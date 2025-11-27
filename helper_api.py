# helper_api.py
# OpenAI API関連とコア機能
# -----------------------------------------
import re
import time
import json
import logging
import logging.handlers

import yaml
import os
from typing import List, Dict, Any, Optional, Union, Literal
from pathlib import Path
from functools import wraps
from datetime import datetime
import hashlib

import tiktoken
from openai import OpenAI

# -----------------------------------------------------
# OpenAI API型定義
# -----------------------------------------------------
from openai.types.responses import (
    EasyInputMessageParam,
    Response
)
from openai.types.chat import (
    ChatCompletionMessageParam,
)

# Role型の定義
RoleType = Literal["user", "assistant", "system", "developer"]


# ==================================================
# 設定管理
# ==================================================
class ConfigManager:
    """設定ファイルの管理"""

    _instance = None

    def __new__(cls, config_path: str = "config.yml"):
        """シングルトンパターンで設定を管理"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = "config.yml"):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._cache = {}
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger('openai_helper')

        # 既に設定済みの場合はスキップ
        if logger.handlers:
            return logger

        log_config = self.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))
        logger.setLevel(level)

        # フォーマッターの設定
        formatter = logging.Formatter(
            log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        # コンソールハンドラー
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # ファイルハンドラー（設定されている場合）
        log_file = log_config.get("file")
        if log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=log_config.get("max_bytes", 10485760),
                backupCount=log_config.get("backup_count", 5)
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger


    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    # 環境変数での設定オーバーライド
                    self._apply_env_overrides(config)
                    return config
            except Exception as e:
                print(f"設定ファイルの読み込みに失敗: {e}")
                return self._get_default_config()
        else:
            print(f"設定ファイルが見つかりません: {self.config_path}")
            return self._get_default_config()


    def _apply_env_overrides(self, config: Dict[str, Any]) -> None:
        """環境変数による設定オーバーライド"""
        # OpenAI API Key
        if os.getenv("OPENAI_API_KEY"):
            config.setdefault("api", {})["openai_api_key"] = os.getenv("OPENAI_API_KEY")

        # ログレベル
        if os.getenv("LOG_LEVEL"):
            config.setdefault("logging", {})["level"] = os.getenv("LOG_LEVEL")

        # デバッグモード
        if os.getenv("DEBUG_MODE"):
            config.setdefault("experimental", {})["debug_mode"] = os.getenv("DEBUG_MODE").lower() == "true"


    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            "models"        : {
                "default"  : "gpt-5-mini",
                "available": ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"]
            },
            "api"           : {
                "timeout"       : 30,
                "max_retries"   : 3,
                "openai_api_key": None
            },
            "ui"            : {
                "page_title": "OpenAI API Demo",
                "page_icon" : "🤖",
                "layout"    : "wide"
            },
            "cache"         : {
                "enabled" : True,
                "ttl"     : 3600,
                "max_size": 100
            },
            "logging"       : {
                "level"       : "INFO",
                "format"      : "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file"        : None,
                "max_bytes"   : 10485760,
                "backup_count": 5
            },
            "error_messages": {
                "general_error"  : "エラーが発生しました",
                "api_key_missing": "APIキーが設定されていません",
                "network_error"  : "ネットワークエラーが発生しました"
            },
            "experimental"  : {
                "debug_mode"            : False,
                "performance_monitoring": True
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """設定値の取得（キャッシュ付き）"""
        if key in self._cache:
            return self._cache[key]

        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                value = default
                break

        result = value if value is not None else default
        self._cache[key] = result
        return result

    def set(self, key: str, value: Any) -> None:
        """設定値の更新"""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value

        # キャッシュクリア
        self._cache.pop(key, None)

    def reload(self):
        """設定の再読み込み"""
        self._config = self._load_config()
        self._cache.clear()

    def save(self, filepath: str = None) -> bool:
        """設定をファイルに保存"""
        try:
            save_path = Path(filepath) if filepath else self.config_path
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(self._config, f, default_flow_style=False, allow_unicode=True)
            return True
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"設定保存エラー: {e}")
            return False


# グローバル設定インスタンス
config = ConfigManager("config.yml")
logger = config.logger


# ==================================================
# メモリベースキャッシュ
# ==================================================
class MemoryCache:
    """メモリベースキャッシュ"""

    def __init__(self):
        self._storage = {}
        self._enabled = config.get("cache.enabled", True)
        self._ttl = config.get("cache.ttl", 3600)
        self._max_size = config.get("cache.max_size", 100)

    def get(self, key: str) -> Any:
        """キャッシュから値を取得"""
        if not self._enabled or key not in self._storage:
            return None

        cached_data = self._storage[key]
        if time.time() - cached_data['timestamp'] > self._ttl:
            del self._storage[key]
            return None

        return cached_data['result']

    def set(self, key: str, value: Any) -> None:
        """キャッシュに値を設定"""
        if not self._enabled:
            return

        self._storage[key] = {
            'result'   : value,
            'timestamp': time.time()
        }

        # サイズ制限チェック
        if len(self._storage) > self._max_size:
            oldest_key = min(self._storage, key=lambda k: self._storage[k]['timestamp'])
            del self._storage[oldest_key]

    def clear(self) -> None:
        """キャッシュクリア"""
        self._storage.clear()

    def size(self) -> int:
        """キャッシュサイズ"""
        return len(self._storage)


# グローバルキャッシュインスタンス
cache = MemoryCache()


# ==================================================
# 安全なJSON処理関数
# ==================================================
def safe_json_serializer(obj: Any) -> Any:
    """
    カスタムJSONシリアライザー
    OpenAI APIのレスポンスオブジェクトなど、標準では処理できないオブジェクトを変換
    """
    # Pydantic モデルの場合
    if hasattr(obj, 'model_dump'):
        try:
            return obj.model_dump()
        except Exception:
            pass

    # dict() メソッドがある場合
    if hasattr(obj, 'dict'):
        try:
            return obj.dict()
        except Exception:
            pass

    # datetime オブジェクトの場合
    if isinstance(obj, datetime):
        return obj.isoformat()

    # OpenAI ResponseUsage オブジェクトの場合（手動属性抽出）
    if hasattr(obj, 'prompt_tokens') and hasattr(obj, 'completion_tokens'):
        return {
            'prompt_tokens'    : getattr(obj, 'prompt_tokens', 0),
            'completion_tokens': getattr(obj, 'completion_tokens', 0),
            'total_tokens'     : getattr(obj, 'total_tokens', 0)
        }

    # その他のオブジェクトは文字列化
    return str(obj)


def safe_json_dumps(data: Any, **kwargs) -> str:
    """安全なJSON文字列化"""
    default_kwargs = {
        'ensure_ascii': False,
        'indent'      : 2,
        'default'     : safe_json_serializer
    }
    default_kwargs.update(kwargs)

    try:
        return json.dumps(data, **default_kwargs)
    except Exception as e:
        logger.error(f"JSON serialization error: {e}")
        # フォールバック: 文字列化
        return json.dumps(str(data), **{k: v for k, v in default_kwargs.items() if k != 'default'})


# ==================================================
# デコレータ（API用）
# ==================================================
def error_handler(func):
    """エラーハンドリングデコレータ（API用）"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            # API用では例外を再発生させる
            raise

    return wrapper


def timer(func):
    """実行時間計測デコレータ（API用）"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} took {execution_time:.2f} seconds")
        return result

    return wrapper


def cache_result(ttl: int = None):
    """結果をキャッシュするデコレータ（メモリベース）"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not config.get("cache.enabled", True):
                return func(*args, **kwargs)

            # キャッシュキーの生成
            cache_key = f"{func.__name__}_{hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()}"

            # キャッシュから取得
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # 関数実行とキャッシュ保存
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result

        return wrapper

    return decorator


# --------------------------------------------------
# デフォルトプロンプト　responses-API（例）ソフトウェア開発用
# --------------------------------------------------
developer_text = (
    "You are a strong developer and good at teaching software developer professionals "
    "please provide an up-to-date, informed overview of the API by function, then show "
    "cookbook programs for each, and explain the API options."
    "あなたは強力な開発者でありソフトウェア開発者の専門家に教えるのが得意です。"
    "OpenAIのAPIを機能別に最新かつ詳細に説明してください。"
    "それぞれのAPIのサンプルプログラムを示しAPIのオプションについて説明してください。"
)
user_text = (
    "Organize and identify the problem and list the issues. "
    "Then, provide a solution procedure for the issues you have organized and identified, "
    "and solve the problems/issues according to the solution procedures."
    "不具合、問題を特定し、整理して箇条書きで列挙・説明してください。"
    "次に、整理・特定した問題点の解決手順を示しなさい。"
    "次に、解決手順に従って問題・課題を解決してください。"
)
assistant_text = "OpenAIのAPIを使用するには、公式openaiライブラリが便利です。回答は日本語で"


def get_default_messages() -> list[EasyInputMessageParam]:

    return [
    EasyInputMessageParam(role="developer", content=developer_text),
    EasyInputMessageParam(role="user",      content=user_text),
    EasyInputMessageParam(role="assistant", content=assistant_text),
]

def append_user_message(append_text, image_url=None):
    return [
    EasyInputMessageParam(role="developer", content=developer_text),
    EasyInputMessageParam(role="user",      content=user_text),
    EasyInputMessageParam(role="assistant", content=assistant_text),
    EasyInputMessageParam(role="user", content=append_text),
]

def append_developer_message(append_text):
    return [
    EasyInputMessageParam(role="developer", content=developer_text),
    EasyInputMessageParam(role="user",      content=user_text),
    EasyInputMessageParam(role="assistant", content=assistant_text),
    EasyInputMessageParam(role="developer", content=append_text),
]

def append_assistant_message(append_text):
    return [
        EasyInputMessageParam(role="developer", content=developer_text),
        EasyInputMessageParam(role="user", content=user_text),
        EasyInputMessageParam(role="assistant", content=assistant_text),
        EasyInputMessageParam(role="assistant", content=append_text),
    ]

# ==================================================
# メッセージ管理
# ==================================================
class MessageManager:
    """メッセージ履歴の管理（API用）"""

    def __init__(self, messages: List[EasyInputMessageParam] = None):
        self._messages = messages or self.get_default_messages()

    @staticmethod
    def get_default_messages() -> List[EasyInputMessageParam]:
        """デフォルトメッセージの取得"""
        default_messages = config.get("default_messages", {})

        developer_content = default_messages.get(
            "developer",
            "You are a helpful assistant specialized in software development."
        )
        user_content = default_messages.get(
            "user",
            "Please help me with my software development tasks."
        )
        assistant_content = default_messages.get(
            "assistant",
            "I'll help you with your software development needs. Please let me know what you'd like to work on."
        )

        return [
            EasyInputMessageParam(role="developer", content=developer_content),
            EasyInputMessageParam(role="user", content=user_content),
            EasyInputMessageParam(role="assistant", content=assistant_content),
        ]

    def add_message(self, role: RoleType, content: str):
        """メッセージの追加"""
        valid_roles: List[RoleType] = ["user", "assistant", "system", "developer"]
        if role not in valid_roles:
            raise ValueError(f"Invalid role: {role}. Must be one of {valid_roles}")

        self._messages.append(EasyInputMessageParam(role=role, content=content))

        # メッセージ数制限
        limit = config.get("api.message_limit", 50)
        if len(self._messages) > limit:
            # 最初のdeveloperメッセージは保持
            developer_msg = self._messages[0] if self._messages[0]['role'] == 'developer' else None
            self._messages = self._messages[-limit:]
            if developer_msg and self._messages[0]['role'] != 'developer':
                self._messages.insert(0, developer_msg)

    def get_messages(self) -> List[EasyInputMessageParam]:
        """メッセージ履歴の取得"""
        return self._messages.copy()

    def clear_messages(self):
        """メッセージ履歴のクリア"""
        self._messages = self.get_default_messages()

    def export_messages(self) -> Dict[str, Any]:
        """メッセージ履歴のエクスポート"""
        return {
            'messages'   : self.get_messages(),
            'exported_at': datetime.now().isoformat()
        }

    def import_messages(self, data: Dict[str, Any]):
        """メッセージ履歴のインポート"""
        if 'messages' in data:
            self._messages = data['messages']


# ==================================================
# トークン管理
# ==================================================
class TokenManager:
    """トークン数の管理（新モデル対応）"""

    # モデル別のエンコーディング対応表
    MODEL_ENCODINGS = {
        "gpt-4o"                   : "cl100k_base",
        "gpt-4o-mini"              : "cl100k_base",
        "gpt-4o-audio-preview"     : "cl100k_base",
        "gpt-4o-mini-audio-preview": "cl100k_base",
        "gpt-4.1"                  : "cl100k_base",
        "gpt-4.1-mini"             : "cl100k_base",
        "o1"                       : "cl100k_base",
        "o1-mini"                  : "cl100k_base",
        "o3"                       : "cl100k_base",
        "o3-mini"                  : "cl100k_base",
        "o4"                       : "cl100k_base",
        "o4-mini"                  : "cl100k_base",
    }

    @classmethod
    def count_tokens(cls, text: str, model: str = None) -> int:
        """テキストのトークン数をカウント"""
        if model is None:
            model = config.get("models.default", "gpt-4o-mini")

        try:
            encoding_name = cls.MODEL_ENCODINGS.get(model, "cl100k_base")
            enc = tiktoken.get_encoding(encoding_name)
            return len(enc.encode(text))
        except Exception as e:
            logger.error(f"トークンカウントエラー: {e}")
            # 簡易的な推定（1文字 = 0.5トークン）
            return len(text) // 2

    @classmethod
    def truncate_text(cls, text: str, max_tokens: int, model: str = None) -> str:
        """テキストを指定トークン数に切り詰め"""
        if model is None:
            model = config.get("models.default", "gpt-4o-mini")

        try:
            encoding_name = cls.MODEL_ENCODINGS.get(model, "cl100k_base")
            enc = tiktoken.get_encoding(encoding_name)
            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return enc.decode(tokens[:max_tokens])
        except Exception as e:
            logger.error(f"テキスト切り詰めエラー: {e}")
            estimated_chars = max_tokens * 2
            return text[:estimated_chars]

    @classmethod
    def estimate_cost(cls, input_tokens: int, output_tokens: int, model: str = None) -> float:
        """API使用コストの推定"""
        if model is None:
            model = config.get("models.default", "gpt-4o-mini")

        pricing = config.get("model_pricing", {})
        model_pricing = pricing.get(model, pricing.get("gpt-4o-mini"))

        if not model_pricing:
            model_pricing = {"input": 0.00015, "output": 0.0006}

        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]

        return input_cost + output_cost

    @classmethod
    def get_model_limits(cls, model: str) -> Dict[str, int]:
        """モデルのトークン制限を取得"""
        limits = {
            "gpt-4o"      : {"max_tokens": 128000, "max_output": 4096},
            "gpt-4o-mini" : {"max_tokens": 128000, "max_output": 4096},
            "gpt-4.1"     : {"max_tokens": 128000, "max_output": 4096},
            "gpt-4.1-mini": {"max_tokens": 128000, "max_output": 4096},
            "o1"          : {"max_tokens": 128000, "max_output": 32768},
            "o1-mini"     : {"max_tokens": 128000, "max_output": 65536},
            "o3"          : {"max_tokens": 200000, "max_output": 100000},
            "o3-mini"     : {"max_tokens": 200000, "max_output": 100000},
            "o4"          : {"max_tokens": 256000, "max_output": 128000},
            "o4-mini"     : {"max_tokens": 256000, "max_output": 128000},
        }
        return limits.get(model, {"max_tokens": 128000, "max_output": 4096})


# ==================================================
# レスポンス処理
# ==================================================
class ResponseProcessor:
    """API レスポンスの処理"""

    @staticmethod
    def extract_text(response: Response) -> List[str]:
        """レスポンスからテキストを抽出"""
        texts = []

        if hasattr(response, 'output'):
            for item in response.output:
                if hasattr(item, 'type') and item.type == "message":
                    if hasattr(item, 'content'):
                        for content in item.content:
                            if hasattr(content, 'type') and content.type == "output_text":
                                if hasattr(content, 'text'):
                                    texts.append(content.text)

        # フォールバック: output_text属性
        if not texts and hasattr(response, 'output_text'):
            texts.append(response.output_text)

        return texts

    @staticmethod
    def _serialize_usage(usage_obj) -> Dict[str, Any]:
        """ResponseUsageオブジェクトを辞書に変換"""
        if usage_obj is None:
            return {}

        # Pydantic モデルの場合
        if hasattr(usage_obj, 'model_dump'):
            try:
                return usage_obj.model_dump()
            except Exception:
                pass

        # dict() メソッドがある場合
        if hasattr(usage_obj, 'dict'):
            try:
                return usage_obj.dict()
            except Exception:
                pass

        # 手動で属性を抽出
        usage_dict = {}
        for attr in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
            if hasattr(usage_obj, attr):
                usage_dict[attr] = getattr(usage_obj, attr)

        return usage_dict

    @staticmethod
    def format_response(response: Response) -> Dict[str, Any]:
        """レスポンスを整形（JSON serializable）"""
        # usage オブジェクトを安全に変換
        usage_obj = getattr(response, "usage", None)
        usage_dict = ResponseProcessor._serialize_usage(usage_obj)

        return {
            "id"        : getattr(response, "id", None),
            "model"     : getattr(response, "model", None),
            "created_at": getattr(response, "created_at", None),
            "text"      : ResponseProcessor.extract_text(response),
            "usage"     : usage_dict,
        }

    @staticmethod
    def save_response(response: Response, filename: str = None) -> str:
        """レスポンスの保存"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"response_{timestamp}.json"

        formatted = ResponseProcessor.format_response(response)

        # ファイルパスの生成
        logs_dir = Path(config.get("paths.logs_dir", "logs"))
        logs_dir.mkdir(exist_ok=True)
        filepath = logs_dir / filename

        # 保存
        save_json_file(formatted, str(filepath))

        return str(filepath)


# ==================================================
# APIクライアント
# ==================================================
class OpenAIClient:
    """OpenAI API クライアント"""

    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = config.get("api.openai_api_key") or os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(config.get("error_messages.api_key_missing", "APIキーが設定されていません"))

        self.client = OpenAI(api_key=api_key)

    @error_handler
    @timer
    def create_response(
        self,
        messages: List[EasyInputMessageParam] = None,
        *,
        input: List[EasyInputMessageParam] = None,
        model: str = None,
        **kwargs,
    ) -> Response:
        """Responses API呼び出し

        `messages` 引数（旧仕様）と `input` 引数（新仕様）の両方に対応する。
        いずれも指定されていない場合はエラーを返す。
        """
        if model is None:
            model = config.get("models.default", "gpt-4o-mini")

        # 新旧両方の引数名をサポート
        if input is None:
            input = messages
        if input is None:
            raise ValueError("messages or input must be provided")

        params = {
            "model": model,
            "input": input,
        }
        params.update(kwargs)

        return self.client.responses.create(**params)

    @error_handler
    @timer
    def create_chat_completion(self, messages: List[ChatCompletionMessageParam], model: str = None, **kwargs):
        """Chat Completions API呼び出し"""
        if model is None:
            model = config.get("models.default", "gpt-4o-mini")

        params = {
            "model"   : model,
            "messages": messages,
        }
        params.update(kwargs)

        return self.client.chat.completions.create(**params)


# ==================================================
# ユーティリティ関数
# ==================================================
def sanitize_key(name: str) -> str:
    """キー用に安全な文字列へ変換"""
    return re.sub(r'[^0-9a-zA-Z_]', '_', name).lower()


def load_json_file(filepath: str) -> Optional[Dict[str, Any]]:
    """JSONファイルの読み込み"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"JSONファイル読み込みエラー: {e}")
        return None


def save_json_file(data: Dict[str, Any], filepath: str) -> bool:
    """JSONファイルの保存"""
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # 安全なJSON保存を使用
        json_str = safe_json_dumps(data)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_str)
        return True
    except Exception as e:
        logger.error(f"JSONファイル保存エラー: {e}")
        return False


def format_timestamp(timestamp: Union[int, float, str] = None) -> str:
    """タイムスタンプのフォーマット"""
    if timestamp is None:
        timestamp = time.time()

    if isinstance(timestamp, str):
        return timestamp

    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def create_session_id() -> str:
    """セッションIDの生成"""
    return hashlib.md5(f"{time.time()}_{id(object())}".encode()).hexdigest()[:8]

# ==================================================
# エクスポート
# ==================================================
__all__ = [
    # 型定義
    'RoleType',

    # クラス
    'ConfigManager',
    'MessageManager',
    'TokenManager',
    'ResponseProcessor',
    'OpenAIClient',
    'MemoryCache',

    # デコレータ
    'error_handler',
    'timer',
    'cache_result',

    # ユーティリティ
    'sanitize_key',
    'load_json_file',
    'save_json_file',
    'format_timestamp',
    'create_session_id',
    'safe_json_serializer',
    'safe_json_dumps',

    # 定数
    'developer_text',
    'user_text',
    'assistant_text',

    # グローバル
    'config',
    'logger',
    'cache',
]
