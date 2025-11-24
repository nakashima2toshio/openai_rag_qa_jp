# helper_api.py 仕様書

作成日: 2024-10-29
更新日: 2024-11-21

## OpenAI API 利用状況一覧

### プロジェクト内で使用中のAPI

| API名 | メソッド | 用途 | 使用箇所 | 説明 |
|------|---------|-----|---------|-----|
| **Responses API** | `client.responses.create()` | テキスト生成（新形式） | helper_api.py:715-743 | 新しいメッセージ形式（developer role含む）をサポート |
| **Chat Completions API** | `client.chat.completions.create()` | チャット形式の対話生成 | helper_api.py:747-758 | 従来のチャット形式API、JSON出力対応 |
| **Structured Outputs API** | `client.responses.parse()` | 構造化出力（Pydantic連携） | celery_tasks.py:198-203,424-429 | Pydanticモデルで型安全な出力を保証 |

### プロジェクトで利用可能な追加API

| API名 | メソッド | 用途 | 対応状況 | 説明 |
|------|---------|-----|---------|-----|
| **Embeddings API** | `client.embeddings.create()` | テキストのベクトル化 | 未実装 | RAGシステムでの類似検索用 |
| **Moderation API** | `client.moderations.create()` | コンテンツの安全性チェック | 未実装 | 不適切なコンテンツの検出 |
| **Images API** | `client.images.generate()` | 画像生成 | 未実装 | DALL-E 3による画像生成 |
| **Audio API** | `client.audio.transcriptions.create()` | 音声認識 | 未実装 | Whisperによる文字起こし |
| **Files API** | `client.files.create()` | ファイル管理 | 未実装 | Fine-tuning用データのアップロード |
| **Fine-tuning API** | `client.fine_tuning.jobs.create()` | モデルのファインチューニング | 未実装 | カスタムモデルの作成 |

### API使用時の主要パラメータ

| パラメータ | 対応API | 型 | 説明 | デフォルト値 |
|-----------|--------|---|------|------------|
| `model` | 全API共通 | str | 使用するモデル名 | "gpt-5-mini" |
| `messages`/`input` | Responses, Chat | List | 入力メッセージ | 必須 |
| `temperature` | Chat Completions | float | 生成の多様性（0-2） | 0.7 |
| `max_tokens` | Chat Completions | int | 最大生成トークン数 | モデル依存 |
| `max_completion_tokens` | Responses (新モデル) | int | 最大出力トークン数 | 1000 |
| `max_output_tokens` | Structured Outputs | int | 構造化出力の最大トークン | 1000-2000 |
| `response_format` | Chat/Structured | dict/Model | 出力形式の指定 | なし |
| `text_format` | responses.parse | Pydantic Model | 構造化出力の型定義 | 必須 |

### サポートモデル一覧

| モデルシリーズ | モデル名 | 最大入力トークン | 最大出力トークン | 用途 | 備考 |
|--------------|---------|----------------|----------------|------|------|
| **GPT-4o** | gpt-4o, gpt-4o-mini | 128,000 | 4,096 | 汎用 | 現行主力モデル |
| **GPT-4.1** | gpt-4.1, gpt-4.1-mini | 128,000 | 4,096 | 汎用 | 改良版 |
| **GPT-5** | gpt-5, gpt-5-mini, gpt-5-nano | 未定 | 未定 | 次世代 | 開発中 |
| **O-Series** | o1, o1-mini | 128,000 | 32,768-65,536 | 推論特化 | temperatureパラメータ非対応 |
| **O-Series (新)** | o3, o3-mini, o4, o4-mini | 200,000-256,000 | 100,000-128,000 | 大規模推論 | 大容量入出力対応 |

## 概要

OpenAI API連携を支援するヘルパーモジュール。設定管理、メッセージ処理、トークン管理、レスポンス処理、キャッシュなどOpenAI API利用のための包括的な機能を提供。

## ファイル情報

- **ファイル名**: helper_api.py
- **行数**: 847行
- **主な機能**: OpenAI API統合とキャッシュ
- **主要API**: Responses API、Chat Completions API

## 主な機能

### 1. 設定管理
- ConfigManager: 階層的設定管理
- YAML設定ファイル読み込み
- 環境変数オーバーライド
- シングルトン実装

### 2. メッセージ管理
- MessageManager: メッセージリスト管理
- ロール種別（user/assistant/system/developer）
- メッセージ制限管理
- エクスポート/インポート機能

### 3. トークン管理
- TokenManager: トークン数計算
- テキスト切り詰め
- コスト推定
- モデル制限確認

### 4. レスポンス処理
- ResponseProcessor: レスポンス解析
- テキスト抽出
- JSON serialization
- ファイル保存

### 5. API クライアント
- OpenAIClient: 統一APIクライアント
- Responses API対応
- Chat Completions API対応
- エラーハンドリング

### 6. キャッシュ
- MemoryCache: インメモリキャッシュ
- TTL（Time To Live）管理
- サイズ制限管理

## アーキテクチャ

### 階層構造

```
ConfigManager (L47-220)
   └── 設定読み込み
   └── YAML解析
   └── 環境変数オーバーライド
   └── シングルトン設定

MemoryCache (L230-276)
   └── TTL付きキャッシュ
   └── サイズ制限管理
   └── 自動削除機能

MessageManager (L451-517)
   └── メッセージリスト管理
   └── ロール検証
   └── エクスポート/インポート

TokenManager (L522-607)
   └── トークン数計算
   └── テキスト切り詰め
   └── コスト推定
   └── モデル制限

ResponseProcessor (L612-696)
   └── レスポンス解析
   └── テキスト抽出
   └── ファイル保存

OpenAIClient (L701-758)
   └── Responses API
   └── Chat Completions API
   └── エラーハンドリング
```

## 主要クラス

### 1. ConfigManager (L47-220)

設定管理のシングルトンクラス。

#### 初期化 (L52-65)

```python
def __new__(cls, config_path: str = "config.yml"):
    """シングルトンインスタンスの生成"""
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
```

**特徴**:
- シングルトンパターン: アプリ全体で1つのインスタンス
- 初期化制御: 重複初期化防止
- キャッシュ機能: 設定値の高速アクセス

#### ログ設定 (L67-100)

```python
def _setup_logger(self) -> logging.Logger:
    """ロガーの設定"""
    logger = logging.getLogger('openai_helper')

    # コンソールハンドラ
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ファイルハンドラ（設定で有効な場合）
    log_file = log_config.get("file")
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=log_config.get("max_bytes", 10485760),
            backupCount=log_config.get("backup_count", 5)
        )
```

**機能特性**:
- コンソール出力
- ローテーションファイル出力（10MB、5ファイル）
- レベル別フォーマット

#### 環境変数オーバーライド (L120-132)

```python
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
```

**サポート環境変数**:
- `OPENAI_API_KEY`: APIキー
- `LOG_LEVEL`: ログレベル
- `DEBUG_MODE`: デバッグモード

#### デフォルト設定 (L135-173)

```python
def _get_default_config(self) -> Dict[str, Any]:
    return {
        "models": {
            "default": "gpt-5-mini",
            "available": ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"]
        },
        "api": {
            "timeout": 30,
            "max_retries": 3,
            "openai_api_key": None
        },
        "cache": {
            "enabled": True,
            "ttl": 3600,
            "max_size": 100
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": None,
            "max_bytes": 10485760,
            "backup_count": 5
        },
        # ...
    }
```

#### 主要メソッド

**get() (L175-191)**
```python
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
```

使用例:
```python
config.get("models.default")  # "gpt-5-mini"
config.get("api.timeout", 30)  # 30
```

**set() (L193-202)**
```python
def set(self, key: str, value: Any) -> None:
    """設定値の更新"""
    keys = key.split('.')
    config = self._config
    for k in keys[:-1]:
        config = config.setdefault(k, {})
    config[keys[-1]] = value

    # キャッシュクリア
    self._cache.pop(key, None)
```

**save() (L209-219)**
```python
def save(self, filepath: str = None) -> bool:
    """設定をファイルに保存"""
    save_path = Path(filepath) if filepath else self.config_path
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(self._config, f, default_flow_style=False, allow_unicode=True)
    return True
```

### 2. MemoryCache (L230-276)

インメモリキャッシュ実装。

#### 初期化 (L233-237)

```python
def __init__(self):
    self._storage = {}
    self._enabled = config.get("cache.enabled", True)
    self._ttl = config.get("cache.ttl", 3600)
    self._max_size = config.get("cache.max_size", 100)
```

**デフォルト設定**:
- 有効状態: True
- TTL: 3600秒（1時間）
- 最大サイズ: 100アイテム

#### 主要メソッド

**get() (L239-248)**
```python
def get(self, key: str) -> Any:
    """キャッシュから値を取得"""
    if not self._enabled or key not in self._storage:
        return None

    cached_data = self._storage[key]
    if time.time() - cached_data['timestamp'] > self._ttl:
        del self._storage[key]
        return None

    return cached_data['result']
```

**set() (L251-264)**
```python
def set(self, key: str, value: Any) -> None:
    """キャッシュに値を設定"""
    if not self._enabled:
        return

    self._storage[key] = {
        'result': value,
        'timestamp': time.time()
    }

    # サイズ制限チェック
    if len(self._storage) > self._max_size:
        oldest_key = min(self._storage, key=lambda k: self._storage[k]['timestamp'])
        del self._storage[oldest_key]
```

**自動削除機能**:
- TTL超過時に自動削除
- サイズ制限超過時に最古のアイテムを削除

### 3. MessageManager (L451-517)

メッセージリストの管理クラス。

#### 初期化 (L454-455)

```python
def __init__(self, messages: List[EasyInputMessageParam] = None):
    self._messages = messages or self.get_default_messages()
```

#### デフォルトメッセージ (L458-479)

```python
@staticmethod
def get_default_messages() -> List[EasyInputMessageParam]:
    """デフォルトメッセージの取得"""
    default_messages = config.get("default_messages", {})

    return [
        EasyInputMessageParam(role="developer", content=developer_content),
        EasyInputMessageParam(role="user", content=user_content),
        EasyInputMessageParam(role="assistant", content=assistant_content),
    ]
```

#### add_message() (L481-496)

```python
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
```

**メッセージ数制限**:
- デフォルト: 50メッセージ
- 常にdeveloperメッセージを保持
- 古いメッセージから自動削除

#### エクスポート/インポート (L506-516)

```python
def export_messages(self) -> Dict[str, Any]:
    """メッセージリストのエクスポート"""
    return {
        'messages': self.get_messages(),
        'exported_at': datetime.now().isoformat()
    }

def import_messages(self, data: Dict[str, Any]):
    """メッセージリストのインポート"""
    if 'messages' in data:
        self._messages = data['messages']
```

### 4. TokenManager (L522-607)

トークン数の計算と管理を行うクラス。

#### モデル別エンコーディング (L526-539)

```python
MODEL_ENCODINGS = {
    "gpt-4o": "cl100k_base",
    "gpt-4o-mini": "cl100k_base",
    "gpt-4o-audio-preview": "cl100k_base",
    "gpt-4o-mini-audio-preview": "cl100k_base",
    "gpt-4.1": "cl100k_base",
    "gpt-4.1-mini": "cl100k_base",
    "o1": "cl100k_base",
    "o1-mini": "cl100k_base",
    "o3": "cl100k_base",
    "o3-mini": "cl100k_base",
    "o4": "cl100k_base",
    "o4-mini": "cl100k_base",
}
```

全モデル`cl100k_base`エンコーディング使用。

#### count_tokens() (L542-554)

```python
@classmethod
def count_tokens(cls, text: str, model: str = None) -> int:
    """テキストのトークン数を計算"""
    if model is None:
        model = config.get("models.default", "gpt-4o-mini")

    try:
        encoding_name = cls.MODEL_ENCODINGS.get(model, "cl100k_base")
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except Exception as e:
        logger.error(f"トークン計算エラー: {e}")
        # 簡易推定（1文字 = 0.5トークン）
        return len(text) // 2
```

**フォールバック処理**:
- エラー時は簡易推定（1文字 = 0.5トークン）

#### truncate_text() (L557-572)

```python
@classmethod
def truncate_text(cls, text: str, max_tokens: int, model: str = None) -> str:
    """テキストを最大トークン数に切り詰め"""
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
```

#### estimate_cost() (L575-589)

```python
@classmethod
def estimate_cost(cls, input_tokens: int, output_tokens: int, model: str = None) -> float:
    """API使用料金の推定"""
    if model is None:
        model = config.get("models.default", "gpt-4o-mini")

    pricing = config.get("model_pricing", {})
    model_pricing = pricing.get(model, pricing.get("gpt-4o-mini"))

    if not model_pricing:
        model_pricing = {"input": 0.00015, "output": 0.0006}

    input_cost = (input_tokens / 1000) * model_pricing["input"]
    output_cost = (output_tokens / 1000) * model_pricing["output"]

    return input_cost + output_cost
```

#### get_model_limits() (L592-606)

```python
@classmethod
def get_model_limits(cls, model: str) -> Dict[str, int]:
    """モデルのトークン制限を取得"""
    limits = {
        "gpt-4o": {"max_tokens": 128000, "max_output": 4096},
        "gpt-4o-mini": {"max_tokens": 128000, "max_output": 4096},
        "gpt-4.1": {"max_tokens": 128000, "max_output": 4096},
        "gpt-4.1-mini": {"max_tokens": 128000, "max_output": 4096},
        "o1": {"max_tokens": 128000, "max_output": 32768},
        "o1-mini": {"max_tokens": 128000, "max_output": 65536},
        "o3": {"max_tokens": 200000, "max_output": 100000},
        "o3-mini": {"max_tokens": 200000, "max_output": 100000},
        "o4": {"max_tokens": 256000, "max_output": 128000},
        "o4-mini": {"max_tokens": 256000, "max_output": 128000},
    }
    return limits.get(model, {"max_tokens": 128000, "max_output": 4096})
```

### 5. ResponseProcessor (L612-696)

APIレスポンスの処理クラス。

#### extract_text() (L616-633)

```python
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
```

#### format_response() (L664-676)

```python
@staticmethod
def format_response(response: Response) -> Dict[str, Any]:
    """レスポンスをJSON serializable形式に変換"""
    usage_obj = getattr(response, "usage", None)
    usage_dict = ResponseProcessor._serialize_usage(usage_obj)

    return {
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
        "created_at": getattr(response, "created_at", None),
        "text": ResponseProcessor.extract_text(response),
        "usage": usage_dict,
    }
```

#### save_response() (L679-695)

```python
@staticmethod
def save_response(response: Response, filename: str = None) -> str:
    """レスポンスの保存"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"response_{timestamp}.json"

    formatted = ResponseProcessor.format_response(response)

    # ファイルパスの準備
    logs_dir = Path(config.get("paths.logs_dir", "logs"))
    logs_dir.mkdir(exist_ok=True)
    filepath = logs_dir / filename

    # 保存
    save_json_file(formatted, str(filepath))

    return str(filepath)
```

### 6. OpenAIClient (L701-758)

OpenAI API統一クライアント。

#### 初期化 (L704-711)

```python
def __init__(self, api_key: str = None):
    if api_key is None:
        api_key = config.get("api.openai_api_key") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(config.get("error_messages.api_key_missing", "APIキーが設定されていません"))

    self.client = OpenAI(api_key=api_key)
```

**APIキー取得優先順位**:
1. 関数引数
2. config.yml
3. 環境変数 `OPENAI_API_KEY`

#### create_response() (L715-743)

```python
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
    """Responses APIを呼び出し

    `messages` パラメータと `input` パラメータの両方を受け付ける
    """
    if model is None:
        model = config.get("models.default", "gpt-4o-mini")

    # 入力パラメータの処理
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
```

**デコレーター**:
- `@error_handler`: エラーハンドリング
- `@timer`: 実行時間計測

#### create_chat_completion() (L747-758)

```python
@error_handler
@timer
def create_chat_completion(self, messages: List[ChatCompletionMessageParam], model: str = None, **kwargs):
    """Chat Completions APIを呼び出し"""
    if model is None:
        model = config.get("models.default", "gpt-4o-mini")

    params = {
        "model": model,
        "messages": messages,
    }
    params.update(kwargs)

    return self.client.chat.completions.create(**params)
```

## デコレーター

### 1. error_handler (L337-349)

```python
def error_handler(func):
    """エラーハンドリングデコレーター（API用）"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            # API用なのでエラーは再送出
            raise
    return wrapper
```

**特徴**:
- エラーログを記録
- エラーは再送出（API用）

### 2. timer (L352-364)

```python
def timer(func):
    """実行時間計測デコレーター（API用）"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} took {execution_time:.2f} seconds")
        return result
    return wrapper
```

### 3. cache_result (L367-391)

```python
def cache_result(ttl: int = None):
    """結果キャッシュデコレーター（関数用）"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not config.get("cache.enabled", True):
                return func(*args, **kwargs)

            # キャッシュキー生成
            cache_key = f"{func.__name__}_{hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()}"

            # キャッシュから取得
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # 実行してキャッシュに保存
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
        return wrapper
    return decorator
```

## ユーティリティ関数

### JSON処理

**safe_json_serializer() (L282-314)**
```python
def safe_json_serializer(obj: Any) -> Any:
    """オブジェクトをJSONシリアライズ可能に変換"""
    # Pydantic モデルの処理
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()

    # datetime オブジェクトの処理
    if isinstance(obj, datetime):
        return obj.isoformat()

    # ResponseUsage オブジェクトの処理
    if hasattr(obj, 'prompt_tokens') and hasattr(obj, 'completion_tokens'):
        return {
            'prompt_tokens': getattr(obj, 'prompt_tokens', 0),
            'completion_tokens': getattr(obj, 'completion_tokens', 0),
            'total_tokens': getattr(obj, 'total_tokens', 0)
        }

    return str(obj)
```

**safe_json_dumps() (L317-331)**
```python
def safe_json_dumps(data: Any, **kwargs) -> str:
    """安全なJSON文字列化"""
    default_kwargs = {
        'ensure_ascii': False,
        'indent': 2,
        'default': safe_json_serializer
    }
    default_kwargs.update(kwargs)

    return json.dumps(data, **default_kwargs)
```

### 文字列処理

**sanitize_key() (L764-766)**
```python
def sanitize_key(name: str) -> str:
    """キー名を安全な形式に変換"""
    return re.sub(r'[^0-9a-zA-Z_]', '_', name).lower()
```

**create_session_id() (L805-807)**
```python
def create_session_id() -> str:
    """セッションIDの生成"""
    return hashlib.md5(f"{time.time()}_{id(object())}".encode()).hexdigest()[:8]
```

## 使用例

### 例1: 設定管理

```python
from helper_api import config

# 設定値の取得
default_model = config.get("models.default")
api_timeout = config.get("api.timeout", 30)

# 設定値の更新
config.set("models.default", "gpt-4o")

# 設定の保存
config.save()
```

### 例2: メッセージ管理

```python
from helper_api import MessageManager

# メッセージマネージャの作成
msg_manager = MessageManager()

# メッセージの追加
msg_manager.add_message("user", "こんにちは")
msg_manager.add_message("assistant", "こんにちは！何かお手伝いできることはありますか？")

# メッセージリストの取得
messages = msg_manager.get_messages()

# エクスポート
exported = msg_manager.export_messages()
```

### 例3: トークン管理

```python
from helper_api import TokenManager

# トークン数計算
text = "これはテストテキストです"
token_count = TokenManager.count_tokens(text, model="gpt-4o-mini")

# テキスト切り詰め
truncated = TokenManager.truncate_text(text, max_tokens=100)

# コスト推定
cost = TokenManager.estimate_cost(1000, 500, model="gpt-4o-mini")
print(f"推定コスト: ${cost:.4f}")

# モデル制限取得
limits = TokenManager.get_model_limits("gpt-4o")
print(f"最大入力: {limits['max_tokens']:,} トークン")
```

### 例4: API クライアント

```python
from helper_api import OpenAIClient, MessageManager

# クライアント作成
client = OpenAIClient()

# メッセージ準備
msg_manager = MessageManager()
msg_manager.add_message("user", "OpenAI APIについて教えてください")
messages = msg_manager.get_messages()

# Responses API呼び出し
response = client.create_response(input=messages, model="gpt-4o-mini")

# レスポンス処理
from helper_api import ResponseProcessor
texts = ResponseProcessor.extract_text(response)
print(texts[0])

# レスポンス保存
filepath = ResponseProcessor.save_response(response)
print(f"保存先: {filepath}")
```

### 例5: キャッシュ

```python
from helper_api import cache, cache_result

# 直接キャッシュ使用
cache.set("my_key", "my_value")
value = cache.get("my_key")

# デコレーターでキャッシュ
@cache_result(ttl=3600)
def expensive_operation(param):
    # 重い処理
    return result

result = expensive_operation("test")  # 初回実行
result = expensive_operation("test")  # 2回目はキャッシュから取得
```

## エラーハンドリング

### APIキーエラー

```python
try:
    client = OpenAIClient()
except ValueError as e:
    print(f"APIキーエラー: {e}")
    # APIキーが設定されていません
```

### API呼び出しエラー

```python
try:
    response = client.create_response(input=messages)
except Exception as e:
    logger.error(f"API呼び出しエラー: {e}")
    # エラーは自動的にログに記録
```

## 設定ファイル (config.yml)

```yaml
models:
  default: gpt-5-mini
  available:
    - gpt-4o-mini
    - gpt-4o
    - gpt-4.1
    - gpt-4.1-mini

api:
  timeout: 30
  max_retries: 3
  message_limit: 50
  openai_api_key: null  # 環境変数から取得

cache:
  enabled: true
  ttl: 3600
  max_size: 100

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/app.log
  max_bytes: 10485760
  backup_count: 5

experimental:
  debug_mode: false
  performance_monitoring: true
```

## パフォーマンス最適化

### 1. シングルトンパターン

```python
# ConfigManagerは1つのインスタンスのみ
config1 = ConfigManager()
config2 = ConfigManager()
assert config1 is config2  # True
```

### 2. キャッシュ活用

```python
# 設定値はキャッシュされる
config.get("models.default")  # ファイルから読み込み
config.get("models.default")  # キャッシュから取得（高速）
```

### 3. TTL付きキャッシュ

```python
# 古いキャッシュは自動削除
cache.set("key", "value")  # 現在時刻を記録
# 3600秒後...
cache.get("key")  # None（自動削除）
```

## 制限事項

1. **APIキー**: 環境変数またはconfig.ymlで設定必須
2. **シングルトン**: ConfigManagerは1インスタンスのみ
3. **キャッシュTTL**: デフォルト1時間で期限切れ
4. **ログファイル**: ローテーション設定（10MB、5ファイル）
5. **メッセージ制限**: デフォルト50メッセージ

## トラブルシューティング

### 問題1: APIキーエラー

**症状**: "APIキーが設定されていません"
**解決策**:
```bash
export OPENAI_API_KEY='your-api-key'
# または config.yml に設定
```

### 問題2: 設定ファイルが見つからない

**症状**: "設定ファイルが見つかりません"
**解決策**: config.yml作成またはデフォルト設定使用

### 問題3: キャッシュが効かない

**症状**: キャッシュから値が取得できない
**解決策**: config.ymlでcache.enabledを確認

## エクスポート定義

```python
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

    # デコレーター
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

    # インスタンス
    'config',
    'logger',
    'cache',
]
```

## まとめ

helper_api.pyはOpenAI API利用のための包括的なヘルパーモジュールです。

### 主要な特徴

1. **統一的な設定管理**: シングルトン、環境変数対応
2. **メッセージ管理**: リスト管理、ロール検証、制限管理
3. **トークン管理**: 計算、切り詰め、コスト推定
4. **レスポンス処理**: テキスト抽出、変換、保存
5. **キャッシュ**: TTL付き、サイズ制限管理
6. **APIクライアント**: Responses API、Chat Completions API統合

### 推奨用途

- OpenAI APIの統一的な利用
- 設定の一元管理
- トークン数とコストの見積もり
- キャッシュによる高速化

---
作成日: 2024-10-29
作成者: OpenAI RAG Q/A JP Development Team