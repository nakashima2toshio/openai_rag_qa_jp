#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
celery_tasks.py - Celery非同期タスク定義
=========================================
Q/Aペア生成の並列処理のためのCeleryタスク定義
"""

import os
import json
import logging
from typing import List, Dict, Optional
from celery import Celery, group
from dotenv import load_dotenv
from openai import OpenAI

# 環境変数読み込み
load_dotenv()

# 共通モジュールからインポート
from models import QAPair, QAPairsResponse
from config import ModelConfig, CeleryConfig

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Celeryアプリケーション設定
app = Celery(
    'qa_generation',
    broker=os.getenv('CELERY_BROKER_URL', CeleryConfig.BROKER_URL),
    backend=os.getenv('CELERY_RESULT_BACKEND', CeleryConfig.RESULT_BACKEND)
)

# Celery設定
app.conf.update(
    task_serializer=CeleryConfig.TASK_SERIALIZER,
    accept_content=CeleryConfig.ACCEPT_CONTENT,
    result_serializer=CeleryConfig.RESULT_SERIALIZER,
    timezone=CeleryConfig.TIMEZONE,
    enable_utc=CeleryConfig.ENABLE_UTC,
    # タスクのタイムアウト設定
    task_time_limit=CeleryConfig.TASK_TIME_LIMIT,
    task_soft_time_limit=CeleryConfig.TASK_SOFT_TIME_LIMIT,
    # 並列度の制御
    worker_concurrency=CeleryConfig.WORKER_CONCURRENCY,
    worker_prefetch_multiplier=CeleryConfig.WORKER_PREFETCH_MULTIPLIER,
    # リトライ設定
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)


# ===========================================
# モデル別パラメータ制約（config.pyから参照）
# ===========================================
def supports_temperature(model: str) -> bool:
    """モデルがtemperatureパラメータをサポートするかチェック"""
    return ModelConfig.supports_temperature(model)


def determine_qa_count(chunk_data: Dict, config: Dict) -> int:
    """
    チャンクのトークン数に基づいてQ/A数を決定

    Args:
        chunk_data: チャンクデータ
        config: 設定情報

    Returns:
        生成するQ/A数
    """
    tokens = chunk_data.get('tokens', 100)
    base_qa_count = config.get('qa_per_chunk', 2)

    # トークン数に応じて調整
    if tokens < 50:
        return max(1, base_qa_count - 1)
    elif tokens > 150:
        return base_qa_count + 1
    else:
        return base_qa_count


def _extract_parsed_response(response, model: str) -> QAPairsResponse:
    """
    responses.parse() API のレスポンスから解析済みデータを抽出

    Args:
        response: OpenAI API レスポンス
        model: 使用したモデル名

    Returns:
        QAPairsResponse オブジェクト

    Raises:
        ValueError: 解析可能なレスポンスがない場合
    """
    parsed_response = None

    # デバッグ: レスポンスの構造をログ出力
    logger.info(f"レスポンスタイプ: {type(response).__name__}")
    response_attrs = [attr for attr in dir(response) if not attr.startswith('_')]
    logger.info(f"レスポンス属性: {response_attrs}")

    # 方法1: output_parsed属性を直接確認（GPT-5シリーズ対応）
    if hasattr(response, 'output_parsed') and response.output_parsed:
        parsed_response = response.output_parsed
        logger.info(f"GPT-5形式のレスポンス（output_parsed）を使用")
        return parsed_response

    # 方法2: output配列から探索（GPT-4o対応）
    if hasattr(response, 'output'):
        logger.info(f"output配列の長さ: {len(response.output)}")
        for idx, output in enumerate(response.output):
            output_type = getattr(output, 'type', 'N/A')
            logger.info(f"output[{idx}] タイプ: {type(output).__name__}, type属性: {output_type}")

            # ReasoningItemはスキップ
            if hasattr(output, 'type'):
                if output.type == "reasoning":
                    logger.info(f"  -> reasoning タイプをスキップ")
                    continue
                elif output.type == "message" and hasattr(output, 'content') and output.content:
                    logger.info(f"  -> message タイプ、content長さ: {len(output.content)}")
                    for item_idx, item in enumerate(output.content):
                        item_type = getattr(item, 'type', 'N/A')
                        logger.info(f"    content[{item_idx}] タイプ: {type(item).__name__}, type属性: {item_type}")

                        # parsed属性をチェック
                        if hasattr(item, 'type') and item.type == "output_text" and hasattr(item, 'parsed') and item.parsed:
                            parsed_response = item.parsed
                            logger.info(f"GPT-4o形式のレスポンス（output配列 -> parsed）を使用")
                            return parsed_response

                        # text属性がある場合もログ出力
                        if hasattr(item, 'text') and item.text:
                            text_preview = item.text[:100] if len(item.text) > 100 else item.text
                            logger.info(f"    text属性あり（最初の100文字）: {text_preview}...")

            if parsed_response:
                break

    # 方法3: 直接text属性からJSON解析を試みる
    if not parsed_response and hasattr(response, 'output'):
        logger.info("方法3: text属性からJSON直接解析を試行")
        for output in response.output:
            if hasattr(output, 'type') and output.type == "message" and hasattr(output, 'content'):
                for item in output.content:
                    if hasattr(item, 'text') and item.text:
                        try:
                            json_data = json.loads(item.text)
                            if 'qa_pairs' in json_data:
                                # JSONからQAPairsResponseを構築
                                parsed_response = QAPairsResponse(**json_data)
                                logger.info(f"JSONテキストから直接解析成功")
                                return parsed_response
                        except json.JSONDecodeError as e:
                            logger.debug(f"JSON解析失敗（JSONDecodeError）: {str(e)[:100]}")
                        except Exception as e:
                            logger.debug(f"JSON解析失敗（その他）: {str(e)[:100]}")
            if parsed_response:
                break

    # 方法4: output_text属性を直接確認
    if not parsed_response and hasattr(response, 'output_text') and response.output_text:
        logger.info("方法4: output_text属性からJSON解析を試行")
        try:
            json_data = json.loads(response.output_text)
            if 'qa_pairs' in json_data:
                parsed_response = QAPairsResponse(**json_data)
                logger.info(f"output_textから直接解析成功")
                return parsed_response
        except Exception as e:
            logger.debug(f"output_text解析失敗: {str(e)[:100]}")

    if not parsed_response:
        logger.error(f"OpenAI APIが解析可能なレスポンスを返しませんでした")
        # レスポンスの詳細をログ出力（デバッグ用）
        try:
            response_str = str(response)[:1000]
            logger.error(f"レスポンス全体（最初の1000文字）: {response_str}")
        except:
            pass
        raise ValueError("No parsable response from OpenAI API")

    return parsed_response


@app.task(bind=True, max_retries=3)
def generate_qa_for_chunk_async(self, chunk_data: Dict, config: Dict, model: str = "gpt-4o-mini") -> Dict:
    """
    単一チャンクからQ/Aペアを非同期生成（Celeryタスク）

    Args:
        chunk_data: チャンクデータ
        config: データセット設定
        model: 使用するモデル

    Returns:
        生成されたQ/Aペアと関連情報を含む辞書
    """
    try:
        logger.info(f"タスク開始: チャンク {chunk_data.get('id', 'unknown')}, モデル: {model}")
        logger.info(f"タスクID: {self.request.id}")

        # OpenAIクライアント初期化
        client = OpenAI()

        # Q/A数の決定（ローカル関数を使用）
        num_pairs = determine_qa_count(chunk_data, config)
        lang = config["lang"]
        logger.info(f"生成予定Q/A数: {num_pairs}, 言語: {lang}")

        # プロンプト設定（言語別）
        if lang == "ja":
            system_prompt = """あなたは教育コンテンツ作成の専門家です。
与えられた日本語テキストから、学習効果の高いQ&Aペアを生成してください。

生成ルール:
1. 質問は明確で具体的に
2. 回答は簡潔で正確に（1-2文程度）
3. テキストの内容に忠実に
4. 多様な観点から質問を作成"""

            question_types_desc = """
- fact: 事実確認型（〜は何ですか？）
- reason: 理由説明型（なぜ〜ですか？）
- comparison: 比較型（〜と〜の違いは？）
- application: 応用型（〜はどのように活用されますか？）"""

            user_prompt = f"""以下のテキストから{num_pairs}個のQ&Aペアを生成してください。

質問タイプ:
{question_types_desc}

テキスト:
{chunk_data['text']}

JSON形式で出力:
{{
  "qa_pairs": [
    {{
      "question": "質問文",
      "answer": "回答文",
      "question_type": "fact/reason/comparison/application"
    }}
  ]
}}"""
        else:
            system_prompt = """You are an expert in educational content creation.
Generate high-quality Q&A pairs from the given English text.

Generation rules:
1. Questions should be clear and specific
2. Answers should be concise and accurate (1-2 sentences)
3. Stay faithful to the text content
4. Create questions from diverse perspectives"""

            question_types_desc = """
- fact: Factual questions (What is...?)
- reason: Explanatory questions (Why...?)
- comparison: Comparative questions (What's the difference...?)
- application: Application questions (How is... used?)"""

            user_prompt = f"""Generate {num_pairs} Q&A pairs from the following text.

Question types:
{question_types_desc}

Text:
{chunk_data['text']}

Output in JSON format:
{{
  "qa_pairs": [
    {{
      "question": "question text",
      "answer": "answer text",
      "question_type": "fact/reason/comparison/application"
    }}
  ]
}}"""

        # OpenAI API呼び出し（構造化出力API - Structured Outputs）
        logger.info(f"OpenAI API呼び出し開始: モデル={model}")

        qa_pairs = []

        # 構造化出力API（client.responses.parse）を使用
        try:
            # プロンプトを結合
            combined_input = f"{system_prompt}\n\n{user_prompt}"

            # 最新のresponses.parse APIを使用
            response = client.responses.parse(
                input=combined_input,
                model=model,
                text_format=QAPairsResponse,  # Pydanticモデルを直接指定
                max_output_tokens=2000  # JSON切れ防止のため増加
            )

            logger.info(f"OpenAI API呼び出し成功（構造化出力API）")

            # レスポンスから解析済みデータを取得
            parsed_response = _extract_parsed_response(response, model)

            # Q/Aペアを抽出
            for qa_data in parsed_response.qa_pairs:
                qa = {
                    "question": qa_data.question,
                    "answer": qa_data.answer,
                    "question_type": qa_data.question_type,
                    "source_chunk_id": chunk_data.get('id', ''),
                    "doc_id": chunk_data.get('doc_id', ''),
                    "dataset_type": chunk_data.get('dataset_type', ''),
                    "chunk_idx": chunk_data.get('chunk_idx', 0)
                }
                qa_pairs.append(qa)

        except Exception as e:
            logger.error(f"構造化出力API呼び出しエラー: {str(e)}")
            logger.info("フォールバック: 通常のChat Completions APIを使用")

            # フォールバック処理
            try:
                # モデルに応じてパラメータを切り替える
                completion_params = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "response_format": {"type": "json_object"},
                }

                # temperatureをサポートするモデルのみ指定
                # GPT-5シリーズ、O-Seriesはtemperature=1のみサポート
                if supports_temperature(model):
                    completion_params["temperature"] = 0.7
                else:
                    logger.info(f"モデル {model} はtemperatureパラメータ非対応のためスキップ")

                # GPT-5シリーズやO3/O4シリーズの場合はmax_completion_tokensを使用
                if model.startswith("gpt-5") or model.startswith("o3") or model.startswith("o4"):
                    completion_params["max_completion_tokens"] = 1000
                else:
                    completion_params["max_tokens"] = 1000

                response = client.chat.completions.create(**completion_params)

                logger.info(f"OpenAI API呼び出し成功（フォールバック）")

                # レスポンス解析（空レスポンス対策）
                import json
                response_text = response.choices[0].message.content

                # 空レスポンスチェック
                if not response_text or response_text.strip() == "":
                    logger.error(f"OpenAI APIが空のレスポンスを返しました")
                    raise ValueError("Empty response from OpenAI API")

                logger.debug(f"API応答（最初の200文字）: {response_text[:200]}...")

                # JSON解析前の検証
                parsed_data = json.loads(response_text)

                for qa_data in parsed_data.get('qa_pairs', []):
                    qa = {
                        "question": qa_data.get('question', ''),
                        "answer": qa_data.get('answer', ''),
                        "question_type": qa_data.get('question_type', 'fact'),
                        "source_chunk_id": chunk_data.get('id', ''),
                        "doc_id": chunk_data.get('doc_id', ''),
                        "dataset_type": chunk_data.get('dataset_type', ''),
                        "chunk_idx": chunk_data.get('chunk_idx', 0)
                    }
                    qa_pairs.append(qa)

            except Exception as api_error:
                logger.error(f"フォールバックAPIも失敗: {str(api_error)}")
                raise

        logger.info(f"タスク完了: チャンク {chunk_data.get('id')} - {len(qa_pairs)}個のQ/A生成")

        return {
            "success": True,
            "chunk_id": chunk_data.get('id'),
            "qa_pairs": qa_pairs,
            "error": None
        }

    except Exception as e:
        logger.error(f"タスクエラー (チャンク {chunk_data.get('id')}): {str(e)}")
        import traceback
        logger.error(f"スタックトレース: {traceback.format_exc()}")

        # リトライ処理
        if self.request.retries < self.max_retries:
            logger.info(f"リトライ {self.request.retries + 1}/{self.max_retries}")
            raise self.retry(exc=e, countdown=5 * (self.request.retries + 1))

        return {
            "success": False,
            "chunk_id": chunk_data.get('id'),
            "qa_pairs": [],
            "error": str(e)
        }


@app.task(bind=True, max_retries=3)
def generate_qa_for_batch_async(self, chunks: List[Dict], config: Dict, model: str = "gpt-4o-mini") -> Dict:
    """
    複数チャンクからQ/Aペアを非同期バッチ生成（Celeryタスク）

    Args:
        chunks: チャンクデータのリスト（1-5個）
        config: データセット設定
        model: 使用するモデル

    Returns:
        生成されたQ/Aペアと関連情報を含む辞書
    """
    try:
        chunk_ids = [c.get('id', 'unknown') for c in chunks]
        logger.info(f"バッチタスク開始: {len(chunks)}チャンク - {chunk_ids}")

        # 単一チャンクの場合は個別処理に委譲
        if len(chunks) == 1:
            return generate_qa_for_chunk_async(chunks[0], config, model)

        # OpenAIクライアント初期化
        client = OpenAI()
        lang = config["lang"]
        all_qa_pairs = []

        # プロンプト構築（言語別）
        if lang == "ja":
            system_prompt = """あなたは教育コンテンツ作成の専門家です。
複数の日本語テキストから、学習効果の高いQ&Aペアを生成してください。

生成ルール:
1. 質問は明確で具体的に
2. 回答は簡潔で正確に（1-2文程度）
3. テキストの内容に忠実に
4. 多様な観点から質問を作成"""
        else:
            system_prompt = """You are an expert in educational content creation.
Generate high-quality Q&A pairs from multiple English texts.

Generation rules:
1. Questions should be clear and specific
2. Answers should be concise and accurate (1-2 sentences)
3. Stay faithful to the text content
4. Create questions from diverse perspectives"""

        # チャンク統合とプロンプト構築
        combined_text = ""
        chunks_data = {}
        total_pairs = 0

        for i, chunk in enumerate(chunks, 1):
            num_pairs = determine_qa_count(chunk, config)
            total_pairs += num_pairs
            chunk_text = chunk['text']

            # 長すぎる場合は短縮
            if len(chunk_text) > 1000:
                chunk_text = chunk_text[:1000] + "..."

            if lang == "ja":
                combined_text += f"\n\n【テキスト{i}】\n{chunk_text}"
            else:
                combined_text += f"\n\n【Text {i}】\n{chunk_text}"

            chunks_data[f"chunk_{i}"] = {"num_pairs": num_pairs, "chunk": chunk}

        # ユーザープロンプト構築
        if lang == "ja":
            user_prompt = f"""以下の{len(chunks)}個のテキストから、合計{total_pairs}個のQ&Aペアを生成してください。
{combined_text}

質問タイプ:
- fact: 事実確認型（〜は何ですか？）
- reason: 理由説明型（なぜ〜ですか？）
- comparison: 比較型（〜と〜の違いは？）
- application: 応用型（〜はどのように活用されますか？）

JSON形式で出力:
{{
  "qa_pairs": [
    {{
      "question": "質問文",
      "answer": "回答文",
      "question_type": "fact/reason/comparison/application"
    }}
  ]
}}"""
        else:
            user_prompt = f"""Generate {total_pairs} Q&A pairs from the following {len(chunks)} texts.
{combined_text}

Question types:
- fact: Factual questions (What is...?)
- reason: Explanatory questions (Why...?)
- comparison: Comparative questions (What's the difference...?)
- application: Application questions (How is... used?)

Output in JSON format:
{{
  "qa_pairs": [
    {{
      "question": "question text",
      "answer": "answer text",
      "question_type": "fact/reason/comparison/application"
    }}
  ]
}}"""

        # OpenAI API呼び出し（構造化出力API - Structured Outputs）
        logger.info(f"OpenAI API呼び出し開始: モデル={model}")

        # 構造化出力API（client.responses.parse）を使用
        try:
            # プロンプトを結合
            combined_input = f"{system_prompt}\n\n{user_prompt}"

            # 最新のresponses.parse APIを使用
            response = client.responses.parse(
                input=combined_input,
                model=model,
                text_format=QAPairsResponse,  # Pydanticモデルを直接指定
                max_output_tokens=4000  # バッチ処理（3チャンク）のJSON切れ防止のため増加
            )

            logger.info(f"OpenAI API呼び出し成功（構造化出力API）")

            # レスポンスから解析済みデータを取得（共通関数を使用）
            parsed_response = _extract_parsed_response(response, model)

            # レスポンス解析とQ/Aペア分配
            qa_index = 0
            for i, chunk in enumerate(chunks, 1):
                chunk_key = f"chunk_{i}"
                expected_pairs = chunks_data[chunk_key]["num_pairs"]

                for _ in range(expected_pairs):
                    if qa_index < len(parsed_response.qa_pairs):
                        qa_data = parsed_response.qa_pairs[qa_index]
                        qa = {
                            "question": qa_data.question,
                            "answer": qa_data.answer,
                            "question_type": qa_data.question_type,
                            "source_chunk_id": chunk.get('id', ''),
                            "doc_id": chunk.get('doc_id', ''),
                            "dataset_type": chunk.get('dataset_type', ''),
                            "chunk_idx": chunk.get('chunk_idx', 0)
                        }
                        all_qa_pairs.append(qa)
                        qa_index += 1

        except Exception as e:
            logger.error(f"構造化出力API呼び出しエラー: {str(e)}")
            logger.info("フォールバック: 通常のChat Completions APIを使用")

            # フォールバック処理
            # モデルに応じてパラメータを切り替える
            completion_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "response_format": {"type": "json_object"},
            }

            # temperatureをサポートするモデルのみ指定
            # GPT-5シリーズ、O-Seriesはtemperature=1のみサポート
            if supports_temperature(model):
                completion_params["temperature"] = 0.7
            else:
                logger.info(f"モデル {model} はtemperatureパラメータ非対応のためスキップ")

            # GPT-5シリーズやO3/O4シリーズの場合はmax_completion_tokensを使用
            if model.startswith("gpt-5") or model.startswith("o3") or model.startswith("o4"):
                completion_params["max_completion_tokens"] = 2000
            else:
                completion_params["max_tokens"] = 2000

            response = client.chat.completions.create(**completion_params)

            import json
            response_text = response.choices[0].message.content

            if not response_text or response_text.strip() == "":
                raise ValueError("Empty response from OpenAI API")

            parsed_data = json.loads(response_text)

            # レスポンス解析とQ/Aペア分配
            qa_index = 0
            for i, chunk in enumerate(chunks, 1):
                chunk_key = f"chunk_{i}"
                expected_pairs = chunks_data[chunk_key]["num_pairs"]

                for _ in range(expected_pairs):
                    if qa_index < len(parsed_data.get('qa_pairs', [])):
                        qa_data = parsed_data['qa_pairs'][qa_index]
                        qa = {
                            "question": qa_data.get('question', ''),
                            "answer": qa_data.get('answer', ''),
                            "question_type": qa_data.get('question_type', 'fact'),
                            "source_chunk_id": chunk.get('id', ''),
                            "doc_id": chunk.get('doc_id', ''),
                            "dataset_type": chunk.get('dataset_type', ''),
                            "chunk_idx": chunk.get('chunk_idx', 0)
                        }
                        all_qa_pairs.append(qa)
                        qa_index += 1

        logger.info(f"バッチタスク完了: {len(chunks)}チャンク - {len(all_qa_pairs)}個のQ/A生成")

        return {
            "success": True,
            "chunk_ids": chunk_ids,
            "qa_pairs": all_qa_pairs,
            "error": None
        }

    except Exception as e:
        logger.error(f"バッチタスクエラー: {str(e)}")

        # リトライ処理
        if self.request.retries < self.max_retries:
            logger.info(f"リトライ {self.request.retries + 1}/{self.max_retries}")
            raise self.retry(exc=e, countdown=5 * (self.request.retries + 1))

        return {
            "success": False,
            "chunk_ids": chunk_ids,
            "qa_pairs": [],
            "error": str(e)
        }


def submit_parallel_qa_generation(chunks: List[Dict], config: Dict, model: str = "gpt-4o-mini",
                                 batch_size: int = 3) -> List:
    """
    並列Q/A生成ジョブを投入

    Args:
        chunks: チャンクのリスト
        config: データセット設定
        model: 使用するモデル
        batch_size: バッチサイズ（1-5）

    Returns:
        Celeryタスクのリスト
    """
    tasks = []

    # バッチ処理の場合
    if batch_size > 1:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            task = generate_qa_for_batch_async.apply_async(
                args=[batch, config, model],
                queue='qa_generation'  # ワーカーが監視しているキューを指定
            )
            tasks.append(task)
            logger.debug(f"タスク投入: {task.id} - {len(batch)}チャンク")
    else:
        # 個別処理の場合
        for chunk in chunks:
            task = generate_qa_for_chunk_async.apply_async(
                args=[chunk, config, model],
                queue='qa_generation'  # ワーカーが監視しているキューを指定
            )
            tasks.append(task)

    logger.info(f"投入されたタスク数: {len(tasks)}")
    return tasks


def collect_results(tasks: List, timeout: int = 300) -> List[Dict]:
    """
    並列処理の結果を収集（改良版：Redis直接アクセスで確実な結果取得）

    Args:
        tasks: Celeryタスクのリスト
        timeout: タイムアウト（秒）

    Returns:
        Q/Aペアのリスト
    """
    import time
    import redis
    import json
    from celery.result import AsyncResult

    all_qa_pairs = []
    failed_chunks = []
    total_tasks = len(tasks)
    completed_tasks = set()  # 完了済みタスクのインデックス
    failed_tasks = set()     # 失敗タスクのインデックス

    logger.info(f"結果収集開始: {total_tasks}個のタスク")

    # Redis接続を確立
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=int(os.getenv('REDIS_DB', 0)),
        decode_responses=True
    )

    # タスクIDをログ出力
    for i, task in enumerate(tasks):
        logger.debug(f"タスク {i+1}: ID={task.id}")

    start_time = time.time()
    last_log_time = start_time
    stall_check_time = start_time
    last_completed_count = 0
    stall_counter = 0

    # シンプルなループで結果を収集
    while len(completed_tasks) + len(failed_tasks) < total_tasks:
        current_time = time.time()
        elapsed = current_time - start_time

        # タイムアウトチェック
        if elapsed > timeout:
            logger.error(f"タイムアウト: {elapsed:.1f}秒経過")
            logger.info(f"収集済み: {len(completed_tasks)}個, 未収集: {total_tasks - len(completed_tasks) - len(failed_tasks)}個")
            break

        # 5秒ごとに進捗表示
        if current_time - last_log_time >= 5:
            logger.info(f"進捗: 完了={len(completed_tasks)}/{total_tasks}, "
                       f"失敗={len(failed_tasks)}, "
                       f"処理中={total_tasks - len(completed_tasks) - len(failed_tasks)}, "
                       f"経過時間={elapsed:.1f}秒")
            last_log_time = current_time

        # 30秒ごとに停滞チェック
        if current_time - stall_check_time >= 30:
            current_completed = len(completed_tasks)
            if current_completed == last_completed_count:
                stall_counter += 1
                logger.warning(f"⚠️ 処理が停滞している可能性があります（{stall_counter * 30}秒間進捗なし）")

                # 3分間進捗がない場合、状態を強制リフレッシュ
                if stall_counter >= 6:
                    logger.warning("📊 タスク状態を強制的に再取得...")
                    for i, task in enumerate(tasks):
                        if i not in completed_tasks and i not in failed_tasks:
                            try:
                                # AsyncResultで再取得
                                refreshed_task = AsyncResult(task.id)
                                if refreshed_task.state in ['SUCCESS', 'FAILURE']:
                                    logger.info(f"タスク {i+1} の状態: {refreshed_task.state}")
                            except Exception as e:
                                logger.debug(f"タスク {i+1} の状態取得エラー: {str(e)}")
                    stall_counter = 0
            else:
                stall_counter = 0
                last_completed_count = current_completed
            stall_check_time = current_time

        # 各タスクをチェック
        pending_indices = []
        for i, task in enumerate(tasks):
            # 既に処理済みのタスクはスキップ
            if i in completed_tasks or i in failed_tasks:
                continue

            try:
                # まずRedisから直接状態を確認（最も確実な方法）
                redis_key = f"celery-task-meta-{task.id}"
                redis_data = redis_client.get(redis_key)

                result = None
                redis_state = None

                if redis_data:
                    try:
                        redis_result = json.loads(redis_data)
                        redis_state = redis_result.get('status')

                        # Redisで成功している場合は結果を直接取得
                        if redis_state == 'SUCCESS':
                            result = redis_result.get('result')
                            if result and isinstance(result, dict) and result.get('success'):
                                # Redis直接取得成功
                                logger.debug(f"タスク {i+1} Redis直接取得成功")
                    except json.JSONDecodeError:
                        logger.warning(f"タスク {i+1} Redis JSONデコードエラー")

                # Redisから正常な結果を取得した場合は処理
                if result and isinstance(result, dict) and result.get('success'):
                    qa_pairs = result.get('qa_pairs', [])
                    all_qa_pairs.extend(qa_pairs)
                    completed_tasks.add(i)

                    # チャンク情報をログ
                    if 'chunk_id' in result:
                        logger.info(f"✓ タスク {i+1}/{total_tasks} 完了（Redis直接）: チャンク {result['chunk_id']} - {len(qa_pairs)}個のQ/A")
                    elif 'chunk_ids' in result:
                        logger.info(f"✓ タスク {i+1}/{total_tasks} 完了（Redis直接）: バッチ {len(result['chunk_ids'])}チャンク - {len(qa_pairs)}個のQ/A")
                else:
                    # Redisから取得できない場合は従来の方法も試す
                    task_state = task.state

                    # デバッグ: 最後のタスクの状態を詳細ログ
                    if i == total_tasks - 1:
                        logger.debug(f"🔍 最後のタスク[{i}] Redis状態={redis_state}, Celery状態={task_state}, id={task.id[:8]}...")

                    # PENDINGでもRedisにSUCCESSがある場合、またはready()の場合
                    if task_state == 'SUCCESS' or task.ready() or (task_state == 'PENDING' and redis_state == 'SUCCESS'):
                        # 結果を取得（短いタイムアウト）
                        try:
                            result = task.get(timeout=1, propagate=False)
                        except:
                            pass

                        if result and isinstance(result, dict) and result.get('success'):
                            qa_pairs = result.get('qa_pairs', [])
                            all_qa_pairs.extend(qa_pairs)
                            completed_tasks.add(i)

                            # チャンク情報をログ
                            if 'chunk_id' in result:
                                logger.info(f"✓ タスク {i+1}/{total_tasks} 完了: チャンク {result['chunk_id']} - {len(qa_pairs)}個のQ/A")
                            elif 'chunk_ids' in result:
                                logger.info(f"✓ タスク {i+1}/{total_tasks} 完了: バッチ {len(result['chunk_ids'])}チャンク - {len(qa_pairs)}個のQ/A")
                        elif result is not None:
                            failed_tasks.add(i)
                            error_msg = result.get('error', 'Unknown error') if result and isinstance(result, dict) else str(result)
                            logger.error(f"✗ タスク {i+1}/{total_tasks} 失敗: {error_msg[:200]}")

                            if result and isinstance(result, dict):
                                if 'chunk_id' in result:
                                    failed_chunks.append(result['chunk_id'])
                                elif 'chunk_ids' in result:
                                    failed_chunks.extend(result['chunk_ids'])

                    elif task_state == 'FAILURE':
                        failed_tasks.add(i)
                        logger.error(f"✗ タスク {i+1}/{total_tasks} 失敗（状態: FAILURE）")

                    elif task_state == 'PENDING':
                        pending_indices.append(i)

            except TimeoutError:
                # タイムアウトは正常（まだ処理中）
                pass
            except Exception as e:
                # その他のエラーは1回だけログ
                if i not in failed_tasks:
                    logger.debug(f"タスク {i+1} チェック中にエラー（処理継続）: {str(e)[:100]}")

        # デバッグ: ペンディング状態のタスクが多い場合警告
        if len(pending_indices) > 10 and current_time - start_time > 60:
            logger.debug(f"ペンディング状態のタスク数: {len(pending_indices)}")

        # 短い待機（処理済みタスクが多い場合は待機時間を長くする）
        completion_ratio = len(completed_tasks) / total_tasks if total_tasks > 0 else 0
        if completion_ratio > 0.9:
            time.sleep(1.0)  # 90%以上完了したら待機時間を長く
        else:
            time.sleep(0.5)

    # 最終的な未完了タスクの処理（より積極的に取得）
    logger.info(f"ループ終了後の最終確認: 未処理タスク {total_tasks - len(completed_tasks) - len(failed_tasks)}個")
    for i, task in enumerate(tasks):
        if i not in completed_tasks and i not in failed_tasks:
            try:
                # AsyncResultで強制的に最新状態を取得
                from celery.result import AsyncResult
                refreshed = AsyncResult(task.id)
                final_state = refreshed.state
                logger.info(f"タスク {i+1} 最終状態: {final_state}")

                # 状態に関わらず結果取得を試みる
                if final_state in ['SUCCESS', 'FAILURE'] or refreshed.ready():
                    result = refreshed.get(timeout=3, propagate=False)
                    if result and isinstance(result, dict) and result.get('success'):
                        qa_pairs = result.get('qa_pairs', [])
                        all_qa_pairs.extend(qa_pairs)
                        completed_tasks.add(i)
                        logger.info(f"✓ タスク {i+1}/{total_tasks} 最終収集で完了: {len(qa_pairs)}個のQ/A")
                    else:
                        failed_tasks.add(i)
                        logger.warning(f"タスク {i+1}/{total_tasks} 最終収集で失敗として処理")
                else:
                    logger.warning(f"タスク {i+1}/{total_tasks} 最終状態: {final_state} - 未完了")
            except Exception as e:
                logger.warning(f"タスク {i+1}/{total_tasks} 最終収集エラー: {str(e)[:100]}")

    # 結果サマリー
    elapsed_total = time.time() - start_time
    logger.info(f"""
    =====================================
    結果収集完了:
    - 成功: {len(completed_tasks)}/{total_tasks}タスク
    - 失敗: {len(failed_tasks)}タスク
    - 未完了: {total_tasks - len(completed_tasks) - len(failed_tasks)}タスク
    - 生成Q/Aペア: {len(all_qa_pairs)}個
    - 所要時間: {elapsed_total:.1f}秒
    =====================================
    """)

    if failed_chunks:
        logger.warning(f"失敗したチャンク（最初の5個）: {failed_chunks[:5]}")

    return all_qa_pairs


if __name__ == "__main__":
    # Celeryワーカーを起動する場合
    # celery -A celery_tasks worker --loglevel=info --concurrency=4
    pass