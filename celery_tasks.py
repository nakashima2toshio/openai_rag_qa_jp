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
from pydantic import BaseModel

# 環境変数読み込み
load_dotenv()

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Celeryアプリケーション設定
app = Celery(
    'qa_generation',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
)

# Celery設定
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Tokyo',
    enable_utc=True,
    # タスクのタイムアウト設定
    task_time_limit=300,  # 5分
    task_soft_time_limit=240,  # 4分（ソフトリミット）
    # 並列度の制御
    worker_concurrency=4,  # ワーカー並列度
    worker_prefetch_multiplier=1,  # プリフェッチ数
    # リトライ設定
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)


class QAPair(BaseModel):
    """Q/Aペアのデータモデル"""
    question: str
    answer: str
    question_type: str
    source_chunk_id: Optional[str] = None
    dataset_type: Optional[str] = None
    auto_generated: bool = False


class QAPairsResponse(BaseModel):
    """Q/Aペア生成レスポンス"""
    qa_pairs: List[QAPair]


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


@app.task(bind=True, max_retries=3)
def generate_qa_for_chunk_async(self, chunk_data: Dict, config: Dict, model: str = "gpt-5-mini") -> Dict:
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

        # OpenAI Chat Completions API呼び出し（標準的な方法）
        logger.info(f"OpenAI API呼び出し開始: モデル={model}")

        try:
            # 標準的なChat Completions APIを使用
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
                response_format={"type": "json_object"}  # JSON形式を強制
            )

            logger.info(f"OpenAI API呼び出し成功")

            # レスポンス解析
            import json
            response_text = response.choices[0].message.content
            logger.debug(f"API応答（最初の200文字）: {response_text[:200]}...")

            parsed_data = json.loads(response_text)
            qa_pairs = []

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
            logger.error(f"OpenAI API呼び出しエラー: {str(api_error)}")
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
def generate_qa_for_batch_async(self, chunks: List[Dict], config: Dict, model: str = "gpt-5-mini") -> Dict:
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

        # OpenAI API呼び出し
        combined_input = f"{system_prompt}\n\n{user_prompt}"

        response = client.responses.parse(
            input=combined_input,
            model=model,
            text_format=QAPairsResponse,
            max_output_tokens=4000
        )

        # レスポンス解析とQ/Aペア分配
        for output in response.output:
            if output.type == "message":
                for item in output.content:
                    if item.type == "output_text" and item.parsed:
                        parsed_data = item.parsed

                        # 各チャンクにQ/Aを割り当て
                        qa_index = 0
                        for i, chunk in enumerate(chunks, 1):
                            chunk_key = f"chunk_{i}"
                            expected_pairs = chunks_data[chunk_key]["num_pairs"]

                            for _ in range(expected_pairs):
                                if qa_index < len(parsed_data.qa_pairs):
                                    qa_data = parsed_data.qa_pairs[qa_index]
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


def submit_parallel_qa_generation(chunks: List[Dict], config: Dict, model: str = "gpt-5-mini",
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
                args=[batch, config, model]
                # queue引数を削除 - デフォルトキューを使用
            )
            tasks.append(task)
            logger.debug(f"タスク投入: {task.id} - {len(batch)}チャンク")
    else:
        # 個別処理の場合
        for chunk in chunks:
            task = generate_qa_for_chunk_async.apply_async(
                args=[chunk, config, model]
                # queue引数を削除
            )
            tasks.append(task)

    logger.info(f"投入されたタスク数: {len(tasks)}")
    return tasks


def collect_results(tasks: List, timeout: int = 300) -> List[Dict]:
    """
    並列処理の結果を収集（シンプルで確実な方法）

    Args:
        tasks: Celeryタスクのリスト
        timeout: タイムアウト（秒）

    Returns:
        Q/Aペアのリスト
    """
    import time

    all_qa_pairs = []
    failed_chunks = []
    total_tasks = len(tasks)
    completed_tasks = set()  # 完了済みタスクのインデックス
    failed_tasks = set()     # 失敗タスクのインデックス

    logger.info(f"結果収集開始: {total_tasks}個のタスク")

    # タスクIDをログ出力
    for i, task in enumerate(tasks):
        logger.debug(f"タスク {i+1}: ID={task.id}")

    start_time = time.time()
    last_log_time = start_time

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

        # 各タスクをチェック
        for i, task in enumerate(tasks):
            # 既に処理済みのタスクはスキップ
            if i in completed_tasks or i in failed_tasks:
                continue

            try:
                # タスクが完了しているかチェック（ノンブロッキング）
                if task.ready():
                    # 結果を取得（短いタイムアウト）
                    result = task.get(timeout=1)

                    if result and result.get('success'):
                        qa_pairs = result.get('qa_pairs', [])
                        all_qa_pairs.extend(qa_pairs)
                        completed_tasks.add(i)

                        # チャンク情報をログ
                        if 'chunk_id' in result:
                            logger.info(f"✓ タスク {i+1}/{total_tasks} 完了: チャンク {result['chunk_id']} - {len(qa_pairs)}個のQ/A")
                        elif 'chunk_ids' in result:
                            logger.info(f"✓ タスク {i+1}/{total_tasks} 完了: バッチ {len(result['chunk_ids'])}チャンク - {len(qa_pairs)}個のQ/A")
                    else:
                        failed_tasks.add(i)
                        error_msg = result.get('error', 'Unknown error') if result else 'No result'
                        logger.error(f"✗ タスク {i+1}/{total_tasks} 失敗: {error_msg}")

                        if result:
                            if 'chunk_id' in result:
                                failed_chunks.append(result['chunk_id'])
                            elif 'chunk_ids' in result:
                                failed_chunks.extend(result['chunk_ids'])

            except TimeoutError:
                # タイムアウトは正常（まだ処理中）
                pass
            except Exception as e:
                # その他のエラーは1回だけログ
                if i not in failed_tasks:
                    logger.debug(f"タスク {i+1} チェック中にエラー（処理継続）: {str(e)[:100]}")

        # 短い待機
        time.sleep(0.5)

    # 最終的な未完了タスクの処理
    for i, task in enumerate(tasks):
        if i not in completed_tasks and i not in failed_tasks:
            try:
                # 最後のチャンス（少し長めのタイムアウト）
                if task.ready():
                    result = task.get(timeout=2)
                    if result and result.get('success'):
                        qa_pairs = result.get('qa_pairs', [])
                        all_qa_pairs.extend(qa_pairs)
                        completed_tasks.add(i)
                        logger.info(f"✓ タスク {i+1}/{total_tasks} 最終収集で完了: {len(qa_pairs)}個のQ/A")
            except:
                logger.warning(f"タスク {i+1}/{total_tasks} 未完了のまま終了")

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