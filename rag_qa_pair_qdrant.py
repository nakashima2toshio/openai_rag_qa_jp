#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# ワーカーを再起動
=============================================
redis-cli FLUSHDB && ./start_celery.sh restart -w 24

=============================================
rag_qa_pair_qdrant.py - RAGデータダウンロード・前処理・Q/A生成ツール

=============================================
起動: streamlit run rag_qa_pair_qdrant.py --server.port=8500

【主要機能】
✅ 4つのデータセットから選択してダウンロード
✅ ローカルファイルのアップロード対応（CSV/TXT/JSON）
✅ 自動前処理・クレンジング
✅ Q/Aペア自動生成（OpenAI API使用）
✅ qa_output/フォルダへ自動保存
✅ 進捗・履歴のリアルタイム表示

【対応データセット】
1. wikipedia_ja: Wikipedia日本語版
2. japanese_text: CC100日本語
3. cc_news: CC-News英語ニュース
4. livedoor: Livedoorニュースコーパス
5. custom_upload: ローカルファイルアップロード
"""

import streamlit as st
import pandas as pd
import json
import urllib.request
import tarfile
import logging
import socket
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterable, Tuple
import time
import tiktoken
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import io
import subprocess
import os
import sys
import tempfile

# Qdrant関連インポート
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

# 環境変数読み込み
load_dotenv()

# ローカルモジュール（必要な関数のみインポート）
from helper_rag import clean_text

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================================================================
# Pydantic モデル定義（Q/A生成用）
# ===================================================================


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


# ===================================================================
# データセット設定
# ===================================================================

DATASET_CONFIGS = {
    "wikipedia_ja": {
        "name": "Wikipedia日本語版",
        "icon": "📚",
        "description": "Wikipedia日本語版の記事データ（百科事典的知識）",
        "hf_dataset": "wikimedia/wikipedia",
        "hf_config": "20231101.ja",
        "split": "train",
        "text_field": "text",
        "title_field": "title",
        "sample_size": 1000,
        "min_text_length": 200,
    },
    "japanese_text": {
        "name": "日本語Webテキスト（CC100）",
        "icon": "📰",
        "description": "日本語Webテキストコーパス",
        "hf_dataset": "range3/cc100-ja",
        "hf_config": None,
        "split": "train",
        "text_field": "text",
        "title_field": None,
        "sample_size": 1000,
        "min_text_length": 10,
    },
    "cc_news": {
        "name": "CC-News（英語ニュース）",
        "icon": "🌐",
        "description": "Common Crawl英語ニュース記事",
        "hf_dataset": "cc_news",
        "hf_config": None,
        "split": "train",
        "text_field": "text",
        "title_field": "title",
        "sample_size": 500,
        "min_text_length": 100,
    },
    "livedoor": {
        "name": "Livedoorニュースコーパス",
        "icon": "📰",
        "description": "Livedoorニュース日本語記事（9カテゴリ、全7,376件）",
        "hf_dataset": None,
        "download_url": "https://www.rondhuit.com/download/ldcc-20140209.tar.gz",
        "split": None,
        "text_field": "content",
        "title_field": "title",
        "sample_size": 7376,
        "min_text_length": 100,
    },
}

# ===================================================================
# Livedoorコーパス用関数
# ===================================================================


def download_livedoor_corpus(save_dir: str = "datasets") -> str:
    """Livedoorニュースコーパスをダウンロード"""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
    tar_filename = "ldcc-20140209.tar.gz"
    tar_path = save_path / tar_filename

    # ダウンロード
    if not tar_path.exists():
        logger.info(f"Livedoorニュースコーパスをダウンロード中: {url}")
        urllib.request.urlretrieve(url, tar_path)
        logger.info(f"ダウンロード完了: {tar_path}")

    # 解凍
    extract_dir = save_path / "livedoor"
    text_dir = extract_dir / "text"

    if not text_dir.exists():
        logger.info(f"アーカイブを解凍中: {tar_path}")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_dir, filter="data")
        logger.info(f"解凍完了: {extract_dir}")

    return str(extract_dir)


def load_livedoor_corpus(data_dir: str) -> pd.DataFrame:
    """Livedoorニュースコーパスを読み込み"""
    categories = [
        "dokujo-tsushin",
        "it-life-hack",
        "kaden-channel",
        "livedoor-homme",
        "movie-enter",
        "peachy",
        "smax",
        "sports-watch",
        "topic-news",
    ]

    articles = []
    text_dir = Path(data_dir) / "text"

    for category in categories:
        category_path = text_dir / category
        if not category_path.exists():
            logger.warning(f"カテゴリディレクトリが見つかりません: {category_path}")
            continue

        txt_files = list(category_path.glob("*.txt"))

        for file_path in txt_files:
            if file_path.name in ["LICENSE.txt", "README.txt"]:
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                if len(lines) >= 3:
                    url = lines[0].strip()
                    date = lines[1].strip() if len(lines) > 1 else ""
                    title = lines[2].strip() if len(lines) > 2 else ""
                    content = "".join(lines[3:]).strip() if len(lines) > 3 else ""

                    articles.append(
                        {
                            "url": url,
                            "date": date,
                            "title": title,
                            "content": content,
                            "category": category,
                        }
                    )
            except Exception as e:
                logger.error(f"ファイル読み込みエラー {file_path}: {e}")

    df = pd.DataFrame(articles)
    logger.info(f"Livedoorコーパス読み込み完了: {len(df)}記事")
    return df


# ===================================================================
# データ処理関数
# ===================================================================


def extract_text_content(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """データセットからテキストコンテンツを抽出"""
    text_field = config["text_field"]
    title_field = config["title_field"]

    df_processed = df.copy()

    # タイトルとテキストを結合
    if title_field and title_field in df.columns and text_field in df.columns:
        df_processed["Combined_Text"] = df_processed.apply(
            lambda row: f"{clean_text(str(row.get(title_field, '')))} {clean_text(str(row.get(text_field, '')))}".strip(),
            axis=1,
        )
    elif text_field in df.columns:
        df_processed["Combined_Text"] = df_processed[text_field].apply(
            lambda x: clean_text(str(x)) if x is not None else ""
        )
    else:
        # フォールバック: 利用可能なテキストフィールドを探す
        text_candidates = ["text", "content", "body", "document", "abstract"]
        found_field = None
        for field in text_candidates:
            if field in df.columns:
                found_field = field
                break

        if found_field:
            df_processed["Combined_Text"] = df_processed[found_field].apply(
                lambda x: clean_text(str(x)) if x is not None else ""
            )
        else:
            df_processed["Combined_Text"] = df_processed.apply(
                lambda row: " ".join([str(v) for v in row.values if v is not None]),
                axis=1,
            )

    # 空のテキストを除外
    df_processed = df_processed[df_processed["Combined_Text"].str.strip() != ""]

    return df_processed


def download_hf_dataset(
    dataset_name: str,
    config_name: Optional[str],
    split: str,
    sample_size: int,
    log_callback,
) -> pd.DataFrame:
    """HuggingFaceからデータセットをダウンロード"""
    from datasets import load_dataset as hf_load_dataset

    samples = []

    if dataset_name == "wikimedia/wikipedia":
        actual_config = config_name if config_name else "20231101.ja"
        log_callback(f"📥 {dataset_name} をロード中 (config: {actual_config})...")
        dataset = hf_load_dataset(
            dataset_name, actual_config, split=split, streaming=True
        )

        for i, item in enumerate(dataset):
            if i >= sample_size:
                break
            samples.append(item)
            if (i + 1) % 100 == 0:
                log_callback(f"進捗: {i + 1}/{sample_size} 件")

    elif dataset_name == "range3/cc100-ja":
        log_callback(f"📥 {dataset_name} をロード中...")
        dataset = hf_load_dataset(dataset_name, split=split, streaming=True)

        for i, item in enumerate(dataset):
            if i >= sample_size:
                break
            samples.append(item)
            if (i + 1) % 100 == 0:
                log_callback(f"進捗: {i + 1}/{sample_size} 件")

    elif dataset_name == "cc_news":
        log_callback(f"📥 {dataset_name} をロード中...")
        if config_name:
            dataset = hf_load_dataset(
                dataset_name, config_name, split=split, streaming=True
            )
        else:
            dataset = hf_load_dataset(dataset_name, split=split, streaming=True)

        for i, item in enumerate(dataset):
            if i >= sample_size:
                break
            samples.append(item)
            if (i + 1) % 50 == 0:
                log_callback(f"進捗: {i + 1}/{sample_size} 件")

    else:
        raise ValueError(f"未対応のデータセット: {dataset_name}")

    df = pd.DataFrame(samples)
    log_callback(f"✅ {len(df)} 件のデータをロードしました")
    return df


def save_to_output(df: pd.DataFrame, dataset_type: str) -> Dict[str, str]:
    """OUTPUTフォルダに保存"""
    output_dir = Path("OUTPUT")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {}

    # CSVファイル
    csv_filename = f"preprocessed_{dataset_type}_{timestamp}.csv"
    csv_path = output_dir / csv_filename
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    saved_files["csv"] = str(csv_path)

    # テキストファイル
    txt_filename = f"{dataset_type}_{timestamp}.txt"
    txt_path = output_dir / txt_filename
    text_data = "\n".join(df["Combined_Text"].dropna().astype(str))
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_data)
    saved_files["txt"] = str(txt_path)

    # メタデータ
    config = DATASET_CONFIGS.get(dataset_type, {})
    metadata = {
        "dataset_type": dataset_type,
        "dataset_name": config.get("name", dataset_type),
        "processed_at": datetime.now().isoformat(),
        "row_count": len(df),
        "csv_file": csv_filename,
        "txt_file": txt_filename,
    }
    json_filename = f"metadata_{dataset_type}_{timestamp}.json"
    json_path = output_dir / json_filename
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    saved_files["json"] = str(json_path)

    return saved_files


def load_qa_output_history() -> pd.DataFrame:
    """qa_output/フォルダから最新のQ&AペアCSVファイル一覧を取得"""
    qa_output_dir = Path("qa_output")

    if not qa_output_dir.exists():
        return pd.DataFrame(columns=["ファイル名", "ファイルサイズ", "作成日付"])

    # CSVファイルを全て取得
    csv_files = list(qa_output_dir.glob("*.csv"))

    if not csv_files:
        return pd.DataFrame(columns=["ファイル名", "ファイルサイズ", "作成日付"])

    history_data = []

    for csv_file in csv_files:
        try:
            # ファイル情報を取得
            file_stat = csv_file.stat()
            file_size = file_stat.st_size
            created_time = datetime.fromtimestamp(file_stat.st_mtime)

            # ファイルサイズを人間が読みやすい形式に変換
            if file_size < 1024:
                size_str = f"{file_size} B"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size / (1024 * 1024):.1f} MB"

            history_data.append(
                {
                    "ファイル名": csv_file.name,
                    "ファイルサイズ": size_str,
                    "作成日付": created_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "_timestamp": created_time,  # ソート用
                }
            )

        except Exception as e:
            logger.error(f"ファイル情報取得エラー {csv_file}: {e}")
            continue

    # DataFrameに変換して日付でソート（昇順：古いものが上）
    df_history = pd.DataFrame(history_data)

    if not df_history.empty:
        df_history = df_history.sort_values("_timestamp", ascending=True)
        df_history = df_history.drop(columns=["_timestamp"])  # ソート用カラムを削除

    return df_history


def load_preprocessed_history() -> pd.DataFrame:
    """OUTPUT/フォルダから前処理済みCSVファイル一覧を取得"""
    output_dir = Path("OUTPUT")

    if not output_dir.exists():
        return pd.DataFrame(columns=["ファイル名", "ファイルサイズ", "作成日付", "データセット名"])

    # preprocessed_*.csvファイルを全て取得
    csv_files = list(output_dir.glob("preprocessed_*.csv"))

    if not csv_files:
        return pd.DataFrame(columns=["ファイル名", "ファイルサイズ", "作成日付", "データセット名"])

    history_data = []

    for csv_file in csv_files:
        try:
            # ファイル情報を取得
            file_stat = csv_file.stat()
            file_size = file_stat.st_size
            created_time = datetime.fromtimestamp(file_stat.st_mtime)

            # ファイルサイズを人間が読みやすい形式に変換
            if file_size < 1024:
                size_str = f"{file_size} B"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size / (1024 * 1024):.1f} MB"

            # データセット名を抽出（preprocessed_XXX.csv → XXX）
            dataset_name = csv_file.stem.replace("preprocessed_", "")

            history_data.append(
                {
                    "ファイル名": csv_file.name,
                    "データセット名": dataset_name,
                    "ファイルサイズ": size_str,
                    "作成日付": created_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "_timestamp": created_time,  # ソート用
                }
            )

        except Exception as e:
            logger.error(f"ファイル情報取得エラー {csv_file}: {e}")
            continue

    # DataFrameに変換して日付でソート（降順：新しいものが上）
    df_history = pd.DataFrame(history_data)

    if not df_history.empty:
        df_history = df_history.sort_values("_timestamp", ascending=False)
        df_history = df_history.drop(columns=["_timestamp"])  # ソート用カラムを削除

    return df_history


# ===================================================================
# ローカルファイル読み込み関数
# ===================================================================


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    アップロードされたファイルを読み込み

    Args:
        uploaded_file: Streamlitのfile_uploaderで取得したファイル

    Returns:
        pd.DataFrame: Combined_Textカラムを含むDataFrame
    """
    file_extension = uploaded_file.name.split(".")[-1].lower()

    try:
        if file_extension == "csv":
            # CSVファイル
            df = pd.read_csv(uploaded_file)

        elif file_extension in ["txt", "text"]:
            # テキストファイル（1行1ドキュメント）
            content = uploaded_file.read().decode("utf-8")
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            df = pd.DataFrame({"text": lines})

        elif file_extension == "json":
            # JSONファイル
            content = uploaded_file.read().decode("utf-8")
            data = json.loads(content)

            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError(
                    "JSONファイルはリストまたはオブジェクトである必要があります"
                )

        elif file_extension == "jsonl":
            # JSON Linesファイル
            content = uploaded_file.read().decode("utf-8")
            lines = [json.loads(line) for line in content.split("\n") if line.strip()]
            df = pd.DataFrame(lines)

        else:
            raise ValueError(f"未対応のファイル形式: {file_extension}")

        # Combined_Textカラムの作成
        if "Combined_Text" not in df.columns:
            # テキストフィールドを探す
            text_candidates = [
                "text",
                "content",
                "body",
                "document",
                "answer",
                "question",
            ]
            found_field = None

            for field in text_candidates:
                if field in df.columns:
                    found_field = field
                    break

            if found_field:
                df["Combined_Text"] = df[found_field].apply(
                    lambda x: clean_text(str(x)) if x is not None else ""
                )
            else:
                # 全カラムを結合
                df["Combined_Text"] = df.apply(
                    lambda row: " ".join([str(v) for v in row.values if v is not None]),
                    axis=1,
                )

        # 空のテキストを除外
        df = df[df["Combined_Text"].str.strip() != ""]
        df = df.reset_index(drop=True)

        return df

    except Exception as e:
        logger.error(f"ファイル読み込みエラー: {e}")
        raise


# ===================================================================
# 高度なQ/A生成関数（a02_make_qa_para.py実行）
# ===================================================================


def run_advanced_qa_generation(
    dataset: Optional[str],
    input_file: Optional[str],
    use_celery: bool,
    celery_workers: int,
    batch_chunks: int,
    max_docs: int,
    merge_chunks: bool,
    min_tokens: int,
    max_tokens: int,
    coverage_threshold: float,
    model: str,
    analyze_coverage: bool,
    log_callback,
) -> Dict[str, Any]:
    """
    a02_make_qa_para.pyをサブプロセスで実行

    Args:
        dataset: データセット名
        input_file: 入力ファイルパス
        use_celery: Celery並列処理を使用
        celery_workers: Celeryワーカー数
        batch_chunks: バッチチャンク数
        max_docs: 最大ドキュメント数
        merge_chunks: チャンク統合
        min_tokens: 最小トークン数
        max_tokens: 最大トークン数
        coverage_threshold: カバレージ閾値
        model: 使用モデル
        analyze_coverage: カバレージ分析を実行
        log_callback: ログコールバック関数

    Returns:
        実行結果の辞書
    """
    import threading
    import queue

    # コマンド構築
    cmd = [sys.executable, "a02_make_qa_para.py"]

    if dataset:
        cmd.extend(["--dataset", dataset])
    elif input_file:
        cmd.extend(["--input-file", input_file])

    if use_celery:
        cmd.append("--use-celery")
        cmd.extend(["--celery-workers", str(celery_workers)])

    cmd.extend(
        [
            "--batch-chunks",
            str(batch_chunks),
            "--max-docs",
            str(max_docs),
            "--min-tokens",
            str(min_tokens),
            "--max-tokens",
            str(max_tokens),
            "--coverage-threshold",
            str(coverage_threshold),
            "--model",
            model,
        ]
    )

    if merge_chunks:
        cmd.append("--merge-chunks")

    if analyze_coverage:
        cmd.append("--analyze-coverage")

    # 環境変数を現在のプロセスからコピー
    env = os.environ.copy()

    log_callback(f"🚀 高度なQ/A生成を開始: {' '.join(cmd)}")

    # 出力をキューに格納
    output_queue = queue.Queue()

    def read_output(pipe, queue):
        """サブプロセスの出力を読み取る"""
        for line in iter(pipe.readline, ""):
            if line:
                queue.put(line.strip())
        pipe.close()

    try:
        # サブプロセス起動
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env,
        )

        # 出力読み取りスレッド開始
        thread = threading.Thread(
            target=read_output, args=(process.stdout, output_queue)
        )
        thread.daemon = True
        thread.start()

        # リアルタイムでログを処理
        saved_files = None
        qa_count = 0
        coverage_results = None

        while True:
            # プロセスが終了したかチェック
            poll = process.poll()

            # キューから出力を取得
            try:
                line = output_queue.get(timeout=0.1)
                log_callback(line)

                # 結果ファイルのパスを抽出
                if "CSV保存:" in line:
                    csv_match = line.split("CSV保存:")[-1].strip()
                    if saved_files is None:
                        saved_files = {}
                    saved_files["csv"] = f"qa_output/{csv_match}"

                elif "JSON保存:" in line:
                    json_match = line.split("JSON保存:")[-1].strip()
                    if saved_files:
                        saved_files["json"] = f"qa_output/{json_match}"

                elif "生成Q/Aペア数:" in line or "生成Q/Aペア:" in line:
                    # Q/A数を抽出
                    import re

                    # "生成Q/Aペア数: 118" または "生成Q/Aペア: 118個" の両方に対応
                    count_match = re.search(r"(\d+)", line)
                    if count_match:
                        qa_count = int(count_match.group(1))

                elif "カバレージ率:" in line:
                    # カバレージ結果を解析
                    import re

                    rate_match = re.search(r"([\d.]+)%", line)
                    if rate_match:
                        coverage_results = {
                            "coverage_rate": float(rate_match.group(1)) / 100
                        }

            except queue.Empty:
                pass

            # プロセスが終了したら残りの出力を処理
            if poll is not None:
                # 残りの出力を全て取得
                while not output_queue.empty():
                    try:
                        line = output_queue.get_nowait()
                        log_callback(line)
                    except queue.Empty:
                        break
                break

        # プロセス終了コード確認
        return_code = process.returncode

        if return_code == 0:
            log_callback("✅ 高度なQ/A生成が正常に完了しました")
            return {
                "success": True,
                "saved_files": saved_files,
                "qa_count": qa_count,
                "coverage_results": coverage_results,
            }
        else:
            log_callback(f"⚠️ 高度なQ/A生成が終了コード {return_code} で終了しました")
            return {"success": False, "return_code": return_code}

    except Exception as e:
        log_callback(f"❌ 高度なQ/A生成でエラーが発生: {str(e)}")
        return {"success": False, "error": str(e)}


# ===================================================================
# Q/A生成関数
# ===================================================================


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """トークン数をカウント"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def split_into_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    テキストをチャンクに分割

    Args:
        text: 分割するテキスト
        chunk_size: チャンクサイズ（トークン数）
        overlap: オーバーラップ（トークン数）

    Returns:
        チャンクのリスト
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

        # オーバーラップを考慮
        start = end - overlap

        # 最後のチャンクの場合は終了
        if end >= len(tokens):
            break

    return chunks


def generate_qa_pairs(
    text: str,
    dataset_type: str,
    chunk_id: str,
    model: str = "gpt-4o-mini",
    qa_per_chunk: int = 3,
    log_callback=None,
) -> List[QAPair]:
    """
    テキストからQ/Aペアを生成

    Args:
        text: 対象テキスト
        dataset_type: データセットタイプ
        chunk_id: チャンクID
        model: 使用するモデル
        qa_per_chunk: チャンクあたりのQ/A数
        log_callback: ログコールバック関数

    Returns:
        Q/Aペアのリスト
    """
    client = OpenAI()

    prompt = f"""以下のテキストから、{qa_per_chunk}個の質問と回答のペアを生成してください。

テキスト:
{text}

要件:
1. 質問は具体的で明確なものにする
2. 回答はテキストの内容に基づいた正確なものにする
3. 質問タイプは以下から選択: factual, conceptual, application, analysis
4. テキストの重要な情報を網羅するようにする

JSON形式で出力してください。
"""

    try:
        # モデルに応じてtemperatureを調整
        # GPT-5シリーズ、O-seriesはtemperatureパラメータをサポートしない（デフォルト1のみ）
        model_lower = model.lower()

        # temperatureをサポートしないモデル
        no_temp_models = ["gpt-5", "o1", "o3", "o4"]
        use_temperature = not any(no_temp in model_lower for no_temp in no_temp_models)

        # API呼び出しパラメータを構築
        api_params = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "あなたは教育用Q/Aペア生成の専門家です。",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": QAPairsResponse,
        }

        # temperatureをサポートするモデルの場合のみ追加
        if use_temperature:
            api_params["temperature"] = 0.7

        response = client.beta.chat.completions.parse(**api_params)

        qa_response = response.choices[0].message.parsed

        # Q/Aペアにメタデータを追加
        for qa in qa_response.qa_pairs:
            qa.source_chunk_id = chunk_id
            qa.dataset_type = dataset_type
            qa.auto_generated = True

        if log_callback:
            log_callback(f"    └─ {len(qa_response.qa_pairs)}個のQ/Aペアを生成")

        return qa_response.qa_pairs

    except Exception as e:
        logger.error(f"Q/A生成エラー: {e}")
        if log_callback:
            log_callback(f"    └─ エラー: {str(e)}")
        return []


def save_qa_pairs_to_file(
    qa_pairs: List[QAPair], dataset_type: str, log_callback=None
) -> Dict[str, str]:
    """
    Q/AペアをCSVとJSONで保存

    Args:
        qa_pairs: Q/Aペアのリスト
        dataset_type: データセットタイプ
        log_callback: ログコールバック関数

    Returns:
        保存されたファイルパスの辞書
    """
    qa_output_dir = Path("qa_output")
    qa_output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {}

    # DataFrameに変換
    qa_data = []
    for qa in qa_pairs:
        qa_data.append(
            {
                "question": qa.question,
                "answer": qa.answer,
                "question_type": qa.question_type,
                "source_chunk_id": qa.source_chunk_id,
                "dataset_type": qa.dataset_type,
                "auto_generated": qa.auto_generated,
            }
        )

    df_qa = pd.DataFrame(qa_data)

    # CSVファイル
    csv_filename = f"qa_pairs_{dataset_type}_{timestamp}.csv"
    csv_path = qa_output_dir / csv_filename
    df_qa.to_csv(csv_path, index=False, encoding="utf-8-sig")
    saved_files["csv"] = str(csv_path)

    if log_callback:
        log_callback(f"  📄 CSV保存: {csv_filename}")

    # JSONファイル
    json_filename = f"qa_pairs_{dataset_type}_{timestamp}.json"
    json_path = qa_output_dir / json_filename

    json_data = {
        "dataset_type": dataset_type,
        "created_at": datetime.now().isoformat(),
        "total_pairs": len(qa_pairs),
        "qa_pairs": qa_data,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    saved_files["json"] = str(json_path)

    if log_callback:
        log_callback(f"  📋 JSON保存: {json_filename}")

    return saved_files


# ===================================================================
# Streamlitアプリ
# ===================================================================


def show_rag_download_page():
    """画面1: RAGデータダウンロード・前処理"""
    st.title("📥 RAGデータダウンロード・前処理ツール")
    st.caption(
        "HuggingFaceデータセットまたはローカルファイルをダウンロード・前処理してOUTPUT/フォルダに保存"
    )

    # ダウンロード・前処理済みデータセット
    st.subheader("📦 ダウンロード・前処理済みデータセット")
    df_preprocessed = load_preprocessed_history()

    if not df_preprocessed.empty:
        st.dataframe(df_preprocessed, use_container_width=True, hide_index=True, height=200)
    else:
        st.info(
            "まだ前処理済みデータがありません。下記からデータセットをダウンロード・前処理してください。"
        )

    st.divider()
    st.caption("データセットの自動ダウンロード → 前処理 → OUTPUT/フォルダに保存")

    # サイドバー：データソース選択
    with st.sidebar:
        st.header("📂 データソース選択")

        # データソース選択（データセット or ローカルファイル）
        data_source = st.radio(
            "データソースを選択",
            options=["dataset", "local_file"],
            format_func=lambda x: "🌐 データセット"
            if x == "dataset"
            else "📁 ローカルファイル",
            key="data_source_selector",
        )

        st.divider()

        if data_source == "dataset":
            # データセット選択
            st.subheader("📥 データセット")

            dataset_options = list(DATASET_CONFIGS.keys())
            dataset_labels = {
                key: f"{DATASET_CONFIGS[key]['icon']} {DATASET_CONFIGS[key]['name']}"
                for key in dataset_options
            }

            selected_dataset = st.radio(
                "ダウンロードするデータセット",
                options=dataset_options,
                format_func=lambda x: dataset_labels[x],
                label_visibility="collapsed",
            )

            uploaded_file = None
            config = DATASET_CONFIGS[selected_dataset]

        else:
            # ローカルファイルアップロード
            st.subheader("📁 ファイルアップロード")

            uploaded_file = st.file_uploader(
                "ファイルを選択",
                type=["csv", "txt", "json", "jsonl"],
                help="CSV, TXT, JSON, JSONL形式に対応",
            )

            selected_dataset = "custom_upload"
            config = {
                "name": "カスタムアップロード",
                "icon": "📁",
                "description": "ローカルファイルからQ/Aペアを生成",
                "text_field": "Combined_Text",
                "title_field": None,
                "sample_size": 0,
                "min_text_length": 50,
            }

    # データソースの表示名
    data_source_name = (
        config["name"]
        if data_source == "dataset"
        else (uploaded_file.name if uploaded_file else "未選択")
    )

    # メインエリア：処理オプション（上部）
    st.subheader("⚙️ 処理オプション")

    # データセット情報と処理オプションを横並び
    col_info, col_opts = st.columns([1, 1])

    with col_info:
        if data_source == "dataset":
            st.info(f"""
**{config["name"]}**

{config["description"]}

- データソース: {config.get("hf_dataset", "直接ダウンロード")}
- デフォルトサンプル数: {config["sample_size"]:,} 件
            """)
        else:
            if uploaded_file:
                st.info(f"""
**📁 ローカルファイル**

ファイル名: {uploaded_file.name}

ファイル形式: {uploaded_file.name.split(".")[-1].upper()}
                """)
            else:
                st.warning("ファイルを選択してください")

    with col_opts:
        if data_source == "dataset":
            sample_size = st.number_input(
                "サンプル数",
                min_value=10,
                max_value=10000,
                value=config["sample_size"],
                step=50,
                help="ダウンロードするデータ件数",
            )
        else:
            sample_size = st.number_input(
                "最大ドキュメント数（上限: 1,000件）",
                min_value=1,
                max_value=1000,
                value=100,
                step=10,
                help="処理する最大ドキュメント数。全件処理する場合は1,000に設定",
            )

        min_length = st.number_input(
            "最小テキスト長",
            min_value=10,
            max_value=1000,
            value=config["min_text_length"],
            step=10,
            help="この長さ未満のテキストを除外",
        )

        remove_duplicates = st.checkbox(
            "重複を除去", value=True, help="完全に同じテキストを除外"
        )

    # 実行ボタン
    run_download = st.button(
        "🚀 ダウンロード＆前処理開始", type="primary", use_container_width=True
    )

    st.divider()

    # メインエリア：処理情報と履歴（下部）
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 処理情報")
        info_container = st.container()

    with col2:
        st.subheader("📜 処理履歴・進捗")
        log_container = st.container()

    # 初期情報表示
    with info_container:
        st.metric("選択データセット", config["name"])
        st.metric("処理予定件数", f"{sample_size:,} 件")
        if "result_count" in st.session_state:
            st.metric("処理完了件数", f"{st.session_state['result_count']:,} 件")

    # ログ表示用
    if "logs" not in st.session_state:
        st.session_state["logs"] = []

    def add_log(message: str):
        """ログを追加"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state["logs"].append(f"[{timestamp}] {message}")

    # 処理実行
    if run_download:
        st.session_state["logs"] = []  # ログクリア
        add_log(f"🚀 処理開始: {data_source_name}")

        # ローカルファイルの場合はファイルチェック
        if data_source == "local_file" and not uploaded_file:
            st.error("ファイルを選択してください")
            st.stop()

        try:
            # ===================================================================
            # ローカルファイルの場合
            # ===================================================================
            if data_source == "local_file":
                # ステップ1: ファイル読み込み
                with st.spinner("📁 ファイル読み込み中..."):
                    add_log(f"📁 ローカルファイル読み込み: {uploaded_file.name}")
                    df = load_uploaded_file(uploaded_file)
                    add_log(f"✅ {len(df)} 件のデータを読み込みました")

                    # サンプリング
                    if len(df) > sample_size:
                        df = df.head(sample_size)
                        add_log(f"📊 {len(df)} 件に制限しました")

                # ステップ2: question, answerカラムの確認と抽出
                with st.spinner("⚙️ データ処理中..."):
                    add_log("⚙️ question, answerカラムを確認中...")

                    # question, answerカラムを探す
                    question_col = None
                    answer_col = None

                    for col in df.columns:
                        col_lower = col.lower()
                        if "question" in col_lower and not question_col:
                            question_col = col
                        if "answer" in col_lower and not answer_col:
                            answer_col = col

                    # question, answerカラムがない場合は通常処理
                    if question_col and answer_col:
                        add_log(f"  ✅ questionカラム: {question_col}")
                        add_log(f"  ✅ answerカラム: {answer_col}")

                        # question, answerのみ抽出
                        df_qa = df[[question_col, answer_col]].copy()
                        df_qa.columns = ["question", "answer"]  # カラム名を統一

                        # 空のデータを除外
                        before_len = len(df_qa)
                        df_qa = df_qa.dropna(subset=["question", "answer"])
                        df_qa = df_qa[
                            (df_qa["question"].str.strip() != "")
                            & (df_qa["answer"].str.strip() != "")
                        ]
                        removed = before_len - len(df_qa)
                        if removed > 0:
                            add_log(
                                f"📊 空データ除外: {removed} 件を除外（残り {len(df_qa)} 件）"
                            )

                        # 重複除去（オプション）
                        if remove_duplicates:
                            before_len = len(df_qa)
                            df_qa = df_qa.drop_duplicates()
                            removed = before_len - len(df_qa)
                            if removed > 0:
                                add_log(
                                    f"📊 重複除去: {removed} 件を除外（残り {len(df_qa)} 件）"
                                )

                        df_qa = df_qa.reset_index(drop=True)
                        add_log(f"✅ データ処理完了: {len(df_qa)} 件")

                        # ステップ3: qa_output/に保存
                        with st.spinner("💾 ファイル保存中..."):
                            add_log("💾 qa_output/フォルダに保存中...")

                            qa_output_dir = Path("qa_output")
                            qa_output_dir.mkdir(exist_ok=True)

                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            csv_filename = f"qa_pairs_upload_{timestamp}.csv"
                            csv_path = qa_output_dir / csv_filename

                            df_qa.to_csv(
                                csv_path, index=False, encoding="utf-8-sig"
                            )
                            add_log(f"  📄 CSV保存: {csv_filename}")
                            add_log("✅ ファイル保存完了")

                        # 結果を保存
                        st.session_state["result_count"] = len(df_qa)
                        st.session_state["qa_saved_files"] = {"csv": str(csv_path)}
                        st.session_state["qa_count"] = len(df_qa)
                        st.session_state["processed_df"] = df_qa

                        add_log("🎉 全処理完了！")
                    else:
                        add_log(
                            "⚠️ Q/Aカラムが見つかりません。テキストデータとして処理します"
                        )

                        # テキストデータとして保存
                        with st.spinner("💾 ファイル保存中..."):
                            output_dir = Path("OUTPUT")
                            output_dir.mkdir(exist_ok=True)

                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            csv_filename = f"preprocessed_upload_{timestamp}.csv"
                            csv_path = output_dir / csv_filename

                            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                            add_log(f"  📄 CSV保存: {csv_filename}")

                            st.session_state["result_count"] = len(df)
                            st.session_state["saved_files"] = {"csv": str(csv_path)}
                            st.session_state["processed_df"] = df

                            add_log("✅ データ保存完了")
                            add_log("🎉 全処理完了！")

            # ===================================================================
            # データセットの場合：既存の処理フロー
            # ===================================================================
            else:
                # ステップ1: データ読み込み
                with st.spinner("📥 データ読み込み中..."):
                    if selected_dataset == "livedoor":
                        # Livedoor特別処理
                        add_log("Livedoorコーパスをダウンロード中...")
                        data_dir = download_livedoor_corpus("datasets")
                        add_log("✅ ダウンロード完了")

                        add_log("データを読み込み中...")
                        df = load_livedoor_corpus(data_dir)
                        add_log(f"✅ {len(df)} 件のデータを読み込みました")

                        # サンプリング
                        if sample_size < len(df):
                            df = df.sample(n=sample_size, random_state=42)
                            add_log(f"📊 {len(df)} 件にサンプリングしました")

                    else:
                        # HuggingFaceからダウンロード
                        df = download_hf_dataset(
                            config["hf_dataset"],
                            config["hf_config"],
                            config["split"],
                            sample_size,
                            add_log,
                        )

                    add_log(f"✅ データ読み込み完了: {len(df)} 件")

                # ステップ2: 前処理
                with st.spinner("⚙️ 前処理実行中..."):
                    add_log("⚙️ 前処理開始")

                    add_log("テキストコンテンツを抽出中...")
                    df_processed = extract_text_content(df, config)
                    add_log(f"✅ テキスト抽出完了: {len(df_processed)} 件")

                    # 短文除外
                    before_len = len(df_processed)
                    df_processed = df_processed[
                        df_processed["Combined_Text"].str.len() >= min_length
                    ]
                    removed = before_len - len(df_processed)
                    if removed > 0:
                        add_log(
                            f"📊 短文除外: {removed} 件を除外（残り {len(df_processed)} 件）"
                        )

                    # 重複除去
                    if remove_duplicates:
                        before_len = len(df_processed)
                        df_processed = df_processed.drop_duplicates(
                            subset=["Combined_Text"]
                        )
                        removed = before_len - len(df_processed)
                        if removed > 0:
                            add_log(
                                f"📊 重複除去: {removed} 件を除外（残り {len(df_processed)} 件）"
                            )

                    df_processed = df_processed.reset_index(drop=True)
                    add_log(f"✅ 前処理完了: {len(df_processed)} 件")

                # ステップ3: OUTPUT保存
                with st.spinner("💾 ファイル保存中..."):
                    add_log("💾 OUTPUTフォルダに保存中...")
                    saved_files = save_to_output(df_processed, selected_dataset)
                    add_log("✅ ファイル保存完了")

                # 結果を保存
                st.session_state["result_count"] = len(df_processed)
                st.session_state["saved_files"] = saved_files
                st.session_state["processed_df"] = df_processed

                add_log("🎉 全処理完了！")

        except Exception as e:
            add_log(f"❌ エラー発生: {str(e)}")
            logger.error(f"処理エラー: {e}")
            st.error(f"エラーが発生しました: {str(e)}")

    # ログ表示
    with log_container:
        if st.session_state["logs"]:
            log_text = "\n".join(st.session_state["logs"])
            st.text_area("処理ログ", value=log_text, height=400, disabled=True)
        else:
            st.info("処理を開始するとここにログが表示されます")

    # 結果表示
    if "saved_files" in st.session_state:
        st.divider()
        st.subheader("📁 出力ファイル一覧")

        saved_files = st.session_state["saved_files"]

        # 前処理ファイル
        st.markdown("### 📄 前処理済みデータ")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.success("**CSV**")
            st.code(Path(saved_files["csv"]).name, language=None)
            st.caption(f"パス: {saved_files['csv']}")

        with col2:
            st.success("**TXT**")
            st.code(Path(saved_files["txt"]).name, language=None)
            st.caption(f"パス: {saved_files['txt']}")

        with col3:
            st.success("**メタデータ**")
            st.code(Path(saved_files["json"]).name, language=None)
            st.caption(f"パス: {saved_files['json']}")

        # データプレビュー
        if "processed_df" in st.session_state:
            with st.expander("📋 データプレビュー（最初の5件）"):
                df_preview = st.session_state["processed_df"][["Combined_Text"]].head(5)
                st.dataframe(df_preview, use_container_width=True)


def show_qa_generation_page():
    """画面2: Q/A生成"""

    st.title("🤖 Q/A生成ツール")
    st.caption(
        "既存データまたはローカルファイルからQ/Aペアを生成（a02_make_qa_para.py機能）"
    )

    # 最新のQ/A履歴表示
    st.subheader("📋 最新のQ&Aペア")
    df_history = load_qa_output_history()

    if not df_history.empty:
        st.dataframe(df_history, use_container_width=True, hide_index=True, height=200)
    else:
        st.info("まだQ&Aペアデータがありません。")

    st.divider()

    # サイドバー：入力ソース選択
    with st.sidebar:
        st.header("📂 入力ソース選択")

        # 入力ソース選択
        input_source = st.radio(
            "入力ソースを選択",
            options=["dataset", "local_file"],
            format_func=lambda x: "🌐 データセット"
            if x == "dataset"
            else "📁 ローカルファイル",
            key="input_source_selector",
        )

        st.divider()

        if input_source == "dataset":
            # データセット選択
            st.subheader("📥 データセット")

            dataset_options = list(DATASET_CONFIGS.keys())
            dataset_labels = {
                key: f"{DATASET_CONFIGS[key]['icon']} {DATASET_CONFIGS[key]['name']}"
                for key in dataset_options
            }

            selected_dataset = st.radio(
                "Q/A生成するデータセット",
                options=dataset_options,
                format_func=lambda x: dataset_labels[x],
                label_visibility="collapsed",
            )

            uploaded_file = None
            input_file_path = None

        else:
            # ローカルファイルアップロード
            st.subheader("📁 ファイルアップロード")

            uploaded_file = st.file_uploader(
                "ファイルを選択",
                type=["csv", "txt", "json", "jsonl"],
                help="CSV, TXT, JSON, JSONL形式に対応",
            )

            selected_dataset = None
            input_file_path = None

        # =========================================================
        # Q/A生成オプション（a02_make_qa_para.py相当）
        # =========================================================
        st.divider()
        st.subheader("🚀 Q/A生成設定")

        # Celery設定
        use_celery = st.checkbox(
            "Celery並列処理", value=True, help="複数ワーカーで並列処理"
        )

        if use_celery:
            celery_workers = st.number_input(
                "Celeryワーカー数",
                min_value=1,
                max_value=48,
                value=24,
                step=1,
                help="並列処理するワーカー数",
            )
        else:
            celery_workers = 1

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            batch_chunks = st.number_input(
                "バッチチャンク数",
                min_value=1,
                max_value=5,
                value=3,
                step=1,
                help="1回のAPIで処理するチャンク数",
            )

            max_docs = st.number_input(
                "最大ドキュメント数",
                min_value=1,
                max_value=10000,
                value=10,
                step=10,
                help="処理する最大ドキュメント数",
            )

        with col_a2:
            min_tokens = st.number_input(
                "最小トークン数",
                min_value=50,
                max_value=500,
                value=150,
                step=10,
                help="統合対象の最小トークン数",
            )

            max_tokens = st.number_input(
                "最大トークン数",
                min_value=100,
                max_value=1000,
                value=400,
                step=50,
                help="統合後の最大トークン数",
            )

        merge_chunks = st.checkbox(
            "チャンク統合", value=True, help="小さいチャンクを統合"
        )

        coverage_threshold = st.slider(
            "カバレージ閾値",
            min_value=0.0,
            max_value=1.0,
            value=0.58,
            step=0.01,
            help="カバレージ判定の類似度閾値",
        )

        qa_model = st.selectbox(
            "モデル",
            options=["gpt-5-nano", "gpt-5-mini", "gpt-5", "gpt-4o-mini", "gpt-4o"],
            index=3,
            help="Q/A生成に使用するモデル",
        )

        analyze_coverage = st.checkbox(
            "カバレージ分析", value=True, help="Q/Aペアのカバレージを分析"
        )

    # メインエリア：処理オプション
    st.subheader("⚙️ 入力情報")

    # 入力情報表示
    col_info, col_opts = st.columns([1, 1])

    with col_info:
        if input_source == "dataset":
            config = DATASET_CONFIGS[selected_dataset]
            st.info(f"""
**{config["name"]}**

{config["description"]}

- データソース: {config.get("hf_dataset", "直接ダウンロード")}
            """)
        else:
            if uploaded_file:
                st.info(f"""
**📁 ローカルファイル**

ファイル名: {uploaded_file.name}

ファイル形式: {uploaded_file.name.split(".")[-1].upper()}
                """)
            else:
                st.warning("ファイルを選択してください")

    with col_opts:
        st.markdown("**処理設定**")
        st.write(f"- Celery並列処理: {'有効' if use_celery else '無効'}")
        if use_celery:
            st.write(f"- ワーカー数: {celery_workers}")
        st.write(f"- バッチチャンク数: {batch_chunks}")
        st.write(f"- 最大ドキュメント数: {max_docs}")
        st.write(f"- モデル: {qa_model}")
        st.write(f"- カバレージ分析: {'実行' if analyze_coverage else 'スキップ'}")

    # 実行ボタン
    run_qa_generation = st.button(
        "🚀 Q/A生成開始", type="primary", use_container_width=True
    )

    st.divider()

    # メインエリア：進捗表示
    st.subheader("📜 処理履歴・進捗")
    log_container = st.container()

    # ログ表示用
    if "qa_logs" not in st.session_state:
        st.session_state["qa_logs"] = []

    def add_log(message: str):
        """ログを追加"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state["qa_logs"].append(f"[{timestamp}] {message}")

    # 処理実行
    if run_qa_generation:
        st.session_state["qa_logs"] = []  # ログクリア

        # 入力チェック
        if input_source == "local_file" and not uploaded_file:
            st.error("ファイルを選択してください")
            st.stop()

        try:
            add_log("🚀 Q/A生成処理開始")

            # ローカルファイルの場合、一時保存
            if input_source == "local_file":
                with st.spinner("📁 ファイル準備中..."):
                    add_log(f"📁 ローカルファイル読み込み: {uploaded_file.name}")

                    # 一時ファイルに保存
                    temp_dir = Path("temp_uploads")
                    temp_dir.mkdir(exist_ok=True)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    temp_filename = f"temp_qa_{timestamp}_{uploaded_file.name}"
                    temp_path = temp_dir / temp_filename

                    # ファイルを保存
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    input_file_path = str(temp_path)
                    add_log(f"  ✅ 一時ファイル作成: {temp_filename}")

            # a02_make_qa_para.pyを実行
            with st.spinner("🚀 Q/Aペア生成中（a02_make_qa_para.py実行）..."):
                add_log("🚀 a02_make_qa_para.py実行開始")

                result = run_advanced_qa_generation(
                    dataset=selected_dataset if input_source == "dataset" else None,
                    input_file=input_file_path,
                    use_celery=use_celery,
                    celery_workers=celery_workers,
                    batch_chunks=batch_chunks,
                    max_docs=max_docs,
                    merge_chunks=merge_chunks,
                    min_tokens=min_tokens,
                    max_tokens=max_tokens,
                    coverage_threshold=coverage_threshold,
                    model=qa_model,
                    analyze_coverage=analyze_coverage,
                    log_callback=add_log,
                )

                # 一時ファイルを削除
                if input_source == "local_file" and input_file_path:
                    try:
                        Path(input_file_path).unlink()
                        add_log("  🗑️ 一時ファイルを削除しました")
                    except:
                        pass

                if result["success"]:
                    qa_saved_files = result.get("saved_files")
                    qa_count = result.get("qa_count", 0)

                    # 結果を保存
                    st.session_state["qa_result_files"] = qa_saved_files
                    st.session_state["qa_result_count"] = qa_count

                    if result.get("coverage_results"):
                        add_log(
                            f"📊 カバレージ率: {result['coverage_results']['coverage_rate']:.1%}"
                        )

                    add_log("🎉 Q/A生成完了！")
                else:
                    add_log("⚠️ Q/A生成に失敗しました")

        except Exception as e:
            add_log(f"❌ エラー発生: {str(e)}")
            logger.error(f"処理エラー: {e}")
            st.error(f"エラーが発生しました: {str(e)}")

    # ログ表示
    with log_container:
        if st.session_state["qa_logs"]:
            log_text = "\n".join(st.session_state["qa_logs"])
            st.text_area("処理ログ", value=log_text, height=400, disabled=True)
        else:
            st.info("Q/A生成を開始するとここにログが表示されます")

    # 結果表示
    if "qa_result_files" in st.session_state and st.session_state["qa_result_files"]:
        st.divider()
        st.subheader("📁 生成結果")

        qa_files = st.session_state["qa_result_files"]
        qa_count = st.session_state.get("qa_result_count", 0)

        st.info(f"✅ 生成されたQ/Aペア: **{qa_count}** 個")

        col_qa1, col_qa2 = st.columns(2)

        with col_qa1:
            if qa_files and "csv" in qa_files and qa_files["csv"]:
                st.success("**Q/A CSV**")
                st.code(Path(qa_files["csv"]).name, language=None)
                st.caption(f"パス: {qa_files['csv']}")

        with col_qa2:
            if qa_files and "json" in qa_files and qa_files["json"]:
                st.success("**Q/A JSON**")
                st.code(Path(qa_files["json"]).name, language=None)
                st.caption(f"パス: {qa_files['json']}")


# ===================================================================
# Qdrant登録機能（a41, a42から移植）
# ===================================================================


# --- a42から移植：ユーティリティ関数 ---
def batched(seq: Iterable, size: int):
    """イテラブルをバッチに分割"""
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


# --- a41から移植：コレクション管理関数 ---
def get_collection_stats(
    client: QdrantClient, collection_name: str
) -> Optional[Dict[str, Any]]:
    """コレクションの統計情報を取得"""
    try:
        collection_info = client.get_collection(collection_name)
        total_points = collection_info.points_count

        # ベクトル設定情報を取得
        vectors_config = collection_info.config.params.vectors
        vector_info = {}

        if isinstance(vectors_config, dict):
            # Named Vectors
            for name, config in vectors_config.items():
                vector_info[name] = {
                    "size": config.size,
                    "distance": str(config.distance),
                }
        elif hasattr(vectors_config, "size"):
            # Single Vector
            vector_info["default"] = {
                "size": vectors_config.size,
                "distance": str(vectors_config.distance),
            }

        return {
            "total_points": total_points,
            "vector_config": vector_info,
            "status": collection_info.status,
        }

    except UnexpectedResponse as e:
        if "doesn't exist" in str(e) or "not found" in str(e).lower():
            return None
        raise
    except Exception as e:
        logger.error(f"統計情報取得エラー: {e}")
        return None


def get_all_collections(client: QdrantClient) -> List[Dict[str, Any]]:
    """全コレクションの情報を取得"""
    collections = client.get_collections()
    collection_list = []

    for collection in collections.collections:
        try:
            info = client.get_collection(collection.name)
            collection_list.append(
                {
                    "name": collection.name,
                    "points_count": info.points_count,
                    "status": info.status,
                }
            )
        except Exception as e:
            collection_list.append(
                {"name": collection.name, "points_count": 0, "status": "unknown"}
            )

    return collection_list


def delete_all_collections(client: QdrantClient, excluded: List[str] = None) -> int:
    """全コレクションを削除"""
    excluded = excluded or []
    collections = get_all_collections(client)

    if not collections:
        return 0

    to_delete = [c for c in collections if c["name"] not in excluded]

    if not to_delete:
        return 0

    deleted_count = 0
    failed_count = 0

    for col in to_delete:
        try:
            client.delete_collection(collection_name=col["name"])
            deleted_count += 1
        except Exception as e:
            logger.error(f"コレクション削除エラー {col['name']}: {e}")
            failed_count += 1

    return deleted_count


# --- a42から移植：データ処理関数 ---
def load_csv_for_qdrant(
    path: str, required=("question", "answer"), limit: int = 0
) -> pd.DataFrame:
    """CSVをロード（Qdrant登録用）"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)

    # 列名マッピング
    column_mappings = {
        "Question": "question",
        "Response": "answer",
        "Answer": "answer",
        "correct_answer": "answer",
    }
    df = df.rename(columns=column_mappings)

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"{path} には '{col}' 列が必要です（列: {list(df.columns)}）"
            )

    df = df.fillna("").drop_duplicates(subset=list(required)).reset_index(drop=True)

    if limit and limit > 0:
        df = df.head(limit).copy()

    return df


def build_inputs_for_embedding(df: pd.DataFrame, include_answer: bool) -> List[str]:
    """埋め込み用入力テキストを構築"""
    if include_answer:
        return (df["question"].astype(str) + "\n" + df["answer"].astype(str)).tolist()
    return df["question"].astype(str).tolist()


def embed_texts_for_qdrant(
    texts: List[str], model: str, batch_size: int = 128
) -> List[List[float]]:
    """テキストをバッチ処理でEmbeddingに変換"""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    client = OpenAI()

    MAX_TOKENS_PER_REQUEST = 8000

    # 空文字列・空白のみの文字列を除外
    valid_texts = []
    valid_indices = []
    for i, text in enumerate(texts):
        if text and text.strip():
            valid_texts.append(text)
            valid_indices.append(i)

    if not valid_texts:
        logger.warning("全てのテキストが空文字列です。ダミーベクトルを返します。")
        return [[0.0] * 1536] * len(texts)

    # 有効なテキストのみで埋め込み生成
    valid_vecs: List[List[float]] = []
    current_batch = []
    current_tokens = 0
    batch_count = 0

    for i, text in enumerate(valid_texts):
        text_tokens = len(enc.encode(text))

        if text_tokens > MAX_TOKENS_PER_REQUEST:
            raise ValueError(
                f"Single text at index {valid_indices[i]} has {text_tokens} tokens, "
                f"which exceeds MAX_TOKENS_PER_REQUEST ({MAX_TOKENS_PER_REQUEST}). "
            )

        if current_tokens + text_tokens > MAX_TOKENS_PER_REQUEST:
            if current_batch:
                batch_count += 1
                resp = client.embeddings.create(model=model, input=current_batch)
                valid_vecs.extend([d.embedding for d in resp.data])
                current_batch = []
                current_tokens = 0

        current_batch.append(text)
        current_tokens += text_tokens

    if current_batch:
        batch_count += 1
        resp = client.embeddings.create(model=model, input=current_batch)
        valid_vecs.extend([d.embedding for d in resp.data])

    # 元のインデックスに合わせてベクトルを再配置
    vecs: List[List[float]] = []
    valid_vec_idx = 0
    for i in range(len(texts)):
        if i in valid_indices:
            vecs.append(valid_vecs[valid_vec_idx])
            valid_vec_idx += 1
        else:
            vecs.append([0.0] * 1536)

    return vecs


def create_or_recreate_collection_for_qdrant(
    client: QdrantClient, name: str, recreate: bool, vector_size: int = 1536
):
    """コレクション作成または再作成"""
    vectors_config = models.VectorParams(
        size=vector_size, distance=models.Distance.COSINE
    )

    if recreate:
        try:
            client.delete_collection(collection_name=name)
        except:
            pass
        client.create_collection(collection_name=name, vectors_config=vectors_config)
    else:
        try:
            client.get_collection(name)
        except Exception:
            client.create_collection(
                collection_name=name, vectors_config=vectors_config
            )

    # ペイロード索引を作成
    try:
        client.create_payload_index(
            name, field_name="domain", field_schema=models.PayloadSchemaType.KEYWORD
        )
    except:
        pass


def build_points_for_qdrant(
    df: pd.DataFrame, vectors: List[List[float]], domain: str, source_file: str
) -> List[models.PointStruct]:
    """Qdrantポイントを構築"""
    n = len(df)
    if len(vectors) != n:
        raise ValueError(f"vectors length mismatch: df={n}, vecs={len(vectors)}")

    now_iso = datetime.now(timezone.utc).isoformat()
    points: List[models.PointStruct] = []

    for i, row in enumerate(df.itertuples(index=False)):
        payload = {
            "domain": domain,
            "question": getattr(row, "question"),
            "answer": getattr(row, "answer"),
            "source": os.path.basename(source_file),
            "created_at": now_iso,
            "schema": "qa:v1",
        }

        pid = abs(hash(f"{domain}-{source_file}-{i}")) & 0x7FFFFFFFFFFFFFFF
        points.append(models.PointStruct(id=pid, vector=vectors[i], payload=payload))

    return points


def upsert_points_to_qdrant(
    client: QdrantClient,
    collection: str,
    points: List[models.PointStruct],
    batch_size: int = 128,
) -> int:
    """ポイントをQdrantにアップサート"""
    count = 0
    for chunk in batched(points, batch_size):
        client.upsert(collection_name=collection, points=chunk)
        count += len(chunk)
    return count


# ===================================================================
# Qdrant Show機能（a40から移植）
# ===================================================================

# Qdrant設定
QDRANT_CONFIG = {
    "name": "Qdrant",
    "host": "localhost",
    "port": 6333,
    "icon": "🎯",
    "url": "http://localhost:6333",
    "health_check_endpoint": "/collections",
    "docker_image": "qdrant/qdrant",
}


class QdrantHealthChecker:
    """Qdrantサーバーの接続状態をチェック"""

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.client = None

    def check_port(self, host: str, port: int, timeout: float = 2.0) -> bool:
        """ポートが開いているかチェック"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception as e:
            if self.debug_mode:
                logger.error(f"Port check failed for {host}:{port}: {e}")
            return False

    def check_qdrant(self) -> Tuple[bool, str, Optional[Dict]]:
        """Qdrant接続チェック"""
        start_time = time.time()

        # まずポートチェック
        if not self.check_port(QDRANT_CONFIG["host"], QDRANT_CONFIG["port"]):
            return False, "Connection refused (port closed)", None

        try:
            self.client = QdrantClient(url=QDRANT_CONFIG["url"], timeout=5)

            # コレクション取得
            collections = self.client.get_collections()

            metrics = {
                "collection_count": len(collections.collections),
                "collections": [c.name for c in collections.collections],
                "response_time_ms": round((time.time() - start_time) * 1000, 2),
            }

            return True, "Connected", metrics

        except Exception as e:
            error_msg = str(e)
            if self.debug_mode:
                error_msg = f"{error_msg}\n{traceback.format_exc()}"
            return False, error_msg, None


class QdrantDataFetcher:
    """Qdrantからデータを取得"""

    def __init__(self, client: QdrantClient):
        self.client = client

    def fetch_collections(self) -> pd.DataFrame:
        """コレクション一覧を取得"""
        try:
            collections = self.client.get_collections()

            data = []
            for collection in collections.collections:
                try:
                    info = self.client.get_collection(collection.name)
                    data.append(
                        {
                            "Collection": collection.name,
                            "Vectors Count": info.vectors_count,
                            "Points Count": info.points_count,
                            "Indexed Vectors": info.indexed_vectors_count,
                            "Status": info.status,
                        }
                    )
                except:
                    data.append(
                        {
                            "Collection": collection.name,
                            "Vectors Count": "N/A",
                            "Points Count": "N/A",
                            "Indexed Vectors": "N/A",
                            "Status": "Error",
                        }
                    )

            return (
                pd.DataFrame(data)
                if data
                else pd.DataFrame({"Info": ["No collections found"]})
            )

        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})

    def fetch_collection_points(
        self, collection_name: str, limit: int = 50
    ) -> pd.DataFrame:
        """コレクションの詳細データを取得"""
        try:
            # スクロールを使ってポイントを取得
            points_result = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            points = points_result[0]  # scrollは (points, next_offset) のタプルを返す

            if not points:
                return pd.DataFrame({"Info": ["No points found in collection"]})

            # ポイントをDataFrameに変換
            data = []
            for point in points:
                row = {"ID": point.id}

                # payloadの各フィールドを列として追加
                if point.payload:
                    for key, value in point.payload.items():
                        # 長すぎる文字列は切り詰め
                        if isinstance(value, str) and len(value) > 200:
                            row[key] = value[:200] + "..."
                        elif isinstance(value, (list, dict)):
                            row[key] = (
                                str(value)[:200] + "..."
                                if len(str(value)) > 200
                                else str(value)
                            )
                        else:
                            row[key] = value

                data.append(row)

            return pd.DataFrame(data)

        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})

    def fetch_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """コレクションの詳細情報を取得"""
        try:
            collection_info = self.client.get_collection(collection_name)

            # configの構造を安全にアクセス
            vector_config = collection_info.config.params.vectors

            # vector_configの型を判定して適切に処理
            if hasattr(vector_config, "size"):
                # 単一ベクトル設定
                vector_size = vector_config.size
                distance = vector_config.distance
            elif hasattr(vector_config, "__iter__"):
                # Named vectors設定の場合
                vector_sizes = {}
                distances = {}
                for name, config in (
                    vector_config.items() if isinstance(vector_config, dict) else []
                ):
                    vector_sizes[name] = (
                        config.size if hasattr(config, "size") else "N/A"
                    )
                    distances[name] = (
                        config.distance if hasattr(config, "distance") else "N/A"
                    )
                vector_size = vector_sizes if vector_sizes else "N/A"
                distance = distances if distances else "N/A"
            else:
                vector_size = "N/A"
                distance = "N/A"

            return {
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "indexed_vectors": collection_info.indexed_vectors_count,
                "status": collection_info.status,
                "config": {
                    "vector_size": vector_size,
                    "distance": distance,
                },
            }
        except Exception as e:
            return {"error": str(e)}

    def fetch_collection_source_info(
        self, collection_name: str, sample_size: int = 200
    ) -> Dict[str, Any]:
        """コレクションのデータソース情報を取得"""
        try:
            collection_info = self.client.get_collection(collection_name)
            total_points = collection_info.points_count

            # サンプルポイントを取得
            points_result = self.client.scroll(
                collection_name=collection_name,
                limit=min(sample_size, total_points),
                with_payload=True,
                with_vectors=False,
            )

            points = points_result[0]

            if not points:
                return {"total_points": total_points, "sources": {}, "sample_size": 0}

            # sourceとgeneration_methodを集計
            source_stats = {}
            for point in points:
                if point.payload:
                    source = point.payload.get("source", "unknown")
                    method = point.payload.get("generation_method", "unknown")
                    domain = point.payload.get("domain", "unknown")

                    if source not in source_stats:
                        source_stats[source] = {
                            "sample_count": 0,
                            "method": method,
                            "domain": domain,
                        }
                    source_stats[source]["sample_count"] += 1

            # 全体のデータ数を推定
            sample_total = len(points)
            for source, stats in source_stats.items():
                ratio = stats["sample_count"] / sample_total
                stats["estimated_total"] = int(total_points * ratio)
                stats["percentage"] = ratio * 100

            return {
                "total_points": total_points,
                "sources": source_stats,
                "sample_size": sample_total,
            }

        except Exception as e:
            return {"error": str(e)}


def display_source_info(source_info: Dict[str, Any]) -> None:
    """データソース情報を表示"""
    if "error" in source_info:
        st.error(f"ソース情報取得エラー: {source_info['error']}")
        return

    total_points = source_info.get("total_points", 0)
    sources = source_info.get("sources", {})
    sample_size = source_info.get("sample_size", 0)

    if not sources:
        st.info("📂 データソース情報が見つかりません")
        return

    # ソース情報を表示
    st.markdown("### 📂 データソース構成 (qa_output/ディレクトリー)")
    st.caption(f"サンプル{sample_size}件から推定 | 総ポイント数: {total_points:,}件")

    # テーブル形式で表示
    source_data = []
    for source, stats in sorted(sources.items()):
        source_data.append(
            {
                "ソースファイル": source,
                "推定件数": f"{stats['estimated_total']:,}件",
                "割合": f"{stats['percentage']:.1f}%",
                "生成方法": stats["method"],
                "ドメイン": stats["domain"],
            }
        )

    df_sources = pd.DataFrame(source_data)
    st.dataframe(df_sources, use_container_width=True, hide_index=True)

    # 詳細情報（折りたたみ可能）
    with st.expander("📊 詳細情報", expanded=False):
        for source, stats in sorted(sources.items()):
            st.markdown(f"**{source}**")
            st.markdown(f"- パス: `qa_output/{source}`")
            st.markdown(
                f"- 推定データ数: {stats['estimated_total']:,}件 ({stats['percentage']:.1f}%)"
            )
            st.markdown(f"- 生成方法: `{stats['method']}`")
            st.markdown(f"- ドメイン: `{stats['domain']}`")
            st.markdown(f"- サンプル内カウント: {stats['sample_count']}件")
            st.divider()


# ===================================================================
# Qdrant登録機能（a41, a42から移植）
# ===================================================================


def show_qdrant_registration_page():
    """画面3: Q/AペアデータQdrant登録"""
    st.title("🗄️ Q/Aペアデータ・Qdrant登録")
    st.caption("qa_output/*.csvのデータをQdrantベクトルDBに登録")

    # サイドバー：設定
    with st.sidebar:
        st.header("⚙️ Qdrant設定")

        qdrant_url = st.text_input(
            "Qdrant URL", value="http://localhost:6333", help="QdrantサーバーのURL"
        )

        st.divider()
        st.header("📋 操作モード")

        operation_mode = st.radio(
            "操作モードを選択",
            options=["all_collections", "individual_csv"],
            format_func=lambda x: "📊 全コレクション操作"
            if x == "all_collections"
            else "📄 個別CSV操作",
            key="qdrant_operation_mode",
        )

        st.divider()

        # モード別設定
        if operation_mode == "individual_csv":
            st.subheader("📄 CSV設定")

            # qa_output/*.csvファイル一覧取得
            qa_output_dir = Path("qa_output")
            if qa_output_dir.exists():
                csv_files = sorted(qa_output_dir.glob("*.csv"))
                csv_options = [f.name for f in csv_files]
            else:
                csv_options = []

            if csv_options:
                selected_csv = st.selectbox(
                    "ファイル選択",
                    options=csv_options,
                    help="登録するCSVファイルを選択",
                )

                # コレクション名を自動生成（カスタマイズ可能）
                default_collection = f"qa_{Path(selected_csv).stem}"
                collection_name = st.text_input(
                    "コレクション名",
                    value=default_collection,
                    help="Qdrantコレクション名",
                )

                recreate_collection = st.checkbox(
                    "既存データ削除",
                    value=True,
                    help="既存コレクションを削除して再作成",
                )

                include_answer = st.checkbox(
                    "answerを含める", value=True, help="埋め込み生成時にanswerも含める"
                )

                data_limit = st.number_input(
                    "データ件数制限",
                    min_value=0,
                    max_value=100000,
                    value=0,
                    step=100,
                    help="0=無制限",
                )
            else:
                st.warning("qa_output/フォルダにCSVファイルが見つかりません")
                selected_csv = None
                collection_name = None
                recreate_collection = False
                include_answer = False
                data_limit = 0

    # Qdrant接続確認
    st.subheader("📡 Qdrant接続状態")

    try:
        client = QdrantClient(url=qdrant_url, timeout=30)
        client.get_collections()
        st.success(f"✅ Qdrant接続成功: {qdrant_url}")
        qdrant_connected = True
    except Exception as e:
        st.error(f"❌ Qdrant接続エラー: {e}")
        st.warning("Qdrantが起動していることを確認してください。")
        st.code("docker run -p 6333:6333 qdrant/qdrant", language="bash")
        qdrant_connected = False
        client = None

    st.divider()

    # モード別メインコンテンツ
    if operation_mode == "all_collections":
        # ===================================================================
        # 全コレクション操作モード
        # ===================================================================
        st.subheader("📊 全コレクション一覧")

        if qdrant_connected and client:
            try:
                collections = get_all_collections(client)

                if collections:
                    total_points = sum(c["points_count"] for c in collections)

                    col_metric1, col_metric2 = st.columns(2)
                    with col_metric1:
                        st.metric("総コレクション数", f"{len(collections)} 個")
                    with col_metric2:
                        st.metric("総ポイント数", f"{total_points:,} 件")

                    # コレクション一覧表
                    df_collections = pd.DataFrame(collections)
                    df_collections = df_collections.sort_values(
                        "points_count", ascending=False
                    )

                    st.dataframe(
                        df_collections,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "name": st.column_config.TextColumn(
                                "コレクション名", width="medium"
                            ),
                            "points_count": st.column_config.NumberColumn(
                                "ポイント数", format="%d"
                            ),
                            "status": st.column_config.TextColumn(
                                "ステータス", width="small"
                            ),
                        },
                    )

                    st.divider()

                    # 危険な操作セクション
                    st.subheader("⚠️ 危険な操作")

                    col_btn1, col_btn2 = st.columns(2)

                    with col_btn1:
                        if st.button(
                            "🗑️ 全コレクション削除",
                            type="secondary",
                            use_container_width=True,
                        ):
                            st.session_state["confirm_delete_all"] = True

                    with col_btn2:
                        if st.button(
                            "📊 詳細統計表示", type="primary", use_container_width=True
                        ):
                            st.session_state["show_detailed_stats"] = True

                    # 削除確認ダイアログ
                    if st.session_state.get("confirm_delete_all", False):
                        st.warning("⚠️ **警告：全コレクション削除**")
                        st.error(
                            f"**{len(collections)}個**のコレクション（合計**{total_points:,}ポイント**）が完全に削除されます。"
                        )
                        st.error("この操作は取り消せません！")

                        col_confirm1, col_confirm2 = st.columns(2)

                        with col_confirm1:
                            if st.button(
                                "✅ 削除を実行",
                                type="primary",
                                use_container_width=True,
                            ):
                                with st.spinner("削除中..."):
                                    deleted = delete_all_collections(client)
                                    st.success(
                                        f"✅ {deleted}個のコレクションを削除しました"
                                    )
                                    st.session_state["confirm_delete_all"] = False
                                    st.rerun()

                        with col_confirm2:
                            if st.button("❌ キャンセル", use_container_width=True):
                                st.session_state["confirm_delete_all"] = False
                                st.rerun()

                    # 詳細統計表示
                    if st.session_state.get("show_detailed_stats", False):
                        st.divider()
                        st.subheader("📊 詳細統計情報")

                        for col_info in collections:
                            with st.expander(
                                f"📦 {col_info['name']} ({col_info['points_count']:,} ポイント)"
                            ):
                                try:
                                    stats = get_collection_stats(
                                        client, col_info["name"]
                                    )
                                    if stats:
                                        st.json(stats)
                                    else:
                                        st.warning("統計情報を取得できませんでした")
                                except Exception as e:
                                    st.error(f"エラー: {e}")

                        if st.button("閉じる"):
                            st.session_state["show_detailed_stats"] = False
                            st.rerun()

                else:
                    st.info("コレクションが存在しません")

            except Exception as e:
                st.error(f"エラー: {e}")
                logger.error(f"コレクション一覧取得エラー: {e}")
        else:
            st.warning("Qdrantに接続できていません")

    else:
        # ===================================================================
        # 個別CSV操作モード
        # ===================================================================
        st.subheader("📄 CSV登録設定")

        if not csv_options:
            st.warning("qa_output/フォルダにCSVファイルがありません")
            st.info("先に「Q/A生成」でデータを作成してください")
            return

        # ファイル情報表示
        csv_path = qa_output_dir / selected_csv
        file_size = csv_path.stat().st_size
        if file_size < 1024:
            size_str = f"{file_size} B"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"

        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info(f"""
**ファイル情報**
- ファイル名: {selected_csv}
- ファイルサイズ: {size_str}
- コレクション名: {collection_name}
            """)

        with col_info2:
            st.info(f"""
**登録設定**
- 既存データ削除: {"はい" if recreate_collection else "いいえ"}
- answerを含める: {"はい" if include_answer else "いいえ"}
- データ件数制限: {data_limit if data_limit > 0 else "無制限"}
            """)

        # データプレビュー
        with st.expander("📋 データプレビュー（最初の3件）"):
            try:
                df_preview = pd.read_csv(csv_path, nrows=3)
                st.dataframe(df_preview, use_container_width=True)
            except Exception as e:
                st.error(f"プレビュー読み込みエラー: {e}")

        st.divider()

        # 登録ボタン
        run_registration = st.button(
            "🚀 Qdrantに登録",
            type="primary",
            use_container_width=True,
            disabled=not qdrant_connected,
        )

        # ログ表示エリア
        st.subheader("📜 処理ログ")
        log_container = st.container()

        if "qdrant_logs" not in st.session_state:
            st.session_state["qdrant_logs"] = []

        def add_log(message: str):
            """ログを追加"""
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state["qdrant_logs"].append(f"[{timestamp}] {message}")

        # 登録処理実行
        if run_registration:
            st.session_state["qdrant_logs"] = []  # ログクリア
            add_log(f"🚀 登録処理開始: {selected_csv}")

            try:
                # ステップ1: CSVロード
                with st.spinner("📁 CSVファイル読み込み中..."):
                    add_log(f"📁 CSV読み込み: {csv_path}")
                    df = load_csv_for_qdrant(str(csv_path), limit=data_limit)
                    add_log(f"✅ {len(df)} 件のデータを読み込みました")

                # ステップ2: コレクション作成
                with st.spinner("🗄️ コレクション準備中..."):
                    add_log(f"🗄️ コレクション準備: {collection_name}")
                    create_or_recreate_collection_for_qdrant(
                        client, collection_name, recreate_collection
                    )
                    add_log(f"✅ コレクション準備完了")

                # ステップ3: 埋め込み生成
                with st.spinner("🔢 埋め込み生成中..."):
                    add_log("🔢 埋め込み生成開始")
                    texts = build_inputs_for_embedding(df, include_answer)
                    vectors = embed_texts_for_qdrant(
                        texts, model="text-embedding-3-small"
                    )
                    add_log(f"✅ {len(vectors)} 件の埋め込みを生成しました")

                # ステップ4: ポイント構築
                with st.spinner("📦 ポイント構築中..."):
                    add_log("📦 Qdrantポイント構築中")
                    # ドメイン名を推定
                    if "cc_news" in selected_csv.lower():
                        domain = "cc_news"
                    elif "livedoor" in selected_csv.lower():
                        domain = "livedoor"
                    else:
                        domain = "custom"

                    points = build_points_for_qdrant(df, vectors, domain, selected_csv)
                    add_log(f"✅ {len(points)} 個のポイントを構築しました")

                # ステップ5: Qdrantアップサート
                with st.spinner("⬆️ Qdrantアップサート中..."):
                    add_log("⬆️ Qdrantにアップサート中")
                    count = upsert_points_to_qdrant(client, collection_name, points)
                    add_log(f"✅ {count} 件をQdrantに登録しました")

                # 完了
                add_log("🎉 全処理完了！")
                st.success(f"✅ {count}件のデータをQdrantに登録しました")

                # 統計情報を表示
                try:
                    stats = get_collection_stats(client, collection_name)
                    if stats:
                        st.divider()
                        st.subheader("📊 登録結果")
                        col_result1, col_result2, col_result3 = st.columns(3)
                        with col_result1:
                            st.metric("総ポイント数", f"{stats['total_points']:,}")
                        with col_result2:
                            vector_size = list(stats["vector_config"].values())[0][
                                "size"
                            ]
                            st.metric("ベクトルサイズ", vector_size)
                        with col_result3:
                            st.metric("ステータス", stats["status"])
                except Exception as e:
                    add_log(f"⚠️ 統計情報取得エラー: {e}")

            except Exception as e:
                add_log(f"❌ エラー発生: {str(e)}")
                logger.error(f"登録エラー: {e}")
                st.error(f"エラーが発生しました: {str(e)}")

        # ログ表示
        with log_container:
            if st.session_state["qdrant_logs"]:
                log_text = "\n".join(st.session_state["qdrant_logs"])
                st.text_area("処理ログ", value=log_text, height=300, disabled=True)
            else:
                st.info("登録を開始するとここにログが表示されます")


def show_qdrant_page():
    """画面4: Qdrant Show - コレクション表示"""
    st.title("🔍 Show-Qdrantコレクション")
    st.caption("Qdrant Vector Database の状態監視とデータ表示")

    # セッションステート初期化
    if "qdrant_debug_mode" not in st.session_state:
        st.session_state.qdrant_debug_mode = False
    if "qdrant_auto_refresh" not in st.session_state:
        st.session_state.qdrant_auto_refresh = False
    if "qdrant_refresh_interval" not in st.session_state:
        st.session_state.qdrant_refresh_interval = 30

    # サイドバー（左ペイン）
    with st.sidebar:
        st.header("⚙️ Qdrant接続状態")

        # デバッグモード切り替え
        debug_mode = st.checkbox(
            "🐛 デバッグモード", value=st.session_state.qdrant_debug_mode
        )
        st.session_state.qdrant_debug_mode = debug_mode

        # 自動リフレッシュ設定
        col1, col2 = st.columns(2)
        with col1:
            auto_refresh = st.checkbox(
                "🔄 自動更新", value=st.session_state.qdrant_auto_refresh
            )
            st.session_state.qdrant_auto_refresh = auto_refresh
        with col2:
            if auto_refresh:
                refresh_interval = st.number_input(
                    "間隔(秒)", min_value=5, max_value=300, value=30
                )
                st.session_state.qdrant_refresh_interval = refresh_interval

        # 接続チェック実行ボタン
        check_button = st.button(
            "🔍 接続チェック実行", type="primary", use_container_width=True
        )

        # HealthCheckerインスタンス
        checker = QdrantHealthChecker(debug_mode=debug_mode)

        # 接続状態表示エリア
        status_container = st.container()

        # 自動リフレッシュまたはボタン押下時に実行
        if check_button or (auto_refresh and time.time() % refresh_interval < 1):
            with status_container:
                with st.spinner("チェック中..."):
                    is_connected, message, metrics = checker.check_qdrant()

                # Qdrantの状態表示
                if is_connected:
                    st.success(f"{QDRANT_CONFIG['icon']} **{QDRANT_CONFIG['name']}**")
                    st.caption(f"✅ {message}")

                    # メトリクス表示
                    if metrics and debug_mode:
                        with st.expander(f"詳細情報", expanded=False):
                            for key, value in metrics.items():
                                st.text(f"{key}: {value}")
                else:
                    st.error(f"{QDRANT_CONFIG['icon']} **{QDRANT_CONFIG['name']}**")
                    st.caption(f"❌ {message}")

                    # エラー詳細（デバッグモード）
                    if debug_mode:
                        with st.expander("エラー詳細", expanded=False):
                            st.code(message)
                            st.caption(
                                f"Host: {QDRANT_CONFIG.get('host')}:{QDRANT_CONFIG.get('port')}"
                            )

                            # Docker起動コマンド表示
                            st.info("Docker起動コマンド:")
                            cmd = f"docker run -d -p {QDRANT_CONFIG['port']}:{QDRANT_CONFIG['port']} {QDRANT_CONFIG['docker_image']}"
                            st.code(cmd, language="bash")

    # メインエリア（右ペイン）
    st.header(f"📊 Qdrant データ表示")

    try:
        # Qdrantクライアントを作成
        client = QdrantClient(url=QDRANT_CONFIG["url"], timeout=5)
        data_fetcher = QdrantDataFetcher(client)

        # コレクション概要表示
        st.subheader("📚 コレクション一覧")

        # コレクション一覧を取得
        df_collections = data_fetcher.fetch_collections()

        if not df_collections.empty and "Collection" in df_collections.columns:
            st.dataframe(df_collections, use_container_width=True)

            # コレクション名のリストを作成
            collection_names = df_collections["Collection"].tolist()

            # ===== データソース情報の表示（メインエリア先頭） =====
            st.divider()
            st.subheader("📂 コレクションのデータソース情報")
            st.caption(
                "各コレクションがqa_output/ディレクトリーのどのファイルから構成されているかを表示します"
            )

            # 各コレクションのソース情報を表示
            for collection_name in collection_names:
                with st.expander(
                    f"📦 {collection_name}", expanded=(collection_name == "qa_corpus")
                ):
                    with st.spinner(f"{collection_name} のソース情報を取得中..."):
                        source_info = data_fetcher.fetch_collection_source_info(
                            collection_name
                        )
                        display_source_info(source_info)

            # エクスポート機能
            col1, col2 = st.columns(2)
            with col1:
                csv = df_collections.to_csv(index=False)
                st.download_button(
                    label="📥 CSVダウンロード",
                    data=csv,
                    file_name=f"qdrant_collections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            with col2:
                json_str = df_collections.to_json(orient="records", indent=2)
                st.download_button(
                    label="📥 JSONダウンロード",
                    data=json_str,
                    file_name=f"qdrant_collections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )

            # コレクション詳細表示
            st.divider()
            st.subheader("🔍 コレクション詳細データ")

            if collection_names:
                selected_collection = st.selectbox(
                    "詳細を表示するコレクションを選択",
                    options=collection_names,
                    key="selected_collection",
                )

                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    limit = st.number_input(
                        "表示件数",
                        min_value=1,
                        max_value=500,
                        value=50,
                        key="qdrant_limit",
                    )
                with col2:
                    show_details = st.button(
                        "📊 詳細情報を表示", key="show_collection_details"
                    )
                with col3:
                    fetch_points = st.button(
                        "🔍 ポイントデータを取得", key="fetch_collection_points"
                    )

                # コレクション詳細情報の表示
                if show_details:
                    with st.spinner(f"{selected_collection} の詳細情報を取得中..."):
                        info = data_fetcher.fetch_collection_info(selected_collection)

                        if "error" not in info:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("ベクトル数", info["vectors_count"])
                            with col2:
                                st.metric("ポイント数", info["points_count"])
                            with col3:
                                st.metric("インデックス済み", info["indexed_vectors"])
                            with col4:
                                st.metric("ステータス", info["status"])

                            # 設定情報
                            st.write("**ベクトル設定:**")
                            st.write(
                                f"  • ベクトル次元: {info['config']['vector_size']}"
                            )
                            st.write(f"  • 距離計算: {info['config']['distance']}")
                        else:
                            st.error(f"エラー: {info['error']}")

                # ポイントデータの表示
                if fetch_points:
                    with st.spinner(
                        f"{selected_collection} のポイントデータを取得中..."
                    ):
                        df_points = data_fetcher.fetch_collection_points(
                            selected_collection, limit
                        )

                        if not df_points.empty and "ID" in df_points.columns:
                            st.write(
                                f"**{selected_collection} のデータサンプル ({len(df_points)} 件):**"
                            )
                            st.dataframe(df_points, use_container_width=True)

                            # エクスポート機能
                            col1, col2 = st.columns(2)
                            with col1:
                                csv = df_points.to_csv(index=False)
                                st.download_button(
                                    label="📥 ポイントデータ CSVダウンロード",
                                    data=csv,
                                    file_name=f"{selected_collection}_points_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                )
                            with col2:
                                json_str = df_points.to_json(orient="records", indent=2)
                                st.download_button(
                                    label="📥 ポイントデータ JSONダウンロード",
                                    data=json_str,
                                    file_name=f"{selected_collection}_points_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                )
                        elif "Info" in df_points.columns:
                            st.info(df_points.iloc[0]["Info"])
                        elif "Error" in df_points.columns:
                            st.error(f"エラー: {df_points.iloc[0]['Error']}")
                        else:
                            st.info("ポイントデータが見つかりません")
        elif "Info" in df_collections.columns:
            st.info(df_collections.iloc[0]["Info"])
        elif "Error" in df_collections.columns:
            error_msg = df_collections.iloc[0]["Error"]

            # より詳細なエラーメッセージを表示
            if "Connection refused" in error_msg or "[Errno 61]" in error_msg:
                st.error("❌ Qdrantサーバーに接続できません")
                st.warning("Qdrantサーバーが起動していることを確認してください")
                st.code("python server.py", language="bash")
                st.caption("または")
                st.code("docker run -p 6333:6333 qdrant/qdrant", language="bash")
                if debug_mode:
                    with st.expander("🔍 詳細エラー情報", expanded=False):
                        st.error(f"詳細エラー: {error_msg}")
            elif "timeout" in error_msg.lower():
                st.error("⏱️ Qdrantサーバーへの接続がタイムアウトしました")
                st.warning("サーバーが応答していないか、ネットワークの問題があります")
            else:
                st.error(f"エラー: {error_msg}")
                st.info("Qdrantサーバーが正しく起動していることを確認してください")
        else:
            st.info("コレクションが見つかりません")

    except Exception as e:
        error_msg = str(e)

        # より詳細なエラーメッセージを表示
        if "Connection refused" in error_msg or "[Errno 61]" in error_msg:
            st.error("❌ Qdrantサーバーに接続できません")
            st.warning("Qdrantサーバーが起動していることを確認してください")
            st.code("python server.py", language="bash")
            st.caption("または")
            st.code("docker run -p 6333:6333 qdrant/qdrant", language="bash")
            if debug_mode:
                with st.expander("🔍 詳細エラー情報", expanded=False):
                    st.error(f"詳細エラー: {error_msg}")
        elif "timeout" in error_msg.lower():
            st.error("⏱️ Qdrantサーバーへの接続がタイムアウトしました")
            st.warning("サーバーが応答していないか、ネットワークの問題があります")
        else:
            st.error(f"Qdrant接続エラー: {error_msg}")
            st.info("Qdrantサーバーが正しく起動していることを確認してください")

    # フッター
    st.divider()
    st.caption(f"最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # デバッグ情報表示
    if debug_mode:
        with st.expander("🐛 デバッグ情報", expanded=False):
            st.subheader("サーバー設定")
            st.json(QDRANT_CONFIG)


# ===================================================================
# Qdrant検索機能（a50から移植）
# ===================================================================

# コレクション固有の埋め込み設定
COLLECTION_EMBEDDINGS_SEARCH = {
    "qa_corpus": {"model": "text-embedding-3-small", "dims": 1536},
    "qa_cc_news_a02_llm": {"model": "text-embedding-3-small", "dims": 1536},
    "qa_cc_news_a03_rule": {"model": "text-embedding-3-small", "dims": 1536},
    "qa_cc_news_a10_hybrid": {"model": "text-embedding-3-small", "dims": 1536},
    "qa_livedoor_a02_20_llm": {"model": "text-embedding-3-small", "dims": 1536},
    "qa_livedoor_a03_rule": {"model": "text-embedding-3-small", "dims": 1536},
    "qa_livedoor_a10_hybrid": {"model": "text-embedding-3-small", "dims": 1536},
}

# コレクション名とCSVファイルのマッピング
COLLECTION_CSV_MAPPING = {
    "qa_cc_news_a02_llm": "a02_qa_pairs_cc_news.csv",
    "qa_cc_news_a03_rule": "a03_qa_pairs_cc_news.csv",
    "qa_cc_news_a10_hybrid": "a10_qa_pairs_cc_news.csv",
    "qa_livedoor_a02_20_llm": "a02_qa_pairs_livedoor.csv",
    "qa_livedoor_a03_rule": "a03_qa_pairs_livedoor.csv",
    "qa_livedoor_a10_hybrid": "a10_qa_pairs_livedoor.csv",
}


def load_sample_questions_from_csv(
    collection_name: str, num_samples: int = 3
) -> List[str]:
    """
    qa_output/からCSVファイルを読み込んで質問例を取得

    Args:
        collection_name: コレクション名
        num_samples: 取得する質問数

    Returns:
        質問のリスト
    """
    # コレクション名に対応するCSVファイルを取得
    csv_filename = COLLECTION_CSV_MAPPING.get(collection_name)
    if not csv_filename:
        return []

    csv_path = Path("qa_output") / csv_filename
    if not csv_path.exists():
        return []

    try:
        df = pd.read_csv(csv_path)
        # questionカラムがあるか確認
        if "question" not in df.columns:
            return []

        # ランダムにサンプリング
        questions = df["question"].dropna().sample(min(num_samples, len(df))).tolist()
        return questions
    except Exception as e:
        logger.error(f"質問例の読み込みエラー {csv_path}: {e}")
        return []


@st.cache_data(ttl=300)  # 5分間キャッシュ
def load_source_qa_data(
    source_filename: str, num_rows: int = 20
) -> Optional[pd.DataFrame]:
    """
    qa_output/*.csvからQ/Aデータを取得

    Args:
        source_filename: ソースファイル名（例: "a02_qa_pairs_cc_news.csv"）
        num_rows: 取得する行数（デフォルト: 20）

    Returns:
        question, answerカラムのDataFrame（上位num_rows行）、エラー時はNone
    """
    csv_path = Path("qa_output") / source_filename
    if not csv_path.exists():
        logger.warning(f"CSVファイルが存在しません: {csv_path}")
        return None

    try:
        # 効率的に最初のnum_rows行だけを読み込み
        df = pd.read_csv(csv_path, nrows=num_rows, usecols=["question", "answer"])

        # カラムの存在確認
        if "question" not in df.columns or "answer" not in df.columns:
            logger.error(
                f"CSVファイルに必要なカラム (question, answer) がありません: {csv_path}"
            )
            return None

        logger.info(f"ソースファイル読み込み成功: {csv_path} ({len(df)}行)")
        return df

    except Exception as e:
        logger.error(f"ソースファイル読み込みエラー {csv_path}: {e}")
        return None


@st.cache_data(ttl=300)  # 5分間キャッシュ
def load_collection_qa_preview(
    collection_name: str, num_rows: int = 20
) -> Optional[pd.DataFrame]:
    """
    コレクションに対応するCSVファイルからQ/Aデータのプレビューを取得

    Args:
        collection_name: コレクション名
        num_rows: 取得する行数（デフォルト: 20）

    Returns:
        question, answerカラムのDataFrame（上位num_rows行）、エラー時はNone
    """
    # コレクション名に対応するCSVファイルを取得
    csv_filename = COLLECTION_CSV_MAPPING.get(collection_name)
    if not csv_filename:
        logger.warning(
            f"コレクション '{collection_name}' に対応するCSVファイルが見つかりません"
        )
        return None

    csv_path = Path("qa_output") / csv_filename
    if not csv_path.exists():
        logger.warning(f"CSVファイルが存在しません: {csv_path}")
        return None

    try:
        # 効率的に最初のnum_rows行だけを読み込み
        df = pd.read_csv(csv_path, nrows=num_rows, usecols=["question", "answer"])

        # カラムの存在確認
        if "question" not in df.columns or "answer" not in df.columns:
            logger.error(
                f"CSVファイルに必要なカラム (question, answer) がありません: {csv_path}"
            )
            return None

        logger.info(f"CSVプレビュー読み込み成功: {csv_path} ({len(df)}行)")
        return df

    except Exception as e:
        logger.error(f"CSVプレビュー読み込みエラー {csv_path}: {e}")
        return None


def embed_query_for_search(
    text: str, model: str, dims: Optional[int] = None
) -> List[float]:
    """
    クエリテキストを埋め込みベクトルに変換

    Args:
        text: 埋め込むテキスト
        model: 使用する埋め込みモデル
        dims: ベクトルの次元数

    Returns:
        埋め込みベクトル
    """
    client = OpenAI()
    if dims and "text-embedding-3" in model:
        return (
            client.embeddings.create(model=model, input=[text], dimensions=dims)
            .data[0]
            .embedding
        )
    else:
        return client.embeddings.create(model=model, input=[text]).data[0].embedding


def show_system_explanation_page():
    """画面0: システム説明"""
    st.title("📖 システム説明")
    st.caption("RAGシステムのデータフロー・処理ステップ・ディレクトリ構造")

    st.markdown("---")

    # データフロー図
    st.subheader("🔄 データフロー例（CC-Newsの場合）")

    st.code("""
┌─────────────────┐
│  HuggingFace    │
│    cc_news      │
└────────┬────────┘
         │ ①ダウンロード
         ↓
┌─────────────────────────────────┐
│  datasets/                      │
│  cc_news_train_500_*.csv        │
└────────┬────────────────────────┘
         │ ①前処理
         ↓
┌─────────────────────────────────┐
│  OUTPUT/                        │
│  preprocessed_cc_news.csv       │
└────────┬────────────────────────┘
         │ ②Q/A生成
         ↓
┌─────────────────────────────────┐
│  qa_output/                     │
│  a02_qa_pairs_cc_news.csv       │
└────────┬────────────────────────┘
         │ ③埋め込み生成
         ↓
┌─────────────────────────────────┐
│  OpenAI                         │
│  text-embedding-3-small         │
└────────┬────────────────────────┘
         │ ③ベクトル登録
         ↓
┌─────────────────────────────────┐
│  Qdrant                         │
│  qa_cc_news_a02_llm             │
└─────────────────────────────────┘
    """, language=None)

    st.markdown("---")

    # ステップ詳細
    st.subheader("📋 ステップ詳細")

    st.markdown("""
| ステップ | スクリプト | 入力 | 出力 | 所要時間目安 |
|---------|-----------|------|------|------------|
| **①-1** | `a01_load_non_qa_rag_data.py` | HuggingFace | `datasets/cc_news_*.csv` | 1-5分 |
| **①-2** | `a01_load_non_qa_rag_data.py` | `datasets/cc_news_*.csv` | `OUTPUT/preprocessed_cc_news.csv` | 1分 |
| **②** | `a02_make_qa_para.py` | `OUTPUT/preprocessed_cc_news.csv` | `qa_output/a02_qa_pairs_cc_news.csv` | 10-60分 |
| **③** | `a42_qdrant_registration.py` | `qa_output/a02_qa_pairs_cc_news.csv` | Qdrant | 5-10分 |
    """)

    st.markdown("---")

    # ディレクトリ構造
    st.subheader("📂 ディレクトリ構造")

    st.markdown("""
```
openai_rag_qa_jp/
├── datasets/                  # ①ダウンロードしたRawデータ
│   ├── wikimedia_wikipedia_train_1000_*.csv
│   ├── range3_cc100_ja_train_1000_*.csv
│   ├── cc_news_train_500_*.csv
│   └── livedoor/
│       └── text/              # 解凍されたLivedoorデータ
│
├── OUTPUT/                    # ①前処理済みデータ
│   ├── preprocessed_wikipedia_ja.csv
│   ├── preprocessed_japanese_text.csv
│   ├── preprocessed_cc_news.csv
│   └── preprocessed_livedoor.csv
│
├── qa_output/                 # ②Q/A生成データ
│   ├── a02_qa_pairs_cc_news.csv
│   ├── a02_qa_pairs_livedoor.csv
│   ├── a03_qa_pairs_cc_news.csv
│   ├── a10_qa_pairs_cc_news.csv
│   └── coverage_*.json
│
└── [Qdrantコレクション]       # ③ベクトルDB
    ├── qa_cc_news_a02_llm
    ├── qa_cc_news_a03_rule
    ├── qa_cc_news_a10_hybrid
    ├── qa_livedoor_a02_20_llm
    ├── qa_livedoor_a03_rule
    └── qa_livedoor_a10_hybrid
```
    """)

    st.markdown("---")

    # 実行コマンド早見表
    st.subheader("🎯 実行コマンド早見表")

    with st.expander("📰 CC-News データセット", expanded=False):
        st.markdown("""
```bash
# ステップ1: ダウンロード・前処理
streamlit run a01_load_non_qa_rag_data.py --server.port=8502
# → UI操作: HuggingFaceから cc_news をロード
# → 「OUTPUTフォルダに保存」ボタンをクリック

# ステップ2: Q/A生成
python a02_make_qa_para.py \\
  --dataset cc_news \\
  --use-celery \\
  --celery-workers 24 \\
  --model gpt-4o-mini \\
  --max-docs 100

# ステップ3: Qdrant登録
python a42_qdrant_registration.py --recreate --include-answer
```
        """)

    with st.expander("📰 Livedoor データセット", expanded=False):
        st.markdown("""
```bash
# ステップ1: ダウンロード・前処理
streamlit run a01_load_non_qa_rag_data.py --server.port=8502
# → UI操作: Livedoor を選択してロード

# ステップ2: Q/A生成
python a02_make_qa_para.py \\
  --dataset livedoor \\
  --use-celery \\
  --celery-workers 24 \\
  --model gpt-4o-mini

# ステップ3: Qdrant登録
python a42_qdrant_registration.py --recreate --include-answer
```
        """)

    with st.expander("📄 カスタムファイル（アップロード）", expanded=False):
        st.markdown("""
```bash
# ステップ2から開始（既にCSVがある場合）
python a02_make_qa_para.py \\
  --input-file my_data.csv \\
  --use-celery \\
  --celery-workers 24 \\
  --model gpt-4o-mini

# ステップ3: Qdrant登録
python a42_qdrant_registration.py \\
  --input-file qa_output/a02_qa_pairs_{dataset}.csv \\
  --recreate --include-answer
```
        """)

    st.markdown("---")

    # 対応データセット一覧
    st.subheader("📊 対応データセット")

    st.markdown("""
| データセット名 | 中間保存先 | 最終出力先 |
|---------------|-----------|-----------|
| **Wikipedia日本語** | `datasets/wikimedia_wikipedia_train_1000_*.csv` | `OUTPUT/preprocessed_wikipedia_ja.csv` |
| **CC100日本語** | `datasets/range3_cc100_ja_train_1000_*.csv` | `OUTPUT/preprocessed_japanese_text.csv` |
| **CC-News英語** | `datasets/cc_news_train_500_*.csv` | `OUTPUT/preprocessed_cc_news.csv` |
| **Livedoor** | `datasets/livedoor_train_7376_*.csv` | `OUTPUT/preprocessed_livedoor.csv` |
    """)


def show_qdrant_search_page():
    """画面5: Qdrant検索"""
    st.title("🔎 Qdrant検索")
    st.caption("Qdrantベクトルデータベースを使用した意味検索")

    # コレクションとCSVファイルの対応表を表示
    st.subheader("📊 コレクションとCSVファイルの対応")
    mapping_data = []
    for collection, csv_file in COLLECTION_CSV_MAPPING.items():
        mapping_data.append(
            {
                "コレクション名": collection,
                "CSVファイル": csv_file,
                "ファイルパス": f"qa_output/{csv_file}",
            }
        )
    mapping_df = pd.DataFrame(mapping_data)
    st.table(mapping_df)
    st.divider()

    # Qdrant接続確認
    qdrant_url = "http://localhost:6333"

    # 利用可能なコレクションを取得
    available_collections = []
    try:
        temp_client = QdrantClient(url=qdrant_url)
        collections_response = temp_client.get_collections()
        available_collections = [col.name for col in collections_response.collections]
    except Exception as e:
        st.error(f"❌ Qdrantサーバーに接続できません: {qdrant_url}")
        st.warning("Qdrantサーバーが起動していることを確認してください")
        st.code("python server.py", language="bash")
        st.caption("または")
        st.code("docker run -p 6333:6333 qdrant/qdrant", language="bash")
        return

    if not available_collections:
        st.warning("利用可能なコレクションがありません")
        st.info("先に「Qdrant登録」でデータを登録してください")
        return

    # サイドバー：検索設定
    with st.sidebar:
        st.header("🔧 検索設定")

        # コレクション選択
        collection = st.selectbox(
            "コレクション",
            options=available_collections,
            help="検索対象のコレクションを選択",
        )

        # コレクション情報表示
        if collection in COLLECTION_EMBEDDINGS_SEARCH:
            col_info = COLLECTION_EMBEDDINGS_SEARCH[collection]
            st.info(f"📊 {col_info['model']} ({col_info['dims']}次元)")

        # Top-K設定
        topk = st.slider(
            "検索結果数（Top-K）", min_value=1, max_value=20, value=5, step=1
        )

        # デバッグモード
        debug_mode = st.checkbox("🐛 デバッグモード", value=False)

    # メインエリア
    # セッション状態の初期化
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""

    # コレクションデータプレビューセクション
    with st.expander("📋 コレクションデータプレビュー", expanded=False):
        # QdrantDataFetcherインスタンスを作成
        try:
            client = QdrantClient(url=qdrant_url)
            data_fetcher = QdrantDataFetcher(client)

            # fetch_collection_source_infoを使用してソース情報を取得
            source_info = data_fetcher.fetch_collection_source_info(collection)

            if "error" not in source_info:
                sources = source_info.get("sources", {})

                if sources:
                    st.caption(f"コレクション: **{collection}**")

                    # 各ソースファイルごとにエキスパンダーを作成
                    for source, stats in sorted(sources.items()):
                        with st.expander(f"📄 {source}", expanded=False):
                            st.markdown(
                                f"- 推定データ数: {stats['estimated_total']:,}件 ({stats['percentage']:.1f}%)"
                            )
                            st.markdown(f"- 生成方法: `{stats['method']}`")
                            st.markdown(f"- ドメイン: `{stats['domain']}`")

                            # question, answerテーブルを表示
                            df_qa = load_source_qa_data(source, num_rows=20)
                            if df_qa is not None:
                                st.dataframe(
                                    df_qa, use_container_width=True, hide_index=True
                                )
                            else:
                                st.info(f"データを読み込めません: qa_output/{source}")
                else:
                    st.info("データソース情報が見つかりません")
            else:
                st.error(f"エラー: {source_info['error']}")
        except Exception as e:
            st.error(f"データ取得エラー: {str(e)}")

    st.divider()

    # 検索入力
    st.subheader("🔍 検索")
    query = st.text_input(
        "検索クエリを入力してください",
        value=st.session_state.search_query,
        placeholder="検索したい質問を入力してください",
    )

    col_search, col_clear = st.columns([4, 1])
    with col_search:
        do_search = st.button("🔍 検索実行", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑️ クリア", use_container_width=True):
            st.session_state.search_query = ""
            st.rerun()

    # 検索実行
    if do_search and query.strip():
        try:
            client = QdrantClient(url=qdrant_url)

            # コレクションに対応した埋め込み設定を取得
            collection_config = COLLECTION_EMBEDDINGS_SEARCH.get(
                collection, {"model": "text-embedding-3-small", "dims": 1536}
            )
            embedding_model = collection_config["model"]
            embedding_dims = collection_config.get("dims")

            if debug_mode:
                st.info(f"🔍 使用モデル: {embedding_model} ({embedding_dims}次元)")

            # クエリを埋め込みベクトルに変換
            with st.spinner("埋め込みベクトルを生成中..."):
                qvec = embed_query_for_search(query, embedding_model, embedding_dims)
                if debug_mode:
                    st.success(f"✅ {len(qvec)}次元のベクトルを生成しました")

            # Qdrantで検索
            with st.spinner("検索中..."):
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    hits = client.search(
                        collection_name=collection, query_vector=qvec, limit=topk
                    )

            # 検索結果を表示
            st.divider()
            st.subheader(f"📊 検索結果 (Top {len(hits)})")

            if not hits:
                st.warning("検索結果が見つかりませんでした")
                return

            # 結果をDataFrameに変換
            rows = []
            for h in hits:
                row_data = {
                    "スコア": f"{h.score:.4f}",
                    "質問": h.payload.get("question", "N/A") if h.payload else "N/A",
                    "回答": h.payload.get("answer", "N/A") if h.payload else "N/A",
                    "ソース": h.payload.get("source", "N/A") if h.payload else "N/A",
                }
                rows.append(row_data)

            df_results = pd.DataFrame(rows)
            st.dataframe(df_results, use_container_width=True, hide_index=True)

            # 最高スコアの結果を詳細表示
            if hits:
                best_hit = hits[0]
                st.divider()
                st.subheader("🏆 最高スコアの結果")

                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("スコア", f"{best_hit.score:.4f}")
                with col2:
                    if best_hit.payload:
                        source = best_hit.payload.get("source", "N/A")
                        st.caption(f"ソース: {source}")

                if best_hit.payload:
                    question = best_hit.payload.get("question", "")
                    answer = best_hit.payload.get("answer", "")

                    st.markdown("**質問:**")
                    st.info(question)

                    st.markdown("**回答:**")
                    st.success(answer)

                    # OpenAIによる日本語応答生成
                    st.divider()
                    st.subheader("🧠 AI応答（OpenAI GPT-4o-mini）")

                    qa_prompt = (
                        "以下の検索結果とユーザーの質問を踏まえて、日本語で簡潔かつ正確に回答してください。\n\n"
                        f"ユーザーの質問:\n{query}\n\n"
                        f"検索結果のスコア: {best_hit.score:.4f}\n"
                        f"検索結果の質問: {question}\n"
                        f"検索結果の回答: {answer}\n"
                    )

                    with st.expander("📝 プロンプト詳細", expanded=False):
                        st.code(qa_prompt)

                    try:
                        with st.spinner("AIが回答を生成中..."):
                            oai_client = OpenAI()
                            oai_resp = oai_client.responses.create(
                                model="gpt-4o-mini", input=qa_prompt
                            )
                            generated_answer = (
                                getattr(oai_resp, "output_text", None) or ""
                            )

                        if generated_answer.strip():
                            st.markdown("**AI応答:**")
                            st.write(generated_answer)
                        else:
                            st.info("応答テキストを取得できませんでした")
                    except Exception as gen_err:
                        st.error(f"AI応答生成に失敗しました: {str(gen_err)}")
                        if debug_mode:
                            st.exception(gen_err)

        except Exception as e:
            st.error(f"❌ エラーが発生しました: {str(e)}")
            if debug_mode:
                st.exception(e)

            if "Connection refused" in str(e):
                st.warning("Qdrantサーバーが起動していることを確認してください")
                st.code("python server.py", language="bash")
            elif "collection" in str(e).lower() and "not found" in str(e).lower():
                st.warning(f"コレクション '{collection}' が見つかりません")
                st.info("先に「Qdrant登録」でデータを登録してください")


def main():
    """メインアプリケーション - 画面選択"""

    # ページ設定
    st.set_page_config(page_title="RAGツール", page_icon="🤖", layout="wide")

    # サイドバー：画面選択
    with st.sidebar:
        st.title("🤖 RAGツール")
        st.divider()

        # メニュー見出し
        st.markdown("**メニュー**")

        # 画面選択
        page = st.radio(
            "機能選択",
            options=[
                "explanation",
                "rag_download",
                "qa_generation",
                "qdrant_registration",
                "show_qdrant",
                "qdrant_search",
            ],
            format_func=lambda x: {
                "explanation": "📖 説明",
                "rag_download": "📥 RAGデータダウンロード",
                "qa_generation": "🤖 Q/A生成",
                "qdrant_registration": "🗄️ Qdrant登録",
                "show_qdrant": "🔍 Show-Qdrant",
                "qdrant_search": "🔎 Qdrant検索",
            }[x],
            label_visibility="collapsed",
        )

        st.divider()

    # 選択された画面を表示
    if page == "explanation":
        show_system_explanation_page()
    elif page == "rag_download":
        show_rag_download_page()
    elif page == "qa_generation":
        show_qa_generation_page()
    elif page == "qdrant_registration":
        show_qdrant_registration_page()
    elif page == "show_qdrant":
        show_qdrant_page()
    elif page == "qdrant_search":
        show_qdrant_search_page()


if __name__ == "__main__":
    main()
