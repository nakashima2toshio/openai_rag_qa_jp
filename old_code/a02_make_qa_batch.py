#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
a02_make_qa_batch.py - Batch API 対応 Q/A 生成と埋め込み
=======================================================

目的:
- 通常APIの約50%コストで大量処理を回すため、OpenAI Batch API を用いて
  - Q/A 生成（/v1/responses）
  - 埋め込み生成（/v1/embeddings）
  をJSONLで準備→投入→ダウンロード→突合（解析）する補助スクリプト。

前提:
- OPENAI_API_KEY が .env などで設定済み
- データの分割（チャンク化）は既存 a02_make_qa.py の関数を再利用
- 1バッチ=1エンドポイント（responses と embeddings を混在させない）
- custom_id を必ず付与し、出力JSONLと突合する

使い方（例）:
  1) Q/A生成のJSONLを用意
     python a02_make_qa_batch.py prepare-responses --dataset wikipedia_ja --model gpt-4o-mini \
         --out qa_output/batch/qa_requests.jsonl --max-docs 600

  2) EmbeddingsのJSONLを用意
     python a02_make_qa_batch.py prepare-embeddings --dataset wikipedia_ja \
         --embedding-model text-embedding-3-small --out qa_output/batch/embed_requests.jsonl --max-docs 600

  3) バッチ投入（Responses）
     python a02_make_qa_batch.py submit --input qa_output/batch/qa_requests.jsonl --endpoint /v1/responses

  4) ステータス確認
     python a02_make_qa_batch.py status --batch-id batch_XXXX

  5) 結果ダウンロード
     python a02_make_qa_batch.py download --batch-id batch_XXXX --out qa_output/batch/qa_output.jsonl

  6) 出力JSONLを突合（Q/A集計）
     python a02_make_qa_batch.py parse-responses --input qa_output/batch/qa_output.jsonl \
         --outdir qa_output --dataset wikipedia_ja

  7) Embeddings 出力のパース（必要な場合）
     python a02_make_qa_batch.py parse-embeddings --input qa_output/batch/embed_output.jsonl --out qa_output/embeddings.parquet

注意:
- Batch API はSLA 24h。急ぐ処理は通常APIへ。待てる処理はBatchへ。
- レートは別枠で大規模処理に有利。
- このスクリプトは投入/DL/突合のパイプを用意する。生成精度は既存のプロンプトに依存。
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

# 既存の前処理・分割ロジックを再利用
from a02_make_qa import (
    DATASET_CONFIGS,
    load_preprocessed_data,
    create_document_chunks,
    merge_small_chunks,
    determine_qa_count,
)


# -------------------------------------------------
# 環境・ログ
# -------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -------------------------------------------------
# ユーティリティ
# -------------------------------------------------
def ensure_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.strip() in {"", "your-openai-api-key-here"}:
        logger.error("OPENAI_API_KEY が未設定です。 .env などで設定してください。")
        sys.exit(1)
    return api_key


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sanitize_id_part(value: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in str(value))


def make_custom_id(kind: str, dataset: str, doc_idx: int | str, chunk_idx: int | str, pairs: int | None = None) -> str:
    """custom_id は突合の鍵。短く・安全に。
    形式: "{kind}|{dataset}|{doc_idx}|{chunk_idx}|{pairs or 0}"
    """
    parts = [sanitize_id_part(kind), sanitize_id_part(dataset), str(doc_idx), str(chunk_idx), str(pairs or 0)]
    return "|".join(parts)


def parse_custom_id(custom_id: str) -> Dict[str, str]:
    try:
        kind, dataset, doc_idx, chunk_idx, pairs = custom_id.split("|")
    except ValueError:
        # 旧/想定外フォーマットは最小限パース
        segs = custom_id.split("|")
        segs += ["", "", "", ""]
        kind, dataset, doc_idx, chunk_idx, pairs = segs[:5]
    return {
        "kind": kind,
        "dataset": dataset,
        "doc_idx": doc_idx,
        "chunk_idx": chunk_idx,
        "pairs": pairs,
    }


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# -------------------------------------------------
# プロンプト（responses用）
# -------------------------------------------------
def build_qa_prompts(text: str, lang: str, num_pairs: int) -> Tuple[str, Dict]:
    """システム＋ユーザーを1入力文字列へまとめる（Batchは"input"で送る）。
    同時にJSONスキーマを返す（Responses APIの response_format 用）。
    """
    if lang == "ja":
        system_prompt = (
            "あなたは教育コンテンツ作成の専門家です。\n"
            "与えられた日本語テキストから、学習効果の高いQ&Aペアを生成してください。\n\n"
            "生成ルール:\n"
            "1. 質問は明確で具体的に\n"
            "2. 回答は簡潔で正確に（1-2文程度）\n"
            "3. テキストの内容に忠実に\n"
            "4. 多様な観点から質問を作成"
        )
        question_types_desc = (
            "- fact: 事実確認型（〜は何ですか？）\n"
            "- reason: 理由説明型（なぜ〜ですか？）\n"
            "- comparison: 比較型（〜と〜の違いは？）\n"
            "- application: 応用型（〜はどのように活用されますか？）"
        )
        user_prompt = (
            f"以下のテキストから{num_pairs}個のQ&Aペアを生成してください。\n\n"
            f"[テキスト]\n{text}\n\n"
            "質問タイプ:\n" + question_types_desc + "\n\n"
            "JSONで出力し、schemaに厳密に従ってください。"
        )
    else:
        system_prompt = (
            "You are an expert in educational content creation.\n"
            "Generate high-quality Q&A pairs from the given English text.\n\n"
            "Generation rules:\n"
            "1. Questions should be clear and specific\n"
            "2. Answers should be concise and accurate (1-2 sentences)\n"
            "3. Stay faithful to the text content\n"
            "4. Create questions from diverse perspectives"
        )
        question_types_desc = (
            "- fact: Factual questions (What is...?)\n"
            "- reason: Explanatory questions (Why...?)\n"
            "- comparison: Comparative questions (What's the difference...?)\n"
            "- application: Application questions (How is... used?)"
        )
        user_prompt = (
            f"Generate {num_pairs} Q&A pairs from the text below.\n\n"
            f"[Text]\n{text}\n\n"
            "Question types:\n" + question_types_desc + "\n\n"
            "Output strict JSON per the schema."
        )

    combined_input = f"{system_prompt}\n\n{user_prompt}"

    json_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "qa_pairs_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "qa_pairs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"},
                                "answer": {"type": "string"},
                                "question_type": {
                                    "type": "string",
                                    "enum": ["fact", "reason", "comparison", "application"]
                                }
                            },
                            "required": ["question", "answer", "question_type"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["qa_pairs"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

    return combined_input, json_schema


# -------------------------------------------------
# JSONL 準備
# -------------------------------------------------
def write_jsonl(lines: List[Dict], out_path: Path) -> None:
    ensure_parent_dir(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def prepare_responses_jsonl(
    *,
    dataset: str,
    model: str,
    out_jsonl: Path,
    max_docs: Optional[int] = None,
    merge_chunks: bool = True,
    min_tokens: int = 150,
    max_tokens: int = 400,
    max_output_tokens: int = 1000,
) -> Tuple[List[Dict], Path]:
    """チャンクを作り、/v1/responses 用JSONLを1行=1リクエストで構築する。
    custom_id を各行に必ず付与。
    """
    df = load_preprocessed_data(dataset)
    chunks = create_document_chunks(df, dataset, max_docs)
    if merge_chunks:
        chunks = merge_small_chunks(chunks, min_tokens=min_tokens, max_tokens=max_tokens)

    cfg = DATASET_CONFIGS[dataset]
    lang = cfg.get("lang", "ja")

    lines: List[Dict] = []
    for c in chunks:
        # チャンクの目標ペア数
        num_pairs = determine_qa_count(c, cfg)
        # 長文は切る（Batchでも無限に長い本文は避ける）
        text = c['text']
        if len(text) > 3000:
            text = text[:3000] + "..."

        combined_input, json_schema = build_qa_prompts(text, lang, num_pairs)

        custom_id = make_custom_id(
            kind="qa",
            dataset=dataset,
            doc_idx=c.get("doc_idx", "0"),
            chunk_idx=c.get("chunk_idx", "0"),
            pairs=num_pairs,
        )

        body = {
            "model": model,
            "input": combined_input,
            "response_format": json_schema,
            "max_output_tokens": max_output_tokens,
        }

        line = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }
        lines.append(line)

    write_jsonl(lines, out_jsonl)
    logger.info(f"responses JSONL を出力: {out_jsonl} ({len(lines)} 行)")
    return chunks, out_jsonl


def prepare_embeddings_jsonl(
    *,
    dataset: str,
    embedding_model: str,
    out_jsonl: Path,
    max_docs: Optional[int] = None,
    merge_chunks: bool = True,
    min_tokens: int = 150,
    max_tokens: int = 400,
) -> Tuple[List[Dict], Path]:
    """チャンクテキストを /v1/embeddings に1行=1リクエストで投げるJSONLへ。"""
    df = load_preprocessed_data(dataset)
    chunks = create_document_chunks(df, dataset, max_docs)
    if merge_chunks:
        chunks = merge_small_chunks(chunks, min_tokens=min_tokens, max_tokens=max_tokens)

    lines: List[Dict] = []
    for c in chunks:
        custom_id = make_custom_id(
            kind="emb",
            dataset=dataset,
            doc_idx=c.get("doc_idx", "0"),
            chunk_idx=c.get("chunk_idx", "0"),
            pairs=None,
        )

        body = {
            "model": embedding_model,
            "input": c["text"]
        }
        line = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/embeddings",
            "body": body,
        }
        lines.append(line)

    write_jsonl(lines, out_jsonl)
    logger.info(f"embeddings JSONL を出力: {out_jsonl} ({len(lines)} 行)")
    return chunks, out_jsonl


# -------------------------------------------------
# Batch API 操作
# -------------------------------------------------
def submit_batch(input_jsonl: Path, endpoint: str, *, metadata: Optional[Dict] = None) -> str:
    """JSONLファイルをアップロードしてBatchを作成。戻り値は batch_id。"""
    ensure_api_key()
    client = OpenAI()

    with input_jsonl.open("rb") as f:
        up = client.files.create(file=f, purpose="batch")

    md = metadata or {}
    md.setdefault("submitted_by", "a02_make_qa_batch.py")
    md.setdefault("endpoint", endpoint)

    batch = client.batches.create(
        input_file_id=up.id,
        endpoint=endpoint,
        completion_window="24h",
        metadata=md,
    )

    logger.info(f"Batch submitted: id={batch.id} input_file_id={up.id} endpoint={endpoint}")
    return batch.id


def get_batch(batch_id: str) -> Dict:
    ensure_api_key()
    client = OpenAI()
    b = client.batches.retrieve(batch_id)
    # openai SDK のオブジェクトを dict に落とす
    data = json.loads(json.dumps(b, default=lambda x: getattr(x, "__dict__", str(x))))
    return data


def download_batch_output(batch_id: str, out_path: Path) -> Path:
    """完了済みバッチの出力JSONLを保存。"""
    ensure_api_key()
    client = OpenAI()
    b = client.batches.retrieve(batch_id)
    if not getattr(b, "output_file_id", None):
        raise RuntimeError(f"Batch {batch_id} に output_file_id がありません。status={getattr(b, 'status', None)}")

    file_id = b.output_file_id
    ensure_parent_dir(out_path)

    content = client.files.content(file_id)
    # content はバイナリストリーム
    with out_path.open("wb") as f:
        for chunk in content.iter_bytes():
            f.write(chunk)

    logger.info(f"Batch output saved: {out_path}")
    return out_path


# -------------------------------------------------
# 出力JSONLの突合（Responses / Embeddings）
# -------------------------------------------------
def extract_output_text_from_response_body(resp_body: Dict) -> Optional[str]:
    """Batchのレスポンスボディからテキストを抜き出す。output_text があれば優先。"""
    # 1) 直接 output_text がある場合
    if isinstance(resp_body, dict) and "output_text" in resp_body and resp_body["output_text"]:
        return resp_body["output_text"]

    # 2) output を辿る（Responses API 仕様）
    try:
        output = resp_body.get("output")
        if isinstance(output, list):
            texts: List[str] = []
            for item in output:
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") == "output_text" and "text" in c:
                            texts.append(c["text"])
            if texts:
                return "\n".join(texts)
    except Exception:
        pass

    return None


def parse_responses_output_jsonl(input_jsonl: Path, *, dataset: Optional[str] = None) -> List[Dict]:
    """Batch出力(JSONL)を読み、custom_idでQ/Aを再構成しフラットな行に展開。"""
    results: List[Dict] = []
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            custom_id = obj.get("custom_id", "")

            # 失敗レコードはスキップ/警告
            if obj.get("error"):
                logger.warning(f"error row custom_id={custom_id}: {obj['error']}")
                continue

            resp_body = obj.get("response", {}).get("body", {})
            text = extract_output_text_from_response_body(resp_body) or ""

            if not text:
                logger.warning(f"no output_text custom_id={custom_id}")
                continue

            # JSONとして解釈（response_formatでJSON出力を強制している想定）
            parsed_json = None
            try:
                parsed_json = json.loads(text)
            except Exception:
                # 念のためJSONが見つかる部分を抽出
                try:
                    start = text.find("{")
                    end = text.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        parsed_json = json.loads(text[start:end+1])
                except Exception:
                    parsed_json = None

            if not parsed_json or "qa_pairs" not in parsed_json:
                logger.warning(f"parsed_json invalid custom_id={custom_id}")
                continue

            meta = parse_custom_id(custom_id)
            ds = dataset or meta.get("dataset", "")
            doc_idx = meta.get("doc_idx", "")
            chunk_idx = meta.get("chunk_idx", "")

            for qa in parsed_json.get("qa_pairs", []):
                # 正規化
                question = qa.get("question", "").strip()
                answer = qa.get("answer", "").strip()
                qtype = qa.get("question_type", "unknown").strip()
                results.append({
                    "dataset_type": ds,
                    "doc_idx": doc_idx,
                    "chunk_idx": chunk_idx,
                    "question": question,
                    "answer": answer,
                    "question_type": qtype,
                    "custom_id": custom_id,
                })

    logger.info(f"Parsed responses: {len(results)} rows")
    return results


def parse_embeddings_output_jsonl(input_jsonl: Path) -> List[Dict]:
    """Batch出力(JSONL)から埋め込みベクトルを抽出し、custom_idでタグ付け。"""
    rows: List[Dict] = []
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            custom_id = obj.get("custom_id", "")

            if obj.get("error"):
                logger.warning(f"error row custom_id={custom_id}: {obj['error']}")
                continue

            resp_body = obj.get("response", {}).get("body", {})
            # Embeddings API の標準レスポンス: { data: [ { embedding: [...], index: 0, object: 'embedding' } ], ... }
            data = resp_body.get("data", [])
            if not data:
                logger.warning(f"no embedding data custom_id={custom_id}")
                continue

            emb = data[0].get("embedding")
            if emb is None:
                logger.warning(f"missing embedding custom_id={custom_id}")
                continue

            rows.append({
                "custom_id": custom_id,
                "embedding": emb,
            })

    logger.info(f"Parsed embeddings: {len(rows)} rows")
    return rows


# -------------------------------------------------
# 保存
# -------------------------------------------------
def save_json(data: Dict | List, path: Path) -> Path:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def save_csv(rows: List[Dict], path: Path) -> Path:
    import csv
    ensure_parent_dir(path)
    if not rows:
        with path.open("w", encoding="utf-8") as f:
            f.write("")
        return path
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)
    return path


# -------------------------------------------------
# CLI
# -------------------------------------------------
def cmd_prepare_responses(args):
    out = Path(args.out)
    prepare_responses_jsonl(
        dataset=args.dataset,
        model=args.model,
        out_jsonl=out,
        max_docs=args.max_docs,
        merge_chunks=not args.no_merge,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_output_tokens=args.max_output_tokens,
    )


def cmd_prepare_embeddings(args):
    out = Path(args.out)
    prepare_embeddings_jsonl(
        dataset=args.dataset,
        embedding_model=args.embedding_model,
        out_jsonl=out,
        max_docs=args.max_docs,
        merge_chunks=not args.no_merge,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )


def cmd_submit(args):
    input_jsonl = Path(args.input)
    batch_id = submit_batch(input_jsonl, args.endpoint, metadata={"job": args.job or "qa"})
    print(batch_id)


def cmd_status(args):
    data = get_batch(args.batch_id)
    print(json.dumps(data, ensure_ascii=False, indent=2))


def cmd_download(args):
    out = Path(args.out)
    download_batch_output(args.batch_id, out)


def cmd_parse_responses(args):
    inp = Path(args.input)
    rows = parse_responses_output_jsonl(inp, dataset=args.dataset)
    # 保存
    ts = timestamp()
    outdir = Path(args.outdir)
    qa_json = outdir / f"qa_pairs_batch_{args.dataset}_{ts}.json"
    qa_csv = outdir / f"qa_pairs_batch_{args.dataset}_{ts}.csv"
    save_json(rows, qa_json)
    save_csv(rows, qa_csv)
    logger.info(f"Saved: {qa_json}, {qa_csv}")


def cmd_parse_embeddings(args):
    inp = Path(args.input)
    rows = parse_embeddings_output_jsonl(inp)
    # Parquet or JSON で保存（Parquetはpandas/pyarrowが必要。なければJSON保存）
    out = Path(args.out)
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        if out.suffix.lower() == ".parquet":
            df.to_parquet(out, index=False)
        elif out.suffix.lower() == ".csv":
            df.to_csv(out, index=False)
        else:
            save_json(rows, out)
        logger.info(f"Saved: {out}")
    except Exception as e:
        logger.warning(f"Parquet/CSV 保存に失敗したため JSON で保存します: {e}")
        save_json(rows, out.with_suffix(".json"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OpenAI Batch API 用のJSONL準備/投入/解析ヘルパー")
    sub = p.add_subparsers(dest="command", required=True)

    # prepare-responses
    pr = sub.add_parser("prepare-responses", help="/v1/responses 用のJSONLを作成")
    pr.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()), required=True)
    pr.add_argument("--model", type=str, default="gpt-4o-mini")
    pr.add_argument("--out", type=str, required=True, help="出力JSONLパス")
    pr.add_argument("--max-docs", type=int, default=None)
    pr.add_argument("--no-merge", action="store_true", help="小チャンク統合を無効化")
    pr.add_argument("--min-tokens", type=int, default=150)
    pr.add_argument("--max-tokens", type=int, default=400)
    pr.add_argument("--max-output-tokens", type=int, default=1000)
    pr.set_defaults(func=cmd_prepare_responses)

    # prepare-embeddings
    pe = sub.add_parser("prepare-embeddings", help="/v1/embeddings 用のJSONLを作成")
    pe.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()), required=True)
    pe.add_argument("--embedding-model", type=str, default="text-embedding-3-small")
    pe.add_argument("--out", type=str, required=True, help="出力JSONLパス")
    pe.add_argument("--max-docs", type=int, default=None)
    pe.add_argument("--no-merge", action="store_true")
    pe.add_argument("--min-tokens", type=int, default=150)
    pe.add_argument("--max-tokens", type=int, default=400)
    pe.set_defaults(func=cmd_prepare_embeddings)

    # submit
    sb = sub.add_parser("submit", help="JSONL を Batch API に投入")
    sb.add_argument("--input", required=True, help="JSONL ファイル")
    sb.add_argument("--endpoint", required=True, choices=["/v1/responses", "/v1/embeddings"])
    sb.add_argument("--job", type=str, default=None, help="任意のジョブ名メタデータ")
    sb.set_defaults(func=cmd_submit)

    # status
    st = sub.add_parser("status", help="Batch ステータス取得")
    st.add_argument("--batch-id", required=True)
    st.set_defaults(func=cmd_status)

    # download
    dl = sub.add_parser("download", help="完了済みBatchの出力を保存")
    dl.add_argument("--batch-id", required=True)
    dl.add_argument("--out", required=True, help="保存先JSONL")
    dl.set_defaults(func=cmd_download)

    # parse-responses
    prr = sub.add_parser("parse-responses", help="Responses出力(JSONL)を突合しQ/A行に展開")
    prr.add_argument("--input", required=True, help="Batch 出力 JSONL")
    prr.add_argument("--dataset", required=False, help="明示的にdatasetを指定（custom_idが持たない場合に使用）")
    prr.add_argument("--outdir", default="qa_output", help="保存ディレクトリ")
    prr.set_defaults(func=cmd_parse_responses)

    # parse-embeddings
    pre = sub.add_parser("parse-embeddings", help="Embeddings出力(JSONL)を抽出")
    pre.add_argument("--input", required=True, help="Batch 出力 JSONL")
    pre.add_argument("--out", required=True, help="保存先（.parquet / .csv / .json）")
    pre.set_defaults(func=cmd_parse_embeddings)

    return p


def main():
    _ = ensure_api_key()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

