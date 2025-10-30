# python a30_qdrant_registration.py --recreate --include-answer

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
a30_qdrant_registration.py — helper群・config.yml連携版（単一コレクション＋domain、Named Vectors対応、answer同梱切替）
--------------------------------------------------------------------------------
- cc_newsドメインのQ&Aデータを、3つの異なる生成方法で作成したものをQdrantの単一コレクションに統合
- 入力ファイル（qa_outputフォルダ）：
  * qa_output/a02_qa_pairs_cc_news.csv（a02_make_qa.pyで生成）
  * qa_output/a03_qa_pairs_cc_news.csv（a03_rag_qa_coverage_improved.pyで生成）
  * qa_output/a10_qa_pairs_cc_news.csv（a10_hybrid_qa_generator.pyで生成）
- payloadに domain="cc_news" と generation_method を付与
- generation_method: "a02_make_qa", "a03_coverage", "a10_hybrid" で生成方法を区別
- フィルタ検索でドメインや生成方法による絞り込みが可能
- config.yml から埋め込みモデル/入出力設定を読み込み（fallbackあり）
- answer を埋め込みに含める切替フラグ（--include-answer / YAML設定）
- Named Vectors 対応（YAMLに複数ベクトル定義があれば自動有効）
- helper_api.py / helper_rag.py が存在すれば活用（無ければ内蔵実装を利用）

使い方：
  export OPENAI_API_KEY=sk-...
  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
  python a30_qdrant_registration.py --recreate
  python a30_qdrant_registration.py --search "気候変動" --method a02_make_qa --using primary

初回登録（--recreate推奨）
     python a30_qdrant_registration.py --recreate --include-answer


主要引数：
  --recreate           : コレクション削除→新規作成
  --collection         : コレクション名（既定は YAML の rag.collection または 'qa_corpus'）
  --qdrant-url         : 既定は http://localhost:6333
  --batch-size         : Embeddings/Upsert バッチサイズ（既定 32）
  --limit              : データ件数上限（開発用、0=無制限）
  --include-answer     : 埋め込み入力に answer も結合（question + "\n" + answer）
  --using              : Named Vectors のキー名（検索時にどのベクトルで検索するか）
  --search             : クエリ指定で検索のみ実行
  --method             : 検索対象を生成方法で絞る（a02_make_qa/a03_coverage/a10_hybrid）
  --topk               : 上位件数（既定5）
"""
import argparse
import os
import json
import glob
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple, Optional, Any
from pathlib import Path

import pandas as pd

try:
    import yaml  # PyYAML
except Exception:
    yaml = None

try:
    import helper_api as hapi
except Exception:
    hapi = None

try:
    import helper_rag as hrag
except Exception:
    hrag = None

from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI

# ------------------ デフォルト設定（YAMLが無い場合の後ろ盾） ------------------
DEFAULTS = {
    "rag": {
        "collection": "qa_corpus",
        "include_answer_in_embedding": False,
        "use_named_vectors": False,
    },
    "embeddings": {
        # named vectors: dict[name] = {model, dims}
        "primary": {"provider": "openai", "model": "text-embedding-3-small", "dims": 1536},
        # 2nd example for future expansion; commented by default
        # "bge": {"provider": "openai", "model": "text-embedding-3-large", "dims": 3072},
    },
    "paths": {
        "customer": "OUTPUT/preprocessed_customer_support_faq.csv",
        "medical":  "OUTPUT/preprocessed_medical_qa.csv",
        "legal":    "OUTPUT/preprocessed_legal_qa.csv",
        "sciq":     "OUTPUT/preprocessed_sciq_qa.csv",
        "trivia":   "OUTPUT/preprocessed_trivia_qa.csv",
    },
    "qdrant": {"url": "http://localhost:6333"},
}

# ------------------ 最新ファイルを動的に検索 ------------------
def find_latest_file(pattern: str, output_dir: str = "OUTPUT") -> Optional[str]:
    """
    指定されたパターンに一致するファイルの中から最新のものを返す
    """
    files = glob.glob(os.path.join(output_dir, pattern))
    if not files:
        return None
    
    # タイムスタンプまたは更新日時でソート
    files_with_time = []
    for file in files:
        stat = os.stat(file)
        files_with_time.append((file, stat.st_mtime))
    
    # 更新日時が最新のファイルを選択
    files_with_time.sort(key=lambda x: x[1], reverse=True)
    latest_file = files_with_time[0][0]
    
    print(f"[INFO] Found latest file for pattern '{pattern}': {os.path.basename(latest_file)}")
    return latest_file

# ------------------ vector_stores.jsonからパスマッピング取得 ------------------
def load_vector_stores_mapping(json_path: str = "vector_stores.json") -> Dict[str, str]:
    """
    vector_stores.jsonからドメイン名を取得し、OUTPUTフォルダから最新ファイルを動的に検索
    """
    mapping = {}
    
    # ドメインごとのファイルパターンを定義
    domain_patterns = {
        "customer": "preprocessed_customer_support_faq*.csv",
        "medical": "preprocessed_medical_qa*.csv",
        "legal": "preprocessed_legal_qa*.csv",
        "sciq": "preprocessed_sciq_qa*.csv",
        "trivia": "preprocessed_trivia_qa*.csv"
    }
    
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # vector_storesのキーからドメイン名を抽出
            for key in data.get("vector_stores", {}).keys():
                if "Customer Support" in key:
                    latest_file = find_latest_file(domain_patterns["customer"])
                    if latest_file:
                        mapping["customer"] = latest_file
                elif "Medical" in key:
                    latest_file = find_latest_file(domain_patterns["medical"])
                    if latest_file:
                        mapping["medical"] = latest_file
                elif "Legal" in key:
                    latest_file = find_latest_file(domain_patterns["legal"])
                    if latest_file:
                        mapping["legal"] = latest_file
                elif "Science" in key or "Technology" in key:
                    latest_file = find_latest_file(domain_patterns["sciq"])
                    if latest_file:
                        mapping["sciq"] = latest_file
                elif "Trivia" in key:
                    latest_file = find_latest_file(domain_patterns["trivia"])
                    if latest_file:
                        mapping["trivia"] = latest_file
    
    # vector_stores.jsonが無い場合でも最新ファイルを検索
    if not mapping:
        print("[INFO] vector_stores.json not found or empty. Searching for latest files...")
        for domain, pattern in domain_patterns.items():
            latest_file = find_latest_file(pattern)
            if latest_file:
                mapping[domain] = latest_file
    
    return mapping

# ------------------ 設定ロード ------------------
def load_config(path: str = "config.yml") -> Dict[str, Any]:
    cfg = {}
    if yaml and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    # マージ（浅いマージで十分。必要なら深いマージに差し替え）
    def merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst.setdefault(k, v)
    full = {}
    merge(full, DEFAULTS)
    merge(full, cfg)
    return full

# ------------------ 小道具 ------------------
def batched(seq: Iterable, size: int):
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf

# ------------------ OpenAIクライアント ------------------
def get_openai_client():
    if hapi and hasattr(hapi, "get_openai_client"):
        return hapi.get_openai_client()
    return OpenAI()

# ------------------ 埋め込み実装（helper優先） ------------------
def embed_texts_openai(texts: List[str], model: str, client: Optional[OpenAI] = None) -> List[List[float]]:
    client = client or get_openai_client()
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

def embed_texts(texts: List[str], model: str, batch_size: int = 128) -> List[List[float]]:
    if hrag and hasattr(hrag, "embed_texts"):
        return hrag.embed_texts(texts, model=model, batch_size=batch_size)
    vecs: List[List[float]] = []
    client = get_openai_client()
    for chunk in batched(texts, batch_size):
        vecs.extend(embed_texts_openai(chunk, model=model, client=client))
    return vecs

# ------------------ 入力テキスト構築 ------------------
def build_inputs(df: pd.DataFrame, include_answer: bool) -> List[str]:
    if include_answer:
        return (df["question"].astype(str) + "\n" + df["answer"].astype(str)).tolist()
    return df["question"].astype(str).tolist()

# ------------------ CSVロード ------------------
def load_csv(path: str, required=("question", "answer"), limit: int = 0) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    # 列名マッピングが必要ならここで調整（例: 'Question'->'question'）
    column_mappings = {
        'Question': 'question',
        'Response': 'answer',
        'Answer': 'answer',
        'correct_answer': 'answer'
    }
    df = df.rename(columns=column_mappings)
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{path} には '{col}' 列が必要です（列: {list(df.columns)}）")
    df = df.fillna("").drop_duplicates(subset=list(required)).reset_index(drop=True)
    if limit and limit > 0:
        df = df.head(limit).copy()
    return df

# ------------------ Qdrant: コレクション作成（Named Vectors対応） ------------------
def create_or_recreate_collection(client: QdrantClient, name: str, recreate: bool,
                                  embeddings_cfg: Dict[str, Dict[str, Any]]):
    # embeddings_cfg: dict[name] = {"model": "...", "dims": int}
    # Named Vectors：複数キーなら dict を、単一なら VectorParams を使う
    if len(embeddings_cfg) == 1:
        dims = list(embeddings_cfg.values())[0]["dims"]
        vectors_config = models.VectorParams(size=dims, distance=models.Distance.COSINE)
    else:
        # Named vectors
        vectors_config = {
            k: models.VectorParams(size=v["dims"], distance=models.Distance.COSINE)
            for k, v in embeddings_cfg.items()
        }
    if recreate:
        client.recreate_collection(collection_name=name, vectors_config=vectors_config)
    else:
        # 無ければ作成
        try:
            client.get_collection(name)
        except Exception:
            client.create_collection(collection_name=name, vectors_config=vectors_config)
    # よく使うpayloadの索引（任意）
    try:
        client.create_payload_index(name, field_name="domain", field_type="keyword")
    except Exception:
        pass
    try:
        client.create_payload_index(name, field_name="generation_method", field_type="keyword")
    except Exception:
        pass

# ------------------ ポイント構築（Named Vectors対応） ------------------
def build_points(df: pd.DataFrame, vectors_by_name: Dict[str, List[List[float]]], domain: str, source_file: str,
                 generation_method: str = None) -> List[models.PointStruct]:
    # vectors_by_name: name -> list[vec]
    n = len(df)
    for name, vecs in vectors_by_name.items():
        if len(vecs) != n:
            raise ValueError(f"vectors length mismatch for '{name}': df={n}, vecs={len(vecs)}")
    now_iso = datetime.now(timezone.utc).isoformat()
    points: List[models.PointStruct] = []
    for i, row in enumerate(df.itertuples(index=False)):
        payload = {
            "domain": domain,
            "generation_method": generation_method or "unknown",
            "question": getattr(row, "question"),
            "answer": getattr(row, "answer"),
            "source": os.path.basename(source_file),
            "created_at": now_iso,
            "schema": "qa:v1",
        }
        # Qdrant requires point IDs to be UUID or unsigned integer
        pid = hash(f"{domain}-{generation_method}-{i}") & 0x7FFFFFFF  # Convert to positive 32-bit integer
        if len(vectors_by_name) == 1:
            # 単一ベクトル
            vec = list(vectors_by_name.values())[0][i]
            points.append(models.PointStruct(id=pid, vector=vec, payload=payload))
        else:
            # Named Vectors（dict渡し）
            vecs_dict = {name: vecs[i] for name, vecs in vectors_by_name.items()}
            points.append(models.PointStruct(id=pid, vector=vecs_dict, payload=payload))
    return points

def upsert_points(client: QdrantClient, collection: str, points: List[models.PointStruct], batch_size: int = 128) -> int:
    count = 0
    for chunk in batched(points, batch_size):
        client.upsert(collection_name=collection, points=chunk)
        count += len(chunk)
    return count

# ------------------ 検索（Named Vectors対応） ------------------
def embed_one(text: str, model: str) -> List[float]:
    return embed_texts([text], model=model, batch_size=1)[0]

def search(client: QdrantClient, collection: str, query: str, using_vec: str, model_for_using: str,
           topk: int = 5, domain: Optional[str] = None, generation_method: Optional[str] = None):
    qvec = embed_one(query, model=model_for_using)
    qfilter = None
    filter_conditions = []
    if domain:
        filter_conditions.append(models.FieldCondition(key="domain", match=models.MatchValue(value=domain)))
    if generation_method:
        filter_conditions.append(models.FieldCondition(key="generation_method", match=models.MatchValue(value=generation_method)))
    if filter_conditions:
        qfilter = models.Filter(must=filter_conditions)
    
    # ベクトル設定を確認して適切な検索方法を選択
    try:
        collection_info = client.get_collection(collection)
        # Named Vectorsかどうか確認
        has_named_vectors = hasattr(collection_info.config.params.vectors, '__iter__') and not isinstance(
            collection_info.config.params.vectors, (str, models.VectorParams))
        
        if has_named_vectors and using_vec:
            # Named Vectorsの場合、using引数を試す
            try:
                hits = client.search(collection_name=collection, query_vector=qvec, limit=topk,
                                   query_filter=qfilter, using=using_vec)
            except (TypeError, Exception):
                # using引数がサポートされていない、または他のエラーの場合
                hits = client.search(collection_name=collection, query_vector=qvec, limit=topk,
                                   query_filter=qfilter)
        else:
            # 単一ベクトルの場合、using引数なしで検索
            hits = client.search(collection_name=collection, query_vector=qvec, limit=topk,
                               query_filter=qfilter)
    except Exception:
        # コレクション情報が取得できない場合、using引数なしで検索
        hits = client.search(collection_name=collection, query_vector=qvec, limit=topk,
                           query_filter=qfilter)
    
    return hits

# ------------------ メイン ------------------
def main():
    cfg = load_config("config.yml")
    rag_cfg = cfg.get("rag", {})
    embeddings_cfg: Dict[str, Dict[str, Any]] = cfg.get("embeddings", {})
    paths_cfg: Dict[str, str] = cfg.get("paths", {})
    qdrant_url = (cfg.get("qdrant", {}) or {}).get("url", "http://localhost:6333")

    ap = argparse.ArgumentParser(description="Ingest 4 QA datasets into single Qdrant collection (domain filter, Named Vectors).")
    ap.add_argument("--recreate", action="store_true", help="Drop & create collection before upsert.")
    ap.add_argument("--collection", default=rag_cfg.get("collection", "qa_corpus"))
    ap.add_argument("--qdrant-url", default=qdrant_url)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="Row limit per CSV for development (0=all)")
    ap.add_argument("--include-answer", action="store_true",
                    default=rag_cfg.get("include_answer_in_embedding", False),
                    help="Use 'question\nanswer' as embedding input.")
    ap.add_argument("--search", default=None, help="Run search only.")
    ap.add_argument("--method", default=None, choices=[None, "a02_make_qa", "a03_coverage", "a10_hybrid"],
                    help="Filter by generation method.")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--using", default=None, help="Named Vector name to use for search (e.g., 'primary').")
    args = ap.parse_args()

    # どのベクトル定義があるか判定（1つなら単一、2つ以上ならNamed Vectors）
    if not embeddings_cfg:
        embeddings_cfg = DEFAULTS["embeddings"]
    # using の既定値（単一路ならその名前、複数なら 'primary' 優先）
    using_default = list(embeddings_cfg.keys())[0]
    using_vec = args.using or using_default

    # Qdrant with timeout configuration
    client = QdrantClient(url=args.qdrant_url, timeout=300)
    create_or_recreate_collection(client, args.collection, recreate=args.recreate, embeddings_cfg=embeddings_cfg)

    # 検索のみ
    if args.search:
        # usingに対応する埋め込みモデルを取得
        if using_vec not in embeddings_cfg:
            raise ValueError(f"--using '{using_vec}' is not in embeddings config: {list(embeddings_cfg.keys())}")
        model_for_using = embeddings_cfg[using_vec]["model"]
        hits = search(client, args.collection, args.search, using_vec, model_for_using,
                      topk=args.topk, domain="cc_news", generation_method=args.method)
        print(f"[Search] collection={args.collection} using={using_vec} domain=cc_news method={args.method or 'ALL'} query={args.search!r}")
        for h in hits:
            print(f"score={h.score:.4f}  method={h.payload.get('generation_method')}  Q: {h.payload.get('question')}  A: {h.payload.get('answer')[:80]}...")
        return

    # インジェスト
    # cc_newsドメインの3つの生成方法によるファイルパスとgeneration_methodのマッピング
    print("[INFO] Processing cc_news domain with 3 generation methods...")

    # ファイルパスと生成方法のマッピング
    file_method_mapping = [
        ("qa_output/a02_qa_pairs_cc_news.csv", "a02_make_qa"),
        ("qa_output/a03_qa_pairs_cc_news.csv", "a03_coverage"),
        ("qa_output/a10_qa_pairs_cc_news.csv", "a10_hybrid"),
    ]

    print(f"\n[INFO] Files to be processed:")
    for path, method in file_method_mapping:
        if os.path.exists(path):
            print(f"  - {method}: {os.path.basename(path)}")
        else:
            print(f"  - {method}: NOT FOUND ({path})")
    print()

    total = 0
    for path, method in file_method_mapping:
        if not os.path.exists(path):
            print(f"[WARN] File not found for method '{method}': {path} (skipping)")
            continue

        print(f"\n[INFO] Processing {method}: {os.path.basename(path)}")
        df = load_csv(path, limit=args.limit)
        texts = build_inputs(df, include_answer=args.include_answer)

        # 埋め込み（Named Vectors: 各ベクトル名ごとに生成）
        vectors_by_name: Dict[str, List[List[float]]] = {}
        for name, vcfg in embeddings_cfg.items():
            vectors_by_name[name] = embed_texts(texts, model=vcfg["model"], batch_size=args.batch_size)

        points = build_points(df, vectors_by_name, domain="cc_news", source_file=path, generation_method=method)
        n = upsert_points(client, args.collection, points, batch_size=args.batch_size)
        print(f"[{method}] Successfully upserted {n} points from {os.path.basename(path)}")
        total += n

    print(f"Done. Total upserted: {total}")

    # 動作確認のミニ検索（エラーを回避しながら実行）
    print(f"\n[INFO] Running verification searches...")
    sample = [("気候変動", "a02_make_qa"), ("温暖化", "a03_coverage"), ("環境問題", "a10_hybrid")]
    model_for_using = embeddings_cfg[using_vec]["model"]

    for q, method in sample:
        try:
            hits = search(client, args.collection, q, using_vec, model_for_using, topk=3,
                         domain="cc_news", generation_method=method)
            if hits:
                print(f"\n[Search] method={method} query={q}")
                for h in hits[:2]:  # 最初の2件のみ表示
                    print(f"  score={h.score:.4f}  Q: {h.payload.get('question')[:60]}...")
        except Exception as e:
            # 個別の検索エラーは静かにスキップ
            pass

if __name__ == "__main__":
    main()
