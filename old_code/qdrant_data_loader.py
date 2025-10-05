#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qdrant データローダー - 簡略版データ投入スクリプト
使用方法:
  python qdrant_data_loader.py --recreate --limit 100
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime, timezone

try:
    import yaml
except ImportError:
    print("PyYAMLが必要です: pip install pyyaml")
    sys.exit(1)

from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI

# 設定読み込み
def load_config(path: str = "config.yml") -> Dict[str, Any]:
    """config.yml から設定を読み込む"""
    defaults = {
        "rag": {"collection": "qa_corpus"},
        "embeddings": {
            "primary": {
                "provider": "openai", 
                "model": "text-embedding-3-small", 
                "dims": 1536
            }
        },
        "qdrant": {"url": "http://localhost:6333"},
    }
    
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            # デフォルトとマージ
            for key in defaults:
                if key not in cfg:
                    cfg[key] = defaults[key]
                elif isinstance(defaults[key], dict):
                    for subkey in defaults[key]:
                        if subkey not in cfg[key]:
                            cfg[key][subkey] = defaults[key][subkey]
    else:
        cfg = defaults
    
    return cfg

def get_data_files() -> Dict[str, str]:
    """処理するデータファイルのパスを取得"""
    base_path = Path("OUTPUT")
    
    # 固定ファイル名またはパターンマッチング
    files = {
        "customer": "preprocessed_customer_support_faq_89rows_20250721_092004.csv",
        "medical": "preprocessed_medical_qa_19704rows_20250721_092658.csv",
        "legal": "preprocessed_legal_qa_4rows_20250721_100302.csv",
        "sciq": "preprocessed_sciq_qa_11679rows_20250721_095451.csv"
    }
    
    # 実際に存在するファイルを確認
    available_files = {}
    for domain, filename in files.items():
        filepath = base_path / filename
        if filepath.exists():
            available_files[domain] = str(filepath)
        else:
            # パターンマッチングで探す
            pattern = f"preprocessed_{domain}*qa*.csv"
            matching = list(base_path.glob(pattern))
            if matching:
                available_files[domain] = str(matching[0])
    
    return available_files

def load_and_prepare_data(filepath: str, limit: int = 0) -> pd.DataFrame:
    """CSVファイルを読み込みクリーニング"""
    df = pd.read_csv(filepath)
    
    # 列名をマッピング
    column_mappings = {
        'Question': 'question',
        'Response': 'answer',
        'Answer': 'answer',
        'correct_answer': 'answer'
    }
    df = df.rename(columns=column_mappings)
    
    # 必要な列が存在することを確認
    required_cols = ['question', 'answer']
    for col in required_cols:
        if col not in df.columns:
            # answerがない場合は空文字で作成
            df[col] = ""
    
    # クリーニング
    df = df.fillna("")
    df = df[df['question'].str.len() > 0]  # 空の質問を除外
    
    if limit > 0:
        df = df.head(limit)
    
    return df

def create_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """OpenAI APIを使用して埋め込みを生成"""
    client = OpenAI()
    embeddings = []
    
    # バッチ処理
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(model=model, input=batch)
        embeddings.extend([data.embedding for data in response.data])
    
    return embeddings

def setup_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int, recreate: bool = False):
    """Qdrantコレクションのセットアップ"""
    if recreate:
        try:
            client.delete_collection(collection_name)
            print(f"既存のコレクション '{collection_name}' を削除しました")
        except:
            pass
    
    # コレクションが存在しない場合は作成
    try:
        client.get_collection(collection_name)
        print(f"コレクション '{collection_name}' は既に存在します")
    except:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        print(f"コレクション '{collection_name}' を作成しました")
        
        # インデックスを作成
        client.create_payload_index(
            collection_name=collection_name,
            field_name="domain",
            field_type="keyword"
        )

def insert_data_to_qdrant(
    client: QdrantClient, 
    collection_name: str,
    df: pd.DataFrame,
    embeddings: List[List[float]],
    domain: str,
    offset: int = 0
):
    """データをQdrantに投入"""
    points = []
    timestamp = datetime.now(timezone.utc).isoformat()
    
    for idx, (_, row) in enumerate(df.iterrows()):
        point_id = offset + idx
        
        payload = {
            "domain": domain,
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "created_at": timestamp,
            "schema": "qa:v1"
        }
        
        points.append(models.PointStruct(
            id=point_id,
            vector=embeddings[idx],
            payload=payload
        ))
    
    # バッチで投入
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(collection_name=collection_name, points=batch)
    
    print(f"  {domain}: {len(points)}件のデータを投入しました")

def main():
    parser = argparse.ArgumentParser(description="Qdrantにデータを投入")
    parser.add_argument("--recreate", action="store_true", help="コレクションを再作成")
    parser.add_argument("--limit", type=int, default=0, help="各ドメインの最大件数（0=制限なし）")
    parser.add_argument("--collection", type=str, default=None, help="コレクション名")
    parser.add_argument("--qdrant-url", type=str, default=None, help="Qdrant URL")
    args = parser.parse_args()
    
    # 設定読み込み
    config = load_config()
    collection_name = args.collection or config.get("rag", {}).get("collection", "qa_corpus")
    qdrant_url = args.qdrant_url or config.get("qdrant", {}).get("url", "http://localhost:6333")
    embedding_config = config.get("embeddings", {}).get("primary", {})
    embedding_model = embedding_config.get("model", "text-embedding-3-small")
    vector_size = embedding_config.get("dims", 1536)
    
    print(f"Qdrantデータローダー開始")
    print(f"  URL: {qdrant_url}")
    print(f"  コレクション: {collection_name}")
    print(f"  埋め込みモデル: {embedding_model}")
    
    # Qdrantクライアント初期化
    try:
        client = QdrantClient(url=qdrant_url, timeout=30)
        # 接続テスト
        client.get_collections()
        print("✅ Qdrantサーバーに接続成功")
    except Exception as e:
        print(f"❌ Qdrantサーバーに接続できません: {e}")
        print("Qdrantサーバーを起動してください:")
        print("  cd docker-compose && docker-compose up -d")
        return 1
    
    # コレクションセットアップ
    setup_qdrant_collection(client, collection_name, vector_size, args.recreate)
    
    # データファイル取得
    data_files = get_data_files()
    if not data_files:
        print("❌ 処理するデータファイルが見つかりません")
        print("まず以下のスクリプトでデータを準備してください:")
        print("  python a30_011_make_rag_data_customer.py")
        print("  python a30_013_make_rag_data_medical.py")
        return 1
    
    print(f"\n📊 {len(data_files)}個のドメインデータを処理します")
    
    # 各ドメインのデータを処理
    total_points = 0
    point_offset = 0
    
    for domain, filepath in data_files.items():
        print(f"\n処理中: {domain}")
        print(f"  ファイル: {filepath}")
        
        # データ読み込み
        df = load_and_prepare_data(filepath, args.limit)
        if len(df) == 0:
            print(f"  ⚠️ 有効なデータがありません")
            continue
        
        # 埋め込みを作成
        print(f"  埋め込み生成中... ({len(df)}件)")
        texts = df['question'].tolist()
        embeddings = create_embeddings(texts, embedding_model)
        
        # Qdrantに投入
        insert_data_to_qdrant(
            client, 
            collection_name, 
            df, 
            embeddings, 
            domain,
            point_offset
        )
        
        point_offset += len(df)
        total_points += len(df)
    
    print(f"\n✅ 完了！合計 {total_points} 件のデータを投入しました")
    
    # 統計情報を表示
    try:
        collection_info = client.get_collection(collection_name)
        print(f"\nコレクション統計:")
        print(f"  総ポイント数: {collection_info.points_count}")
        print(f"  ベクトル次元: {collection_info.config.params.vectors.size}")
    except:
        pass
    
    print("\n検索UIを起動するには:")
    print("  streamlit run a50_rag_search_local_qdrant.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
