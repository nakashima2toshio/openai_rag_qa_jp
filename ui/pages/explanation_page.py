#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
explanation_page.py - システム説明ページ
========================================
RAGシステムのデータフロー、処理ステップ、ディレクトリ構造の説明
"""

import streamlit as st


def show_system_explanation_page():
    """システム説明ページ"""
    st.title("📖 システム説明")
    st.caption("RAGシステムのデータフロー・処理ステップ・ディレクトリ構造")

    st.markdown("---")

    # データフロー図
    st.subheader("🔄 データフロー例（CC-Newsの場合）")

    st.code(
        """
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
    """,
        language=None,
    )

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