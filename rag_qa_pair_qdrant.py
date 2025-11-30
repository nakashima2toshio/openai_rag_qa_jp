#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG Q&A生成・Qdrant管理 Streamlit アプリケーション

================================================================================
【実行方法】
================================================================================
# 基本起動
streamlit run rag_qa_pair_qdrant.py --server.port=8500

# 事前準備（必要な場合）
1. Qdrantサーバー起動: docker-compose -f docker-compose/docker-compose.yml up -d
2. Celeryワーカー起動: redis-cli FLUSHDB && ./start_celery.sh restart -w 24
3. Flower監視（任意）: celery -A celery_config flower --port=5555

================================================================================
【仕様概要】
================================================================================
本アプリケーションは以下の6画面で構成される統合RAGツールです。

■ 画面構成
  1. 📖 説明           - システムのデータフロー・ディレクトリ構造を表示
  2. 📥 RAGデータDL    - HuggingFaceまたはローカルファイルからデータ取得・前処理
  3. 🤖 Q/A生成        - OpenAI APIによるQ&Aペア自動生成（Celery並列処理対応）
  4. 🗄️ Qdrant登録    - Q&AペアをQdrantベクトルDBに登録
  5. 🔍 Show-Qdrant    - Qdrantコレクション内容の閲覧
  6. 🔎 Qdrant検索     - セマンティック検索によるQ&A検索

■ 対応データセット
  - wikipedia_ja   : Wikipedia日本語版
  - japanese_text  : CC100日本語（Webテキスト）
  - cc_news        : CC-News英語ニュース
  - livedoor       : Livedoorニュースコーパス（9カテゴリ、7,376件）
  - custom_upload  : ローカルファイル（CSV/TXT/JSON/JSONL）

■ データフロー
  HuggingFace/ファイル → datasets/ → OUTPUT/ → qa_output/ → Qdrant
     ↓ダウンロード         ↓前処理      ↓Q&A生成     ↓ベクトル登録

■ 主要機能
  - データセット自動ダウンロード・前処理・クレンジング
  - OpenAI API（GPT-4o-mini等）によるQ&Aペア生成
  - Celery並列処理による大規模データ対応（a02_make_qa_para.py連携）
  - text-embedding-3-smallによるベクトル化・Qdrant登録
  - コサイン類似度によるセマンティック検索

■ 出力先ディレクトリ
  - datasets/   : ダウンロードした生データ
  - OUTPUT/     : 前処理済みCSV（preprocessed_*.csv）
  - qa_output/  : 生成Q&AペアCSV/JSON

■ 依存サービス
  - OpenAI API  : Q&A生成・埋め込み生成
  - Qdrant      : ベクトルデータベース（localhost:6333）
  - Redis       : Celeryブローカー（並列処理時）

================================================================================
【モジュール構成】
================================================================================
本ファイルはエントリポイントとして機能し、実際の処理は以下のモジュールに分離：

- services/
  - dataset_service.py  : データセット操作（ダウンロード、前処理）
  - qdrant_service.py   : Qdrant操作（CRUD、ヘルスチェック）
  - file_service.py     : ファイル操作（履歴読み込み、保存）
  - qa_service.py       : Q/A生成（OpenAI API、サブプロセス実行）

- ui/pages/
  - explanation_page.py       : システム説明ページ
  - download_page.py          : RAGデータダウンロードページ
  - qa_generation_page.py     : Q/A生成ページ
  - qdrant_registration_page.py : Qdrant登録ページ
  - qdrant_show_page.py       : Qdrant表示ページ
  - qdrant_search_page.py     : Qdrant検索ページ
"""

import streamlit as st

# UIページをインポート
from ui.pages import (
    show_system_explanation_page,
    show_rag_download_page,
    show_qa_generation_page,
    show_qdrant_registration_page,
    show_qdrant_page,
    show_qdrant_search_page,
)


def main():
    """メインアプリケーション - 画面選択"""

    # ページ設定
    st.set_page_config(page_title="OpenAI-RAGツール", page_icon="🤖", layout="wide")

    # サイドバー：画面選択
    with st.sidebar:
        st.title("🤖 OpenAI-RAGツール")
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
    page_mapping = {
        "explanation": show_system_explanation_page,
        "rag_download": show_rag_download_page,
        "qa_generation": show_qa_generation_page,
        "qdrant_registration": show_qdrant_registration_page,
        "show_qdrant": show_qdrant_page,
        "qdrant_search": show_qdrant_search_page,
    }
    page_mapping[page]()


if __name__ == "__main__":
    main()