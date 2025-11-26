#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_page.py - RAGデータダウンロードページ
==============================================
HuggingFaceからのデータダウンロードと前処理
"""

import streamlit as st


def show_rag_download_page():
    """RAGデータダウンロードページ（スタブ）

    Note: 完全な実装は rag_qa_pair_qdrant.py にあります。
    このファイルは段階的移行のためのスタブです。
    """
    # 完全な機能は元のファイルからインポート
    try:
        from rag_qa_pair_qdrant import show_rag_download_page as _show_rag_download_page
        _show_rag_download_page()
    except ImportError as e:
        st.error(f"ページの読み込みに失敗しました: {e}")
        st.info("rag_qa_pair_qdrant.py が見つかりません")