#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qdrant_search_page.py - Qdrant検索ページ
========================================
Qdrantを使用した類似検索機能
"""

import streamlit as st


def show_qdrant_search_page():
    """Qdrant検索ページ（スタブ）

    Note: 完全な実装は rag_qa_pair_qdrant.py にあります。
    このファイルは段階的移行のためのスタブです。
    """
    # 完全な機能は元のファイルからインポート
    try:
        from rag_qa_pair_qdrant import show_qdrant_search_page as _show_qdrant_search_page
        _show_qdrant_search_page()
    except ImportError as e:
        st.error(f"ページの読み込みに失敗しました: {e}")
        st.info("rag_qa_pair_qdrant.py が見つかりません")