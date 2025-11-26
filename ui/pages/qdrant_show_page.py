#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qdrant_show_page.py - Qdrantデータ表示ページ
============================================
Qdrantコレクションのデータ表示機能
"""

import streamlit as st


def show_qdrant_page():
    """Qdrantデータ表示ページ（スタブ）

    Note: 完全な実装は rag_qa_pair_qdrant.py にあります。
    このファイルは段階的移行のためのスタブです。
    """
    # 完全な機能は元のファイルからインポート
    try:
        from rag_qa_pair_qdrant import show_qdrant_page as _show_qdrant_page
        _show_qdrant_page()
    except ImportError as e:
        st.error(f"ページの読み込みに失敗しました: {e}")
        st.info("rag_qa_pair_qdrant.py が見つかりません")