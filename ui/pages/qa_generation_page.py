#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qa_generation_page.py - Q/A生成ページ
=====================================
Q/Aペアの自動生成機能
"""

import streamlit as st


def show_qa_generation_page():
    """Q/A生成ページ（スタブ）

    Note: 完全な実装は rag_qa_pair_qdrant.py にあります。
    このファイルは段階的移行のためのスタブです。
    """
    # 完全な機能は元のファイルからインポート
    try:
        from rag_qa_pair_qdrant import show_qa_generation_page as _show_qa_generation_page
        _show_qa_generation_page()
    except ImportError as e:
        st.error(f"ページの読み込みに失敗しました: {e}")
        st.info("rag_qa_pair_qdrant.py が見つかりません")