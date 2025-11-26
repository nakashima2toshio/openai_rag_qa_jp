#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qdrant_registration_page.py - Qdrant登録ページ
==============================================
Q/AデータのQdrantへの登録機能

機能:
- CSVファイルからQdrantへの登録
- コレクション管理（作成・削除）
- 埋め込みベクトル生成
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from qdrant_client import QdrantClient

# サービスモジュールからインポート
from services.qdrant_service import (
    get_collection_stats,
    get_all_collections,
    delete_all_collections,
    load_csv_for_qdrant,
    build_inputs_for_embedding,
    embed_texts_for_qdrant,
    create_or_recreate_collection_for_qdrant,
    build_points_for_qdrant,
    upsert_points_to_qdrant,
)

logger = logging.getLogger(__name__)


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
                    add_log("✅ コレクション準備完了")

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
                        st.json(stats)
                except Exception as e:
                    logger.warning(f"統計情報取得エラー: {e}")

            except Exception as e:
                add_log(f"❌ エラー発生: {str(e)}")
                st.error(f"エラーが発生しました: {str(e)}")

        # ログ表示
        with log_container:
            if st.session_state["qdrant_logs"]:
                log_text = "\n".join(st.session_state["qdrant_logs"])
                st.text_area("処理ログ", value=log_text, height=300, disabled=True)
            else:
                st.info("登録処理を開始するとここにログが表示されます")