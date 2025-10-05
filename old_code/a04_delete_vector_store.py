# a04_delete_vector_store.py
# OpenAI Vector StoreとFiles削除バッチスクリプト
# streamlit run a04_delete_vector_store.py --server.port=8504

import streamlit as st
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

# OpenAI SDK のインポート
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError as e:
    OPENAI_AVAILABLE = False
    st.error(f"OpenAI SDK が見つかりません: {e}")
    st.stop()

# ===================================================================
# ログ設定
# ===================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================================================================
# 定数定義
# ===================================================================
class DeletionMode(Enum):
    """削除モード"""
    INDIVIDUAL = "individual"  # 個別選択削除
    BATCH = "batch"            # 複数選択削除
    ALL = "all"                # 全削除

@dataclass
class DeletionResult:
    """削除結果データクラス"""
    success: bool
    item_type: str  # "vector_store" または "file"
    item_id: str
    item_name: str
    error: Optional[str] = None
    timestamp: Optional[datetime] = None

# ===================================================================
# Vector Store/File管理クラス
# ===================================================================
class OpenAIResourceManager:
    """OpenAIリソース管理クラス"""
    
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OpenAI APIキーが設定されていません。環境変数 OPENAI_API_KEY を確認してください。")
        
        self.client = OpenAI(api_key=api_key)
        self.deletion_history = []
    
    def list_vector_stores(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Vector Store一覧を取得"""
        try:
            stores = self.client.vector_stores.list(limit=limit)
            store_list = []
            
            for store in stores.data:
                store_info = {
                    "id": store.id,
                    "name": store.name or "Unnamed",
                    "file_counts": getattr(store.file_counts, 'total', 0) if store.file_counts else 0,
                    "created_at": store.created_at,
                    "usage_bytes": store.usage_bytes or 0,
                    "metadata": store.metadata or {}
                }
                store_list.append(store_info)
            
            return sorted(store_list, key=lambda x: x['created_at'], reverse=True)
        except Exception as e:
            logger.error(f"Vector Store一覧取得エラー: {e}")
            return []
    
    def list_files(self, limit: int = 100, purpose: str = "assistants") -> List[Dict[str, Any]]:
        """ファイル一覧を取得"""
        try:
            files = self.client.files.list(limit=limit)
            file_list = []
            
            for file in files.data:
                # purpose でフィルタリング（オプション）
                if purpose and hasattr(file, 'purpose') and file.purpose != purpose:
                    continue
                
                file_info = {
                    "id": file.id,
                    "filename": file.filename or "Unnamed",
                    "purpose": getattr(file, 'purpose', 'unknown'),
                    "bytes": file.bytes or 0,
                    "created_at": file.created_at,
                    "status": getattr(file, 'status', 'unknown')
                }
                file_list.append(file_info)
            
            return sorted(file_list, key=lambda x: x['created_at'], reverse=True)
        except Exception as e:
            logger.error(f"ファイル一覧取得エラー: {e}")
            return []
    
    def get_vector_store_files(self, vector_store_id: str) -> List[Dict[str, str]]:
        """特定のVector Storeに関連付けられたファイル情報を取得"""
        try:
            vector_store_files = self.client.vector_stores.files.list(
                vector_store_id=vector_store_id
            )
            file_info = []
            for vsf in vector_store_files.data:
                # ファイルIDから詳細情報を取得
                try:
                    file = self.client.files.retrieve(vsf.id)
                    file_info.append({
                        "id": vsf.id,
                        "filename": getattr(file, 'filename', 'Unknown')
                    })
                except:
                    file_info.append({
                        "id": vsf.id,
                        "filename": "Unknown"
                    })
            return file_info
        except Exception as e:
            logger.error(f"Vector Storeファイル取得エラー: {e}")
            return []
    
    def delete_vector_store(self, vector_store_id: str, store_name: str = None, delete_associated_files: bool = False) -> Tuple[DeletionResult, List[DeletionResult]]:
        """Vector Storeを削除（オプションで関連ファイルも削除）"""
        file_deletion_results = []
        
        # 関連ファイルの削除（オプション）
        if delete_associated_files:
            associated_files = self.get_vector_store_files(vector_store_id)
            logger.info(f"Vector Store {vector_store_id} に関連付けられたファイル: {len(associated_files)}個")
            
            for file_info in associated_files:
                file_result = self.delete_file(file_info['id'], file_info['filename'])
                file_deletion_results.append(file_result)
                time.sleep(0.2)  # API rate limit対策
        
        # Vector Store本体を削除
        try:
            self.client.vector_stores.delete(vector_store_id)
            
            result = DeletionResult(
                success=True,
                item_type="vector_store",
                item_id=vector_store_id,
                item_name=store_name or vector_store_id,
                timestamp=datetime.now()
            )
            
            self.deletion_history.append(result)
            logger.info(f"Vector Store削除成功: {vector_store_id} ({store_name})")
            
            if delete_associated_files and file_deletion_results:
                success_files = sum(1 for r in file_deletion_results if r.success)
                logger.info(f"関連ファイル削除: {success_files}/{len(file_deletion_results)} 成功")
            
            return result, file_deletion_results
            
        except Exception as e:
            error_msg = str(e)
            result = DeletionResult(
                success=False,
                item_type="vector_store",
                item_id=vector_store_id,
                item_name=store_name or vector_store_id,
                error=error_msg,
                timestamp=datetime.now()
            )
            
            self.deletion_history.append(result)
            logger.error(f"Vector Store削除失敗: {vector_store_id} - {error_msg}")
            return result, file_deletion_results
    
    def delete_file(self, file_id: str, filename: str = None) -> DeletionResult:
        """ファイルを削除"""
        try:
            # ファイルを削除
            self.client.files.delete(file_id)
            
            result = DeletionResult(
                success=True,
                item_type="file",
                item_id=file_id,
                item_name=filename or file_id,
                timestamp=datetime.now()
            )
            
            self.deletion_history.append(result)
            logger.info(f"ファイル削除成功: {file_id} ({filename})")
            return result
            
        except Exception as e:
            error_msg = str(e)
            result = DeletionResult(
                success=False,
                item_type="file",
                item_id=file_id,
                item_name=filename or file_id,
                error=error_msg,
                timestamp=datetime.now()
            )
            
            self.deletion_history.append(result)
            logger.error(f"ファイル削除失敗: {file_id} - {error_msg}")
            return result
    
    def batch_delete_vector_stores(self, vector_store_ids: List[Tuple[str, str]], delete_associated_files: bool = False) -> Tuple[List[DeletionResult], List[DeletionResult]]:
        """複数のVector Storeを一括削除（オプションで関連ファイルも削除）"""
        vs_results = []
        all_file_results = []
        
        for store_id, store_name in vector_store_ids:
            vs_result, file_results = self.delete_vector_store(store_id, store_name, delete_associated_files)
            vs_results.append(vs_result)
            all_file_results.extend(file_results)
            time.sleep(0.5)  # API rate limit対策
        
        return vs_results, all_file_results
    
    def batch_delete_files(self, file_ids: List[Tuple[str, str]]) -> List[DeletionResult]:
        """複数のファイルを一括削除"""
        results = []
        for file_id, filename in file_ids:
            result = self.delete_file(file_id, filename)
            results.append(result)
            time.sleep(0.5)  # API rate limit対策
        return results
    
    def delete_all_vector_stores(self) -> List[DeletionResult]:
        """全てのVector Storeを削除（危険な操作）"""
        stores = self.list_vector_stores()
        store_ids = [(store['id'], store['name']) for store in stores]
        return self.batch_delete_vector_stores(store_ids)
    
    def delete_all_files(self, purpose: str = "assistants") -> List[DeletionResult]:
        """全てのファイルを削除（危険な操作）"""
        files = self.list_files(purpose=purpose)
        file_ids = [(file['id'], file['filename']) for file in files]
        return self.batch_delete_files(file_ids)
    
    def save_deletion_history(self, filepath: str = "deletion_history.json"):
        """削除履歴を保存"""
        history_data = []
        for result in self.deletion_history:
            history_data.append({
                "success": result.success,
                "item_type": result.item_type,
                "item_id": result.item_id,
                "item_name": result.item_name,
                "error": result.error,
                "timestamp": result.timestamp.isoformat() if result.timestamp else None
            })
        
        output_dir = Path("OUTPUT")
        output_dir.mkdir(exist_ok=True)
        
        filepath = output_dir / filepath
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"削除履歴を保存しました: {filepath}")
        return filepath

# ===================================================================
# Streamlit UI
# ===================================================================
class DeletionUI:
    """削除UI管理クラス"""
    
    def __init__(self):
        self.manager = None
    
    def setup_page(self):
        """ページ設定"""
        try:
            st.set_page_config(
                page_title="Vector Store/Files削除ツール",
                page_icon="🗑️",
                layout="wide",
                initial_sidebar_state="expanded"
            )
        except st.errors.StreamlitAPIException:
            pass
    
    def setup_header(self):
        """ヘッダー設定"""
        st.title("🗑️ OpenAI Vector Store/Files削除ツール")
        st.caption("Vector StoreとFilesの管理・削除を行うツール")
        
        # 警告表示
        st.warning("⚠️ **注意**: このツールは削除操作を行います。削除されたデータは復元できません。")
        st.markdown("---")
    
    def setup_sidebar(self) -> Dict[str, Any]:
        """サイドバー設定"""
        st.sidebar.title("🗑️ 削除設定")
        st.sidebar.markdown("---")
        
        # 削除モード選択
        deletion_mode = st.sidebar.selectbox(
            "削除モード",
            [DeletionMode.INDIVIDUAL.value, DeletionMode.BATCH.value, DeletionMode.ALL.value],
            format_func=lambda x: {
                DeletionMode.INDIVIDUAL.value: "個別選択削除",
                DeletionMode.BATCH.value: "複数選択削除",
                DeletionMode.ALL.value: "全削除（危険）"
            }[x]
        )
        
        # 安全確認
        confirm_delete = False
        if deletion_mode == DeletionMode.ALL.value:
            st.sidebar.error("⚠️ 全削除モードが選択されています")
            confirm_delete = st.sidebar.checkbox(
                "本当に全削除を実行しますか？",
                value=False,
                help="この操作は取り消せません"
            )
        
        # APIキー確認
        with st.sidebar.expander("🔑 API設定確認", expanded=False):
            api_key_status = "✅ 設定済み" if os.getenv("OPENAI_API_KEY") else "❌ 未設定"
            st.write(f"**OpenAI APIキー**: {api_key_status}")
            
            if not os.getenv("OPENAI_API_KEY"):
                st.error("環境変数 OPENAI_API_KEY を設定してください")
                st.code("export OPENAI_API_KEY='your-api-key-here'")
        
        return {
            "deletion_mode": deletion_mode,
            "confirm_delete": confirm_delete
        }
    
    def display_vector_stores(self, stores: List[Dict]) -> List[Tuple[str, str]]:
        """Vector Store一覧表示と選択"""
        st.subheader("📚 Vector Store一覧")
        
        if not stores:
            st.info("Vector Storeが見つかりません")
            return []
        
        # データフレーム用データ準備
        selected_stores = []
        
        # テーブル表示用のデータ
        for idx, store in enumerate(stores):
            col1, col2, col3, col4, col5, col6 = st.columns([0.5, 3, 2, 1.5, 2, 2])
            
            with col1:
                selected = st.checkbox("", key=f"vs_{store['id']}")
                if selected:
                    selected_stores.append((store['id'], store['name']))
            
            with col2:
                st.write(f"**{store['name']}**")
            
            with col3:
                st.caption(f"ID: {store['id'][:8]}...")
            
            with col4:
                st.write(f"📁 {store['file_counts']} files")
            
            with col5:
                size_mb = store['usage_bytes'] / (1024 * 1024)
                st.write(f"💾 {size_mb:.1f} MB")
            
            with col6:
                created_date = datetime.fromtimestamp(store['created_at'])
                st.caption(created_date.strftime("%Y-%m-%d %H:%M"))
        
        st.write(f"**選択数**: {len(selected_stores)}/{len(stores)}")
        return selected_stores
    
    def display_files(self, files: List[Dict]) -> List[Tuple[str, str]]:
        """ファイル一覧表示と選択"""
        st.subheader("📁 ファイル一覧")
        
        if not files:
            st.info("ファイルが見つかりません")
            return []
        
        # データフレーム用データ準備
        selected_files = []
        
        # テーブル表示用のデータ
        for idx, file in enumerate(files):
            col1, col2, col3, col4, col5, col6 = st.columns([0.5, 3, 2, 1.5, 1.5, 2])
            
            with col1:
                selected = st.checkbox("", key=f"file_{file['id']}")
                if selected:
                    selected_files.append((file['id'], file['filename']))
            
            with col2:
                st.write(f"**{file['filename']}**")
            
            with col3:
                st.caption(f"ID: {file['id'][:8]}...")
            
            with col4:
                st.caption(f"Type: {file['purpose']}")
            
            with col5:
                size_mb = file['bytes'] / (1024 * 1024)
                st.write(f"💾 {size_mb:.1f} MB")
            
            with col6:
                created_date = datetime.fromtimestamp(file['created_at'])
                st.caption(created_date.strftime("%Y-%m-%d %H:%M"))
        
        st.write(f"**選択数**: {len(selected_files)}/{len(files)}")
        return selected_files
    
    def display_deletion_results(self, results: List[DeletionResult]):
        """削除結果表示"""
        st.subheader("📊 削除結果")
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        # サマリー
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("処理総数", len(results))
        with col2:
            st.metric("成功", len(successful))
        with col3:
            st.metric("失敗", len(failed))
        
        # 成功リスト
        if successful:
            with st.expander(f"✅ 削除成功 ({len(successful)}件)", expanded=True):
                for result in successful:
                    icon = "📚" if result.item_type == "vector_store" else "📁"
                    st.success(f"{icon} {result.item_name} (ID: {result.item_id[:8]}...)")
        
        # 失敗リスト
        if failed:
            with st.expander(f"❌ 削除失敗 ({len(failed)}件)", expanded=True):
                for result in failed:
                    icon = "📚" if result.item_type == "vector_store" else "📁"
                    st.error(f"{icon} {result.item_name}: {result.error}")
        
        return successful, failed

# ===================================================================
# メイン処理
# ===================================================================
def initialize_session_state():
    """セッション状態の初期化"""
    if 'deletion_history' not in st.session_state:
        st.session_state.deletion_history = []
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = None

def main():
    """メイン処理関数"""
    
    # セッション状態初期化
    initialize_session_state()
    
    # UI初期化
    ui = DeletionUI()
    ui.setup_page()
    ui.setup_header()
    
    # OpenAI SDK確認
    if not OPENAI_AVAILABLE:
        st.error("OpenAI SDKが利用できません。`pip install openai` でインストールしてください。")
        return
    
    # APIキー確認
    if not os.getenv("OPENAI_API_KEY"):
        st.error("🔑 OpenAI APIキーが設定されていません")
        st.code("export OPENAI_API_KEY='your-api-key-here'")
        st.info("APIキーを設定してからアプリを再起動してください")
        return
    
    # マネージャー初期化
    try:
        manager = OpenAIResourceManager()
        ui.manager = manager
    except Exception as e:
        st.error(f"マネージャーの初期化に失敗: {e}")
        return
    
    # サイドバー設定
    settings = ui.setup_sidebar()
    deletion_mode = settings["deletion_mode"]
    confirm_delete = settings["confirm_delete"]
    
    # メインコンテンツ
    tab1, tab2, tab3 = st.tabs(["📚 Vector Store削除", "📁 ファイル削除", "📊 削除履歴"])
    
    with tab1:
        st.header("📚 Vector Store削除")
        
        # リフレッシュボタン
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("🔄 一覧を更新", key="refresh_vs"):
                st.session_state.last_refresh = datetime.now()
                st.rerun()
        
        # Vector Store一覧取得
        stores = manager.list_vector_stores()
        
        if stores:
            st.info(f"合計 {len(stores)} 個のVector Storeが見つかりました")
            
            # モード別処理
            if deletion_mode == DeletionMode.INDIVIDUAL.value:
                # 個別選択モード
                selected_stores = ui.display_vector_stores(stores)
                
                if selected_stores:
                    st.markdown("---")
                    delete_with_files = st.checkbox("🔗 関連ファイルも同時に削除", value=False, help="Vector Storeに関連付けられたファイルも削除します")
                    if st.button(f"🗑️ 選択した{len(selected_stores)}個のVector Storeを削除", type="primary"):
                        with st.spinner("削除中..."):
                            vs_results, file_results = manager.batch_delete_vector_stores(selected_stores, delete_with_files)
                            all_results = vs_results + file_results
                            ui.display_deletion_results(all_results)
                            
                            # 履歴保存
                            if all_results:
                                filepath = manager.save_deletion_history(
                                    f"vs_deletion_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                                )
                                st.success(f"削除履歴を保存しました: {filepath}")
                
            elif deletion_mode == DeletionMode.BATCH.value:
                # 複数選択モード（INDIVIDUALと同じ）
                selected_stores = ui.display_vector_stores(stores)
                
                if selected_stores:
                    st.markdown("---")
                    delete_with_files = st.checkbox("🔗 関連ファイルも同時に削除", value=False, help="Vector Storeに関連付けられたファイルも削除します", key="batch_delete_files")
                    if st.button(f"🗑️ 選択した{len(selected_stores)}個のVector Storeを削除", type="primary"):
                        with st.spinner("削除中..."):
                            vs_results, file_results = manager.batch_delete_vector_stores(selected_stores, delete_with_files)
                            all_results = vs_results + file_results
                            ui.display_deletion_results(all_results)
                            
                            # 履歴保存
                            if all_results:
                                filepath = manager.save_deletion_history(
                                    f"vs_deletion_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                                )
                                st.success(f"削除履歴を保存しました: {filepath}")
                
            elif deletion_mode == DeletionMode.ALL.value:
                # 全削除モード
                st.error(f"⚠️ 全{len(stores)}個のVector Storeが削除対象です")
                
                # Vector Store名一覧表示
                with st.expander("削除対象一覧", expanded=False):
                    for store in stores:
                        st.write(f"- {store['name']} (ID: {store['id'][:8]}...)")
                
                if confirm_delete:
                    st.markdown("---")
                    # 二重確認
                    final_confirm = st.checkbox("最終確認: 本当に全て削除してもよろしいですか？")
                    
                    if final_confirm:
                        if st.button("🗑️ 全Vector Storeを削除", type="primary"):
                            with st.spinner("全削除中..."):
                                results = manager.delete_all_vector_stores()
                                ui.display_deletion_results(results)
                                
                                # 履歴保存
                                if results:
                                    filepath = manager.save_deletion_history(
                                        f"vs_deletion_history_ALL_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                                    )
                                    st.success(f"削除履歴を保存しました: {filepath}")
                else:
                    st.warning("全削除を実行するには、サイドバーで確認チェックボックスをオンにしてください")
        else:
            st.info("Vector Storeが見つかりません")
    
    with tab2:
        st.header("📁 ファイル削除")
        
        # リフレッシュボタン
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("🔄 一覧を更新", key="refresh_files"):
                st.session_state.last_refresh = datetime.now()
                st.rerun()
        
        # ファイル一覧取得
        files = manager.list_files()
        
        if files:
            st.info(f"合計 {len(files)} 個のファイルが見つかりました")
            
            # モード別処理
            if deletion_mode == DeletionMode.INDIVIDUAL.value:
                # 個別選択モード
                selected_files = ui.display_files(files)
                
                if selected_files:
                    st.markdown("---")
                    if st.button(f"🗑️ 選択した{len(selected_files)}個のファイルを削除", type="primary"):
                        with st.spinner("削除中..."):
                            results = manager.batch_delete_files(selected_files)
                            ui.display_deletion_results(results)
                            
                            # 履歴保存
                            if results:
                                filepath = manager.save_deletion_history(
                                    f"file_deletion_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                                )
                                st.success(f"削除履歴を保存しました: {filepath}")
                
            elif deletion_mode == DeletionMode.BATCH.value:
                # 複数選択モード（INDIVIDUALと同じ）
                selected_files = ui.display_files(files)
                
                if selected_files:
                    st.markdown("---")
                    if st.button(f"🗑️ 選択した{len(selected_files)}個のファイルを削除", type="primary"):
                        with st.spinner("削除中..."):
                            results = manager.batch_delete_files(selected_files)
                            ui.display_deletion_results(results)
                            
                            # 履歴保存
                            if results:
                                filepath = manager.save_deletion_history(
                                    f"file_deletion_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                                )
                                st.success(f"削除履歴を保存しました: {filepath}")
                
            elif deletion_mode == DeletionMode.ALL.value:
                # 全削除モード
                st.error(f"⚠️ 全{len(files)}個のファイルが削除対象です")
                
                # ファイル名一覧表示
                with st.expander("削除対象一覧", expanded=False):
                    for file in files:
                        st.write(f"- {file['filename']} (ID: {file['id'][:8]}...)")
                
                if confirm_delete:
                    st.markdown("---")
                    # 二重確認
                    final_confirm = st.checkbox("最終確認: 本当に全て削除してもよろしいですか？", key="final_confirm_files")
                    
                    if final_confirm:
                        if st.button("🗑️ 全ファイルを削除", type="primary"):
                            with st.spinner("全削除中..."):
                                results = manager.delete_all_files()
                                ui.display_deletion_results(results)
                                
                                # 履歴保存
                                if results:
                                    filepath = manager.save_deletion_history(
                                        f"file_deletion_history_ALL_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                                    )
                                    st.success(f"削除履歴を保存しました: {filepath}")
                else:
                    st.warning("全削除を実行するには、サイドバーで確認チェックボックスをオンにしてください")
        else:
            st.info("ファイルが見つかりません")
    
    with tab3:
        st.header("📊 削除履歴")
        
        # 現在のセッションの履歴
        if manager.deletion_history:
            st.subheader("現在のセッションの削除履歴")
            
            history_data = []
            for result in manager.deletion_history:
                history_data.append({
                    "時刻": result.timestamp.strftime("%H:%M:%S") if result.timestamp else "-",
                    "タイプ": "Vector Store" if result.item_type == "vector_store" else "ファイル",
                    "名前": result.item_name,
                    "ID": result.item_id[:8] + "...",
                    "状態": "✅ 成功" if result.success else f"❌ 失敗: {result.error}"
                })
            
            import pandas as pd
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True)
            
            # 履歴をJSONで保存
            if st.button("💾 履歴を保存"):
                filepath = manager.save_deletion_history(
                    f"deletion_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                st.success(f"削除履歴を保存しました: {filepath}")
        else:
            st.info("このセッションでの削除履歴はまだありません")
        
        # 過去の履歴ファイル表示
        st.markdown("---")
        st.subheader("保存済み履歴ファイル")
        
        output_dir = Path("OUTPUT")
        if output_dir.exists():
            history_files = list(output_dir.glob("*deletion_history*.json"))
            
            if history_files:
                for file in sorted(history_files, reverse=True):
                    with st.expander(f"📄 {file.name}", expanded=False):
                        try:
                            with open(file, 'r', encoding='utf-8') as f:
                                history = json.load(f)
                            
                            st.write(f"**記録数**: {len(history)}件")
                            
                            # 簡易表示
                            for item in history[:10]:  # 最初の10件のみ表示
                                status = "✅" if item['success'] else "❌"
                                st.write(f"{status} {item['item_type']}: {item['item_name']}")
                            
                            if len(history) > 10:
                                st.write(f"... 他 {len(history) - 10} 件")
                        except Exception as e:
                            st.error(f"ファイル読み込みエラー: {e}")
            else:
                st.info("保存済みの履歴ファイルはありません")
        else:
            st.info("OUTPUTディレクトリが存在しません")
    
    # フッター
    st.markdown("---")
    st.markdown("### 🗑️ Vector Store/Files削除ツール")
    st.markdown("⚠️ **注意**: 削除操作は取り消せません。慎重に実行してください。")

if __name__ == "__main__":
    main()

# 実行コマンド:
# streamlit run a04_delete_vector_store.py --server.port=8504