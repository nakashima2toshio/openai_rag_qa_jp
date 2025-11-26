# helper_rag.py
# RAGデータ前処理の共通機能
# -----------------------------------------

import streamlit as st
import pandas as pd
import re
import io
import logging
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from functools import wraps

# ===================================================================
# 基本ログ設定
# ===================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================================================================
# 共通モジュールからインポート
# ===================================================================
# テキスト処理関数をhelper_text.pyから参照可能にする
from helper_text import clean_text as _clean_text_impl
from helper_text import count_tokens, split_into_chunks

# 設定はconfig.pyから参照可能（後方互換性維持）
try:
    from config import ModelConfig
except ImportError:
    ModelConfig = None


# ==================================================
# 設定管理クラス（共通）
# ==================================================
class AppConfig:
    """アプリケーション設定（全アプリ共通）"""

    # 利用可能なモデル
    AVAILABLE_MODELS = [
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4o-audio-preview",
        "gpt-4o-mini-audio-preview",
        "gpt-4.1",
        "gpt-4.1-mini",
        "o1",
        "o1-mini",
        "o3",
        "o3-mini",
        "o4",
        "o4-mini"
    ]

    DEFAULT_MODEL = "gpt-5-mini"

    # モデル料金（1000トークンあたりのドル）
    MODEL_PRICING = {
        "gpt-5"                    : {"input": 0.01, "output": 0.03},
        "gpt-5-mini"               : {"input": 0.0001, "output": 0.0004},
        "gpt-5-nano"               : {"input": 0.00005, "output": 0.0002},
        "gpt-4o"                   : {"input": 0.005, "output": 0.015},
        "gpt-4o-mini"              : {"input": 0.00015, "output": 0.0006},
        "gpt-4o-audio-preview"     : {"input": 0.01, "output": 0.02},
        "gpt-4o-mini-audio-preview": {"input": 0.00025, "output": 0.001},
        "gpt-4.1"                  : {"input": 0.0025, "output": 0.01},
        "gpt-4.1-mini"             : {"input": 0.0001, "output": 0.0004},
        "o1"                       : {"input": 0.015, "output": 0.06},
        "o1-mini"                  : {"input": 0.003, "output": 0.012},
        "o3"                       : {"input": 0.03, "output": 0.12},
        "o3-mini"                  : {"input": 0.006, "output": 0.024},
        "o4"                       : {"input": 0.05, "output": 0.20},
        "o4-mini"                  : {"input": 0.01, "output": 0.04},
    }

    # モデル制限
    MODEL_LIMITS = {
        "gpt-5"                    : {"max_tokens": 256000, "max_output": 8192},
        "gpt-5-mini"               : {"max_tokens": 128000, "max_output": 4096},
        "gpt-5-nano"               : {"max_tokens": 64000, "max_output": 2048},
        "gpt-4o"                   : {"max_tokens": 128000, "max_output": 4096},
        "gpt-4o-mini"              : {"max_tokens": 128000, "max_output": 4096},
        "gpt-4o-audio-preview"     : {"max_tokens": 128000, "max_output": 4096},
        "gpt-4o-mini-audio-preview": {"max_tokens": 128000, "max_output": 4096},
        "gpt-4.1"                  : {"max_tokens": 128000, "max_output": 4096},
        "gpt-4.1-mini"             : {"max_tokens": 128000, "max_output": 4096},
        "o1"                       : {"max_tokens": 128000, "max_output": 32768},
        "o1-mini"                  : {"max_tokens": 128000, "max_output": 65536},
        "o3"                       : {"max_tokens": 200000, "max_output": 100000},
        "o3-mini"                  : {"max_tokens": 200000, "max_output": 100000},
        "o4"                       : {"max_tokens": 256000, "max_output": 128000},
        "o4-mini"                  : {"max_tokens": 256000, "max_output": 128000},
    }

    @classmethod
    def get_model_limits(cls, model: str) -> Dict[str, int]:
        """モデルの制限を取得"""
        return cls.MODEL_LIMITS.get(model, {"max_tokens": 128000, "max_output": 4096})

    @classmethod
    def get_model_pricing(cls, model: str) -> Dict[str, float]:
        """モデルの料金を取得"""
        return cls.MODEL_PRICING.get(model, {"input": 0.00015, "output": 0.0006})


# ==================================================
# RAG設定クラス（全データセット対応）
# ==================================================
class RAGConfig:
    """RAGデータ前処理の設定（全データセット統合）"""

    DATASET_CONFIGS = {
        # カスタマーサポートFAQ
        "customer_support_faq": {
            "name"            : "カスタマーサポート・FAQ",
            "icon"            : "💬",
            "required_columns": ["question", "answer"],
            "description"     : "カスタマーサポートFAQデータセット",
            "combine_template": "{question} {answer}",
            "port"            : 8501
        },

        # 医療QA
        "medical_qa"          : {
            "name"            : "医療QAデータ",
            "icon"            : "🏥",
            "required_columns": ["Question", "Complex_CoT", "Response"],
            "description"     : "医療質問回答データセット",
            "combine_template": "{question} {complex_cot} {response}",
            "port"            : 8503
        },

        # 科学・技術QA
        "sciq_qa"             : {
            "name"            : "科学・技術QA（SciQ）",
            "icon"            : "🔬",
            "required_columns": ["question", "correct_answer"],
            "description"     : "科学・技術質問回答データセット",
            "combine_template": "{question} {correct_answer}",
            "port"            : 8504
        },

        # 法律・判例QA
        "legal_qa"            : {
            "name"            : "法律・判例QA",
            "icon"            : "⚖️",
            "required_columns": ["question", "answer"],
            "description"     : "法律・判例質問回答データセット",
            "combine_template": "{question} {answer}",
            "port"            : 8505
        },
        
        # TriviaQA
        "trivia_qa"           : {
            "name"            : "雑学QA（TriviaQA）",
            "icon"            : "🎯",
            "required_columns": ["question", "answer"],
            "description"     : "雑学質問回答データセット",
            "combine_template": "{question} {answer} {entity_pages} {search_results}",
            "port"            : 8506
        }
    }

    @classmethod
    def get_config(cls, dataset_type: str) -> Dict[str, Any]:
        """データセット設定の取得"""
        return cls.DATASET_CONFIGS.get(dataset_type, {
            "name"            : "未知のデータセット",
            "icon"            : "❓",
            "required_columns": [],
            "description"     : "未知のデータセット",
            "combine_template": "{}",
            "port"            : 8500
        })

    @classmethod
    def get_all_datasets(cls) -> List[str]:
        """全データセットタイプのリストを取得"""
        return list(cls.DATASET_CONFIGS.keys())

    @classmethod
    def get_dataset_by_port(cls, port: int) -> Optional[str]:
        """ポート番号からデータセットタイプを取得"""
        for dataset_type, config in cls.DATASET_CONFIGS.items():
            if config.get("port") == port:
                return dataset_type
        return None


# ==================================================
# トークン管理クラス（共通）
# ==================================================
class TokenManager:
    """トークン数の管理（簡易版）"""

    @staticmethod
    def count_tokens(text: str, model: str = None) -> int:
        """テキストのトークン数をカウント（簡易推定）"""
        if not text:
            return 0

        # 簡易推定: 日本語文字は0.5トークン、英数字は0.25トークン
        japanese_chars = len([c for c in text if ord(c) > 127])
        english_chars = len(text) - japanese_chars
        estimated_tokens = int(japanese_chars * 0.5 + english_chars * 0.25)

        # 最低1トークンは必要
        return max(1, estimated_tokens)

    @staticmethod
    def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
        """API使用コストの推定"""
        pricing = AppConfig.get_model_pricing(model)
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost


# ==================================================
# エラーハンドリングデコレータ（共通）
# ==================================================
def safe_execute(func):
    """安全実行デコレータ"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            st.error(f"エラーが発生しました: {str(e)}")
            return None

    return wrapper


# ==================================================
# UI関数群（共通）
# ==================================================
def select_model(key: str = "model_selection") -> str:
    """モデル選択UI"""
    models = AppConfig.AVAILABLE_MODELS
    default_model = AppConfig.DEFAULT_MODEL

    try:
        default_index = models.index(default_model)
    except ValueError:
        default_index = 0

    selected = st.sidebar.selectbox(
        "🤖 モデルを選択",
        models,
        index=default_index,
        key=key,
        help="利用するOpenAIモデルを選択してください"
    )

    return selected


def show_model_info(selected_model: str) -> None:
    """選択されたモデルの情報を表示"""
    try:
        limits = AppConfig.get_model_limits(selected_model)
        pricing = AppConfig.get_model_pricing(selected_model)

        with st.sidebar.expander("📊 選択モデル情報", expanded=False):
            # 基本情報
            col1, col2 = st.columns(2)
            with col1:
                st.write("**最大入力**")
                st.write(f"{limits['max_tokens']:,}")
            with col2:
                st.write("**最大出力**")
                st.write(f"{limits['max_output']:,}")

            # 料金情報
            st.write("**料金（1000トークン）**")
            st.write(f"- 入力: ${pricing['input']:.5f}")
            st.write(f"- 出力: ${pricing['output']:.5f}")

            # モデル特性
            if selected_model.startswith("o"):
                st.info("🧠 推論特化モデル")
                st.caption("高度な推論タスクに最適化")
            elif "audio" in selected_model:
                st.info("🎵 音声対応モデル")
                st.caption("音声入力・出力に対応")
            elif "gpt-4o" in selected_model:
                st.info("👁️ マルチモーダルモデル")
                st.caption("テキスト・画像の理解が可能")
            else:
                st.info("💬 標準対話モデル")
                st.caption("一般的な対話・テキスト処理")

            # RAG用途での推奨度
            st.write("**RAG用途推奨度**")
            if selected_model in ["gpt-4o-mini", "gpt-4.1-mini"]:
                st.success("✅ 最適（コスト効率良好）")
            elif selected_model in ["gpt-4o", "gpt-4.1"]:
                st.info("💡 高品質（コスト高）")
            elif selected_model.startswith("o"):
                st.warning("⚠️ 推論特化（RAG用途には過剰）")
            else:
                st.info("💬 標準的な性能")

    except Exception as e:
        logger.error(f"モデル情報表示エラー: {e}")
        st.sidebar.error("モデル情報の取得に失敗しました")


def estimate_token_usage(df_processed: pd.DataFrame, selected_model: str) -> None:
    """処理済みデータのトークン使用量推定"""
    try:
        if 'Combined_Text' in df_processed.columns:
            # サンプルテキストでトークン数を推定
            sample_size = min(10, len(df_processed))
            sample_texts = df_processed['Combined_Text'].head(sample_size).tolist()
            total_chars = df_processed['Combined_Text'].str.len().sum()

            if sample_texts:
                sample_text = " ".join(sample_texts)
                sample_tokens = TokenManager.count_tokens(sample_text, selected_model)
                sample_chars = len(sample_text)

                if sample_chars > 0:
                    # 全体のトークン数を推定
                    estimated_total_tokens = int((total_chars / sample_chars) * sample_tokens)

                    with st.expander("🔢 トークン使用量推定", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("推定総トークン数", f"{estimated_total_tokens:,}")
                        with col2:
                            avg_tokens_per_record = estimated_total_tokens / len(df_processed)
                            st.metric("平均トークン/レコード", f"{avg_tokens_per_record:.0f}")
                        with col3:
                            # embedding用のコスト推定（参考値）
                            embedding_cost = (estimated_total_tokens / 1000) * 0.0001
                            st.metric("推定embedding費用", f"${embedding_cost:.4f}")

                        st.info(f"💡 選択モデル「{selected_model}」での推定値")
                        st.caption("※ 実際のトークン数とは異なる場合があります")

    except Exception as e:
        logger.error(f"トークン使用量推定エラー: {e}")
        st.error("トークン使用量の推定に失敗しました")


# ==================================================
# データ処理関数群（共通）
# ==================================================
def clean_text(text: str) -> str:
    """テキストのクレンジング処理"""
    if pd.isna(text) or text == "":
        return ""

    # 文字列に変換
    text = str(text)

    # 改行を空白に置換
    text = text.replace('\n', ' ').replace('\r', ' ')

    # 連続した空白を1つの空白にまとめる
    text = re.sub(r'\s+', ' ', text)

    # 先頭・末尾の空白を除去
    text = text.strip()

    # 引用符の正規化
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")

    return text


def combine_columns(row: pd.Series, dataset_type: str) -> str:
    """複数列を結合して1つのテキストにする（データセット対応）"""
    config_data = RAGConfig.get_config(dataset_type)
    required_columns = config_data["required_columns"]

    # 各列からテキストを抽出・クレンジング
    cleaned_values = []
    for col in required_columns:
        if col in row.index:
            value = row.get(col, '')
            cleaned_text = clean_text(str(value))
            if cleaned_text:  # 空でない場合のみ追加
                cleaned_values.append(cleaned_text)

    # 医療QAの特別処理（Question, Complex_CoT, Response）
    if dataset_type == "medical_qa":
        # 大文字小文字を考慮した列名マッピング
        medical_cols = {}
        for col in row.index:
            col_lower = col.lower()
            if 'question' in col_lower:
                medical_cols['question'] = clean_text(str(row.get(col, '')))
            elif 'complex_cot' in col_lower or 'cot' in col_lower:
                medical_cols['complex_cot'] = clean_text(str(row.get(col, '')))
            elif 'response' in col_lower:
                medical_cols['response'] = clean_text(str(row.get(col, '')))

        # 医療QA用の結合
        medical_values = [v for v in medical_cols.values() if v]
        if medical_values:
            return " ".join(medical_values).strip()

    # 結合
    combined = " ".join(cleaned_values)
    return combined.strip()


def validate_data(df: pd.DataFrame, dataset_type: str = None) -> List[str]:
    """データの検証"""
    issues = []

    # 基本統計
    issues.append(f"総行数: {len(df):,}")
    issues.append(f"総列数: {len(df.columns)}")

    # 必須列の確認
    if dataset_type:
        config_data = RAGConfig.get_config(dataset_type)
        required_columns = config_data["required_columns"]

        # 大文字小文字を考慮した列名チェック
        available_columns = [col.lower() for col in df.columns]
        missing_columns = []
        found_columns = []

        for req_col in required_columns:
            req_col_lower = req_col.lower()
            # 部分一致も許可（例：Question -> question, Complex_CoT -> complex_cot）
            found = False
            for available_col in df.columns:
                if req_col_lower in available_col.lower() or available_col.lower() in req_col_lower:
                    found_columns.append(available_col)
                    found = True
                    break
            if not found:
                missing_columns.append(req_col)

        if missing_columns:
            issues.append(f"⚠️ 必須列が不足: {missing_columns}")
        else:
            issues.append(f"✅ 必須列確認済み: {found_columns}")

    # 各列の空値確認
    for col in df.columns:
        empty_count = df[col].isna().sum() + (df[col] == '').sum()
        if empty_count > 0:
            percentage = (empty_count / len(df)) * 100
            issues.append(f"{col}列: 空値 {empty_count:,}個 ({percentage:.1f}%)")

    # 重複行の確認
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"⚠️ 重複行: {duplicate_count:,}個")
    else:
        issues.append("✅ 重複行なし")

    return issues


@safe_execute
def load_dataset(uploaded_file, dataset_type: str = None) -> Tuple[pd.DataFrame, List[str]]:
    """データセットの読み込みと基本検証"""
    # CSVファイルの読み込み
    df = pd.read_csv(uploaded_file)

    # 基本検証
    validation_results = validate_data(df, dataset_type)

    logger.info(f"データセット読み込み完了: {len(df):,}行, {len(df.columns)}列")
    return df, validation_results


@safe_execute
def process_rag_data(df: pd.DataFrame, dataset_type: str, combine_columns_option: bool = True) -> pd.DataFrame:
    """RAGデータの前処理を実行"""
    # 基本的な前処理
    df_processed = df.copy()

    # 重複行の除去
    initial_rows = len(df_processed)
    df_processed = df_processed.drop_duplicates()
    duplicates_removed = initial_rows - len(df_processed)

    # 空行の除去（全列がNAの行）
    df_processed = df_processed.dropna(how='all')
    empty_rows_removed = initial_rows - duplicates_removed - len(df_processed)

    # インデックスのリセット
    df_processed = df_processed.reset_index(drop=True)

    logger.info(f"前処理完了: 重複除去={duplicates_removed:,}行, 空行除去={empty_rows_removed:,}行")

    # 各列のクレンジング（データセット対応）
    config_data = RAGConfig.get_config(dataset_type)
    required_columns = config_data["required_columns"]

    # 大文字小文字を考慮した列名処理
    for col in df_processed.columns:
        for req_col in required_columns:
            if req_col.lower() in col.lower() or col.lower() in req_col.lower():
                df_processed[col] = df_processed[col].apply(clean_text)

    # 列の結合（オプション）
    if combine_columns_option:
        df_processed['Combined_Text'] = df_processed.apply(
            lambda row: combine_columns(row, dataset_type),
            axis=1
        )

        # 空の結合テキストを除去
        before_filter = len(df_processed)
        df_processed = df_processed[df_processed['Combined_Text'].str.strip() != '']
        after_filter = len(df_processed)
        empty_combined_removed = before_filter - after_filter

        if empty_combined_removed > 0:
            logger.info(f"空の結合テキストを除去: {empty_combined_removed:,}行")

    return df_processed


@safe_execute
def create_download_data(df: pd.DataFrame, include_combined: bool = True, dataset_type: str = None) -> Tuple[
    str, Optional[str]]:
    """ダウンロード用データの作成"""
    # CSVデータの作成
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_data = csv_buffer.getvalue()

    # 結合テキストデータの作成
    text_data = None
    if include_combined and 'Combined_Text' in df.columns:
        # インデックスなしで結合テキストのみを出力
        text_lines = []
        for text in df['Combined_Text']:
            if text and str(text).strip():
                text_lines.append(str(text).strip())
        text_data = '\n'.join(text_lines)

    return csv_data, text_data


def display_statistics(df_original: pd.DataFrame, df_processed: pd.DataFrame, dataset_type: str = None) -> None:
    """処理前後の統計情報を表示"""
    st.subheader("📊 統計情報")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("元の行数", f"{len(df_original):,}")
    with col2:
        st.metric("処理後の行数", f"{len(df_processed):,}")
    with col3:
        removed_rows = len(df_original) - len(df_processed)
        st.metric("除去された行数", f"{removed_rows:,}")

    # 結合テキストの分析
    if 'Combined_Text' in df_processed.columns:
        st.subheader("📝 結合後テキスト分析")
        text_lengths = df_processed['Combined_Text'].str.len()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("平均文字数", f"{text_lengths.mean():.0f}")
        with col2:
            st.metric("最大文字数", f"{text_lengths.max():,}")
        with col3:
            st.metric("最小文字数", f"{text_lengths.min():,}")

        # パーセンタイル表示
        percentiles = text_lengths.quantile([0.25, 0.5, 0.75])
        st.write("**文字数分布:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"25%点: {percentiles[0.25]:.0f}文字")
        with col2:
            st.write(f"中央値: {percentiles[0.5]:.0f}文字")
        with col3:
            st.write(f"75%点: {percentiles[0.75]:.0f}文字")


# ==================================================
# ファイル保存関数群（共通）
# ==================================================
def create_output_directory() -> Path:
    """OUTPUTディレクトリの作成"""
    try:
        output_dir = Path("OUTPUT")
        output_dir.mkdir(exist_ok=True)

        # 書き込み権限のテスト
        test_file = output_dir / ".test_write"
        try:
            test_file.write_text("test", encoding='utf-8')
            if test_file.exists():
                test_file.unlink()
                logger.info("書き込み権限テスト: 成功")
        except Exception as e:
            raise PermissionError(f"書き込み権限テストに失敗: {e}")

        logger.info(f"OUTPUTディレクトリ準備完了: {output_dir.absolute()}")
        return output_dir

    except Exception as e:
        logger.error(f"ディレクトリ作成エラー: {e}")
        raise


@safe_execute
def save_files_to_output(df_processed, dataset_type: str, csv_data: str, text_data: str = None) -> Dict[str, str]:
    """処理済みデータをOUTPUTフォルダに保存"""
    output_dir = create_output_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {}

    # CSVファイルの保存
    csv_filename = f"preprocessed_{dataset_type}.csv"
    csv_path = output_dir / csv_filename

    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(csv_data)

    if csv_path.exists():
        saved_files['csv'] = str(csv_path)
        logger.info(f"CSVファイル保存完了: {csv_path}")

    # テキストファイルの保存
    if text_data and len(text_data.strip()) > 0:
        txt_filename = f"{dataset_type}.txt"
        txt_path = output_dir / txt_filename

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text_data)

        if txt_path.exists():
            saved_files['txt'] = str(txt_path)
            logger.info(f"テキストファイル保存完了: {txt_path}")

    # メタデータファイルの保存
    metadata = {
        "dataset_type"        : dataset_type,
        "processed_rows"      : len(df_processed),
        "processing_timestamp": timestamp,
        "created_at"          : datetime.now().isoformat(),
        "files_created"       : list(saved_files.keys()),
        "processing_info"     : {
            "original_rows": st.session_state.get('original_rows', 0),
            "removed_rows" : st.session_state.get('original_rows', 0) - len(df_processed)
        }
    }

    metadata_filename = f"metadata_{dataset_type}.json"
    metadata_path = output_dir / metadata_filename

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if metadata_path.exists():
        saved_files['metadata'] = str(metadata_path)
        logger.info(f"メタデータファイル保存完了: {metadata_path}")

    return saved_files


# ==================================================
# 使用方法説明関数（データセット対応）
# ==================================================
def show_usage_instructions(dataset_type: str) -> None:
    """使用方法の説明を表示（データセット別対応）"""
    config_data = RAGConfig.get_config(dataset_type)
    required_columns_str = ", ".join(config_data["required_columns"])

    st.markdown("---")
    st.subheader("📖 使用方法")

    # 基本的な使用方法
    basic_usage = f"""
    ### 📋 前処理手順
    1. **モデル選択**: サイドバーでRAG用途に適したモデルを選択
    2. **CSVファイルアップロード**: {required_columns_str} 列を含むCSVファイルを選択
    3. **前処理実行**: 以下の処理が自動で実行されます：
       - 改行・空白の正規化
       - 重複行の除去
       - 空行の除去
       - 引用符の正規化
    4. **列結合**: Vector Store/RAG用に最適化された自然な文章として結合
    5. **トークン使用量確認**: 選択モデルでのトークン数とコストを推定
    6. **ダウンロード**: 前処理済みデータを各種形式でダウンロード

    ### 🎯 RAG最適化の特徴
    - **自然な文章結合**: ラベルなしで読みやすい文章として結合
    - **OpenAI embedding対応**: text-embedding-ada-002等に最適化
    - **検索性能向上**: 意味的検索の精度向上

    ### 💡 推奨モデル
    - **コスト重視**: gpt-4o-mini, gpt-4.1-mini
    - **品質重視**: gpt-4o, gpt-4.1
    - **推論タスク**: o1-mini, o3-mini（RAG用途には過剰）
    """

    # データセット特有の説明
    dataset_specific = ""
    if dataset_type == "customer_support_faq":
        dataset_specific = """
    ### 💬 カスタマーサポートFAQの特徴
    - **FAQ形式**: 質問と回答のペアによる構造
    - **実用的な内容**: 実際の顧客からの質問に基づく
    - **簡潔な回答**: 分かりやすく実用的な回答
        """
    elif dataset_type == "medical_qa":
        dataset_specific = """
    ### 🏥 医療QAデータの特徴
    - **複雑な推論**: Complex_CoT列による段階的推論過程
    - **専門用語**: 医療専門用語の適切な処理
    - **詳細な回答**: 医療情報に特化した包括的な回答
        """
    elif dataset_type == "sciq_qa":
        dataset_specific = """
    ### 🔬 SciQデータの特徴
    - **科学・技術問題**: 化学、物理、生物学、数学などの分野をカバー
    - **多肢選択形式**: distractor列による選択肢問題
    - **補足説明**: support列による詳細な背景情報
    - **幅広い難易度**: 基礎から応用まで様々なレベルの科学問題
        """
    elif dataset_type == "legal_qa":
        dataset_specific = """
    ### ⚖️ 法律・判例QAデータの特徴
    - **法律専門用語**: 条文、判例、法的概念の適切な処理
    - **詳細な回答**: 法的根拠を含む包括的な説明
    - **正確性重視**: 法的情報の正確性を保持した前処理
    - **引用・参照**: 条文番号や判例番号などの法的根拠の保護
        """

    st.markdown(basic_usage + dataset_specific)


# ==================================================
# ページ設定関数（共通）
# ==================================================
def setup_page_config(dataset_type: str) -> None:
    """ページ設定の初期化"""
    config_data = RAGConfig.get_config(dataset_type)

    try:
        st.set_page_config(
            page_title=f"{config_data['name']}前処理（完全独立版）",
            page_icon=config_data['icon'],
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except st.errors.StreamlitAPIException:
        pass


def setup_page_header(dataset_type: str) -> None:
    """ページヘッダーの設定"""
    config_data = RAGConfig.get_config(dataset_type)

    st.title(f"{config_data['icon']} {config_data['name']}前処理アプリ")
    st.caption("RAG（Retrieval-Augmented Generation）用データ前処理 - 完全独立版")
    st.markdown("---")


def setup_sidebar_header(dataset_type: str) -> None:
    """サイドバーヘッダーの設定"""
    config_data = RAGConfig.get_config(dataset_type)

    st.sidebar.title(f"{config_data['icon']} {config_data['name']}")
    st.sidebar.markdown("---")


# ==================================================
# エクスポート（共通関数一覧）
# ==================================================
__all__ = [
    # 設定クラス
    'AppConfig',
    'RAGConfig',
    'TokenManager',

    # デコレータ
    'safe_execute',

    # UI関数
    'select_model',
    'show_model_info',
    'estimate_token_usage',

    # データ処理関数
    'clean_text',
    'combine_columns',
    'validate_data',
    'load_dataset',
    'process_rag_data',
    'create_download_data',
    'display_statistics',

    # ファイル保存関数
    'create_output_directory',
    'save_files_to_output',

    # 使用方法・ページ設定関数
    'show_usage_instructions',
    'setup_page_config',
    'setup_page_header',
    'setup_sidebar_header',
]
