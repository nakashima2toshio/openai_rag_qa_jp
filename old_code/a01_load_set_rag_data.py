#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
a01_load_set_rag_data.py - 統合RAGデータ処理ツール
===============================================
起動: streamlit run a01_load_set_rag_data.py --server.port=8501

【主要機能】
✅ 5種類のデータセットタイプを統合処理
   - カスタマーサポート・FAQ
   - 医療QAデータ
   - 科学・技術QA（SciQ）
   - 法律・判例QA
   - 雑学QA（TriviaQA）
✅ データ検証・品質チェック
✅ RAG用テキスト結合・前処理
✅ トークン使用量推定
✅ CSV/TXT/JSONフォーマット出力

【対応データセット】
1. customer_support_faq: カスタマーサポート向けFAQデータ
2. medical_qa: 医療分野のQ&Aデータ（推論過程付き）
3. sciq_qa: 科学・技術分野のQ&Aデータ（選択肢付き）
4. legal_qa: 法律・判例に関するQ&Aデータ
5. trivia_qa: 雑学のQA
"""

import streamlit as st
import pandas as pd
import json
import io
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any

# ローカルモジュール
from helper_rag import (
    setup_page_config,
    setup_page_header,
    setup_sidebar_header,
    select_model,
    show_model_info,
    validate_data,
    load_dataset,
    process_rag_data,
    estimate_token_usage,
    create_download_data,
    display_statistics,
    save_files_to_output,
    show_usage_instructions,
    RAGConfig
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================================================================
# データセット別検証関数
# ===================================================================

def validate_customer_support_data_specific(df: pd.DataFrame) -> List[str]:
    """カスタマーサポートデータ固有の検証"""
    issues = []
    
    # 必須列の検証
    if 'question' in df.columns and 'answer' in df.columns:
        # サポート関連キーワードの確認
        support_keywords = ['問題', '解決', 'トラブル', 'エラー', 'help', 'support']
        has_support_content = df['question'].str.contains('|'.join(support_keywords), na=False, case=False).any()
        if not has_support_content:
            issues.append("⚠️ サポート関連のキーワードが見つかりません")
        
        # 回答の長さチェック
        avg_answer_length = df['answer'].str.len().mean()
        if avg_answer_length < 50:
            issues.append(f"⚠️ 回答の平均文字数が短すぎます（{avg_answer_length:.0f}文字）")
        
        # 質問タイプの分析
        question_types = df['question'].str.contains('？|\\?', na=False).sum()
        st.info(f"💡 疑問形式の質問: {question_types}/{len(df)} ({question_types/len(df)*100:.1f}%)")
    
    return issues

def validate_medical_data_specific(df: pd.DataFrame) -> List[str]:
    """医療QAデータ固有の検証"""
    issues = []
    
    if 'Question' in df.columns and 'Response' in df.columns:
        # 医療用語のチェック
        medical_keywords = ['症状', '診断', '治療', '薬', '病気', 'disease', 'treatment', 'symptom']
        has_medical_content = df['Question'].str.contains('|'.join(medical_keywords), na=False, case=False).any()
        if not has_medical_content:
            issues.append("⚠️ 医療関連のキーワードが見つかりません")
        
        # Complex_CoTの存在確認
        if 'Complex_CoT' in df.columns:
            cot_non_empty = df['Complex_CoT'].notna().sum()
            st.info(f"💡 推論過程(Complex_CoT)付きデータ: {cot_non_empty}/{len(df)} ({cot_non_empty/len(df)*100:.1f}%)")
        
        # 回答の詳細度チェック
        avg_response_length = df['Response'].str.len().mean()
        if avg_response_length < 100:
            issues.append(f"⚠️ 回答の平均文字数が短すぎます（{avg_response_length:.0f}文字）")
    
    return issues

def validate_sciq_data_specific(df: pd.DataFrame) -> List[str]:
    """科学・技術QAデータ固有の検証"""
    issues = []
    
    if 'question' in df.columns and 'correct_answer' in df.columns:
        # 科学用語のチェック
        science_keywords = ['化学', '物理', '生物', '数学', 'science', 'physics', 'chemistry', 'biology']
        has_science_content = df['question'].str.contains('|'.join(science_keywords), na=False, case=False).any()
        if not has_science_content:
            issues.append("⚠️ 科学関連のキーワードが見つかりません")
        
        # 選択肢の存在確認
        distractor_cols = ['distractor1', 'distractor2', 'distractor3']
        available_distractors = [col for col in distractor_cols if col in df.columns]
        if available_distractors:
            st.info(f"💡 選択肢データあり: {len(available_distractors)}個の誤答選択肢")
        
        # サポートテキストの確認
        if 'support' in df.columns:
            support_non_empty = df['support'].notna().sum()
            st.info(f"💡 補足説明付きデータ: {support_non_empty}/{len(df)} ({support_non_empty/len(df)*100:.1f}%)")
    
    return issues

def validate_legal_data_specific(df: pd.DataFrame) -> List[str]:
    """法律QAデータ固有の検証"""
    issues = []
    
    if 'question' in df.columns and 'answer' in df.columns:
        # 法律用語のチェック
        legal_keywords = ['法律', '条文', '判例', '契約', 'law', 'legal', 'contract', 'regulation']
        has_legal_content = df['question'].str.contains('|'.join(legal_keywords), na=False, case=False).any()
        if not has_legal_content:
            issues.append("⚠️ 法律関連のキーワードが見つかりません")
        
        # 法的参照の確認
        has_references = df['answer'].str.contains('条|法|規則|判例', na=False).any()
        if has_references:
            ref_count = df['answer'].str.contains('条|法|規則|判例', na=False).sum()
            st.info(f"💡 法的参照を含むデータ: {ref_count}/{len(df)} ({ref_count/len(df)*100:.1f}%)")
        
        # 回答の分類
        answer_lengths = df['answer'].str.len()
        categories = pd.cut(answer_lengths, bins=[0, 50, 100, 200, 500, float('inf')], 
                          labels=['超短文', '短文', '中文', '長文', '超長文'])
        st.info(f"💡 回答の長さ分布: {categories.value_counts().to_dict()}")
    
    return issues

def validate_trivia_data_specific(df: pd.DataFrame) -> List[str]:
    """雑学QAデータ固有の検証"""
    issues = []
    
    if 'question' in df.columns and 'answer' in df.columns:
        # 雑学的なキーワードの確認（幅広いトピック）
        trivia_keywords = ['誰', '何', 'どこ', 'いつ', 'なぜ', 'どの', 'who', 'what', 'where', 'when', 'why', 'which']
        has_trivia_content = df['question'].str.contains('|'.join(trivia_keywords), na=False, case=False).any()
        if not has_trivia_content:
            issues.append("⚠️ 一般的な質問形式のキーワードが見つかりません")
        
        # 回答の簡潔性チェック
        avg_answer_length = df['answer'].str.len().mean()
        if avg_answer_length > 500:
            issues.append(f"⚠️ 回答が長すぎる可能性があります（平均{avg_answer_length:.0f}文字）")
        
        # エンティティページの存在確認
        if 'entity_pages' in df.columns:
            entity_non_empty = df['entity_pages'].notna().sum()
            st.info(f"💡 エンティティページ付きデータ: {entity_non_empty}/{len(df)} ({entity_non_empty/len(df)*100:.1f}%)")
        
        # 検索結果の存在確認
        if 'search_results' in df.columns:
            search_non_empty = df['search_results'].notna().sum()
            st.info(f"💡 検索結果（コンテキスト）付きデータ: {search_non_empty}/{len(df)} ({search_non_empty/len(df)*100:.1f}%)")
        
        # 質問の多様性チェック
        question_starts = df['question'].str[:10]  # 最初の10文字
        unique_starts = question_starts.nunique()
        diversity_ratio = unique_starts / len(df) * 100
        if diversity_ratio < 50:
            st.warning(f"⚠️ 質問の多様性が低い可能性があります（{diversity_ratio:.1f}%）")
        else:
            st.info(f"💡 質問の多様性: {diversity_ratio:.1f}%")
    
    return issues

# ===================================================================
# メイン処理
# ===================================================================

def main():
    """メイン処理関数"""
    
    # ページ設定（デフォルトのデータセットタイプで初期化）
    setup_page_config("customer_support_faq")
    
    # サイドバー設定
    with st.sidebar:
        setup_sidebar_header("customer_support_faq")
        
        # データセットタイプ選択
        st.subheader("📊 データセットタイプ選択")
        
        dataset_options = RAGConfig.get_all_datasets()
        dataset_labels = {
            dt: f"{RAGConfig.get_config(dt)['icon']} {RAGConfig.get_config(dt)['name']}"
            for dt in dataset_options
        }
        
        selected_dataset = st.selectbox(
            "処理するデータセットタイプ",
            options=dataset_options,
            format_func=lambda x: dataset_labels[x],
            help="処理したいデータセットのタイプを選択してください"
        )
        
        # データセット設定を取得
        dataset_config = RAGConfig.get_config(selected_dataset)
        
        # データセット情報表示
        st.info(f"""
        **選択中のデータセット:**
        - タイプ: {dataset_config['name']}
        - 必須列: {', '.join(dataset_config['required_columns'])}
        """)
        
        # モデル選択
        st.divider()
        selected_model = select_model()
        show_model_info(selected_model)
        
        # データセット固有のオプション
        st.divider()
        st.subheader("⚙️ データセット固有設定")
        
        dataset_specific_options = {}
        
        if selected_dataset == "customer_support_faq":
            dataset_specific_options['preserve_formatting'] = st.checkbox(
                "フォーマットを保持", 
                value=True,
                help="改行や箇条書きなどのフォーマットを保持します"
            )
            dataset_specific_options['normalize_questions'] = st.checkbox(
                "質問を正規化",
                value=False,
                help="質問文を疑問形に統一します"
            )
            
        elif selected_dataset == "medical_qa":
            dataset_specific_options['preserve_medical_terms'] = st.checkbox(
                "医学用語を保持",
                value=True,
                help="医学専門用語をそのまま保持します"
            )
            if 'Complex_CoT' in st.session_state.get('uploaded_columns', []):
                dataset_specific_options['include_cot'] = st.checkbox(
                    "Complex_CoTを含める",
                    value=True,
                    help="推論過程（Complex_CoT）を含めます"
                )
                
        elif selected_dataset == "sciq_qa":
            dataset_specific_options['include_distractors'] = st.checkbox(
                "誤答選択肢を含める",
                value=False,
                help="選択肢問題の誤答も含めます"
            )
            dataset_specific_options['include_support'] = st.checkbox(
                "サポートテキストを含める",
                value=True,
                help="補足説明テキストを含めます"
            )
            dataset_specific_options['preserve_scientific_notation'] = st.checkbox(
                "科学的記法を保持",
                value=True,
                help="数式や化学式などの記法を保持します"
            )
            
        elif selected_dataset == "legal_qa":
            dataset_specific_options['preserve_legal_terms'] = st.checkbox(
                "法律用語を保持",
                value=True,
                help="法律専門用語をそのまま保持します"
            )
            dataset_specific_options['preserve_references'] = st.checkbox(
                "条文参照を保持",
                value=True,
                help="法令や判例への参照を保持します"
            )
            dataset_specific_options['normalize_case_names'] = st.checkbox(
                "事件名を正規化",
                value=False,
                help="判例の事件名を統一形式に正規化します"
            )
            
        elif selected_dataset == "trivia_qa":
            dataset_specific_options['include_entity_pages'] = st.checkbox(
                "エンティティページを含める",
                value=True,
                help="Wikipediaなどのエンティティページ情報を含めます"
            )
            dataset_specific_options['include_search_results'] = st.checkbox(
                "検索結果を含める",
                value=True,
                help="検索結果（コンテキスト）を含めます"
            )
            dataset_specific_options['preserve_formatting'] = st.checkbox(
                "フォーマットを保持",
                value=True,
                help="テキストのフォーマットを保持します"
            )
    
    # メインコンテンツ
    setup_page_header(selected_dataset)
    
    # 使い方例をExpanderで表示
    with st.expander("📖 **使い方例**", expanded=False):
        st.markdown("""
        ### 🎯 基本的な使い方
        
        1. **左ペインで設定**
           - 📊 **データセットタイプを選択** (カスタマーサポート、医療QA、科学QA、法律QA)
           - 🤖 使用するモデルを選択
           - ⚙️ データセット固有の設定を調整
        
        2. **右ペインで処理実行**
        
           **📁 データアップロード**
           - CSVファイルをアップロード、または
           - HuggingFaceからデータセットを自動ダウンロード
             - データセット名を入力（例: `sciq`）
             - Split名とサンプル数を指定
             - 「📥 HuggingFaceからロード」をクリック
           
           **🔍 データ検証**
           - データの品質をチェック
           - 必須列の確認
           - データセット固有の検証結果を確認
           
           **⚙️ 前処理実行**
           - 結合するカラムを選択
           - セパレータを選択
           - 「🚀 前処理を実行」をクリック
           
           **📊 結果・ダウンロード**
           - 処理済みデータをCSV、TXT、JSON形式でダウンロード
           - OUTPUTフォルダに保存
        
        ### 💡 ヒント
        - HuggingFaceからダウンロードしたデータは自動的に`datasets/`フォルダに保存されます
        - メタデータも同時に保存され、後で処理の履歴を確認できます
        """)
    
    # タブ設定
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 データアップロード",
        "🔍 データ検証",
        "⚙️ 前処理実行",
        "📊 結果・ダウンロード"
    ])
    
    # Tab 1: データアップロード
    with tab1:
        st.header("データファイルのアップロード")
        
        # ファイルアップロード
        uploaded_file = st.file_uploader(
            f"{dataset_config['name']}のCSVファイルを選択",
            type=['csv'],
            help=f"必須列: {', '.join(dataset_config['required_columns'])}"
        )
        
        # または、データセット名を入力してHuggingFaceから自動ロード
        st.divider()
        st.subheader("または、HuggingFaceから自動ロード")
        
        # デフォルトのデータセット名を設定
        default_datasets = {
            "customer_support_faq": "MakTek/Customer_support_faqs_dataset",
            "medical_qa": "FreedomIntelligence/medical-o1-reasoning-SFT",
            "sciq_qa": "sciq",
            "legal_qa": "nguha/legalbench",
            "trivia_qa": "trivia_qa"  # TriviaQA用のデフォルト追加
        }
        
        dataset_name = st.text_input(
            "HuggingFaceデータセット名",
            value=default_datasets.get(selected_dataset, ""),
            placeholder="例: sciq"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            split_name = st.text_input("Split名", value="train")
        with col2:
            sample_size = st.number_input("サンプル数", min_value=10, value=1000)
        
        if st.button("📥 HuggingFaceからロード", type="primary"):
            # HuggingFaceからデータをロード
            try:
                from datasets import load_dataset as hf_load_dataset
                
                # データセット固有のconfig設定
                config_mapping = {
                    "medical_qa": "en",  # 医療データセット用
                    "legal_qa": "consumer_contracts_qa",  # 法律データセット用
                    "trivia_qa": "rc"  # TriviaQA用 (Reading Comprehension)
                }
                
                # configパラメータの決定（nameパラメータとして渡す）
                config_param = config_mapping.get(selected_dataset, None)
                
                # データセットをロード
                with st.spinner(f"HuggingFaceから{dataset_name}をダウンロード中..."):
                    if selected_dataset == "trivia_qa":
                        # TriviaQA用の特別な処理
                        dataset = hf_load_dataset(dataset_name, config_param, split=split_name)
                        # データを辞書形式からDataFrameに変換
                        data_list = []
                        for i, item in enumerate(dataset):
                            if i >= sample_size:
                                break
                            # TriviaQAのデータ構造に応じて必要なフィールドを抽出
                            record = {
                                'question_id': item.get('question_id', ''),
                                'question': item.get('question', ''),
                                'answer': '',  # answerは辞書型なので後で処理
                                'entity_pages': '',  # entity_pagesは辞書型なので後で処理
                                'search_results': ''  # search_resultsは辞書型なので後で処理
                            }
                            
                            # answerフィールドの処理
                            if 'answer' in item and item['answer']:
                                answer_data = item['answer']
                                if isinstance(answer_data, dict):
                                    # 正規化された値を優先
                                    if 'normalized_value' in answer_data and answer_data['normalized_value']:
                                        record['answer'] = answer_data['normalized_value']
                                    elif 'value' in answer_data:
                                        record['answer'] = answer_data['value']
                                    elif 'aliases' in answer_data and answer_data['aliases']:
                                        record['answer'] = answer_data['aliases'][0] if isinstance(answer_data['aliases'], list) else str(answer_data['aliases'])
                                else:
                                    record['answer'] = str(answer_data)
                            
                            # entity_pagesフィールドの処理
                            if 'entity_pages' in item and item['entity_pages']:
                                pages = item['entity_pages']
                                if isinstance(pages, dict):
                                    # 辞書の値からタイトルを抽出
                                    titles = []
                                    for key, page in list(pages.items())[:3]:  # 最初の3つ
                                        if isinstance(page, dict) and 'title' in page:
                                            titles.append(page['title'])
                                        elif isinstance(page, str):
                                            titles.append(page)
                                    record['entity_pages'] = ' | '.join(titles)
                                elif isinstance(pages, list):
                                    # リストの場合
                                    titles = []
                                    for page in pages[:3]:
                                        if isinstance(page, dict) and 'title' in page:
                                            titles.append(page['title'])
                                        elif isinstance(page, str):
                                            titles.append(page)
                                    record['entity_pages'] = ' | '.join(titles)
                            
                            # search_resultsフィールドの処理
                            if 'search_results' in item and item['search_results']:
                                results = item['search_results']
                                if isinstance(results, dict):
                                    # 辞書の値から検索結果を抽出
                                    contexts = []
                                    for key, result in list(results.items())[:2]:  # 最初の2つ
                                        if isinstance(result, dict):
                                            context = result.get('search_context', result.get('description', ''))
                                            if context and len(context) > 500:
                                                context = context[:500] + '...'
                                            if context:
                                                contexts.append(context)
                                        elif isinstance(result, str):
                                            context = result[:500] + '...' if len(result) > 500 else result
                                            contexts.append(context)
                                    record['search_results'] = ' '.join(contexts)
                                elif isinstance(results, list):
                                    # リストの場合
                                    contexts = []
                                    for result in results[:2]:
                                        if isinstance(result, dict):
                                            context = result.get('search_context', result.get('description', ''))
                                            if context and len(context) > 500:
                                                context = context[:500] + '...'
                                            if context:
                                                contexts.append(context)
                                        elif isinstance(result, str):
                                            context = result[:500] + '...' if len(result) > 500 else result
                                            contexts.append(context)
                                    record['search_results'] = ' '.join(contexts)
                            
                            data_list.append(record)
                        
                        df = pd.DataFrame(data_list)
                        
                    elif config_param:
                        dataset = hf_load_dataset(dataset_name, name=config_param, split=split_name)
                        df = dataset.to_pandas().head(sample_size) if hasattr(dataset, 'to_pandas') else pd.DataFrame(dataset[:sample_size])
                    else:
                        dataset = hf_load_dataset(dataset_name, split=split_name)
                        df = dataset.to_pandas().head(sample_size) if hasattr(dataset, 'to_pandas') else pd.DataFrame(dataset[:sample_size])
                
                # datasetsフォルダに保存
                if df is not None:
                    # datasetsフォルダを作成
                    datasets_dir = Path("datasets")
                    datasets_dir.mkdir(exist_ok=True)
                    
                    # ファイル名を生成
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_dataset_name = dataset_name.replace("/", "_").replace("-", "_")
                    csv_filename = f"{safe_dataset_name}_{split_name}_{sample_size}_{timestamp}.csv"
                    csv_path = datasets_dir / csv_filename
                    
                    # CSVファイルとして保存
                    df.to_csv(csv_path, index=False)
                    st.info(f"📂 データをdatasets/{csv_filename}に保存しました")
                    
                    # メタデータも保存
                    metadata = {
                        'dataset_name': dataset_name,
                        'dataset_type': selected_dataset,
                        'split': split_name,
                        'sample_size': sample_size,
                        'actual_size': len(df),
                        'config': config_param,
                        'downloaded_at': datetime.now().isoformat(),
                        'columns': df.columns.tolist()
                    }
                    
                    metadata_filename = f"{safe_dataset_name}_{split_name}_{sample_size}_{timestamp}_metadata.json"
                    metadata_path = datasets_dir / metadata_filename
                    
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                st.error(f"データセットのロードに失敗しました: {str(e)}")
                df = None
            
            if df is not None:
                st.session_state['uploaded_data'] = df
                st.session_state['uploaded_columns'] = df.columns.tolist()
                st.success(f"✅ {len(df)}件のデータをロードし、datasets/フォルダに保存しました")
        
        # アップロードされたデータの処理
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state['uploaded_data'] = df
            st.session_state['uploaded_columns'] = df.columns.tolist()
            st.success(f"✅ ファイルをアップロードしました: {uploaded_file.name}")
        
        # データプレビュー
        if 'uploaded_data' in st.session_state:
            df = st.session_state['uploaded_data']
            st.subheader("📋 データプレビュー")
            st.info(f"データ件数: {len(df)}件 | カラム数: {len(df.columns)}列")
            st.dataframe(df.head(10), use_container_width=True)
            
            # カラム情報
            with st.expander("📊 カラム詳細"):
                col_info = pd.DataFrame({
                    'カラム名': df.columns,
                    'データ型': df.dtypes.astype(str),
                    '非NULL数': df.count(),
                    'NULL数': df.isnull().sum(),
                    'ユニーク数': [df[col].nunique() for col in df.columns]
                })
                st.dataframe(col_info, use_container_width=True)
    
    # Tab 2: データ検証
    with tab2:
        st.header("データ品質チェック")
        
        if 'uploaded_data' not in st.session_state:
            st.warning("⚠️ まずデータをアップロードしてください")
        else:
            df = st.session_state['uploaded_data']
            
            # 共通検証
            st.subheader("📋 基本検証")
            issues = validate_data(df, selected_dataset)
            
            # データセット固有の検証
            st.subheader(f"🔍 {dataset_config['name']}固有の検証")
            
            if selected_dataset == "customer_support_faq":
                specific_issues = validate_customer_support_data_specific(df)
            elif selected_dataset == "medical_qa":
                specific_issues = validate_medical_data_specific(df)
            elif selected_dataset == "sciq_qa":
                specific_issues = validate_sciq_data_specific(df)
            elif selected_dataset == "legal_qa":
                specific_issues = validate_legal_data_specific(df)
            elif selected_dataset == "trivia_qa":
                specific_issues = validate_trivia_data_specific(df)
            else:
                specific_issues = []
            
            issues.extend(specific_issues)
            
            # 検証結果の表示
            if issues:
                st.warning("以下の問題が検出されました：")
                for issue in issues:
                    st.write(f"• {issue}")
            else:
                st.success("✅ データ検証に合格しました！")
            
            # データ統計（処理前後のデータ比較用に同じデータを渡す）
            display_statistics(df, df, selected_dataset)
            
            # 必須列のサンプル表示
            st.subheader("📝 必須列のサンプル")
            for col in dataset_config['required_columns']:
                if col in df.columns:
                    with st.expander(f"{col} のサンプル（最初の3件）"):
                        for i, value in enumerate(df[col].head(3), 1):
                            st.text(f"[{i}] {value}")
    
    # Tab 3: 前処理実行
    with tab3:
        st.header("RAG用データ前処理")
        
        if 'uploaded_data' not in st.session_state:
            st.warning("⚠️ まずデータをアップロードしてください")
        else:
            df = st.session_state['uploaded_data']
            
            # カラム結合オプション
            st.subheader("📝 テキスト結合設定")
            
            # データセット固有のカラム結合設定
            if selected_dataset == "medical_qa":
                # 実際のカラム名を大文字小文字を無視して検索
                actual_columns = []
                medical_columns_map = {}
                for col in df.columns:
                    col_lower = col.lower()
                    if 'question' in col_lower:
                        actual_columns.append(col)
                        medical_columns_map['Question'] = col
                    elif 'complex_cot' in col_lower or 'cot' in col_lower:
                        actual_columns.append(col)
                        medical_columns_map['Complex_CoT'] = col
                    elif 'response' in col_lower:
                        actual_columns.append(col)
                        medical_columns_map['Response'] = col
                
                # デフォルト値の設定
                if dataset_specific_options.get('include_cot', True) and 'Complex_CoT' in medical_columns_map:
                    default_cols = list(medical_columns_map.values())
                else:
                    default_cols = [v for k, v in medical_columns_map.items() if k != 'Complex_CoT']
                
                combine_columns = st.multiselect(
                    "結合するカラムを選択",
                    options=actual_columns,
                    default=default_cols if default_cols else actual_columns
                )
            elif selected_dataset == "sciq_qa":
                available_cols = ['question', 'correct_answer']
                if 'support' in df.columns and dataset_specific_options.get('include_support', True):
                    available_cols.append('support')
                if dataset_specific_options.get('include_distractors', False):
                    for col in ['distractor1', 'distractor2', 'distractor3']:
                        if col in df.columns:
                            available_cols.append(col)
                combine_columns = st.multiselect(
                    "結合するカラムを選択",
                    options=available_cols,
                    default=available_cols[:2]  # デフォルトは question と correct_answer
                )
            elif selected_dataset == "trivia_qa":
                # TriviaQA用のカラム設定
                available_cols = []
                if 'question' in df.columns:
                    available_cols.append('question')
                if 'answer' in df.columns:
                    available_cols.append('answer')
                if 'entity_pages' in df.columns and dataset_specific_options.get('include_entity_pages', True):
                    available_cols.append('entity_pages')
                if 'search_results' in df.columns and dataset_specific_options.get('include_search_results', True):
                    available_cols.append('search_results')
                
                # デフォルトは全カラムを選択
                default_cols = ['question', 'answer']
                if 'entity_pages' in df.columns:
                    default_cols.append('entity_pages')
                if 'search_results' in df.columns:
                    default_cols.append('search_results')
                    
                combine_columns = st.multiselect(
                    "結合するカラムを選択",
                    options=available_cols,
                    default=[col for col in default_cols if col in available_cols]
                )
            else:
                combine_columns = st.multiselect(
                    "結合するカラムを選択",
                    options=dataset_config['required_columns'],
                    default=dataset_config['required_columns']
                )
            
            # セパレータ選択
            separator_options = {
                "スペース": " ",
                "改行": "\n",
                "タブ": "\t",
                "カスタム": ""
            }
            separator_type = st.radio("セパレータ", options=list(separator_options.keys()), horizontal=True)
            
            if separator_type == "カスタム":
                custom_separator = st.text_input("カスタムセパレータ", value=" | ")
                separator = custom_separator
            else:
                separator = separator_options[separator_type]
            
            # 処理実行ボタン
            if st.button("🚀 前処理を実行", type="primary"):
                with st.spinner("処理中..."):
                    # データ処理
                    df_processed = process_rag_data(
                        df, 
                        selected_dataset,
                        combine_columns_option=True  # カラムを結合するオプション
                    )
                    
                    # Combined_Textカラムが既に存在する場合は、セパレータを適用して上書き
                    # (process_rag_data関数でCombined_Textが作成されているため)
                    if combine_columns and 'Combined_Text' in df_processed.columns:
                        # 選択されたカラムが実際に存在するか確認
                        missing_cols = [col for col in combine_columns if col not in df.columns]
                        if missing_cols:
                            st.error(f"❌ 選択されたカラムが見つかりません: {missing_cols}")
                            st.info(f"利用可能なカラム: {list(df.columns)}")
                        else:
                            # 元のデータフレームから選択されたカラムを結合
                            df_processed['Combined_Text'] = df[combine_columns].apply(
                                lambda row: separator.join(row.dropna().astype(str)), axis=1
                            )
                    
                    # 結果を保存
                    st.session_state['processed_data'] = df_processed
                    st.session_state['processing_config'] = {
                        'dataset_type': selected_dataset,
                        'combine_columns': combine_columns,
                        'separator': separator,
                        'options': dataset_specific_options
                    }
                    
                    st.success("✅ 前処理が完了しました！")
            
            # 処理済みデータのプレビュー
            if 'processed_data' in st.session_state:
                df_processed = st.session_state['processed_data']
                
                st.subheader("📋 処理済みデータプレビュー")
                st.dataframe(df_processed.head(10), use_container_width=True)
                
                # トークン使用量推定
                st.subheader("💰 トークン使用量推定")
                estimate_token_usage(df_processed, selected_model)  # この関数は内部で表示を行う
    
    # Tab 4: 結果・ダウンロード
    with tab4:
        st.header("処理結果とダウンロード")
        
        if 'processed_data' not in st.session_state:
            st.warning("⚠️ まず前処理を実行してください")
        else:
            df_processed = st.session_state['processed_data']
            config = st.session_state['processing_config']
            
            # 処理サマリー
            st.subheader("📊 処理サマリー")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("処理件数", f"{len(df_processed):,}件")
            with col2:
                st.metric("カラム数", len(df_processed.columns))
            with col3:
                st.metric("結合カラム数", len(config['combine_columns']))
            
            # ファイルダウンロード
            st.subheader("📥 ファイルダウンロード")
            
            # ダウンロードデータ作成
            csv_data, txt_data = create_download_data(
                df_processed,
                include_combined=True,
                dataset_type=config['dataset_type']
            )
            
            # メタデータを作成
            metadata = {
                'dataset_type': config['dataset_type'],
                'processed_at': datetime.now().isoformat(),
                'row_count': len(df_processed),
                'column_count': len(df_processed.columns),
                'config': config
            }
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="📄 CSVファイル",
                    data=csv_data,
                    file_name=f"preprocessed_{config['dataset_type']}.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.download_button(
                    label="📝 テキストファイル",
                    data=txt_data,
                    file_name=f"{config['dataset_type']}.txt",
                    mime="text/plain"
                )
            
            with col3:
                st.download_button(
                    label="📋 メタデータ(JSON)",
                    data=json.dumps(metadata, ensure_ascii=False, indent=2),
                    file_name=f"metadata_{config['dataset_type']}.json",
                    mime="application/json"
                )
            
            # OUTPUTフォルダへの保存
            st.divider()
            if st.button("💾 OUTPUTフォルダに保存", type="primary"):
                output_dir = Path("OUTPUT")
                # CSVデータを作成
                csv_buffer = io.StringIO()
                df_processed.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                # テキストデータを作成
                text_data = None
                if 'Combined_Text' in df_processed.columns:
                    text_data = '\n'.join(df_processed['Combined_Text'].dropna().astype(str))
                
                saved_files = save_files_to_output(
                    df_processed,
                    config['dataset_type'],
                    csv_data,
                    text_data
                )
                
                if saved_files:
                    st.success("✅ ファイルを保存しました：")
                    for file in saved_files:
                        st.write(f"• {file}")
                else:
                    st.error("❌ ファイル保存に失敗しました")
            
            # データサンプル表示
            st.divider()
            st.subheader("📝 処理済みデータのサンプル")
            st.dataframe(df_processed.head(5), use_container_width=True)
    
    # 使用方法
    with st.expander("📚 使用方法"):
        show_usage_instructions(selected_dataset)

if __name__ == "__main__":
    main()