#!/usr/bin/env python3
"""
Q&A生成に最適化されたキーワード抽出システム
preprocessed CSVファイル対応版（a02_make_qa.py参考）

使用方法:
    # 基本使用（cc_newsデータセット）
    python a10_qa_optimized.py --dataset cc_news --output qa_output

    # データセット指定
    python a10_qa_optimized.py --dataset cc_news --max-docs 10

    # 出力ディレクトリ指定
    python a10_qa_optimized.py --dataset japanese_text --output qa_output

    # Q/A数を手動指定
    python a10_qa_optimized.py --dataset wikipedia_ja --qa-count 10
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging

from helper_rag_qa import QAOptimizedExtractor, QACountOptimizer

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# データセット設定（a02_make_qa.pyから引用）
# ==========================================

DATASET_CONFIGS = {
    "cc_news": {
        "name": "CC-News英語ニュース",
        "file": "OUTPUT/preprocessed_cc_news.csv",
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "en",
    },
    "japanese_text": {
        "name": "日本語Webテキスト",
        "file": "OUTPUT/preprocessed_japanese_text.csv",
        "text_column": "Combined_Text",
        "title_column": None,
        "lang": "ja",
    },
    "wikipedia_ja": {
        "name": "Wikipedia日本語版",
        "file": "OUTPUT/preprocessed_wikipedia_ja.csv",
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "ja",
    }
}


# ==========================================
# データ読み込み関数
# ==========================================

def load_preprocessed_data(dataset_type: str) -> pd.DataFrame:
    """preprocessedデータを読み込み
    Args:
        dataset_type: データセットタイプ
    Returns:
        読み込んだDataFrame
    """
    config = DATASET_CONFIGS.get(dataset_type)
    if not config:
        raise ValueError(f"未対応のデータセット: {dataset_type}")

    file_path = config["file"]
    if not Path(file_path).exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

    logger.info(f"データ読み込み中: {file_path}")
    df = pd.read_csv(file_path)

    # 必要なカラムの確認
    text_col = config["text_column"]
    if text_col not in df.columns:
        raise ValueError(f"テキストカラム '{text_col}' が見つかりません")

    # 空のテキストを除外
    df = df[df[text_col].notna() & (df[text_col].str.strip() != '')]

    logger.info(f"読み込み完了: {len(df)}件のデータ")
    return df


# ==========================================
# キーワード抽出処理
# ==========================================

def extract_keywords_from_dataset(
    df: pd.DataFrame,
    dataset_type: str,
    max_docs: Optional[int] = None,
    qa_count: Optional[int] = None,
    use_progressive: bool = True,
    output_dir: str = "qa_keywords_output"
) -> Dict:
    """データセットからキーワード抽出とQ/Aテンプレート生成

    Args:
        df: データフレーム
        dataset_type: データセットタイプ
        max_docs: 処理する最大文書数
        qa_count: 生成するQ/A数（Noneの場合は自動決定）
        use_progressive: 段階的生成を使用するか
        output_dir: 出力ディレクトリ

    Returns:
        抽出結果の辞書
    """
    config = DATASET_CONFIGS[dataset_type]
    text_col = config["text_column"]
    title_col = config.get("title_column")

    # 処理する文書数を制限
    docs_to_process = df.head(max_docs) if max_docs else df

    logger.info(f"キーワード抽出開始: {len(docs_to_process)}件の文書")

    extractor = QAOptimizedExtractor()
    optimizer = QACountOptimizer()

    all_results = []
    total_keywords = 0
    total_relations = 0
    total_qa_templates = 0

    for idx, row in docs_to_process.iterrows():
        # テキスト取得
        text = str(row[text_col]) if pd.notna(row[text_col]) else ""

        # タイトル取得（あれば）
        title = ""
        if title_col and title_col in row and pd.notna(row[title_col]):
            title = str(row[title_col])

        # ドキュメントID作成
        doc_id = f"{dataset_type}_{idx}"
        if title:
            doc_id += f"_{title[:30]}"

        logger.debug(f"処理中: {doc_id}")

        # Q/A数を自動決定（未指定の場合）
        if qa_count is None:
            qa_result = optimizer.calculate_optimal_qa_count(text, mode="auto")
            doc_qa_count = qa_result['optimal_count']
            logger.debug(f"  自動決定Q/A数: {doc_qa_count}")
        else:
            doc_qa_count = qa_count

        # キーワード抽出とQ/Aテンプレート生成
        result = extractor.extract_for_qa_generation(
            text,
            qa_count=doc_qa_count,
            mode="auto",
            use_progressive=use_progressive,
            return_details=True
        )

        # メタデータ追加
        result['doc_id'] = doc_id
        result['doc_idx'] = idx
        result['title'] = title

        # 統計更新
        total_keywords += result['metadata']['total_keywords_extracted']
        total_relations += result['metadata']['total_relations_found']
        total_qa_templates += len(result.get('suggested_qa_pairs', []))

        all_results.append(result)

    # 結果のサマリー作成
    summary = {
        "dataset_type": dataset_type,
        "dataset_name": config["name"],
        "documents_processed": len(docs_to_process),
        "total_keywords": total_keywords,
        "total_relations": total_relations,
        "total_qa_templates": total_qa_templates,
        "avg_keywords_per_doc": total_keywords / len(docs_to_process) if docs_to_process.shape[0] > 0 else 0,
        "avg_relations_per_doc": total_relations / len(docs_to_process) if docs_to_process.shape[0] > 0 else 0,
        "avg_qa_templates_per_doc": total_qa_templates / len(docs_to_process) if docs_to_process.shape[0] > 0 else 0,
        "extraction_timestamp": datetime.now().isoformat()
    }

    return {
        "summary": summary,
        "results": all_results
    }


# ==========================================
# 結果保存
# ==========================================

def save_results(
    extraction_results: Dict,
    dataset_type: str,
    output_dir: str = "qa_keywords_output"
) -> Dict[str, str]:
    """結果をファイルに保存

    Args:
        extraction_results: 抽出結果
        dataset_type: データセットタイプ
        output_dir: 出力ディレクトリ

    Returns:
        保存したファイルパス
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. サマリーファイル
    summary_file = output_path / f"keyword_summary_{dataset_type}_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(extraction_results['summary'], f, ensure_ascii=False, indent=2)

    # 2. 詳細結果ファイル（JSON）
    details_file = output_path / f"keyword_details_{dataset_type}_{timestamp}.json"
    with open(details_file, 'w', encoding='utf-8') as f:
        json.dump(extraction_results['results'], f, ensure_ascii=False, indent=2)

    # 3. キーワードリスト（CSV）
    keywords_data = []
    for doc_result in extraction_results['results']:
        doc_id = doc_result['doc_id']
        for kw in doc_result.get('keywords', []):
            keywords_data.append({
                'doc_id': doc_id,
                'keyword': kw['keyword'],
                'difficulty': kw['difficulty'],
                'category': kw['category'],
                'frequency': kw['frequency'],
                'context': kw.get('best_context', '')[:100]  # 最初の100文字
            })

    if keywords_data:
        keywords_df = pd.DataFrame(keywords_data)
        keywords_csv = output_path / f"keywords_{dataset_type}_{timestamp}.csv"
        keywords_df.to_csv(keywords_csv, index=False, encoding='utf-8')
    else:
        keywords_csv = None

    # 4. Q/Aテンプレート（CSV）
    qa_data = []
    for doc_result in extraction_results['results']:
        doc_id = doc_result['doc_id']
        for qa in doc_result.get('suggested_qa_pairs', []):
            for q_template in qa.get('question_templates', []):
                qa_data.append({
                    'doc_id': doc_id,
                    'keyword': qa['keyword'],
                    'difficulty': qa['difficulty'],
                    'question_template': q_template,
                    'suggested_answer_length': qa.get('suggested_answer_length', 'medium')
                })

    if qa_data:
        qa_df = pd.DataFrame(qa_data)
        qa_csv = output_path / f"qa_templates_{dataset_type}_{timestamp}.csv"
        qa_df.to_csv(qa_csv, index=False, encoding='utf-8')
    else:
        qa_csv = None

    logger.info(f"結果を保存しました: {output_path}")

    return {
        "summary": str(summary_file),
        "details": str(details_file),
        "keywords_csv": str(keywords_csv) if keywords_csv else None,
        "qa_templates_csv": str(qa_csv) if qa_csv else None
    }


# ==========================================
# メイン処理
# ==========================================

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="Q&A生成最適化キーワード抽出システム（preprocessed対応版）"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        default="cc_news",
        help="処理するデータセット"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="処理する最大文書数（テスト用）"
    )
    parser.add_argument(
        "--qa-count",
        type=int,
        default=None,
        help="文書あたりのQ/A数（未指定の場合は自動決定）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="qa_keywords_output",
        help="出力ディレクトリ"
    )
    parser.add_argument(
        "--no-progressive",
        action="store_true",
        help="段階的生成を無効化"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="デモンストレーションモードで実行"
    )

    args = parser.parse_args()

    # デモモードの場合は元の処理を実行
    if args.demo:
        logger.info("デモモードで実行中...")
        run_original_demo()
        return

    logger.info(f"""
    =====================================
    Q&A最適化キーワード抽出開始
    =====================================
    データセット: {DATASET_CONFIGS[args.dataset]['name']}
    出力先: {args.output}
    最大文書数: {args.max_docs if args.max_docs else '制限なし'}
    Q/A数: {args.qa_count if args.qa_count else '自動決定'}
    段階的生成: {'無効' if args.no_progressive else '有効'}
    """)

    try:
        # 1. データ読み込み
        logger.info("\n[1/3] データ読み込み...")
        df = load_preprocessed_data(args.dataset)

        # 2. キーワード抽出
        logger.info("\n[2/3] キーワード抽出とQ/Aテンプレート生成...")
        extraction_results = extract_keywords_from_dataset(
            df,
            args.dataset,
            max_docs=args.max_docs,
            qa_count=args.qa_count,
            use_progressive=not args.no_progressive,
            output_dir=args.output
        )

        # 3. 結果保存
        logger.info("\n[3/3] 結果保存...")
        saved_files = save_results(extraction_results, args.dataset, args.output)

        # 完了メッセージ
        summary = extraction_results['summary']
        logger.info(f"""
        =====================================
        処理完了
        =====================================
        処理文書数: {summary['documents_processed']}
        抽出キーワード総数: {summary['total_keywords']}
        関係性総数: {summary['total_relations']}
        Q/Aテンプレート総数: {summary['total_qa_templates']}

        平均値（文書あたり）:
        - キーワード: {summary['avg_keywords_per_doc']:.1f}個
        - 関係性: {summary['avg_relations_per_doc']:.1f}個
        - Q/Aテンプレート: {summary['avg_qa_templates_per_doc']:.1f}個

        保存ファイル:
        - サマリー: {saved_files['summary']}
        - 詳細: {saved_files['details']}
        - キーワードCSV: {saved_files['keywords_csv']}
        - Q/AテンプレートCSV: {saved_files['qa_templates_csv']}
        """)

        # OpenAI API利用回数の説明
        print("\n" + "=" * 80)
        print("📊 OpenAI API利用回数について")
        print("=" * 80)
        print(f"""
このツール（a10_qa_optimized.py）はテンプレートベースでQ/Aを生成するため、
OpenAI APIを使用しません（0回）。

ただし、生成されたテンプレートから実際のQ/Aペアを生成する場合は、
後処理でLLMを使用することができます（オプション）。

【497記事（preprocessed_cc_news.csv全体）を処理した場合】
- キーワード抽出: 0回（ルールベース）
- 関係性分析: 0回（パターンマッチング）
- Q/Aテンプレート生成: 0回（テンプレート）
- 合計: 0回

【LLMと併用する場合（オプション）】
各記事に対してLLMでQ/Aを生成する場合:
- 497記事 × 1回 = 497回のAPI呼び出し
- コスト試算: 約$0.15（gpt-5-mini使用時）
""")

    except Exception as e:
        logger.error(f"処理中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_original_demo():
    """元のデモンストレーション関数を実行"""
    from helper_rag_qa import QAOptimizedExtractor, QACountOptimizer

    # 元のデモコードを実行
    print("=" * 80)
    print("Q&A生成最適化キーワード抽出システム（デモ）")
    print("=" * 80)

    test_text = """
    人工知能（AI）は、機械学習と深層学習を基盤として急速に発展しています。
    特に自然言語処理（NLP）の分野では、トランスフォーマーモデルが革命的な成果を上げました。
    BERTやGPTなどの大規模言語モデルは、文脈理解能力を大幅に向上させています。
    """

    extractor = QAOptimizedExtractor()
    result = extractor.extract_for_qa_generation(
        test_text,
        qa_count=3,
        difficulty_distribution={'basic': 0.4, 'intermediate': 0.4, 'advanced': 0.2}
    )

    print(f"\n入力テキスト長: {len(test_text)}文字")
    print(f"抽出キーワード数: {result['metadata']['total_keywords_extracted']}")
    print(f"関係性数: {result['metadata']['total_relations_found']}")

    print("\n抽出されたキーワード:")
    for kw in result['keywords'][:5]:
        print(f"  • {kw['keyword']} ({kw['difficulty']})")

    print("\n生成されたQ&Aテンプレート:")
    for qa in result['suggested_qa_pairs'][:3]:
        print(f"  • {qa['keyword']}: {qa['question_templates'][0] if qa['question_templates'] else 'N/A'}")


if __name__ == "__main__":
    main()