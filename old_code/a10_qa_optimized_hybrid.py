#!/usr/bin/env python3
"""
ハイブリッドQ&A生成システム（最適化版）
ルールベース抽出 + LLM品質向上 + セマンティックカバレージ計算

使用方法:
    # 基本使用（cc_newsデータセット、gpt-5-mini使用）
    python a10_qa_optimized_hybrid.py --dataset cc_news --output qa_output

    # モデル指定
    python a10_qa_optimized_hybrid.py --dataset cc_news --model gpt-4o-mini

    # LLMなし（ルールベースのみ）
    python a10_qa_optimized_hybrid.py --dataset cc_news --no-llm

    # カバレージ計算なし
    python a10_qa_optimized_hybrid.py --dataset cc_news --no-coverage

    # 文書タイプ指定
    python a10_qa_optimized_hybrid.py --dataset cc_news --doc-type news
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
from tqdm import tqdm

# helper_rag_qa から新しいクラスをインポート
from helper_rag_qa import OptimizedHybridQAGenerator

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# データセット設定
# ==========================================

DATASET_CONFIGS = {
    "cc_news": {
        "name": "CC-News英語ニュース",
        "file": "OUTPUT/preprocessed_cc_news.csv",
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "en",
        "default_doc_type": "news"
    },
    "japanese_text": {
        "name": "日本語Webテキスト",
        "file": "OUTPUT/preprocessed_japanese_text.csv",
        "text_column": "Combined_Text",
        "title_column": None,
        "lang": "ja",
        "default_doc_type": "auto"
    },
    "wikipedia_ja": {
        "name": "Wikipedia日本語版",
        "file": "OUTPUT/preprocessed_wikipedia_ja.csv",
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "ja",
        "default_doc_type": "academic"
    }
}

# ==========================================
# データ読み込み
# ==========================================

def load_preprocessed_data(dataset_type: str) -> pd.DataFrame:
    """preprocessedデータを読み込み"""
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
# ハイブリッドQ/A生成処理
# ==========================================

def generate_hybrid_qa_from_dataset(
    df: pd.DataFrame,
    dataset_type: str,
    model: str = "gpt-5-mini",
    max_docs: Optional[int] = None,
    qa_count: Optional[int] = None,
    use_llm: bool = True,
    calculate_coverage: bool = True,
    doc_type: Optional[str] = None,
    output_dir: str = "qa_output"
) -> Dict:
    """データセットからハイブリッドQ/A生成

    Args:
        df: データフレーム
        dataset_type: データセットタイプ
        model: 使用するLLMモデル
        max_docs: 処理する最大文書数
        qa_count: 生成するQ/A数（Noneで自動決定）
        use_llm: LLMを使用するか
        calculate_coverage: カバレージ計算するか
        doc_type: 文書タイプ（Noneの場合はデータセットのデフォルト）
        output_dir: 出力ディレクトリ

    Returns:
        生成結果の辞書
    """
    config = DATASET_CONFIGS[dataset_type]
    text_col = config["text_column"]
    title_col = config.get("title_column")

    # 文書タイプの決定
    if doc_type is None:
        doc_type = config.get("default_doc_type", "auto")

    # 処理する文書数を制限
    docs_to_process = df.head(max_docs) if max_docs else df

    logger.info(f"ハイブリッドQ/A生成開始: {len(docs_to_process)}件の文書")
    logger.info(f"使用モデル: {model if use_llm else 'ルールベースのみ'}")

    # Q/A生成器の初期化
    generator = OptimizedHybridQAGenerator(model=model)

    all_results = []
    total_api_calls = 0
    total_tokens = 0
    total_cost = 0.0
    total_qa_generated = 0
    coverage_scores = []

    # プログレスバー付きで処理
    for idx, row in tqdm(docs_to_process.iterrows(), total=len(docs_to_process), desc="Q/A生成"):
        # テキスト取得
        text = str(row[text_col]) if pd.notna(row[text_col]) else ""

        # タイトル取得（あれば）
        title = ""
        if title_col and title_col in row and pd.notna(row[title_col]):
            title = str(row[title_col])

        # ドキュメントID作成
        doc_id = f"{dataset_type}_{idx}"
        if title:
            doc_id += f"_{title[:30].replace(' ', '_')}"

        logger.debug(f"処理中: {doc_id}")

        # ハイブリッドQ/A生成
        try:
            result = generator.generate_hybrid_qa(
                text=text,
                qa_count=qa_count,
                use_llm=use_llm,
                calculate_coverage=calculate_coverage,
                document_type=doc_type
            )

            # メタデータ追加
            result['doc_id'] = doc_id
            result['doc_idx'] = idx
            result['title'] = title
            result['text_length'] = len(text)

            # 統計更新
            total_api_calls += result['api_usage']['calls']
            total_tokens += result['api_usage']['tokens']
            total_cost += result['api_usage']['cost']
            total_qa_generated += len(result['qa_pairs'])

            if calculate_coverage:
                coverage_scores.append(result['coverage'].get('coverage_percentage', 0))

            all_results.append(result)

        except Exception as e:
            logger.error(f"文書 {doc_id} の処理中にエラー: {e}")
            continue

    # 全体のサマリー作成
    summary = {
        "dataset_type": dataset_type,
        "dataset_name": config["name"],
        "model_used": model if use_llm else "rule-based",
        "documents_processed": len(all_results),
        "total_qa_generated": total_qa_generated,
        "avg_qa_per_doc": total_qa_generated / len(all_results) if all_results else 0,
        "api_usage": {
            "total_calls": total_api_calls,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "cost_per_doc": total_cost / len(all_results) if all_results else 0
        },
        "coverage": {
            "calculated": calculate_coverage,
            "avg_coverage": sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0,
            "min_coverage": min(coverage_scores) if coverage_scores else 0,
            "max_coverage": max(coverage_scores) if coverage_scores else 0
        },
        "generation_timestamp": datetime.now().isoformat()
    }

    return {
        "summary": summary,
        "results": all_results
    }

# ==========================================
# 結果保存
# ==========================================

def save_results(
    generation_results: Dict,
    dataset_type: str,
    model: str,
    output_dir: str = "qa_output"
) -> Dict[str, str]:
    """結果をファイルに保存

    Args:
        generation_results: 生成結果
        dataset_type: データセットタイプ
        model: 使用したモデル
        output_dir: 出力ディレクトリ

    Returns:
        保存したファイルパス
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_suffix = model.replace("-", "_").replace(".", "_")

    # 1. サマリーファイル
    summary_file = output_path / f"hybrid_summary_{dataset_type}_{model_suffix}_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(generation_results['summary'], f, ensure_ascii=False, indent=2)

    # 2. 詳細結果ファイル（JSON）
    details_file = output_path / f"hybrid_details_{dataset_type}_{model_suffix}_{timestamp}.json"
    with open(details_file, 'w', encoding='utf-8') as f:
        json.dump(generation_results['results'], f, ensure_ascii=False, indent=2)

    # 3. Q/Aペア（CSV）
    qa_data = []
    for doc_result in generation_results['results']:
        doc_id = doc_result['doc_id']
        for qa in doc_result.get('qa_pairs', []):
            qa_data.append({
                'doc_id': doc_id,
                'question': qa.get('question', ''),
                'answer': qa.get('answer', ''),
                'doc_title': doc_result.get('title', ''),
                'text_length': doc_result.get('text_length', 0)
            })

    if qa_data:
        qa_df = pd.DataFrame(qa_data)
        qa_csv = output_path / f"hybrid_qa_pairs_{dataset_type}_{model_suffix}_{timestamp}.csv"
        qa_df.to_csv(qa_csv, index=False, encoding='utf-8')
    else:
        qa_csv = None

    # 4. カバレージレポート（CSV）
    if generation_results['summary']['coverage']['calculated']:
        coverage_data = []
        for doc_result in generation_results['results']:
            if 'coverage' in doc_result:
                coverage_data.append({
                    'doc_id': doc_result['doc_id'],
                    'total_chunks': doc_result['coverage'].get('total_chunks', 0),
                    'covered_chunks': doc_result['coverage'].get('covered_chunks', 0),
                    'coverage_percentage': doc_result['coverage'].get('coverage_percentage', 0),
                    'average_similarity': doc_result['coverage'].get('average_similarity', 0)
                })

        if coverage_data:
            coverage_df = pd.DataFrame(coverage_data)
            coverage_csv = output_path / f"hybrid_coverage_{dataset_type}_{model_suffix}_{timestamp}.csv"
            coverage_df.to_csv(coverage_csv, index=False, encoding='utf-8')
        else:
            coverage_csv = None
    else:
        coverage_csv = None

    logger.info(f"結果を保存しました: {output_path}")

    return {
        "summary": str(summary_file),
        "details": str(details_file),
        "qa_pairs_csv": str(qa_csv) if qa_csv else None,
        "coverage_csv": str(coverage_csv) if coverage_csv else None
    }

# ==========================================
# コスト見積もり
# ==========================================

def estimate_cost(dataset_type: str, model: str, use_llm: bool = True) -> Dict:
    """処理コストの見積もり"""

    # データセットサイズの取得
    config = DATASET_CONFIGS[dataset_type]
    file_path = config["file"]

    if Path(file_path).exists():
        df = pd.read_csv(file_path)
        doc_count = len(df)
    else:
        # デフォルト値
        doc_count = 497 if dataset_type == "cc_news" else 100

    # モデル別の料金（1Mトークンあたり、ドル）
    pricing = {
        "gpt-5-mini": {"input": 0.15, "output": 0.60},
        "gpt-5": {"input": 1.50, "output": 6.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "o1-mini": {"input": 3.00, "output": 12.00},
        "o1": {"input": 15.00, "output": 60.00},
        "o3-mini": {"input": 3.00, "output": 12.00}
    }

    if not use_llm:
        return {
            "document_count": doc_count,
            "model": "rule-based",
            "estimated_cost": 0.0,
            "api_calls": 0,
            "note": "ルールベースのみ（API使用なし）"
        }

    model_pricing = pricing.get(model, pricing["gpt-5-mini"])

    # 1文書あたりの推定トークン数
    avg_tokens_per_doc = 500  # 入力300 + 出力200

    # LLM呼び出し（Q/A生成）
    llm_calls = doc_count
    llm_tokens = doc_count * avg_tokens_per_doc
    llm_cost = (llm_tokens * 0.7 * model_pricing["input"] +
                llm_tokens * 0.3 * model_pricing["output"]) / 1_000_000

    # 埋め込み生成（カバレージ計算用）
    embedding_calls = doc_count * 2  # チャンク + Q/A
    embedding_tokens = doc_count * 200  # 簡略化
    embedding_cost = embedding_tokens * 0.00002  # text-embedding-3-smallの料金

    total_cost = llm_cost + embedding_cost

    return {
        "document_count": doc_count,
        "model": model,
        "llm_calls": llm_calls,
        "embedding_calls": embedding_calls,
        "total_api_calls": llm_calls + embedding_calls,
        "estimated_tokens": llm_tokens + embedding_tokens,
        "llm_cost": round(llm_cost, 4),
        "embedding_cost": round(embedding_cost, 4),
        "estimated_total_cost": round(total_cost, 4),
        "cost_per_document": round(total_cost / doc_count, 6) if doc_count > 0 else 0
    }

# ==========================================
# メイン処理
# ==========================================

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="ハイブリッドQ&A生成システム（LLM + ルールベース + カバレージ計算）"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        default="cc_news",
        help="処理するデータセット"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="使用するLLMモデル（デフォルト: gpt-5-mini）"
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
        "--doc-type",
        type=str,
        choices=["news", "technical", "academic", "auto"],
        default=None,
        help="文書タイプ（未指定の場合はデータセットのデフォルト）"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="LLMを使用しない（ルールベースのみ）"
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="カバレージ計算を行わない"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="qa_output",
        help="出力ディレクトリ"
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="コスト見積もりのみ実行"
    )

    args = parser.parse_args()

    # コスト見積もりモード
    if args.estimate_only:
        logger.info("コスト見積もりモード")
        estimate = estimate_cost(args.dataset, args.model, not args.no_llm)

        print("\n" + "=" * 80)
        print("📊 処理コスト見積もり")
        print("=" * 80)
        print(f"データセット: {DATASET_CONFIGS[args.dataset]['name']}")
        print(f"文書数: {estimate['document_count']}")
        print(f"使用モデル: {estimate['model']}")

        if not args.no_llm:
            print(f"\nAPI呼び出し:")
            print(f"  - LLM呼び出し: {estimate['llm_calls']}回")
            print(f"  - 埋め込み生成: {estimate['embedding_calls']}回")
            print(f"  - 合計: {estimate['total_api_calls']}回")

            print(f"\nコスト内訳:")
            print(f"  - LLM: ${estimate['llm_cost']:.4f}")
            print(f"  - 埋め込み: ${estimate['embedding_cost']:.4f}")
            print(f"  - 合計: ${estimate['estimated_total_cost']:.4f}")
            print(f"  - 1文書あたり: ${estimate['cost_per_document']:.6f}")
        else:
            print(f"\nコスト: $0.00 （ルールベースのみ）")

        print("=" * 80)
        return

    # 通常の処理
    logger.info(f"""
    =====================================
    ハイブリッドQ&A生成開始
    =====================================
    データセット: {DATASET_CONFIGS[args.dataset]['name']}
    モデル: {args.model if not args.no_llm else 'ルールベースのみ'}
    出力先: {args.output}
    最大文書数: {args.max_docs if args.max_docs else '制限なし'}
    Q/A数: {args.qa_count if args.qa_count else '自動決定'}
    LLM使用: {'はい' if not args.no_llm else 'いいえ'}
    カバレージ計算: {'はい' if not args.no_coverage else 'いいえ'}
    文書タイプ: {args.doc_type if args.doc_type else 'デフォルト'}
    """)

    # OpenAI APIキーの確認
    if not args.no_llm and not os.getenv('OPENAI_API_KEY'):
        logger.error("OpenAI APIキーが設定されていません。環境変数 OPENAI_API_KEY を設定してください。")
        sys.exit(1)

    try:
        # 1. データ読み込み
        logger.info("\n[1/3] データ読み込み...")
        df = load_preprocessed_data(args.dataset)

        # 2. ハイブリッドQ/A生成
        logger.info("\n[2/3] ハイブリッドQ/A生成...")
        generation_results = generate_hybrid_qa_from_dataset(
            df,
            args.dataset,
            model=args.model,
            max_docs=args.max_docs,
            qa_count=args.qa_count,
            use_llm=not args.no_llm,
            calculate_coverage=not args.no_coverage,
            doc_type=args.doc_type,
            output_dir=args.output
        )

        # 3. 結果保存
        logger.info("\n[3/3] 結果保存...")
        saved_files = save_results(
            generation_results,
            args.dataset,
            args.model,
            args.output
        )

        # 完了メッセージ
        summary = generation_results['summary']
        logger.info(f"""
        =====================================
        処理完了
        =====================================
        処理文書数: {summary['documents_processed']}
        生成Q/A総数: {summary['total_qa_generated']}
        平均Q/A数/文書: {summary['avg_qa_per_doc']:.1f}

        API使用状況:
        - 総呼び出し回数: {summary['api_usage']['total_calls']}
        - 総トークン数: {summary['api_usage']['total_tokens']}
        - 総コスト: ${summary['api_usage']['total_cost']:.4f}
        - 文書あたりコスト: ${summary['api_usage']['cost_per_doc']:.6f}

        カバレージ:
        - 平均カバレージ: {summary['coverage']['avg_coverage']:.1f}%
        - 最小: {summary['coverage']['min_coverage']:.1f}%
        - 最大: {summary['coverage']['max_coverage']:.1f}%

        保存ファイル:
        - サマリー: {saved_files['summary']}
        - 詳細: {saved_files['details']}
        - Q/A CSV: {saved_files['qa_pairs_csv']}
        - カバレージCSV: {saved_files['coverage_csv']}
        """)

        # 改善効果の説明
        print("\n" + "=" * 80)
        print("🚀 ハイブリッドアプローチの効果")
        print("=" * 80)
        print("""
このハイブリッド版では、以下の改善が実現されました：

1. **品質向上**（LLM使用時）
   - テンプレートベースから自然な質問文へ
   - 文書タイプに応じた適切な質問生成
   - 文脈を考慮した包括的な回答

2. **カバレージ測定**
   - セマンティックな類似度による正確な測定
   - チャンクレベルでの詳細な分析
   - Q/Aペアの網羅性の定量化

3. **コスト最適化**
   - ルールベースで候補を絞り込み
   - LLMは品質向上のみに使用
   - 必要に応じてルールベースのみも選択可能

4. **柔軟性**
   - 複数のLLMモデルから選択可能
   - 文書タイプごとの最適化
   - カバレージ目標の調整可能
""")

        if not args.no_llm:
            print(f"実際のAPI使用コスト: ${summary['api_usage']['total_cost']:.4f}")
        else:
            print("API使用コスト: $0.00（ルールベースのみ）")

    except Exception as e:
        logger.error(f"処理中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()