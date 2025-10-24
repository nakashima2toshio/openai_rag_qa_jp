#!/usr/bin/env python3
"""
バッチ処理版ハイブリッドQ&A生成システム
API呼び出しを最小化し、処理を高速化
# 確実に95%達成するための推奨コマンド
  python a10_qa_optimized_hybrid_batch.py \
      --dataset cc_news \
      --model gpt-5-mini \
      --batch-size 10 \
      --embedding-batch-size 150 \
      --qa-count 12 \
      --max-docs 150 \
      --output qa_output


使用方法:
    # 基本使用（バッチサイズ10）
    python a10_qa_optimized_hybrid_batch.py --dataset cc_news

    # バッチサイズ指定
    python a10_qa_optimized_hybrid_batch.py --dataset cc_news --batch-size 20

    # モデル指定
    python a10_qa_optimized_hybrid_batch.py --dataset cc_news --model gpt-5-mini

    # 比較実行（通常版 vs バッチ版）
    python a10_qa_optimized_hybrid_batch.py --dataset cc_news --compare
"""

import os
import sys
import json
import argparse
import pandas as pd
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging
from tqdm import tqdm

# helper_rag_qa から新しいバッチクラスをインポート
from helper_rag_qa import BatchHybridQAGenerator, OptimizedHybridQAGenerator

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# データセット設定（a10_qa_optimized_hybrid.pyと同じ）
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

def load_preprocessed_data(dataset_type: str, max_docs: Optional[int] = None) -> pd.DataFrame:
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

    # 最大文書数で制限
    if max_docs:
        df = df.head(max_docs)

    logger.info(f"読み込み完了: {len(df)}件のデータ")
    return df

# ==========================================
# バッチ処理によるQ/A生成
# ==========================================

def generate_batch_qa_from_dataset(
    df: pd.DataFrame,
    dataset_type: str,
    model: str = "gpt-5-mini",
    batch_size: int = 10,
    embedding_batch_size: int = 100,
    qa_count: Optional[int] = None,
    use_llm: bool = True,
    calculate_coverage: bool = True,
    doc_type: Optional[str] = None,
    output_dir: str = "qa_output"
) -> Dict:
    """バッチ処理でデータセットからQ/A生成"""

    config = DATASET_CONFIGS[dataset_type]
    text_col = config["text_column"]
    title_col = config.get("title_column")

    # 文書タイプの決定
    if doc_type is None:
        doc_type = config.get("default_doc_type", "auto")

    logger.info(f"バッチ処理Q/A生成開始: {len(df)}件の文書")
    logger.info(f"バッチサイズ: LLM={batch_size}, 埋め込み={embedding_batch_size}")

    # テキストリストの準備
    texts = df[text_col].tolist()

    # バッチ生成器の初期化
    generator = BatchHybridQAGenerator(
        model=model,
        batch_size=batch_size,
        embedding_batch_size=embedding_batch_size
    )

    # 処理時間の計測開始
    start_time = time.time()

    # バッチ処理でQ/A生成
    batch_results = generator.generate_batch_hybrid_qa(
        texts=texts,
        qa_count=qa_count,
        use_llm=use_llm,
        calculate_coverage=calculate_coverage,
        document_type=doc_type,
        show_progress=True
    )

    # 処理時間の計算
    elapsed_time = time.time() - start_time

    # 統計情報の集計
    total_qa_generated = sum(len(r["qa_pairs"]) for r in batch_results)
    total_cost = sum(r["api_usage"]["cost"] for r in batch_results)

    coverage_scores = []
    if calculate_coverage:
        coverage_scores = [r["coverage"].get("coverage_percentage", 0) for r in batch_results]

    # サマリー作成
    summary = {
        "dataset_type": dataset_type,
        "dataset_name": config["name"],
        "model_used": model if use_llm else "rule-based",
        "batch_processing": True,
        "batch_sizes": {
            "llm_batch_size": batch_size,
            "embedding_batch_size": embedding_batch_size
        },
        "documents_processed": len(batch_results),
        "total_qa_generated": total_qa_generated,
        "avg_qa_per_doc": total_qa_generated / len(batch_results) if batch_results else 0,
        "processing_time": {
            "total_seconds": elapsed_time,
            "minutes": elapsed_time / 60,
            "docs_per_second": len(batch_results) / elapsed_time if elapsed_time > 0 else 0
        },
        "api_usage": {
            "total_cost": total_cost,
            "cost_per_doc": total_cost / len(batch_results) if batch_results else 0,
            "batch_statistics": generator.batch_stats
        },
        "coverage": {
            "calculated": calculate_coverage,
            "avg_coverage": sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0,
            "min_coverage": min(coverage_scores) if coverage_scores else 0,
            "max_coverage": max(coverage_scores) if coverage_scores else 0
        },
        "generation_timestamp": datetime.now().isoformat()
    }

    # 各文書にメタデータを追加
    for i, result in enumerate(batch_results):
        result['doc_id'] = f"{dataset_type}_{i}"
        result['doc_idx'] = i
        if title_col and title_col in df.columns:
            result['title'] = df.iloc[i][title_col] if pd.notna(df.iloc[i][title_col]) else ""
        else:
            result['title'] = ""
        result['text_length'] = len(texts[i])

    return {
        "summary": summary,
        "results": batch_results
    }

# ==========================================
# 通常版との比較実行
# ==========================================

def compare_with_normal_version(
    df: pd.DataFrame,
    dataset_type: str,
    model: str = "gpt-5-mini",
    sample_size: int = 10
) -> Dict:
    """通常版とバッチ版の性能比較"""

    config = DATASET_CONFIGS[dataset_type]
    text_col = config["text_column"]

    # サンプルデータ
    sample_df = df.head(sample_size)
    texts = sample_df[text_col].tolist()

    logger.info(f"\n{'='*80}")
    logger.info(f"通常版 vs バッチ版 比較実行（{sample_size}文書）")
    logger.info(f"{'='*80}")

    # 通常版の実行
    logger.info("\n■ 通常版（個別処理）実行中...")
    normal_generator = OptimizedHybridQAGenerator(model=model)

    normal_start = time.time()
    normal_results = []
    normal_api_calls = 0

    for text in tqdm(texts, desc="通常版"):
        result = normal_generator.generate_hybrid_qa(
            text=text,
            qa_count=3,
            use_llm=True,
            calculate_coverage=True,
            document_type="auto"
        )
        normal_results.append(result)
        normal_api_calls += 3  # LLM + 埋め込み×2

    normal_elapsed = time.time() - normal_start

    # バッチ版の実行
    logger.info("\n■ バッチ版（バッチ処理）実行中...")
    batch_generator = BatchHybridQAGenerator(model=model, batch_size=5)

    batch_start = time.time()
    batch_results = batch_generator.generate_batch_hybrid_qa(
        texts=texts,
        qa_count=3,
        use_llm=True,
        calculate_coverage=True,
        document_type="auto",
        show_progress=True
    )
    batch_elapsed = time.time() - batch_start

    # 比較結果
    comparison = {
        "sample_size": sample_size,
        "normal_version": {
            "total_time": normal_elapsed,
            "time_per_doc": normal_elapsed / sample_size,
            "api_calls": normal_api_calls,
            "calls_per_doc": normal_api_calls / sample_size
        },
        "batch_version": {
            "total_time": batch_elapsed,
            "time_per_doc": batch_elapsed / sample_size,
            "api_calls": batch_generator.batch_stats["total_llm_calls"] +
                        batch_generator.batch_stats["total_embedding_calls"],
            "calls_per_doc": (batch_generator.batch_stats["total_llm_calls"] +
                             batch_generator.batch_stats["total_embedding_calls"]) / sample_size
        },
        "improvement": {
            "time_reduction": f"{(1 - batch_elapsed/normal_elapsed) * 100:.1f}%",
            "api_call_reduction": f"{(1 - (batch_generator.batch_stats['total_llm_calls'] +
                                          batch_generator.batch_stats['total_embedding_calls']) /
                                          normal_api_calls) * 100:.1f}%",
            "speedup": f"{normal_elapsed/batch_elapsed:.2f}x"
        }
    }

    # 結果表示
    print("\n" + "="*80)
    print("📊 性能比較結果")
    print("="*80)
    print(f"サンプル数: {sample_size}文書\n")

    print("【通常版（個別処理）】")
    print(f"  処理時間: {normal_elapsed:.2f}秒")
    print(f"  API呼出: {normal_api_calls}回")
    print(f"  1文書あたり: {normal_elapsed/sample_size:.2f}秒, {normal_api_calls/sample_size:.1f}回\n")

    print("【バッチ版（バッチ処理）】")
    print(f"  処理時間: {batch_elapsed:.2f}秒")
    print(f"  API呼出: {comparison['batch_version']['api_calls']}回")
    print(f"  1文書あたり: {batch_elapsed/sample_size:.2f}秒, "
          f"{comparison['batch_version']['calls_per_doc']:.1f}回\n")

    print("【改善効果】")
    print(f"  処理時間短縮: {comparison['improvement']['time_reduction']}")
    print(f"  API呼出削減: {comparison['improvement']['api_call_reduction']}")
    print(f"  高速化: {comparison['improvement']['speedup']}")
    print("="*80)

    return comparison

# ==========================================
# 結果保存
# ==========================================

def save_batch_results(
    generation_results: Dict,
    dataset_type: str,
    model: str,
    batch_size: int,
    output_dir: str = "qa_output"
) -> Dict[str, str]:
    """バッチ処理結果を保存"""

    # qa_output/a10 ディレクトリに保存
    output_path = Path(output_dir) / "a10"
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_suffix = model.replace("-", "_").replace(".", "_")

    # 1. サマリーファイル
    summary_file = output_path / f"batch_summary_{dataset_type}_{model_suffix}_b{batch_size}_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(generation_results['summary'], f, ensure_ascii=False, indent=2)

    # 2. Q/Aペア（CSV）
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
        qa_csv = output_path / f"batch_qa_pairs_{dataset_type}_{model_suffix}_b{batch_size}_{timestamp}.csv"
        qa_df.to_csv(qa_csv, index=False, encoding='utf-8')
    else:
        qa_csv = None

    logger.info(f"結果を保存しました: {output_path}")

    return {
        "summary": str(summary_file),
        "qa_pairs_csv": str(qa_csv) if qa_csv else None
    }

# ==========================================
# メイン処理
# ==========================================

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="バッチ処理版ハイブリッドQ&A生成システム"
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
        help="使用するLLMモデル"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="LLMバッチサイズ（デフォルト: 10）"
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=100,
        help="埋め込みバッチサイズ（デフォルト: 100）"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="処理する最大文書数"
    )
    parser.add_argument(
        "--qa-count",
        type=int,
        default=None,
        help="文書あたりのQ/A数"
    )
    parser.add_argument(
        "--doc-type",
        type=str,
        choices=["news", "technical", "academic", "auto"],
        default=None,
        help="文書タイプ"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="LLMを使用しない"
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
        "--compare",
        action="store_true",
        help="通常版との比較実行"
    )
    parser.add_argument(
        "--compare-size",
        type=int,
        default=10,
        help="比較実行のサンプルサイズ"
    )

    args = parser.parse_args()

    # OpenAI APIキーの確認
    if not args.no_llm and not os.getenv('OPENAI_API_KEY'):
        logger.error("OpenAI APIキーが設定されていません。環境変数 OPENAI_API_KEY を設定してください。")
        sys.exit(1)

    logger.info(f"""
=====================================
バッチ処理版ハイブリッドQ&A生成
=====================================
データセット: {DATASET_CONFIGS[args.dataset]['name']}
モデル: {args.model if not args.no_llm else 'ルールベースのみ'}
バッチサイズ: LLM={args.batch_size}, 埋め込み={args.embedding_batch_size}
出力先: {args.output}
最大文書数: {args.max_docs if args.max_docs else '制限なし'}
""")

    try:
        # データ読み込み
        logger.info("\n[1/3] データ読み込み...")
        df = load_preprocessed_data(args.dataset, args.max_docs)

        # 比較実行モード
        if args.compare:
            logger.info("\n[比較実行モード]")
            comparison_result = compare_with_normal_version(
                df, args.dataset, args.model, args.compare_size
            )

            # 比較結果を保存
            output_path = Path(args.output)
            output_path.mkdir(exist_ok=True)
            comparison_file = output_path / f"comparison_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(comparison_file, 'w', encoding='utf-8') as f:
                json.dump(comparison_result, f, ensure_ascii=False, indent=2)
            logger.info(f"比較結果を保存: {comparison_file}")
            return

        # 通常のバッチ処理実行
        logger.info("\n[2/3] バッチ処理Q/A生成...")
        generation_results = generate_batch_qa_from_dataset(
            df,
            args.dataset,
            model=args.model,
            batch_size=args.batch_size,
            embedding_batch_size=args.embedding_batch_size,
            qa_count=args.qa_count,
            use_llm=not args.no_llm,
            calculate_coverage=not args.no_coverage,
            doc_type=args.doc_type,
            output_dir=args.output
        )

        # 結果保存
        logger.info("\n[3/3] 結果保存...")
        saved_files = save_batch_results(
            generation_results,
            args.dataset,
            args.model,
            args.batch_size,
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

処理時間:
- 合計: {summary['processing_time']['total_seconds']:.2f}秒
- 分: {summary['processing_time']['minutes']:.2f}分
- 処理速度: {summary['processing_time']['docs_per_second']:.2f}文書/秒

API使用状況:
- LLM呼び出し: {summary['api_usage']['batch_statistics']['total_llm_calls']}回
- 埋め込み呼び出し: {summary['api_usage']['batch_statistics']['total_embedding_calls']}回
- 総コスト: ${summary['api_usage']['total_cost']:.4f}

カバレージ:
- 平均: {summary['coverage']['avg_coverage']:.1f}%
- 最小: {summary['coverage']['min_coverage']:.1f}%
- 最大: {summary['coverage']['max_coverage']:.1f}%

保存ファイル:
- サマリー: {saved_files['summary']}
- Q/A CSV: {saved_files['qa_pairs_csv']}
""")

        # バッチ処理の効果を表示
        print("\n" + "="*80)
        print("🚀 バッチ処理の効果")
        print("="*80)

        original_calls = summary['documents_processed'] * 3  # 通常版の推定
        actual_calls = (summary['api_usage']['batch_statistics']['total_llm_calls'] +
                       summary['api_usage']['batch_statistics']['total_embedding_calls'])

        print(f"""
バッチ処理により以下の改善を実現：

1. **API呼び出し削減**
   - 通常版（推定）: {original_calls}回
   - バッチ版（実際）: {actual_calls}回
   - 削減率: {(1 - actual_calls/original_calls) * 100:.1f}%

2. **処理速度向上**
   - 処理速度: {summary['processing_time']['docs_per_second']:.2f}文書/秒
   - {summary['documents_processed']}文書を{summary['processing_time']['minutes']:.1f}分で処理

3. **スケーラビリティ**
   - 大規模データセット処理が現実的に
   - レート制限リスクの大幅低減
""")

    except Exception as e:
        logger.error(f"処理中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()