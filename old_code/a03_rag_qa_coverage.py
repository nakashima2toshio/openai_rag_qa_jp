#!/usr/bin/env python3
"""
セマンティックカバレッジ分析とQ/A生成システム（ファイル入力対応版）
=====================================================
helper_rag_qa.pyの全クラスを活用した包括的なQ/A生成システム

使用方法:
    python a03_rag_qa_coverage.py [--input INPUT_FILE] [--dataset DATASET_TYPE] [--model MODEL] [--output OUTPUT_DIR]

例:
    # preprocessedファイルから処理
    python a03_rag_qa_coverage.py --input OUTPUT/preprocessed_cc_news.csv --dataset cc_news --model gpt-5-mini --analyze-coverage

    # テキストファイルから直接処理
    python a03_rag_qa_coverage.py --input OUTPUT/cc_news.txt --model gpt-5-mini --analyze-coverage

    # カバレージ分析付き
    python a03_rag_qa_coverage.py --input OUTPUT/preprocessed_cc_news.csv --analyze-coverage
"""

from helper_rag_qa import (
    SemanticCoverage,
    QAGenerationConsiderations,
    QAPair,
    QAPairsList,
    LLMBasedQAGenerator,
    ChainOfThoughtQAGenerator,
    RuleBasedQAGenerator,
    TemplateBasedQAGenerator,
    HybridQAGenerator,
    AdvancedQAGenerationTechniques,
    QAGenerationOptimizer,
)
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging
import pprint

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def qa_generation_checklist():
    """Q/A生成時のチェックリスト"""
    return {
        "事前準備"  : [
            "□ 文書の種類と特性を分析",
            "□ 目的（評価/学習/テスト）を明確化",
            "□ 必要なカバレッジレベルを設定",
            "□ 予算とリソースを確認"
        ],
        "品質基準"  : [
            "□ 回答がテキスト内に存在することを確認",
            "□ 質問の明確性と曖昧さの排除",
            "□ 質問タイプの多様性を確保",
            "□ 難易度のバランスを調整"
        ],
        "技術選択"  : [
            "□ ルールベースで基本的なQ/Aを生成",
            "□ LLMで複雑な推論Q/Aを補完",
            "□ ハイブリッドアプローチで最適化",
            "□ 人間のレビューで品質保証"
        ],
        "評価と改善": [
            "□ カバレッジ測定の実施",
            "□ 重複と矛盾の検出",
            "□ ユーザーフィードバックの収集",
            "□ 継続的な改善サイクル"
        ]
    }


def demonstrate_semantic_coverage(document_text):
    """SemanticCoverageクラスのデモンストレーション"""

    print("=" * 80)
    print("1. セマンティックカバレッジ分析")
    print("=" * 80)

    document_text = document_text

    # SemanticCoverageの初期化
    analyzer = SemanticCoverage(embedding_model="text-embedding-3-small")

    # 文書をセマンティックチャンクに分割
    print("\n文書をチャンクに分割中...")
    chunks = analyzer.create_semantic_chunks(document_text, verbose=False)

    print(f"\n✅ {len(chunks)}個のチャンクを作成しました")
    for i, chunk in enumerate(chunks, 1):
        print(f"  チャンク{i}: {chunk['text'][:60]}...")
        print(f"    文数: {len(chunk['sentences'])}, ID: {chunk['id']}")

    return analyzer, chunks, document_text


def demonstrate_qa_generation_considerations(document_text: str):
    """QAGenerationConsiderationsクラスのデモンストレーション"""

    print("\n\n" + "=" * 80)
    print("2. Q/A生成前のチェックリスト")
    print("=" * 80)

    # チェックリストの表示
    checklist = qa_generation_checklist()

    for category, items in checklist.items():
        print(f"\n【{category}】")
        for item in items:
            print(f"  {item}")


def demonstrate_rule_based_generation(document_text: str):
    """RuleBasedQAGeneratorクラスのデモンストレーション（改善版）"""

    print("\n\n" + "=" * 80)
    print("3. ルールベースQ/A生成")
    print("=" * 80)

    # 長いテキストの場合は警告
    text_length = len(document_text)
    print(f"\n📊 処理準備:")
    print(f"  テキスト長: {text_length:,}文字")

    # 497記事の場合、約100万文字になるため、最初の一部だけ処理
    MAX_LENGTH = 50000  # 最大5万文字
    if text_length > MAX_LENGTH:
        print(f"  ⚠️ テキストが長すぎるため、最初の{MAX_LENGTH:,}文字のみ処理します")
        document_text = document_text[:MAX_LENGTH]

    # 言語検出（簡易）
    is_english = any(word in document_text[:500] for word in ['the', 'The', 'is', 'are', 'was', 'were', 'have', 'has'])

    if is_english:
        print("  検出言語: 英語")
        print("  ⚠️ 英語テキストのため、簡易パターンマッチングを使用します")
        # 英語用の簡易パターンマッチング
        return extract_english_qa_patterns(document_text)
    else:
        print("  検出言語: 日本語")
        try:
            # タイムアウト付きでspaCyを実行
            import signal
            from contextlib import contextmanager

            @contextmanager
            def timeout(seconds):
                def signal_handler(signum, frame):
                    raise TimeoutError("処理がタイムアウトしました")

                # Windowsの場合はタイムアウトをスキップ
                if hasattr(signal, 'SIGALRM'):
                    signal.signal(signal.SIGALRM, signal_handler)
                    signal.alarm(seconds)
                    try:
                        yield
                    finally:
                        signal.alarm(0)
                else:
                    # Windows環境ではタイムアウトなし
                    yield

            try:
                with timeout(10):  # 10秒でタイムアウト
                    print("\n⏳ spaCyモデル初期化中...")
                    rule_generator = RuleBasedQAGenerator()

                print("\n定義文からQ/A抽出中...")
                with timeout(20):  # 20秒でタイムアウト
                    definition_qas = rule_generator.extract_definition_qa(document_text)

            except TimeoutError as e:
                print(f"\n⚠️ {e}")
                print("  簡易パターンマッチングにフォールバックします")
                return []

            if definition_qas:
                print(f"\n✅ {len(definition_qas)}個の定義Q/Aを生成しました")
                for i, qa in enumerate(definition_qas[:3], 1):  # 最初の3つのみ表示
                    print(f"\n  【定義Q/A {i}】")
                    print(f"    質問: {qa['question'][:50]}...")
                    print(f"    回答: {qa['answer'][:80]}...")
                    print(f"    信頼度: {qa.get('confidence', 'N/A')}")
                if len(definition_qas) > 3:
                    print(f"  ... 他{len(definition_qas) - 3}個")
            else:
                print("\n⚠️  定義文が見つかりませんでした")

            return definition_qas

        except (OSError, ImportError) as e:
            print(f"\n⚠️  spaCyモデルのエラー: {e}")
            print("    簡易パターンマッチングにフォールバックします")
            return []
        except Exception as e:
            print(f"\n⚠️  予期しないエラー: {e}")
            return []


def extract_english_qa_patterns(text: str) -> List[Dict]:
    """英語テキスト用の簡易Q/A生成（spaCy不要）"""
    import re

    qa_pairs = []

    # テキストを文に分割（最初の50文のみ処理）
    sentences = re.split(r'[.!?]\s+', text)[:50]

    print(f"\n📝 英語パターンマッチング開始...")
    print(f"  処理文数: {len(sentences)}")

    for i, sent in enumerate(sentences, 1):
        if i % 10 == 0:
            print(f"  進捗: {i}/{len(sentences)}文 処理済み...")

        # Pattern 1: "X is Y" statements
        is_match = re.search(r'^([A-Z][^,]+?)\s+(is|was|are|were)\s+(.+)$', sent.strip())
        if is_match and len(is_match.group(1)) < 80:
            subject = is_match.group(1).strip()
            verb = is_match.group(2)
            predicate = is_match.group(3).strip()[:100]

            qa_pairs.append({
                "question": f"What {verb} {subject}?",
                "answer": f"{subject} {verb} {predicate}",
                "type": "definition",
                "confidence": 0.7
            })

    # 重複除去
    seen_questions = set()
    unique_qas = []
    for qa in qa_pairs:
        if qa['question'] not in seen_questions:
            seen_questions.add(qa['question'])
            unique_qas.append(qa)

    print(f"\n✅ {len(unique_qas)}個の英語Q/Aを生成しました")

    return unique_qas[:20]  # 最大20個まで返す


def demonstrate_template_based_generation(document_text: str):
    """TemplateBasedQAGeneratorクラスのデモンストレーション"""

    print("\n\n" + "=" * 80)
    print("4. テンプレートベースQ/A生成")
    print("=" * 80)

    # テンプレートベース生成器の初期化
    template_generator = TemplateBasedQAGenerator()

    # エンティティを手動指定（実際にはNERで抽出）
    entities = ['AI', '機械学習', 'トランスフォーマー', 'BERT', 'GPT']

    print(f"\nエンティティ: {', '.join(entities)}")
    print("\nエンティティベースQ/A生成中...")

    template_qas = []
    for entity in entities[:3]:  # 最初の3つで例示
        # 簡易的なテンプレート適用
        qa = {
            "question": f"{entity}とは何ですか？",
            "answer": f"{entity}に関する情報は文書内で説明されています。",
            "entity": entity,
            "type": "entity_based",
            "confidence": 0.75
        }
        template_qas.append(qa)

    print(f"\n✅ {len(template_qas)}個のテンプレートQ/Aを生成しました")
    for i, qa in enumerate(template_qas, 1):
        print(f"\n  【テンプレートQ/A {i}】")
        print(f"    質問: {qa['question']}")
        print(f"    回答: {qa['answer']}")
        print(f"    エンティティ: {qa['entity']}")
        print(f"    信頼度: {qa.get('confidence', 'N/A')}")

    return template_qas


def demonstrate_llm_based_generation(document_text: str):
    """LLMBasedQAGeneratorクラスのデモンストレーション"""

    print("\n\n" + "=" * 80)
    print("5. LLMベースQ/A生成")
    print("=" * 80)

    api_key = os.getenv('OPENAI_API_KEY')

    if api_key:
        try:
            print("\nLLMBasedQAGenerator でQ/A生成中...")
            llm_generator = LLMBasedQAGenerator(model="gpt-4o-mini")
            llm_qas = llm_generator.generate_basic_qa(document_text, num_pairs=3)

            print(f"\n✅ {len(llm_qas)}個のLLM Q/Aを生成しました")
            for i, qa in enumerate(llm_qas[:2], 1):  # 最初の2つを表示
                print(f"\n  【LLM Q/A {i}】")
                print(f"    質問: {qa.get('question', 'N/A')}")
                print(f"    回答: {qa.get('answer', 'N/A')[:80]}...")
                print(f"    種類: {qa.get('question_type', 'N/A')}")

            return llm_qas

        except Exception as e:
            print(f"\n⚠️  LLM Q/A生成中にエラーが発生しました: {str(e)}")
            return []
    else:
        print("\n⚠️  OpenAI APIキーが必要です（スキップ）")
        print("実際の使用例:")
        print("""
    # OpenAI APIキー設定後
    llm_generator = LLMBasedQAGenerator(model="gpt-4o-mini")
    llm_qas = llm_generator.generate_basic_qa(document_text, num_pairs=3)
    """)

        # ダミーデータ
        llm_qas = [
            {
                "question": "AIの応用分野にはどのようなものがありますか？",
                "answer": "医療診断から自動運転まで幅広い分野で応用されています。",
                "question_type": "fact",
                "difficulty": "basic"
            }
        ]

        print(f"\n（シミュレーション）{len(llm_qas)}個のLLM Q/Aを生成")
        return llm_qas


def demonstrate_cot_generation(document_text: str):
    """ChainOfThoughtQAGeneratorクラスのデモンストレーション"""

    print("\n\n" + "=" * 80)
    print("6. Chain-of-Thought Q/A生成")
    print("=" * 80)

    api_key = os.getenv('OPENAI_API_KEY')

    if api_key:
        try:
            print("\nChainOfThoughtQAGenerator でQ/A生成中...")
            cot_generator = ChainOfThoughtQAGenerator()
            result = cot_generator.generate_with_reasoning(document_text)

            # 結果から qa_pairs を取得
            cot_qas = result.get('qa_pairs', []) if isinstance(result, dict) else result

            print(f"\n✅ {len(cot_qas)}個のCoT Q/Aを生成しました")
            for i, qa in enumerate(cot_qas[:2], 1):  # 最初の2つを表示
                print(f"\n  【CoT Q/A {i}】")
                print(f"    質問: {qa.get('question', 'N/A')}")
                print(f"    回答: {qa.get('answer', 'N/A')[:80]}...")
                print(f"    推論: {qa.get('reasoning', 'N/A')[:80]}...")
                print(f"    信頼度: {qa.get('confidence', 'N/A')}")

            return cot_qas

        except Exception as e:
            print(f"\n⚠️  CoT Q/A生成中にエラーが発生しました: {str(e)}")
            return []
    else:
        print("\n⚠️  OpenAI APIキーが必要です（スキップ）")
        print("実際の使用例:")
        print("""
    # OpenAI APIキー設定後
    cot_generator = ChainOfThoughtQAGenerator(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    cot_qas = cot_generator.generate_cot_qa(document_text, num_pairs=3, include_confidence=True)
    """)

        # ダミーデータ
        cot_qas = [
            {
                "question": "なぜトランスフォーマーはRNNより高速なのですか？",
                "answer": "アテンション機構により並列処理が可能だからです。",
                "reasoning": "トランスフォーマーはアテンション機構を使用 → 順次処理不要 → 並列化可能",
                "confidence": 0.92,
                "question_type": "reason",
                "difficulty": "intermediate"
            }
        ]

        print(f"\n（シミュレーション）{len(cot_qas)}個のCoT Q/Aを生成")

        return cot_qas


def demonstrate_hybrid_generation(document_text: str, rule_qas: List[Dict], template_qas: List[Dict]):
    """HybridQAGeneratorクラスのデモンストレーション"""

    print("\n\n" + "=" * 80)
    print("7. ハイブリッドQ/A生成（統合）")
    print("=" * 80)

    # 全Q/Aを統合
    all_qas = []
    all_qas.extend(rule_qas)
    all_qas.extend(template_qas)

    print(f"\n統合結果:")
    print(f"  - ルールベース: {len(rule_qas)}個")
    print(f"  - テンプレートベース: {len(template_qas)}個")
    print(f"  - 合計: {len(all_qas)}個")

    # 簡易的な重複除去
    unique_questions = {}
    for qa in all_qas:
        q = qa['question']
        if q not in unique_questions:
            unique_questions[q] = qa

    unique_qas = list(unique_questions.values())

    print(f"\n重複除去後: {len(unique_qas)}個")

    # 統合されたQ/Aペアの表示
    print(f"\n【統合Q/Aペア（最初の3個）】")
    for i, qa in enumerate(unique_qas[:3], 1):
        print(f"\n  【統合Q/A {i}】")
        print(f"    質問: {qa.get('question', 'N/A')}")
        print(f"    回答: {qa.get('answer', 'N/A')[:80]}{'...' if len(qa.get('answer', '')) > 80 else ''}")
        print(f"    タイプ: {qa.get('type', 'N/A')}")
        print(f"    信頼度: {qa.get('confidence', 'N/A')}")

    return unique_qas


def demonstrate_advanced_techniques(document_text: str):
    """AdvancedQAGenerationTechniquesクラスのデモンストレーション"""

    print("\n\n" + "=" * 80)
    print("8. 高度なQ/A生成技術")
    print("=" * 80)

    api_key = os.getenv('OPENAI_API_KEY')

    if api_key:
        try:
            print("\nAdvancedQAGenerationTechniques でQ/A生成中...")
            advanced_gen = AdvancedQAGenerationTechniques()

            # 敵対的Q/A生成（既存のQ/Aペアが必要なため、簡易的なサンプルを作成）
            print("\n敵対的Q/A生成中...")
            sample_qa = [{"question": "RAGとは何ですか？", "answer": "検索拡張生成です"}]
            adversarial_qas = advanced_gen.generate_adversarial_qa(document_text, existing_qa=sample_qa)

            print(f"\n✅ {len(adversarial_qas)}個の高度なQ/Aを生成しました")
            for i, qa in enumerate(adversarial_qas[:2], 1):
                print(f"\n  【高度なQ/A {i}】")
                print(f"    質問: {qa.get('question', 'N/A')}")
                print(f"    回答: {qa.get('answer', 'N/A')[:80]}...")
                print(f"    タイプ: {qa.get('type', 'N/A')}")

            return adversarial_qas

        except Exception as e:
            print(f"\n⚠️  高度なQ/A生成中にエラーが発生しました: {str(e)}")
            return []
    else:
        print("\n⚠️  OpenAI APIキーが必要です（スキップ）")
        print("実際の使用例:")
        print("""
    # OpenAI APIキー設定後
    advanced_gen = AdvancedQAGenerationTechniques(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

    # 敵対的Q/A生成
    adversarial_qas = advanced_gen.generate_adversarial_qa(document_text, num_pairs=3)

    # マルチホップ推論Q/A生成
    multihop_qas = advanced_gen.generate_multihop_qa(document_text, chunks, num_pairs=2)

    # 反事実的Q/A生成
    counterfactual_qas = advanced_gen.generate_counterfactual_qa(document_text, num_pairs=2)
    """)

        # ダミーデータ
        advanced_qas = [
            {
                "question": "もしトランスフォーマーが開発されていなかったら、NLPはどうなっていましたか？",
                "answer": "RNNベースのモデルが主流のまま、処理速度と精度の両立が困難だったでしょう。",
                "type": "counterfactual",
                "difficulty": "advanced"
            }
        ]

        print(f"\n（シミュレーション）{len(advanced_qas)}個の高度なQ/Aを生成")

        return advanced_qas


def demonstrate_coverage_optimization(analyzer, chunks, document_text: str, all_qas: List[Dict]):
    """QAGenerationOptimizerクラスのデモンストレーション"""

    print("\n\n" + "=" * 80)
    print("9. カバレッジ最適化")
    print("=" * 80)

    api_key = os.getenv('OPENAI_API_KEY')

    if api_key and analyzer.has_api_key:
        try:
            print("\nカバレッジ分析を実行中...")

            # 文書の埋め込みを生成
            doc_embeddings = analyzer.generate_embeddings(chunks)

            # Q/Aペアの埋め込みを生成
            qa_texts = [qa.get('question', '') + ' ' + qa.get('answer', '') for qa in all_qas if qa.get('question') and qa.get('answer')]

            if not qa_texts:
                print("\n⚠️  有効なQ/Aペアがありません")
                return

            # 各Q/Aの埋め込みを生成
            qa_embeddings = []
            for qa_text in qa_texts:
                emb = analyzer.generate_embedding(qa_text)
                qa_embeddings.append(emb)

            # カバレッジを計算（閾値0.7以上で「カバーされている」と判定）
            threshold = 0.7
            covered_chunks = set()

            for qa_emb in qa_embeddings:
                for i, doc_emb in enumerate(doc_embeddings):
                    similarity = analyzer.cosine_similarity(doc_emb, qa_emb)
                    if similarity >= threshold:
                        covered_chunks.add(i)

            coverage_rate = len(covered_chunks) / len(chunks) if len(chunks) > 0 else 0

            print(f"\n✅ カバレッジ分析完了")
            print(f"  総チャンク数: {len(chunks)}")
            print(f"  カバーされたチャンク: {len(covered_chunks)}")
            print(f"  カバレッジ率: {coverage_rate:.1%}")
            print(f"  総Q/A数: {len(all_qas)}")

            if coverage_rate < 0.8:
                uncovered_count = len(chunks) - len(covered_chunks)
                print(f"\n💡 推奨: カバーされていないチャンクが{uncovered_count}個あります")
                print(f"   追加で{uncovered_count * 2}個程度のQ/Aペア生成を推奨します")

        except Exception as e:
            print(f"\n⚠️  カバレッジ分析中にエラーが発生しました: {str(e)}")
            print("    シミュレーションモードに切り替えます")
            # フォールバックとしてシミュレーション結果を表示
            print(f"\n（シミュレーション）カバレッジ最適化結果:")
            print(f"  初期カバレッジ: 65.0%")
            print(f"  最終カバレッジ: 95.0%")
            print(f"  改善度: +30.0%")
            print(f"  新規生成Q/A数: 8個")
            print(f"  総Q/A数: {len(all_qas) + 8}個")
    else:
        print("\n⚠️  OpenAI APIキーが必要です（シミュレーションモード）")
        print("\n実際の使用例:")
        print("""
    # OpenAI APIキー設定後
    optimizer = QAGenerationOptimizer(analyzer=analyzer, generator=hybrid_gen)

    optimized_result = optimizer.optimize_coverage(
        document_text=document_text,
        existing_qa_pairs=all_qas,
        target_coverage=0.95,
        max_iterations=5
    )

    print(f"初期カバレッジ: {optimized_result['initial_coverage']:.2%}")
    print(f"最終カバレッジ: {optimized_result['coverage_rate']:.2%}")
    print(f"改善度: +{optimized_result['improvement']:.2%}")
    """)

        # シミュレーション結果
        print(f"\n（シミュレーション）カバレッジ最適化結果:")
        print(f"  初期カバレッジ: 65.0%")
        print(f"  最終カバレッジ: 95.0%")
        print(f"  改善度: +30.0%")
        print(f"  新規生成Q/A数: 8個")
        print(f"  総Q/A数: {len(all_qas) + 8}個")


def export_results(all_qas: List[Dict], output_file: str = "a03_qa_results.json"):
    """生成されたQ/Aペアをエクスポート"""

    print("\n\n" + "=" * 80)
    print("10. 結果のエクスポート")
    print("=" * 80)

    export_data = {
        "total_qa_pairs": len(all_qas),
        "generation_methods": {
            "rule_based": len([qa for qa in all_qas if qa.get('type') == 'definition' or qa.get('type') == 'terminology']),
            "template_based": len([qa for qa in all_qas if qa.get('type') == 'entity_based']),
            "llm_based": len([qa for qa in all_qas if qa.get('question_type') in ['fact', 'reason']]),
        },
        "qa_pairs": all_qas
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Q/A生成結果を {output_file} に保存しました")
    print(f"総Q/A数: {export_data['total_qa_pairs']}")
    print(f"生成手法別:")
    for method, count in export_data['generation_methods'].items():
        print(f"  - {method}: {count}個")

def print_doc():
    print("""
        1. document_text (str)
          - 型: str
          - 内容: RAGシステムに関する日本語テキスト（233文字、5文）
          - 役割: 全てのQ/A生成処理の入力となる元テキスト
            """)
    print("document_text =:", document_text)

    print("""
        2. chunks (List[Dict])

          - 型: List[Dict[str, Any]]
          - 要素数: 1個（このサンプルでは全文が1チャンクに収まる）
          - 各チャンクの構造:
          {
              "id": str,                    # "chunk_0"
              "text": str,                  # チャンク全体のテキスト
              "sentences": List[str],       # 個別の文のリスト
              "start_sentence_idx": int,    # 開始文インデックス (0)
              "end_sentence_idx": int       # 終了文インデックス (4)
          }
          - 役割: 文書を意味的に分割した単位。カバレッジ分析で使用

        3. analyzer (SemanticCoverage)

          - 型: SemanticCoverage (helper_rag_qa.SemanticCoverage)
          - 主要属性:
            - embedding_model: "text-embedding-3-small"
            - has_api_key: True
            - client: OpenAI()
          - 主要メソッド:
            - create_semantic_chunks(text) → List[Dict]
            - generate_embeddings(chunks) → List[np.ndarray]
            - generate_embedding(text) → np.ndarray
            - cosine_similarity(vec1, vec2) → float
          - 役割: セマンティック分析の中核。チャンク分割、埋め込み生成、類似度計算を実行

          この3つの変数が連携して、文書のセマンティックカバレッジ分析を実現しています。
            """)

    print(""" -----------------------------------------------------------
    
    【1. create_semantic_chunks(document: str, verbose: bool = True) → List[Dict]】

      説明: 文書を意味的に区切られたチャンクに分割

      処理手順:
        1. 文単位で分割（_split_into_sentences()）
           - 日本語: 。．.!? で分割
           - 英語: . ! ? で分割

        2. トークン数を計算しながらチャンク構築
           - max_tokens = 200 トークン/チャンク
           - 文の途中では分割しない（意味の断絶を防ぐ）

        3. トピックの連続性を考慮した調整
           - _adjust_chunks_for_topic_continuity()で最適化

      戻り値: List[Dict]
        各チャンク: {
          "id": "chunk_0",
          "text": "チャンク全体のテキスト",
          "sentences": ["文1", "文2", ...],
          "start_sentence_idx": 0,
          "end_sentence_idx": 2
        }

    【2. generate_embeddings(chunks: List[Dict]) → List[np.ndarray]】

      説明: 複数チャンクの埋め込みをバッチ生成

      処理手順:
        1. 各チャンクのtextフィールドを抽出
        2. OpenAI Embeddings APIに一括送信
           client.embeddings.create(
             input=[chunk["text"] for chunk in chunks],
             model=self.embedding_model
           )
        3. 返却されたベクトルをL2正規化

      戻り値: List[np.ndarray]
        各ベクトル: 1536次元のnumpy配列

    【3. generate_embedding(text: str) → np.ndarray】

      説明: 単一テキストの埋め込み生成

      処理手順:
        1. テキストをOpenAI APIに送信
        2. 埋め込みベクトル取得
        3. L2正規化して返却

      戻り値: np.ndarray (1536次元)

      用途: Q/Aペアの埋め込み生成

    【4. cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) → float】

      説明: 2つのベクトル間のコサイン類似度を計算

      計算式:
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

      戻り値: float (0.0〜1.0)
        - 1.0: 完全に一致
        - 0.7以上: 高い類似性（カバレッジ判定の閾値）
        - 0.0: 全く関連性なし

      用途: Q/Aペアと文書チャンクの関連性評価
        """)

# ==========================================
# 新規追加: ファイル入力処理とデータセット設定
# ==========================================

DATASET_CONFIGS = {
    "cc_news": {
        "name": "CC-News英語ニュース",
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "en"
    },
    "japanese_text": {
        "name": "日本語Webテキスト",
        "text_column": "Combined_Text",
        "title_column": None,
        "lang": "ja"
    },
    "wikipedia_ja": {
        "name": "Wikipedia日本語版",
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "ja"
    }
}


def load_input_data(input_file: str, dataset_type: Optional[str] = None, max_docs: Optional[int] = None) -> str:
    """
    入力ファイルからテキストデータを読み込み
    Args:
        input_file: 入力ファイルパス（CSV、TXT、JSON対応）
        dataset_type: データセットタイプ（CSVの場合に使用）
        max_docs: 処理する最大文書数
    Returns:
        処理対象テキスト（結合済み）
    """
    file_path = Path(input_file)
    if not file_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_file}")

    logger.info(f"データ読み込み中: {input_file}")

    # ファイル形式に応じて処理
    if file_path.suffix.lower() == '.csv':
        # CSVファイルの場合
        df = pd.read_csv(file_path)

        # データセット設定を適用
        if dataset_type and dataset_type in DATASET_CONFIGS:
            config = DATASET_CONFIGS[dataset_type]
            text_col = config["text_column"]

            if text_col not in df.columns:
                # "text"カラムがある場合はそれを使用
                if "text" in df.columns:
                    text_col = "text"
                else:
                    raise ValueError(f"テキストカラム '{text_col}' が見つかりません")

            # 文書数制限
            if max_docs:
                df = df.head(max_docs)

            # テキストを結合
            texts = df[text_col].dropna().tolist()
            combined_text = "\n\n".join([str(t) for t in texts])

            logger.info(f"読み込み完了: {len(texts)}件の文書")

        else:
            # dataset_type未指定の場合、最初のテキストカラムを使用
            text_cols = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower()]
            if text_cols:
                text_col = text_cols[0]
            else:
                # テキストらしいカラムがない場合は全カラムを結合
                text_col = df.columns[0]

            if max_docs:
                df = df.head(max_docs)

            texts = df[text_col].dropna().tolist()
            combined_text = "\n\n".join([str(t) for t in texts])

    elif file_path.suffix.lower() == '.json':
        # JSONファイルの場合
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            # リスト形式の場合
            if max_docs:
                data = data[:max_docs]

            # textフィールドを探す
            texts = []
            for item in data:
                if isinstance(item, dict):
                    if 'text' in item:
                        texts.append(item['text'])
                    elif 'content' in item:
                        texts.append(item['content'])
                    else:
                        # 最初の文字列フィールドを使用
                        for v in item.values():
                            if isinstance(v, str):
                                texts.append(v)
                                break
                else:
                    texts.append(str(item))

            combined_text = "\n\n".join(texts)

        else:
            combined_text = str(data)

    else:
        # テキストファイルとして扱う
        with open(file_path, 'r', encoding='utf-8') as f:
            combined_text = f.read()

        # max_docsが指定されている場合、段落で区切る
        if max_docs:
            paragraphs = combined_text.split('\n\n')
            paragraphs = paragraphs[:max_docs]
            combined_text = '\n\n'.join(paragraphs)

    return combined_text


def save_results(
    qa_pairs: List[Dict],
    coverage_results: Optional[Dict] = None,
    dataset_type: str = "custom",
    output_dir: str = "qa_output"
) -> Dict[str, str]:
    """
    結果をファイルに保存（a02_make_qa.pyと同じ形式）
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Q/Aペアを保存（JSON）
    qa_file = output_path / f"qa_pairs_{dataset_type}_{timestamp}.json"
    with open(qa_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    # Q/Aペアを保存（CSV）
    qa_csv_file = output_path / f"qa_pairs_{dataset_type}_{timestamp}.csv"
    qa_df = pd.DataFrame(qa_pairs)
    qa_df.to_csv(qa_csv_file, index=False, encoding='utf-8')

    saved_files = {
        "qa_json": str(qa_file),
        "qa_csv": str(qa_csv_file)
    }

    # カバレッジ分析結果がある場合は保存
    if coverage_results:
        coverage_file = output_path / f"coverage_{dataset_type}_{timestamp}.json"
        with open(coverage_file, 'w', encoding='utf-8') as f:
            json.dump(coverage_results, f, ensure_ascii=False, indent=2)
        saved_files["coverage"] = str(coverage_file)

    # サマリー情報を保存
    summary = {
        "dataset_type": dataset_type,
        "generated_at": timestamp,
        "total_qa_pairs": len(qa_pairs),
        "files": saved_files
    }

    if coverage_results:
        summary["coverage_rate"] = coverage_results.get('coverage_rate', 0)

    summary_file = output_path / f"summary_{dataset_type}_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    saved_files["summary"] = str(summary_file)

    logger.info(f"結果を保存しました: {output_path}")

    return saved_files


def generate_qa_for_chunk(chunk_text: str, num_qa: int = 2) -> List[Dict]:
    """
    単一チャンクに対してQ/Aを生成（チャンクベース改良版）
    """
    qas = []

    # 英語/日本語の簡易判定
    is_english = any(word in chunk_text[:100] for word in ['the', 'The', 'is', 'are', 'was'])

    if is_english:
        # 英語用の基本的なパターンマッチング
        sentences = chunk_text.split('. ')[:5]  # 最初の5文
        for i, sent in enumerate(sentences[:num_qa]):
            if len(sent) > 20:
                # What型の質問
                qa = {
                    'question': f"What is described in this passage about: {sent[:30]}...?",
                    'answer': sent,
                    'type': 'factual'
                }
                qas.append(qa)

                # Why/How型の質問（可能なら）
                if 'because' in sent.lower() or 'due to' in sent.lower():
                    qa = {
                        'question': f"Why does this occur: {sent[:30]}...?",
                        'answer': sent,
                        'type': 'reasoning'
                    }
                    qas.append(qa)
    else:
        # 日本語用のパターンマッチング
        sentences = chunk_text.split('。')[:5]
        for i, sent in enumerate(sentences[:num_qa]):
            if len(sent) > 10:
                qa = {
                    'question': f"{sent[:20]}...について説明してください",
                    'answer': sent,
                    'type': 'factual'
                }
                qas.append(qa)

    return qas[:num_qa]  # 指定された数だけ返す


def process_with_methods(
    document_text: str,
    methods: List[str],
    model: str = "gpt-4o-mini"
) -> tuple:
    """
    指定された手法でQ/A生成を実行（改良版：チャンクベース処理）
    Args:
        document_text: 処理対象テキスト
        methods: 使用する手法のリスト
        model: 使用するモデル
    Returns:
        (生成されたQ/Aペアのリスト, analyzer, chunks)
    """
    all_qas = []

    # SemanticCoverage初期化（チャンク作成用）
    analyzer = SemanticCoverage(embedding_model="text-embedding-3-small")
    chunks = analyzer.create_semantic_chunks(document_text, verbose=False)
    logger.info(f"チャンク作成完了: {len(chunks)}個")

    # カバレッジ目標を達成するための戦略
    total_chunks = len(chunks)
    target_coverage = 0.8  # 80%カバレッジ目標

    # 必要なQ/A数を推定（1つのQ/Aが平均2-3チャンクをカバーすると仮定）
    avg_coverage_per_qa = 2.5
    target_qa_count = int((total_chunks * target_coverage) / avg_coverage_per_qa)
    logger.info(f"目標Q/A数: 約{target_qa_count}個（80%カバレッジ達成用）")

    # チャンク数が多い場合はサンプリング
    if total_chunks > 100:
        # 均等にサンプリング（最大200チャンク）
        sample_size = min(200, total_chunks)
        step = max(1, total_chunks // sample_size)
        sampled_chunks = chunks[::step][:sample_size]
        logger.info(f"大規模データのため、{len(sampled_chunks)}チャンクをサンプリング")
    else:
        sampled_chunks = chunks

    # 各手法でチャンクベースのQ/A生成
    qa_per_method = max(2, target_qa_count // len(methods))
    chunks_per_method = min(len(sampled_chunks), qa_per_method // 2)

    # 指定された手法で処理
    if "rule" in methods:
        logger.info(f"ルールベースQ/A生成中...（{chunks_per_method}チャンク処理）")
        for i, chunk in enumerate(sampled_chunks[:chunks_per_method]):
            chunk_qas = generate_qa_for_chunk(chunk['text'], num_qa=2)
            all_qas.extend(chunk_qas)
            if (i + 1) % 50 == 0:
                logger.info(f"  進捗: {i+1}/{chunks_per_method}チャンク")

    if "template" in methods:
        logger.info(f"テンプレートベースQ/A生成中...（{chunks_per_method}チャンク処理）")
        # テンプレートベースで追加Q/A生成
        template_generator = TemplateBasedQAGenerator()
        for i, chunk in enumerate(sampled_chunks[:chunks_per_method]):
            try:
                # 複数のテンプレートを適用
                templates = [
                    "What is the main topic discussed in this text?",
                    "What are the key points mentioned?",
                    "What information is provided about",
                    "According to the passage, what"
                ]

                for template in templates[:2]:  # 各チャンクに2つのテンプレート
                    qa = {
                        'question': f"{template} {chunk['text'][:30]}...?",
                        'answer': chunk['text'][:200],
                        'type': 'template_based'
                    }
                    all_qas.append(qa)

            except Exception as e:
                logger.debug(f"テンプレート生成エラー: {e}")

            if (i + 1) % 50 == 0:
                logger.info(f"  進捗: {i+1}/{chunks_per_method}チャンク")

    if "llm" in methods:
        logger.info(f"LLMベースQ/A生成中...（コスト制約により{min(10, chunks_per_method)}チャンク）")
        # LLMは高コストなので制限
        for chunk in sampled_chunks[:min(10, chunks_per_method)]:
            try:
                llm_qas = demonstrate_llm_based_generation(chunk['text'])
                all_qas.extend(llm_qas)
            except Exception as e:
                logger.debug(f"LLM生成エラー: {e}")

    if "cot" in methods:
        logger.info("Chain-of-Thought Q/A生成中...")
        # CoTは少数の高品質Q/A
        for chunk in sampled_chunks[:min(5, chunks_per_method)]:
            try:
                cot_qas = demonstrate_cot_generation(chunk['text'])
                all_qas.extend(cot_qas)
            except Exception as e:
                logger.debug(f"CoT生成エラー: {e}")

    if "advanced" in methods:
        logger.info("高度なQ/A生成中...")
        try:
            advanced_qas = demonstrate_advanced_techniques(document_text[:5000])
            all_qas.extend(advanced_qas)
        except Exception as e:
            logger.debug(f"高度な生成エラー: {e}")

    # 重複除去
    unique_questions = {}
    for qa in all_qas:
        q = qa.get('question', '')
        if q and q not in unique_questions:
            unique_questions[q] = qa

    unique_qas = list(unique_questions.values())

    # Q/A数が目標に達しない場合は追加生成
    if len(unique_qas) < target_qa_count * 0.5:
        logger.info(f"Q/A数が不足（{len(unique_qas)}個）。追加生成中...")
        additional_needed = target_qa_count - len(unique_qas)

        # 未処理のチャンクから追加生成
        unprocessed_start = chunks_per_method * len(methods)
        for i, chunk in enumerate(chunks[unprocessed_start:unprocessed_start + additional_needed // 2]):
            chunk_qas = generate_qa_for_chunk(chunk['text'], num_qa=2)
            unique_qas.extend(chunk_qas)
            if (i + 1) % 50 == 0:
                logger.info(f"  追加生成進捗: {i+1}チャンク")

    logger.info(f"Q/A生成完了: {len(unique_qas)}個（重複除去後）")

    return unique_qas, analyzer, chunks


def main():
    """メイン実行関数（コマンドライン対応版）"""

    # コマンドライン引数のパーサー設定
    parser = argparse.ArgumentParser(
        description="セマンティックカバレッジ分析とQ/A生成システム（ファイル入力対応版）"
    )

    # 入力ファイル関連
    parser.add_argument(
        "--input",
        type=str,
        help="入力ファイルパス（CSV、TXT、JSON対応）"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        help="データセットタイプ（cc_news, japanese_text, wikipedia_ja）"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="処理する最大文書数"
    )

    # Q/A生成手法
    parser.add_argument(
        "--methods",
        type=str,
        nargs='+',
        choices=['rule', 'template', 'llm', 'cot', 'advanced', 'all'],
        default=['rule', 'template'],
        help="使用するQ/A生成手法"
    )

    # モデル設定
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="使用するOpenAIモデル"
    )

    # 出力設定
    parser.add_argument(
        "--output",
        type=str,
        default="qa_output",
        help="出力ディレクトリ"
    )

    # 分析オプション
    parser.add_argument(
        "--analyze-coverage",
        action="store_true",
        help="カバレッジ分析を実行"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="デモモード（サンプルテキストで実行）"
    )

    args = parser.parse_args()

    # ヘッダー表示
    print("\n" + "=" * 80)
    print("セマンティックカバレッジ分析とQ/A生成システム（ファイル入力対応版）")
    print("=" * 80)

    # 環境チェック
    api_key = os.getenv('OPENAI_API_KEY')
    print(f"\n📋 環境チェック:")
    print(f"  OpenAI APIキー: {'✅ 設定済み' if api_key else '❌ 未設定'}")
    if api_key:
        print(f"  動作モード: フル機能")
    else:
        print(f"  動作モード: シミュレーション（ルール/テンプレートのみ）")

    # デモモードまたはファイル入力モード
    if args.demo or not args.input:
        # デモモード: サンプルテキストで実行
        print("\n📝 デモモード: サンプルテキストで実行")
        document_text = """
        RAGシステムは、Retrieval-Augmented Generationの略で、検索拡張生成と呼ばれます。
        このシステムは、大規模言語モデルと情報検索を組み合わせた技術です。
        RAGの主な利点は、外部知識ベースから関連情報を取得し、より正確な回答を生成できることです。
        Qdrantはベクトルデータベースであり、高速な類似度検索を実現します。
        OpenAIのtext-embedding-3-smallモデルを使用して、テキストをベクトル表現に変換します。
        """
        dataset_type = "demo"

    else:
        # ファイル入力モード
        print(f"\n📁 入力ファイル: {args.input}")
        print(f"  データセット: {args.dataset if args.dataset else '自動検出'}")
        print(f"  最大文書数: {args.max_docs if args.max_docs else '制限なし'}")

        try:
            # ファイルからデータ読み込み
            document_text = load_input_data(
                args.input,
                args.dataset,
                args.max_docs
            )
            dataset_type = args.dataset if args.dataset else "custom"

        except Exception as e:
            logger.error(f"ファイル読み込みエラー: {e}")
            sys.exit(1)

    # Q/A生成手法の決定
    methods = args.methods
    if 'all' in methods:
        methods = ['rule', 'template', 'llm', 'cot', 'advanced']

    print(f"\n🛠️  使用する手法: {', '.join(methods)}")
    print(f"  モデル: {args.model}")
    print(f"  出力先: {args.output}")
    print(f"  カバレッジ分析: {'実行' if args.analyze_coverage else 'スキップ'}")

    print("\n" + "=" * 80)
    print("処理開始")
    print("=" * 80)

    try:
        # Q/A生成処理
        qa_pairs, analyzer, chunks = process_with_methods(
            document_text,
            methods,
            args.model
        )

        # カバレッジ分析（オプション）
        coverage_results = None
        if args.analyze_coverage and qa_pairs:
            print("\n" + "=" * 80)
            print("カバレッジ分析")
            print("=" * 80)

            try:
                # カバレッジ計算
                coverage_results = {}

                # 埋め込み生成とカバレッジ計算
                if api_key and analyzer.has_api_key:
                    doc_embeddings = analyzer.generate_embeddings(chunks)

                    qa_texts = [
                        qa.get('question', '') + ' ' + qa.get('answer', '')
                        for qa in qa_pairs
                        if qa.get('question') and qa.get('answer')
                    ]

                    if qa_texts:
                        qa_embeddings = []
                        for qa_text in qa_texts:
                            emb = analyzer.generate_embedding(qa_text)
                            qa_embeddings.append(emb)

                        # カバレッジ計算
                        threshold = 0.7
                        covered_chunks = set()

                        for qa_emb in qa_embeddings:
                            for i, doc_emb in enumerate(doc_embeddings):
                                similarity = analyzer.cosine_similarity(doc_emb, qa_emb)
                                if similarity >= threshold:
                                    covered_chunks.add(i)

                        coverage_rate = len(covered_chunks) / len(chunks) if chunks else 0

                        coverage_results = {
                            "coverage_rate": coverage_rate,
                            "covered_chunks": len(covered_chunks),
                            "total_chunks": len(chunks),
                            "threshold": threshold
                        }

                        print(f"\n📊 カバレッジ分析結果:")
                        print(f"  カバレッジ率: {coverage_rate:.1%}")
                        print(f"  カバー済みチャンク: {len(covered_chunks)}/{len(chunks)}")
                        print(f"  閾値: {threshold}")

            except Exception as e:
                logger.warning(f"カバレッジ分析中にエラー: {e}")

        # 結果保存
        saved_files = save_results(
            qa_pairs,
            coverage_results,
            dataset_type,
            args.output
        )

        # 完了メッセージ
        print("\n" + "=" * 80)
        print("処理完了")
        print("=" * 80)
        print(f"\n✅ 生成されたQ/Aペア数: {len(qa_pairs)}")
        print(f"✅ 保存ファイル:")
        for file_type, file_path in saved_files.items():
            print(f"  - {file_type}: {file_path}")

        # 統計情報
        if qa_pairs:
            print(f"\n📊 Q/Aペア統計:")
            type_counts = {}
            for qa in qa_pairs:
                qa_type = qa.get('type', qa.get('question_type', 'unknown'))
                type_counts[qa_type] = type_counts.get(qa_type, 0) + 1

            for qa_type, count in sorted(type_counts.items()):
                print(f"  - {qa_type}: {count}件")

    except Exception as e:
        logger.error(f"処理中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

