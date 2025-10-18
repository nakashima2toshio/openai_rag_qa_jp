#!/usr/bin/env python3
"""
セマンティックカバレッジ分析とQ/A生成システム
helper_rag_qa.pyの全クラスを活用した包括的なQ/A生成のデモンストレーション
python a03_rag_qa_coverage.py
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
import json
from typing import List, Dict
import pprint

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
    """RuleBasedQAGeneratorクラスのデモンストレーション"""

    print("\n\n" + "=" * 80)
    print("3. ルールベースQ/A生成")
    print("=" * 80)

    try:
        # ルールベース生成器の初期化
        rule_generator = RuleBasedQAGenerator()

        # 定義文からQ/A抽出
        print("\n定義文からQ/A抽出中...")
        definition_qas = rule_generator.extract_definition_qa(document_text)

        if definition_qas:
            print(f"\n✅ {len(definition_qas)}個の定義Q/Aを生成しました")
            for i, qa in enumerate(definition_qas, 1):
                print(f"\n  【定義Q/A {i}】")
                print(f"    質問: {qa['question']}")
                print(f"    回答: {qa['answer']}")
                print(f"    信頼度: {qa.get('confidence', 'N/A')}")
        else:
            print("\n⚠️  定義文が見つかりませんでした")

        return definition_qas

    except OSError as e:
        print(f"\n⚠️  spaCy日本語モデルがインストールされていません")
        print("    インストールコマンド: python -m spacy download ja_core_news_lg")
        print("    ルールベースQ/A生成をスキップします")
        return []


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

def main():
    # 1. セマンティックカバレッジ分析
    import pprint

    # サンプル文書
    document_text = """
    RAGシステムは、Retrieval-Augmented Generationの略で、検索拡張生成と呼ばれます。
    このシステムは、大規模言語モデルと情報検索を組み合わせた技術です。
    RAGの主な利点は、外部知識ベースから関連情報を取得し、より正確な回答を生成できることです。
    Qdrantはベクトルデータベースであり、高速な類似度検索を実現します。
    OpenAIのtext-embedding-3-smallモデルを使用して、テキストをベクトル表現に変換します。
    """

    analyzer, chunks, document_text = demonstrate_semantic_coverage(document_text)

    print("document_text =:", document_text)

    print("\n全ての属性: -------------------------------------")
    for attr in dir(analyzer):
        if not attr.startswith('_'):
            value = getattr(analyzer, attr)
            if not callable(value):
                print(f"  - {attr}: {value}")

    print("\n全てのメソッド:------------------------------------")
    for attr in dir(analyzer):
        if not attr.startswith('_') and callable(getattr(analyzer, attr)):
            print(f"  - {attr}()")



def main2():
    import pprint

    """メイン実行関数"""

    print("\n" + "=" * 80)
    print("セマンティックカバレッジ分析とQ/A生成システム")
    print("=" * 80)

    # 環境チェック
    api_key = os.getenv('OPENAI_API_KEY')
    print(f"\n📋 環境チェック:")
    print(f"  OpenAI APIキー: {'✅ 設定済み' if api_key else '❌ 未設定'}")
    if api_key:
        print(f"  動作モード: フル機能（全てのLLM APIを使用）")
    else:
        print(f"  動作モード: 完全シミュレーション")
    print()

    # 1. セマンティックカバレッジ分析
    import pprint

    # サンプル文書
    document_text = """
    RAGシステムは、Retrieval-Augmented Generationの略で、検索拡張生成と呼ばれます。
    このシステムは、大規模言語モデルと情報検索を組み合わせた技術です。
    RAGの主な利点は、外部知識ベースから関連情報を取得し、より正確な回答を生成できることです。
    Qdrantはベクトルデータベースであり、高速な類似度検索を実現します。
    OpenAIのtext-embedding-3-smallモデルを使用して、テキストをベクトル表現に変換します。
    """
    analyzer, chunks, document_text = demonstrate_semantic_coverage(document_text)

    pprint.pprint(analyzer)

    # 2. Q/A生成前のチェックリスト
    demonstrate_qa_generation_considerations(document_text)

    # 3. ルールベースQ/A生成
    rule_qas = demonstrate_rule_based_generation(document_text)

    # 4. テンプレートベースQ/A生成
    template_qas = demonstrate_template_based_generation(document_text)

    # 5. LLMベースQ/A生成（シミュレーション）
    llm_qas = demonstrate_llm_based_generation(document_text)

    # 6. Chain-of-Thought Q/A生成（シミュレーション）
    cot_qas = demonstrate_cot_generation(document_text)

    # 7. ハイブリッドQ/A生成
    all_qas = demonstrate_hybrid_generation(document_text, rule_qas, template_qas)

    # 8. 高度なQ/A生成技術（シミュレーション）
    advanced_qas = demonstrate_advanced_techniques(document_text)

    # 9. カバレッジ最適化（シミュレーション）
    demonstrate_coverage_optimization(analyzer, chunks, document_text, all_qas)

    # 10. 結果のエクスポート
    export_results(all_qas)

    # まとめ
    print("\n\n" + "=" * 80)
    print("まとめ")
    print("=" * 80)
    print("\n本システムの特徴:")
    print("  ✅ SemanticCoverage - 文書の意味的チャンク分割と埋め込み生成")
    print("  ✅ RuleBasedQAGenerator - パターンマッチングによる確実なQ/A生成")
    print("  ✅ TemplateBasedQAGenerator - エンティティベースQ/A生成")
    print("  ✅ LLMBasedQAGenerator - GPTによる多様なQ/A生成（要APIキー）")
    print("  ✅ ChainOfThoughtQAGenerator - 推論過程付き高品質Q/A（要APIキー）")
    print("  ✅ HybridQAGenerator - 複数手法の統合と品質検証")
    print("  ✅ AdvancedQAGenerationTechniques - 敵対的・マルチホップ・反事実的Q/A（要APIキー）")
    print("  ✅ QAGenerationOptimizer - カバレッジ最適化とコスト管理（要APIキー）")
    print("=" * 80)


if __name__ == "__main__":
    main()

