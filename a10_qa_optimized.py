#!/usr/bin/env python3
"""
Q&A生成に最適化されたキーワード抽出システム
キーワード間の関係性、難易度分類、文脈情報を含む
包括的なQ&A生成支援システム
自動Q/A数決定機能付き
python a10_qa_optimized.py
"""

from helper_rag_qa import QAOptimizedExtractor, QACountOptimizer
import json


def demonstrate_automatic_qa_count_optimization():
    """文書に応じた最適なQ/A数の自動決定デモ"""

    print("=" * 80)
    print("最適Q/A数の自動決定システム")
    print("=" * 80)

    # 異なる長さのテストテキスト
    test_texts = {
        "短文": """
        AIは人工知能の略称です。機械学習により学習します。
        """,

        "中文": """
        人工知能（AI）は、機械学習と深層学習を基盤として急速に発展しています。
        特に自然言語処理（NLP）の分野では、トランスフォーマーモデルが革命的な成果を上げました。
        BERTやGPTなどの大規模言語モデルは、文脈理解能力を大幅に向上させています。
        """,

        "長文": """
        アンパサンド（&, ）は、並立助詞「…と…」を意味する記号である。
        ラテン語で「…と…」を表す接続詞 "et" の合字を起源とする字形を使用している。
        英語で教育を行う学校でアルファベットを復唱する場合、その文字自体が単語となる文字（"A", "I"）が存在する。
        また、アルファベットの最後に、27番目の文字のように "&" を加えることも広く行われていた。
        "&" はラテン語で et と読み、アルファベットの復唱の最後は "X, Y, Z, and per se and" という形になった。
        この最後のフレーズが繰り返されるうちに "ampersand" となった。
        アンパサンドの起源は1世紀の古ローマ筆記体にまでさかのぼることができる。
        """
    }

    optimizer = QACountOptimizer()

    for text_type, text in test_texts.items():
        print(f"\n【{text_type}の分析】")
        print("-" * 40)

        # 各モードでの最適Q/A数を計算
        for mode in ["auto", "evaluation", "learning", "faq"]:
            result = optimizer.calculate_optimal_qa_count(text, mode)
            print(f"{mode:12} モード: {result['optimal_count']:2}個")
            if mode == "auto":
                print(f"  決定理由: {result['reasoning']}")

        print(f"\nメトリクス:")
        metrics = optimizer.calculate_optimal_qa_count(text, "auto")['metrics']
        print(f"  文書長: {metrics['doc_length']}文字")
        print(f"  文数: {metrics['sentence_count']}文")
        print(f"  複雑度スコア: {metrics['complexity_score']:.2f}")
        print(f"  キーワード密度: {metrics['keyword_density']:.1f}")


def demonstrate_progressive_generation():
    """段階的Q/A生成（収穫逓減検出付き）のデモ"""

    print("\n" + "=" * 80)
    print("段階的Q/A生成システム（収穫逓減検出付き）")
    print("=" * 80)

    test_text = """
    ブロックチェーン技術は、分散型台帳として、取引データをネットワーク上の複数のノードで共有・管理します。
    各ブロックにはハッシュ値が含まれ、前のブロックと暗号学的に連結されています。
    この仕組みにより、データの改ざんが極めて困難になります。
    ビットコインをはじめとする暗号通貨の基盤技術として注目されていますが、
    サプライチェーン管理、医療記録、投票システムなど、様々な分野での応用が期待されています。
    スマートコントラクトと組み合わせることで、自動執行可能な契約も実現できます。
    """

    extractor = QAOptimizedExtractor()

    print("\n【段階的生成モード】")
    result = extractor.extract_for_qa_generation(
        test_text,
        qa_count=None,  # 自動決定
        mode="auto",
        use_progressive=True,  # 段階的生成を使用
        return_details=True
    )

    # 生成過程の表示
    if 'metadata' in result and 'progressive_generation' in result['metadata']:
        prog_info = result['metadata']['progressive_generation']
        print(f"\n処理バッチ数: {prog_info['batches_processed']}")
        print(f"最終カバレッジ: {prog_info['final_coverage']:.2%}")

        print("\nカバレッジ改善履歴:")
        for batch in prog_info.get('coverage_history', []):
            print(f"  バッチ{batch['batch']}: "
                  f"カバレッジ {batch['coverage']:.2%} "
                  f"(改善 +{batch['improvement']:.2%}, "
                  f"追加キーワード {batch['keywords_added']}個)")

    print(f"\n最終的に生成されたQ/A数: {result['metadata']['qa_count']}個")


def demonstrate_qa_extraction():
    """Q&A最適化抽出のデモンストレーション"""

    print("=" * 80)
    print("Q&A生成最適化キーワード抽出システム")
    print("=" * 80)

    # テストテキスト
    test_text2 = """
    人工知能（AI）は、機械学習と深層学習を基盤として急速に発展しています。
    特に自然言語処理（NLP）の分野では、トランスフォーマーモデルが革命的な成果を上げました。
    BERTやGPTなどの大規模言語モデルは、文脈理解能力を大幅に向上させています。

    トランスフォーマーは、アテンション機構を使用して、
    文章中の単語間の関係を効率的に学習します。
    これにより、従来のRNNやLSTMよりも高速で正確な処理が可能になりました。

    AIの応用は医療診断から自動運転まで幅広く、社会に大きな影響を与えています。
    しかし、AIの倫理的な課題やバイアスの問題も重要な議論となっています。
    """

    test_text = """
    アンパサンド アンパサンド（&, ）は、並立助詞「…と…」を意味する記号である。
    ラテン語で「…と…」を表す接続詞 "et" の合字を起源とする字形を使用している。
    語源 英語で教育を行う学校でアルファベットを復唱する場合、その文字自体が単語となる文字（"A", "I", かつては唱えられていた。
    また、アルファベットの最後に、27番目の文字のように "&" を加えることも広く行われていた。
    "&" はラテン語で et と読して、アルファベットの復唱の最後は "X, Y, Z, and per se and" という形になった。
    この最後のフレーズが繰り返されるうちに "ampersaマリ・アンペールがこの記号を自身の著作で使い、これが広く読まれたため、この記号が "Ampère's and" と呼ばれるようになったという誤った語源俗説がある。
    歴史 アンパサンドの起源は1世紀の古ローマ筆記体にまでさかのぼることができる。
    """

    # Q&A最適化抽出を実行
    extractor = QAOptimizedExtractor()
    result = extractor.extract_for_qa_generation(
        test_text,
        qa_count=5,
        difficulty_distribution={'basic': 0.4, 'intermediate': 0.4, 'advanced': 0.2}
    )

    print(f"\n入力テキスト長: {len(test_text)}文字")
    print(f"抽出方法: {result['metadata']['extraction_method']}")
    print(f"総キーワード数: {result['metadata']['total_keywords_extracted']}")
    print(f"関係性数: {result['metadata']['total_relations_found']}")

    # キーワードと文脈を表示
    print("\n" + "=" * 80)
    print("抽出されたキーワード（文脈付き）")
    print("=" * 80)

    for i, item in enumerate(result['keywords'], 1):
        print(f"\n{i}. {item['keyword']}")
        print(f"   難易度: {item['difficulty']}")
        print(f"   カテゴリ: {item['category']}")
        print(f"   出現回数: {item['frequency']}")
        if item['best_context']:
            print(f"   文脈: {item['best_context'][:60]}...")

    # 関係性を表示
    if result['relations']:
        print("\n" + "=" * 80)
        print("キーワード間の関係性")
        print("=" * 80)

        for rel in result['relations'][:5]:
            print(f"\n• {rel['from']} → {rel['to']}")
            print(f"  関係: {rel['relation_type']}")
            print(f"  文脈: {rel['context'][:60]}...")

    # Q&Aテンプレートを表示
    print("\n" + "=" * 80)
    print("生成されたQ&Aテンプレート")
    print("=" * 80)

    for i, qa in enumerate(result['suggested_qa_pairs'], 1):
        print(f"\n{i}. キーワード: {qa['keyword']} ({qa['difficulty']})")
        print(f"   質問例:")
        for q in qa['question_templates'][:2]:
            print(f"   - {q}")
        print(f"   推奨回答長: {qa['suggested_answer_length']}")

    # 実際のQ&Aペアを生成
    print("\n" + "=" * 80)
    print("生成されたQ&Aペア例")
    print("=" * 80)

    qa_pairs = extractor.generate_qa_pairs(result)
    for i, qa in enumerate(qa_pairs[:3], 1):
        print(f"\n【Q{i}】{qa['question']}")
        print(f"【A{i}】{qa['answer']}")
        print(f"   難易度: {qa['difficulty']}, 推奨長: {qa['suggested_length']}")
        print(f"   関連キーワード: {', '.join(qa['keywords'])}")


def analyze_extraction_quality():
    """抽出品質の分析"""

    print("\n\n" + "=" * 80)
    print("抽出品質の比較分析")
    print("=" * 80)

    test_cases = {
        "技術文書": """
            ブロックチェーン技術は、分散型台帳として、取引データを
            ネットワーク上の複数のノードで共有・管理します。
            各ブロックにはハッシュ値が含まれ、改ざんが極めて困難です。
        """,

        "説明文書": """
            量子コンピュータとは、量子力学の原理を利用した計算機です。
            量子ビット（キュービット）は0と1の重ね合わせ状態を取ることができ、
            これにより並列計算が可能になります。
        """
    }

    extractor = QAOptimizedExtractor()

    for doc_type, text in test_cases.items():
        print(f"\n【{doc_type}】")
        print("-" * 40)

        # Q&A最適化抽出
        result = extractor.extract_for_qa_generation(
            text,
            qa_count=3,
            return_details=False
        )

        print("抽出キーワード:", ', '.join(result['keywords']))
        print("\n生成可能な質問:")
        for qa in result['qa_templates']:
            if qa['question']:
                print(f"  • {qa['question']} ({qa['difficulty']})")


def export_qa_data(text: str, output_file: str = "qa_extraction_result.json"):
    """Q&A抽出結果をJSON形式でエクスポート"""

    extractor = QAOptimizedExtractor()
    result = extractor.extract_for_qa_generation(text, qa_count=5)

    # エクスポート用データを整形
    export_data = {
        'input_text': text[:200] + '...' if len(text) > 200 else text,
        'extraction_metadata': result['metadata'],
        'keywords': [
            {
                'keyword': kw['keyword'],
                'difficulty': kw['difficulty'],
                'category': kw['category'],
                'best_context': kw['best_context']
            }
            for kw in result['keywords']
        ],
        'relations': result['relations'],
        'qa_templates': [
            {
                'keyword': qa['keyword'],
                'difficulty': qa['difficulty'],
                'questions': qa['question_templates']
            }
            for qa in result['suggested_qa_pairs']
        ]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Q&A抽出結果を {output_file} に保存しました")
    return export_data


def main():
    """メイン実行関数"""

    # デモンストレーション
    demonstrate_qa_extraction()

    # 品質分析
    analyze_extraction_quality()

    # 新機能のデモンストレーション
    print("\n" + "=" * 80)
    print("新機能: 文書に応じた最適Q/A数の自動決定")
    print("=" * 80)
    demonstrate_automatic_qa_count_optimization()

    print("\n" + "=" * 80)
    print("新機能: 段階的Q/A生成（収穫逓減検出付き）")
    print("=" * 80)
    demonstrate_progressive_generation()

    # エクスポート例
    sample_text = """
        深層学習は、多層ニューラルネットワークを用いた機械学習の手法です。
        特に画像認識や自然言語処理で優れた性能を発揮します。
        CNNは画像処理に、RNNは時系列データに適しています。
    """

    export_qa_data(sample_text)

    print("\n" + "=" * 80)
    print("Q&A最適化抽出システムの特徴:")
    print("- キーワード間の関係性を抽出")
    print("- 難易度に応じた分類")
    print("- 文脈情報の保持")
    print("- Q&Aテンプレートの自動生成")
    print("- 冗長なキーワードの除去")
    print("- 文書特性に応じた最適Q/A数の自動決定")
    print("- 段階的生成と収穫逓減の検出")
    print("=" * 80)


if __name__ == "__main__":
    main()