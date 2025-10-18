#!/usr/bin/env python3
"""
KeywordExtractor統合版処理のデモスクリプト
MeCabと正規表現を統合したキーワード抽出システムの実装例
python a10_make_qa.py
"""

from regex_mecab import KeywordExtractor
import json
from typing import List, Dict, Tuple


def demonstrate_basic_extraction(example_str: str) -> None:
    """基本的なキーワード抽出のデモンストレーション"""
    print("=" * 80)
    print("1. 基本的なキーワード抽出")
    print("=" * 80)
    print(f"\n入力テキスト:\n{example_str}\n")

    # KeywordExtractorのインスタンス化
    extractor = KeywordExtractor(prefer_mecab=True)

    # シンプルな抽出（スコアリングあり）
    print("【統合版抽出結果（スコアリングあり、上位5件）】")
    keywords = extractor.extract(example_str, top_n=5, use_scoring=True)
    for i, keyword in enumerate(keywords, 1):
        print(f"  {i}. {keyword}")

    # シンプルな抽出（スコアリングなし、頻度ベース）
    print("\n【統合版抽出結果（頻度ベース、上位5件）】")
    keywords_freq = extractor.extract(example_str, top_n=5, use_scoring=False)
    for i, keyword in enumerate(keywords_freq, 1):
        print(f"  {i}. {keyword}")


def demonstrate_detailed_extraction(example_str: str) -> None:
    """詳細情報付きキーワード抽出のデモンストレーション"""
    print("\n" + "=" * 80)
    print("2. 詳細情報付きキーワード抽出")
    print("=" * 80)

    extractor = KeywordExtractor(prefer_mecab=True)

    # 各手法での抽出結果を取得
    results = extractor.extract_with_details(example_str, top_n=10)

    for method, keyword_scores in results.items():
        print(f"\n【{method}】")
        print("-" * 40)
        for i, (keyword, score) in enumerate(keyword_scores, 1):
            # スコアをプログレスバー風に表示
            bar_length = int(score * 20)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            print(f"  {i:2d}. {keyword:20s} [{bar}] {score:.3f}")

    # 手法間の比較分析
    analyze_method_comparison(results)


def analyze_method_comparison(results: Dict[str, List[Tuple[str, float]]]) -> None:
    """手法間の比較分析"""
    print("\n" + "=" * 80)
    print("3. 手法間の比較分析")
    print("=" * 80)

    # 各手法のキーワードセットを作成
    keyword_sets = {}
    for method, keywords_scored in results.items():
        keyword_sets[method] = set(kw for kw, _ in keywords_scored)

    # 統計情報
    print("\n【統計情報】")
    for method, keywords in keyword_sets.items():
        avg_length = sum(len(kw) for kw in keywords) / len(keywords) if keywords else 0
        print(f"  {method:12s}: {len(keywords):2d}個のキーワード（平均長: {avg_length:.1f}文字）")

    # 共通キーワードの分析
    print("\n【共通キーワード分析】")

    # すべての手法で共通
    if len(keyword_sets) > 1:
        common_all = set.intersection(*keyword_sets.values())
        if common_all:
            print(f"  全手法で共通: {', '.join(sorted(common_all))}")
        else:
            print(f"  全手法で共通: なし")

    # ペアワイズの共通性
    methods = list(keyword_sets.keys())
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method1, method2 = methods[i], methods[j]
            common = keyword_sets[method1] & keyword_sets[method2]
            if common:
                overlap_rate = len(common) / min(len(keyword_sets[method1]), len(keyword_sets[method2])) * 100
                print(f"  {method1} ∩ {method2}: {len(common)}個 ({overlap_rate:.1f}%重複)")

    # ユニークキーワード
    print("\n【各手法のユニークキーワード】")
    for method, keywords in keyword_sets.items():
        other_keywords = set()
        for other_method, other_kw_set in keyword_sets.items():
            if other_method != method:
                other_keywords.update(other_kw_set)
        unique = keywords - other_keywords
        if unique:
            print(f"  {method}のみ: {', '.join(sorted(unique)[:5])}")


def demonstrate_different_texts() -> None:
    """異なるタイプのテキストでのデモンストレーション"""
    print("\n" + "=" * 80)
    print("4. 異なるタイプのテキストでの抽出")
    print("=" * 80)

    test_texts = {
        "技術文書": """
            機械学習アルゴリズムの最適化において、ハイパーパラメータチューニングは
            重要な役割を果たします。グリッドサーチやベイズ最適化などの手法により、
            モデルの性能を大幅に向上させることができます。特にディープラーニングでは、
            学習率やバッチサイズの調整が精度に大きく影響します。
        """,

        "ビジネス文書": """
            デジタルトランスフォーメーション（DX）を推進する企業が増加しています。
            クラウドコンピューティングとビッグデータ分析を活用することで、
            業務効率化とコスト削減を実現できます。アジャイル開発手法の導入により、
            市場変化への迅速な対応が可能となりました。
        """,

        "ニュース記事": """
            人工知能スタートアップのOpenAIが、最新の言語モデルGPT-4を発表しました。
            このモデルは、従来のGPT-3.5と比較して推論能力が大幅に向上し、
            複雑なタスクの処理が可能になりました。Microsoft社との提携により、
            Azure上でのサービス展開も予定されています。
        """,

        "医療文書": """
            新型コロナウイルス感染症（COVID-19）の診断において、PCR検査と
            抗原検査が主要な検査方法となっています。ワクチン接種により、
            重症化リスクの低減と集団免疫の獲得が期待されます。変異株の出現により、
            ブースター接種の重要性が高まっています。
        """
    }

    extractor = KeywordExtractor(prefer_mecab=True)

    for doc_type, text in test_texts.items():
        print(f"\n【{doc_type}】")
        print("-" * 40)
        keywords = extractor.extract(text, top_n=5, use_scoring=True)
        print("  キーワード: " + " | ".join(keywords))


def demonstrate_custom_extraction() -> None:
    """カスタマイズされた抽出のデモンストレーション"""
    print("\n" + "=" * 80)
    print("5. カスタム設定での抽出")
    print("=" * 80)

    example_str = """
    量子コンピュータは、量子ビット（キュービット）を用いて計算を行います。
    量子もつれと量子重ね合わせの原理により、従来のコンピュータでは
    不可能な並列計算が実現可能です。量子アルゴリズムの開発により、
    暗号解読や創薬分野での応用が期待されています。量子エラー訂正技術の
    向上により、実用的な量子コンピュータの実現が近づいています。
    """

    print(f"\n入力テキスト:\n{example_str}\n")

    # 異なるtop_n設定での抽出
    extractor = KeywordExtractor(prefer_mecab=True)

    for top_n in [3, 5, 10, 15]:
        keywords = extractor.extract(example_str, top_n=top_n, use_scoring=True)
        print(f"\nTop-{top_n:2d}: {', '.join(keywords)}")

    # スコア詳細の表示
    print("\n【スコア詳細分析】")
    results = extractor.extract_with_details(example_str, top_n=10)

    if '統合版' in results:
        print("\n統合版のスコア内訳:")
        for keyword, score in results['統合版'][:5]:
            # スコア計算の詳細を取得
            freq = example_str.count(keyword)
            length = len(keyword)
            print(f"\n  {keyword}:")
            print(f"    総合スコア: {score:.3f}")
            print(f"    出現回数: {freq}回")
            print(f"    文字数: {length}文字")

            # 文字種判定
            import re
            if re.match(r'^[ァ-ヴー]{3,}$', keyword):
                char_type = "カタカナ"
            elif re.match(r'^[A-Z]{2,}$', keyword):
                char_type = "英大文字"
            elif re.match(r'^[一-龥]{4,}$', keyword):
                char_type = "漢字"
            else:
                char_type = "混合"
            print(f"    文字種: {char_type}")


def export_results_to_json(example_str: str, output_file: str = "keyword_extraction_results.json") -> None:
    """結果をJSON形式でエクスポート"""
    print("\n" + "=" * 80)
    print("6. 結果のエクスポート")
    print("=" * 80)

    extractor = KeywordExtractor(prefer_mecab=True)

    # 全手法での抽出結果を取得
    results = extractor.extract_with_details(example_str, top_n=10)

    # JSON用にデータを整形
    export_data = {
        "input_text": example_str,
        "extraction_results": {}
    }

    for method, keywords_scores in results.items():
        export_data["extraction_results"][method] = [
            {"keyword": kw, "score": float(score)}
            for kw, score in keywords_scores
        ]

    # ファイルに保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    print(f"結果を {output_file} に保存しました")

    # 保存内容のサマリー表示
    print("\n【保存された内容のサマリー】")
    for method, keywords_list in export_data["extraction_results"].items():
        print(f"  {method}: {len(keywords_list)}個のキーワード")
        top_3 = [item["keyword"] for item in keywords_list[:3]]
        print(f"    Top-3: {', '.join(top_3)}")


def main():
    """メイン実行関数"""

    # サンプルテキスト
    example_str = """
    人工知能（AI）は、機械学習と深層学習を基盤として急速に発展しています。
    特に自然言語処理（NLP）の分野では、トランスフォーマーモデルが革命的な成果を上げました。
    BERTやGPTなどの大規模言語モデルは、文脈理解能力を大幅に向上させています。
    画像認識の分野では、CNNが主流でしたが、最近ではVision Transformerも注目されています。
    AIの応用は医療診断から自動運転まで幅広く、社会に大きな影響を与えています。
    しかし、AIの倫理的な課題やバイアスの問題も重要な議論となっています。
    """

    # 1. 基本的な抽出
    demonstrate_basic_extraction(example_str)

    # 2. 詳細情報付き抽出
    demonstrate_detailed_extraction(example_str)

    # 3. 異なるタイプのテキストでの抽出
    demonstrate_different_texts()

    # 4. カスタム設定での抽出
    demonstrate_custom_extraction()

    # 5. 結果のエクスポート
    export_results_to_json(example_str)

    print("\n" + "=" * 80)
    print("すべての処理が完了しました")
    print("=" * 80)


if __name__ == "__main__":
    main()