#!/usr/bin/env python3
"""
スマートキーワード選択システム（ハイブリッド方式）
テキスト特性に応じて最適なtop_nを自動決定し、
3手法から最良の結果を選択する統合システム
python a10_smart_selection.py
"""

from regex_mecab import KeywordExtractor
from a10_best_selection import BestKeywordSelector
from typing import List, Dict, Tuple, Optional
import re
import math
import json


class SmartKeywordSelector(BestKeywordSelector):
    """テキスト特性に応じた最適なキーワード抽出を行うクラス"""

    def __init__(self, prefer_mecab: bool = True):
        super().__init__(prefer_mecab=prefer_mecab)

        # モード別のデフォルトtop_n
        self.mode_defaults = {
            "summary": 5,        # 要約・概要把握用
            "standard": 10,      # 標準的な分析
            "detailed": 15,      # 詳細分析
            "exhaustive": 20,    # 網羅的抽出
            "tag": 3,           # タグ付け用
        }

    def calculate_auto_top_n(self, text: str) -> Tuple[int, str]:
        """
        テキスト特性から最適なtop_nを自動計算

        Returns:
            (top_n, 判定理由)
        """
        # 基本メトリクス
        text_length = len(text)
        sentences = len(re.split(r'[。！？\.\!\?]+', text))

        # 専門用語の密度を推定（カタカナ・英字・漢字複合語）
        technical_pattern = r'[ァ-ヴー]{3,}|[A-Z]{2,}|[一-龥]{4,}'
        technical_terms = len(re.findall(technical_pattern, text))
        technical_density = technical_terms / (text_length / 100) if text_length > 0 else 0

        # ルールベースの決定
        if text_length < 100:
            return 3, f"超短文（{text_length}文字）"
        elif text_length < 300:
            return 5, f"短文（{text_length}文字）"
        elif text_length < 500:
            base = 7
            if technical_density > 2:
                base += 2  # 専門用語が多い場合は増やす
            return base, f"中文（{text_length}文字、専門用語密度: {technical_density:.1f}）"
        elif text_length < 1000:
            base = 10
            if sentences > 10:
                base += 2  # 文が多い場合は増やす
            return base, f"標準文（{text_length}文字、{sentences}文）"
        elif text_length < 2000:
            return 15, f"長文（{text_length}文字）"
        else:
            # 非常に長い文書は対数的に増加
            extra = int(math.log(text_length / 2000, 2))
            return min(20 + extra, 30), f"超長文（{text_length}文字）"

    def find_optimal_by_coverage(
        self,
        text: str,
        target_coverage: float = 0.7,
        min_n: int = 3,
        max_n: int = 20
    ) -> Tuple[int, float]:
        """
        目標カバレッジを達成する最小のtop_nを探索

        Returns:
            (最適なtop_n, 達成カバレッジ率)
        """
        best_n = min_n
        best_coverage = 0

        for n in range(min_n, max_n + 1):
            # 3手法で抽出して最良を選択
            result = self.extract_best(text, n, return_details=False)
            keywords = result['keywords']

            # カバレッジ計算（キーワードがカバーする文字数の割合）
            covered_chars = 0
            for keyword in keywords:
                occurrences = text.count(keyword)
                covered_chars += len(keyword) * occurrences

            coverage = covered_chars / len(text) if len(text) > 0 else 0

            if coverage >= target_coverage:
                return n, coverage

            best_coverage = coverage
            best_n = n

        # 目標に届かない場合は最大値を返す
        return best_n, best_coverage

    def find_optimal_by_diminishing_returns(
        self,
        text: str,
        min_n: int = 3,
        max_n: int = 20,
        threshold: float = 0.05
    ) -> Tuple[int, float, List[float]]:
        """
        収穫逓減の法則に基づいて最適なtop_nを決定

        Args:
            threshold: 改善率がこの値以下になったら停止

        Returns:
            (最適なtop_n, 最終スコア, 各nでのスコアリスト)
        """
        scores = []
        previous_score = 0
        optimal_n = min_n

        for n in range(min_n, max_n + 1):
            result = self.extract_best(text, n, return_details=False)
            current_score = result['total_score']
            scores.append(current_score)

            if n > min_n:
                improvement = current_score - previous_score
                improvement_rate = improvement / previous_score if previous_score > 0 else 1

                # 改善が閾値以下なら前の値が最適
                if improvement_rate < threshold:
                    optimal_n = n - 1
                    break

            previous_score = current_score
            optimal_n = n

        return optimal_n, scores[optimal_n - min_n], scores

    def extract_best_auto(
        self,
        text: str,
        mode: str = "auto",
        min_keywords: int = 3,
        max_keywords: int = 20,
        target_coverage: float = 0.7,
        return_analysis: bool = False
    ) -> Dict[str, any]:
        """
        自動的に最適なtop_nを決定してキーワード抽出

        Args:
            text: 分析対象テキスト
            mode: 抽出モード
                - "auto": テキスト長に基づく自動決定
                - "summary": 要約用（5個）
                - "standard": 標準（10個）
                - "detailed": 詳細（15個）
                - "coverage": カバレッジベース
                - "diminishing": 収穫逓減ベース
            min_keywords: 最小キーワード数
            max_keywords: 最大キーワード数
            target_coverage: カバレッジモードでの目標率
            return_analysis: 分析詳細を返すか

        Returns:
            抽出結果と決定根拠
        """
        analysis = {
            "mode": mode,
            "text_length": len(text),
            "sentence_count": len(re.split(r'[。！？\.\!\?]+', text))
        }

        # モードに応じてtop_nを決定
        if mode == "auto":
            top_n, reason = self.calculate_auto_top_n(text)
            analysis["decision_reason"] = reason

        elif mode in self.mode_defaults:
            top_n = self.mode_defaults[mode]
            analysis["decision_reason"] = f"固定値モード: {mode}"

        elif mode == "coverage":
            top_n, achieved_coverage = self.find_optimal_by_coverage(
                text, target_coverage, min_keywords, max_keywords
            )
            analysis["decision_reason"] = f"カバレッジ {achieved_coverage:.1%} 達成"
            analysis["target_coverage"] = target_coverage
            analysis["achieved_coverage"] = achieved_coverage

        elif mode == "diminishing":
            top_n, final_score, score_progression = self.find_optimal_by_diminishing_returns(
                text, min_keywords, max_keywords
            )
            analysis["decision_reason"] = f"収穫逓減点: n={top_n}"
            analysis["score_progression"] = score_progression
            analysis["final_score"] = final_score

        else:
            top_n = 10
            analysis["decision_reason"] = "デフォルト値"

        # 範囲制限
        top_n = max(min_keywords, min(max_keywords, top_n))
        analysis["selected_top_n"] = top_n

        # 最良の手法でキーワード抽出
        result = self.extract_best(text, top_n, return_details=True)

        # 結果に分析情報を追加
        result["optimization"] = analysis

        if not return_analysis:
            # 簡潔な結果のみ返す
            return {
                "keywords": result["keywords"],
                "method": result["best_method"],
                "top_n": top_n,
                "mode": mode,
                "reason": analysis["decision_reason"]
            }

        return result


def demonstrate_smart_selection():
    """スマート選択システムのデモンストレーション"""

    print("=" * 80)
    print("スマートキーワード選択システム - デモンストレーション")
    print("=" * 80)

    # 様々な長さのテストケース
    test_cases = {
        "超短文（SNS投稿）": "AIが医療診断を革新。精度95%達成！",

        "短文（ニュースヘッドライン）": """
            OpenAIが最新モデルGPT-5を発表。従来モデルの10倍の性能向上を実現。
            企業向けAPIも同時リリース。
        """,

        "中文（要約）": """
            機械学習の分野では、深層学習が主流となっています。
            特にTransformerアーキテクチャは自然言語処理に革命をもたらしました。
            BERTやGPTなどのモデルが次々と登場し、人間レベルの言語理解を実現しつつあります。
            画像認識分野でもVision Transformerが注目を集めています。
        """,

        "長文（技術記事）": """
            量子コンピュータは、従来のコンピュータとは根本的に異なる原理で動作します。
            量子ビット（キュービット）は、0と1の重ね合わせ状態を取ることができ、
            これにより並列計算が可能になります。量子もつれという現象を利用することで、
            複数のキュービット間で情報を瞬時に共有できます。

            現在、IBMやGoogleが量子コンピュータの開発を進めており、
            量子優位性の実証に成功しています。特に、素因数分解や最適化問題において、
            従来のコンピュータを大きく上回る性能を示しています。

            しかし、量子エラーやデコヒーレンスといった課題も残されており、
            実用化にはまだ時間がかかると予想されています。量子エラー訂正技術の
            発展が、今後の鍵となるでしょう。
        """
    }

    selector = SmartKeywordSelector()

    # 各モードでテスト
    modes = ["auto", "summary", "coverage", "diminishing"]

    for case_name, text in test_cases.items():
        print(f"\n\n【テストケース: {case_name}】")
        print("-" * 80)
        print(f"テキスト長: {len(text)}文字")
        print()

        for mode in modes:
            result = selector.extract_best_auto(
                text,
                mode=mode,
                target_coverage=0.6,
                return_analysis=False
            )

            print(f"\n  ◆ モード: {mode}")
            print(f"    決定されたtop_n: {result['top_n']}")
            print(f"    選択理由: {result['reason']}")
            print(f"    選択手法: {result['method']}")
            print(f"    キーワード: {', '.join(result['keywords'][:5])}")
            if len(result['keywords']) > 5:
                print(f"               ... 他{len(result['keywords'])-5}個")


def analyze_optimization_effect():
    """最適化効果の分析"""

    print("\n\n" + "=" * 80)
    print("最適化効果の分析")
    print("=" * 80)

    sample_text = """
        人工知能技術の発展により、様々な産業で自動化が進んでいます。
        特に製造業では、品質検査の自動化により不良品率が大幅に低下しました。
        また、物流業界では、AIによる配送ルート最適化で効率が30%向上しています。
        医療分野でも、画像診断AIが医師の診断を支援し、早期発見率が向上しています。
        金融業界では、不正検知システムの精度が向上し、被害額が減少しています。
    """

    selector = SmartKeywordSelector()

    print("\n【収穫逓減分析】")
    print("-" * 60)

    # 各top_nでのスコアを計算
    scores_by_n = []
    for n in range(3, 16):
        result = selector.extract_best(sample_text, n, return_details=False)
        scores_by_n.append((n, result['total_score']))

    # グラフ風の表示
    print("\n  top_n  スコア   改善率   グラフ")
    print("  " + "-" * 50)

    prev_score = 0
    for n, score in scores_by_n:
        improvement = score - prev_score if prev_score > 0 else score
        improvement_rate = (improvement / prev_score * 100) if prev_score > 0 else 100

        # バーグラフ
        bar_length = int(score * 30)
        bar = '█' * bar_length

        # 改善率の表示
        if prev_score > 0:
            imp_str = f"+{improvement_rate:5.1f}%"
        else:
            imp_str = "  ---  "

        print(f"  {n:5d}  {score:.3f}  {imp_str}  {bar}")

        # 収穫逓減点の判定
        if prev_score > 0 and improvement_rate < 5:
            print(f"         ↑ 収穫逓減点（改善率 {improvement_rate:.1f}% < 5%）")
            break

        prev_score = score

    # 最適値での結果
    optimal_result = selector.extract_best_auto(
        sample_text,
        mode="diminishing",
        return_analysis=False
    )

    print(f"\n  → 最適なtop_n: {optimal_result['top_n']}")
    print(f"     キーワード: {', '.join(optimal_result['keywords'])}")


def main():
    """メイン実行関数"""

    # スマート選択のデモ
    demonstrate_smart_selection()

    # 最適化効果の分析
    analyze_optimization_effect()

    # 実用例
    print("\n\n" + "=" * 80)
    print("実用例：Q&A生成用のキーワード抽出")
    print("=" * 80)

    qa_text = """
        ブロックチェーン技術は、分散型台帳技術として知られ、
        取引記録を複数のコンピュータで共有・管理します。
        各ブロックには取引データとハッシュ値が含まれ、
        チェーン状に連結されています。これにより、
        データの改ざんが極めて困難になります。
        ビットコインなどの暗号通貨だけでなく、
        サプライチェーン管理や医療記録管理など、
        様々な分野での応用が期待されています。
    """

    selector = SmartKeywordSelector()

    # Q&A生成には「auto」モードが最適
    result = selector.extract_best_auto(qa_text, mode="auto")

    print(f"\n自動決定されたtop_n: {result['top_n']}")
    print(f"選択理由: {result['reason']}")
    print(f"選択された手法: {result['method']}")
    print("\n抽出されたキーワード:")
    for i, keyword in enumerate(result['keywords'], 1):
        print(f"  {i:2d}. {keyword}")

    print("\n" + "=" * 80)
    print("スマート選択システムの利点:")
    print("- テキスト特性に応じた自動最適化")
    print("- 複数の最適化戦略から選択可能")
    print("- 収穫逓減を考慮した効率的な抽出")
    print("- 3手法から最良を選択する品質保証")
    print("=" * 80)


if __name__ == "__main__":
    main()