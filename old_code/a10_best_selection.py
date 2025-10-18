#!/usr/bin/env python3
"""
キーワード抽出の最良選択システム
3つの手法（MeCab、正規表現、統合版）をすべて実行し、
最良の結果を自動選択して採用する
python a10_best_selection.py
"""

from regex_mecab import KeywordExtractor
from typing import List, Dict, Tuple, Optional
import re
import json


class BestKeywordSelector:
    """3手法から最良のキーワードを選択するクラス"""

    def __init__(self, prefer_mecab: bool = True):
        """
        Args:
            prefer_mecab: MeCabを優先的に使用するか
        """
        self.extractor = KeywordExtractor(prefer_mecab=prefer_mecab)

        # 評価重み付け（調整可能）
        self.weights = {
            'coverage': 0.25,      # カバレージ率
            'diversity': 0.15,     # 多様性
            'technicality': 0.25,  # 専門性
            'coherence': 0.20,     # 一貫性
            'length_balance': 0.15 # 長さのバランス
        }

    def evaluate_keywords(self, keywords: List[str], text: str) -> Dict[str, float]:
        """
        キーワードセットの品質を多面的に評価
        Args:
            keywords: 評価対象のキーワードリスト
            text: 元のテキスト
        Returns:
            評価指標の辞書
        """
        if not keywords:
            return {metric: 0.0 for metric in self.weights.keys()}

        metrics = {}

        # 1. カバレージ率（キーワードがテキストに存在する割合）
        coverage_count = sum(1 for kw in keywords if kw in text)
        metrics['coverage'] = coverage_count / len(keywords)

        # 2. 多様性（文字数の分散）
        lengths = [len(kw) for kw in keywords]
        avg_len = sum(lengths) / len(lengths)
        if len(lengths) > 1:
            variance = sum((l - avg_len) ** 2 for l in lengths) / (len(lengths) - 1)
            # 適度な分散を評価（標準偏差2-4文字が理想）
            std_dev = variance ** 0.5
            metrics['diversity'] = min(1.0, (std_dev / 3.0) if std_dev < 3 else (6 - std_dev) / 3.0)
        else:
            metrics['diversity'] = 0.5

        # 3. 専門性（カタカナ・英語・漢字複合語の割合）
        technical_patterns = [
            (r'^[ァ-ヴー]{3,}$', 1.0),      # カタカナ3文字以上
            (r'^[A-Z]{2,}[A-Z0-9]*$', 1.2), # 英大文字（略語）
            (r'^[一-龥]{4,}$', 0.9),        # 漢字4文字以上
            (r'^[A-Za-z]+[A-Za-z0-9]*$', 0.8) # 英単語
        ]

        tech_score = 0
        for kw in keywords:
            kw_tech = 0
            for pattern, weight in technical_patterns:
                if re.match(pattern, kw):
                    kw_tech = max(kw_tech, weight)
            tech_score += kw_tech
        metrics['technicality'] = min(1.0, tech_score / len(keywords))

        # 4. 一貫性（キーワード間の関連性）
        # 同じ文字を含むキーワードのペア数で評価
        coherence_score = 0
        for i, kw1 in enumerate(keywords):
            for kw2 in keywords[i+1:]:
                # 部分文字列の共有
                if len(kw1) >= 2 and len(kw2) >= 2:
                    if any(sub in kw2 for sub in [kw1[i:i+2] for i in range(len(kw1)-1)]):
                        coherence_score += 1
        max_pairs = len(keywords) * (len(keywords) - 1) / 2
        metrics['coherence'] = coherence_score / max_pairs if max_pairs > 0 else 0

        # 5. 長さのバランス（2-8文字が理想）
        ideal_length_ratio = sum(1 for kw in keywords if 2 <= len(kw) <= 8) / len(keywords)
        metrics['length_balance'] = ideal_length_ratio

        return metrics

    def calculate_total_score(self, metrics: Dict[str, float]) -> float:
        """
        評価指標から総合スコアを計算
        Args:
            metrics: 各評価指標の辞書
        Returns:
            総合スコア（0.0-1.0）
        """
        total = sum(metrics.get(metric, 0) * weight
                   for metric, weight in self.weights.items())
        return min(1.0, total)

    def extract_best(self, text: str, top_n: int = 10,
                     return_details: bool = False) -> Dict[str, any]:
        """
        3つの手法で抽出し、最良の結果を選択
        Args:
            text: 分析対象テキスト
            top_n: 抽出するキーワード数
            return_details: 詳細情報を返すか
        Returns:
            最良のキーワードと選択理由
        """
        # 各手法で抽出
        all_results = self.extractor.extract_with_details(text, top_n)

        # 各手法の評価
        evaluations = {}
        for method, keywords_scores in all_results.items():
            keywords = [kw for kw, _ in keywords_scores[:top_n]]

            # 評価指標を計算
            metrics = self.evaluate_keywords(keywords, text)
            total_score = self.calculate_total_score(metrics)

            evaluations[method] = {
                'keywords': keywords,
                'metrics': metrics,
                'total_score': total_score,
                'keyword_scores': keywords_scores[:top_n]
            }

        # 最良の手法を選択
        best_method = max(evaluations.items(),
                         key=lambda x: x[1]['total_score'])

        result = {
            'best_method': best_method[0],
            'keywords': best_method[1]['keywords'],
            'total_score': best_method[1]['total_score'],
            'reason': self._generate_reason(best_method[0], evaluations)
        }

        if return_details:
            result['all_evaluations'] = evaluations

        return result

    def _generate_reason(self, best_method: str,
                        evaluations: Dict[str, Dict]) -> str:
        """選択理由を生成"""
        best_eval = evaluations[best_method]
        metrics = best_eval['metrics']

        # 最も優れた指標を特定
        best_metric = max(metrics.items(), key=lambda x: x[1])

        reasons = {
            'coverage': 'テキストカバレージが最も高い',
            'diversity': 'キーワードの多様性が優れている',
            'technicality': '専門用語の抽出精度が高い',
            'coherence': 'キーワード間の一貫性が優れている',
            'length_balance': 'キーワード長のバランスが良い'
        }

        return f"{reasons.get(best_metric[0], '総合的に優れている')} (スコア: {best_eval['total_score']:.3f})"

    def compare_methods_visual(self, text: str, top_n: int = 10) -> None:
        """
        3手法の比較を視覚的に表示

        Args:
            text: 分析対象テキスト
            top_n: 抽出するキーワード数
        """
        result = self.extract_best(text, top_n, return_details=True)

        print("=" * 80)
        print("3手法の比較と最良選択")
        print("=" * 80)
        print(f"\n入力テキスト（冒頭100文字）:\n{text[:100]}...\n")

        # 各手法の結果を表示
        for method, evaluation in result['all_evaluations'].items():
            is_best = (method == result['best_method'])
            mark = "🏆 " if is_best else "   "

            print(f"\n{mark}【{method}】")
            print("-" * 60)

            # キーワード表示
            print("  キーワード:")
            for i, kw in enumerate(evaluation['keywords'][:5], 1):
                print(f"    {i}. {kw}")

            # 評価指標を表示
            print("\n  評価指標:")
            for metric, value in evaluation['metrics'].items():
                bar_length = int(value * 15)
                bar = '█' * bar_length + '░' * (15 - bar_length)
                metric_name = {
                    'coverage': 'カバレージ',
                    'diversity': '多様性',
                    'technicality': '専門性',
                    'coherence': '一貫性',
                    'length_balance': '長さバランス'
                }.get(metric, metric)
                print(f"    {metric_name:10s}: [{bar}] {value:.3f}")

            print(f"\n  総合スコア: {evaluation['total_score']:.3f}")

        # 最終選択
        print("\n" + "=" * 80)
        print(f"✅ 選択された手法: {result['best_method']}")
        print(f"   理由: {result['reason']}")
        print("=" * 80)


def demonstrate_best_selection():
    """最良選択システムのデモンストレーション"""

    # テストケース
    test_cases = {
        "AI技術文書": """
            人工知能（AI）は、機械学習と深層学習を基盤として急速に発展しています。
            特に自然言語処理（NLP）の分野では、トランスフォーマーモデルが革命的な成果を上げました。
            BERTやGPTなどの大規模言語モデルは、文脈理解能力を大幅に向上させています。
        """,

        "ビジネス文書": """
            デジタルトランスフォーメーション（DX）により、企業の業務プロセスが大きく変化しています。
            クラウドコンピューティングとビッグデータ分析を活用し、リアルタイムな意思決定が可能になりました。
            アジャイル開発手法とDevOpsの導入により、開発スピードが飛躍的に向上しています。
        """,

        "医療技術文書": """
            遺伝子編集技術CRISPRは、遺伝性疾患の治療に革命をもたらしています。
            ゲノムシークエンシングのコスト低下により、個別化医療が現実的になってきました。
            バイオインフォマティクスとAIの融合により、創薬プロセスが加速しています。
        """,

        "量子コンピューティング": """
            量子コンピュータは量子ビット（キュービット）を使用し、量子もつれと重ね合わせの原理を利用します。
            量子アルゴリズムにより、従来のコンピュータでは不可能な問題を解決できます。
            量子エラー訂正と量子優位性の実現が、実用化への鍵となっています。
        """
    }

    selector = BestKeywordSelector()

    # 各テストケースで最良選択を実行
    for case_name, text in test_cases.items():
        print(f"\n\n{'#' * 80}")
        print(f"テストケース: {case_name}")
        print('#' * 80)

        selector.compare_methods_visual(text, top_n=10)

    # 統計サマリー
    print("\n\n" + "=" * 80)
    print("最良選択システムの利点")
    print("=" * 80)
    print("""
1. 自動的に最適な手法を選択
2. テキストの特性に応じた適応的な抽出
3. 多面的な評価による品質保証
4. 透明性のある選択理由の提示
5. 各手法の長所を活かした結果

推奨使用方法:
- 重要な文書: 最良選択システムを使用
- 大量処理: 事前にテキストタイプを分類し、適切な手法を選択
- 探索的分析: 全手法の結果を比較検討
    """)


def export_best_results(text: str, output_file: str = "best_keywords.json"):
    """最良の結果をJSON形式でエクスポート"""

    selector = BestKeywordSelector()
    result = selector.extract_best(text, top_n=15, return_details=True)

    # エクスポート用データ構造
    export_data = {
        "input_text": text[:500] + "..." if len(text) > 500 else text,
        "best_method": result['best_method'],
        "keywords": result['keywords'],
        "total_score": result['total_score'],
        "selection_reason": result['reason'],
        "all_methods_comparison": {}
    }

    # 各手法の詳細を追加
    for method, eval_data in result['all_evaluations'].items():
        export_data["all_methods_comparison"][method] = {
            "keywords": eval_data['keywords'],
            "total_score": eval_data['total_score'],
            "metrics": eval_data['metrics']
        }

    # ファイルに保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 最良の結果を {output_file} に保存しました")
    return export_data


def main():
    """メイン実行関数"""

    # デモンストレーション
    demonstrate_best_selection()

    # サンプルテキストで最良結果をエクスポート
    sample_text = """
        最新のAI技術により、自然言語処理、画像認識、音声認識などの分野で
        ブレークスルーが起きています。特にTransformerアーキテクチャを
        基盤としたGPT-4やClaude 3などの大規模言語モデルは、
        人間レベルの文章生成能力を実現しています。
    """

    export_best_results(sample_text, "best_keywords.json")


if __name__ == "__main__":
    main()