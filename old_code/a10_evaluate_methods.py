#!/usr/bin/env python3
"""
キーワード抽出手法の比較評価スクリプト
異なる特性のテキストで各手法の性能を検証
python a10_evaluate_methods.py
"""

from regex_mecab import KeywordExtractor
from typing import Dict, List, Tuple
import re


def create_test_cases() -> Dict[str, str]:
    """評価用の多様なテストケースを作成"""

    test_cases = {
        # ケース1: 複合名詞が多い技術文書
        "複合名詞優位": """
            自然言語処理技術、機械学習アルゴリズム、深層学習モデル、
            ニューラルネットワーク構造、畳み込みニューラルネットワーク、
            リカレントニューラルネットワーク、敵対的生成ネットワーク、
            ベイズ最適化手法、強化学習エージェント、転移学習技術
        """,

        # ケース2: 単一名詞が多い文書
        "単一名詞優位": """
            データを分析する。モデルを構築する。アルゴリズムを改善する。
            パラメータを調整する。精度を向上させる。コストを削減する。
            システムを最適化する。プロセスを自動化する。結果を評価する。
            レポートを作成する。
        """,

        # ケース3: カタカナ語中心
        "カタカナ語中心": """
            クラウドコンピューティング、ビッグデータアナリティクス、
            ブロックチェーンテクノロジー、サイバーセキュリティ、
            デジタルトランスフォーメーション、アジャイルメソドロジー、
            マイクロサービスアーキテクチャ、コンテナオーケストレーション
        """,

        # ケース4: 英語略語中心
        "英語略語中心": """
            AI、ML、DL、NLP、CV、GAN、CNN、RNN、LSTM、BERT、
            GPT、API、SDK、IDE、CI、CD、DevOps、SaaS、PaaS、IaaS、
            IoT、AR、VR、MR、XR、5G、WiFi、HTTP、REST、JSON
        """,

        # ケース5: 日本語一般文書
        "日本語一般文書": """
            日本の伝統文化は、長い歴史の中で培われてきました。
            茶道、華道、書道などの芸道は、精神性を重視します。
            和食は、季節の食材を大切にし、美しい盛り付けが特徴です。
            着物は、日本の美意識を表現する伝統的な衣装です。
            祭りや年中行事は、地域社会の絆を深める重要な役割を果たしています。
        """,

        # ケース6: 混在型（実際のブログ記事風）
        "混在型ブログ": """
            最近のAIトレンドについて解説します。ChatGPTの登場により、
            自然言語処理の分野が大きく変化しました。大規模言語モデルは、
            従来の機械学習手法とは異なり、プロンプトエンジニアリングという
            新しいスキルが重要になってきています。APIを使用することで、
            誰でも簡単にAIを活用できる時代になりました。
        """,

        # ケース7: 短文の羅列
        "短文羅列": """
            AI革命。データ分析。機械学習。深層学習。
            ビッグデータ。クラウド。IoT。5G通信。
            量子コンピュータ。ブロックチェーン。
            メタバース。NFT。Web3。DX推進。
        """,

        # ケース8: 説明的な長文
        "説明的長文": """
            人工知能とは、人間の知的能力をコンピュータ上で実現する技術であり、
            その応用範囲は非常に広く、画像認識から自然言語処理、音声認識、
            ゲームプレイング、自動運転など多岐にわたります。特に近年では、
            ディープラーニングと呼ばれる多層ニューラルネットワークを用いた
            手法が大きな成果を上げており、これまで困難とされていた問題を
            次々と解決しています。
        """,

        # ケース9: 専門用語なし日常会話
        "日常会話": """
            今日は天気が良かったので、公園に散歩に行きました。
            桜の花が満開で、とても綺麗でした。家族連れや友達同士で
            お花見を楽しんでいる人がたくさんいました。
            春の陽気に誘われて、アイスクリームを食べながら
            ベンチでのんびり過ごしました。
        """,

        # ケース10: 数字・記号混在
        "数字記号混在": """
            2024年のAI市場規模は約1500億ドルに達し、前年比25%の成長率。
            GPT-4は1750億のパラメータを持ち、GPT-3.5の10倍の性能。
            5G通信は最大20Gbpsの通信速度を実現し、4Gの100倍高速。
            量子コンピュータは2^100の計算を同時に処理可能。
        """
    }

    return test_cases


def evaluate_extraction_quality(keywords: List[str], text: str, case_type: str) -> Dict[str, float]:
    """抽出品質を評価する指標を計算"""

    metrics = {}

    # 1. カバレージ率（抽出キーワードがテキストに存在する割合）
    coverage = sum(1 for kw in keywords if kw in text) / len(keywords) if keywords else 0
    metrics['カバレージ率'] = coverage

    # 2. 多様性スコア（キーワードの文字数のばらつき）
    if keywords:
        lengths = [len(kw) for kw in keywords]
        avg_len = sum(lengths) / len(lengths)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
        metrics['多様性スコア'] = min(variance / 10, 1.0)  # 正規化
    else:
        metrics['多様性スコア'] = 0

    # 3. 専門性スコア（カタカナ・英語・漢字複合語の割合）
    if keywords:
        technical_pattern = r'^([ァ-ヴー]{3,}|[A-Z]{2,}|[一-龥]{4,})$'
        technical_ratio = sum(1 for kw in keywords if re.match(technical_pattern, kw)) / len(keywords)
        metrics['専門性スコア'] = technical_ratio
    else:
        metrics['専門性スコア'] = 0

    # 4. ケース別適合度
    case_scores = {
        "複合名詞優位": lambda kws: sum(1 for kw in kws if len(kw) >= 6) / len(kws) if kws else 0,
        "単一名詞優位": lambda kws: sum(1 for kw in kws if 2 <= len(kw) <= 4) / len(kws) if kws else 0,
        "カタカナ語中心": lambda kws: sum(1 for kw in kws if re.match(r'^[ァ-ヴー]+$', kw)) / len(kws) if kws else 0,
        "英語略語中心": lambda kws: sum(1 for kw in kws if re.match(r'^[A-Z]+$', kw)) / len(kws) if kws else 0,
        "日本語一般文書": lambda kws: sum(1 for kw in kws if re.match(r'^[ぁ-ん一-龥]+$', kw)) / len(kws) if kws else 0,
    }

    if case_type in case_scores:
        metrics['ケース適合度'] = case_scores[case_type](keywords)
    else:
        metrics['ケース適合度'] = coverage  # デフォルトはカバレージ率

    # 5. 総合スコア
    metrics['総合スコア'] = (
        metrics['カバレージ率'] * 0.3 +
        metrics['多様性スコア'] * 0.1 +
        metrics['専門性スコア'] * 0.3 +
        metrics['ケース適合度'] * 0.3
    )

    return metrics


def run_comprehensive_evaluation():
    """包括的な評価を実行"""

    print("=" * 100)
    print("キーワード抽出手法の包括的評価")
    print("=" * 100)

    test_cases = create_test_cases()
    extractor = KeywordExtractor(prefer_mecab=True)

    # 結果を格納する辞書
    all_results = {}
    method_scores = {"MeCab複合名詞": [], "正規表現": [], "統合版": []}

    for case_name, text in test_cases.items():
        print(f"\n【テストケース: {case_name}】")
        print("-" * 80)
        print(f"テキスト（冒頭50文字）: {text[:50].strip()}...")
        print()

        # 各手法で抽出
        results = extractor.extract_with_details(text, top_n=5)

        case_results = {}
        for method, keywords_scores in results.items():
            keywords = [kw for kw, _ in keywords_scores]

            # 品質評価
            metrics = evaluate_extraction_quality(keywords, text, case_name)
            case_results[method] = {
                'keywords': keywords[:5],  # 上位5件
                'metrics': metrics
            }

            # スコアを記録
            if method in method_scores:
                method_scores[method].append(metrics['総合スコア'])

            print(f"  {method}:")
            print(f"    キーワード: {', '.join(keywords[:5]) if keywords else 'なし'}")
            print(f"    総合スコア: {metrics['総合スコア']:.3f}")

        all_results[case_name] = case_results

        # 最良の手法を特定
        best_method = max(case_results.items(),
                         key=lambda x: x[1]['metrics']['総合スコア'])
        print(f"\n  🏆 最良手法: {best_method[0]} (スコア: {best_method[1]['metrics']['総合スコア']:.3f})")

    # 総合分析
    print("\n" + "=" * 100)
    print("総合分析結果")
    print("=" * 100)

    # 各手法の平均スコア
    print("\n【平均総合スコア】")
    avg_scores = {}
    for method, scores in method_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            avg_scores[method] = avg
            print(f"  {method}: {avg:.3f}")

    # 各手法が最良だったケース数
    print("\n【最良手法となった回数】")
    best_count = {"MeCab複合名詞": 0, "正規表現": 0, "統合版": 0}
    for case_name, case_results in all_results.items():
        best = max(case_results.items(),
                  key=lambda x: x[1]['metrics']['総合スコア'])
        if best[0] in best_count:
            best_count[best[0]] += 1

    for method, count in best_count.items():
        percentage = count / len(test_cases) * 100
        print(f"  {method}: {count}回 ({percentage:.1f}%)")

    # ケース別の優位性分析
    print("\n【ケース別優位性】")
    for case_name, case_results in all_results.items():
        scores = {method: results['metrics']['総合スコア']
                 for method, results in case_results.items()}
        best = max(scores.items(), key=lambda x: x[1])
        print(f"  {case_name:15s}: {best[0]} (スコア差: {best[1] - min(scores.values()):.3f})")

    # 統合版の相対的優位性
    print("\n【統合版の相対的パフォーマンス】")
    integrated_advantages = []
    for case_name, case_results in all_results.items():
        if '統合版' in case_results:
            integrated_score = case_results['統合版']['metrics']['総合スコア']
            other_scores = [r['metrics']['総合スコア']
                          for m, r in case_results.items() if m != '統合版']
            if other_scores:
                advantage = integrated_score - max(other_scores)
                integrated_advantages.append(advantage)
                if advantage > 0:
                    print(f"  {case_name}: +{advantage:.3f} (優位)")
                elif advantage < 0:
                    print(f"  {case_name}: {advantage:.3f} (劣位)")
                else:
                    print(f"  {case_name}: ±0.000 (同等)")

    if integrated_advantages:
        avg_advantage = sum(integrated_advantages) / len(integrated_advantages)
        print(f"\n  統合版の平均優位性: {avg_advantage:+.3f}")
        if avg_advantage > 0:
            print("  → 統合版は平均的に他手法より優れている")
        elif avg_advantage < 0:
            print("  → 統合版は平均的に他手法より劣る")
        else:
            print("  → 統合版は他手法と同等")

    # 結論
    print("\n" + "=" * 100)
    print("評価結論")
    print("=" * 100)

    # 最高平均スコアの手法
    if avg_scores:
        best_avg_method = max(avg_scores.items(), key=lambda x: x[1])
        print(f"\n📊 最高平均スコア: {best_avg_method[0]} ({best_avg_method[1]:.3f})")

    # 最多勝利の手法
    if best_count:
        most_wins = max(best_count.items(), key=lambda x: x[1])
        print(f"🏅 最多勝利: {most_wins[0]} ({most_wins[1]}回)")

    # 統合版の評価
    if '統合版' in avg_scores:
        integrated_rank = sorted(avg_scores.values(), reverse=True).index(avg_scores['統合版']) + 1
        print(f"\n統合版の順位: {integrated_rank}位 / {len(avg_scores)}手法中")

        # 統合版が優位なケース
        integrated_best_cases = [case for case, results in all_results.items()
                                if '統合版' in results and
                                max(results.items(), key=lambda x: x[1]['metrics']['総合スコア'])[0] == '統合版']
        if integrated_best_cases:
            print(f"\n統合版が最良のケース:")
            for case in integrated_best_cases:
                print(f"  • {case}")


if __name__ == "__main__":
    run_comprehensive_evaluation()