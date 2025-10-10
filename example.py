#
# 日本語文書のセマンティックカバレージ分析とQ/A自動生成デモンストレーション

import numpy as np
from typing import List, Dict, Tuple
import re
import os
import json
import time
from dotenv import load_dotenv

# 環境変数を読み込む
load_dotenv()

# ------------------
# サンプルデータ定義
# ------------------

# 例文：AIに関する技術文書
example_document = """
人工知能（AI）は、機械学習と深層学習を基盤として急速に発展しています。
特に自然言語処理（NLP）の分野では、トランスフォーマーモデルが革命的な成果を上げました。
BERTやGPTなどの大規模言語モデルは、文脈理解能力を大幅に向上させています。
画像認識の分野では、CNNが主流でしたが、最近ではVision Transformerも注目されています。
AIの応用は医療診断から自動運転まで幅広く、社会に大きな影響を与えています。
しかし、AIの倫理的な課題やバイアスの問題も重要な議論となっています。
"""

example_qa_pairs = [
    {
        "question": "トランスフォーマーモデルはどの分野で成果を上げていますか？",
        "answer": "自然言語処理（NLP）の分野で革命的な成果を上げています。"
    },
    {
        "question": "AIの応用分野にはどのようなものがありますか？",
        "answer": "医療診断から自動運転まで幅広い分野で応用されています。"
    },
    # 注意：画像認識やAI倫理についてのQ/Aが欠けている
]


def demonstrate_semantic_coverage():
    """セマンティックカバレージの処理を実演"""

    from a03_rag_qa_coverage import SemanticCoverage

    # インスタンス化
    analyzer = SemanticCoverage()

    print("=" * 60)
    print("1. ドキュメントのチャンク化")
    print("=" * 60)

    # ドキュメントをチャンク化
    doc_chunks = analyzer.create_semantic_chunks(example_document)

    for chunk in doc_chunks:
        print(f"\n{chunk['id']}:")
        print(f"  テキスト: {chunk['text'][:100]}...")
        print(f"  文の数: {len(chunk['sentences'])}")

    print("\n" + "=" * 60)
    print("2. 埋め込みベクトル生成")
    print("=" * 60)

    # 埋め込み生成
    doc_embeddings = analyzer.generate_embeddings(doc_chunks)
    print(f"ドキュメント埋め込み shape: {doc_embeddings.shape}")

    # Q/Aペアの埋め込み
    qa_embeddings = []
    for qa in example_qa_pairs:
        qa_text = f"{qa['question']} {qa['answer']}"
        embedding = analyzer.generate_embedding(qa_text)
        qa_embeddings.append(embedding)

    qa_embeddings = np.array(qa_embeddings)
    print(f"Q/A埋め込み shape: {qa_embeddings.shape}")

    print("\n" + "=" * 60)
    print("3. カバレージ計算")
    print("=" * 60)

    # 各チャンクに対する最大類似度を計算
    coverage_matrix = np.zeros((len(doc_chunks), len(example_qa_pairs)))

    for i in range(len(doc_embeddings)):
        for j in range(len(qa_embeddings)):
            similarity = analyzer.cosine_similarity(doc_embeddings[i], qa_embeddings[j])
            coverage_matrix[i, j] = similarity

    # 各チャンクの最大類似度
    max_similarities = coverage_matrix.max(axis=1)

    # 結果表示
    threshold = 0.7
    for i, (chunk, max_sim) in enumerate(zip(doc_chunks, max_similarities)):
        status = "✓ カバー" if max_sim > threshold else "✗ 未カバー"
        print(f"\n{chunk['id']} [{status}] (類似度: {max_sim:.3f})")
        print(f"  内容: {chunk['text'][:80]}...")

        if max_sim > threshold:
            best_qa_idx = coverage_matrix[i].argmax()
            print(f"  最も関連するQ/A: Q{best_qa_idx + 1}")

    # 全体のカバレージ率
    covered_chunks = sum(1 for s in max_similarities if s > threshold)
    coverage_rate = covered_chunks / len(doc_chunks)

    print("\n" + "=" * 60)
    print("4. カバレージサマリー")
    print("=" * 60)
    print(f"総チャンク数: {len(doc_chunks)}")
    print(f"カバーされたチャンク: {covered_chunks}")
    print(f"カバレージ率: {coverage_rate:.1%}")

    # 未カバー領域の特定
    uncovered_content = []
    for chunk, sim in zip(doc_chunks, max_similarities):
        if sim <= threshold:
            uncovered_content.append(chunk['text'])

    if uncovered_content:
        print("\n⚠️ 未カバー領域:")
        for content in uncovered_content:
            print(f"  - {content[:100]}...")

    return coverage_matrix, max_similarities


# ------------------
# 可視化による理解
# ------------------
import matplotlib.pyplot as plt
import seaborn as sns

# 日本語フォントの設定
import matplotlib
import platform

# OSに応じた日本語フォント設定
system = platform.system()
if system == 'Darwin':  # macOS
    matplotlib.rcParams['font.family'] = 'Hiragino Sans'
elif system == 'Windows':
    matplotlib.rcParams['font.family'] = 'MS Gothic'
else:  # Linux
    matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'

# マイナス記号の文字化けを防ぐ
matplotlib.rcParams['axes.unicode_minus'] = False


def visualize_semantic_coverage(coverage_matrix, doc_chunks, qa_pairs):
    """カバレージマトリックスの可視化"""

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 1. ヒートマップ
    ax1 = axes[0]
    sns.heatmap(
        coverage_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0, vmax=1,
        ax=ax1,
        xticklabels=[f"Q{i + 1}" for i in range(len(qa_pairs))],
        yticklabels=[f"Chunk{i}" for i in range(len(doc_chunks))]
    )
    ax1.set_title('セマンティック類似度マトリックス')
    ax1.set_xlabel('Q/Aペア')
    ax1.set_ylabel('ドキュメントチャンク')

    # 2. カバレージの分布
    ax2 = axes[1]
    max_similarities = coverage_matrix.max(axis=1)

    # ヒストグラム
    bars = ax2.bar(
        range(len(max_similarities)),
        max_similarities,
        color=['green' if s > 0.7 else 'red' for s in max_similarities]
    )

    ax2.axhline(y=0.7, color='blue', linestyle='--', label='閾値 (0.7)')
    ax2.set_xlabel('チャンクID')
    ax2.set_ylabel('最大類似度')
    ax2.set_title('チャンクごとのカバレージ状況')
    ax2.legend()

    plt.tight_layout()
    return fig


# ----------------
# 実行結果の解釈
# ----------------
def extract_keywords(text, top_n=5, use_mecab=False):
    """テキストから重要なキーワードを抽出

    Args:
        text: 分析対象のテキスト
        top_n: 抽出するキーワード数
        use_mecab: MeCabを使用するか（True: MeCab, False: 正規表現）

    Returns:
        重要キーワードのリスト
    """
    if use_mecab:
        try:
            # MeCab版の関数を呼び出し
            from example_mecab import extract_keywords_mecab
            return extract_keywords_mecab(text, top_n, use_compound=True)
        except ImportError:
            # MeCabが利用できない場合は正規表現版にフォールバック
            pass

    # 正規表現版（既存の実装）
    import collections

    # ストップワード（除外する一般的な語）
    stopwords = {'こと', 'もの', 'これ', 'それ', 'ため', 'よう', 'さん',
                 'ます', 'です', 'ある', 'いる', 'する', 'なる', 'できる',
                 'いう', '的', 'な', 'に', 'を', 'は', 'が', 'で', 'と', 'の', 'から', 'まで'}

    # カタカナ語、漢字複合語、英数字を抽出
    pattern = r'[ァ-ヴー]+|[一-龥]{2,}|[A-Za-z]+[A-Za-z0-9]*'
    words = re.findall(pattern, text)

    # 単語頻度をカウント（ストップワードを除外）
    word_freq = collections.Counter(
        word for word in words
        if word not in stopwords and len(word) > 1
    )

    # 頻出上位のキーワードを返す
    keywords = [word for word, freq in word_freq.most_common(top_n)]

    # キーワードが少ない場合は元の単語リストから補完
    if len(keywords) < 3:
        keywords.extend([w for w in words if w not in keywords][:3-len(keywords)])

    return keywords


def calculate_priority(uncovered_chunks):
    """未カバーチャンクの優先度を計算

    Args:
        uncovered_chunks: 未カバーのテキストチャンクのリスト

    Returns:
        (チャンク, 優先度スコア)のタプルのリスト（降順）
    """
    priority_scores = []

    for chunk in uncovered_chunks:
        score = 0.0

        # 1. 文字数による重要度（長いチャンクは情報量が多い）
        length_score = min(len(chunk) / 200, 1.0)  # 200文字を基準に正規化
        score += length_score * 0.3

        # 2. 専門用語の密度（カタカナ語、英語、漢字複合語）
        technical_pattern = r'[ァ-ヴー]+|[A-Z][a-z]*|[一-龥]{3,}'
        technical_terms = re.findall(technical_pattern, chunk)
        term_density = len(technical_terms) / max(len(chunk.split()), 1)
        score += min(term_density * 10, 1.0) * 0.4  # 専門用語密度を重視

        # 3. 数字や記号の存在（具体的な情報を含む可能性）
        has_numbers = bool(re.search(r'\d+', chunk))
        has_symbols = bool(re.search(r'[％%＄$€¥]', chunk))
        if has_numbers:
            score += 0.15
        if has_symbols:
            score += 0.15

        # 4. キーワードの重要度（特定の重要語を含むか）
        important_keywords = ['AI', '人工知能', '機械学習', '深層学習', 'ディープラーニング',
                            '医療', '診断', '自動運転', '倫理', 'バイアス', '課題', '問題',
                            'トランスフォーマー', 'CNN', 'NLP', 'BERT', 'GPT']

        keyword_matches = sum(1 for kw in important_keywords if kw in chunk)
        keyword_score = min(keyword_matches / 3, 1.0)  # 3つ以上で満点
        score += keyword_score * 0.3

        # 最終スコアを0-100の範囲に調整
        final_score = min(score * 100, 100)
        priority_scores.append((chunk, final_score))

    # スコアの降順でソート
    priority_scores.sort(key=lambda x: x[1], reverse=True)

    return priority_scores


def interpret_results(coverage_rate, uncovered_chunks):
    """結果の解釈とアクション提案"""

    interpretations = {
        "excellent": (0.8, "優秀：主要な内容は網羅されています"),
        "good"     : (0.6, "良好：基本的な内容はカバーされています"),
        "fair"     : (0.4, "改善必要：重要な領域が未カバーです"),
        "poor"     : (0.0, "要対策：大幅なQ/A追加が必要です")
    }

    for level, (threshold, message) in interpretations.items():
        if coverage_rate >= threshold:
            print(f"評価: {message}")
            break

    # 具体的なアクション提案
    if uncovered_chunks:
        print("\n推奨アクション:")
        print("1. 以下の領域に関するQ/A追加を検討:")
        for chunk in uncovered_chunks[:3]:  # 上位3つ
            keywords = extract_keywords(chunk)
            print(f"   - {', '.join(keywords)}")

        print("\n2. Q/A生成の優先順位:")
        priority_scores = calculate_priority(uncovered_chunks)
        for i, (chunk, score) in enumerate(priority_scores[:3]):
            print(f"   {i + 1}. スコア{score:.2f}: {chunk[:50]}...")


def main():
    """メイン関数：セマンティックカバレージ分析の実行と改善提案"""

    # APIキーの確認
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-openai-api-key-here":
        print("⚠️  警告: OPENAI_API_KEYが正しく設定されていません。")
        print(".envファイルまたは環境変数を確認してください。")
        return

    print("=" * 60)
    print("セマンティックカバレージ分析デモンストレーション")
    print("=" * 60)

    # 1. カバレージ分析の実行
    try:
        coverage_matrix, max_similarities = demonstrate_semantic_coverage()
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 未カバー領域の特定
    from a03_rag_qa_coverage import SemanticCoverage
    analyzer = SemanticCoverage()
    doc_chunks = analyzer.create_semantic_chunks(example_document)

    threshold = 0.7
    uncovered_chunks = []
    for chunk, sim in zip(doc_chunks, max_similarities):
        if sim <= threshold:
            uncovered_chunks.append(chunk['text'])

    # 3. 結果の解釈
    coverage_rate = sum(1 for s in max_similarities if s > threshold) / len(doc_chunks)

    print("\n" + "=" * 60)
    print("5. 分析結果と改善提案")
    print("=" * 60)

    interpret_results(coverage_rate, uncovered_chunks)

    # 4. 可視化（オプション）
    user_input = input("\n結果を可視化しますか？ (y/n): ")
    if user_input.lower() == 'y':
        try:
            fig = visualize_semantic_coverage(coverage_matrix, doc_chunks, example_qa_pairs)
            plt.show()
            print("✅ 可視化が完了しました")
        except Exception as e:
            print(f"可視化エラー: {e}")

    # 5. 主要トピックベースの改善提案
    if uncovered_chunks:
        print("\n" + "=" * 60)
        print("6. 主要トピックベースの改善提案")
        print("=" * 60)

        print("\n未カバーチャンクの主要トピック:")
        for i, chunk in enumerate(uncovered_chunks, 1):
            keywords = extract_keywords(chunk, top_n=5, use_mecab=False)
            print(f"\nチャンク {i}:")
            print(f"  テキスト: {chunk[:80]}...")
            print(f"  主要トピック: {', '.join(keywords)}")
            print(f"  推奨Q/A:")

            # トピックに基づくQ/A提案
            for keyword in keywords[:3]:
                if keyword in ['CNN', 'Vision Transformer', '画像認識']:
                    print(f"    Q: {keyword}はどのような技術ですか？")
                    print(f"    A: {keyword}は画像認識分野で使用される...")
                elif keyword in ['倫理', 'バイアス', '課題', '問題']:
                    print(f"    Q: AIの{keyword}についてどのような議論がありますか？")
                    print(f"    A: AIの{keyword}として...")
                else:
                    print(f"    Q: {keyword}について説明してください。")
                    print(f"    A: {keyword}は...")

    print("\n" + "=" * 60)
    print("分析が完了しました。")
    print("=" * 60)


if __name__ == "__main__":
    main()