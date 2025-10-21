# python make_qa.py
# QAの作成、および、評価
import numpy as np
from typing import List, Dict, Tuple, cast
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
import re
import os
import json
import time
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

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

    from helper_rag_qa import SemanticCoverage
    
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
    from typing import List
    
    # 日本語の名詞・固有名詞を簡易的に抽出（正規表現ベース）
    # より精度を上げる場合はMeCabやJanomeなどの形態素解析器を使用
    
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
    from typing import List, Tuple
    
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


def determine_qa_pairs_count(chunk: Dict) -> int:
    """
    チャンクごとに最適なQ/Aペア数を決定
    
    Args:
        chunk: チャンクデータ
    
    Returns:
        最適なQ/Aペア数（2-5個）
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    text_length = len(chunk['text'])
    token_count = len(tokenizer.encode(chunk['text']))
    sentence_count = len(chunk.get('sentences', []))
    
    # トークン数に基づく基本的な決定
    if token_count < 50:
        base_count = 2
    elif token_count < 100:
        base_count = 3
    elif token_count < 150:
        base_count = 4
    else:
        base_count = 5
    
    # 文の数による調整
    if sentence_count <= 2:
        base_count = min(base_count, 2)
    elif sentence_count >= 5:
        base_count = min(base_count + 1, 5)
    
    # 最小2、最大5の制限
    return min(max(base_count, 2), 5)


# Pydantic model for structured output
class QAPair(BaseModel):
    question: str
    answer: str
    question_type: str

class QAPairsResponse(BaseModel):
    qa_pairs: List[QAPair]

def generate_qa_pairs_from_chunk(chunk: Dict, model: str = "gpt-5-mini") -> List[Dict]:
    """
    チャンクテキストからQ/Aペアを生成（GPT-5-miniを使用）
    
    Args:
        chunk: チャンクデータ（text, idを含む辞書）
        model: 使用するOpenAIモデル
    
    Returns:
        生成されたQ/Aペアのリスト
    """
    client = OpenAI()
    
    # 最適なQ/Aペア数を決定
    num_pairs = determine_qa_pairs_count(chunk)
    
    # システムプロンプト
    system_prompt = """あなたは教育コンテンツ作成の専門家です。
与えられたテキストから重要な情報を抽出し、学習に効果的な日本語のQ&Aペアを生成してください。

生成ルール：
1. 質問は明確で具体的に
2. 回答は簡潔で正確に（1-2文程度）
3. テキストの内容に忠実に
4. 重複を避け、多様な観点から質問を作成"""

    # ユーザープロンプト
    user_prompt = f"""以下のテキストから{num_pairs}個の日本語Q&Aペアを生成してください。

【質問タイプの指針】
- 事実確認型（〜は何ですか？）
- 理由説明型（なぜ〜ですか？）
- 比較型（〜と〜の違いは？）
- 応用型（〜はどのように活用されますか？）

【テキスト】
{chunk['text']}

【出力形式】
以下のJSON形式で出力してください：
{{
  "qa_pairs": [
    {{
      "question": "質問文",
      "answer": "回答文",
      "question_type": "fact/reason/comparison/application"
    }}
  ]
}}"""

    try:
        # API呼び出し (Responses API with structured output)
        # Responses APIでは、システムプロンプトとユーザープロンプトを結合
        combined_input = f"""{system_prompt}

{user_prompt}"""
        
        # Using the latest Responses API with parse for structured output
        # Note: gpt-5-mini doesn't support temperature parameter
        completion = client.responses.parse(
            model=model,
            input=combined_input,
            text_format=QAPairsResponse
        )
        # レスポンスの解析
        # ParsedResponseオブジェクトのoutput_parsed属性にアクセス
        if hasattr(completion, 'output_parsed') and completion.output_parsed:
            qa_pairs = [qa.model_dump() for qa in completion.output_parsed.qa_pairs]
        else:
            qa_pairs = []
        
        # チャンクIDを追加
        for qa in qa_pairs:
            qa['source_chunk_id'] = chunk['id']
        
        return qa_pairs
        
    except Exception as e:
        print(f"⚠️ Q/A生成エラー (チャンク {chunk['id']}): {e}")
        # エラー時は空のリストを返す
        return []


def generate_qa_for_all_chunks(chunks: List[Dict], model: str = "gpt-5-mini") -> List[Dict]:
    """
    全チャンクからQ/Aペアを生成
    
    Args:
        chunks: チャンクのリスト
        model: 使用するOpenAIモデル
    
    Returns:
        全Q/Aペアのリスト
    """
    all_qa_pairs = []
    
    print(f"\n【Q/Aペア生成】")
    print(f"使用モデル: {model}")
    print("-" * 40)
    
    for i, chunk in enumerate(chunks):
        print(f"\nチャンク {i+1}/{len(chunks)} を処理中...")
        
        # Q/A生成（リトライ機能付き）
        max_retries = 3
        for attempt in range(max_retries):
            try:
                qa_pairs = generate_qa_pairs_from_chunk(chunk, model)
                if qa_pairs:
                    all_qa_pairs.extend(qa_pairs)
                    print(f"  ✅ {len(qa_pairs)}個のQ/Aペアを生成")
                    break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"  ❌ 生成失敗: {e}")
                else:
                    print(f"  リトライ {attempt + 1}/{max_retries - 1}...")
                    time.sleep(2 ** attempt)  # 指数バックオフ
    
    print(f"\n合計: {len(all_qa_pairs)}個のQ/Aペアを生成")
    return all_qa_pairs


def display_qa_pairs(qa_pairs: List[Dict]):
    """
    生成されたQ/Aペアを表示
    
    Args:
        qa_pairs: Q/Aペアのリスト
    """
    print("\n【生成されたQ/Aペア】")
    print("=" * 60)
    
    # チャンクごとにグループ化
    from collections import defaultdict
    grouped = defaultdict(list)
    for qa in qa_pairs:
        grouped[qa.get('source_chunk_id', 'unknown')].append(qa)
    
    for chunk_id, pairs in grouped.items():
        print(f"\n◆ {chunk_id} からのQ/Aペア:")
        print("-" * 40)
        for i, qa in enumerate(pairs, 1):
            print(f"\nQ{i}: {qa['question']}")
            print(f"A{i}: {qa['answer']}")
            print(f"タイプ: {qa.get('question_type', 'N/A')}")


def calculate_coverage_matrix(doc_chunks, qa_pairs, analyzer):
    """
    ドキュメントチャンクとQ/Aペア間のカバレージマトリックスを計算
    
    Args:
        doc_chunks: ドキュメントチャンクのリスト
        qa_pairs: Q/Aペアのリスト
        analyzer: SemanticCoverageインスタンス
    
    Returns:
        coverage_matrix: カバレージマトリックス
        max_similarities: 各チャンクの最大類似度
    """
    # 埋め込み生成
    doc_embeddings = analyzer.generate_embeddings(doc_chunks)
    
    # Q/Aペアの埋め込み
    qa_embeddings = []
    for qa in qa_pairs:
        qa_text = f"{qa['question']} {qa['answer']}"
        embedding = analyzer.generate_embedding(qa_text)
        qa_embeddings.append(embedding)
    
    qa_embeddings = np.array(qa_embeddings)
    
    # カバレージマトリックス計算
    coverage_matrix = np.zeros((len(doc_chunks), len(qa_pairs)))
    for i in range(len(doc_embeddings)):
        for j in range(len(qa_embeddings)):
            similarity = analyzer.cosine_similarity(doc_embeddings[i], qa_embeddings[j])
            coverage_matrix[i, j] = similarity
    
    max_similarities = coverage_matrix.max(axis=1) if len(qa_pairs) > 0 else np.zeros(len(doc_chunks))
    
    return coverage_matrix, max_similarities


def identify_uncovered_chunks(doc_chunks, max_similarities, threshold=0.7):
    """
    未カバーチャンクを特定
    
    Args:
        doc_chunks: ドキュメントチャンクのリスト
        max_similarities: 各チャンクの最大類似度
        threshold: カバレージ判定閾値
    
    Returns:
        未カバーチャンクの情報リスト
    """
    uncovered_chunks = []
    
    for i, (chunk, sim) in enumerate(zip(doc_chunks, max_similarities)):
        if sim < threshold:
            uncovered_chunks.append({
                'chunk': chunk,
                'index': i,
                'similarity': sim,
                'gap': threshold - sim,
                'text': chunk['text']
            })
    
    return uncovered_chunks


def generate_topic_based_qa(chunk_text, keywords, existing_qa, model="gpt-5-mini", num_pairs=3):
    """
    主要トピックに基づいてQ/Aペアを生成
    
    Args:
        chunk_text: チャンクのテキスト
        keywords: 主要キーワード
        existing_qa: 既存のQ/Aペア（重複回避用）
        model: 使用するモデル
        num_pairs: 生成するQ/Aペア数
    
    Returns:
        生成されたQ/Aペアのリスト
    """
    client = OpenAI()
    
    # 既存Q/Aの要約（重複回避のため）
    existing_questions = [qa['question'] for qa in existing_qa[:5]]  # 最初の5つ
    existing_context = "\n".join([f"- {q}" for q in existing_questions]) if existing_questions else "なし"
    
    system_prompt = """あなたは教育コンテンツ作成の専門家です。
主要トピックに焦点を当てた、学習効果の高い日本語Q&Aペアを生成してください。"""
    
    user_prompt = f"""
以下のテキストから、主要トピック「{', '.join(keywords[:3])}」に焦点を当てて、
{num_pairs}個の日本語Q&Aペアを生成してください。

【重要な指示】
1. 各キーワードについて最低1つの質問を含める
2. 技術的な詳細や具体的な手法を問う質問を優先
3. 定義、特徴、関連性を明確にする

【既存のQ/A（重複を避ける）】
{existing_context}

【テキスト】
{chunk_text}

【出力形式】
{{
  "qa_pairs": [
    {{
      "question": "質問文",
      "answer": "回答文",
      "question_type": "definition/feature/relation/technical"
    }}
  ]
}}
"""
    
    combined_input = f"{system_prompt}\n\n{user_prompt}"
    
    try:
        completion = client.responses.parse(
            model=model,
            input=combined_input,
            text_format=QAPairsResponse
        )
        
        if hasattr(completion, 'output_parsed') and completion.output_parsed:
            qa_pairs = [qa.model_dump() for qa in completion.output_parsed.qa_pairs]
            return qa_pairs
        else:
            return []
    except Exception as e:
        print(f"⚠️ トピックベースQ/A生成エラー: {e}")
        return []


def predict_coverage_improvement(chunk, new_qa_pairs, analyzer):
    """
    新しいQ/Aペアによるカバレージ改善を予測
    
    Args:
        chunk: チャンク
        new_qa_pairs: 新しいQ/Aペア
        analyzer: SemanticCoverageインスタンス
    
    Returns:
        予測される新しい類似度
    """
    if not new_qa_pairs:
        return 0.0
    
    # チャンクの埋め込み
    chunk_embedding = analyzer.generate_embedding(chunk['text'])
    
    # 新Q/Aの埋め込み
    max_similarity = 0.0
    for qa in new_qa_pairs:
        qa_text = f"{qa['question']} {qa['answer']}"
        qa_embedding = analyzer.generate_embedding(qa_text)
        similarity = analyzer.cosine_similarity(chunk_embedding, qa_embedding)
        max_similarity = max(max_similarity, similarity)
    
    return max_similarity


def improve_coverage_with_auto_qa(doc_chunks, existing_qa_pairs, analyzer, threshold=0.7):
    """
    未カバーチャンクに対してQ/Aを自動生成してカバレージを改善
    
    Args:
        doc_chunks: ドキュメントチャンク
        existing_qa_pairs: 既存のQ/Aペア
        analyzer: SemanticCoverageインスタンス
        threshold: カバレージ判定閾値
    
    Returns:
        改善されたQ/Aペアリスト、改善前後のカバレージ率
    """
    print("\n" + "=" * 60)
    print("主要トピックベースのカバレージ改善")
    print("=" * 60)
    
    # 1. 現在のカバレージ分析
    coverage_matrix, max_similarities = calculate_coverage_matrix(
        doc_chunks, existing_qa_pairs, analyzer
    )
    
    initial_coverage_rate = sum(s > threshold for s in max_similarities) / len(doc_chunks)
    print(f"\n初期カバレージ率: {initial_coverage_rate:.1%}")
    
    # 2. 未カバーチャンクの特定
    uncovered_chunks = identify_uncovered_chunks(doc_chunks, max_similarities, threshold)
    
    if not uncovered_chunks:
        print("すべてのチャンクがカバーされています。")
        return existing_qa_pairs, initial_coverage_rate, initial_coverage_rate
    
    print(f"未カバーチャンク数: {len(uncovered_chunks)}")
    
    # 3. 各未カバーチャンクの処理
    new_qa_pairs = []
    
    for i, uc in enumerate(uncovered_chunks):
        print(f"\n未カバーチャンク {i+1}/{len(uncovered_chunks)} 処理中...")
        print(f"  現在の類似度: {uc['similarity']:.3f}")
        print(f"  カバレージギャップ: {uc['gap']:.3f}")
        
        # 主要トピック抽出（MeCab版使用）
        keywords = extract_keywords(uc['text'], top_n=5, use_mecab=True)
        print(f"  主要トピック: {', '.join(keywords)}")
        
        # Q/A生成
        chunk_qa = generate_topic_based_qa(
            chunk_text=uc['text'],
            keywords=keywords,
            existing_qa=existing_qa_pairs + new_qa_pairs,
            num_pairs=3
        )
        
        if chunk_qa:
            # チャンクIDを追加
            for qa in chunk_qa:
                qa['source_chunk_id'] = uc['chunk']['id']
                qa['auto_generated'] = True  # 自動生成フラグ
            
            new_qa_pairs.extend(chunk_qa)
            
            # カバレージ改善の予測
            predicted_coverage = predict_coverage_improvement(
                uc['chunk'], 
                chunk_qa, 
                analyzer
            )
            print(f"  予測カバレージ: {uc['similarity']:.3f} → {predicted_coverage:.3f}")
            print(f"  生成Q/A数: {len(chunk_qa)}")
        else:
            print(f"  ⚠️ Q/A生成に失敗しました")
    
    # 4. 全Q/Aペアの統合
    all_qa_pairs = existing_qa_pairs + new_qa_pairs
    
    # 5. 新カバレージ率の計算
    new_coverage_matrix, new_max_similarities = calculate_coverage_matrix(
        doc_chunks, all_qa_pairs, analyzer
    )
    final_coverage_rate = sum(s > threshold for s in new_max_similarities) / len(doc_chunks)
    
    # 6. 改善結果の表示
    print("\n" + "=" * 60)
    print("カバレージ改善結果")
    print("=" * 60)
    print(f"初期カバレージ率: {initial_coverage_rate:.1%}")
    print(f"最終カバレージ率: {final_coverage_rate:.1%}")
    print(f"改善度: +{(final_coverage_rate - initial_coverage_rate):.1%}")
    print(f"新規生成Q/A数: {len(new_qa_pairs)}")
    print(f"総Q/A数: {len(all_qa_pairs)}")
    
    return all_qa_pairs, initial_coverage_rate, final_coverage_rate


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
    """メイン関数：意味的チャンク分割の実装
    1. 意味的チャンク分割の実行: SemanticCoverageクラスを使用してexample_documentを意味的にチャンク分割
    2. 詳細な分割結果表示: 各チャンクのID、文の数、トークン数、開始・終了位置、内容を表示
    3. 統計情報: 総トークン数、平均トークン数、最大・最小トークン数を計算・表示
    4. キーワード抽出: 各チャンクから主要キーワードを3つ抽出して表示
    5. オプション機能: セマンティックカバレージ分析と可視化を選択的に実行可能
    """
    # helper_rag_qa.pyからSemanticCoverageクラスをインポート
    from helper_rag_qa import SemanticCoverage
    
    print("=" * 60)
    print("意味的チャンク分割デモンストレーション")
    print("=" * 60)
    
    # APIキーの確認と表示
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key != "your-openai-api-key-here":
        print(f"\n✅ OPENAI_API_KEYが設定されています (長さ: {len(api_key)}文字)")
    else:
        print("\n⚠️  警告: OPENAI_API_KEYが正しく設定されていません。")
        print(".envファイルまたは環境変数を確認してください。")
        return
    
    # SemanticCoverageインスタンスの作成
    print("\nSemanticCoverageクラスを初期化中...")
    try:
        analyzer = SemanticCoverage()
        print("✅ 初期化成功")
    except Exception as e:
        print(f"❌ 初期化エラー: {e}")
        print("\n詳細なエラー情報:")
        import traceback
        traceback.print_exc()
        return
    
    # 元の文書を表示
    print("\n【入力文書】")
    print("-" * 40)
    print(example_document)
    print("-" * 40)
    
    # 意味的チャンク分割の実行
    print("\n【意味的チャンク分割の実行】")
    chunks = analyzer.create_semantic_chunks(example_document)
    
    # 分割結果の詳細表示
    print(f"\n【分割結果】")
    print(f"総チャンク数: {len(chunks)}")
    print("-" * 40)
    
    for i, chunk in enumerate(chunks):
        print(f"\n■ チャンク {i+1} (ID: {chunk['id']})")
        print(f"  文の数: {len(chunk['sentences'])}")
        # トークン数を計算
        token_count = len(analyzer.tokenizer.encode(chunk['text']))
        print(f"  トークン数: {token_count}")
        print(f"  開始文インデックス: {chunk.get('start_sentence_idx', 'N/A')}")
        print(f"  終了文インデックス: {chunk.get('end_sentence_idx', 'N/A')}")
        print(f"  内容:")
        print(f"    {chunk['text']}")
    
    # チャンクの統計情報
    print("\n【統計情報】")
    print("-" * 40)
    # 各チャンクのトークン数を計算
    token_counts = [len(analyzer.tokenizer.encode(chunk['text'])) for chunk in chunks]
    total_tokens = sum(token_counts)
    avg_tokens = total_tokens / len(chunks) if chunks else 0
    max_tokens = max(token_counts, default=0)
    min_tokens = min(token_counts, default=0)
    
    print(f"総トークン数: {total_tokens}")
    print(f"平均トークン数/チャンク: {avg_tokens:.1f}")
    print(f"最大トークン数: {max_tokens}")
    print(f"最小トークン数: {min_tokens}")
    
    # 各チャンクが含む主要なトピック/キーワードを抽出
    print("\n【各チャンクの主要キーワード】")
    print("-" * 40)
    for i, chunk in enumerate(chunks):
        keywords = extract_keywords(chunk['text'], top_n=3)
        print(f"チャンク {i+1}: {', '.join(keywords)}")
    
    # Q/Aペア生成のオプション
    qa_generation_input = input("\n各チャンクからQ/Aペアを生成しますか？ (y/n): ")
    generated_qa_pairs = []
    if qa_generation_input.lower() == 'y':
        # GPT-5-miniを使用してQ/Aペアを生成
        generated_qa_pairs = generate_qa_for_all_chunks(chunks, model="gpt-5-mini")
        
        # 生成されたQ/Aペアを表示
        if generated_qa_pairs:
            display_qa_pairs(generated_qa_pairs)
            
            # Q/AペアをJSONファイルに保存するオプション
            save_input = input("\n生成されたQ/Aペアを保存しますか？ (y/n): ")
            if save_input.lower() == 'y':
                filename = f"generated_qa_pairs_{time.strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(generated_qa_pairs, f, ensure_ascii=False, indent=2)
                print(f"✅ Q/Aペアを {filename} に保存しました")
    
    # オプション：セマンティックカバレージの完全な分析も実行
    user_input = input("\n完全なセマンティックカバレージ分析を実行しますか？ (y/n): ")
    if user_input.lower() == 'y':
        print("\n" + "=" * 60)
        print("セマンティックカバレージ分析")
        print("=" * 60)
        coverage_matrix, max_similarities = demonstrate_semantic_coverage()
        
        # 可視化のオプション
        visualize_input = input("\n結果を可視化しますか？ (y/n): ")
        if visualize_input.lower() == 'y':
            fig = visualize_semantic_coverage(coverage_matrix, chunks, example_qa_pairs)
            plt.show()
        
        # カバレージ改善のオプション
        improve_input = input("\n主要トピックベースでカバレージを自動改善しますか？ (y/n): ")
        if improve_input.lower() == 'y':
            # 既存のQ/Aペアと生成済みQ/Aペアを統合
            all_existing_qa = example_qa_pairs.copy()
            if generated_qa_pairs:
                all_existing_qa.extend(generated_qa_pairs)
            
            # カバレージ改善実行
            improved_qa_pairs, initial_rate, final_rate = improve_coverage_with_auto_qa(
                chunks, all_existing_qa, analyzer
            )
            
            # 改善されたQ/Aペアの表示
            if final_rate > initial_rate:
                display_improved_input = input("\n改善されたQ/Aペアを表示しますか？ (y/n): ")
                if display_improved_input.lower() == 'y':
                    # 自動生成されたQ/Aのみ表示
                    auto_generated = [qa for qa in improved_qa_pairs if qa.get('auto_generated')]
                    if auto_generated:
                        print("\n【自動生成されたQ/Aペア】")
                        display_qa_pairs(auto_generated)
                
                # 改善後のQ/Aペアを保存
                save_improved_input = input("\n改善後の全Q/Aペアを保存しますか？ (y/n): ")
                if save_improved_input.lower() == 'y':
                    filename = f"improved_qa_pairs_{time.strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(improved_qa_pairs, f, ensure_ascii=False, indent=2)
                    print(f"✅ 改善後のQ/Aペアを {filename} に保存しました")
    
    print("\n処理が完了しました。")

if __name__ == "__main__":
    main()
