# a03_rag_qa_coverage.py - 詳細設計書

## 目次

1. [概要](#1-概要)
   - 1.1 [目的](#11-目的)
   - 1.2 [主要機能](#12-主要機能)
   - 1.3 [アーキテクチャ概要](#13-アーキテクチャ概要)
2. [SemanticCoverageクラス](#2-semanticcoverageクラス)
   - 2.1 [初期化](#21-初期化)
   - 2.2 [セマンティックチャンク作成](#22-セマンティックチャンク作成)
   - 2.3 [埋め込み生成](#23-埋め込み生成)
   - 2.4 [コサイン類似度計算](#24-コサイン類似度計算)
3. [QAGenerationConsiderationsクラス](#3-qagenerationconsiderationsクラス)
   - 3.1 [文書特性分析](#31-文書特性分析)
   - 3.2 [Q/A要件定義](#32-qa要件定義)
4. [LLMベースQ/A生成](#4-llmベースqa生成)
   - 4.1 [LLMBasedQAGeneratorクラス](#41-llmbasedqageneratorクラス)
   - 4.2 [基本Q/A生成](#42-基本qa生成)
   - 4.3 [多様なQ/A生成](#43-多様なqa生成)
5. [Chain-of-Thought Q/A生成](#5-chain-of-thought-qa生成)
   - 5.1 [ChainOfThoughtQAGeneratorクラス](#51-chainofthoughtqageneratorクラス)
   - 5.2 [推論過程付きQ/A生成](#52-推論過程付きqa生成)
6. [ルールベースQ/A生成](#6-ルールベースqa生成)
   - 6.1 [RuleBasedQAGeneratorクラス](#61-rulebasedqageneratorクラス)
   - 6.2 [定義文抽出](#62-定義文抽出)
   - 6.3 [事実情報抽出](#63-事実情報抽出)
   - 6.4 [列挙抽出](#64-列挙抽出)
7. [テンプレートベースQ/A生成](#7-テンプレートベースqa生成)
   - 7.1 [TemplateBasedQAGeneratorクラス](#71-templatebasedqageneratorクラス)
   - 7.2 [テンプレート定義](#72-テンプレート定義)
   - 7.3 [エンティティベース生成](#73-エンティティベース生成)
8. [ハイブリッドQ/A生成](#8-ハイブリッドqa生成)
   - 8.1 [HybridQAGeneratorクラス](#81-hybridqageneratorクラス)
   - 8.2 [包括的生成パイプライン](#82-包括的生成パイプライン)
   - 8.3 [品質検証](#83-品質検証)
9. [高度なQ/A生成技術](#9-高度なqa生成技術)
   - 9.1 [敵対的Q/A生成](#91-敵対的qa生成)
   - 9.2 [マルチホップQ/A生成](#92-マルチホップqa生成)
   - 9.3 [反事実的Q/A生成](#93-反事実的qa生成)
10. [Q/A生成最適化](#10-qa生成最適化)
    - 10.1 [QAGenerationOptimizerクラス](#101-qagenerationoptimizerクラス)
    - 10.2 [カバレッジ最適化戦略](#102-カバレッジ最適化戦略)
    - 10.3 [適応的生成](#103-適応的生成)
11. [チェックリストとベストプラクティス](#11-チェックリストとベストプラクティス)
    - 11.1 [Q/A生成チェックリスト](#111-qa生成チェックリスト)
    - 11.2 [品質基準](#112-品質基準)
12. [使用例とワークフロー](#12-使用例とワークフロー)
    - 12.1 [基本的な使用例](#121-基本的な使用例)
    - 12.2 [高度な使用例](#122-高度な使用例)

---

## 1. 概要

### 1.1 目的
RAG（Retrieval-Augmented Generation）システムにおけるセマンティックカバレッジ測定とQ/A自動生成のための包括的なフレームワーク。文書からの意味的チャンク分割、埋め込み生成、コサイン類似度計算、そして複数の手法を組み合わせた高品質Q/A生成を提供。

### 1.2 主要機能

#### セマンティックカバレッジ測定
- ✅ 文書の意味的チャンク分割
- ✅ OpenAI埋め込みAPI連携
- ✅ コサイン類似度計算
- ✅ トピック連続性を考慮した最適化

#### Q/A生成手法（5種類）
- ✅ **LLMベース生成**: GPT-4oを使用した高品質Q/A
- ✅ **Chain-of-Thought生成**: 推論過程付きQ/A
- ✅ **ルールベース生成**: パターンマッチングによる確実なQ/A
- ✅ **テンプレートベース生成**: エンティティ抽出とテンプレート適用
- ✅ **ハイブリッド生成**: 複数手法の組み合わせ

#### 高度な機能
- ✅ 敵対的Q/A生成（システムテスト用）
- ✅ マルチホップ推論Q/A
- ✅ 反事実的Q/A（What-if シナリオ）
- ✅ カバレッジ最適化戦略
- ✅ 適応的生成アルゴリズム

### 1.3 アーキテクチャ概要

```
[文書入力]
    ↓
[SemanticCoverage]
    ├── チャンク分割
    ├── 埋め込み生成
    └── 類似度計算
    ↓
[Q/A生成パイプライン]
    ├── Phase 1: ルールベース生成（高信頼度）
    ├── Phase 2: テンプレートベース生成（補完）
    ├── Phase 3: LLM生成（ギャップ埋め）
    └── Phase 4: 品質検証・改善
    ↓
[最適化・検証]
    ├── カバレッジ分析
    ├── 重複除去
    ├── 矛盾検出
    └── 品質スコアリング
    ↓
[Q/Aペア出力]
```

## 2. SemanticCoverageクラス

### 2.1 初期化

```python
class SemanticCoverage:
    def __init__(self, embedding_model="text-embedding-3-small"):
        self.embedding_model = embedding_model
        self.client = OpenAI()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
```

**パラメータ**:
- embedding_model: 埋め込みモデル（デフォルト: "text-embedding-3-small"）

**属性**:
- client: OpenAI APIクライアント
- tokenizer: トークンカウンター（cl100k_base）

### 2.2 セマンティックチャンク作成

#### create_semantic_chunks(document: str, verbose: bool = True) -> List[Dict]

**目的**: 文書を意味的に区切られたチャンクに分割

**処理フロー**:

```
1. 文単位分割
   ↓
   _split_into_sentences()
   - 日本語・英語の文末記号で分割
   - 正規表現: r'(?<=[。．.!?])\s*'
   ↓
2. トークンベースグループ化
   ↓
   max_tokens = 200
   for sentence in sentences:
       if current_tokens + sentence_tokens > max_tokens:
           チャンク保存
           新規チャンク開始
       else:
           current_chunk.append(sentence)
   ↓
3. トピック連続性調整
   ↓
   _adjust_chunks_for_topic_continuity()
   - 短すぎるチャンク（< 2文）は前のチャンクとマージ
   - マージ後のトークン数 < 300を条件
   ↓
4. チャンク返却
```

**返却値構造**:
```python
[
    {
        "id": "chunk_0",
        "text": "チャンクのテキスト",
        "sentences": ["文1", "文2", ...],
        "start_sentence_idx": 0,
        "end_sentence_idx": 3
    },
    ...
]
```

**重要ポイント**:
1. **文の境界で分割**: 意味の断絶を防ぐ
2. **トピックの変化を検出**: 連続性を考慮
3. **適切なサイズを維持**: 埋め込みモデルの制限内（200トークン）

#### _split_into_sentences(text: str) -> List[str]
**目的**: 文単位で分割（日本語対応）

**正規表現パターン**:
```python
r'(?<=[。．.!?])\s*'
```
- 日本語: 。．!?
- 英語: .!?
- 後続の空白も削除

**例**:
```python
text = "これは文1です。これは文2です。"
sentences = ["これは文1です。", "これは文2です。"]
```

#### _adjust_chunks_for_topic_continuity(chunks: List[Dict]) -> List[Dict]
**目的**: トピックの連続性を考慮してチャンクを調整

**アルゴリズム**:
```python
for i, chunk in enumerate(chunks):
    if i > 0 and len(chunk["sentences"]) < 2:
        # 短すぎるチャンク
        prev_chunk = adjusted_chunks[-1]
        combined_text = prev_chunk["text"] + " " + chunk["text"]

        if len(tokenizer.encode(combined_text)) < 300:
            # マージ実行
            prev_chunk["text"] = combined_text
            prev_chunk["sentences"].extend(chunk["sentences"])
            prev_chunk["end_sentence_idx"] = chunk["end_sentence_idx"]
            continue

    adjusted_chunks.append(chunk)
```

**条件**:
- チャンク長 < 2文
- マージ後トークン数 < 300

### 2.3 埋め込み生成

#### generate_embeddings(doc_chunks: List[Dict]) -> np.ndarray
**目的**: チャンクのリストから埋め込みベクトルを生成

**処理フロー**:
```python
embeddings = []
batch_size = 20  # OpenAI API制限

for i in range(0, len(doc_chunks), batch_size):
    batch = doc_chunks[i:i + batch_size]
    texts = [chunk["text"] for chunk in batch]

    try:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )

        for embedding_data in response.data:
            embedding = np.array(embedding_data.embedding)
            # L2正規化
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)

    except Exception as e:
        # エラー時はゼロベクトル
        for _ in batch:
            embeddings.append(np.zeros(1536))

return np.array(embeddings)
```

**重要ポイント**:
1. **バッチ処理**: 20件ずつ処理（API効率化）
2. **L2正規化**: コサイン類似度計算の高速化
3. **エラーハンドリング**: ゼロベクトルで代替

**埋め込み次元**: 1536（text-embedding-3-small）

#### generate_embedding(text: str) -> np.ndarray
**目的**: 単一テキストの埋め込み生成

**処理**:
```python
try:
    response = self.client.embeddings.create(
        model=self.embedding_model,
        input=text
    )
    embedding = np.array(response.data[0].embedding)
    return embedding / np.linalg.norm(embedding)
except Exception as e:
    return np.zeros(1536)
```

### 2.4 コサイン類似度計算

#### cosine_similarity(doc_emb: np.ndarray, qa_emb: np.ndarray) -> float
**目的**: 2つのベクトル間のコサイン類似度を計算

**最適化アルゴリズム**:
```python
# 正規化済みの場合は内積で計算
if np.allclose(np.linalg.norm(doc_emb), 1.0) and \
   np.allclose(np.linalg.norm(qa_emb), 1.0):
    return float(np.dot(doc_emb, qa_emb))

# 正規化されていない場合は完全な計算
dot_product = np.dot(doc_emb, qa_emb)
norm_doc = np.linalg.norm(doc_emb)
norm_qa = np.linalg.norm(qa_emb)

if norm_doc == 0 or norm_qa == 0:
    return 0.0

return float(dot_product / (norm_doc * norm_qa))
```

**計算式**:
- 正規化済み: `similarity = dot(doc_emb, qa_emb)`
- 未正規化: `similarity = dot(doc_emb, qa_emb) / (||doc_emb|| × ||qa_emb||)`

**範囲**: [-1, 1]（1に近いほど類似）

## 3. QAGenerationConsiderationsクラス

### 3.1 文書特性分析

#### analyze_document_characteristics(document) -> Dict
**目的**: 文書の特性を多角的に分析

**分析項目**:
```python
{
    "document_type": self.detect_document_type(document),
    # 技術文書、物語、レポート等

    "complexity_level": self.assess_complexity(document),
    # 専門性のレベル

    "factual_density": self.measure_factual_content(document),
    # 事実情報の密度

    "structure": self.analyze_structure(document),
    # 構造化の度合い

    "language": self.detect_language(document),
    # 言語と文体

    "domain": self.identify_domain(document),
    # ドメイン特定

    "length": len(document.split()),
    # 文書長

    "ambiguity_level": self.assess_ambiguity(document)
    # 曖昧さの度合い
}
```

**用途**:
- Q/A生成戦略の決定
- 質問タイプの選択
- 難易度レベルの設定

### 3.2 Q/A要件定義

#### define_qa_requirements() -> Dict
**目的**: Q/A生成の要件を定義

**返却値**:
```python
{
    "purpose": ["評価", "学習", "検索テスト", "ユーザー支援"],

    "question_types": [
        "事実確認型",  # What/Who/When/Where
        "理解確認型",  # Why/How
        "推論型",     # What if/影響は
        "要約型",     # 要点は何か
        "比較型"      # 違いは何か
    ],

    "difficulty_levels": ["基礎", "中級", "上級", "専門家"],

    "answer_formats": ["短答", "説明", "リスト", "段落"],

    "coverage_targets": {
        "minimum": 0.3,       # 最低限のカバレッジ
        "optimal": 0.6,       # 最適なカバレッジ
        "comprehensive": 0.8  # 包括的なカバレッジ
    }
}
```

## 4. LLMベースQ/A生成

### 4.1 LLMBasedQAGeneratorクラス

```python
class LLMBasedQAGenerator:
    def __init__(self, model="gpt-4o"):
        self.client = OpenAI()
        self.model = model
```

### 4.2 基本Q/A生成

#### generate_basic_qa(text: str, num_pairs: int = 5) -> List[Dict]
**目的**: 基本的なQ/A生成

**プロンプト構造**:
```python
prompt = f"""
以下のテキストから{num_pairs}個の質問と回答のペアを生成してください。

要件：
1. 質問は具体的で明確にする
2. 回答はテキストから直接答えられるものにする
3. 質問の種類を多様にする（What/Why/How/When/Where）
4. 回答は簡潔かつ正確にする

テキスト：
{text[:3000]}  # トークン制限のため切り詰め

出力形式（JSON）：
{{
    "qa_pairs": [
        {{
            "question": "質問文",
            "answer": "回答文",
            "question_type": "種類（factual/reasoning/summary等）",
            "difficulty": "難易度（easy/medium/hard）",
            "source_span": "回答の根拠となる元テキストの一部"
        }}
    ]
}}
"""
```

**API呼び出し**:
```python
response = self.client.chat.completions.create(
    model=self.model,
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},
    temperature=0.7
)

return json.loads(response.choices[0].message.content)["qa_pairs"]
```

**パラメータ**:
- temperature: 0.7（多様性のバランス）
- response_format: json_object（構造化出力）

### 4.3 多様なQ/A生成

#### generate_diverse_qa(text: str) -> List[Dict]
**目的**: 多様な種類のQ/A生成

**質問タイプ定義**:
```python
qa_types = {
    "factual": "事実確認の質問（Who/What/When/Where）",
    "causal": "因果関係の質問（Why/How）",
    "comparative": "比較の質問（違い、類似点）",
    "inferential": "推論が必要な質問",
    "summary": "要約を求める質問",
    "application": "応用・活用に関する質問"
}
```

**処理フロー**:
```python
all_qa_pairs = []

for qa_type, description in qa_types.items():
    prompt = f"""
    以下のテキストから「{description}」を2個生成してください。

    テキスト：{text[:2000]}

    JSON形式で出力：
    {{"qa_pairs": [...]}}
    """

    response = self.client.chat.completions.create(
        model=self.model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    qa_pairs = json.loads(response.choices[0].message.content)["qa_pairs"]
    for qa in qa_pairs:
        qa["question_type"] = qa_type
    all_qa_pairs.extend(qa_pairs)

return all_qa_pairs
```

**結果**: 各タイプ2個 × 6タイプ = 12個のQ/Aペア

## 5. Chain-of-Thought Q/A生成

### 5.1 ChainOfThoughtQAGeneratorクラス

```python
class ChainOfThoughtQAGenerator:
    """思考の連鎖を使った高品質Q/A生成"""
```

### 5.2 推論過程付きQ/A生成

#### generate_with_reasoning(text: str) -> List[Dict]
**目的**: 推論過程を含む高品質Q/A生成

**プロンプト戦略**:
```python
prompt = f"""
以下のテキストから質の高いQ/Aペアを生成します。
各ステップを踏んで考えてください。

ステップ1: テキストの主要なトピックと概念を抽出
ステップ2: 各トピックについて重要な情報を特定
ステップ3: その情報を問う質問を設計
ステップ4: テキストから正確な回答を抽出
ステップ5: 質問と回答の妥当性を検証

テキスト：
{text}

出力形式：
{{
    "analysis": {{
        "main_topics": ["トピック1", "トピック2"],
        "key_concepts": ["概念1", "概念2"],
        "information_density": "high/medium/low"
    }},
    "qa_pairs": [
        {{
            "question": "質問",
            "answer": "回答",
            "reasoning": "なぜこの質問が重要か",
            "confidence": 0.95
        }}
    ]
}}
"""
```

**API設定**:
```python
response = self.client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},
    temperature=0.3  # より確定的な出力のため低温
)

return json.loads(response.choices[0].message.content)
```

**特徴**:
- 段階的思考プロセス
- 分析結果を含む
- 信頼度スコア付き
- 推論の根拠を明示

## 6. ルールベースQ/A生成

### 6.1 RuleBasedQAGeneratorクラス

```python
class RuleBasedQAGenerator:
    def __init__(self):
        self.nlp = spacy.load("ja_core_news_lg")
```

**依存**: spaCy日本語モデル（ja_core_news_lg）

### 6.2 定義文抽出

#### extract_definition_qa(text: str) -> List[Dict]
**目的**: 定義文からQ/A生成

**パターン1**: 「〜とは〜である」
```python
pattern1 = r'([^。]+)とは([^。]+)(?:である|です)'
matches = re.findall(pattern1, text)

for term, definition in matches:
    qa_pairs.append({
        "question": f"{term.strip()}とは何ですか？",
        "answer": f"{term.strip()}とは{definition.strip()}です。",
        "type": "definition",
        "confidence": 0.9
    })
```

**パターン2**: 「〜は〜と呼ばれる」
```python
pattern2 = r'([^。]+)は([^。]+)と呼ばれ'
matches = re.findall(pattern2, text)

for subject, name in matches:
    qa_pairs.append({
        "question": f"{subject.strip()}は何と呼ばれますか？",
        "answer": f"{name.strip()}と呼ばれます。",
        "type": "terminology",
        "confidence": 0.85
    })
```

**例**:
```
入力: "機械学習とは、データから学習するアルゴリズムです。"
出力:
  question: "機械学習とは何ですか？"
  answer: "機械学習とは、データから学習するアルゴリズムです。"
```

### 6.3 事実情報抽出

#### extract_fact_qa(text: str) -> List[Dict]
**目的**: 事実情報からQ/A生成

**spaCy解析**:
```python
doc = self.nlp(text)

for sent in doc.sents:
    # 主語と動詞を含む文を対象
    subjects = [token for token in sent if token.dep_ == "nsubj"]
    verbs = [token for token in sent if token.pos_ == "VERB"]

    if subjects and verbs:
        # 日付を含む文
        dates = [ent for ent in sent.ents if ent.label_ == "DATE"]
        if dates:
            qa_pairs.append({
                "question": f"{subjects[0].text}はいつ{verbs[0].text}ましたか？",
                "answer": f"{dates[0].text}です。",
                "type": "temporal",
                "confidence": 0.7
            })

        # 場所を含む文
        locations = [ent for ent in sent.ents if ent.label_ in ["GPE", "LOC"]]
        if locations:
            qa_pairs.append({
                "question": f"{subjects[0].text}はどこで{verbs[0].text}ますか？",
                "answer": f"{locations[0].text}です。",
                "type": "location",
                "confidence": 0.7
            })
```

**抽出対象エンティティ**:
- DATE: 日付・時刻
- GPE: 地政学的エンティティ（国・都市）
- LOC: 場所

### 6.4 列挙抽出

#### extract_list_qa(text: str) -> List[Dict]
**目的**: 列挙からQ/A生成

**パターン**: 「〜には、A、B、Cがある」
```python
pattern = r'([^。]+)(?:には|に|では)、([^。]+(?:、[^。]+)+)が(?:ある|あります|含まれ|存在)'
matches = re.findall(pattern, text)

for topic, items in matches:
    item_list = [item.strip() for item in items.split('、')]
    qa_pairs.append({
        "question": f"{topic.strip()}には何がありますか？",
        "answer": f"{topic.strip()}には、{'、'.join(item_list)}があります。",
        "type": "enumeration",
        "items": item_list,
        "confidence": 0.8
    })
```

**例**:
```
入力: "機械学習には、教師あり学習、教師なし学習、強化学習があります。"
出力:
  question: "機械学習には何がありますか？"
  answer: "機械学習には、教師あり学習、教師なし学習、強化学習があります。"
  items: ["教師あり学習", "教師なし学習", "強化学習"]
```

## 7. テンプレートベースQ/A生成

### 7.1 TemplateBasedQAGeneratorクラス

```python
class TemplateBasedQAGenerator:
    def __init__(self):
        self.templates = self._load_templates()
```

### 7.2 テンプレート定義

#### _load_templates() -> Dict
**目的**: 質問テンプレートの定義

**テンプレート種類**:
```python
{
    "comparison": [
        "{A}と{B}の違いは何ですか？",
        "{A}と{B}のどちらが{property}ですか？",
        "{A}と{B}の共通点は何ですか？"
    ],

    "process": [
        "{process}のプロセスを説明してください。",
        "{process}にはどのようなステップがありますか？",
        "{process}の最初のステップは何ですか？"
    ],

    "cause_effect": [
        "{event}の原因は何ですか？",
        "{cause}の結果として何が起こりますか？",
        "なぜ{phenomenon}が発生するのですか？"
    ],

    "characteristics": [
        "{entity}の特徴は何ですか？",
        "{entity}の主な機能は何ですか？",
        "{entity}はどのように使用されますか？"
    ]
}
```

### 7.3 エンティティベース生成

#### generate_from_entities(text: str, entities: List[Dict]) -> List[Dict]
**目的**: 抽出されたエンティティからQ/A生成

**エンティティタイプ別テンプレート**:
```python
for entity in entities:
    entity_type = entity['type']
    entity_text = entity['text']

    if entity_type == "PERSON":
        questions = [
            f"{entity_text}は誰ですか？",
            f"{entity_text}は何をしましたか？",
            f"{entity_text}の役割は何ですか？"
        ]

    elif entity_type == "ORG":
        questions = [
            f"{entity_text}はどのような組織ですか？",
            f"{entity_text}の目的は何ですか？",
            f"{entity_text}はいつ設立されましたか？"
        ]

    elif entity_type == "PRODUCT":
        questions = [
            f"{entity_text}とは何ですか？",
            f"{entity_text}の用途は何ですか？",
            f"{entity_text}の特徴は何ですか？"
        ]

    # テキストから回答を探索
    for question in questions:
        answer = self.find_answer_in_text(text, entity_text, question)
        if answer:
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "entity": entity_text,
                "type": "entity_based",
                "confidence": 0.75
            })
```

## 8. ハイブリッドQ/A生成

### 8.1 HybridQAGeneratorクラス

```python
class HybridQAGenerator:
    def __init__(self):
        self.llm_generator = LLMBasedQAGenerator()
        self.rule_generator = RuleBasedQAGenerator()
        self.template_generator = TemplateBasedQAGenerator()
```

### 8.2 包括的生成パイプライン

#### generate_comprehensive_qa(text, target_count=20, quality_threshold=0.7) -> List[Dict]
**目的**: 包括的なQ/A生成パイプライン

**4フェーズ戦略**:

```python
# Phase 1: ルールベースで確実なQ/Aを生成
print("Phase 1: ルールベース生成...")
rule_based_qa = []
rule_based_qa.extend(self.rule_generator.extract_definition_qa(text))
rule_based_qa.extend(self.rule_generator.extract_fact_qa(text))
rule_based_qa.extend(self.rule_generator.extract_list_qa(text))

# 高信頼度のものだけを選択
rule_based_qa = [qa for qa in rule_based_qa
                 if qa.get('confidence', 0) >= quality_threshold]
all_qa_pairs.extend(rule_based_qa)

# Phase 2: テンプレートベースで補完
print("Phase 2: テンプレートベース生成...")
entities = self.extract_entities(text)
template_qa = self.template_generator.generate_from_entities(text, entities)

# 重複を除去
template_qa = self.remove_duplicates(template_qa, all_qa_pairs)
all_qa_pairs.extend(template_qa)

# Phase 3: LLMで不足分を補完
remaining_count = target_count - len(all_qa_pairs)
if remaining_count > 0:
    print(f"Phase 3: LLM生成（残り{remaining_count}個）...")

    # カバーされていない領域を特定
    uncovered_text = self.identify_uncovered_sections(text, all_qa_pairs)

    llm_qa = self.llm_generator.generate_diverse_qa(uncovered_text)
    llm_qa = llm_qa[:remaining_count]

    all_qa_pairs.extend(llm_qa)

# Phase 4: 品質検証と改善
print("Phase 4: 品質検証...")
validated_qa = self.validate_and_improve_qa(all_qa_pairs, text)

return validated_qa
```

**フェーズ別特徴**:
| フェーズ | 手法 | コスト | 信頼度 | 生成数目安 |
|---------|------|--------|--------|----------|
| Phase 1 | ルールベース | 無料 | 高 | 10件 |
| Phase 2 | テンプレート | 無料 | 中 | 15件 |
| Phase 3 | LLM | 有料 | 中〜高 | 20件 |
| Phase 4 | 検証 | 無料 | - | フィルタ |

### 8.3 品質検証

#### validate_and_improve_qa(qa_pairs, source_text) -> List[Dict]
**目的**: Q/Aペアの品質検証と改善

**検証項目**:
```python
for qa in qa_pairs:
    validations = {
        "answer_found": self.verify_answer_in_text(
            qa['answer'], source_text
        ),
        "question_clear": self.check_question_clarity(
            qa['question']
        ),
        "no_contradiction": self.check_no_contradiction(
            qa, validated_qa
        ),
        "appropriate_length": self.check_length_appropriateness(qa)
    }

    # すべての検証をパスしたものだけを採用
    if all(validations.values()):
        qa['validations'] = validations
        qa['quality_score'] = self.calculate_quality_score(qa)
        validated_qa.append(qa)

# 品質スコアでソート
validated_qa.sort(key=lambda x: x['quality_score'], reverse=True)
```

**検証基準**:
1. **answer_found**: 回答がソーステキストに存在
2. **question_clear**: 質問が明確で曖昧でない
3. **no_contradiction**: 他のQ/Aと矛盾しない
4. **appropriate_length**: 適切な長さ

## 9. 高度なQ/A生成技術

### 9.1 敵対的Q/A生成

#### generate_adversarial_qa(text, existing_qa) -> List[Dict]
**目的**: システムを混乱させる質問の生成（テスト用）

**戦略**:

1. **否定質問**:
```python
for qa in existing_qa[:5]:
    adversarial_qa.append({
        "question": qa['question'].replace("何ですか", "何ではありませんか"),
        "answer": f"{qa['answer']}ではないものを指します。",
        "type": "adversarial_negation"
    })
```

2. **文脈外質問**:
```python
adversarial_qa.append({
    "question": "この文書に書かれていない情報は何ですか？",
    "answer": "文書には含まれていない情報です。",
    "type": "out_of_context"
})
```

3. **曖昧な参照**:
```python
adversarial_qa.append({
    "question": "それは何を指していますか？",
    "answer": "文脈により異なります。",
    "type": "ambiguous_reference"
})
```

### 9.2 マルチホップQ/A生成

#### generate_multi_hop_qa(text: str) -> List[Dict]
**目的**: マルチホップ推論が必要なQ/A生成

**プロンプト例**:
```python
prompt = f"""
以下のテキストから、複数の情報を組み合わせて答える必要がある質問を生成してください。

例：
- AがBである、BがCである → AとCの関係は？
- XはYより大きい、YはZより大きい → X、Y、Zの順序は？

テキスト：{text}

JSON形式で3つ生成してください。
"""
```

**特徴**:
- 複数ステップの推論が必要
- 情報の結合・統合が求められる
- システムの高度な理解力をテスト

### 9.3 反事実的Q/A生成

#### generate_counterfactual_qa(text: str) -> List[Dict]
**目的**: 反事実的Q/A（もし〜だったら）の生成

**テンプレート**:
```python
counterfactual_templates = [
    "もし{condition}でなかったら、{outcome}はどうなっていましたか？",
    "{event}が起こらなかった場合、何が変わっていましたか？",
    "{factor}が異なっていたら、結果はどう変わりますか？"
]
```

**用途**:
- 因果関係の理解テスト
- 推論能力の評価
- What-ifシナリオ分析

## 10. Q/A生成最適化

### 10.1 QAGenerationOptimizerクラス

```python
class QAGenerationOptimizer:
    """Q/A生成の最適化"""
```

### 10.2 カバレッジ最適化戦略

#### optimize_for_coverage(text, budget) -> Dict
**目的**: カバレッジを最大化する生成戦略

**5フェーズ戦略**:
```python
strategy = {
    "phase1": {
        "method": "rule_based",
        "target": "high_confidence_facts",
        "cost": 0,
        "expected_qa": 10
    },
    "phase2": {
        "method": "template_based",
        "target": "entities_and_concepts",
        "cost": 0,
        "expected_qa": 15
    },
    "phase3": {
        "method": "llm_cheap",
        "model": "gpt-3.5-turbo",
        "target": "gap_filling",
        "cost": budget * 0.3,
        "expected_qa": 20
    },
    "phase4": {
        "method": "llm_quality",
        "model": "gpt-4o",
        "target": "complex_reasoning",
        "cost": budget * 0.5,
        "expected_qa": 10
    },
    "phase5": {
        "method": "human_validation",
        "target": "quality_assurance",
        "cost": budget * 0.2,
        "expected_qa": "validation_only"
    }
}
```

**予算配分**:
- Phase 3 (安価LLM): 30%
- Phase 4 (高品質LLM): 50%
- Phase 5 (人間検証): 20%

### 10.3 適応的生成

#### adaptive_generation(text, initial_qa) -> List[Dict]
**目的**: 既存Q/Aを分析して適応的に生成

**アルゴリズム**:
```python
# カバレッジ分析
coverage_analysis = self.analyze_coverage(text, initial_qa)

# 不足している質問タイプを特定
missing_types = self.identify_missing_question_types(initial_qa)

# ギャップを埋める新しいQ/A生成
new_qa = []
for missing_type in missing_types:
    new_qa.extend(
        self.generate_specific_type(text, missing_type, count=3)
    )

return new_qa
```

**適応ポイント**:
1. カバレッジの低い領域を特定
2. 不足する質問タイプを補完
3. 難易度のバランスを調整

## 11. チェックリストとベストプラクティス

### 11.1 Q/A生成チェックリスト

#### qa_generation_checklist() -> Dict

```python
{
    "事前準備": [
        "□ 文書の種類と特性を分析",
        "□ 目的（評価/学習/テスト）を明確化",
        "□ 必要なカバレッジレベルを設定",
        "□ 予算とリソースを確認"
    ],

    "品質基準": [
        "□ 回答がテキスト内に存在することを確認",
        "□ 質問の明確性と曖昧さの排除",
        "□ 質問タイプの多様性を確保",
        "□ 難易度のバランスを調整"
    ],

    "技術選択": [
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
```

### 11.2 品質基準

| 基準 | 説明 | 閾値 |
|-----|------|------|
| 回答存在性 | 回答がソーステキストに存在 | 必須 |
| 質問明確性 | 質問が明確で曖昧でない | 0.8以上 |
| タイプ多様性 | 質問タイプの分散 | 各タイプ15%以上 |
| 難易度バランス | 難易度レベルの分布 | 基礎:中級:上級 = 3:5:2 |
| 信頼度 | 生成Q/Aの信頼度スコア | 0.7以上 |
| カバレッジ | 文書のカバレッジ率 | 0.6以上（最適） |

## 12. 使用例とワークフロー

### 12.1 基本的な使用例

#### セマンティックチャンク作成
```python
# 初期化
coverage = SemanticCoverage(embedding_model="text-embedding-3-small")

# 文書をチャンクに分割
document = "長い文書テキスト..."
chunks = coverage.create_semantic_chunks(document, verbose=True)

# 埋め込み生成
doc_embeddings = coverage.generate_embeddings(chunks)

# Q/Aペアの埋め込み
qa_text = "質問と回答のテキスト"
qa_embedding = coverage.generate_embedding(qa_text)

# 類似度計算
similarity = coverage.cosine_similarity(doc_embeddings[0], qa_embedding)
print(f"類似度: {similarity:.3f}")
```

#### LLMベースQ/A生成
```python
# 初期化
generator = LLMBasedQAGenerator(model="gpt-4o")

# 基本Q/A生成
qa_pairs = generator.generate_basic_qa(text, num_pairs=5)

# 多様なQ/A生成
diverse_qa = generator.generate_diverse_qa(text)

# 結果表示
for qa in qa_pairs:
    print(f"Q: {qa['question']}")
    print(f"A: {qa['answer']}")
    print(f"Type: {qa['question_type']}")
    print("---")
```

### 12.2 高度な使用例

#### ハイブリッド生成パイプライン
```python
# 初期化
hybrid_gen = HybridQAGenerator()

# 包括的Q/A生成
qa_pairs = hybrid_gen.generate_comprehensive_qa(
    text=document,
    target_count=20,
    quality_threshold=0.7
)

# 結果分析
print(f"生成されたQ/A数: {len(qa_pairs)}")

# 質問タイプ別集計
type_counts = {}
for qa in qa_pairs:
    qa_type = qa.get('type', 'unknown')
    type_counts[qa_type] = type_counts.get(qa_type, 0) + 1

print("質問タイプ別集計:")
for qa_type, count in type_counts.items():
    print(f"  {qa_type}: {count}件")
```

#### カバレッジ最適化
```python
# 最適化戦略の取得
optimizer = QAGenerationOptimizer()
strategy = optimizer.optimize_for_coverage(text, budget=100)

print("最適化戦略:")
for phase, config in strategy.items():
    print(f"{phase}:")
    print(f"  手法: {config['method']}")
    print(f"  コスト: {config['cost']}")
    print(f"  期待Q/A数: {config['expected_qa']}")

# 適応的生成
initial_qa = generator.generate_basic_qa(text, num_pairs=10)
additional_qa = optimizer.adaptive_generation(text, initial_qa)

print(f"初期Q/A数: {len(initial_qa)}")
print(f"追加Q/A数: {len(additional_qa)}")
```

#### Chain-of-Thought生成
```python
# CoT生成
cot_gen = ChainOfThoughtQAGenerator()
result = cot_gen.generate_with_reasoning(text)

# 分析結果
print("文書分析:")
print(f"  主要トピック: {result['analysis']['main_topics']}")
print(f"  重要概念: {result['analysis']['key_concepts']}")
print(f"  情報密度: {result['analysis']['information_density']}")

# Q/Aペア
print("\nQ/Aペア:")
for qa in result['qa_pairs']:
    print(f"Q: {qa['question']}")
    print(f"A: {qa['answer']}")
    print(f"理由: {qa['reasoning']}")
    print(f"信頼度: {qa['confidence']}")
    print("---")
```

---

## まとめ

本設計書では、RAGシステムにおけるセマンティックカバレッジ測定とQ/A自動生成の包括的なフレームワークを詳述しました。

**主要コンポーネント**:
1. **SemanticCoverage**: チャンク分割、埋め込み生成、類似度計算
2. **LLMベース生成**: GPT-4oを活用した高品質Q/A
3. **ルールベース生成**: パターンマッチングによる確実なQ/A
4. **テンプレートベース生成**: エンティティ抽出とテンプレート適用
5. **ハイブリッド生成**: 複数手法の最適な組み合わせ

**推奨ワークフロー**:
1. 文書特性分析 → 適切な手法選択
2. Phase 1: ルールベース（無料、高信頼度）
3. Phase 2: テンプレート（無料、補完）
4. Phase 3: LLM（有料、ギャップ埋め）
5. Phase 4: 品質検証・改善

このフレームワークにより、コスト効率と品質のバランスを取りながら、高カバレッジのQ/Aデータセットを構築できます。