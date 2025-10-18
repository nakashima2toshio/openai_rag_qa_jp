# a03_rag_qa_coverage.py - 詳細設計書

## 目次

1. [概要](#1-概要)
   - 1.1 [目的](#11-目的)
   - 1.2 [主要機能](#12-主要機能)
   - 1.3 [8つの生成手法の概要](#13-8つの生成手法の概要)
2. [アーキテクチャ](#2-アーキテクチャ)
   - 2.1 [システム構成図](#21-システム構成図)
   - 2.2 [主要コンポーネント](#22-主要コンポーネント)
3. [Q/A生成手法の詳細](#3-qa生成手法の詳細)
   - 3.1 [SemanticCoverage](#31-semanticcoverage)
   - 3.2 [RuleBasedQAGenerator](#32-rulebasedqagenerator)
   - 3.3 [TemplateBasedQAGenerator](#33-templatebasedqagenerator)
   - 3.4 [LLMBasedQAGenerator](#34-llmbasedqagenerator)
   - 3.5 [ChainOfThoughtQAGenerator](#35-chainofthoughtqagenerator)
   - 3.6 [HybridQAGenerator](#36-hybridqagenerator)
   - 3.7 [AdvancedQAGenerationTechniques](#37-advancedqagenerationtechniques)
   - 3.8 [QAGenerationOptimizer](#38-qagenerationoptimizer)
4. [8つの手法比較表](#4-8つの手法比較表)
   - 4.1 [総合比較表](#41-総合比較表)
   - 4.2 [OpenAI API利用比較](#42-openai-api利用比較)
   - 4.3 [生成品質・特性比較](#43-生成品質特性比較)
5. [処理フロー](#5-処理フロー)
   - 5.1 [メイン処理フロー](#51-メイン処理フロー)
   - 5.2 [カバレッジ分析フロー](#52-カバレッジ分析フロー)
6. [結果保存](#6-結果保存)
   - 6.1 [出力ファイル](#61-出力ファイル)
   - 6.2 [統計情報](#62-統計情報)
7. [実行方法](#7-実行方法)
   - 7.1 [環境変数設定](#71-環境変数設定)
   - 7.2 [実行コマンド](#72-実行コマンド)
8. [依存関係](#8-依存関係)
   - 8.1 [外部ライブラリ](#81-外部ライブラリ)
   - 8.2 [内部モジュール](#82-内部モジュール)
9. [パフォーマンスとコスト](#9-パフォーマンスとコスト)
   - 9.1 [API呼び出しコスト](#91-api呼び出しコスト)
   - 9.2 [実行時間](#92-実行時間)
10. [注意事項・制約](#10-注意事項制約)
    - 10.1 [制約事項](#101-制約事項)
    - 10.2 [推奨設定](#102-推奨設定)
11. [今後の改善案](#11-今後の改善案)
    - 11.1 [機能拡張](#111-機能拡張)
    - 11.2 [品質向上](#112-品質向上)

---

## 1. 概要

### 1.1 目的
セマンティックカバレッジ分析と8つの異なる手法によるQ/A生成を統合した包括的なQ/Aペア生成システム。各手法の特性を理解し、最適な組み合わせでRAGシステム用の高品質なQ/Aペアを生成する。

### 1.2 主要機能
- **セマンティックカバレッジ分析**: 文書を意味的に分割し、Q/Aペアのカバレッジを測定
- **多様な生成手法**: 8つの異なるアプローチでQ/A生成
- **ハイブリッド統合**: 複数手法の結果を統合し品質検証
- **カバレッジ最適化**: 埋め込みベクトルを使用した文書カバレッジ分析
- **結果のエクスポート**: JSON形式での詳細な結果保存

### 1.3 8つの生成手法の概要

| No | 手法名 | 主な用途 | API使用 |
|----|-------|---------|---------|
| 1 | SemanticCoverage | 文書分析・カバレッジ測定 | ✅ (Embeddings) |
| 2 | RuleBasedQAGenerator | 基本的な定義・事実のQ/A生成 | ❌ |
| 3 | TemplateBasedQAGenerator | エンティティベースのQ/A生成 | ❌ |
| 4 | LLMBasedQAGenerator | 多様で自然なQ/A生成 | ✅ (Chat/Responses) |
| 5 | ChainOfThoughtQAGenerator | 推論過程付き高品質Q/A | ✅ (Chat) |
| 6 | HybridQAGenerator | 複数手法の統合・品質検証 | ⚠️ (部分的) |
| 7 | AdvancedQAGenerationTechniques | 敵対的・高度なQ/A生成 | ❌ (将来実装) |
| 8 | QAGenerationOptimizer | カバレッジ最適化 | ✅ (Embeddings) |

## 2. アーキテクチャ

### 2.1 システム構成図

**処理フロー:**

1. サンプル文書テキスト入力
2. SemanticCoverage: チャンク分割、埋め込み生成
3. チェックリスト確認
4. Q/A生成（複数手法並列実行可能）:
   - RuleBasedQAGenerator: パターンマッチング
   - TemplateBasedQAGenerator: テンプレート適用
   - LLMBasedQAGenerator: OpenAI API使用
   - ChainOfThoughtQAGenerator: OpenAI API (chat) 使用
5. HybridQAGenerator: 統合、重複除去、品質検証
6. AdvancedQAGenerationTechniques: 敵対的Q/A生成
7. QAGenerationOptimizer: カバレッジ分析、最適化推奨
8. 結果エクスポート: JSON出力、統計情報

### 2.2 主要コンポーネント

#### 2.2.1 データモデル（Pydantic）
- **QAPair**: 個別Q/Aペアのデータモデル
  - question: str - 質問文
  - answer: str - 回答文
  - question_type: str - 質問タイプ
  - difficulty: str - 難易度
  - source_span: str - ソーステキスト

- **QAPairsList**: Q/Aペアのリスト構造
  - qa_pairs: List[QAPair] - Q/Aペアのリスト

#### 2.2.2 設定管理
- サンプル文書テキスト（RAGシステムに関する日本語テキスト）
- 埋め込みモデル: `text-embedding-3-small`
- LLMモデル: `gpt-4o-mini`（デフォルト）


## 3. Q/A生成手法の詳細

### 3.1 SemanticCoverage
### ----------------------
SemanticCoverage クラスの詳細説明

概要

SemanticCoverageクラスは、文書のセマンティック（意味的）カバレッジを測定するためのクラスです。文書を意味的なチャンク（塊
）に分割し、OpenAIの埋め込みモデルを使って各チャンクのベクトル表現を生成し、Q&Aペアがその文書をどの程度カバーしているか
をコサイン類似度で評価します。

主な用途

- RAG（Retrieval-Augmented Generation）システムの評価
- Q&Aデータセットの品質評価
- 文書の意味的な分割と分析

---
処理の流れ

1. 初期化 (__init__)
 ↓
2. 文書をセマンティックチャンクに分割 (create_semantic_chunks)
 ↓
3. 各チャンクの埋め込みベクトルを生成 (generate_embeddings)
 ↓
4. Q&Aペアの埋め込みベクトルを生成 (generate_embedding)
 ↓
5. コサイン類似度を計算してカバレッジを測定 (cosine_similarity)

---
内部関数の処理の流れと概要

1. __init__(self, embedding_model="text-embedding-3-small") (984-994行)

処理の流れ:
1. 埋め込みモデル名を設定
2. 環境変数からOpenAI APIキーを取得
3. APIキーが有効ならOpenAIクライアントを初期化
4. tiktokenトークナイザーを初期化

概要: クラスの初期化。OpenAI APIクライアントとトークンカウンタを準備します。

---
2. create_semantic_chunks(self, document: str, verbose: bool = True) (996-1050行)

処理の流れ:
Step 1: _split_into_sentences() で文単位に分割
 ↓
Step 2: 文をトークン数が200以下になるようにグループ化
 ↓
Step 3: _adjust_chunks_for_topic_continuity() でチャンクを調整
 ↓
返り値: チャンクのリスト（各チャンクはID、テキスト、文リスト、開始/終了インデックスを含む辞書）

概要: 文書を意味的なチャンクに分割します。文の境界を尊重し、最大200トークンのチャンクを作成します。

重要ポイント:
- 文の途中で分割しない（意味の断絶を防ぐ）
- トークン制限（200トークン）を守る
- トピックの連続性を考慮

---
3. _split_into_sentences(self, text: str) (1052-1058行)

処理の流れ:
1. 正規表現で日本語と英語の文末記号（。．.!?）で分割
2. 空文字列を除去
3. 前後の空白を削除

概要: テキストを文単位に分割する内部関数。日本語・英語両対応。

---
4. _adjust_chunks_for_topic_continuity(self, chunks: List[Dict]) (1060-1080行)

処理の流れ:
for each chunk:
  if chunk が短すぎる（2文未満）:
      前のチャンクと結合可能かチェック
      if 結合後も300トークン未満:
          前のチャンクとマージ
      else:
          そのまま追加
  else:
      そのまま追加

概要: 短すぎるチャンクを前のチャンクとマージして、トピックの連続性を保ちます。

---
5. generate_embeddings(self, doc_chunks: List[Dict]) (1082-1124行)

処理の流れ:
1. APIキーがない場合はゼロベクトルを返す
2. チャンクを20個ずつバッチ処理
3. for each batch:
   - OpenAI APIで埋め込みベクトル生成
   - L2正規化（コサイン類似度計算の最適化）
   - エラー時はゼロベクトルを追加
4. NumPy配列として返す

概要: チャンクのリストから埋め込みベクトルを生成します。バッチ処理で効率化し、エラーハンドリングも実装。

重要ポイント:
- バッチサイズ20（OpenAI APIの制限考慮）
- L2正規化でコサイン類似度計算を高速化
- エラー時のフォールバック（ゼロベクトル）

---
6. generate_embedding(self, text: str) (1126-1141行)

処理の流れ:
1. APIキーがない場合はゼロベクトルを返す
2. OpenAI APIで単一テキストの埋め込み生成
3. L2正規化
4. エラー時はゼロベクトルを返す

概要: 単一テキスト（Q&Aペアなど）の埋め込みベクトルを生成します。

---
7. cosine_similarity(self, doc_emb: np.ndarray, qa_emb: np.ndarray) (1143-1165行)

処理の流れ:
1. ベクトルが正規化済みかチェック
2. if 正規化済み:
   内積で計算（高速）
 else:
   完全なコサイン類似度の式で計算
3. ゼロ除算チェック
4. 類似度を返す（-1.0 ~ 1.0）

概要: 2つの埋め込みベクトル間のコサイン類似度を計算します。正規化済みなら内積で高速計算。

重要ポイント:
- 正規化済みベクトルは内積で計算可能
- 範囲は[-1, 1]、1に近いほど類似
- ゼロベクトルのエラーハンドリング

---
IPO（Input-Process-Output）分析

全体の IPO

| 項目      | 内容                                                     |
|---------|--------------------------------------------------------|
| Input   | 文書テキスト（str）、OpenAI APIキー（環境変数）                         |
| Process | 文書のセマンティックチャンク分割 → 埋め込みベクトル生成 → 類似度計算                  |
| Output  | チャンクリスト（List[Dict]）、埋め込みベクトル（np.ndarray）、類似度スコア（float） |

---
各関数の IPO

1. __init__(embedding_model)

| IPO | 内容                                           |
|-----|----------------------------------------------|
| I   | embedding_model (str), OPENAI_API_KEY (環境変数) |
| P   | APIキー検証、OpenAIクライアント初期化、トークナイザー初期化           |
| O   | 初期化済みのインスタンス（self.client, self.tokenizer）    |

2. create_semantic_chunks(document, verbose)

| IPO | 内容                                                                          |
|-----|-----------------------------------------------------------------------------|
| I   | document (str), verbose (bool)                                              |
| P   | 文分割 → トークンカウント → チャンクグループ化 → トピック調整                                         |
| O   | List[Dict] (各要素: id, text, sentences, start_sentence_idx, end_sentence_idx) |

3. _split_into_sentences(text)

| IPO | 内容                           |
|-----|------------------------------|
| I   | text (str)                   |
| P   | 正規表現で文末記号分割 → 空文字列除去 → トリミング |
| O   | List[str] (文のリスト)            |

4. _adjust_chunks_for_topic_continuity(chunks)

| IPO | 内容                                   |
|-----|--------------------------------------|
| I   | chunks (List[Dict])                  |
| P   | 短いチャンクを前のチャンクとマージ判定 → 300トークン以下ならマージ |
| O   | List[Dict] (調整後のチャンクリスト)             |

5. generate_embeddings(doc_chunks)

| IPO | 内容                                                |
|-----|---------------------------------------------------|
| I   | doc_chunks (List[Dict])                           |
| P   | バッチ処理（20個ずつ） → OpenAI API呼び出し → L2正規化 → エラーハンドリング |
| O   | np.ndarray (shape: [チャンク数, 1536])                 |

6. generate_embedding(text)

| IPO | 内容                                 |
|-----|------------------------------------|
| I   | text (str)                         |
| P   | OpenAI API呼び出し → L2正規化 → エラーハンドリング |
| O   | np.ndarray (shape: [1536])         |

7. cosine_similarity(doc_emb, qa_emb)

| IPO | 内容                                        |
|-----|-------------------------------------------|
| I   | doc_emb (np.ndarray), qa_emb (np.ndarray) |
| P   | 正規化チェック → 内積 or 完全計算 → ゼロ除算チェック           |
| O   | float (類似度スコア: -1.0 ~ 1.0)                |

---
データフロー図

[文書テキスト]
  ↓
[create_semantic_chunks]
  ├→ _split_into_sentences
  ├→ トークンカウント & グループ化
  └→ _adjust_chunks_for_topic_continuity
  ↓
[チャンクリスト]
  ↓
[generate_embeddings]
  ├→ バッチ処理（20個ずつ）
  ├→ OpenAI API呼び出し
  └→ L2正規化
  ↓
[埋め込みベクトル配列]
  ↓
[cosine_similarity] ← [Q&Aペアの埋め込み]
  ↓
[類似度スコア]

---
使用例

# 初期化
coverage = SemanticCoverage(embedding_model="text-embedding-3-small")

# 文書をチャンクに分割
chunks = coverage.create_semantic_chunks(document_text)

# チャンクの埋め込み生成
doc_embeddings = coverage.generate_embeddings(chunks)

# Q&Aペアの埋め込み生成
qa_embedding = coverage.generate_embedding("日本の首都は?")

# 類似度計算
  similarity = coverage.cosine_similarity(doc_embeddings[0], qa_embedding)



### ----------------------
#### 概要
文書を意味的にチャンク分割し、埋め込みベクトルを生成してQ/Aペアのカバレッジを測定する基盤クラス。

#### 処理の流れ
1. 文書テキスト入力
2. 日本語・英語の文末記号で文分割
3. トークン数を計算しながらチャンク構築（200トークン/チャンク）
4. 短すぎるチャンクは前チャンクとマージ
5. OpenAI Embeddings APIで埋め込み生成
6. L2正規化でベクトルを正規化

#### IPO (Input-Process-Output)

**Input:**
- `text`: str - 分析対象の文書テキスト
- `embedding_model`: str - 埋め込みモデル名（デフォルト: "text-embedding-3-small"）
- `verbose`: bool - 詳細ログ出力フラグ

**Process:**
- `create_semantic_chunks()`: 文書を意味的にチャンク分割
- `generate_embeddings()`: チャンクのバッチ埋め込み生成
- `generate_embedding()`: 単一テキストの埋め込み生成
- `cosine_similarity()`: コサイン類似度計算

**Output:**
```python
[
    {
        "id": "chunk_0",
        "text": "RAGシステムは...",
        "sentences": ["文1", "文2"],
        "start_sentence_idx": 0,
        "end_sentence_idx": 2
    },
    ...
]
```

#### 効果
- ✅ **意味的一貫性**: 文を単位としてチャンク化するため、意味が途中で切れない
- ✅ **カバレッジ測定**: Q/Aペアが文書をどれだけカバーしているか定量化
- ✅ **埋め込み品質**: OpenAI Embeddings APIによる高品質なベクトル表現

#### 評価
| 項目 | 評価 | 備考 |
|------|------|------|
| 処理速度 | ⭐⭐⭐⭐ | 高速（API呼び出しは必要） |
| 精度 | ⭐⭐⭐⭐⭐ | OpenAI Embeddings使用で高精度 |
| コスト効率 | ⭐⭐⭐ | Embeddings API呼び出しコストあり |
| 拡張性 | ⭐⭐⭐⭐⭐ | 他の手法の基盤として機能 |

---

### 3.2 RuleBasedQAGenerator

#### 概要
正規表現パターンマッチングにより、定義文・事実情報から確実にQ/Aペアを抽出する。

#### 処理の流れ
1. テキスト入力
2. 定義文パターンマッチング（`〜とは〜である`）
3. 呼称パターンマッチング（`〜は〜と呼ばれる`）
4. 列挙パターンマッチング（`〜には、A、B、Cがある`）
5. Q/Aペア生成（信頼度スコア付き）

#### IPO (Input-Process-Output)

**Input:**
- `text`: str - 抽出対象のテキスト
- `language`: str - 言語コード（"ja" or "en"）

**Process:**
```python
# パターン1: 定義文
pattern = r'([^。]+)とは([^。]+)(?:である|です)'
matches = re.findall(pattern, text)

for term, definition in matches:
    qa_pairs.append({
        "question": f"{term}とは何ですか？",
        "answer": f"{term}とは{definition}です。",
        "type": "definition",
        "confidence": 0.9
    })
```

**Output:**
```python
[
    {
        "question": "RAGシステムとは何ですか？",
        "answer": "RAGシステムとはRetrieval-Augmented Generationの略で、検索拡張生成と呼ばれます。",
        "type": "definition",
        "confidence": 0.9
    }
]
```

#### 効果
- ✅ **確実性**: パターンマッチングによる100%再現性
- ✅ **コスト**: API呼び出し不要
- ✅ **高速**: 正規表現による高速処理
- ⚠️ **カバレッジ**: パターンに一致する箇所のみ

#### 評価
| 項目 | 評価 | 備考 |
|------|------|------|
| 処理速度 | ⭐⭐⭐⭐⭐ | 非常に高速（API不要） |
| 精度 | ⭐⭐⭐⭐ | パターン一致時は高精度 |
| 多様性 | ⭐⭐ | パターン依存で限定的 |
| コスト効率 | ⭐⭐⭐⭐⭐ | APIコストなし |
| 依存性 | ⚠️ | spaCy日本語モデル必要 |

---

### 3.3 TemplateBasedQAGenerator

#### 概要
事前定義されたテンプレートとエンティティを組み合わせて構造化されたQ/Aペアを生成。

#### 処理の流れ
1. エンティティ抽出（手動指定または自動抽出）
2. エンティティタイプ判定
3. 対応するテンプレート選択
4. テンプレートに変数を埋め込み
5. Q/Aペア生成

#### IPO (Input-Process-Output)

**Input:**
- `text`: str - 対象テキスト
- `entities`: List[str] - エンティティリスト（例: ['AI', '機械学習', 'RAG']）

**Process:**
```python
templates = {
    "comparison": [
        "{A}と{B}の違いは何ですか？",
        "{A}と{B}の共通点は何ですか？"
    ],
    "characteristics": [
        "{entity}の特徴は何ですか？",
        "{entity}はどのように使用されますか？"
    ]
}

for entity in entities:
    template = select_template(entity_type)
    qa = apply_template(template, entity, text)
    qa_pairs.append(qa)
```

**Output:**
```python
[
    {
        "question": "AIとは何ですか？",
        "answer": "AIに関する情報は文書内で説明されています。",
        "entity": "AI",
        "type": "entity_based",
        "confidence": 0.75
    }
]
```

#### 効果
- ✅ **構造化**: 一貫した質問形式
- ✅ **網羅性**: エンティティごとに生成
- ✅ **カスタマイズ性**: テンプレート編集で柔軟に対応
- ⚠️ **汎用性**: テンプレート設計次第

#### 評価
| 項目 | 評価 | 備考 |
|------|------|------|
| 処理速度 | ⭐⭐⭐⭐⭐ | 非常に高速 |
| 精度 | ⭐⭐⭐ | テンプレート品質依存 |
| 多様性 | ⭐⭐⭐ | テンプレート数に依存 |
| コスト効率 | ⭐⭐⭐⭐⭐ | APIコストなし |
| 保守性 | ⭐⭐⭐⭐ | テンプレート管理が必要 |

---

### 3.4 LLMBasedQAGenerator

#### 概要
OpenAI APIを使用して、多様で自然なQ/Aペアを生成する。

#### 処理の流れ
1. テキスト入力とQ/A数決定
2. 言語別プロンプト構築
3. OpenAI API呼び出し（responses.parse）
4. JSON形式でQ/Aペア受信
5. メタデータ付与

#### IPO (Input-Process-Output)

**Input:**
- `text`: str - 対象テキスト
- `num_pairs`: int - 生成するQ/A数
- `model`: str - 使用モデル（デフォルト: "gpt-4o-mini"）

**Process:**
```python
prompt = f"""
以下のテキストから{num_pairs}個の質問と回答のペアを生成してください。

要件：
1. 質問は具体的で明確にする
2. 回答はテキストから直接答えられるものにする
3. 質問の種類を多様にする（What/Why/How/When/Where）
4. 回答は簡潔かつ正確にする

テキスト：{text[:3000]}

出力形式（JSON）：
{{
    "qa_pairs": [
        {{
            "question": "質問",
            "answer": "回答",
            "question_type": "factual/causal/comparative",
            "difficulty": "basic/intermediate/advanced"
        }}
    ]
}}
"""

response = client.responses.parse(
    input=prompt,
    model=model,
    text_format=QAPairsResponse
)
```

**Output:**
```python
[
    {
        "question": "RAGシステムの主な利点は何ですか？",
        "answer": "外部知識ベースから関連情報を取得し、より正確な回答を生成できることです。",
        "question_type": "factual",
        "difficulty": "basic"
    }
]
```

#### 効果
- ✅ **多様性**: 様々な視点からの質問生成
- ✅ **自然さ**: 人間らしい表現
- ✅ **柔軟性**: プロンプト調整で出力制御
- ⚠️ **コスト**: API呼び出しコストあり

#### 評価
| 項目 | 評価 | 備考 |
|------|------|------|
| 処理速度 | ⭐⭐⭐ | API応答時間に依存 |
| 精度 | ⭐⭐⭐⭐ | モデル性能依存 |
| 多様性 | ⭐⭐⭐⭐⭐ | 非常に多様な質問生成 |
| コスト効率 | ⭐⭐ | API呼び出しコスト高 |
| 再現性 | ⭐⭐⭐ | temperature設定で制御 |

---

### 3.5 ChainOfThoughtQAGenerator

#### 概要
LLMに推論過程を段階的に説明させながら、高品質なQ/Aペアを生成する。

#### 処理の流れ
1. テキスト入力
2. Chain-of-Thoughtプロンプト構築（5ステップ）
3. OpenAI API呼び出し（chat + JSON mode）
4. 分析結果とQ/Aペア受信
5. 推論過程と信頼度スコア付与

#### IPO (Input-Process-Output)

**Input:**
- `text`: str - 対象テキスト
- `model`: str - 使用モデル（デフォルト: "gpt-4o-mini"）

**Process:**
```python
prompt = f"""
以下のテキストから質の高いQ/Aペアを生成します。
各ステップを踏んで考えてください。

ステップ1: テキストの主要なトピックと概念を抽出
ステップ2: 各トピックについて重要な情報を特定
ステップ3: その情報を問う質問を設計
ステップ4: テキストから正確な回答を抽出
ステップ5: 質問と回答の妥当性を検証

テキスト：{text}

必ずJSON形式で出力してください：
{{
    "analysis": {{
        "main_topics": ["トピック1", "トピック2"],
        "key_concepts": ["概念1", "概念2"],
        "information_density": "high/medium/low"
    }},
    "qa_pairs": [{{
        "question": "質問",
        "answer": "回答",
        "reasoning": "なぜこの質問が重要か",
        "confidence": 0.95
    }}]
}}
"""
```

**Output:**
```python
{
    "analysis": {
        "main_topics": ["RAG", "Qdrant", "OpenAI"],
        "key_concepts": ["検索拡張生成", "ベクトルDB"],
        "information_density": "high"
    },
    "qa_pairs": [{
        "question": "RAGシステムの主な利点は何ですか？",
        "answer": "外部知識ベースから関連情報を取得し、より正確な回答を生成できること",
        "reasoning": "テキスト中で明示的に述べられている主要な利点",
        "confidence": 0.92
    }]
}
```

#### 効果
- ✅ **品質**: 段階的思考による高品質Q/A
- ✅ **説明可能性**: 推論過程が明示的
- ✅ **信頼度**: スコアによる品質評価
- ⚠️ **コスト**: 高い（プロンプトが長い）

#### 評価
| 項目 | 評価 | 備考 |
|------|------|------|
| 処理速度 | ⭐⭐ | 低速（複雑なプロンプト） |
| 精度 | ⭐⭐⭐⭐⭐ | 非常に高精度 |
| 多様性 | ⭐⭐⭐⭐ | 高い |
| コスト効率 | ⭐ | 非常に高コスト |
| 説明可能性 | ⭐⭐⭐⭐⭐ | reasoning付きで透明 |

---

### 3.6 HybridQAGenerator

#### 概要
複数の生成手法（ルールベース、テンプレートベース、LLMベース）を統合し、品質検証を実施。

#### 処理の流れ
1. Phase 1: ルールベースQ/A生成（信頼度 >= 0.7 のみ採用）
2. Phase 2: テンプレートベースQ/A生成（重複除去）
3. Phase 3: LLM補完（カバーされていない領域を特定）
4. Phase 4: 品質検証（妥当性・明確性・矛盾チェック）
5. Phase 5: 統合・ソート（品質スコア順）

#### IPO (Input-Process-Output)

**Input:**
- `text`: str - 対象テキスト
- `target_count`: int - 目標Q/A数

**Process:**
```python
# Phase 1: ルールベース
rule_qas = rule_generator.extract_definition_qa(text)
rule_qas += rule_generator.extract_fact_qa(text)

# Phase 2: テンプレートベース
entities = extract_entities(text)
template_qas = template_generator.generate_from_entities(entities, text)

# Phase 3: LLM補完
covered_topics = analyze_coverage(rule_qas + template_qas)
uncovered_count = target_count - len(rule_qas + template_qas)
llm_qas = llm_generator.generate_diverse_qa(text, uncovered_count)

# Phase 4: 品質検証
all_qas = rule_qas + template_qas + llm_qas
validated_qas = [qa for qa in all_qas if validate_qa(qa, text)]

# Phase 5: 統合・ソート
final_qas = remove_duplicates(validated_qas)
final_qas.sort(key=lambda x: x['quality_score'], reverse=True)
```

**Output:**
```python
[
    {
        "question": "質問",
        "answer": "回答",
        "source": "rule_based/template/llm",
        "quality_score": 0.95,
        "validation": {
            "answer_found": True,
            "question_clear": True,
            "no_contradiction": True
        }
    }
]
```

#### 効果
- ✅ **総合品質**: 複数手法の長所を統合
- ✅ **カバレッジ**: 高い文書カバレッジ
- ✅ **信頼性**: 品質検証による保証
- ⚠️ **複雑性**: 実装が複雑

#### 評価
| 項目 | 評価 | 備考 |
|------|------|------|
| 処理速度 | ⭐⭐⭐ | 中速（複数手法実行） |
| 精度 | ⭐⭐⭐⭐⭐ | 非常に高精度 |
| 多様性 | ⭐⭐⭐⭐⭐ | 非常に多様 |
| コスト効率 | ⭐⭐⭐ | 中程度（LLM部分のみコスト） |
| 総合品質 | ⭐⭐⭐⭐⭐ | 最高品質 |

---

### 3.7 AdvancedQAGenerationTechniques

#### 概要
システムの堅牢性を評価するための高度なQ/A生成（敵対的、マルチホップ、反事実的）。

#### 処理の流れ
1. 既存Q/Aペアの分析
2. 敵対的Q/A生成（否定質問、文脈外質問）
3. マルチホップ推論Q/A生成（複数情報の組み合わせ）
4. 反事実的Q/A生成（"もし〜だったら"形式）

#### IPO (Input-Process-Output)

**Input:**
- `text`: str - 対象テキスト
- `existing_qa`: List[Dict] - 既存のQ/Aペア

**Process:**
```python
adversarial_qa = []

# 1. 否定質問
for qa in existing_qa[:5]:
    adversarial_qa.append({
        "question": qa['question'].replace("何ですか", "何ではありませんか"),
        "answer": f"{qa['answer']}ではないものを指します。",
        "type": "adversarial_negation"
    })

# 2. 文脈外質問
adversarial_qa.append({
    "question": "この文書に書かれていない情報は何ですか？",
    "answer": "文書には含まれていない情報です。",
    "type": "out_of_context"
})

# 3. マルチホップ推論
# 複数の事実を組み合わせた質問生成

# 4. 反事実的質問
adversarial_qa.append({
    "question": "もし{concept}が存在しなかったら、どうなりますか？",
    "answer": "代替手段として{alternative}が使われるでしょう。",
    "type": "counterfactual"
})
```

**Output:**
```python
[
    {
        "question": "RAGシステムは何ではありませんか？",
        "answer": "単純な検索システムではありません。",
        "type": "adversarial_negation",
        "difficulty": "advanced"
    },
    {
        "question": "もしベクトルデータベースがなかったら、RAGシステムはどう実装されますか？",
        "answer": "従来のキーワード検索や全文検索に依存することになります。",
        "type": "counterfactual",
        "difficulty": "advanced"
    }
]
```

#### 効果
- ✅ **堅牢性テスト**: システムの限界を評価
- ✅ **推論能力評価**: 複雑な推論が必要
- ✅ **エッジケース**: 通常の質問では見つからない問題発見
- ⚠️ **実用性**: 特定の評価目的向け

#### 評価
| 項目 | 評価 | 備考 |
|------|------|------|
| 処理速度 | ⭐⭐⭐⭐ | 高速（現状ルールベース） |
| 精度 | ⭐⭐⭐ | 評価用途では有効 |
| 多様性 | ⭐⭐⭐⭐ | 高度な質問タイプ |
| コスト効率 | ⭐⭐⭐⭐⭐ | APIコストなし（現状） |
| 実装状態 | ⚠️ | 一部シミュレーション |

---

### 3.8 QAGenerationOptimizer

#### 概要
生成されたQ/Aペアのカバレッジを分析し、最適化推奨を提供する。

#### 処理の流れ
1. 文書チャンクの埋め込み生成
2. Q/Aペアの埋め込み生成（質問+回答を結合）
3. カバレッジ計算（コサイン類似度 >= 0.7）
4. カバーされていないチャンクの特定
5. 追加Q/A生成の推奨

#### IPO (Input-Process-Output)

**Input:**
- `analyzer`: SemanticCoverage - 分析器インスタンス
- `chunks`: List[Dict] - 文書チャンク
- `all_qas`: List[Dict] - 既存Q/Aペア

**Process:**
```python
# 1. 文書チャンクの埋め込み生成
doc_embeddings = analyzer.generate_embeddings(chunks)

# 2. Q/Aペアの埋め込み生成
qa_texts = [qa['question'] + ' ' + qa['answer'] for qa in all_qas]
qa_embeddings = [analyzer.generate_embedding(text) for text in qa_texts]

# 3. カバレッジ計算（閾値0.7）
threshold = 0.7
covered_chunks = set()

for qa_emb in qa_embeddings:
    for i, doc_emb in enumerate(doc_embeddings):
        similarity = analyzer.cosine_similarity(doc_emb, qa_emb)
        if similarity >= threshold:
            covered_chunks.add(i)

# 4. カバレッジ率計算
coverage_rate = len(covered_chunks) / len(chunks)
```

**Output:**
```
✅ カバレッジ分析完了
  総チャンク数: 5
  カバーされたチャンク: 4
  カバレッジ率: 80.0%
  総Q/A数: 12

💡 推奨: カバーされていないチャンクが1個あります
   追加で2個程度のQ/Aペア生成を推奨します
```

#### 効果
- ✅ **定量的分析**: 数値によるカバレッジ測定
- ✅ **最適化推奨**: 具体的な改善提案
- ✅ **ギャップ特定**: カバーされていない領域を明示
- ⚠️ **コスト**: 埋め込み生成コストあり

#### 評価
| 項目 | 評価 | 備考 |
|------|------|------|
| 処理速度 | ⭐⭐⭐ | 中速（埋め込み生成） |
| 精度 | ⭐⭐⭐⭐⭐ | 埋め込みベースで高精度 |
| 有用性 | ⭐⭐⭐⭐⭐ | 具体的な改善指標 |
| コスト効率 | ⭐⭐ | Embeddings APIコスト |
| 実用性 | ⭐⭐⭐⭐⭐ | 継続的改善に有効 |

