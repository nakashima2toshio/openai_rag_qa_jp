# a50_rag_search_local_qdrant.py

## 概要

Qdrant RAG検索用のStreamlit UIアプリケーション。ローカルまたはリモートのQdrantベクトルデータベースに対して、自然言語クエリを使用したセマンティック検索を実行し、関連するQ&Aペアを取得して表示します。

## 対応データセット

### 基本コレクション
- `product_embeddings`: 製品情報検索用（384次元）
- `qa_corpus`: Q&Aコーパス検索用（1536次元、5つのドメインをサポート）

### CC News データセット（7,376件の生データから派生）
- `raw_cc_news`: CC Newsの生データ（7,376件）
- `qa_cc_news_a02_llm`: LLM生成方式によるQ&Aペア（1,344件）
  - AIによる自然な質問・回答ペアの生成
- `qa_cc_news_a03_rule`: ルールベース生成方式によるQ&Aペア（1,638件）
  - テンプレートベースの構造化されたQ&A生成
- `qa_cc_news_a10_hybrid`: ハイブリッド生成方式によるQ&Aペア（1,638件）
  - LLMとルールベースを組み合わせた生成

### Livedoor News データセット（7,376件の生データから派生）
- `raw_livedoor`: Livedoorの生データ（7,376件）
- `qa_livedoor_a02_20_llm`: LLM生成方式によるQ&Aペア（1,317件）
  - AIによる自然な質問・回答ペアの生成
- `qa_livedoor_a03_rule`: ルールベース生成方式によるQ&Aペア（1,638件）
  - テンプレートベースの構造化されたQ&A生成
- `qa_livedoor_a10_hybrid`: ハイブリッド生成方式によるQ&Aペア（1,638件）
  - LLMとルールベースを組み合わせた生成

## 主な機能

### 1. 複数コレクション対応
- 動的にQdrantサーバーから利用可能なコレクションを取得
- 各コレクションごとに最適な埋め込み設定を自動適用
- コレクション情報（モデル・次元数）をサイドバーに表示

### 2. ドメイン別検索（qa_corpusコレクションのみ）
以下の5つのドメインをサポート：
- **customer**: カスタマーサポート・FAQ
- **medical**: 医療QAデータ
- **sciq**: 科学・技術QA
- **legal**: 法律・判例QA
- **trivia**: TriviaQA（トリビアQA）

※他のコレクションではドメインフィルタリングは無効化され、"ALL"として扱われます

### 3. 多言語対応
- OpenAIの埋め込みモデルによる多言語サポート
- 日本語クエリで英語データを検索可能（逆も可能）
- 翻訳不要で日英間の意味的検索を実現
- 例: 日本語「返金は可能ですか？」と英語「Can I get a refund?」の高い類似度（0.4957）

### 4. 動的埋め込み次元対応
- text-embedding-3系モデルで次元数の動的指定をサポート
- 384次元: 高速処理用（product_embeddings）
- 1536次元: 高精度用（qa_corpus、CC News、Livedoor）
- コレクションごとに最適な次元数を自動設定

### 5. Named Vectors切替
config.ymlで定義された複数の埋め込みモデル設定を切替可能：
- primary: text-embedding-3-small (1536次元)
- ada-002: text-embedding-ada-002 (1536次元)
- 3-small: text-embedding-3-small (1536次元)

### 6. リアルタイム類似度スコア表示
類似度スコアの閾値目安：
- **0.8以上**: 非常に関連性が高い（ほぼ一致）
- **0.6-0.8**: 関連性がある（有用な結果）
- **0.4-0.6**: 部分的に関連（参考程度）
- **0.4未満**: 関連性が低い（フィルタリング推奨）

### 7. OpenAI GPT統合
- 検索結果を基にGPT-4o-miniで日本語回答を生成
- 検索結果の最高スコアのQ&Aペアを利用
- ユーザーの元の質問と検索結果を組み合わせて回答を生成

### 8. サンプル質問機能
- コレクション・ドメインに応じた質問例を自動表示
- CC Newsコレクション用の英語質問サンプル（5件）
- Livedoorコレクション用の日本語質問サンプル（5件）
- ドメイン別質問サンプル（customer, medical, legal, sciq, trivia）
- ワンクリックでサンプル質問を入力フィールドに設定

## 起動方法

```bash
# デフォルトポート（8501）で起動
streamlit run a50_rag_search_local_qdrant.py

# カスタムポート（8504）で起動
streamlit run a50_rag_search_local_qdrant.py --server.port=8504
```

## 前提条件

### 1. 環境変数
`.env`ファイルに以下を設定：
```bash
OPENAI_API_KEY=your-openai-api-key
```

### 2. Qdrantサーバー
```bash
# Dockerを使用する場合
cd docker-compose
docker-compose up -d

# ローカルでQdrantを起動する場合
qdrant
```

### 3. データ登録
```bash
# データをQdrantに登録
python a42_qdrant_registration.py --recreate --limit 100
```

## 設定ファイル

### config.yml
```yaml
rag:
  collection: "product_embeddings"  # デフォルトコレクション

embeddings:
  primary:
    provider: "openai"
    model: "text-embedding-3-small"
    dims: 1536
  ada-002:
    provider: "openai"
    model: "text-embedding-ada-002"
    dims: 1536

qdrant:
  url: "http://localhost:6333"
```

## UI構成

### サイドバー設定
- **Collection**: 検索対象のコレクションを選択
- **Using vector (named)**: 使用する埋め込みモデル設定を選択
- **Domain**: ドメインフィルタ（qa_corpusのみ）
- **TopK**: 取得する検索結果の数（1-20）
- **Qdrant URL**: Qdrantサーバーのアドレス
- **Debug Mode**: デバッグ情報の表示

### メイン画面
1. **クエリ入力フィールド**: 検索クエリを入力
2. **検索ボタン**: 検索を実行
3. **結果表示**:
   - DataFrameで全検索結果を表示
   - 最高スコアの結果を詳細表示
   - OpenAI応答を日本語で表示

### サイドバーの質問例
コレクション・ドメインに応じた質問例を自動表示：

**CC Newsコレクション選択時:**
- 英語での検索サンプル（5件）
- コレクションタイプに応じたラベル表示（生データ/LLM生成/ルールベース/ハイブリッド）

**Livedoorコレクション選択時:**
- 日本語での検索サンプル（5件）
- コレクションタイプに応じたラベル表示（生データ/LLM生成/ルールベース/ハイブリッド）

**qa_corpusコレクション + 特定ドメイン選択時:**
- 選択ドメインに対応した質問例（3-5件）

**qa_corpusコレクション + ALL選択時:**
- 各ドメイン（customer, medical, legal, sciq, trivia）から2件ずつ表示

**product_embeddingsコレクション選択時:**
- 製品関連の質問例を追加表示（4件）

## エラーハンドリング

### 接続エラー
- Qdrantサーバーへの接続失敗時に詳細なエラーメッセージを表示
- 解決方法の提案（Docker起動コマンド等）

### コレクションエラー
- コレクションが存在しない場合の対処法を表示
- データ登録コマンドの案内

### 埋め込み生成エラー
- モデルや次元数の不一致を検出して報告
- デバッグモードで詳細情報を表示

## 技術的詳細

### 埋め込みベクトル生成
```python
def embed_query(text: str, model: str, dims: Optional[int] = None) -> List[float]:
    """
    クエリテキストを埋め込みベクトルに変換
    text-embedding-3系モデルは次元数の動的指定をサポート
    """
    client = OpenAI()
    if dims and "text-embedding-3" in model:
        return client.embeddings.create(
            model=model,
            input=[text],
            dimensions=dims
        ).data[0].embedding
    else:
        return client.embeddings.create(
            model=model,
            input=[text]
        ).data[0].embedding
```

### ベクトル検索
```python
# Qdrantでの類似検索
hits = client.search(
    collection_name=collection,
    query_vector=qvec,
    limit=topk,
    query_filter=qfilter  # ドメインフィルタ（オプション、qa_corpusのみ）
)
```

### OpenAI応答生成（GPT-4o-mini使用）
```python
# 検索結果を基にした日本語回答生成
qa_prompt_jp = (
    "以下の検索結果（スコア・質問・回答）とユーザーの元の質問を踏まえて、"
    "日本語で簡潔かつ正確に回答してください。必要に応じて箇条書きを用いてください。\n\n"
    f"ユーザーの元の質問（query）:\n{query}\n\n"
    f"検索結果のスコア: {br_score:.4f}\n"
    f"検索結果の質問: {br_q}\n"
    f"検索結果の回答: {br_a}\n"
)

oai_client = OpenAI()
oai_resp = oai_client.responses.create(
    model="gpt-4o-mini",
    input=qa_prompt_jp
)
generated_answer = getattr(oai_resp, "output_text", None) or ""
```

### フィールドマッピング（柔軟な対応）
検索結果から以下のフィールドを抽出（複数のフィールド名に対応）：
- `score`: 類似度スコア
- `domain`: ドメイン情報（存在する場合）
- `question`: 質問テキスト
  - 対応フィールド名: `question`, `text`, `content`
  - テキストフィールドがない場合は最初に見つかった文字列フィールド（200文字まで）
- `answer`: 回答テキスト
  - 対応フィールド名: `answer`, `response`, `metadata`
- `source`: ソース情報
  - 対応フィールド名: `source`, `file`

※デバッグモードでpayload構造の詳細を確認可能

## 依存関係

```python
# 主要なライブラリ
pandas          # データフレーム処理
streamlit       # Web UI フレームワーク
qdrant-client   # Qdrantクライアント
openai          # OpenAI API
pyyaml          # 設定ファイル読み込み（オプション）
```

## 今後の改善点

1. **パフォーマンス最適化**
   - 埋め込みベクトルのキャッシング
   - バッチ処理の実装

2. **機能拡張**
   - より多くのコレクションタイプへの対応
   - カスタムフィルタの追加
   - 検索履歴の保存

3. **UI改善**
   - より直感的なフィルタリングインターフェース
   - 検索結果のビジュアライゼーション
   - エクスポート機能の追加