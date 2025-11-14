# a02_make_qa.py - Q&Aペア自動生成システム

## 目次
- [概要](#概要)
- [実行環境の準備](#実行環境の準備)
- [実行方法](#実行方法)
- [システムアーキテクチャ](#システムアーキテクチャ)
- [主要機能](#主要機能)
- [パフォーマンス特性](#パフォーマンス特性)
- [トラブルシューティング](#トラブルシューティング)

---

## 概要

`a02_make_qa.py`は、preprocessedされた文書データから自動的にQ&A（質問・回答）ペアを生成するシステムです。OpenAI APIを活用して高品質な学習用Q&Aペアを生成し、生成されたQ&Aペアの文書カバレージを分析します。

---

## 実行環境の準備

### 1. システム要件

- **Python**: 3.9以上推奨
- **OS**: Linux、macOS、Windows（WSL推奨）
- **メモリ**: 最小2GB、推奨4GB以上
- **ストレージ**: 1GB以上の空き容量

### 2. Pythonパッケージのインストール

プロジェクトルートディレクトリで以下のコマンドを実行：

```bash
# 依存パッケージをインストール
pip install -r requirements.txt
```

#### 主要な依存パッケージ：
- `openai>=2.6.1` - OpenAI API クライアント
- `pandas>=2.3.1` - データ処理
- `numpy>=2.3.2` - 数値計算
- `tiktoken>=0.11.0` - トークンカウント
- `pydantic>=2.11.7` - データバリデーション
- `python-dotenv>=1.1.1` - 環境変数管理
- `scikit-learn>=1.7.2` - 類似度計算
- `mecab-python3>=1.0.10` - 日本語形態素解析（オプション）

### 3. 環境変数の設定

プロジェクトルートに `.env` ファイルを作成し、以下の内容を設定：

```bash
# OpenAI API Key (必須)
OPENAI_API_KEY=sk-proj-YOUR_API_KEY_HERE

# Optional: Qdrant URL (デフォルト: http://localhost:6333)
QDRANT_URL=http://localhost:6333

# Optional: PostgreSQL接続文字列
PG_CONN_STR=postgresql://user:pass@localhost:5432/dbname
```

**重要**:
- OpenAI APIキーは必須です。[OpenAI Platform](https://platform.openai.com/)で取得してください。
- APIキーは絶対に公開リポジトリにコミットしないでください。

### 4. データの準備

処理対象の preprocessed データファイルが `OUTPUT/` ディレクトリに存在することを確認：

```bash
# 必要なファイル（いずれか）
OUTPUT/preprocessed_cc_news.csv        # CC-News英語ニュース
OUTPUT/preprocessed_japanese_text.csv  # 日本語Webテキスト
OUTPUT/preprocessed_wikipedia_ja.csv   # Wikipedia日本語版
```

各CSVファイルには以下のカラムが必要：
- `Combined_Text`: 文書の本文
- `title`: 文書タイトル（オプション、データセットによる）

**データの生成方法**:
```bash
# a01_load_set_rag_data.py でデータを準備（前処理）
python a01_load_set_rag_data.py --dataset cc_news
```

### 5. MeCabのインストール（オプション、日本語処理の精度向上）

MeCabは日本語の形態素解析に使用されます。インストールしない場合は自動的に正規表現ベースの処理にフォールバックします。

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8
pip install mecab-python3
```

#### macOS:
```bash
brew install mecab mecab-ipadic
pip install mecab-python3
```

#### Windows:
- [MeCab公式サイト](https://taku910.github.io/mecab/)からインストーラーをダウンロード
- インストール後、`pip install mecab-python3`

**確認方法**:
```python
# Pythonで確認
import MeCab
tagger = MeCab.Tagger()
print(tagger.parse("テスト"))  # 正常に動作すればOK
```

### 6. 出力ディレクトリの作成

```bash
# 出力ディレクトリを作成
mkdir -p qa_output/a02
```

---

## 実行方法

### 基本的な使い方

```bash
# 最小構成での実行
python a02_make_qa.py --dataset cc_news

# カバレージ分析を含む実行
python a02_make_qa.py --dataset cc_news --analyze-coverage

# テスト実行（最初の10文書のみ処理）
python a02_make_qa.py --dataset cc_news --max-docs 10 --analyze-coverage
```

### コマンドラインオプション

#### 必須オプション

| オプション | 説明 | デフォルト | 選択肢 |
|----------|------|-----------|--------|
| `--dataset` | 処理するデータセット | `cc_news` | `cc_news`, `japanese_text`, `wikipedia_ja` |

#### 推奨オプション

| オプション | 説明 | デフォルト | 推奨値 |
|----------|------|-----------|--------|
| `--analyze-coverage` | カバレージ分析を実行 | `False` | 常に有効化推奨 |
| `--batch-chunks` | 1回のAPIで処理するチャンク数 | `3` | `3`（品質重視）〜`5`（効率重視） |
| `--merge-chunks` | 小さいチャンクを統合 | `True` | 有効推奨 |

#### 詳細設定オプション

| オプション | 説明 | デフォルト | 範囲 |
|----------|------|-----------|------|
| `--model` | 使用するOpenAIモデル | `gpt-5-mini` | `gpt-4`, `gpt-5-mini` など |
| `--output` | 出力ディレクトリ | `qa_output/a02` | 任意のパス |
| `--max-docs` | 処理する最大文書数（テスト用） | `None`（全件） | 整数値 |
| `--min-tokens` | 統合対象の最小トークン数 | `150` | `100`〜`200` |
| `--max-tokens` | 統合後の最大トークン数 | `400` | `300`〜`500` |
| `--no-merge-chunks` | チャンク統合を無効化 | - | - |

### 実行例

#### 1. 本番運用向け（推奨設定）

```bash
# 高品質なQ&A生成とカバレージ分析
python a02_make_qa.py \
    --dataset cc_news \
    --batch-chunks 3 \
    --merge-chunks \
    --min-tokens 100 \
    --max-tokens 300 \
    --model gpt-5-mini \
    --analyze-coverage
```

**処理時間の見積もり（497文書の場合）**:
- 処理文書数: 497件
- チャンク数: 約1,825個 → 統合後 約1,820個
- API呼び出し: 約365回（バッチサイズ3）
- 推定実行時間: 60-75分
- カバレージ分析: +3-5分
- **合計**: 約65-80分

#### 2. テスト実行（開発・検証用）

```bash
# 最初の10文書のみ処理
python a02_make_qa.py \
    --dataset cc_news \
    --max-docs 10 \
    --analyze-coverage
```

**処理時間**: 約2-3分

#### 3. 日本語テキスト処理

```bash
# 日本語Webテキストの処理
python a02_make_qa.py \
    --dataset japanese_text \
    --model gpt-5-mini \
    --analyze-coverage \
    --max-docs 10

# Wikipedia日本語版の処理
python a02_make_qa.py \
    --dataset wikipedia_ja \
    --model gpt-5-mini \
    --analyze-coverage \
    --max-docs 10
```

#### 4. 高速処理（効率重視）

```bash
# バッチサイズを大きくして高速化（品質は若干低下）
python a02_make_qa.py \
    --dataset cc_news \
    --batch-chunks 5 \
    --merge-chunks \
    --model gpt-5-mini \
    --analyze-coverage
```

### 出力ファイル

実行が完了すると、`qa_output/a02/` ディレクトリに以下のファイルが生成されます：

```
qa_output/a02/
├── qa_pairs_cc_news_20251114_123045.json      # Q&Aペア（JSON形式）
├── qa_pairs_cc_news_20251114_123045.csv       # Q&Aペア（CSV形式）
├── coverage_cc_news_20251114_123045.json      # カバレージ分析結果
└── summary_cc_news_20251114_123045.json       # サマリー情報
```

#### ファイルの内容

**1. qa_pairs_{dataset}_{timestamp}.json**
```json
[
  {
    "question": "What is the main topic of the article?",
    "answer": "The article discusses...",
    "question_type": "fact",
    "source_chunk_id": "cc_news_0_chunk_0",
    "doc_id": "cc_news_0",
    "dataset_type": "cc_news",
    "chunk_idx": 0
  }
]
```

**2. coverage_{dataset}_{timestamp}.json**
```json
{
  "coverage_rate": 0.706,
  "covered_chunks": 1285,
  "total_chunks": 1820,
  "multi_threshold": {
    "strict": {"threshold": 0.80, "coverage_rate": 0.543},
    "standard": {"threshold": 0.70, "coverage_rate": 0.706},
    "lenient": {"threshold": 0.60, "coverage_rate": 0.891}
  },
  "chunk_analysis": {
    "by_length": {...},
    "by_position": {...}
  }
}
```

**3. summary_{dataset}_{timestamp}.json**
```json
{
  "dataset_type": "cc_news",
  "dataset_name": "CC-News英語ニュース",
  "generated_at": "20251114_123045",
  "total_qa_pairs": 4646,
  "coverage_rate": 0.706,
  "covered_chunks": 1285,
  "total_chunks": 1820
}
```

### 実行ログの見方

```
2025-11-14 12:30:00 - INFO - =====================================
2025-11-14 12:30:00 - INFO - Q/Aペア生成開始
2025-11-14 12:30:00 - INFO - =====================================
2025-11-14 12:30:00 - INFO - データセット: CC-News英語ニュース
2025-11-14 12:30:00 - INFO - モデル: gpt-5-mini
2025-11-14 12:30:00 - INFO -
2025-11-14 12:30:05 - INFO - [1/4] データ読み込み...
2025-11-14 12:30:07 - INFO - 読み込み完了: 497件のデータ
2025-11-14 12:30:07 - INFO - [2/4] チャンク作成...
2025-11-14 12:30:10 - INFO - チャンク作成完了: 1825個のチャンク
2025-11-14 12:30:10 - INFO - [3/4] Q/Aペア生成...
2025-11-14 12:30:10 - INFO - バッチ 1/365 処理中 (3チャンク)...
...
2025-11-14 13:45:00 - INFO - Q/Aペア生成完了: 4646個
2025-11-14 13:45:00 - INFO - [4/4] カバレージ分析...
2025-11-14 13:50:00 - INFO - カバレージ分析完了
2025-11-14 13:50:05 - INFO - 処理完了
```

---

## システムアーキテクチャ（処理フロー図）
### 詳細内容

1. バッチ処理の明確化

- 1チャンク vs 2-5チャンクの分岐
- 単一処理とバッチ処理の違いを明示

2. データセット別設定

- cc_news → 基本数5
- japanese_text → 基本数2
- wikipedia_ja → 基本数3

3. Q&A数決定の詳細プロセス

- トークン数解析ステップを追加
- 各トークン範囲での基本数設定を明示
- チャンク位置確認と位置補正判定を分離
- 上限チェック（8個制限）を明示

4. プロンプト生成の詳細

- 言語別プロンプト作成の分岐
- 質問タイプ指定（fact/reason/comparison/application）
- プロンプト構築プロセス
- バッチ統合（単一/複数）の処理

5. エラー処理とリトライロジック

- レスポンス確認の分岐
- リトライ可否の判定
- 指数バックオフ待機
- フォールバック処理への遷移

```mermaid
flowchart TD
    Start([開始]) --> LoadData[データ読み込み]
    LoadData --> CSV[(preprocessedファイル)]
    CSV --> CreateChunks[チャンク作成]

    CreateChunks --> MecabChunk[MeCabチャンク分割]
    MecabChunk --> LangCheck{言語判定}
    LangCheck -->|日本語| JaSplit[日本語文分割]
    LangCheck -->|英語| EnSplit[英語文分割]

    JaSplit --> TokenCount[トークン数計算]
    EnSplit --> TokenCount

    TokenCount --> ChunkSize{サイズ判定}
    ChunkSize -->|適切| AddChunk[チャンク追加]
    ChunkSize -->|大きい| SplitWords[単語分割]
    SplitWords --> AddChunk

    AddChunk --> MergeOpt{統合オプション}
    MergeOpt -->|Yes| MergeSmall[小チャンク統合]
    MergeOpt -->|No| GenerateQA[Q&A生成]

    MergeSmall --> CheckTokens{統合可能判定}
    CheckTokens -->|可能| MergeChunks[チャンク統合]
    CheckTokens -->|不可| GenerateQA
    MergeChunks --> GenerateQA

    GenerateQA --> BatchCheck{バッチサイズ判定}
    BatchCheck -->|1チャンク| SingleGen[単一チャンク処理]
    BatchCheck -->|2-5チャンク| BatchGen[バッチ処理]

    SingleGen --> GetConfig[設定取得]
    BatchGen --> GetConfig

    GetConfig --> DatasetType{データセット判定}
    DatasetType -->|cc_news| BaseCount5[基本数5]
    DatasetType -->|japanese_text| BaseCount2[基本数2]
    DatasetType -->|wikipedia_ja| BaseCount3[基本数3]

    BaseCount5 --> CountQA[Q&A数決定]
    BaseCount2 --> CountQA
    BaseCount3 --> CountQA

    CountQA --> TokenAnalysis[トークン数解析]
    TokenAnalysis --> TokenBased{トークン数判定}
    TokenBased -->|50未満| QA2[基本2個]
    TokenBased -->|50-100| QA3[基本3個]
    TokenBased -->|100-200| QABase1[基本数+1]
    TokenBased -->|200-300| QABase2[基本数+2]
    TokenBased -->|300以上| QABase3[基本数+3]

    QA2 --> ChunkPos[チャンク位置確認]
    QA3 --> ChunkPos
    QABase1 --> ChunkPos
    QABase2 --> ChunkPos
    QABase3 --> ChunkPos

    ChunkPos --> PosCheck{位置補正判定}
    PosCheck -->|6番目以降| AddOne[+1個追加]
    PosCheck -->|5番目以前| FinalCount[最終Q&A数確定]
    AddOne --> FinalCount

    FinalCount --> MaxCheck{上限チェック}
    MaxCheck -->|8以下| PrepPrompt[プロンプト準備]
    MaxCheck -->|8超過| Cap8[8個に制限]
    Cap8 --> PrepPrompt

    PrepPrompt --> LangPrompt{言語別プロンプト}
    LangPrompt -->|日本語| JaPrompt[日本語プロンプト作成]
    LangPrompt -->|英語| EnPrompt[英語プロンプト作成]

    JaPrompt --> QTypes[質問タイプ指定]
    EnPrompt --> QTypes

    QTypes --> TypeList[fact/reason/comparison/application]
    TypeList --> BuildPrompt[プロンプト構築]

    BuildPrompt --> BatchMerge{バッチ統合}
    BatchMerge -->|単一| SinglePrompt[単一プロンプト]
    BatchMerge -->|複数| MergedPrompt[統合プロンプト]

    SinglePrompt --> CallAPI[OpenAI API呼び出し]
    MergedPrompt --> CallAPI

    CallAPI --> ResponseCheck{レスポンス確認}
    ResponseCheck -->|成功| ParseResp[Pydantic解析]
    ResponseCheck -->|エラー| RetryLogic{リトライ判定}

    RetryLogic -->|リトライ可| WaitBackoff[指数バックオフ待機]
    RetryLogic -->|リトライ不可| Fallback[個別処理フォールバック]
    WaitBackoff --> CallAPI
    Fallback --> SingleGen

    ParseResp --> ValidateQA[Q&A検証]
    ValidateQA --> AssignMeta[メタデータ付与]
    AssignMeta --> StoreQA[Q&Aペア保存]

    StoreQA --> CovCheck{カバレージ分析}
    CovCheck -->|Yes| AnalyzeCov[カバレージ分析実行]
    CovCheck -->|No| SaveFiles[結果保存]

    AnalyzeCov --> GenEmbed[埋め込み生成]
    GenEmbed --> DocEmbed[文書埋め込み]
    GenEmbed --> QAEmbed[Q&A埋め込み]

    DocEmbed --> CalcMatrix[行列計算]
    QAEmbed --> CalcMatrix

    CalcMatrix --> MultiThresh[多段階評価]
    MultiThresh --> Strict[Strict評価]
    MultiThresh --> Standard[Standard評価]
    MultiThresh --> Lenient[Lenient評価]

    CalcMatrix --> ChunkAnal[特性分析]
    ChunkAnal --> ByLength[長さ別分析]
    ChunkAnal --> ByPosition[位置別分析]

    ByLength --> Insights[インサイト]
    ByPosition --> Insights
    Strict --> CovResults[カバレージ結果]
    Standard --> CovResults
    Lenient --> CovResults
    Insights --> CovResults

    CovResults --> SaveFiles

    SaveFiles --> JSONFile[(JSON出力)]
    SaveFiles --> CSVFile[(CSV出力)]
    SaveFiles --> CovFile[(カバレージ)]
    SaveFiles --> SummaryFile[(サマリー)]

    JSONFile --> End([終了])
    CSVFile --> End
    CovFile --> End
    SummaryFile --> End
```

### 主要機能

1. **マルチデータセット対応**
   - CC-News英語ニュース
   - 日本語Webテキスト
   - Wikipedia日本語版

2. **インテリジェントな文書処理**
   - MeCabベースの文境界検出（日本語）
   - トークン数に基づく適切なサイズへの分割
   - 小さいチャンクの自動統合機能

3. **高効率Q&Aペア生成**
   - OpenAI APIを使用した自動生成
   - バッチ処理対応（3-5チャンクを同時処理）
   - 動的なQ&A数調整（チャンクサイズ・位置に基づく）
   - 4種類の質問タイプ（fact/reason/comparison/application）

4. **詳細なカバレージ分析**
   - 生成Q&Aペアの文書カバー率計算
   - 多段階閾値評価（strict/standard/lenient）
   - チャンク特性別分析（長さ別・位置別）
   - データセット別の最適閾値設定

---

## 1. Q&A生成戦略

### 1.1 基本アーキテクチャ

```
文書 → MeCabベースチャンキング → チャンク統合 → バッチQ&A生成 → 多段階カバレージ分析
```

### 1.2 LLMベースアプローチ
- **GPT-5-mini使用**: 高品質な自然言語Q&A生成
- **構造化出力**: Pydanticモデルによる型安全な出力
- **多様な質問タイプ**: fact、reason、comparison、application

---

## 2. チャンク処理戦略

### 2.1 MeCabベースチャンキング

```python
def create_mecab_chunks(text: str, lang: str = "ja", max_tokens: int = 200) -> List[Dict]:
    """
    MeCabを使った文境界検出によるチャンク作成
    - 文境界を保持（日本語: 「。」、英語: ". "）
    - 最大トークン数: 200-300（データセット依存）
    - チャンクID付与: doc_id_chunk_番号
    """
```

**特徴**:
- MeCab利用可能時は形態素解析で正確な文境界検出
- フォールバック: 正規表現による文分割
- トークン数制限でチャンクサイズ最適化

### 2.2 チャンク統合戦略

```python
def merge_small_chunks(chunks: List[Dict], min_tokens: int = 150, max_tokens: int = 400):
    """
    小さいチャンクを統合して適切なサイズに調整

    統合ルール:
    - min_tokens未満のチャンクは統合対象
    - 同じ文書のチャンクのみ統合
    - max_tokensを超えない範囲で統合
    """
```

**統合効果**:
- 元チャンク数の5-10%削減
- コンテキストの改善
- API呼び出し数の削減

---

## 3. 動的Q&A数調整メカニズム

### 3.1 改善版determine_qa_count関数

```python
def determine_qa_count(chunk: Dict, config: Dict) -> int:
    """チャンクに最適なQ/A数を決定（改善版）"""
    base_count = config["qa_per_chunk"]  # cc_news: 5
    token_count = len(tokenizer.encode(chunk['text']))
    chunk_position = chunk.get('chunk_idx', 0)

    # トークン数に基づく基本Q&A数
    if token_count < 50:
        qa_count = 2  # 旧: 1 → 新: 2
    elif token_count < 100:
        qa_count = 3  # 旧: 2 → 新: 3
    elif token_count < 200:
        qa_count = base_count + 1  # 6個
    elif token_count < 300:
        qa_count = base_count + 2  # 7個
    else:
        qa_count = base_count + 3  # 8個

    # 位置バイアス補正（文書後半+1）
    if chunk_position >= 5:
        qa_count += 1

    return min(qa_count, 8)  # 上限8個
```

### 3.2 チャンク特性別Q&A数

| チャンクタイプ | トークン数 | 基本Q&A数 | 位置補正後 |
|---------------|-----------|-----------|------------|
| Very Short | < 50 | 2 | 2-3 |
| Short | 50-100 | 3 | 3-4 |
| Medium | 100-200 | 6 | 6-7 |
| Long | 200-300 | 7 | 7-8 |
| Very Long | > 300 | 8 | 8 |

---

## 4. バッチQ&A生成

### 4.1 バッチ処理の仕組み

```python
def generate_qa_pairs_for_batch(chunks: List[Dict], config: Dict, model: str) -> List[Dict]:
    """
    複数チャンクから一度にQ/Aペアを生成

    処理フロー:
    1. 3-5チャンクをバッチ化
    2. 各チャンクのQ&A数を計算
    3. 統合プロンプトを生成
    4. 一度のAPI呼び出しで全Q&A生成
    """
```

### 4.2 プロンプト設計

#### 日本語用プロンプト
```python
system_prompt = """あなたは教育コンテンツ作成の専門家です。
複数の日本語テキストから、学習効果の高いQ&Aペアを生成してください。

生成ルール:
1. 質問は明確で具体的に
2. 回答は簡潔で正確に（1-2文程度）
3. テキストの内容に忠実に
4. 多様な観点から質問を作成"""

# 質問タイプの指定
question_types = """
- fact: 事実確認型（〜は何ですか？）
- reason: 理由説明型（なぜ〜ですか？）
- comparison: 比較型（〜と〜の違いは？）
- application: 応用型（〜はどのように活用されますか？）
"""
```

### 4.3 バッチ処理の利点
- **API効率**: 5チャンク個別 → 1回のAPI呼び出し（80%削減）
- **コンテキスト共有**: 複数チャンクの関連性を考慮
- **コスト削減**: API呼び出し数減少による費用削減

---

## 5. 多段階カバレージ分析

### 5.1 3段階評価システム

```python
OPTIMAL_THRESHOLDS = {
    "cc_news": {
        "strict": 0.80,    # 厳格: 高品質なマッチングのみ
        "standard": 0.70,  # 標準: 実用的な基準
        "lenient": 0.60    # 寛容: 緩い基準
    }
}
```

### 5.2 カバレージ分析の詳細

```python
def multi_threshold_coverage(coverage_matrix, chunks, qa_pairs, thresholds):
    """複数閾値でカバレージを評価"""

    for level, threshold in thresholds.items():
        # 各レベルでのカバレージ計算
        covered = sum(1 for s in max_similarities if s >= threshold)
        coverage_rate = covered / len(chunks)

        # 未カバーチャンクの特定
        uncovered_chunks = [
            chunk for chunk, sim in zip(chunks, max_similarities)
            if sim < threshold
        ]
```

### 5.3 チャンク特性別分析

```python
def analyze_chunk_characteristics(chunks, coverage_matrix):
    """チャンクの長さ・位置別カバレージ分析"""

    # 長さ別分類
    length_categories = {
        'short': token_count < 100,
        'medium': 100 <= token_count < 200,
        'long': token_count >= 200
    }

    # 位置別分類
    position_categories = {
        'beginning': position < total/3,
        'middle': total/3 <= position < 2*total/3,
        'end': position >= 2*total/3
    }
```

---

## 6. データセット設定

### 6.1 データセット別パラメータ

```python
DATASET_CONFIGS = {
    "cc_news": {
        "name": "CC-News英語ニュース",
        "chunk_size": 300,      # トークン数
        "qa_per_chunk": 5,      # 改善: 3→5
        "lang": "en"
    },
    "japanese_text": {
        "name": "日本語Webテキスト",
        "chunk_size": 200,
        "qa_per_chunk": 2,
        "lang": "ja"
    },
    "wikipedia_ja": {
        "name": "Wikipedia日本語版",
        "chunk_size": 250,
        "qa_per_chunk": 3,
        "lang": "ja"
    }
}
```

---

## 7. 実行パラメータと最適化

### 7.1 推奨実行コマンド

```bash
# 本番運用向け（改善版）
python a02_make_qa.py \
    --dataset cc_news \
    --batch-chunks 3 \      # 5→3: より丁寧な処理
    --merge-chunks \         # チャンク統合有効
    --min-tokens 100 \       # 150→100: 小チャンク削減
    --max-tokens 300 \       # 400→300: 過度な統合防止
    --model gpt-5-mini \
    --analyze-coverage       # カバレージ分析有効
```

### 7.2 パラメータの影響

| パラメータ | デフォルト | 推奨値 | 影響 |
|----------|-----------|--------|------|
| batch-chunks | 5 | 3 | API精度とのトレードオフ |
| min-tokens | 150 | 100 | 統合対象チャンクの閾値 |
| max-tokens | 400 | 300 | 統合後の最大サイズ |
| qa-per-chunk | 3 | 5 | カバレージ向上 |

---

## 8. パフォーマンス特性

### 8.1 実測値（497文書処理時）

| 指標 | 値 |
|------|-----|
| 処理文書数 | 497 |
| 元チャンク数 | 1,325 |
| 統合後チャンク数 | ~1,320 |
| 生成Q&A数 | 4,646 |
| API呼び出し数 | ~265回 |
| 推定実行時間 | 60-75分 |
| カバレージ（Standard） | 70.6% |

### 8.2 質問タイプ分布

```
- fact: 3,157件 (68.0%)        # 事実確認が最多
- reason: 804件 (17.3%)         # 理由説明
- application: 410件 (8.8%)     # 応用
- comparison: 274件 (5.9%)      # 比較
- explanation: 1件 (0.02%)      # 説明（稀）
```

### 8.3 カバレージ詳細

#### チャンク長別
- Short (<100 tokens): 68.0% ⚠️
- Medium (100-200): 81.4% ✅
- Long (>200): 68.7% ⚠️

#### 位置別
- Beginning: 73.5%
- Middle: 69.8% ⚠️
- End: 68.6% ⚠️

---

## 9. キーワード抽出機能

### 9.1 KeywordExtractorクラス

```python
class KeywordExtractor:
    """MeCabと正規表現を統合したキーワード抽出"""

    def extract(self, text: str, top_n: int = 5) -> List[str]:
        # MeCab優先、フォールバックで正規表現
        if self.mecab_available:
            return self._extract_with_mecab(text, top_n)
        else:
            return self._extract_with_regex(text, top_n)
```

**特徴**:
- MeCab利用時: 複合名詞の正確な抽出
- フォールバック: カタカナ語・漢字複合語・英数字の抽出
- ストップワード除去: 頻出する意味のない語を除外

---

## 10. エラー処理とリトライ

### 10.1 リトライメカニズム

```python
max_retries = 3
for attempt in range(max_retries):
    try:
        qa_pairs = generate_qa_pairs_for_batch(batch, config, model)
        break
    except Exception as e:
        if attempt == max_retries - 1:
            # 最終試行失敗時は個別処理にフォールバック
            for chunk in batch:
                qa_pairs = generate_qa_pairs_for_chunk(chunk)
        else:
            wait_time = 2 ** attempt  # 指数バックオフ
            time.sleep(wait_time)
```

---

## 11. 長所と短所

### 長所
✅ **高品質Q&A**: LLMによる自然な質問生成
✅ **詳細分析**: 多段階カバレージと特性別評価
✅ **柔軟性**: チャンク統合とバッチ処理の最適化
✅ **堅牢性**: リトライとフォールバック機能
✅ **日英対応**: 言語別の最適化プロンプト

### 短所
❌ **カバレージ**: Standard閾値で70.6%（改善余地あり）
❌ **処理時間**: 60-75分（大規模処理には時間がかかる）
❌ **API依存**: LLM APIコストが発生
❌ **位置バイアス**: 文書後半のカバレージが低い

---

## 12. 改善提案

### 12.1 カバレージ向上策
1. **qa_per_chunk**: 3→5に増加済み（更に6-7へ）
2. **バッチサイズ**: 5→3に削減済み（品質向上）
3. **未カバーチャンク**: 追加Q&A生成パスの実装

### 12.2 効率化策
1. **並列処理**: 複数バッチの並行処理
2. **キャッシュ**: 生成済みQ&Aの再利用
3. **増分処理**: 差分のみ処理

---

## 使用推奨シナリオ

### 最適な用途
- ✅ **高品質Q&A必要**: 自然な質問文が求められる場合
- ✅ **詳細分析必要**: カバレージの多角的評価が必要
- ✅ **中規模データ**: 数百〜数千文書の処理

### 不適切な用途
- ❌ **低予算**: API費用を最小化したい場合
- ❌ **超大規模**: 数万文書以上の処理
- ❌ **リアルタイム**: 即座の結果が必要な場合

---

## トラブルシューティング

### よくあるエラーと解決方法

#### 1. OpenAI APIキーエラー

**エラーメッセージ**:
```
ERROR - OPENAI_API_KEYが設定されていません
```

**解決方法**:
1. `.env` ファイルが存在することを確認
2. `OPENAI_API_KEY=sk-proj-...` の形式で設定されているか確認
3. APIキーが有効であることを確認（[OpenAI Platform](https://platform.openai.com/)）
4. 環境変数が正しく読み込まれているか確認:
   ```bash
   python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('OPENAI_API_KEY'))"
   ```

#### 2. データファイルが見つからない

**エラーメッセージ**:
```
FileNotFoundError: ファイルが見つかりません: OUTPUT/preprocessed_cc_news.csv
```

**解決方法**:
1. `OUTPUT/` ディレクトリが存在することを確認
2. preprocessed データファイルを生成:
   ```bash
   python a01_load_set_rag_data.py --dataset cc_news
   ```
3. ファイル名が正しいことを確認:
   ```bash
   ls -la OUTPUT/preprocessed_*.csv
   ```

#### 3. MeCabインストールエラー

**エラーメッセージ**:
```
RuntimeError: MeCab initialization failed
```

**解決方法**:
1. MeCabをインストール（オプションなので、なくても動作します）
2. 正規表現モードで動作することを確認（自動フォールバック）
3. ログで以下のメッセージを確認:
   ```
   ⚠️ MeCabが利用できません（正規表現モード）
   ```

#### 4. メモリ不足エラー

**エラーメッセージ**:
```
MemoryError: Unable to allocate array
```

**解決方法**:
1. `--max-docs` オプションで処理文書数を制限:
   ```bash
   python a02_make_qa.py --dataset cc_news --max-docs 100
   ```
2. `--batch-chunks` を小さくする（3 → 1）
3. システムメモリを増やす

#### 5. API レート制限エラー

**エラーメッセージ**:
```
RateLimitError: Rate limit exceeded
```

**解決方法**:
1. スクリプトは自動的にリトライします（3回まで）
2. `--batch-chunks` を小さくして API 呼び出しを分散:
   ```bash
   python a02_make_qa.py --dataset cc_news --batch-chunks 1
   ```
3. OpenAI のレート制限を確認し、アップグレードを検討

#### 6. カバレージ分析が遅い

**症状**:
カバレージ分析に非常に長い時間がかかる

**解決方法**:
1. `--analyze-coverage` オプションを省略してQ&A生成のみ実行
2. 処理文書数を制限: `--max-docs 50`
3. カバレージ分析は別途実行する

#### 7. 生成されるQ&A数が少ない

**症状**:
期待よりも少ないQ&Aペアしか生成されない

**解決方法**:
1. ログで API エラーがないか確認
2. `--batch-chunks` を小さくする（5 → 3）
3. チャンク統合を無効化: `--no-merge-chunks`
4. データセット設定の `qa_per_chunk` を確認

#### 8. 出力ディレクトリ作成エラー

**エラーメッセージ**:
```
PermissionError: [Errno 13] Permission denied: 'qa_output/a02'
```

**解決方法**:
1. ディレクトリを手動で作成:
   ```bash
   mkdir -p qa_output/a02
   ```
2. 書き込み権限を確認:
   ```bash
   chmod 755 qa_output/a02
   ```
3. 別の出力パスを指定:
   ```bash
   python a02_make_qa.py --dataset cc_news --output /tmp/qa_output
   ```

### パフォーマンス最適化のヒント

#### 処理時間を短縮する

1. **バッチサイズを増やす**（品質とのトレードオフ）:
   ```bash
   python a02_make_qa.py --dataset cc_news --batch-chunks 5
   ```

2. **チャンク統合を最適化**:
   ```bash
   python a02_make_qa.py \
       --dataset cc_news \
       --merge-chunks \
       --min-tokens 100 \
       --max-tokens 300
   ```

3. **カバレージ分析をスキップ**:
   ```bash
   python a02_make_qa.py --dataset cc_news  # --analyze-coverageなし
   ```

4. **処理文書数を制限**:
   ```bash
   python a02_make_qa.py --dataset cc_news --max-docs 100
   ```

#### カバレージを向上させる

1. **Q&A数を増やす**:
   - コード内の `qa_per_chunk` を 5 → 7 に変更
   - `determine_qa_count` 関数の上限を 8 → 10 に変更

2. **バッチサイズを小さくする**（品質向上）:
   ```bash
   python a02_make_qa.py --dataset cc_news --batch-chunks 3
   ```

3. **チャンク統合を調整**:
   ```bash
   python a02_make_qa.py \
       --dataset cc_news \
       --merge-chunks \
       --min-tokens 100 \
       --max-tokens 250
   ```

#### コストを削減する

1. **安価なモデルを使用**:
   ```bash
   python a02_make_qa.py --dataset cc_news --model gpt-4o-mini
   ```

2. **バッチサイズを最大化**（API呼び出し削減）:
   ```bash
   python a02_make_qa.py --dataset cc_news --batch-chunks 5
   ```

3. **処理文書数を制限**:
   ```bash
   python a02_make_qa.py --dataset cc_news --max-docs 50
   ```

### デバッグ方法

#### ログレベルを変更

```python
# a02_make_qa.py の logging.basicConfig を変更
logging.basicConfig(
    level=logging.DEBUG,  # INFO → DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

#### 中間結果を確認

```bash
# 実行後、出力ファイルを確認
ls -lh qa_output/a02/

# JSON ファイルを確認
cat qa_output/a02/summary_cc_news_*.json | python -m json.tool

# カバレージ結果を確認
cat qa_output/a02/coverage_cc_news_*.json | python -m json.tool | head -50
```

#### API呼び出しをモニタリング

```bash
# 実行中にログファイルをリアルタイムで確認
python a02_make_qa.py --dataset cc_news --max-docs 10 2>&1 | tee execution.log

# 別のターミナルで
tail -f execution.log
```

---

## まとめ

`a02_make_qa.py`は、**品質とカバレージのバランスを重視**したQ&Aペア生成システムです。動的Q&A数調整、チャンク統合、バッチ処理などの最適化により、70.6%のカバレージを達成しています。

多段階カバレージ分析により、生成されたQ&Aの品質を詳細に評価でき、継続的な改善が可能な設計となっています。

### 推奨ワークフロー

1. **開発・テスト段階**:
   ```bash
   python a02_make_qa.py --dataset cc_news --max-docs 10 --analyze-coverage
   ```

2. **本番実行前の検証**:
   ```bash
   python a02_make_qa.py --dataset cc_news --max-docs 50 --analyze-coverage
   ```

3. **本番運用**:
   ```bash
   python a02_make_qa.py \
       --dataset cc_news \
       --batch-chunks 3 \
       --merge-chunks \
       --min-tokens 100 \
       --max-tokens 300 \
       --model gpt-5-mini \
       --analyze-coverage
   ```

### 関連ドキュメント

- [CLAUDE.md](../CLAUDE.md) - プロジェクト全体の概要
- [a03_rag_qa_coverage_improved.py](../a03_rag_qa_coverage_improved.py) - カバレージ分析のコア実装
- [requirements.txt](../requirements.txt) - 依存パッケージ一覧

---

**作成日**: 2025年11月6日
**最終更新**: 2025年11月14日
**バージョン**: 2.0