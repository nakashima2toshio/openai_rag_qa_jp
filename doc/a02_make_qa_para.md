# a02_make_qa_para.py - 改善版Q/Aペア自動生成システム

## 目次

1. [概要](#1-概要)
2. [アーキテクチャ](#2-アーキテクチャ)
3. [キーワード抽出・複雑度分析](#3-キーワード抽出複雑度分析)
4. [セマンティックチャンク分割](#4-セマンティックチャンク分割)
5. [Q/Aペア生成](#5-qaペア生成)
6. [Celery非同期並列処理](#6-celery非同期並列処理)
7. [カバレージ分析](#7-カバレージ分析)
8. [コマンドラインオプション](#8-コマンドラインオプション)
9. [実行方法](#9-実行方法)
10. [トラブルシューティング](#10-トラブルシューティング)

---

## 1. 概要

### 1.1 目的

`a02_make_qa_para.py`は、OUTPUTフォルダ内のpreprocessedファイルから高品質なQ/Aペアを自動生成するシステムです。バッチ処理による並列化でAPI呼び出し回数を大幅削減し、Celeryによる非同期並列処理をサポートします。

### 1.2 起動コマンド

```bash
# 基本実行（同期処理）
python a02_make_qa_para.py --dataset livedoor --model gpt-4o-mini --max-docs 20

# Celery並列処理
python a02_make_qa_para.py --dataset cc_news --use-celery --celery-workers 24 --batch-chunks 3
```

### 1.3 主要機能

- **セマンティック分割によるチャンク作成**（段落境界を優先）
- **バッチ処理による並列Q/A生成**（1-5チャンク同時処理）
- **Celeryによる非同期並列処理**（複数ワーカーで同時実行）
- **小チャンク自動統合による効率化**
- **多段階カバレージ分析**（strict/standard/lenient）
- **チャンク特性別カバレージ分析**（長さ別・位置別）

### 1.4 対応データセット

| データセット | キー | 言語 | 説明 |
|------------|------|------|------|
| CC-News | `cc_news` | 英語 | 英語ニュース記事（7,376件） |
| CC100日本語 | `japanese_text` | 日本語 | Webテキストコーパス |
| Wikipedia日本語版 | `wikipedia_ja` | 日本語 | 百科事典的知識 |
| Livedoorニュース | `livedoor` | 日本語 | ニュースコーパス（7,376件） |

### 1.5 処理モード比較

| モード | API呼び出し | 実行時間 | 効率化率 | 推奨用途 |
|--------|------------|---------|---------|---------|
| 同期処理 | 1800回 | 180分 | 1.0x | 小規模テスト |
| Celery並列 | 1800回 | 23分 | 7.8x | 中規模処理 |
| **ハイブリッド** | **600回** | **8分** | **22.5x** | **大規模処理（推奨）** |

---

## 2. アーキテクチャ

### 2.1 システム構成図

```
┌─────────────────────────────────────────────────────────────────┐
│                    a02_make_qa_para.py                          │
├─────────────────────────────────────────────────────────────────┤
│  [1] データ読み込み                                              │
│      load_preprocessed_data() / load_uploaded_file()            │
│                              │                                  │
│                              ▼                                  │
│  [2] チャンク作成                                                │
│      create_document_chunks() → create_semantic_chunks()        │
│                              │                                  │
│                              ▼                                  │
│  [3] チャンク統合（オプション）                                    │
│      merge_small_chunks()                                       │
│                              │                                  │
│                              ▼                                  │
│  [4] Q/A生成                                                    │
│      ┌──────────────────┬──────────────────┐                    │
│      │  同期処理          │  Celery並列       │                  │
│      │  generate_qa_*()  │  submit_parallel_qa_generation()    │
│      └──────────────────┴──────────────────┘                    │
│                              │                                  │
│                              ▼                                  │
│  [5] カバレージ分析                                              │
│      analyze_coverage() → multi_threshold_coverage()           │
│                              │                                  │
│                              ▼                                  │
│  [6] 結果保存                                                    │
│      save_results() → qa_output/a02/                           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 依存モジュール

```python
# 共通モジュール
from models import QAPairsResponse
from config import DATASET_CONFIGS, QAGenerationConfig

# ローカルモジュール
from a03_rag_qa_coverage_improved import SemanticCoverage
from helper_rag_qa import SemanticCoverage as SemanticChunker
from helper_rag import clean_text

# Celeryタスク（オプション）
from celery_tasks import submit_parallel_qa_generation, collect_results
```

### 2.3 データセット拡張設定

```python
_LOCAL_DATASET_EXTENSIONS = {
    "cc_news": {
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "en",
    },
    "japanese_text": {
        "text_column": "Combined_Text",
        "title_column": None,
        "lang": "ja",
    },
    "wikipedia_ja": {
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "ja",
    },
    "livedoor": {
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "ja",
    }
}
```

---

## 3. キーワード抽出・複雑度分析

### 3.1 KeywordExtractorクラス

MeCabと正規表現を統合したキーワード抽出クラスです。MeCabが利用可能な場合は複合名詞抽出を優先し、利用不可の場合は正規表現版に自動フォールバックします。

```python
class KeywordExtractor:
    def __init__(self, prefer_mecab: bool = True):
        """MeCab優先設定"""

    def extract(self, text: str, top_n: int = 5) -> List[str]:
        """キーワード抽出（自動フォールバック対応）"""

    def _extract_with_mecab(self, text: str, top_n: int) -> List[str]:
        """MeCabによる複合名詞抽出"""

    def _extract_with_regex(self, text: str, top_n: int) -> List[str]:
        """正規表現によるキーワード抽出"""
```

**ストップワード**:
```python
self.stopwords = {
    'こと', 'もの', 'これ', 'それ', 'ため', 'よう', 'さん',
    'ます', 'です', 'ある', 'いる', 'する', 'なる', 'できる',
    'いう', '的', 'な', 'に', 'を', 'は', 'が', 'で', 'と',
    'の', 'から', 'まで', '等', 'など', 'よる', 'おく', 'くる'
}
```

### 3.2 複雑度分析関数

```python
def analyze_chunk_complexity(chunk_text: str, lang: str = "ja") -> Dict:
    """チャンクの複雑度を分析

    Returns:
        {
            "complexity_level": "high" | "medium" | "low",
            "technical_terms": List[str],  # 上位10個
            "avg_sentence_length": float,
            "concept_density": float,
            "sentence_count": int,
            "token_count": int
        }
    """
```

**複雑度レベル判定**:
| レベル | 条件 |
|--------|------|
| high | 概念密度 > 5% OR 平均文長 > 30トークン |
| medium | 概念密度 > 2% OR 平均文長 > 20トークン |
| low | その他 |

### 3.3 主要概念抽出

```python
def extract_key_concepts(chunk_text: str, lang: str = "ja", top_n: int = 5) -> List[str]:
    """チャンクから主要概念を抽出
    KeywordExtractorと複雑度分析の結果を統合
    """
```

---

## 4. セマンティックチャンク分割

### 4.1 チャンク作成関数

```python
def create_semantic_chunks(
    text: str,
    lang: str = "ja",
    max_tokens: int = 200,
    chunk_id_prefix: str = "chunk"
) -> List[Dict]:
    """
    セマンティック分割によるチャンク作成（段落優先）

    helper_rag_qa.pyのSemanticCoverage.create_semantic_chunks()を使用し、
    段落境界を最優先したセマンティック分割を実行。

    Returns:
        [{
            'id': str,
            'text': str,
            'tokens': int,
            'type': 'paragraph' | 'sentence_group' | 'forced_split',
            'sentences': List[str]
        }, ...]
    """
```

### 4.2 文書チャンク作成

```python
def create_document_chunks(
    df: pd.DataFrame,
    dataset_type: str,
    max_docs: Optional[int] = None,
    config: Optional[Dict] = None
) -> List[Dict]:
    """DataFrameから文書チャンクを作成（セマンティック分割）

    各チャンクに付加されるメタデータ:
    - doc_id: 文書ID
    - doc_idx: 文書インデックス
    - chunk_idx: チャンクインデックス
    - dataset_type: データセットタイプ
    """
```

### 4.3 小チャンク統合

```python
def merge_small_chunks(
    chunks: List[Dict],
    min_tokens: int = 150,
    max_tokens: int = 400
) -> List[Dict]:
    """小さいチャンクを統合して適切なサイズにする

    - min_tokens未満のチャンクは統合対象
    - 同じ文書からのチャンクのみ統合
    - 統合後のトークン数がmax_tokensを超えない範囲で統合
    """
```

---

## 5. Q/Aペア生成

### 5.1 動的Q/A数決定

```python
def determine_qa_count(chunk: Dict, config: Dict) -> int:
    """チャンクに最適なQ/A数を決定（動的調整）"""
```

| トークン数 | 基本Q/A数 | 備考 |
|-----------|----------|------|
| < 50 | 2個 | 短いチャンクでも最低2個 |
| 50-100 | 3個 | Shortチャンク強化 |
| 100-200 | base + 1 | Mediumチャンク |
| 200-300 | base + 2 | Longチャンク |
| > 300 | base + 3 | 超長文（上限8個） |

**位置バイアス補正**: 文書後半（6番目以降のチャンク）は+1個追加

### 5.2 質問タイプ

コードで使用される4種類の質問タイプ:

| タイプ | 日本語 | 英語 |
|--------|--------|------|
| fact | 事実確認型（〜は何ですか？） | Factual (What is...?) |
| reason | 理由説明型（なぜ〜ですか？） | Explanatory (Why...?) |
| comparison | 比較型（〜と〜の違いは？） | Comparative (What's the difference...?) |
| application | 応用型（〜はどのように活用されますか？） | Application (How is... used?) |

### 5.3 バッチ処理

```python
def generate_qa_pairs_for_batch(
    chunks: List[Dict],
    config: Dict,
    model: str = "gpt-4o-mini",
    client: Optional[OpenAI] = None
) -> List[Dict]:
    """複数チャンクから一度にQ/Aペアを生成（バッチ処理対応）

    - 1-5チャンクを1つのプロンプトに統合
    - OpenAI Responses API (client.responses.parse) を使用
    - QAPairsResponse Pydanticモデルで型安全な出力
    """
```

**バッチ処理の効果**:
| バッチサイズ | API呼び出し削減率 |
|------------|-----------------|
| 1 | 0% |
| 2 | 50% |
| **3** | **67%（推奨）** |
| 4 | 75% |
| 5 | 80% |

### 5.4 単一チャンク処理

```python
def generate_qa_pairs_for_chunk(
    chunk: Dict,
    config: Dict,
    model: str = "gpt-4o-mini",
    client: Optional[OpenAI] = None
) -> List[Dict]:
    """単一チャンクからQ/Aペアを生成（後方互換性のため維持）"""
```

### 5.5 データセット全体処理

```python
def generate_qa_for_dataset(
    chunks: List[Dict],
    dataset_type: str,
    model: str = "gpt-4o-mini",
    chunk_batch_size: int = 3,
    merge_chunks: bool = True,
    min_tokens: int = 150,
    max_tokens: int = 400,
    config: Optional[Dict] = None
) -> List[Dict]:
    """データセット全体のQ/Aペア生成

    - チャンク統合 → バッチ分割 → API呼び出し
    - 最大3回リトライ
    - フォールバック: バッチ失敗時は個別処理
    """
```

---

## 6. Celery非同期並列処理

### 6.1 ワーカー管理

```python
def check_celery_workers(required_workers: int = 8) -> bool:
    """Celeryワーカーの状態を確認（リトライ機能付き）

    - 最大3回リトライ
    - ワーカー数不足でも続行可能
    """
```

### 6.2 ワーカー起動・管理コマンド

```bash
# 起動
./start_celery.sh start -w 8

# ステータス確認
./start_celery.sh status

# 停止
./start_celery.sh stop

# 再起動（推奨）
redis-cli FLUSHDB
./start_celery.sh restart -w 24
```

### 6.3 並列タスク投入・結果収集

```python
from celery_tasks import submit_parallel_qa_generation, collect_results

# タスク投入
tasks = submit_parallel_qa_generation(
    processed_chunks, config, model, batch_size=3
)

# 結果収集（タイムアウト計算: タスク数 × 10秒、最低600秒、最大1800秒）
timeout_seconds = min(max(len(tasks) * 10, 600), 1800)
qa_pairs = collect_results(tasks, timeout=timeout_seconds)
```

### 6.4 主要ファイル

| ファイル | 役割 |
|---------|------|
| `celery_tasks.py` | タスク定義とワーカー設定 |
| `a02_make_qa_para.py` | タスク投入と結果収集 |
| `start_celery.sh` | ワーカー管理スクリプト |

---

## 7. カバレージ分析

### 7.1 データセット別最適閾値

```python
OPTIMAL_THRESHOLDS = {
    "cc_news": {
        "strict": 0.80,
        "standard": 0.70,
        "lenient": 0.60
    },
    "japanese_text": {
        "strict": 0.75,
        "standard": 0.65,
        "lenient": 0.55
    },
    "wikipedia_ja": {
        "strict": 0.85,
        "standard": 0.75,
        "lenient": 0.65
    },
    "livedoor": {
        "strict": 0.78,
        "standard": 0.68,
        "lenient": 0.58
    }
}
```

### 7.2 多段階カバレージ分析

```python
def multi_threshold_coverage(
    coverage_matrix: np.ndarray,
    chunks: List[Dict],
    qa_pairs: List[Dict],
    thresholds: Dict[str, float]
) -> Dict:
    """複数閾値でカバレージを評価

    Returns:
        {
            "strict": {
                "threshold": float,
                "covered_chunks": int,
                "coverage_rate": float,
                "uncovered_count": int,
                "uncovered_chunks": List[Dict]
            },
            "standard": {...},
            "lenient": {...}
        }
    """
```

### 7.3 チャンク特性別分析

```python
def analyze_chunk_characteristics_coverage(
    chunks: List[Dict],
    coverage_matrix: np.ndarray,
    qa_pairs: List[Dict],
    threshold: float = 0.7
) -> Dict:
    """チャンク特性別のカバレージ分析

    Returns:
        {
            "by_length": {
                "short": {"count", "covered", "avg_similarity", "coverage_rate"},
                "medium": {...},
                "long": {...}
            },
            "by_position": {
                "beginning": {...},
                "middle": {...},
                "end": {...}
            },
            "summary": {
                "total_chunks": int,
                "total_qa_pairs": int,
                "threshold_used": float,
                "insights": List[str]
            }
        }
    """
```

**長さ別分類**:
- short: < 100トークン
- medium: 100-200トークン
- long: >= 200トークン

**位置別分類**:
- beginning: 前半33%
- middle: 中盤33%
- end: 後半33%

### 7.4 メイン分析関数

```python
def analyze_coverage(
    chunks: List[Dict],
    qa_pairs: List[Dict],
    dataset_type: str = "wikipedia_ja",
    custom_threshold: Optional[float] = None
) -> Dict:
    """生成されたQ/Aペアのカバレージを分析

    - 埋め込み生成（バッチAPI最適化）
    - カバレージ行列計算
    - 多段階カバレージ分析
    - チャンク特性別分析
    """
```

---

## 8. コマンドラインオプション

### 8.1 全オプション一覧

| オプション | 型 | デフォルト | 説明 |
|-----------|---|----------|------|
| `--dataset` | str | None | データセット（--input-fileと排他） |
| `--input-file` | str | None | ローカルファイルパス（--datasetと排他） |
| `--model` | str | gpt-4o-mini | 使用するOpenAIモデル |
| `--output` | str | qa_output/a02 | 出力ディレクトリ |
| `--max-docs` | int | None | 処理する最大文書数 |
| `--analyze-coverage` | flag | False | カバレージ分析を実行 |
| `--batch-chunks` | int | 3 | バッチサイズ（1-5） |
| `--merge-chunks` | flag | True | チャンク統合を有効化 |
| `--no-merge-chunks` | flag | - | チャンク統合を無効化 |
| `--min-tokens` | int | 150 | 統合対象の最小トークン数 |
| `--max-tokens` | int | 400 | 統合後の最大トークン数 |
| `--use-celery` | flag | False | Celery並列処理を使用 |
| `--celery-workers` | int | 4 | Celeryワーカー数 |
| `--coverage-threshold` | float | None | カスタム閾値 |

### 8.2 入力ソース

**--dataset**: OUTPUTフォルダのpreprocessedファイルを使用
```bash
python a02_make_qa_para.py --dataset livedoor
```

**--input-file**: 任意のローカルファイルを使用
```bash
python a02_make_qa_para.py --input-file qa_output/qa_pairs_upload_20251122_182355.csv
```

---

## 9. 実行方法

### 9.1 環境準備

```bash
# 1. Redisを起動
brew services start redis  # macOS
# または: redis-server

# 2. 既存タスクのクリア（推奨）
redis-cli FLUSHDB

# 3. Celeryワーカーを起動
./start_celery.sh restart -w 24
```

### 9.2 テスト実行

```bash
# 同期処理（小規模テスト）
python a02_make_qa_para.py \
  --dataset livedoor \
  --batch-chunks 3 \
  --model gpt-4o-mini \
  --max-docs 20 \
  --analyze-coverage
```

### 9.3 Celery並列実行

```bash
python a02_make_qa_para.py \
  --dataset cc_news \
  --use-celery \
  --celery-workers 24 \
  --batch-chunks 3 \
  --merge-chunks \
  --min-tokens 150 \
  --max-tokens 400 \
  --model gpt-4o-mini \
  --analyze-coverage
```

### 9.4 実行時の進捗表示

```
進捗: 成功=3/17, 失敗=0, 実行中=4, 待機中=10, 経過時間=15.2秒
進捗: 成功=7/17, 失敗=0, 実行中=4, 待機中=6, 経過時間=20.4秒
進捗: 成功=17/17, 失敗=0, 実行中=0, 待機中=0, 経過時間=45.8秒
✓ すべてのタスクが完了しました
```

### 9.5 出力ファイル

```
qa_output/a02/
├── qa_pairs_{dataset}_{timestamp}.json    # Q/Aペア（JSON）
├── qa_pairs_{dataset}_{timestamp}.csv     # Q/Aペア（CSV全カラム）
├── coverage_{dataset}_{timestamp}.json    # カバレージ分析結果
└── summary_{dataset}_{timestamp}.json     # サマリー情報

qa_output/
└── a02_qa_pairs_{dataset}.csv             # 統一フォーマット（question/answerのみ）
```

### 9.6 実行時間の見積もり

| 項目 | 値 |
|-----|-----|
| 処理文書数 | 497件（全件） |
| チャンク数 | ~1,825個 → 統合後 ~1,820個 |
| API呼び出し | 約365回（バッチサイズ5） |
| 推定実行時間 | 60-75分 |
| カバレージ分析 | +3-5分 |
| 合計 | 約65-80分 |

---

## 10. トラブルシューティング

### 10.1 Celeryワーカーが起動しない

**診断手順**:
```bash
# Redisの状態確認
redis-cli ping

# ログを確認
tail -f logs/celery_qa_*.log

# プロセスを確認
ps aux | grep celery
```

**解決策**:
```bash
# 既存ワーカーを停止
./start_celery.sh stop

# タスクキューをクリア
redis-cli FLUSHDB

# 再起動
./start_celery.sh start -w 8
```

### 10.2 タスクが処理されない

**診断**:
```bash
# キューの状態確認
redis-cli LLEN celery
redis-cli LLEN qa_generation

# ワーカーの状態確認
celery -A celery_tasks inspect active
```

**解決策**:
```bash
# ワーカー数を減らして再起動
./start_celery.sh restart -w 2

# それでも解決しない場合は同期処理に切り替え
python a02_make_qa_para.py --dataset cc_news --batch-chunks 3 --max-docs 20
```

### 10.3 JSONDecodeError

**症状**: `Expecting value: line 1 column 1 (char 0)`

**原因**: OpenAI APIが空のレスポンスを返している

**解決策**:
1. ワーカーを再起動
```bash
./start_celery.sh restart -w 8
```

2. ログを確認
```bash
tail -f logs/celery_qa_*.log
```

### 10.4 空レスポンスエラー

**症状**: `No parseable response from OpenAI API`

**対策（コード内で自動処理）**:
```python
# 空レスポンスチェック
if parsed_count == 0:
    logger.error("OpenAI APIから解析可能なレスポンスが返されませんでした")
    raise ValueError("No parseable response from OpenAI API")
```

### 10.5 タイムアウトエラー

**症状**: `task_time_limit exceeded`

**解決策**: `celery_tasks.py`でタイムアウトを延長
```python
app.conf.update(
    task_time_limit=600,  # 10分に延長
    task_soft_time_limit=540,
)
```

---

## 付録: データ読み込み関数

### A.1 ローカルファイル読み込み

```python
def load_uploaded_file(file_path: str) -> pd.DataFrame:
    """
    ローカルファイルを読み込み

    対応形式: CSV, TXT, JSON, JSONL
    Combined_Textカラムを自動生成
    """
```

### A.2 Q/A CSVファイル読み込み

```python
def load_local_qa_file(file_path: str) -> pd.DataFrame:
    """ローカルのQ/A CSVファイルを読み込み

    question, answerカラムを自動検出
    空データ・重複を除去
    """
```

### A.3 preprocessedデータ読み込み

```python
def load_preprocessed_data(dataset_type: str) -> pd.DataFrame:
    """preprocessedデータを読み込み

    - 固定名ファイルを優先
    - タイムスタンプ付きファイルを自動検索
    - 最新ファイルを自動選択
    """
```