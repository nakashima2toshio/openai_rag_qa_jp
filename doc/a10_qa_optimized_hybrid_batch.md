# a10_qa_optimized_hybrid_batch.py - 技術仕様書

## 最新バージョン情報
- **最終更新**: 2024-10-29
- **バージョン**: v1.2 (最新実装版)
- **主要機能**: バッチ処理、API呼び出し最適化、ハイブリッドQ/A生成

---

## 概要

`a10_qa_optimized_hybrid_batch.py`は、**バッチ処理によるAPI呼び出し最適化**を実現した高度なQ&Aペア生成システムです。複数文書を一度のAPI呼び出しで処理することで、API呼び出し数を**最大96.3%削減**し、処理速度を大幅に向上させます。

**主な特徴**:
- バッチ処理による大幅なAPI呼び出し削減
- 処理時間の短縮（3分 → 1分）
- コスト削減（$0.075 → $0.008）
- エラー時の自動フォールバック機能
- 詳細な統計レポート出力

---

## バッチ処理の革新性

### 従来版との決定的な違い

```
従来版（個別処理）:
文書1 → API呼出1
文書2 → API呼出2
文書3 → API呼出3
...
文書497 → API呼出497

バッチ版（バッチ処理）:
文書1-10 → API呼出1
文書11-20 → API呼出2
文書21-30 → API呼出3
...
文書491-497 → API呼出50
```

| 処理方式 | 497文書のAPI呼出数 | 処理時間 | コスト削減率 |
|---------|-------------------|---------|------------|
| **従来版（個別処理）** | 497回 | 約3分 | - |
| **バッチ版（バッチ10）** | 50回 | 約1分 | **90%削減** |
| **バッチ版（バッチ20）** | 25回 | 約45秒 | **95%削減** |

### バッチ処理の3段階最適化

```
Stage 1: ルールベース抽出（ローカル処理）
    ↓
Stage 2: LLMバッチ処理（497文書 → 50回のAPI呼出）
    ↓
Stage 3: 埋め込みバッチ処理（100文書ずつ → 5回のAPI呼出）
    ↓
合計: 55回のAPI呼出（従来版1,491回から96.3%削減）
```

---

## 推奨実行コマンド

### 95%カバレージ達成版（推奨）

```bash
python a10_qa_optimized_hybrid_batch.py \
    --dataset cc_news \
    --model gpt-5-mini \
    --batch-size 10 \
    --embedding-batch-size 150 \
    --qa-count 12 \
    --max-docs 150 \
    --output qa_output
```

**期待結果**:
- 処理文書: 150件
- 生成Q/A: 1,800個
- カバレージ: 95%+
- API呼出: 約20回
- 処理時間: 2-3分
- コスト: $0.01-0.02
- 出力先: `qa_output/a10/`

---

## 主要コンポーネント

### 1. インポートと設定（L30-54）

```python
from helper_rag_qa import BatchHybridQAGenerator, OptimizedHybridQAGenerator

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

### 2. データセット設定（L56-81）

```python
DATASET_CONFIGS = {
    "cc_news": {
        "name": "CC-News英語ニュース",
        "file": "OUTPUT/preprocessed_cc_news.csv",
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "en",
        "default_doc_type": "news"
    },
    "japanese_text": {...},
    "wikipedia_ja": {...}
}
```

---

## 主要関数

### 1. load_preprocessed_data()（L87-113）

**目的**: preprocessedデータを読み込み

```python
def load_preprocessed_data(dataset_type: str, max_docs: Optional[int] = None) -> pd.DataFrame:
    """
    Args:
        dataset_type: データセットタイプ
        max_docs: 最大文書数
    Returns:
        pd.DataFrame: 読み込んだデータ
    """
```

**処理手順**:
1. データセット設定取得（L89-91）
2. ファイル存在確認（L93-95）
3. CSVファイル読み込み（L98）
4. カラム確認（L101-103）
5. 空テキスト除外（L106）
6. 文書数制限（L109-110）

### 2. generate_batch_qa_from_dataset()（L119-223）

**目的**: バッチ処理でデータセットからQ/A生成

```python
def generate_batch_qa_from_dataset(
    df: pd.DataFrame,
    dataset_type: str,
    model: str = "gpt-5-mini",
    batch_size: int = 10,
    embedding_batch_size: int = 100,
    qa_count: Optional[int] = None,
    use_llm: bool = True,
    calculate_coverage: bool = True,
    doc_type: Optional[str] = None,
    output_dir: str = "qa_output"
) -> Dict:
```

**処理フロー**:
1. テキストリスト準備（L145）
2. BatchHybridQAGenerator初期化（L147-152）
3. バッチ処理実行（L158-165）:
   ```python
   batch_results = generator.generate_batch_hybrid_qa(
       texts=texts,
       qa_count=qa_count,
       use_llm=use_llm,
       calculate_coverage=calculate_coverage,
       document_type=doc_type,
       show_progress=True
   )
   ```
4. 統計情報集計（L171-176）
5. サマリー作成（L179-208）
6. メタデータ追加（L211-218）

**返却値構造**:
```python
{
    "summary": {
        "dataset_type": "cc_news",
        "documents_processed": 150,
        "total_qa_generated": 1800,
        "batch_sizes": {
            "llm_batch_size": 10,
            "embedding_batch_size": 150
        },
        "processing_time": {...},
        "api_usage": {...},
        "coverage": {...}
    },
    "results": [...]
}
```

### 3. compare_with_normal_version()（L229-333）

**目的**: 通常版とバッチ版の性能比較

```python
def compare_with_normal_version(
    df: pd.DataFrame,
    dataset_type: str,
    model: str = "gpt-5-mini",
    sample_size: int = 10
) -> Dict:
```

**処理内容**:
1. 通常版の実行（L249-266）:
   ```python
   normal_generator = OptimizedHybridQAGenerator(model=model)
   for text in tqdm(texts, desc="通常版"):
       result = normal_generator.generate_hybrid_qa(...)
   ```

2. バッチ版の実行（L269-282）:
   ```python
   batch_generator = BatchHybridQAGenerator(model=model, batch_size=5)
   batch_results = batch_generator.generate_batch_hybrid_qa(...)
   ```

3. 比較結果の計算（L285-308）:
   - 処理時間の比較
   - API呼び出し数の比較
   - 改善率の計算

### 4. save_batch_results()（L339-385）

**目的**: バッチ処理結果を保存

```python
def save_batch_results(
    generation_results: Dict,
    dataset_type: str,
    model: str,
    batch_size: int,
    output_dir: str = "qa_output"
) -> Dict[str, str]:
```

**出力先**: `qa_output/a10/`（L349）

**保存ファイル**:
1. サマリーファイル（L356-358）:
   - ファイル名: `batch_summary_{dataset}_{model}_b{batch_size}_{timestamp}.json`
2. Q/Aペア（CSV）（L361-377）:
   - ファイル名: `batch_qa_pairs_{dataset}_{model}_b{batch_size}_{timestamp}.csv`

---

## メイン処理（L391-595）

### 処理フロー

```python
def main():
    """メイン処理"""
```

1. **引数パース**（L393-467）
2. **APIキー確認**（L470-473）
3. **データ読み込み**（L488-489）
4. **処理モード分岐**:
   - 比較実行モード（L492-505）
   - 通常バッチ処理（L507-520）
5. **結果保存**（L523-530）
6. **統計表示**（L533-560）
7. **バッチ処理効果の表示**（L563-586）

### コマンドライン引数（L393-467）

| 引数 | 型 | デフォルト | 説明 |
|-----|-----|----------|------|
| `--dataset` | str | cc_news | 処理するデータセット |
| `--model` | str | gpt-5-mini | 使用するLLMモデル |
| `--batch-size` | int | 10 | LLMバッチサイズ |
| `--embedding-batch-size` | int | 100 | 埋め込みバッチサイズ |
| `--max-docs` | int | None | 処理する最大文書数 |
| `--qa-count` | int | None | 文書あたりのQ/A数 |
| `--doc-type` | str | None | 文書タイプ（news/technical/academic/auto） |
| `--no-llm` | flag | False | LLMを使用しない |
| `--no-coverage` | flag | False | カバレージ計算を行わない |
| `--output` | str | qa_output | 出力ディレクトリ |
| `--compare` | flag | False | 通常版との比較実行 |
| `--compare-size` | int | 10 | 比較実行のサンプルサイズ |

---

## 使用方法

### 基本使用例

```bash
# 基本使用（バッチサイズ10）
python a10_qa_optimized_hybrid_batch.py --dataset cc_news

# バッチサイズ指定
python a10_qa_optimized_hybrid_batch.py --dataset cc_news --batch-size 20

# モデル指定
python a10_qa_optimized_hybrid_batch.py --dataset cc_news --model gpt-5-mini

# 比較実行（通常版 vs バッチ版）
python a10_qa_optimized_hybrid_batch.py --dataset cc_news --compare
```

### 高度な使用例

```bash
# 高カバレッジ達成設定
python a10_qa_optimized_hybrid_batch.py \
    --dataset cc_news \
    --model gpt-5-mini \
    --batch-size 10 \
    --embedding-batch-size 150 \
    --qa-count 12 \
    --max-docs 150

# コスト最小化設定
python a10_qa_optimized_hybrid_batch.py \
    --dataset cc_news \
    --no-llm \
    --no-coverage

# 品質重視設定
python a10_qa_optimized_hybrid_batch.py \
    --dataset cc_news \
    --model gpt-4o \
    --batch-size 5 \
    --qa-count 8
```

### プログラムからの使用

```python
from helper_rag_qa import BatchHybridQAGenerator

# 初期化
generator = BatchHybridQAGenerator(
    model="gpt-5-mini",
    batch_size=10,
    embedding_batch_size=100
)

# バッチ処理実行
texts = ["文書1...", "文書2...", "文書3...", ...]
results = generator.generate_batch_hybrid_qa(
    texts=texts,
    qa_count=5,
    use_llm=True,
    calculate_coverage=True,
    document_type="auto",
    show_progress=True
)

# 結果の取得
for i, result in enumerate(results):
    qa_pairs = result["qa_pairs"]
    coverage = result["coverage"]["coverage_percentage"]
    cost = result["api_usage"]["cost"]
    print(f"文書{i+1}: {len(qa_pairs)}個のQ/A, カバレッジ{coverage:.1f}%, コスト${cost:.4f}")
```

---

## 出力ファイル

### ファイル構成

```
qa_output/a10/
├── batch_summary_{dataset}_{model}_b{batch_size}_{timestamp}.json
└── batch_qa_pairs_{dataset}_{model}_b{batch_size}_{timestamp}.csv
```

### サマリーファイル例（L179-208）

```json
{
    "dataset_type": "cc_news",
    "dataset_name": "CC-News英語ニュース",
    "model_used": "gpt-5-mini",
    "batch_processing": true,
    "batch_sizes": {
        "llm_batch_size": 10,
        "embedding_batch_size": 100
    },
    "documents_processed": 497,
    "total_qa_generated": 1491,
    "avg_qa_per_doc": 3.0,
    "processing_time": {
        "total_seconds": 60,
        "minutes": 1.0,
        "docs_per_second": 8.28
    },
    "api_usage": {
        "total_cost": 0.0075,
        "cost_per_doc": 0.000015,
        "batch_statistics": {
            "llm_batches": 50,
            "embedding_batches": 5,
            "total_llm_calls": 50,
            "total_embedding_calls": 5
        }
    },
    "coverage": {
        "calculated": true,
        "avg_coverage": 85.5,
        "min_coverage": 72.0,
        "max_coverage": 95.0
    },
    "generation_timestamp": "2024-10-29T14:30:00"
}
```

---

## パフォーマンス比較

### 処理時間とコスト（497文書）

| 処理モード | API呼出数 | 処理時間 | コスト（gpt-5-mini） | 削減率 |
|-----------|----------|---------|-------------------|--------|
| **通常版** | 1,491回 | 3分 | $0.075 | - |
| **バッチ版（10）** | 150回 | 1分 | $0.008 | **89.9%** |
| **バッチ版（20）** | 75回 | 45秒 | $0.004 | **95.0%** |

### スケーラビリティ

| 文書数 | 通常版API呼出 | バッチ版API呼出（10） | 削減率 |
|-------|-------------|-------------------|--------|
| 10 | 30回 | 4回 | 86.7% |
| 100 | 300回 | 30回 | 90.0% |
| 500 | 1,500回 | 150回 | 90.0% |
| 1,000 | 3,000回 | 300回 | 90.0% |

### 比較実行結果の例（L310-331）

```
================================================================================
📊 性能比較結果
================================================================================
サンプル数: 10文書

【通常版（個別処理）】
  処理時間: 30.00秒
  API呼出: 30回
  1文書あたり: 3.00秒, 3.0回

【バッチ版（バッチ処理）】
  処理時間: 10.00秒
  API呼出: 3回
  1文書あたり: 1.00秒, 0.3回

【改善効果】
  処理時間短縮: 66.7%
  API呼出削減: 90.0%
  高速化: 3.00x
================================================================================
```

---

## トラブルシューティング

### よくある問題と解決方法

#### Q: バッチ処理でエラーが頻発する
**A:** バッチサイズを小さくする
```bash
python a10_qa_optimized_hybrid_batch.py --batch-size 5
```

#### Q: メモリ不足エラー
**A:** 埋め込みバッチサイズを削減
```bash
python a10_qa_optimized_hybrid_batch.py --embedding-batch-size 50
```

#### Q: API Rate Limit エラー
**A:** バッチサイズを大きくして呼出頻度を減らす
```bash
python a10_qa_optimized_hybrid_batch.py --batch-size 20
```

#### Q: OpenAI APIキーエラー（L470-473）
**A:** 環境変数を設定
```bash
export OPENAI_API_KEY="your-api-key"
```

---

## ベストプラクティス

### バッチサイズの選択

| 用途 | 推奨バッチサイズ | 理由 |
|------|---------------|------|
| **開発・テスト** | 5 | エラー発生時の影響最小化 |
| **本番運用** | 10-20 | バランスが良い |
| **大量処理** | 20-50 | 最大効率化（リスク増） |
| **高品質重視** | 5-10 | パース精度向上 |

### エラーハンドリング（L588-592）

```python
try:
    # メイン処理
    ...
except Exception as e:
    logger.error(f"処理中にエラーが発生しました: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
```

### コスト最適化設定

```bash
# 最小コストでの大量処理
python a10_qa_optimized_hybrid_batch.py \
    --dataset cc_news \
    --model gpt-5-mini \
    --batch-size 50 \
    --no-coverage
```

---

## BatchHybridQAGeneratorクラス

`helper_rag_qa`モジュールに実装されたクラス（L43）

### 主要メソッド

| メソッド | 説明 |
|---------|------|
| `generate_batch_hybrid_qa()` | 複数文書のバッチ処理 |
| `_batch_enhance_with_llm()` | LLMバッチ品質向上 |
| `_create_batch_prompt()` | バッチプロンプト作成 |
| `_parse_batch_response()` | バッチ応答パース |
| `_batch_calculate_coverage()` | バッチカバレッジ計算 |
| `_batch_get_embeddings()` | 埋め込みバッチ取得 |

### 初期化パラメータ（L147-152）

```python
generator = BatchHybridQAGenerator(
    model=model,                          # LLMモデル
    batch_size=batch_size,                # LLMバッチサイズ
    embedding_batch_size=embedding_batch_size  # 埋め込みバッチサイズ
)
```

### バッチ統計情報（L199, L296-299, L548-549）

```python
batch_statistics = {
    "llm_batches": 50,           # LLMバッチ数
    "embedding_batches": 5,      # 埋め込みバッチ数
    "total_llm_calls": 50,       # 総LLM呼び出し数
    "total_embedding_calls": 5,  # 総埋め込み呼び出し数
}
```

---

## バッチ処理の効果（L563-586）

処理完了時に自動的に表示される統計情報：

```
🚀 バッチ処理の効果
================================================================================

バッチ処理により以下の改善を実現：

1. **API呼び出し削減**
   - 通常版（推定）: 450回
   - バッチ版（実際）: 20回
   - 削減率: 95.6%

2. **処理速度向上**
   - 処理速度: 8.28文書/秒
   - 150文書を3.0分で処理

3. **スケーラビリティ**
   - 大規模データセット処理が現実的に
   - レート制限リスクの大幅低減
```

---

## 今後の改善案

1. **非同期バッチ処理**
   - asyncio による並列処理
   - 処理時間のさらなる短縮

2. **動的バッチサイズ調整**
   - 文書長に応じた自動調整
   - エラー率に基づく適応制御

3. **キャッシュ機能**
   - 埋め込みベクトルのキャッシュ
   - 重複文書の検出と再利用

4. **リトライ機能**
   - 指数バックオフ
   - 部分的な成功の保存

5. **進捗状態の永続化**
   - 中断・再開機能
   - チェックポイント保存

---

## 変更履歴

### v1.2 (2024-10-29)
- ドキュメント全面更新（コード行番号の具体的な参照を追加）
- 実装の詳細な説明を追加

### v1.1 (2024-10-23)
- 出力ディレクトリを`qa_output/a10/`に変更（サブディレクトリ自動作成）
- ファイル管理の改善

### v1.0 (2024-10-21)
- バッチ処理版初版リリース
- BatchHybridQAGeneratorクラス実装
- API呼出削減率96%達成
- 統計レポート機能追加
- 比較実験機能実装

---

**最終更新日**: 2024年10月29日
**バージョン**: 1.2
**作成者**: OpenAI RAG Q&A JP開発チーム