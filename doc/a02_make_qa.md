# a02_make_qa.py - 詳細設計書

## 最新バージョン情報
- **最終更新**: 2025-10-23
- **バージョン**: v2.5 (バッチ処理最適化版)
- **主要機能**: バッチ処理、多段階カバレージ分析、チャンク統合最適化

---

## 推奨実行設定

### 本番運用向け設定（品質重視）

```bash
python a02_make_qa.py \
    --dataset cc_news \
    --model gpt-5-mini \
    --batch-chunks 5 \
    --merge-chunks \
    --min-tokens 150 \
    --max-tokens 400 \
    --analyze-coverage
```

**期待される実行時間と結果**:
| 項目 | 値 |
|------|-----|
| 処理文書数 | 497件（全件） |
| チャンク数 | ~1,825個 → 統合後 ~365個 |
| API呼び出し | 約73回（バッチサイズ5） |
| 推定実行時間 | 12-15分 |
| カバレージ分析 | +3-5分 |
| 合計 | 約15-20分 |
| 生成Q/Aペア数 | 500-700個 |
| 推定コスト | $0.10-0.15 (gpt-5-mini) |

### バッチサイズによる効果比較

| バッチサイズ | API呼び出し | 実行時間 | 効果 |
|------------|-----------|---------|------|
| 1（個別処理） | ~365回 | 60-80分 | 基準（100%） |
| 3（デフォルト） | ~122回 | 20-25分 | 67%削減 |
| 5（推奨） | ~73回 | 12-15分 | 80%削減 |

---

## 目次

1. [概要](#1-概要)
   - 1.1 [目的](#11-目的)
   - 1.2 [主要機能](#12-主要機能)
   - 1.3 [対応データセット](#13-対応データセット)
2. [アーキテクチャ](#2-アーキテクチャ)
   - 2.1 [システム構成図](#21-システム構成図)
   - 2.2 [主要コンポーネント](#22-主要コンポーネント)
3. [データ処理フロー](#3-データ処理フロー)
   - 3.1 [データ読み込み・前処理](#31-データ読み込み前処理)
   - 3.2 [チャンク作成](#32-チャンク作成)
   - 3.3 [チャンク統合](#33-チャンク統合)
4. [Q/Aペア生成](#4-qaペア生成)
   - 4.1 [Q/A数決定ロジック](#41-qa数決定ロジック)
   - 4.2 [バッチ処理](#42-バッチ処理)
   - 4.3 [単一チャンク処理](#43-単一チャンク処理)
   - 4.4 [データセット全体の生成制御](#44-データセット全体の生成制御)
5. [カバレージ分析](#5-カバレージ分析)
   - 5.1 [多段階カバレージ計算](#51-多段階カバレージ計算)
   - 5.2 [チャンク特性別分析](#52-チャンク特性別分析)
6. [結果保存](#6-結果保存)
7. [コマンドライン引数](#7-コマンドライン引数)
8. [メイン処理フロー](#8-メイン処理フロー)
9. [依存関係](#9-依存関係)
10. [パフォーマンス最適化](#10-パフォーマンス最適化)
11. [出力統計情報](#11-出力統計情報)
12. [注意事項・制約](#12-注意事項制約)
13. [今後の改善案](#13-今後の改善案)

---

## 1. 概要

### 1.1 目的
preprocessedファイルからOpenAI APIを使用してQ/Aペアを自動生成し、生成されたQ/Aペアのセマンティックカバレージを分析するシステム。

### 1.2 主要機能
- preprocessed CSVファイルの読み込みと前処理
- 文書のセマンティックチャンク分割
- **チャンクの統合最適化（--merge-chunks、デフォルト有効）**
- **バッチ処理によるQ/Aペア生成（1-5チャンク同時処理、デフォルト: 3）**
- **多段階カバレージ分析（strict/standard/lenient、データセット別最適閾値）**
- **チャンク特性別分析（長さ別・位置別カバレージ、自動インサイト生成）**
- 結果のJSON/CSV形式での保存（4ファイル出力: qa_output/a02/）

### 1.3 対応データセット
| データセット | ファイルパス | 言語 | チャンクサイズ | Q/Aペア数/チャンク | 最適閾値（standard） |
|------------|-------------|------|--------------|------------------|-------------------|
| cc_news | OUTPUT/preprocessed_cc_news.csv | 英語 | 300トークン | 3 | 0.70 |
| japanese_text | OUTPUT/preprocessed_japanese_text.csv | 日本語 | 200トークン | 2 | 0.65 |
| wikipedia_ja | OUTPUT/preprocessed_wikipedia_ja.csv | 日本語 | 250トークン | 3 | 0.75 |

---

## 2. アーキテクチャ

### 2.1 システム構成図

```
preprocessed CSV読み込み
         ↓
データ読み込み・前処理
         ↓
チャンク作成（SemanticCoverage）
         ↓
チャンク統合（小チャンク統合）
         ↓
バッチ処理でQ/A生成
  ├─ OpenAI API: responses.parse
  ├─ リトライ機能（最大3回）
  └─ フォールバック（個別処理）
         ↓
カバレージ分析（オプション）
  ├─ 埋め込み生成
  ├─ 類似度行列計算
  ├─ 多段階閾値評価
  └─ チャンク特性別分析
         ↓
結果保存（qa_output/a02/）
  ├─ qa_pairs_{dataset}_{timestamp}.json
  ├─ qa_pairs_{dataset}_{timestamp}.csv
  ├─ coverage_{dataset}_{timestamp}.json
  └─ summary_{dataset}_{timestamp}.json
```

### 2.2 主要コンポーネント

#### 2.2.1 データモデル（Pydantic）

**QAPair**: 個別Q/Aペアのデータモデル
```python
class QAPair(BaseModel):
    question: str                      # 質問文
    answer: str                        # 回答文
    question_type: str                 # 質問タイプ（fact/reason/comparison/application）
    source_chunk_id: Optional[str]     # ソースチャンクID
    dataset_type: Optional[str]        # データセット種別
    auto_generated: bool = False       # 自動生成フラグ
```

**QAPairsResponse**: API応答用モデル
```python
class QAPairsResponse(BaseModel):
    qa_pairs: List[QAPair]
```

#### 2.2.2 設定管理

**DATASET_CONFIGS**: データセット別設定辞書
- name: データセット名
- file: ファイルパス
- text_column: テキストカラム名
- title_column: タイトルカラム名（オプション）
- lang: 言語コード（"ja" or "en"）
- chunk_size: チャンクサイズ（トークン数、注: 内部では200固定）
- qa_per_chunk: チャンクあたりのQ/Aペア数

**OPTIMAL_THRESHOLDS**: データセット別最適閾値
```python
{
    "cc_news": {"strict": 0.80, "standard": 0.70, "lenient": 0.60},
    "japanese_text": {"strict": 0.75, "standard": 0.65, "lenient": 0.55},
    "wikipedia_ja": {"strict": 0.85, "standard": 0.75, "lenient": 0.65}
}
```

---

## 3. データ処理フロー

### 3.1 データ読み込み・前処理

#### `load_preprocessed_data(dataset_type: str) -> pd.DataFrame`

**目的**: preprocessed CSVファイルを読み込み、前処理を実行

**処理手順**:
1. DATASET_CONFIGSから設定取得
2. ファイル存在チェック
3. CSVファイル読み込み
4. 必須カラム存在確認
5. 空テキスト除外
6. DataFrameを返却

**エラーハンドリング**:
- 未対応データセット: `ValueError`
- ファイル不在: `FileNotFoundError`
- カラム不在: `ValueError`

### 3.2 チャンク作成

#### `create_document_chunks(df: pd.DataFrame, dataset_type: str, max_docs: Optional[int]) -> List[Dict]`

**目的**: DataFrameから文書チャンクを作成

**処理手順**:
1. 設定情報取得（text_column, title_column, chunk_size）
2. SemanticCoverageインスタンス生成
3. 処理文書数制限（max_docsが指定された場合）
4. 各文書を反復処理:
   - テキスト抽出（str型に変換）
   - doc_id生成（タイトル含む場合は先頭30文字）
   - `SemanticCoverage.create_semantic_chunks()`でチャンク分割
   - メタデータ追加（doc_id, doc_idx, chunk_idx, dataset_type）
5. 全チャンクリストを返却

**注意事項**:
- `create_semantic_chunks()`は内部で200トークン固定を使用
- 設定のchunk_sizeは現在未使用
- エラー発生時は警告ログ出力してcontinue

### 3.3 チャンク統合

#### `merge_small_chunks(chunks: List[Dict], min_tokens: int = 150, max_tokens: int = 400) -> List[Dict]`

**目的**: 小さいチャンクを統合して適切なサイズに最適化

**アルゴリズム**:
1. tiktoken（cl100k_base）でトークンカウント
2. 各チャンクを反復:
   - トークン数 >= min_tokens → そのまま追加
   - トークン数 < min_tokens → 統合候補
3. 統合条件:
   - 統合後のトークン数 <= max_tokens
   - 同一文書（doc_id一致）からのチャンク
4. 統合時の処理:
   - テキストを"\n\n"で連結
   - `original_chunks`リストに元チャンクIDを記録
   - chunk_idxを範囲形式で記録（例: "0-2"）
5. 統合チャンクリストを返却

**効果**:
- API呼び出し回数削減（例: 1,825個 → 365個、80%削減）
- コスト削減
- 文脈の連続性向上

**パラメータ**:
- `min_tokens`: デフォルト150（このトークン数未満は統合対象）
- `max_tokens`: デフォルト400（統合後の最大サイズ）

---

## 4. Q/Aペア生成

### 4.1 Q/A数決定ロジック

#### `determine_qa_count(chunk: Dict, config: Dict) -> int`

**目的**: チャンクのトークン数に基づいて最適なQ/A数を決定

**ロジック**:
| トークン数範囲 | Q/A数 |
|--------------|-------|
| < 50 | min(base_count, 1) |
| 50-99 | min(base_count, 2) |
| 100-199 | base_count |
| >= 200 | min(base_count + 1, 5) |

### 4.2 バッチ処理

#### `generate_qa_pairs_for_batch(chunks: List[Dict], config: Dict, model: str, client: OpenAI) -> List[Dict]`

**目的**: 複数チャンク（最大5個、推奨3個）から一度にQ/Aペアを生成

**処理フロー**:

1. **チャンク数チェック**:
   - 0個: 空リスト返却
   - 1個: `generate_qa_pairs_for_chunk()`へ委譲
   - 2個以上: バッチ処理継続

2. **プロンプト構築（言語別）**:

   **日本語の場合**:
   ```python
   system_prompt = """あなたは教育コンテンツ作成の専門家です。
   複数の日本語テキストから、学習効果の高いQ&Aペアを生成してください。

   生成ルール:
   1. 質問は明確で具体的に
   2. 回答は簡潔で正確に（1-2文程度）
   3. テキストの内容に忠実に
   4. 多様な観点から質問を作成"""

   # 複数チャンクを結合
   combined_text = """
   【テキスト1】
   {chunk1_text}

   【テキスト2】
   {chunk2_text}

   【テキスト3】
   {chunk3_text}
   """
   ```

   **英語の場合**: 同様の構造を英語で構築

3. **OpenAI API呼び出し**（最新のResponses API使用）:
   ```python
   response = client.responses.parse(
       input=f"{system_prompt}\n\n{user_prompt}",
       model=model,
       text_format=QAPairsResponse,  # Pydanticモデル指定
       max_output_tokens=4000        # バッチ処理のため増加
   )
   ```

4. **レスポンス解析**:
   - `parsed_data`からQ/Aペアを取得
   - 各チャンクに期待される数だけQ/Aを順次割り当て
   - メタデータ追加（source_chunk_id, doc_id, dataset_type, chunk_idx）

5. **エラーハンドリング**:
   - 例外発生時はフォールバックで個別処理（`generate_qa_pairs_for_chunk()`）

### 4.3 単一チャンク処理

#### `generate_qa_pairs_for_chunk(chunk: Dict, config: Dict, model: str, client: OpenAI) -> List[Dict]`

**目的**: 単一チャンクからQ/Aペアを生成（後方互換性維持、フォールバック用）

**処理フロー**:
1. Q/A数決定（`determine_qa_count()`）
2. 言語別プロンプト構築:
   - システムプロンプト: 生成ルール指示
   - ユーザープロンプト: テキスト + 質問タイプ + JSON形式指示
3. テキスト長制限:
   - 2000文字超の場合は切り詰め（"..."付加）
4. API呼び出し（responses.parse使用、max_output_tokens=1000）
5. レスポンス解析とメタデータ付与

**質問タイプ**:
- **fact**: 事実確認型（What is...? / 〜は何ですか？）
- **reason**: 理由説明型（Why...? / なぜ〜ですか？）
- **comparison**: 比較型（What's the difference...? / 〜と〜の違いは？）
- **application**: 応用型（How is... used? / 〜はどのように活用されますか？）

### 4.4 データセット全体の生成制御

#### `generate_qa_for_dataset(chunks, dataset_type, model, chunk_batch_size, merge_chunks, min_tokens, max_tokens) -> List[Dict]`

**目的**: データセット全体のQ/Aペア生成を統括

**処理フロー**:

1. **前処理**:
   ```python
   if merge_chunks:
       processed_chunks = merge_small_chunks(chunks, min_tokens, max_tokens)
   else:
       processed_chunks = chunks

   api_calls = (len(processed_chunks) + chunk_batch_size - 1) // chunk_batch_size
   ```

2. **バッチ処理ループ**:
   ```python
   for i in range(0, total_chunks, chunk_batch_size):
       batch = processed_chunks[i:i+chunk_batch_size]

       # リトライ機能付きQ/A生成（最大3回）
       for attempt in range(max_retries):
           try:
               if chunk_batch_size == 1:
                   qa_pairs = generate_qa_pairs_for_chunk(batch[0], ...)
               else:
                   qa_pairs = generate_qa_pairs_for_batch(batch, ...)

               if qa_pairs:
                   all_qa_pairs.extend(qa_pairs)
               break
           except Exception as e:
               # リトライまたはフォールバック
   ```

3. **リトライ制御**:
   - 最大3回リトライ
   - 指数バックオフ（2^attempt秒待機）
   - 最終失敗時は個別処理にフォールバック

4. **API制限対策**:
   - バッチ間で0.2秒待機（バッチ処理により短縮）

**パラメータ**:
- `chunk_batch_size`: 1-5（デフォルト: 3、推奨: 5）
- `merge_chunks`: bool（デフォルト: True）
- `min_tokens`: 統合対象最小トークン（デフォルト: 150）
- `max_tokens`: 統合後最大トークン（デフォルト: 400）

---

## 5. カバレージ分析

### 5.1 多段階カバレージ計算

#### `analyze_coverage(chunks: List[Dict], qa_pairs: List[Dict], dataset_type: str) -> Dict`

**目的**: 生成Q/Aペアがドキュメントチャンクをどれだけカバーしているか多段階で分析

**処理手順**:

1. **埋め込み生成**:
   ```python
   doc_embeddings = analyzer.generate_embeddings(chunks)

   qa_embeddings = []
   for qa in qa_pairs:
       qa_text = f"{qa['question']} {qa['answer']}"
       embedding = analyzer.generate_embedding(qa_text)
       qa_embeddings.append(embedding)
   ```

2. **カバレージ行列計算**:
   ```python
   coverage_matrix = np.zeros((len(chunks), len(qa_pairs)))
   for i in range(len(chunks)):
       for j in range(len(qa_pairs)):
           similarity = cosine_similarity(doc_embeddings[i], qa_embeddings[j])
           coverage_matrix[i, j] = similarity
   ```

3. **多段階カバレージ判定**:
   - データセット別最適閾値を自動取得（`get_optimal_thresholds()`）
   - 3段階評価（strict/standard/lenient）
   - 各閾値でカバレージ率を算出
   - 未カバーチャンクとギャップを記録

**データセット別閾値**:
| データセット | Strict | Standard | Lenient |
|------------|--------|----------|---------|
| cc_news | 0.80 | 0.70 | 0.60 |
| japanese_text | 0.75 | 0.65 | 0.55 |
| wikipedia_ja | 0.85 | 0.75 | 0.65 |

### 5.2 チャンク特性別分析

#### `analyze_chunk_characteristics_coverage(chunks, coverage_matrix, qa_pairs, threshold) -> Dict`

**目的**: チャンクの特性（長さ、位置）別にカバレージを分析

**分析軸**:

1. **長さ別分析**:
   - **short**: < 100トークン
   - **medium**: 100-200トークン
   - **long**: >= 200トークン

   各カテゴリで以下を計算:
   - チャンク数（count）
   - カバー済み数（covered）
   - カバレージ率（coverage_rate）
   - 平均類似度（avg_similarity）

2. **位置別分析**:
   - **beginning**: 文書前半33%
   - **middle**: 文書中盤34%
   - **end**: 文書後半33%

   各カテゴリで同様の指標を計算

3. **自動インサイト生成**:
   - カバレージ率70%未満のカテゴリを自動検出
   - 改善提案を生成（例: "shortチャンクのカバレージが低い（65.0%）"）

**返却データ構造**:
```python
{
    # 基本メトリクス
    "coverage_rate": 0.85,
    "covered_chunks": 43,
    "total_chunks": 50,
    "threshold": 0.70,

    # 多段階カバレージ
    "multi_threshold": {
        "strict": {
            "threshold": 0.80,
            "covered_chunks": 40,
            "coverage_rate": 0.80,
            "uncovered_count": 10,
            "uncovered_chunks": [...]
        },
        "standard": {...},
        "lenient": {...}
    },

    # チャンク特性別分析
    "chunk_analysis": {
        "by_length": {
            "short": {"count": 15, "covered": 12, "coverage_rate": 0.80, "avg_similarity": 0.75},
            "medium": {...},
            "long": {...}
        },
        "by_position": {
            "beginning": {"count": 17, "covered": 16, "coverage_rate": 0.94, "avg_similarity": 0.78},
            "middle": {...},
            "end": {...}
        },
        "summary": {
            "total_chunks": 50,
            "total_qa_pairs": 150,
            "threshold_used": 0.70,
            "insights": [
                "shortチャンクのカバレージが低い（65.0%）",
                "文書end部分のカバレージが低い（68.0%）"
            ]
        }
    },

    # データセット情報
    "dataset_type": "cc_news",
    "optimal_thresholds": {"strict": 0.80, "standard": 0.70, "lenient": 0.60}
}
```

---

## 6. 結果保存

### `save_results(qa_pairs, coverage_results, dataset_type, output_dir) -> Dict[str, str]`

**出力ディレクトリ**: `qa_output/a02/` （サブディレクトリ自動作成）

**保存ファイル**:

1. **Q/Aペア（JSON）**: `qa_pairs_{dataset_type}_{timestamp}.json`
   - ensure_ascii=False（日本語対応）
   - indent=2（可読性向上）

2. **Q/Aペア（CSV）**: `qa_pairs_{dataset_type}_{timestamp}.csv`
   - pd.DataFrameで変換
   - encoding='utf-8'

3. **カバレージ分析結果（JSON）**: `coverage_{dataset_type}_{timestamp}.json`
   - multi_threshold結果含む
   - chunk_analysis結果含む
   - uncovered_chunksはプレビュー版（200文字まで）

4. **サマリー（JSON）**: `summary_{dataset_type}_{timestamp}.json`
   ```json
   {
     "dataset_type": "cc_news",
     "dataset_name": "CC-News英語ニュース",
     "generated_at": "20251023_141030",
     "total_qa_pairs": 525,
     "coverage_rate": 0.85,
     "covered_chunks": 43,
     "total_chunks": 50,
     "files": {
       "qa_json": "qa_output/a02/qa_pairs_cc_news_20251023_141030.json",
       "qa_csv": "qa_output/a02/qa_pairs_cc_news_20251023_141030.csv",
       "coverage": "qa_output/a02/coverage_cc_news_20251023_141030.json",
       "summary": "qa_output/a02/summary_cc_news_20251023_141030.json"
     }
   }
   ```

---

## 7. コマンドライン引数

### 7.1 必須引数
なし（すべてオプション）

### 7.2 オプション引数

| 引数 | 型 | デフォルト | 選択肢 | 説明 |
|-----|-----|----------|-------|------|
| --dataset | str | cc_news | cc_news, japanese_text, wikipedia_ja | 処理するデータセット |
| --model | str | gpt-5-mini | - | 使用するOpenAIモデル |
| --output | str | qa_output | - | 出力ディレクトリ（a02サブディレクトリ自動作成） |
| --max-docs | int | None | - | 処理する最大文書数（テスト用） |
| --analyze-coverage | flag | False | - | カバレージ分析を実行 |
| --batch-chunks | int | 3 | 1, 2, 3, 4, 5 | 1回のAPIで処理するチャンク数 |
| --merge-chunks | flag | True | - | 小さいチャンクを統合する（デフォルト有効） |
| --no-merge-chunks | flag | False | - | チャンク統合を無効化 |
| --min-tokens | int | 150 | - | 統合対象の最小トークン数 |
| --max-tokens | int | 400 | - | 統合後の最大トークン数 |

### 7.3 使用例

```bash
# 【推奨】本番運用設定（全文書、バッチ5、15-20分、$0.10-0.15）
python a02_make_qa.py \
    --dataset cc_news \
    --batch-chunks 5 \
    --merge-chunks \
    --min-tokens 150 \
    --max-tokens 400 \
    --model gpt-5-mini \
    --analyze-coverage

# 【最効率】API呼び出し最小化（バッチ5）
python a02_make_qa.py \
    --dataset cc_news \
    --batch-chunks 5 \
    --merge-chunks \
    --analyze-coverage

# テスト実行（10文書のみ）
python a02_make_qa.py \
    --dataset cc_news \
    --model gpt-5-mini \
    --analyze-coverage \
    --max-docs 10

# Wikipedia日本語版、カバレージ分析あり
python a02_make_qa.py \
    --dataset wikipedia_ja \
    --model gpt-5-mini \
    --analyze-coverage \
    --max-docs 10

# 日本語テキスト、個別処理（バッチサイズ1）
python a02_make_qa.py \
    --dataset japanese_text \
    --batch-chunks 1 \
    --no-merge-chunks

# カスタムチャンク統合設定
python a02_make_qa.py \
    --min-tokens 100 \
    --max-tokens 500 \
    --batch-chunks 5
```

---

## 8. メイン処理フロー

### 8.1 main()関数の実行ステップ

```
[初期化]
  ↓
APIキーチェック（OPENAI_API_KEY）
  ↓
[1/4] データ読み込み
  ↓
  load_preprocessed_data(args.dataset)
  ↓
[2/4] チャンク作成
  ↓
  create_document_chunks(df, args.dataset, args.max_docs)
  ↓
[3/4] Q/Aペア生成
  ↓
  generate_qa_for_dataset(
      chunks,
      dataset_type=args.dataset,
      model=args.model,
      chunk_batch_size=args.batch_chunks,
      merge_chunks=args.merge_chunks,
      min_tokens=args.min_tokens,
      max_tokens=args.max_tokens
  )
  ↓
[4/4] カバレージ分析（オプション）
  ↓
  if args.analyze_coverage:
      analyze_coverage(chunks, qa_pairs, args.dataset)
  ↓
結果保存
  ↓
  save_results(qa_pairs, coverage_results, args.dataset, args.output)
  ↓
統計情報表示
  ├─ 質問タイプ別統計
  ├─ 多段階カバレージ結果
  └─ チャンク特性別分析結果
```

### 8.2 エラーハンドリング

1. **API Key未設定**:
   - 環境変数`OPENAI_API_KEY`チェック
   - "your-openai-api-key-here"も不正と判定
   - `sys.exit(1)`で終了

2. **チャンク作成失敗**:
   - 空のチャンクリスト → エラーログ + `sys.exit(1)`

3. **Q/A生成失敗**:
   - 個別チャンクエラー: 警告ログ出力して継続
   - バッチエラー: リトライ（最大3回） → フォールバック（個別処理）

4. **全体エラー**:
   - try-exceptでキャッチ
   - `traceback.print_exc()`で詳細出力
   - `sys.exit(1)`

---

## 9. 依存関係

### 9.1 外部ライブラリ
- **openai** (>= 1.100.2): OpenAI API クライアント（responses.parse使用）
- **pandas**: データフレーム操作
- **numpy**: 数値計算、行列演算
- **tiktoken**: トークン数カウント（cl100k_base）
- **pydantic**: データモデル定義・バリデーション
- **python-dotenv**: 環境変数管理

### 9.2 内部モジュール
- **a03_rag_qa_coverage_improved.SemanticCoverage**:
  - `create_semantic_chunks()`: チャンク分割
  - `generate_embeddings()`: 埋め込み生成（バッチ）
  - `generate_embedding()`: 埋め込み生成（単一）
  - `cosine_similarity()`: コサイン類似度計算

---

## 10. パフォーマンス最適化

### 10.1 API呼び出し削減

**チャンク統合による削減**:
- 小さいチャンクを統合（min_tokens=150）
- 例: 1,825チャンク → 365チャンク（80%削減）

**バッチ処理による削減**:
- 複数チャンク同時処理（chunk_batch_size=1-5）
- 効果例（バッチサイズ5の場合）:
  - 365チャンク ÷ 5 = 約73回のAPI呼び出し
  - 従来比（個別処理）: 365回 → 73回（**80%削減**）

**総合効果**:
| 段階 | チャンク数 | API呼び出し | 削減率 |
|------|----------|-----------|-------|
| 元データ | 1,825個 | 1,825回 | - |
| チャンク統合後 | 365個 | 365回 | 80% |
| バッチ処理後（サイズ5） | 365個 | 73回 | 96% |

### 10.2 エラー回復

- **指数バックオフ**: 2^attempt秒待機（attempt: 0, 1, 2 → 1秒, 2秒, 4秒）
- **フォールバック**: バッチ失敗時は個別処理に切替
- **最大リトライ**: 3回

### 10.3 レート制限対策

- バッチ間0.2秒待機（バッチ処理により短縮）
- max_output_tokens制御:
  - 単一チャンク: 1000トークン
  - バッチ処理: 4000トークン

---

## 11. 出力統計情報

### 11.1 質問タイプ別統計

生成完了後、質問タイプごとの件数を表示:
```
質問タイプ別統計:
  application: 128件
  comparison: 102件
  fact: 156件
  reason: 139件
```

### 11.2 カバレージ統計（多段階分析対応）

```
多段階カバレージ分析結果:
- Strict  (閾値0.80): 80.0%
- Standard(閾値0.70): 85.0%
- Lenient (閾値0.60): 92.0%

チャンク特性別カバレージ:
長さ別:
- Short チャンク: 80.0%
- Medium チャンク: 88.0%
- Long チャンク: 85.0%

位置別:
- Beginning (前半): 94.0%
- Middle (中盤): 82.0%
- End (後半): 78.0%

📊 分析インサイト:
  • 文書end部分のカバレージが低い（78.0%）
  • Shortチャンクで追加Q/A生成の余地あり
```

---

## 12. 注意事項・制約

### 12.1 制約事項

1. **チャンクサイズ**:
   - `create_semantic_chunks()`は内部で200トークン固定
   - DATASET_CONFIGSのchunk_sizeは現在未使用

2. **テキスト長制限**:
   - 単一チャンク処理: 2000文字まで
   - バッチ処理（チャンク結合時）: 1000文字/チャンク

3. **バッチサイズ**: 最大5チャンク（推奨: 3-5）

4. **カバレージ閾値**:
   - データセット別に自動設定（OPTIMAL_THRESHOLDS）
   - コマンドライン引数での変更は不可（将来の改善予定）

### 12.2 推奨設定

**日本語データセット**:
- `batch_chunks`: 2-3（日本語は長くなる傾向）
- `min_tokens`: 150, `max_tokens`: 400

**英語データセット**:
- `batch_chunks`: 3-5
- `min_tokens`: 100, `max_tokens`: 500

### 12.3 コスト管理

- `--max-docs`でテスト実行推奨（10文書で約$0.005-0.01）
- バッチサイズ調整でAPI呼び出し削減
- カバレージ分析は埋め込み生成コストが追加（必要時のみ実行）

**推定コスト（gpt-5-mini使用時）**:
| 文書数 | API呼び出し | 推定コスト |
|-------|-----------|----------|
| 10文書 | ~2-3回 | $0.005-0.01 |
| 100文書 | ~20-25回 | $0.05-0.08 |
| 全文書（497件） | ~73回 | $0.10-0.15 |

---

## 13. 今後の改善案

### 13.1 機能拡張

1. **カバレージ閾値のコマンドライン引数化**:
   - `--coverage-threshold-strict`, `--coverage-threshold-standard`, `--coverage-threshold-lenient`

2. **create_semantic_chunksのトークン数制御**:
   - 設定のchunk_sizeを実際に使用

3. **質問タイプの重み付け設定**:
   - `--question-type-weights fact:0.3,reason:0.3,comparison:0.2,application:0.2`

4. **複数モデル同時実行（比較用）**:
   - `--models gpt-5-mini,gpt-4o-mini`

### 13.2 最適化

1. **埋め込みキャッシュ機構**:
   - 同一チャンクの埋め込みを再利用

2. **並列処理（asyncio対応）**:
   - 複数バッチの並列実行

3. **プログレスバー表示（tqdm導入）**:
   - リアルタイム進捗表示

4. **中間結果の自動保存（レジューム機能）**:
   - 処理中断時の再開機能

### 13.3 品質向上

1. **Q/Aペアの自動品質評価**:
   - 質問の明確性、回答の正確性を自動評価

2. **重複Q/Aの検出・除去**:
   - 類似Q/Aペアの自動検出

3. **難易度別の質問生成**:
   - easy/medium/hardの3段階

4. **ユーザーフィードバック機構**:
   - Q/Aペアの有用性を評価

---

## 変更履歴

### v2.5 (2025-10-23)
- 出力ディレクトリを`qa_output/a02/`に変更（サブディレクトリ自動作成）
- ドキュメント全面更新（最新仕様を反映）

### v2.4
- バッチ処理最適化（バッチサイズ1-5対応）
- 多段階カバレージ分析実装

### v2.3
- チャンク統合機能実装
- データセット別最適閾値設定

### v2.2
- OpenAI Responses API対応（responses.parse）
- Pydanticモデル定義追加

### v2.1
- 初期リリース