# a02_make_qa.py - 詳細設計書

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
   - 4.2 [バッチ処理（3チャンク同時）](#42-バッチ処理3チャンク同時)
   - 4.3 [単一チャンク処理](#43-単一チャンク処理)
   - 4.4 [データセット全体の生成制御](#44-データセット全体の生成制御)
5. [カバレージ分析](#5-カバレージ分析)
   - 5.1 [セマンティックカバレージ計算](#51-セマンティックカバレージ計算)
6. [結果保存](#6-結果保存)
   - 6.1 [save_results](#61-save_resultsqa_pairs-coverage_results-dataset_type-output_dir---dictstr-str)
7. [コマンドライン引数](#7-コマンドライン引数)
   - 7.1 [必須引数](#71-必須引数)
   - 7.2 [オプション引数](#72-オプション引数)
   - 7.3 [使用例](#73-使用例)
8. [メイン処理フロー](#8-メイン処理フロー)
   - 8.1 [main()関数の実行ステップ](#81-main関数の実行ステップ)
   - 8.2 [エラーハンドリング](#82-エラーハンドリング)
9. [依存関係](#9-依存関係)
   - 9.1 [外部ライブラリ](#91-外部ライブラリ)
   - 9.2 [内部モジュール](#92-内部モジュール)
10. [パフォーマンス最適化](#10-パフォーマンス最適化)
    - 10.1 [API呼び出し削減](#101-api呼び出し削減)
    - 10.2 [エラー回復](#102-エラー回復)
    - 10.3 [レート制限対策](#103-レート制限対策)
11. [出力統計情報](#11-出力統計情報)
    - 11.1 [質問タイプ別統計](#111-質問タイプ別統計)
    - 11.2 [カバレージ統計](#112-カバレージ統計)
12. [注意事項・制約](#12-注意事項制約)
    - 12.1 [制約事項](#121-制約事項)
    - 12.2 [推奨設定](#122-推奨設定)
    - 12.3 [コスト管理](#123-コスト管理)
13. [今後の改善案](#13-今後の改善案)
    - 13.1 [機能拡張](#131-機能拡張)
    - 13.2 [最適化](#132-最適化)
    - 13.3 [品質向上](#133-品質向上)

---

## 1. 概要

### 1.1 目的
preprocessedファイルからOpenAI APIを使用してQ/Aペアを自動生成し、生成されたQ/Aペアのセマンティックカバレージを分析するシステム。

### 1.2 主要機能
- preprocessed CSVファイルの読み込みと前処理
- 文書のセマンティックチャンク分割
- チャンクの統合最適化
- バッチ処理によるQ/Aペア生成（3チャンク同時処理対応）
- セマンティックカバレージ分析
- 結果のJSON/CSV形式での保存

### 1.3 対応データセット
| データセット | ファイルパス | 言語 | チャンクサイズ | Q/Aペア数/チャンク |
|------------|-------------|------|--------------|------------------|
| cc_news | OUTPUT/preprocessed_cc_news.csv | 英語 | 300トークン | 3 |
| japanese_text | OUTPUT/preprocessed_japanese_text.csv | 日本語 | 200トークン | 2 |
| wikipedia_ja | OUTPUT/preprocessed_wikipedia_ja.csv | 日本語 | 250トークン | 3 |

## 2. アーキテクチャ

### 2.1 システム構成図

**処理フロー:**

1. preprocessed CSV読み込み
2. データ読み込み処理
3. チャンク作成
4. チャンク統合（小さいチャンクをマージ）
5. バッチ処理でQ/A生成（OpenAI API: responses.parse使用）
6. カバレージ分析（セマンティック類似度計算）
7. 結果保存
8. JSON/CSV出力

### 2.2 主要コンポーネント

#### 2.2.1 データモデル（Pydantic）
- **QAPair**: 個別Q/Aペアのデータモデル
  - question: str - 質問文
  - answer: str - 回答文
  - question_type: str - 質問タイプ（fact/reason/comparison/application）
  - source_chunk_id: Optional[str] - ソースチャンクID
  - dataset_type: Optional[str] - データセット種別
  - auto_generated: bool - 自動生成フラグ

- **QAPairsResponse**: API応答用モデル
  - qa_pairs: List[QAPair] - Q/Aペアのリスト

#### 2.2.2 設定管理
- **DATASET_CONFIGS**: データセット別設定辞書
  - name: データセット名
  - file: ファイルパス
  - text_column: テキストカラム名
  - title_column: タイトルカラム名（オプション）
  - lang: 言語コード（"ja" or "en"）
  - chunk_size: チャンクサイズ（トークン数）
  - qa_per_chunk: チャンクあたりのQ/Aペア数

## 3. データ処理フロー

### 3.1 データ読み込み・前処理

#### load_preprocessed_data(dataset_type: str) -> pd.DataFrame
**目的**: preprocessed CSVファイルを読み込み、前処理を実行

**処理手順**:
1. DATASET_CONFIGSから設定取得
2. ファイル存在チェック
3. CSVファイル読み込み
4. 必須カラム存在確認
5. 空テキスト除外
6. DataFrameを返却

**エラーハンドリング**:
- 未対応データセット: ValueError
- ファイル不在: FileNotFoundError
- カラム不在: ValueError

### 3.2 チャンク作成

#### create_document_chunks(df: pd.DataFrame, dataset_type: str, max_docs: Optional[int]) -> List[Dict]
**目的**: DataFrameから文書チャンクを作成

**処理手順**:
1. 設定情報取得（text_column, title_column, chunk_size）
2. SemanticCoverageインスタンス生成
3. 処理文書数制限（max_docsが指定された場合）
4. 各文書を反復処理:
   - テキスト抽出（str型に変換）
   - doc_id生成（タイトル含む場合は先頭30文字）
   - SemanticCoverage.create_semantic_chunksでチャンク分割
   - メタデータ追加（doc_id, doc_idx, chunk_idx, dataset_type）
5. 全チャンクリストを返却

**注意事項**:
- create_semantic_chunksは内部で200トークン固定を使用（設定のchunk_sizeは反映されない）
- エラー発生時は警告ログ出力してcontinue

### 3.3 チャンク統合

#### merge_small_chunks(chunks: List[Dict], min_tokens: int = 150, max_tokens: int = 400) -> List[Dict]
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
   - original_chunksリストに元チャンクIDを記録
   - chunk_idxを範囲形式で記録（例: "0-2"）
5. 統合チャンクリストを返却

**効果**:
- API呼び出し回数削減
- コスト削減
- 文脈の連続性向上

## 4. Q/Aペア生成

### 4.1 Q/A数決定ロジック

#### determine_qa_count(chunk: Dict, config: Dict) -> int
**目的**: チャンクのトークン数に基づいて最適なQ/A数を決定

**ロジック**:
| トークン数範囲 | Q/A数 |
|--------------|-------|
| < 50 | min(base_count, 1) |
| 50-99 | min(base_count, 2) |
| 100-199 | base_count |
| >= 200 | min(base_count + 1, 5) |

### 4.2 バッチ処理（3チャンク同時）

#### generate_qa_pairs_for_batch(chunks: List[Dict], config: Dict, model: str, client: OpenAI) -> List[Dict]
**目的**: 複数チャンク（最大3個）から一度にQ/Aペアを生成

**処理フロー**:
1. チャンク数チェック:
   - 0個: 空リスト返却
   - 1個: generate_qa_pairs_for_chunk()へ委譲
   - 2個以上: バッチ処理継続

2. プロンプト構築（言語別）:
   - 日本語の場合:
     - システムプロンプト: 教育コンテンツ作成専門家
     - 複数テキストを【テキスト1】【テキスト2】...で結合
     - 各チャンクのQ/A数を計算し合計を指示
   - 英語の場合: 同様の構造を英語で構築

3. OpenAI API呼び出し:
   ```python
   response = client.responses.parse(
       input=combined_input,
       model=model,
       text_format=QAPairsResponse,
       max_output_tokens=4000
   )
   ```

4. レスポンス解析:
   - parsed_dataからQ/Aペアを取得
   - 各チャンクに期待される数だけQ/Aを順次割り当て
   - メタデータ追加（source_chunk_id, doc_id, dataset_type, chunk_idx）

5. エラーハンドリング:
   - 例外発生時はフォールバックで個別処理

### 4.3 単一チャンク処理

#### generate_qa_pairs_for_chunk(chunk: Dict, config: Dict, model: str, client: OpenAI) -> List[Dict]
**目的**: 単一チャンクからQ/Aペアを生成（後方互換性維持）

**処理フロー**:
1. Q/A数決定（determine_qa_count）
2. 言語別プロンプト構築:
   - システムプロンプト: 生成ルール指示
   - ユーザープロンプト: テキスト + 質問タイプ + JSON形式指示
3. テキスト長制限:
   - 2000文字超の場合は切り詰め
4. API呼び出し（responses.parse使用）
5. レスポンス解析とメタデータ付与

**質問タイプ**:
- fact: 事実確認型（What is...?）
- reason: 理由説明型（Why...?）
- comparison: 比較型（What's the difference...?）
- application: 応用型（How is... used?）

### 4.4 データセット全体の生成制御

#### generate_qa_for_dataset(chunks, dataset_type, model, chunk_batch_size, merge_chunks, min_tokens, max_tokens) -> List[Dict]
**目的**: データセット全体のQ/Aペア生成を統括

**処理フロー**:
1. 前処理:
   - merge_chunks=Trueの場合、merge_small_chunksで統合
   - API呼び出し回数計算

2. バッチ処理ループ:
   ```python
   for i in range(0, total_chunks, chunk_batch_size):
       batch = processed_chunks[i:i+chunk_batch_size]
       # リトライ機能付きQ/A生成
       for attempt in range(max_retries):
           # 生成実行
   ```

3. リトライ制御:
   - 最大3回リトライ
   - 指数バックオフ（2^attempt秒待機）
   - 最終失敗時は個別処理にフォールバック

4. API制限対策:
   - バッチ間で0.5秒待機

**パラメータ**:
- chunk_batch_size: 1-5（デフォルト: 3）
- merge_chunks: bool（デフォルト: True）
- min_tokens: 統合対象最小トークン（デフォルト: 150）
- max_tokens: 統合後最大トークン（デフォルト: 400）

## 5. カバレージ分析

### 5.1 セマンティックカバレージ計算

#### analyze_coverage(chunks: List[Dict], qa_pairs: List[Dict]) -> Dict
**目的**: 生成Q/Aペアがドキュメントチャンクをどれだけカバーしているか分析

**処理手順**:
1. 埋め込み生成:
   - チャンク埋め込み: SemanticCoverage.generate_embeddings()
   - Q/A埋め込み: 質問+回答を結合してgenerate_embedding()

2. カバレージ行列計算:
   ```python
   coverage_matrix = np.zeros((len(chunks), len(qa_pairs)))
   for i, j in itertools.product(range(len(chunks)), range(len(qa_pairs))):
       similarity = cosine_similarity(doc_embeddings[i], qa_embeddings[j])
       coverage_matrix[i, j] = similarity
   ```

3. カバレージ判定:
   - 閾値: 0.7（コサイン類似度）
   - 各チャンクの最大類似度 > 0.7 → カバー済み
   - coverage_rate = covered_chunks / total_chunks

4. 未カバーチャンク特定:
   - 最大類似度 < 0.7 のチャンクをリスト化
   - gap（閾値までの距離）も記録

**返却データ**:
```python
{
    "coverage_rate": float,
    "covered_chunks": int,
    "total_chunks": int,
    "uncovered_chunks": List[Dict],
    "max_similarities": List[float],
    "threshold": float
}
```

## 6. 結果保存

### 6.1 save_results(qa_pairs, coverage_results, dataset_type, output_dir) -> Dict[str, str]

**保存ファイル**:
1. **Q/Aペア（JSON）**: `qa_pairs_{dataset_type}_{timestamp}.json`
   - ensure_ascii=False（日本語対応）
   - indent=2（可読性向上）

2. **Q/Aペア（CSV）**: `qa_pairs_{dataset_type}_{timestamp}.csv`
   - pd.DataFrameで変換
   - encoding='utf-8'

3. **カバレージ分析結果（JSON）**: `coverage_{dataset_type}_{timestamp}.json`
   - uncovered_chunksはプレビュー版（200文字まで）

4. **サマリー（JSON）**: `summary_{dataset_type}_{timestamp}.json`
   ```json
   {
     "dataset_type": "cc_news",
     "dataset_name": "CC-News英語ニュース",
     "generated_at": "20241004_141030",
     "total_qa_pairs": 150,
     "coverage_rate": 0.85,
     "covered_chunks": 43,
     "total_chunks": 50,
     "files": {...}
   }
   ```

## 7. コマンドライン引数

### 7.1 必須引数
なし（すべてオプション）

### 7.2 オプション引数

| 引数 | 型 | デフォルト | 説明 |
|-----|-----|----------|------|
| --dataset | str | cc_news | 処理するデータセット（cc_news/japanese_text/wikipedia_ja） |
| --model | str | gpt-5-mini | 使用するOpenAIモデル |
| --output | str | qa_output | 出力ディレクトリ |
| --max-docs | int | None | 処理する最大文書数（テスト用） |
| --analyze-coverage | flag | False | カバレージ分析を実行 |
| --batch-chunks | int | 3 | 1回のAPIで処理するチャンク数（1-5） |
| --merge-chunks | flag | True | 小さいチャンクを統合する |
| --no-merge-chunks | flag | False | チャンク統合を無効化 |
| --min-tokens | int | 150 | 統合対象の最小トークン数 |
| --max-tokens | int | 400 | 統合後の最大トークン数 |

### 7.3 使用例

```bash
# 基本使用（cc_newsデータセット、デフォルト設定）
python a02_make_qa.py

# Wikipedia日本語版、カバレージ分析あり
python a02_make_qa.py --dataset wikipedia_ja --analyze-coverage

# 日本語テキスト、バッチサイズ1（個別処理）、チャンク統合なし
python a02_make_qa.py --dataset japanese_text --batch-chunks 1 --no-merge-chunks

# テスト実行（最大10文書、gpt-4o-mini使用）
python a02_make_qa.py --max-docs 10 --model gpt-4o-mini

# カスタムチャンク統合設定
python a02_make_qa.py --min-tokens 100 --max-tokens 500 --batch-chunks 5
```

## 8. メイン処理フロー

### 8.1 main()関数の実行ステップ

```
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
  generate_qa_for_dataset(chunks, ...)
  ↓
[4/4] カバレージ分析（オプション）
  ↓
  analyze_coverage(chunks, qa_pairs)
  ↓
結果保存
  ↓
  save_results(qa_pairs, coverage_results, ...)
  ↓
統計情報表示
```

### 8.2 エラーハンドリング
1. **API Key未設定**:
   - 環境変数OPENAI_API_KEYチェック
   - "your-openai-api-key-here"も不正と判定
   - sys.exit(1)で終了

2. **チャンク作成失敗**:
   - 空のチャンクリスト → エラーログ + sys.exit(1)

3. **Q/A生成失敗**:
   - 個別チャンクエラー: 警告ログ出力して継続
   - バッチエラー: リトライ → フォールバック

4. **全体エラー**:
   - try-exceptでキャッチ
   - traceback.print_exc()で詳細出力
   - sys.exit(1)

## 9. 依存関係

### 9.1 外部ライブラリ
- **openai**: OpenAI API クライアント（responses.parse使用）
- **pandas**: データフレーム操作
- **numpy**: 数値計算、行列演算
- **tiktoken**: トークン数カウント（cl100k_base）
- **pydantic**: データモデル定義・バリデーション
- **python-dotenv**: 環境変数管理

### 9.2 内部モジュール
- **rag_qa.SemanticCoverage**:
  - create_semantic_chunks(): チャンク分割
  - generate_embeddings(): 埋め込み生成（バッチ）
  - generate_embedding(): 埋め込み生成（単一）
  - cosine_similarity(): コサイン類似度計算

## 10. パフォーマンス最適化

### 10.1 API呼び出し削減
- **チャンク統合**: 小さいチャンクを統合して呼び出し回数削減
- **バッチ処理**: 3チャンク同時処理でAPI呼び出し1/3に削減
- **効果例**:
  - 元150チャンク → 統合100チャンク → バッチ34回呼び出し
  - 従来比: 150回 → 34回（約77%削減）

### 10.2 エラー回復
- **指数バックオフ**: 2^attempt秒待機
- **フォールバック**: バッチ失敗時は個別処理
- **最大リトライ**: 3回

### 10.3 レート制限対策
- バッチ間0.5秒待機
- max_output_tokens制御（単一: 1000、バッチ: 4000）

## 11. 出力統計情報

### 11.1 質問タイプ別統計
生成完了後、質問タイプごとの件数を表示:
```
質問タイプ別統計:
  application: 45件
  comparison: 38件
  fact: 42件
  reason: 25件
```

### 11.2 カバレージ統計
```
カバレージ分析結果:
- カバレージ率: 85.0%
- カバー済みチャンク: 43/50
- 未カバーチャンク: 7
```

## 12. 注意事項・制約

### 12.1 制約事項
1. **チャンクサイズ**: create_semantic_chunksは内部で200トークン固定（設定のchunk_sizeは使用されない）
2. **テキスト長制限**:
   - 単一チャンク処理: 2000文字まで
   - バッチ処理（チャンク結合時）: 1000文字まで
3. **バッチサイズ**: 最大5チャンク（推奨: 3）
4. **カバレージ閾値**: 0.7固定（ハードコード）

### 12.2 推奨設定
- **日本語データセット**:
  - batch_chunks: 2-3（日本語は長くなる傾向）
  - min_tokens: 150, max_tokens: 400

- **英語データセット**:
  - batch_chunks: 3-5
  - min_tokens: 100, max_tokens: 500

### 12.3 コスト管理
- `--max-docs`でテスト実行推奨
- バッチサイズ調整でAPI呼び出し削減
- カバレージ分析は埋め込み生成コストが追加（必要時のみ実行）

## 13. 今後の改善案

### 13.1 機能拡張
1. カバレージ閾値をコマンドライン引数化
2. create_semantic_chunksのトークン数を設定から制御
3. 質問タイプの重み付け設定
4. 複数モデル同時実行（比較用）

### 13.2 最適化
1. 埋め込みキャッシュ機構
2. 並列処理（asyncio対応）
3. プログレスバー表示（tqdm導入）
4. 中間結果の自動保存（レジューム機能）

### 13.3 品質向上
1. Q/Aペアの自動品質評価
2. 重複Q/Aの検出・除去
3. 難易度別の質問生成
4. ユーザーフィードバック機構