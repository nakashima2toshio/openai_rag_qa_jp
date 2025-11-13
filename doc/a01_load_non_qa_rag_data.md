# a01_load_non_qa_rag_data.py - 技術仕様書

## 目次

1. [概要](#1-概要)
   - 1.1 [目的](#11-目的)
   - 1.2 [主要機能](#12-主要機能)
   - 1.3 [対応データセット](#13-対応データセット)
2. [アーキテクチャ](#2-アーキテクチャ)
   - 2.1 [システム構成図](#21-システム構成図)
   - 2.2 [UIレイアウト](#22-uiレイアウト)
   - 2.3 [主要コンポーネント](#23-主要コンポーネント)
3. [データセット設定](#3-データセット設定)
   - 3.1 [NonQARAGConfigクラス](#31-nonqaragconfigクラス)
   - 3.2 [データセット別設定](#32-データセット別設定)
4. [データ検証機能](#4-データ検証機能)
   - 4.1 [Wikipedia特有の検証](#41-wikipedia特有の検証)
   - 4.2 [ニュースデータ特有の検証](#42-ニュースデータ特有の検証)
   - 4.3 [学術論文データ特有の検証](#43-学術論文データ特有の検証)
   - 4.4 [コードデータ特有の検証](#44-コードデータ特有の検証)
   - 4.5 [Stack Overflow特有の検証](#45-stack-overflow特有の検証)
5. [データ処理](#5-データ処理)
   - 5.1 [テキスト抽出](#51-テキスト抽出)
   - 5.2 [前処理オプション](#52-前処理オプション)
6. [HuggingFace統合](#6-huggingface統合)
   - 6.1 [自動ダウンロード機能](#61-自動ダウンロード機能)
   - 6.2 [ストリーミングモード](#62-ストリーミングモード)
   - 6.3 [メタデータ管理](#63-メタデータ管理)
7. [UI機能](#7-ui機能)
   - 7.1 [タブ構成](#71-タブ構成)
   - 7.2 [インタラクティブ設定](#72-インタラクティブ設定)
8. [出力・保存](#8-出力保存)
   - 8.1 [ダウンロード機能](#81-ダウンロード機能)
   - 8.2 [OUTPUTフォルダ保存](#82-outputフォルダ保存)
9. [helper_rag連携](#9-helper_rag連携)
   - 9.1 [インポート関数一覧](#91-インポート関数一覧)
   - 9.2 [主要クラス](#92-主要クラス)
10. [使用方法](#10-使用方法)
    - 10.1 [起動方法](#101-起動方法)
    - 10.2 [基本ワークフロー](#102-基本ワークフロー)
    - 10.3 [推奨設定](#103-推奨設定)
11. [エラーハンドリング](#11-エラーハンドリング)
    - 11.1 [HuggingFaceエラー](#111-huggingfaceエラー)
    - 11.2 [データ処理エラー](#112-データ処理エラー)
12. [今後の改善案](#12-今後の改善案)
    - 12.1 [機能拡張](#121-機能拡張)
    - 12.2 [パフォーマンス最適化](#122-パフォーマンス最適化)

---

## 1. 概要

### 1.1 目的
非Q&A型RAGデータ（Wikipedia、ニュース、学術論文など）を処理し、RAG（Retrieval-Augmented Generation）システム用の前処理済みテキストデータを生成するStreamlitベースのWebアプリケーション。

### 1.2 主要機能
- ✅ **日本語・英語データセットの処理**
  - Wikipedia日本語版（動作確認済み）
  - CC100日本語Webテキスト（動作確認済み）
  - CC-News英語ニュース（動作確認済み）
  - Livedoorニュースコーパス（動作確認済み、7,376件）
- ✅ **データ検証・品質チェック**
  - 基本検証（必須フィールド、NULL値、重複）
  - データセット固有の詳細検証
- ✅ **RAG用テキスト抽出・前処理**
  - タイトル・本文の自動結合
  - テキストクレンジング処理
  - 短いテキスト・重複の除去
- ✅ **トークン使用量推定**
- ✅ **CSV/TXT/JSONフォーマット出力**
- ✅ **HuggingFace Hub自動ダウンロード**
- ✅ **直接ダウンロード対応**（Livedoorコーパス等）

### 1.3 対応データセット

| データセット | 説明 | HuggingFace | Config | 主要フィールド | デフォルトサンプル数 |
|------------|------|------------|--------|--------------|----------|
| wikipedia_ja | Wikipedia日本語版 | wikimedia/wikipedia | 20231101.ja | title, text | 1000 |
| japanese_text | CC100日本語Webテキスト | range3/cc100-ja | - | text | 1000 |
| cc_news | CC-News英語ニュース | cc_news | - | title, text | 500 |
| livedoor | Livedoorニュースコーパス | - (直接DL) | - | url, title, content, category | 7376 |

## 2. アーキテクチャ

### 2.1 システム構成図

```
[Streamlit Webアプリ]
    ↓
[データソース選択]
    ├── CSVファイルアップロード
    └── HuggingFace Hub自動ダウンロード
    ↓
[データ検証エンジン]
    ├── 基本検証（共通）
    └── データセット固有検証
    ↓
[前処理パイプライン]
    ├── テキスト抽出
    ├── タイトル・本文結合
    ├── フィルタリング（長さ、重複）
    └── クレンジング処理
    ↓
[出力管理]
    ├── ダウンロード（CSV/TXT/JSON）
    ├── OUTPUTフォルダ保存
    └── datasets/フォルダ保存（HuggingFace）
```

### 2.2 UIレイアウト

```
┌─────────────────────────────────────────────┐
│ サイドバー              │ メインコンテンツ   │
│                        │                   │
│ ┌─────────────────┐   │ ┌──────────────┐ │
│ │📊 データセット    │   │ │タブ1:        │ │
│ │   タイプ選択      │   │ │データ        │ │
│ │- Wikipedia日本語  │   │ │アップロード   │ │
│ │- CC100日本語     │   │ └──────────────┘ │
│ │- CC-News英語     │   │                   │
│ └─────────────────┘   │ ┌──────────────┐ │
│                        │ │タブ2:        │ │
│ ┌─────────────────┐   │ │データ検証     │ │
│ │🤖 モデル選択      │   │ └──────────────┘ │
│ └─────────────────┘   │                   │
│                        │ ┌──────────────┐ │
│ ┌─────────────────┐   │ │タブ3:        │ │
│ │⚙️ データセット    │   │ │前処理実行     │ │
│ │   固有設定        │   │ └──────────────┘ │
│ └─────────────────┘   │                   │
│                        │ ┌──────────────┐ │
│                        │ │タブ4:        │ │
│                        │ │結果・        │ │
│                        │ │ダウンロード   │ │
│                        │ └──────────────┘ │
└─────────────────────────────────────────────┘
```

### 2.3 主要コンポーネント

#### 2.3.1 設定管理
- **NonQARAGConfig**: データセット設定の一元管理クラス

#### 2.3.2 データ検証
- **validate_wikipedia_data_specific**: Wikipedia固有検証（L161-187）
- **validate_news_data_specific**: ニュース固有検証（L190-223）
- **validate_scientific_data_specific**: 学術論文固有検証（L226-262）
- **validate_code_data_specific**: コード固有検証（L265-289）
- **validate_stackoverflow_data_specific**: Stack Overflow固有検証（L292-330）

#### 2.3.3 Livedoorコーパス処理
- **download_livedoor_corpus**: Livedoorコーパスをダウンロード・解凍（L337-371）
- **load_livedoor_corpus**: Livedoorコーパスを読み込み（L374-437）

#### 2.3.4 データ処理
- **extract_text_content**: テキスト抽出・結合（L444-489）

#### 2.3.5 メイン処理
- **main**: アプリケーション全体の制御（L496-1127）

## 3. データセット設定

### 3.1 NonQARAGConfigクラス

**場所**: L62-133

**目的**: 非Q&A型RAGデータセットの設定を一元管理

**クラスメソッド**:

#### get_config(dataset_type: str) -> Dict[str, Any]
データセット設定の取得

**パラメータ**:
- dataset_type: データセットタイプ ("wikipedia_ja", "japanese_text", "cc_news")

**返却値**:
```python
{
    "name": "データセット名",
    "icon": "アイコン絵文字",
    "required_columns": ["必須カラムリスト"],
    "description": "説明",
    "hf_dataset": "HuggingFaceデータセット名",
    "hf_config": "Config名 or None",
    "split": "train/test/validation",
    "streaming": True/False,
    "text_field": "テキストフィールド名",
    "title_field": "タイトルフィールド名 or None",
    "sample_size": デフォルトサンプル数
}
```

#### get_all_datasets() -> List[str]
全データセットタイプのリストを取得

**返却値**: ["wikipedia_ja", "japanese_text", "cc_news", "livedoor"]

### 3.2 データセット別設定

#### 3.2.1 Wikipedia日本語版（L66-79）
```python
{
    "name": "Wikipedia日本語版",
    "icon": "📚",
    "required_columns": ["title", "text"],
    "description": "Wikipedia日本語版の記事データ",
    "hf_dataset": "wikimedia/wikipedia",
    "hf_config": "20231101.ja",
    "split": "train",
    "streaming": True,
    "text_field": "text",
    "title_field": "title",
    "sample_size": 1000
}
```

**データセット固有設定**（L427-438）:
- remove_markup: Wikiマークアップ除去（デフォルト: True）
- min_text_length: 最小テキスト長（デフォルト: 200）

#### 3.2.2 CC100日本語（L81-94）
```python
{
    "name": "日本語Webテキスト（CC100）",
    "icon": "📰",
    "required_columns": ["text"],
    "description": "日本語Webテキストコーパス",
    "hf_dataset": "range3/cc100-ja",
    "hf_config": None,
    "split": "train",
    "streaming": True,
    "text_field": "text",
    "title_field": None,
    "sample_size": 1000
}
```

**データセット固有設定**（L440-451）:
- remove_urls: URL除去（デフォルト: True）
- min_text_length: 最小テキスト長（デフォルト: 10）

#### 3.2.3 CC-News英語ニュース（L96-109）
```python
{
    "name": "CC-News（英語ニュース）",
    "icon": "🌐",
    "required_columns": ["title", "text"],
    "description": "Common Crawl英語ニュース記事",
    "hf_dataset": "cc_news",
    "hf_config": None,
    "split": "train",
    "streaming": True,
    "text_field": "text",
    "title_field": "title",
    "sample_size": 500
}
```

**データセット固有設定**（L453-464）:
- remove_urls: URL除去（デフォルト: True）
- min_text_length: 最小テキスト長（デフォルト: 100）

#### 3.2.4 Livedoorニュースコーパス（L117-132）
```python
{
    "name": "Livedoorニュースコーパス",
    "icon": "📰",
    "required_columns": ["url", "title", "content", "category"],
    "description": "Livedoorニュース日本語記事（9カテゴリ）",
    "hf_dataset": None,  # 直接ダウンロード
    "download_url": "https://www.rondhuit.com/download/ldcc-20140209.tar.gz",
    "hf_config": None,
    "split": None,
    "streaming": False,
    "text_field": "content",
    "title_field": "title",
    "sample_size": 7376  # 全記事数
}
```

**対応カテゴリ**（9種類）:
- dokujo-tsushin（独女通信）
- it-life-hack（ITライフハック）
- kaden-channel（家電チャンネル）
- livedoor-homme（ライブドアオム）
- movie-enter（映画エンタメ）
- peachy（ピーチィ）
- smax（エスマックス）
- sports-watch（スポーツウォッチ）
- topic-news（トピックニュース）

**ファイル形式**:
```
1行目: URL
2行目: 日付
3行目: タイトル
4行目以降: 本文
```

## 4. データ検証機能

### 4.1 Wikipedia特有の検証

#### validate_wikipedia_data_specific(df: pd.DataFrame) -> List[str]
**場所**: L139-165

**検証項目**:
1. **テキスト長チェック**:
   - 平均テキスト長 < 100文字 → 警告
   - 平均テキスト長 >= 100文字 → 適切

2. **Wikiマークアップチェック**:
   - `==`, `[[`, `]]` を含む記事を検出
   - 検出率をパーセンテージで表示

3. **タイトル重複チェック**:
   - 重複タイトルの件数を警告

**返却値例**:
```python
[
    "✅ 適切なテキスト長: 平均850文字",
    "💡 Wikiマークアップ含む記事: 450件 (45.0%)",
    "⚠️ 重複タイトル: 3件"
]
```

### 4.2 ニュースデータ特有の検証

#### validate_news_data_specific(df: pd.DataFrame, dataset_type: str) -> List[str]
**場所**: L168-201

**検証項目**:
1. **記事長分析**:
   - テキストフィールド自動検出（content/body/text）
   - 平均記事長を表示

2. **短い記事検出**:
   - 100文字未満の記事を検出・警告
   - パーセンテージを表示

3. **カテゴリ情報分析**（categoryフィールドがある場合）:
   - カテゴリ総数
   - Top3カテゴリと件数

### 4.3 学術論文データ特有の検証

#### validate_scientific_data_specific(df: pd.DataFrame, dataset_type: str) -> List[str]
**場所**: L204-241

**検証項目**:
1. **要旨長分析**:
   - 平均要旨長を表示

2. **学術用語検出**:
   - 日英学術キーワード: 'research', 'study', 'method', '研究', '方法', '結果', '考察'
   - 含有率を表示

3. **PubMed特有**（dataset_type == "pubmed"）:
   - 医学用語検出: 'patient', 'treatment', 'disease', '患者', '治療', '疾患'

4. **arXiv特有**（dataset_type == "arxiv"）:
   - 本文（article）存在チェック

### 4.4 コードデータ特有の検証

#### validate_code_data_specific(df: pd.DataFrame) -> List[str]
**場所**: L243-267

**検証項目**:
1. **コード長分析**:
   - 平均コード長を表示

2. **ドキュメント存在確認**:
   - func_documentation_string フィールドの検証
   - ドキュメント付きコードの割合

3. **プログラミング言語キーワード検出**:
   - キーワード: 'def ', 'class ', 'import ', 'function', 'return'
   - 含有率を表示

### 4.5 Stack Overflow特有の検証

#### validate_stackoverflow_data_specific(df: pd.DataFrame) -> List[str]
**場所**: L270-308

**検証項目**:
1. **質問長分析**:
   - 平均質問長（bodyフィールド）

2. **タグ情報分析**:
   - タグ付き質問の割合
   - 人気タグTop5の集計と表示

3. **技術キーワード検出**:
   - キーワード: 'python', 'javascript', 'java', 'error', 'function', 'code'
   - 含有率を表示

## 5. データ処理

### 5.1 Livedoorコーパスダウンロード

#### download_livedoor_corpus(save_dir: str = "datasets") -> str
**場所**: L337-371

**目的**: Livedoorニュースコーパスのtar.gzファイルをダウンロード・解凍

**処理フロー**:
```python
1. ダウンロード先ディレクトリ作成（L346-347）
   ↓
2. tar.gzファイルをダウンロード（L354-358）
   URL: https://www.rondhuit.com/download/ldcc-20140209.tar.gz
   ↓
3. datasets/livedoor/に解凍（L360-369）
   ⚠️ セキュリティ対策: tar.extractall(filter='data')
   ↓
4. 解凍先ディレクトリパスを返却
```

**セキュリティ対策**:
- Python 3.12+で`tar.extractall()`の`filter='data'`パラメータを使用
- パストラバーサル攻撃を防止

**返却値**: 解凍先ディレクトリパス（例: "datasets/livedoor"）

### 5.2 Livedoorコーパス読み込み

#### load_livedoor_corpus(data_dir: str) -> pd.DataFrame
**場所**: L374-437

**目的**: 解凍済みのLivedoorコーパスをDataFrameに読み込み

**処理フロー**:
```python
1. カテゴリリスト定義（L384-394）
   ↓
2. 各カテゴリディレクトリを走査（L399-433）
   ├── text/{category}/*.txt を取得
   ├── LICENSE.txt, README.txt を除外
   └── 各txtファイルを解析
       ├── 1行目: URL
       ├── 2行目: 日付
       ├── 3行目: タイトル
       └── 4行目以降: 本文
   ↓
3. DataFrame変換（L434）
   ↓
4. ログ出力とDataFrame返却（L435-436）
```

**エラーハンドリング**:
- カテゴリディレクトリ不在: logger.warning（L402-403）
- ファイル読み込みエラー: logger.error（L432）

**返却値**:
```python
pd.DataFrame({
    'url': str,
    'date': str,
    'title': str,
    'content': str,
    'category': str
})
```

### 5.3 テキスト抽出

#### extract_text_content(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame
**場所**: L444-489

**目的**: データセットからテキストコンテンツを抽出し、Combined_Textカラムを作成

**処理フロー**:
```python
1. データセット設定取得（L447-449）
   ↓
2. text_field と title_field を特定
   ↓
3. フィールド存在チェック（L453-485）
   ├── タイトル + テキスト両方あり
   │   → f"{clean_text(title)} {clean_text(text)}"
   ├── テキストのみ
   │   → clean_text(text)
   └── フィールド見つからない
       ├── フォールバック候補を検索
       │   ['text', 'content', 'body', 'document', 'abstract', 'description']
       └── 候補も見つからない
           → 全カラムを結合
   ↓
4. 空テキストを除外（L487）
   ↓
5. df_processedを返却
```

**デコレーター**: @safe_execute（エラーハンドリング自動化）

### 5.4 前処理オプション

#### 短いテキストの除外（L968-976）
```python
if remove_short_text:
    before_len = len(df_processed)
    df_processed = df_processed[
        df_processed['Combined_Text'].str.len() >= min_length
    ]
    removed = before_len - len(df_processed)
    if removed > 0:
        st.info(f"📊 {removed}件の短いテキストを除外しました")
```

**パラメータ**:
- remove_short_text: bool（デフォルト: True）
- min_length: int（デフォルト: 100）

#### 重複除去（L978-984）
```python
if remove_duplicates:
    before_len = len(df_processed)
    df_processed = df_processed.drop_duplicates(subset=['Combined_Text'])
    removed = before_len - len(df_processed)
    if removed > 0:
        st.info(f"📊 {removed}件の重複テキストを除外しました")
```

**パラメータ**:
- remove_duplicates: bool（デフォルト: True）

## 6. HuggingFace統合

### 6.1 自動ダウンロード機能

**場所**: L694-855

**処理フロー**:
```python
1. ユーザー入力取得（L674-692）
   ├── dataset_name: HuggingFaceデータセット名
   ├── config_name: Config名（オプション）
   ├── split_name: Split名（デフォルト: train）
   └── sample_size: サンプル数
   ↓
2. データセット別処理（L707-800）
   ├── livedoor（Livedoorニュースコーパス）
   │   ├── download_livedoor_corpus()でダウンロード・解凍
   │   ├── load_livedoor_corpus()で読み込み
   │   └── サンプリング（オプション）
   ├── wikimedia/wikipedia
   │   → config必須（例: 20231101.ja）
   ├── range3/cc100-ja
   │   → configなし
   ├── cc_news
   │   → configオプション
   └── その他（非推奨）
   ↓
3. ストリーミングロード（HuggingFaceのみ）
   ↓
4. サンプリング（プログレスバー表示）
   ↓
5. DataFrame変換
   ↓
6. datasets/フォルダに保存（L802-813）
   ├── CSV: {dataset_name}_{split}_{size}_{timestamp}.csv
   └── JSON: {dataset_name}_{split}_{size}_{timestamp}_metadata.json
   ↓
7. メタデータ保存（L815-830）
   ↓
8. セッションステートに保存（L832-834）
```

### 6.2 ストリーミングモード

**利点**:
- メモリ効率: 全データをロードせずサンプリング可能
- 大規模データセット対応: GBサイズのデータセットでも処理可能
- プログレス表示: リアルタイムで進捗確認

**実装例（Wikipedia日本語版）**（L728-745）:
```python
# ストリーミングモードでロード
actual_dataset = "wikimedia/wikipedia"
actual_config = config_name if config_name else "20231101.ja"

dataset = hf_load_dataset(
    actual_dataset,
    actual_config,
    split=split_name,
    streaming=True  # メモリ効率的な逐次処理
)

# サンプリング
samples = []
progress_bar = st.progress(0)
for i, item in enumerate(dataset):
    if i >= sample_size:
        break
    samples.append(item)
    progress_bar.progress((i + 1) / sample_size)

df = pd.DataFrame(samples)
progress_bar.empty()
```

**Livedoor実装例**（L711-725）:
```python
# Livedoorニュースコーパスの特別処理
if selected_dataset == "livedoor":
    st.info("📥 Livedoorニュースコーパスをダウンロード中...")

    # ダウンロードと解凍
    with st.spinner("ダウンロードと解凍中..."):
        data_dir = download_livedoor_corpus("datasets")

    # データ読み込み
    with st.spinner("データを読み込み中..."):
        df = load_livedoor_corpus(data_dir)

    # サンプリング（必要に応じて）
    if sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        st.info(f"📊 {len(df)}件にサンプリングしました")
```

### 6.3 メタデータ管理

**場所**: L815-830

**保存内容**:
```json
{
  "dataset_name": "wikimedia/wikipedia",
  "dataset_type": "wikipedia_ja",
  "config": "20231101.ja",
  "split": "train",
  "sample_size": 1000,
  "actual_size": 1000,
  "downloaded_at": "2024-10-29T14:30:45.123456",
  "columns": ["id", "url", "title", "text"]
}
```

**用途**:
- データセット来歴追跡
- 再現性確保
- デバッグ支援

## 7. UI機能

### 7.1 タブ構成

#### タブ1: データアップロード（L648-880）
**機能**:
1. **CSVファイルアップロード**（L651-656）
   - file_uploader（type=['csv']）
   - アップロード後、session_stateに保存

2. **HuggingFace/直接ダウンロード自動ロード**（L659-855）
   - データセット名入力
   - Config/Split/サンプル数設定
   - Livedoor対応: 直接ダウンロード・解凍・読み込み
   - ストリーミングダウンロード（HuggingFace）
   - datasets/フォルダにCSV・メタデータJSON保存

3. **データプレビュー**（L864-880）
   - 先頭10件表示
   - カラム詳細（データ型、NULL数、ユニーク数）

#### タブ2: データ検証（L882-926）
**機能**:
1. **基本検証**（L890-892）
   - validate_data()を実行
   - 必須フィールドチェック

2. **データセット固有検証**（L895-906）
   - validate_*_data_specific()を実行
   - データセットタイプに応じた詳細検証
   - Livedoor対応: validate_news_data_specific()

3. **検証結果表示**（L910-917）
   - ⚠️: st.warning()
   - ✅: st.success()
   - 💡: st.info()

4. **テキストサンプル表示**（L920-925）
   - 先頭3件のテキストプレビュー（500文字まで）

#### タブ3: 前処理実行（L928-1029）
**機能**:
1. **前処理設定**（L937-959）
   - 短いテキスト除外設定
   - 重複除去設定

2. **前処理実行**（L962-1003）
   - extract_text_content()実行
   - フィルタリング処理
   - session_stateに保存

3. **処理済みデータプレビュー**（L1006-1010）
   - Combined_Textカラムの先頭10件

4. **トークン使用量推定**（L1013-1014）
   - estimate_token_usage()呼び出し

5. **テキスト長分布表示**（L1017-1028）
   - 平均、最小、最大、中央値をメトリクス表示

#### タブ4: 結果・ダウンロード（L1031-1123）
**機能**:
1. **処理サマリー**（L1040-1051）
   - 処理件数、除外件数、残存率をメトリクス表示

2. **ファイルダウンロード**（L1054-1099）
   - CSVファイル: 処理済みデータ全体
   - テキストファイル: Combined_Textのみ
   - メタデータ(JSON): 処理設定情報

3. **OUTPUTフォルダ保存**（L1102-1116）
   - save_files_to_output()呼び出し
   - 保存先パス表示

4. **データサンプル表示**（L1119-1123）
   - 先頭3件のテキスト（1000文字まで）

### 7.2 インタラクティブ設定

#### サイドバー設定（L514-564）
```python
with st.sidebar:
    # データセットタイプ選択
    selected_dataset = st.selectbox(
        "処理するデータセットタイプ",
        options=dataset_options,
        format_func=lambda x: dataset_labels[x],
        help="処理したいデータセットのタイプを選択してください"
    )

    # モデル選択
    selected_model = select_model()
    show_model_info(selected_model)

    # データセット固有設定
    dataset_specific_options = {}
    # （各データセットに応じた設定UI）
```

#### データセット別オプション（L551-594）

| データセット | オプション | デフォルト | 説明 |
|------------|----------|----------|------|
| wikipedia_ja | remove_markup | True | Wikiマークアップ除去 |
|  | min_text_length | 200 | 最小テキスト長 |
| japanese_text | remove_urls | True | URL除去 |
|  | min_text_length | 10 | 最小テキスト長 |
| cc_news | remove_urls | True | URL除去 |
|  | min_text_length | 100 | 最小テキスト長 |
| livedoor | （なし） | - | カテゴリ自動処理 |

## 8. 出力・保存

### 8.1 ダウンロード機能

#### CSVファイル（L1056-1083）
```python
csv_buffer = io.StringIO()
df_processed.to_csv(csv_buffer, index=False)
csv_data = csv_buffer.getvalue()

st.download_button(
    label="📄 CSVファイル",
    data=csv_data,
    file_name=f"preprocessed_{config['dataset_type']}.csv",
    mime="text/csv"
)
```

#### テキストファイル（L1062-1091）
```python
text_data = '\n'.join(df_processed['Combined_Text'].dropna().astype(str))

st.download_button(
    label="📝 テキストファイル",
    data=text_data,
    file_name=f"{config['dataset_type']}.txt",
    mime="text/plain"
)
```

#### メタデータ(JSON)（L1064-1099）
```python
metadata = {
    'dataset_type': config['dataset_type'],
    'dataset_name': dataset_config['name'],
    'processed_at': datetime.now().isoformat(),
    'row_count': len(df_processed),
    'original_count': original_count,
    'removed_count': removed,
    'config': config
}

st.download_button(
    label="📋 メタデータ(JSON)",
    data=json.dumps(metadata, ensure_ascii=False, indent=2),
    file_name=f"metadata_{config['dataset_type']}.json",
    mime="application/json"
)
```

### 8.2 OUTPUTフォルダ保存

#### save_files_to_output()の呼び出し（L1103-1116）
```python
if st.button("💾 OUTPUTフォルダに保存", type="primary"):
    saved_files = save_files_to_output(
        df_processed,
        config['dataset_type'],
        csv_data,
        text_data
    )

    if saved_files:
        st.success("✅ ファイルを保存しました：")
        for file_type, file_path in saved_files.items():
            st.write(f"• {file_path}")
```

#### 保存ファイル構造
```
OUTPUT/
├── preprocessed_wikipedia_ja.csv
├── preprocessed_japanese_text.csv
├── preprocessed_cc_news.csv
├── preprocessed_livedoor.csv
├── wikipedia_ja.txt
├── japanese_text.txt
├── cc_news.txt
└── livedoor.txt
```

## 9. helper_rag連携

### 9.1 インポート関数一覧（L41-57）

| 関数名 | 用途 | 使用箇所 |
|-------|------|---------|
| setup_page_config | ページ設定 | L503-511 |
| setup_page_header | ページヘッダー設定 | - |
| setup_sidebar_header | サイドバーヘッダー設定 | - |
| select_model | モデル選択UI | L547 |
| show_model_info | モデル情報表示 | L548 |
| validate_data | 基本データ検証 | L892 |
| load_dataset | データセットロード | - |
| estimate_token_usage | トークン使用量推定 | L1014 |
| create_download_data | ダウンロードデータ作成 | - |
| display_statistics | 統計情報表示 | - |
| save_files_to_output | OUTPUTフォルダ保存 | L1104-1108 |
| show_usage_instructions | 使用方法表示 | - |
| clean_text | テキストクレンジング | L457, L462 |
| TokenManager | トークン管理クラス | - |
| safe_execute | エラーハンドリング | L444（デコレーター） |

### 9.2 主要クラス

#### AppConfig
アプリケーション全体の設定管理

**主な属性**:
- MODEL_NAMES: 利用可能なモデル名一覧
- DEFAULT_MODEL: デフォルトモデル
- OUTPUT_DIR: 出力ディレクトリパス

#### RAGConfig
RAG固有の設定管理

**主な属性**:
- CHUNK_SIZE: チャンクサイズ
- OVERLAP_SIZE: オーバーラップサイズ
- EMBEDDING_MODEL: 埋め込みモデル

#### TokenManager
トークン数のカウント・推定

**主なメソッド**:
- count_tokens(text: str) -> int: テキストのトークン数カウント
- estimate_cost(token_count: int, model: str) -> float: コスト推定

## 10. 使用方法

### 10.1 起動方法

```bash
# 基本起動
streamlit run a01_load_non_qa_rag_data.py

# ポート指定起動（デフォルト: 8501）
streamlit run a01_load_non_qa_rag_data.py --server.port=8502

# ブラウザ自動オープンなし
streamlit run a01_load_non_qa_rag_data.py --server.headless=true
```

### 10.2 基本ワークフロー

```
1. データセットタイプ選択（サイドバー）
   ├── wikipedia_ja: Wikipedia日本語版
   ├── japanese_text: CC100日本語
   ├── cc_news: CC-News英語
   └── livedoor: Livedoorニュースコーパス
   ↓
2. モデル選択（サイドバー）
   └── GPT-4o、GPT-4o-mini等
   ↓
3. データセット固有設定（サイドバー）
   └── 各データセット特有のオプション
   ↓
4. データアップロード（タブ1）
   ├── CSVファイルアップロード または
   ├── HuggingFace Hub自動ダウンロード または
   └── 直接ダウンロード（Livedoor等）
   ↓
5. データ検証（タブ2）
   ├── 基本検証（NULL、重複等）
   └── データセット固有検証
   ↓
6. 前処理実行（タブ3）
   ├── 前処理設定（フィルタ条件）
   ├── 前処理実行ボタンクリック
   └── 結果プレビュー
   ↓
7. 結果ダウンロード（タブ4）
   ├── CSV/TXT/JSONダウンロード
   └── OUTPUTフォルダ保存
```

### 10.3 推奨設定

#### Wikipedia日本語版
```python
データセット: wikipedia_ja
HuggingFace: wikimedia/wikipedia
Config: 20231101.ja
サンプル数: 1000
オプション:
  - remove_markup: True
  - min_text_length: 200
前処理:
  - remove_short_text: True
  - min_length: 200
  - remove_duplicates: True
```

#### CC100日本語
```python
データセット: japanese_text
HuggingFace: range3/cc100-ja
Config: なし
サンプル数: 1000
オプション:
  - remove_urls: True
  - min_text_length: 10
前処理:
  - remove_short_text: True
  - min_length: 10
  - remove_duplicates: True
```

#### CC-News英語
```python
データセット: cc_news
HuggingFace: cc_news
Config: なし
サンプル数: 500
オプション:
  - remove_urls: True
  - min_text_length: 100
前処理:
  - remove_short_text: True
  - min_length: 100
  - remove_duplicates: True
```

#### Livedoorニュースコーパス
```python
データセット: livedoor
ダウンロード: 直接DL（https://www.rondhuit.com/download/ldcc-20140209.tar.gz）
サンプル数: 7376（全記事）
オプション:
  - （なし、カテゴリ自動処理）
前処理:
  - remove_short_text: True
  - min_length: 100
  - remove_duplicates: True
注意事項:
  - 9カテゴリ全記事を読み込み
  - datasets/livedoor/に保存
  - ファイル形式: URL/日付/タイトル/本文（4行構造）
```

## 11. エラーハンドリング

### 11.1 HuggingFaceエラー（L836-854）

#### スクリプトベース廃止エラー
```python
if "Dataset scripts are no longer supported" in error_msg:
    st.error("❌ このデータセットはスクリプトベースで廃止されています")
    st.info("""
    💡 **動作確認済みのデータセットをご利用ください：**
    - `wikimedia/wikipedia` (Config: 20231101.ja)
    - `range3/cc100-ja`
    """)
```

#### データセット不在エラー
```python
elif "doesn't exist on the Hub" in error_msg:
    st.error("❌ データセットが見つかりません")
    st.info("""
    💡 **データセット名を確認してください。推奨：**
    - `wikimedia/wikipedia`
    - `range3/cc100-ja`
    """)
```

#### その他のエラー
```python
else:
    st.error(f"データセットのロードに失敗しました: {error_msg}")
    st.info("💡 データセット名、config名、split名を確認してください")
```

### 11.2 データ処理エラー（L1001-1003）

#### 前処理エラー
```python
try:
    df_processed = extract_text_content(df, selected_dataset)
    # 処理続行...
except Exception as e:
    st.error(f"前処理エラー: {str(e)}")
    logger.error(f"前処理エラー: {e}")
```

#### safe_executeデコレーター
```python
@safe_execute
def extract_text_content(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    # 自動的にtry-exceptでラップされる
    # エラー時はログ出力とNone返却
    ...
```

## 12. 今後の改善案

### 12.1 機能拡張

1. **追加データセット対応**
   - ✅ Livedoorニュース（対応済み）
   - arXiv学術論文
   - PubMed医学論文
   - Stack Overflow Q&A
   - CodeSearchNet（コードデータ）

2. **高度な前処理**
   - 言語自動検出
   - 不適切コンテンツフィルタリング
   - エンティティ認識・マスキング
   - 文章品質スコアリング

3. **カスタマイズ機能**
   - ユーザー定義フィールドマッピング
   - カスタム検証ルール
   - プリセット設定の保存・読み込み

4. **バッチ処理**
   - 複数ファイル一括処理
   - 自動スケジューリング
   - 処理キューの管理

### 12.2 パフォーマンス最適化

1. **並列処理**
   - Daskによる並列データ処理
   - マルチスレッドダウンロード
   - 非同期処理の活用

2. **キャッシング**
   - @st.cache_dataでデータキャッシュ
   - ダウンロード済みデータの再利用
   - 処理結果のキャッシュ

3. **メモリ最適化**
   - チャンク単位での読み込み
   - 不要カラムの早期削除
   - メモリマップファイルの活用

4. **進捗表示改善**
   - 詳細な進捗バー（残り時間表示）
   - 処理時間推定
   - キャンセル機能の実装

---

## 変更履歴

### v1.1.0 (2025-11-12)
- ✅ Livedoorニュースコーパス対応追加（7,376記事）
- ✅ 直接ダウンロード機能追加（download_livedoor_corpus）
- ✅ Livedoorコーパス読み込み機能追加（load_livedoor_corpus）
- ✅ セキュリティ対策: tar.extractall(filter='data')
- ✅ メタデータJSON保存の改善
- ✅ 9カテゴリ対応（独女通信、ITライフハック等）
- ✅ カテゴリ別データ分析機能

### v1.0.0 (2024-10-29)
- 🎉 初回リリース
- Wikipedia日本語版対応
- CC100日本語対応
- CC-News英語対応
- HuggingFace Hub統合
- ストリーミングモード実装
- データ検証機能
- 前処理パイプライン
- CSV/TXT/JSON出力

---

**最終更新日**: 2025年11月12日
**バージョン**: 1.1.0
**作成者**: OpenAI RAG Q&A JP開発チーム