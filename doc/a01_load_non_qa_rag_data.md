# a01_load_non_qa_rag_data.py - 詳細設計書

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
非Q&A型RAGデータ（Wikipedia、ニュース、学術論文など）を処理し、RAG（Retrieval-Augmented Generation）用の前処理済みテキストデータを生成するStreamlitベースのWebアプリケーション。

### 1.2 主要機能
- ✅ **日本語・英語データセットの処理**
  - Wikipedia日本語版（動作確認済み）
  - CC100日本語（動作確認済み）
  - CC-News英語ニュース（動作確認済み）
- ✅ **データ検証・品質チェック**
  - 基本検証（必須フィールド、NULL値、重複）
  - データセット固有の検証
- ✅ **RAG用テキスト抽出・前処理**
  - タイトル・本文の結合
  - クレンジング処理
  - 短いテキスト・重複の除去
- ✅ **トークン使用量推定**
- ✅ **CSV/TXT/JSONフォーマット出力**
- ✅ **HuggingFace自動ダウンロード**

### 1.3 対応データセット

| データセット | 説明 | HuggingFace | Config | 主要フィールド | サンプル数 |
|------------|------|------------|--------|--------------|----------|
| wikipedia_ja | Wikipedia日本語版 | wikimedia/wikipedia | 20231101.ja | title, text | 1000 |
| japanese_text | CC100日本語 | range3/cc100-ja | - | text | 1000 |
| cc_news | CC-News英語ニュース | cc_news | - | title, text | 500 |

## 2. アーキテクチャ

### 2.1 システム構成図

```
[Streamlit UI]
    ↓
[データソース選択]
    ├── CSVアップロード
    └── HuggingFace自動ダウンロード
    ↓
[データ検証]
    ├── 基本検証
    └── データセット固有検証
    ↓
[前処理実行]
    ├── テキスト抽出
    ├── 結合処理
    ├── フィルタリング
    └── クレンジング
    ↓
[出力・保存]
    ├── CSV/TXT/JSON
    └── OUTPUTフォルダ保存
```

### 2.2 UIレイアウト

```
┌─────────────────────────────────────────────┐
│ Sidebar                  │ Main Content     │
│                          │                  │
│ ┌─────────────────┐     │ ┌──────────────┐ │
│ │データセット選択  │     │ │タブ1:        │ │
│ │- wikipedia_ja   │     │ │データ        │ │
│ │- japanese_text  │     │ │アップロード   │ │
│ │- cc_news        │     │ └──────────────┘ │
│ └─────────────────┘     │                  │
│                          │ ┌──────────────┐ │
│ ┌─────────────────┐     │ │タブ2:        │ │
│ │モデル選択        │     │ │データ検証     │ │
│ └─────────────────┘     │ └──────────────┘ │
│                          │                  │
│ ┌─────────────────┐     │ ┌──────────────┐ │
│ │データセット      │     │ │タブ3:        │ │
│ │固有設定          │     │ │前処理実行     │ │
│ └─────────────────┘     │ └──────────────┘ │
│                          │                  │
│                          │ ┌──────────────┐ │
│                          │ │タブ4:        │ │
│                          │ │結果・        │ │
│                          │ │ダウンロード   │ │
│                          │ └──────────────┘ │
└─────────────────────────────────────────────┘
```

### 2.3 主要コンポーネント

#### 2.3.1 設定管理
- **NonQARAGConfig**: データセット設定の一元管理

#### 2.3.2 データ検証
- **validate_wikipedia_data_specific**: Wikipedia固有検証
- **validate_news_data_specific**: ニュース固有検証
- **validate_scientific_data_specific**: 学術論文固有検証
- **validate_code_data_specific**: コード固有検証
- **validate_stackoverflow_data_specific**: Stack Overflow固有検証

#### 2.3.3 データ処理
- **extract_text_content**: テキスト抽出・結合

#### 2.3.4 helper_rag連携
- setup_page_config, setup_page_header, setup_sidebar_header
- select_model, show_model_info
- validate_data, load_dataset
- estimate_token_usage
- create_download_data, display_statistics
- save_files_to_output
- show_usage_instructions, clean_text
- TokenManager, safe_execute

## 3. データセット設定

### 3.1 NonQARAGConfigクラス

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
    "icon": "アイコン",
    "required_columns": ["必須カラムリスト"],
    "description": "説明",
    "hf_dataset": "HuggingFaceデータセット名",
    "hf_config": "Config名 or None",
    "split": "train/test/validation",
    "streaming": True/False,
    "text_field": "テキストフィールド名",
    "title_field": "タイトルフィールド名 or None",
    "sample_size": サンプル数
}
```

#### get_all_datasets() -> List[str]
全データセットタイプのリストを取得

**返却値**: ["wikipedia_ja", "japanese_text", "cc_news"]

### 3.2 データセット別設定

#### 3.2.1 Wikipedia日本語版
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

**データセット固有設定**:
- remove_markup: Wikiマークアップ除去（デフォルト: True）
- min_text_length: 最小テキスト長（デフォルト: 200）

#### 3.2.2 CC100日本語
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

**データセット固有設定**:
- remove_urls: URL除去（デフォルト: True）
- min_text_length: 最小テキスト長（デフォルト: 10）

#### 3.2.3 CC-News英語ニュース
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

**データセット固有設定**:
- remove_urls: URL除去（デフォルト: True）
- min_text_length: 最小テキスト長（デフォルト: 100）

## 4. データ検証機能

### 4.1 Wikipedia特有の検証

#### validate_wikipedia_data_specific(df: pd.DataFrame) -> List[str]

**検証項目**:
1. **テキスト長チェック**:
   - 平均テキスト長 < 100文字 → 警告
   - 平均テキスト長 >= 100文字 → 適切

2. **Wikiマークアップチェック**:
   - `==`, `[[`, `]]` を含む記事を検出
   - パーセンテージを表示

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

**検証項目**:
1. **記事長分析**:
   - テキストフィールド自動検出（content/body/text）
   - 平均記事長を表示

2. **短い記事検出**:
   - 100文字未満の記事を検出
   - パーセンテージを表示

3. **カテゴリ情報分析**（livedoorなど）:
   - カテゴリ数
   - Top3カテゴリと件数

**返却値例**:
```python
[
    "📊 平均記事長: 1250文字",
    "⚠️ 短い記事（<100文字）: 15件 (3.0%)",
    "📂 カテゴリ数: 9種類",
    "  - sports: 120件",
    "  - entertainment: 95件",
    "  - technology: 80件"
]
```

### 4.3 学術論文データ特有の検証

#### validate_scientific_data_specific(df: pd.DataFrame, dataset_type: str) -> List[str]

**検証項目**:
1. **要旨長分析**:
   - 平均要旨長を表示

2. **学術用語検出**:
   - キーワード: 'research', 'study', 'method', 'result', 'conclusion', '研究', '方法', '結果', '考察'
   - 含有率を表示

3. **PubMed特有**:
   - 医学用語検出: 'patient', 'treatment', 'disease', 'clinical', '患者', '治療', '疾患', '臨床'

4. **arXiv特有**:
   - 本文（article）存在チェック

**返却値例**:
```python
[
    "📄 平均要旨長: 320文字",
    "📚 学術的キーワード含む: 850件 (85.0%)",
    "🏥 医学用語含む: 720件 (72.0%)"
]
```

### 4.4 コードデータ特有の検証

#### validate_code_data_specific(df: pd.DataFrame) -> List[str]

**検証項目**:
1. **コード長分析**:
   - 平均コード長を表示

2. **ドキュメント存在確認**:
   - func_documentation_string フィールド
   - ドキュメント有無の割合

3. **コードキーワード検出**:
   - キーワード: 'def ', 'class ', 'import ', 'function', 'return'
   - 含有率を表示

**返却値例**:
```python
[
    "💻 平均コード長: 450文字",
    "📝 ドキュメントあり: 680件 (68.0%)",
    "🔧 コードキーワード含む: 920件 (92.0%)"
]
```

### 4.5 Stack Overflow特有の検証

#### validate_stackoverflow_data_specific(df: pd.DataFrame) -> List[str]

**検証項目**:
1. **質問長分析**:
   - 平均質問長（bodyフィールド）

2. **タグ情報分析**:
   - タグ付き質問の割合
   - 人気タグTop5

3. **技術キーワード検出**:
   - キーワード: 'python', 'javascript', 'java', 'error', 'function', 'code'
   - 含有率を表示

**返却値例**:
```python
[
    "❓ 平均質問長: 580文字",
    "🏷️ タグ付き: 950件 (95.0%)",
    "🔝 人気タグTop5:",
    "  - python: 320件",
    "  - javascript: 250件",
    "  - java: 180件",
    "  - error: 150件",
    "  - function: 120件",
    "💡 技術キーワード含む: 880件 (88.0%)"
]
```

## 5. データ処理

### 5.1 テキスト抽出

#### extract_text_content(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame
**目的**: データセットからテキストコンテンツを抽出し、Combined_Textカラムを作成

**処理ロジック**:
```python
1. データセット設定取得
   ↓
2. text_field と title_field を特定
   ↓
3. フィールド存在チェック
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
4. 空テキストを除外
   ↓
5. df_processedを返却
```

**デコレーター**: @safe_execute（エラーハンドリング自動化）

**例**:
```python
# Wikipedia日本語版
df = extract_text_content(df, "wikipedia_ja")
# → Combined_Text = "記事タイトル 記事本文..."

# CC100日本語（タイトルなし）
df = extract_text_content(df, "japanese_text")
# → Combined_Text = "Webテキスト本文..."
```

### 5.2 前処理オプション

#### 短いテキストの除外
```python
if remove_short_text:
    df_processed = df_processed[
        df_processed['Combined_Text'].str.len() >= min_length
    ]
```

**パラメータ**:
- remove_short_text: bool（デフォルト: True）
- min_length: int（デフォルト: 100）

#### 重複除去
```python
if remove_duplicates:
    df_processed = df_processed.drop_duplicates(subset=['Combined_Text'])
```

**パラメータ**:
- remove_duplicates: bool（デフォルト: True）

## 6. HuggingFace統合

### 6.1 自動ダウンロード機能

**処理フロー**:
```python
1. ユーザー入力
   ├── dataset_name: HuggingFaceデータセット名
   ├── config_name: Config名（オプション）
   ├── split_name: Split名（デフォルト: train）
   └── sample_size: サンプル数
   ↓
2. データセット別処理
   ├── wikimedia/wikipedia
   │   → config必須（例: 20231101.ja）
   ├── range3/cc100-ja
   │   → configなし
   ├── cc_news
   │   → configオプション
   └── その他（非推奨）
   ↓
3. ストリーミングロード
   ↓
4. サンプリング（プログレスバー表示）
   ↓
5. DataFrame変換
   ↓
6. datasets/フォルダに保存
   ├── CSV: {dataset_name}_{split}_{size}_{timestamp}.csv
   └── JSON: {dataset_name}_{split}_{size}_{timestamp}_metadata.json
   ↓
7. セッションステートに保存
```

**実装例**:
```python
# Wikipedia日本語版
dataset = hf_load_dataset(
    "wikimedia/wikipedia",
    "20231101.ja",
    split="train",
    streaming=True
)

# CC100日本語
dataset = hf_load_dataset(
    "range3/cc100-ja",
    split="train",
    streaming=True
)

# サンプリング
samples = []
for i, item in enumerate(dataset):
    if i >= sample_size:
        break
    samples.append(item)
    progress_bar.progress((i + 1) / sample_size)

df = pd.DataFrame(samples)
```

### 6.2 ストリーミングモード

**利点**:
- メモリ効率: 全データをロードせずサンプリング可能
- 大規模データセット対応: GBサイズのデータセットでも処理可能
- プログレス表示: リアルタイムで進捗確認

**設定**:
```python
DATASET_CONFIGS = {
    "wikipedia_ja": {
        ...
        "streaming": True,
        ...
    }
}
```

### 6.3 メタデータ管理

**保存内容**:
```json
{
  "dataset_name": "wikimedia/wikipedia",
  "dataset_type": "wikipedia_ja",
  "config": "20231101.ja",
  "split": "train",
  "sample_size": 1000,
  "actual_size": 1000,
  "downloaded_at": "2024-10-04T14:30:45.123456",
  "columns": ["id", "url", "title", "text"]
}
```

**用途**:
- データセット来歴追跡
- 再現性確保
- デバッグ支援

## 7. UI機能

### 7.1 タブ構成

#### タブ1: データアップロード
**機能**:
1. **CSVファイルアップロード**
   - file_uploader（type=['csv']）
   - アップロード後、session_stateに保存

2. **HuggingFace自動ロード**
   - データセット名入力
   - Config/Split/サンプル数設定
   - ロードボタン → ストリーミングダウンロード

3. **データプレビュー**
   - 先頭10件表示
   - カラム詳細（データ型、NULL数、ユニーク数）

#### タブ2: データ検証
**機能**:
1. **基本検証**
   - validate_data()を実行
   - 必須フィールドチェック

2. **データセット固有検証**
   - validate_*_data_specific()を実行
   - データセットタイプに応じた検証

3. **検証結果表示**
   - ⚠️: st.warning()
   - ✅: st.success()
   - その他: st.info()

4. **テキストサンプル表示**
   - 先頭3件のテキストプレビュー（500文字まで）

#### タブ3: 前処理実行
**機能**:
1. **前処理設定**
   - 短いテキスト除外（チェックボックス + 最小文字数入力）
   - 重複除去（チェックボックス）

2. **前処理実行ボタン**
   - extract_text_content()
   - フィルタリング処理
   - session_stateに保存

3. **処理済みデータプレビュー**
   - Combined_Textカラムの先頭10件

4. **トークン使用量推定**
   - estimate_token_usage()

5. **テキスト長分布**
   - 平均、最小、最大、中央値をメトリクス表示

#### タブ4: 結果・ダウンロード
**機能**:
1. **処理サマリー**
   - 処理件数、除外件数、残存率をメトリクス表示

2. **ファイルダウンロード**
   - CSVファイル: st.download_button()
   - テキストファイル: Combined_Textを改行区切り
   - メタデータ(JSON): 処理設定を含む

3. **OUTPUTフォルダ保存**
   - save_files_to_output()
   - preprocessed_{dataset_type}.csv
   - {dataset_type}.txt

4. **データサンプル表示**
   - 先頭3件のテキスト（1000文字まで）

### 7.2 インタラクティブ設定

#### サイドバー設定
```python
with st.sidebar:
    # データセットタイプ選択
    selected_dataset = st.selectbox(
        "処理するデータセットタイプ",
        options=dataset_options,
        format_func=lambda x: dataset_labels[x]
    )

    # モデル選択
    selected_model = select_model()
    show_model_info(selected_model)

    # データセット固有設定
    if selected_dataset == "wikipedia_ja":
        dataset_specific_options['remove_markup'] = st.checkbox(...)
        dataset_specific_options['min_text_length'] = st.number_input(...)
```

#### データセット別オプション

| データセット | オプション | デフォルト | 説明 |
|------------|----------|----------|------|
| wikipedia_ja | remove_markup | True | Wikiマークアップ除去 |
|  | min_text_length | 200 | 最小テキスト長 |
| japanese_text | remove_urls | True | URL除去 |
|  | min_text_length | 10 | 最小テキスト長 |
| cc_news | remove_urls | True | URL除去 |
|  | min_text_length | 100 | 最小テキスト長 |

## 8. 出力・保存

### 8.1 ダウンロード機能

#### CSVファイル
```python
csv_buffer = io.StringIO()
df_processed.to_csv(csv_buffer, index=False)
csv_data = csv_buffer.getvalue()

st.download_button(
    label="📄 CSVファイル",
    data=csv_data,
    file_name=f"preprocessed_{dataset_type}.csv",
    mime="text/csv"
)
```

#### テキストファイル
```python
text_data = '\n'.join(df_processed['Combined_Text'].dropna().astype(str))

st.download_button(
    label="📝 テキストファイル",
    data=text_data,
    file_name=f"{dataset_type}.txt",
    mime="text/plain"
)
```

#### メタデータ(JSON)
```python
metadata = {
    'dataset_type': dataset_type,
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
    file_name=f"metadata_{dataset_type}.json",
    mime="application/json"
)
```

### 8.2 OUTPUTフォルダ保存

#### save_files_to_output()の呼び出し
```python
if st.button("💾 OUTPUTフォルダに保存"):
    saved_files = save_files_to_output(
        df_processed,
        dataset_type,
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
├── wikipedia_ja.txt
├── japanese_text.txt
└── cc_news.txt
```

## 9. helper_rag連携

### 9.1 インポート関数一覧

| 関数名 | 用途 | 使用箇所 |
|-------|------|---------|
| setup_page_config | ページ設定 | main() |
| setup_page_header | ページヘッダー設定 | main() |
| setup_sidebar_header | サイドバーヘッダー設定 | main() |
| select_model | モデル選択UI | サイドバー |
| show_model_info | モデル情報表示 | サイドバー |
| validate_data | 基本データ検証 | タブ2 |
| load_dataset | データセットロード | タブ1 |
| estimate_token_usage | トークン使用量推定 | タブ3 |
| create_download_data | ダウンロードデータ作成 | タブ4 |
| display_statistics | 統計情報表示 | タブ4 |
| save_files_to_output | OUTPUTフォルダ保存 | タブ4 |
| show_usage_instructions | 使用方法表示 | Expander |
| clean_text | テキストクレンジング | extract_text_content |
| TokenManager | トークン管理 | 全体 |
| safe_execute | エラーハンドリング | デコレーター |

### 9.2 主要クラス

#### AppConfig
アプリケーション全体の設定管理

**主な属性**:
- MODEL_NAMES: モデル名一覧
- DEFAULT_MODEL: デフォルトモデル
- OUTPUT_DIR: 出力ディレクトリ

#### RAGConfig
RAG固有の設定管理

**主な属性**:
- CHUNK_SIZE: チャンクサイズ
- OVERLAP_SIZE: オーバーラップサイズ
- EMBEDDING_MODEL: 埋め込みモデル

#### TokenManager
トークン数のカウント・推定

**主なメソッド**:
- count_tokens(text: str) -> int
- estimate_cost(token_count: int, model: str) -> float

## 10. 使用方法

### 10.1 起動方法

```bash
# 基本起動
streamlit run a01_load_non_qa_rag_data.py

# ポート指定起動
streamlit run a01_load_non_qa_rag_data.py --server.port=8502
```

### 10.2 基本ワークフロー

```
1. データセットタイプ選択（サイドバー）
   ↓
2. モデル選択（サイドバー）
   ↓
3. データセット固有設定（サイドバー）
   ↓
4. データアップロード（タブ1）
   ├── CSVアップロード または
   └── HuggingFace自動ダウンロード
   ↓
5. データ検証（タブ2）
   ├── 基本検証
   └── データセット固有検証
   ↓
6. 前処理実行（タブ3）
   ├── 前処理設定
   ├── 前処理実行ボタン
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

## 11. エラーハンドリング

### 11.1 HuggingFaceエラー

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

### 11.2 データ処理エラー

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
   - arXiv学術論文
   - PubMed医学論文
   - Stack Overflow Q&A
   - CodeSearchNet（コードデータ）
   - Livedoorニュース

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

### 12.2 パフォーマンス最適化

1. **並列処理**
   - Daskによる並列データ処理
   - マルチスレッドダウンロード

2. **キャッシング**
   - @st.cache_dataでデータキャッシュ
   - ダウンロード済みデータの再利用

3. **メモリ最適化**
   - チャンク読み込み
   - 不要カラムの早期削除

4. **進捗表示改善**
   - 詳細な進捗バー
   - 処理時間推定
   - キャンセル機能