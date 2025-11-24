# a01_load_non_qa_rag_data.py - 技術仕様書

## 目次

1. [概要](#1-概要)
   - 1.1 [目的](#11-目的)
   - 1.2 [主要機能](#12-主要機能)
   - 1.3 [対応データセット](#13-対応データセット)
2. [推奨データセット（サイエンス・AI・プログラミング）](#2-推奨データセットサイエンスaiプログラミング)
   - 2.1 [arXiv論文データセット](#21-arxiv論文データセット)
   - 2.2 [Scientific Papers](#22-scientific-papers)
   - 2.3 [CodeSearchNet](#23-codesearchnet)
   - 2.4 [その他のデータセット](#24-その他のデータセット)
3. [アーキテクチャ](#3-アーキテクチャ)
   - 3.1 [システム構成図](#31-システム構成図)
   - 3.2 [UIレイアウト](#32-uiレイアウト)
   - 3.3 [主要コンポーネント](#33-主要コンポーネント)
4. [データセット設定](#4-データセット設定)
   - 4.1 [NonQARAGConfigクラス](#41-nonqaragconfigクラス)
   - 4.2 [データセット別設定](#42-データセット別設定)
5. [データ検証機能](#5-データ検証機能)
6. [データ処理](#6-データ処理)
7. [使用方法](#7-使用方法)
8. [出力・保存](#8-出力保存)
9. [エラーハンドリング](#9-エラーハンドリング)

---

## 1. 概要

### 1.1 目的

非Q&A型のRAGデータセットをHugging Faceや外部ソースからダウンロードし、前処理・検証を行うStreamlitベースのWebアプリケーション。

### 1.2 主要機能

- ✅ **多言語対応**: 日本語・英語データセットの処理
- ✅ **データ検証**: 品質チェック・必須カラム確認
- ✅ **RAG用前処理**: テキスト抽出・クレンジング
- ✅ **トークン推定**: OpenAI APIトークン使用量計算
- ✅ **複数形式出力**: CSV/TXT/JSON形式でエクスポート
- ✅ **HuggingFace統合**: 自動ダウンロード・ストリーミング対応

### 1.3 対応データセット

#### 動作確認済み（日本語）
1. **Wikipedia日本語版** 📚 - 百科事典的知識
2. **CC100日本語** 📰 - Webテキストコーパス
3. **Livedoorニュースコーパス** 📰 - 日本語ニュース（9カテゴリ、7,376件）

#### 動作確認済み（英語）
4. **CC-News** 🌐 - 英語ニュース記事

---

## 2. 推奨データセット（サイエンス・AI・プログラミング）

### 2.1 arXiv論文データセット 📄 ⭐推奨

#### 概要
```python
"arxiv": {
    "name": "arXiv論文アブストラクト",
    "icon": "📄",
    "description": "arXiv論文のアブストラクト（CS/AI/ML分野中心）",
    "hf_dataset": "arxiv_dataset",
    "hf_config": None,
    "split": "train",
    "text_field": "abstract",
    "title_field": "title",
    "sample_size": 1000,
    "min_text_length": 100,
    "categories": ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "stat.ML"]
}
```

#### 詳細情報
- **内容**: 科学論文のアブストラクト
- **分野**: Computer Science, AI, Machine Learning, NLP, Computer Vision
- **件数**: 約217万件
- **言語**: 英語
- **用途**: 技術的なRAG、学術検索、最新研究トレンド把握
- **特徴**: 高品質な論文テキスト、最新研究が随時追加

#### 推奨理由
1. AI/ML/CS分野の最新研究が豊富
2. 高品質で構造化されたデータ
3. カテゴリ別フィルタリングが可能
4. 学術的な正確性が高い

#### カテゴリ一覧
- `cs.AI`: Artificial Intelligence
- `cs.LG`: Machine Learning
- `cs.CL`: Computation and Language (NLP)
- `cs.CV`: Computer Vision
- `stat.ML`: Statistics - Machine Learning

---

### 2.2 Scientific Papers 🔬 ⭐推奨

#### 概要
```python
"scientific_papers": {
    "name": "科学論文（PubMed + arXiv）",
    "icon": "🔬",
    "description": "PubMedとarXivの論文データセット",
    "hf_dataset": "scientific_papers",
    "hf_config": "arxiv",  # または "pubmed"
    "split": "train",
    "text_field": "article",
    "title_field": "abstract",
    "sample_size": 1000,
    "min_text_length": 100
}
```

#### 詳細情報
- **内容**: 論文のフルテキストまたはアブストラクト
- **分野**: サイエンス全般（医学・物理・CS）
- **件数**: 数十万件
- **言語**: 英語
- **用途**: 学術研究支援、文献調査、科学的知識ベース構築
- **特徴**: フルテキストとアブストラクト両方利用可能

#### Config選択
- **`arxiv`**: Computer Science論文中心
- **`pubmed`**: 医学・生命科学論文中心

#### 推奨理由
1. 幅広いサイエンス分野をカバー
2. フルテキストで詳細な情報を取得可能
3. 医学・CS両方に対応

---

### 2.3 CodeSearchNet 💻 ⭐推奨

#### 概要
```python
"code_search_net": {
    "name": "CodeSearchNet（コード+ドキュメント）",
    "icon": "💻",
    "description": "GitHubのコードとドキュメントペア",
    "hf_dataset": "code_search_net",
    "hf_config": "python",  # python, java, go, php, javascript, ruby
    "split": "train",
    "text_field": "func_documentation_string",
    "title_field": "func_name",
    "sample_size": 1000,
    "min_text_length": 50
}
```

#### 詳細情報
- **内容**: 関数コード + ドキュメンテーション
- **分野**: プログラミング（6言語対応）
- **件数**: 約600万件
- **言語**: 英語（コメント・ドキュメント）
- **用途**: プログラミング支援RAG、コード検索、API学習
- **特徴**: 実用的なコードとドキュメントのペア

#### 対応プログラミング言語
- Python
- Java
- Go
- PHP
- JavaScript
- Ruby

#### 推奨理由
1. コード + 説明のペアでプログラミング学習に最適
2. 実用的なコードサンプルが豊富
3. 複数言語対応で幅広い用途

---

### 2.4 その他のデータセット

#### Papers with Code 🤖
```python
"papers_with_code": {
    "name": "Papers with Code",
    "icon": "🤖",
    "description": "AI/ML論文とコード実装",
    "hf_dataset": "neural-bridge/papers-with-code",
    "split": "train",
    "text_field": "abstract",
    "title_field": "title",
    "sample_size": 1000
}
```
- **内容**: AI/ML論文 + GitHub実装リンク
- **分野**: AI/ML専門
- **特徴**: 論文と実装コードの紐付け

#### Wikipedia英語版（科学技術）🌐
```python
"wikipedia_en_science": {
    "name": "Wikipedia英語版（科学技術）",
    "icon": "🌐",
    "hf_dataset": "wikimedia/wikipedia",
    "hf_config": "20231101.en",
    "text_field": "text",
    "title_field": "title",
    "filter_categories": ["Science", "Technology", "Computing"]
}
```
- **内容**: 百科事典記事（科学技術分野）
- **件数**: 数百万件（フィルタ前）

#### S2ORC（Semantic Scholar）📚
```python
"s2orc": {
    "name": "Semantic Scholar Open Research Corpus",
    "icon": "📚",
    "hf_dataset": "allenai/s2orc",
    "text_field": "abstract",
    "title_field": "title"
}
```
- **内容**: 学術論文のメタデータ + アブストラクト
- **分野**: 全学術分野
- **件数**: 約1億件以上

#### The Stack（GitHub）🐙
```python
"github_code": {
    "name": "The Stack（GitHubコード）",
    "icon": "🐙",
    "hf_dataset": "bigcode/the-stack",
    "hf_config": "python",
    "text_field": "content",
    "title_field": "path"
}
```
- **内容**: GitHubのソースコード
- **件数**: 数TB規模
- **注意**: 非常に大規模なデータセット

---

## 3. アーキテクチャ

### 3.1 システム構成図

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit WebUI                           │
├─────────────────────────────────────────────────────────────┤
│  Sidebar              │  Main Content Area                   │
│  ┌─────────────────┐ │  ┌─────────────────────────────────┐ │
│  │ データセット    │ │  │ Tab1: データアップロード         │ │
│  │ タイプ選択      │ │  │  - ファイルアップロード          │ │
│  │                 │ │  │  - HuggingFace自動ロード         │ │
│  ├─────────────────┤ │  ├─────────────────────────────────┤ │
│  │ モデル選択      │ │  │ Tab2: データ検証                 │ │
│  │                 │ │  │  - 基本検証                      │ │
│  ├─────────────────┤ │  │  - データセット固有検証          │ │
│  │ データセット    │ │  ├─────────────────────────────────┤ │
│  │ 固有設定        │ │  │ Tab3: 前処理実行                 │ │
│  └─────────────────┘ │  │  - テキスト抽出                  │ │
│                       │  │  - クレンジング                  │ │
│                       │  ├─────────────────────────────────┤ │
│                       │  │ Tab4: 結果・ダウンロード         │ │
│                       │  │  - 統計情報表示                  │ │
│                       │  │  - CSV/TXT/JSON出力              │ │
│                       │  └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    helper_rag.py                             │
│  - validate_data()                                           │
│  - load_dataset()                                            │
│  - estimate_token_usage()                                    │
│  - clean_text()                                              │
│  - save_files_to_output()                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               External Data Sources                          │
│  - HuggingFace Hub                                          │
│  - Rondhuit (Livedoor)                                      │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 UIレイアウト

- **サイドバー**: データセット選択、モデル選択、固有設定
- **メインエリア**: 4つのタブでワークフロー管理

### 3.3 主要コンポーネント

1. **NonQARAGConfig**: データセット設定管理
2. **データ検証関数**: データセット別の検証ロジック
3. **データ処理関数**: テキスト抽出・前処理
4. **Livedoor専用関数**: tar.gzダウンロード・解凍

---

## 4. データセット設定

### 4.1 NonQARAGConfigクラス

```python
class NonQARAGConfig:
    """非Q&A型RAGデータセットの設定"""

    DATASET_CONFIGS = {
        "dataset_key": {
            "name": "表示名",
            "icon": "アイコン",
            "required_columns": ["必須カラム"],
            "description": "説明",
            "hf_dataset": "HuggingFaceデータセット名",
            "hf_config": "Config名",
            "split": "train/test/validation",
            "streaming": True/False,
            "text_field": "テキストフィールド名",
            "title_field": "タイトルフィールド名",
            "sample_size": サンプル数
        }
    }
```

### 4.2 データセット別設定

#### Wikipedia日本語版
- **HuggingFace**: `wikimedia/wikipedia`
- **Config**: `20231101.ja`
- **フィールド**: title, text
- **サンプル数**: 1000件

#### CC100日本語
- **HuggingFace**: `range3/cc100-ja`
- **フィールド**: text
- **サンプル数**: 1000件

#### CC-News
- **HuggingFace**: `cc_news`
- **フィールド**: title, text
- **サンプル数**: 500件

#### Livedoor
- **ダウンロード**: 直接tar.gz取得
- **カテゴリ**: 9種類
- **フィールド**: url, date, title, content, category
- **サンプル数**: 7,376件（全記事）

---

## 5. データ検証機能

### 5.1 Wikipedia特有の検証
- 平均テキスト長チェック（100文字未満で警告）
- Wikiマークアップ検出（`==`, `[[`, `]]`）
- タイトル重複チェック

### 5.2 ニュースデータ特有の検証
- 記事長分析
- 短い記事検出（<100文字）
- カテゴリ分布表示（Livedoorの場合）

### 5.3 学術論文データ特有の検証
- 要旨長分析
- 学術キーワード検出（research, study, method等）
- 医学用語検出（PubMedの場合）

### 5.4 コードデータ特有の検証
- コード長分析
- ドキュメント存在確認
- プログラミング言語キーワード検出

### 5.5 Stack Overflow特有の検証
- 質問長分析
- タグ付き質問の割合
- 人気タグTop5表示
- 技術キーワード検出

---

## 6. データ処理

### 6.1 テキスト抽出（extract_text_content）

```python
def extract_text_content(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    """データセットからテキストコンテンツを抽出"""
    # タイトルとテキストを結合
    # 空のテキストを除外
    # Combined_Textカラムを作成
```

**処理フロー**:
1. title_field + text_field → Combined_Text
2. clean_text()でクレンジング
3. 空文字列の除外

### 6.2 前処理オプション

#### データセット別オプション

**Wikipedia**:
- Wikiマークアップ除去
- 最小テキスト長: 200文字

**CC100日本語**:
- URL除去
- 最小テキスト長: 10文字

**CC-News**:
- URL除去
- 最小テキスト長: 100文字

**arXiv/Scientific Papers**:
- アブストラクト長フィルタ
- 最小テキスト長: 100文字

**CodeSearchNet**:
- ドキュメント文字列抽出
- 最小テキスト長: 50文字

---

## 7. 使用方法

### 7.1 起動方法

```bash
streamlit run a01_load_non_qa_rag_data.py --server.port=8502
```

### 7.2 基本ワークフロー

1. **データセット選択** (サイドバー)
2. **データアップロード** (Tab1)
   - CSVファイルアップロード、または
   - HuggingFace自動ロード
3. **データ検証** (Tab2)
   - 基本検証結果確認
   - データセット固有検証結果確認
4. **前処理実行** (Tab3)
   - オプション設定
   - 前処理実行
   - トークン使用量推定
5. **結果ダウンロード** (Tab4)
   - CSV/TXT/JSON形式でダウンロード
   - OUTPUTフォルダに保存

### 7.3 推奨設定

#### arXiv論文の場合
- サンプル数: 500〜1000件
- 最小テキスト長: 100文字
- ストリーミング: ON

#### CodeSearchNetの場合
- サンプル数: 500〜1000件
- 最小テキスト長: 50文字
- Config: python（または希望の言語）

#### Scientific Papersの場合
- サンプル数: 500件
- Config: arxiv（CS論文）または pubmed（医学論文）

---

## 8. 出力・保存

### 8.1 出力ファイル形式

#### CSVファイル
```csv
Combined_Text,title,text,...
"タイトル テキスト内容...",タイトル,テキスト,...
```

#### TXTファイル
```
タイトル テキスト内容...
タイトル テキスト内容...
```

#### JSONメタデータ
```json
{
  "dataset_type": "arxiv",
  "dataset_name": "arXiv論文アブストラクト",
  "processed_at": "2025-11-21T14:30:22",
  "row_count": 987,
  "csv_file": "preprocessed_arxiv.csv",
  "txt_file": "arxiv.txt"
}
```

### 8.2 保存先

**OUTPUTフォルダ**:
```
OUTPUT/
├── preprocessed_{dataset_type}.csv
├── {dataset_type}.txt
└── metadata_{dataset_type}.json
```

**datasetsフォルダ（中間ファイル）**:
```
datasets/
├── {dataset_name}_{split}_{size}_{timestamp}.csv
└── {dataset_name}_{split}_{size}_{timestamp}_metadata.json
```

---

## 9. エラーハンドリング

### 9.1 HuggingFaceエラー

#### "Dataset scripts are no longer supported"
- **原因**: スクリプトベースのデータセットが廃止
- **解決**: 動作確認済みデータセットを使用

#### "doesn't exist on the Hub"
- **原因**: データセット名が間違っている
- **解決**: HuggingFace Hubでデータセット名を確認

### 9.2 データ処理エラー

#### メモリ不足
- **原因**: 大規模データセットの一括読み込み
- **解決**: サンプル数を減らす、ストリーミングモード有効化

#### フィールド不一致
- **原因**: 想定カラムが存在しない
- **解決**: データセット設定のtext_field/title_fieldを確認

---

## 10. 今後の改善案

### 10.1 機能拡張
- カテゴリフィルタリング機能（arXivカテゴリ等）
- 多言語対応の拡張
- バッチ処理機能

### 10.2 パフォーマンス最適化
- 並列処理の実装
- キャッシュ機構の強化
- ストリーミング処理の最適化

---

## 付録: データセット追加方法

新しいデータセットを追加する場合:

```python
"新データセット名": {
    "name": "表示名",
    "icon": "📄",
    "required_columns": ["必須カラム"],
    "description": "説明",
    "hf_dataset": "HuggingFaceパス",
    "hf_config": "Config名（オプション）",
    "split": "train",
    "streaming": True,
    "text_field": "テキストフィールド",
    "title_field": "タイトルフィールド",
    "sample_size": 1000
}
```

対応する検証関数も追加推奨:
```python
def validate_新データセット_specific(df: pd.DataFrame) -> List[str]:
    """新データセット特有の検証"""
    issues = []
    # 検証ロジック
    return issues
```