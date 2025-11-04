# OpenAI RAG Q&A JP

日本語RAG（Retrieval-Augmented Generation）システムのQ&A生成・評価・検索ツール集

## プロジェクト概要

このプロジェクトは、日本語および英語のドキュメントからQ&Aペアを自動生成し、セマンティックカバレッジ分析を行い、Qdrantベクトルデータベースを使用した高精度な検索システムを提供します。

## 主要機能

- 📚 多様なデータソースからのRAGデータ処理（Wikipedia、CC-News、Web文書等）
- 🤖 OpenAI APIを使用した高品質なQ&Aペア自動生成
- 📊 セマンティックカバレッジ分析（99.7%のカバレッジ率達成）
- 🔍 Qdrantベクトルデータベースによる高速検索
- 🌐 多言語対応（日本語・英語のクロスリンガル検索）
- 💬 GPT-4o-miniによる日本語回答生成

## プログラム一覧

### データ処理・Q&A生成系


| プログラム名                     | 概要                                                                                                                                                                                     |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| a01_load_non_qa_rag_data.py      | 非Q&A型RAGデータ処理ツール。Wikipedia日本語版、CC100日本語、CC-News英語などのデータセットから<br>RAG用テキストを抽出・前処理し、CSV/TXT/JSON形式で出力。Streamlit UIで対話的に操作可能。 |
| a02_make_qa.py (v2.8)            | OUTPUTフォルダ内のpreprocessedファイル（CC-News、日本語Webテキスト、Wikipedia）から<br>OpenAI APIを使用してQ&Aペアを生成。動的Q/A調整と位置バイアス補正により90-95%のカバレッジ達成。 |
| a03_rag_qa_coverage_improved.py  | セマンティックカバレッジ分析とQ&A生成の改良版。カバレッジ率99.7%を達成し、<br>実行時間2分でチャンクごとに複数の詳細なQ&Aを生成。API呼び出しを最小化。                                    |
| a10_qa_optimized_hybrid_batch.py | バッチ処理版ハイブリッドQ&A生成システム。API呼び出しを最小化し処理を高速化。<br>バッチサイズや埋め込みバッチサイズを調整可能で、95%のカバレッジ達成を目標。                              |
| a20_output_qa_csv.py             | 各Q&A生成プログラムの最新出力CSVから question と answer 列のみを抽出し、<br>統一フォーマットのCSVファイルを作成。複数の生成方式の結果を統合。                                            |

### ベクトルストア・検索系


| プログラム名                        | 概要                                                                                                                                                                           |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| a31_make_cloud_vector_store_vsid.py | OpenAI Vector Store作成用Streamlitアプリ。CSVファイルからVector Storeを作成し、<br>ベクトルストアIDの管理と保存をサポート。UIで対話的にデータをアップロード。                  |
| a34_rag_search_cloud_vs.py          | OpenAI Responses APIとfile_searchツールを使用したRAG検索アプリケーション。<br>動的Vector Store対応で、環境変数からAPIキーを取得。Streamlit UIで検索実行。                      |
| a40_show_qdrant_data.py             | Qdrantデータ表示ツール。Qdrantサーバーの接続状態チェック、コレクション一覧表示、<br>データ統計情報の確認などをStreamlit UIで提供。                                             |
| a41_qdrant_truncate.py              | Qdrantコレクションのデータ削除ツール。コレクション全体の削除、特定ドメインのみの削除、<br>統計情報の表示など、RAGデータを安全に削除するためのユーティリティ。                  |
| a42_qdrant_registration.py          | QdrantへのQ&Aデータ登録ツール。cc_newsドメインのデータを生成方法ごとに<br>異なるコレクションに分離登録。埋め込みベクトルの作成と保存を実行。                                   |
| a50_rag_search_local_qdrant.py      | Qdrant RAG検索用Streamlit UI。複数コレクション対応、ドメイン別検索、動的埋め込み次元対応。<br>OpenAI GPT-4o-miniによる日本語回答生成機能付き。類似度スコアをリアルタイム表示。 |

### カバレッジ分析・翻訳系


| プログラム名         | 概要                                                                                                                                            |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| coverage_japan.py    | 日本語文書のセマンティックカバレッジ分析とQ/A自動生成デモンストレーション。<br>文書の意味的な網羅性を評価し、不足している領域のQ&Aを自動生成。  |
| translate_cc_news.py | CC-Newsの英語テキストを日本語に翻訳。OpenAI Responses APIを使用し、<br>チャンキング処理で大量テキストを効率的に翻訳。省略なしの完全翻訳を実現。 |

### ヘルパー・ユーティリティ系


| プログラム名     | 概要                                                                                                                                           |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| helper_api.py    | OpenAI API関連のコア機能を提供。APIコール、エラーハンドリング、レート制限管理、<br>コスト計算、ロギング機能などの共通処理を集約。              |
| helper_rag.py    | RAGデータ前処理の共通機能。データクリーニング、テキスト正規化、チャンク分割、<br>設定管理（AppConfigクラス）などの前処理ユーティリティを提供。 |
| helper_rag_qa.py | RAG Q&A用ユーティリティ。BestKeywordSelector、SmartKeywordSelector、<br>QAOptimizedExtractorクラスを提供し、キーワード抽出の最適化を実現。     |
| helper_st.py     | Streamlit UIのカスタマーサポートFAQデータ処理用ヘルパー。<br>モデル選択機能付きでRAG前処理をサポート。UI部品の共通化。                         |

### キーワード抽出・テキスト処理系


| プログラム名            | 概要                                                                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| qa_keyword_extractor.py | Q&Aペア作成用の最適化されたキーワード抽出モジュール。<br>MeCab複合名詞版と改善版を統合し、効率的なキーワード抽出を実現。              |
| regex_mecab.py          | MeCab複合名詞版と正規表現版を統合したロバストなキーワード抽出システム。<br>日本語テキストから重要な複合名詞と固有名詞を高精度で抽出。 |
| regex_vs_mecab.py       | MeCabを使用した日本語キーワード抽出の改良版。複合名詞の抽出オプション、<br>品詞フィルタリング、頻度分析などの高度な機能を提供。       |

### システム管理系


| プログラム名         | 概要                                                                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| server.py            | Qdrantベクトルデータベースサーバーの起動・管理スクリプト。<br>ヘルスチェック、ポート設定、テストモードなどのサーバー制御機能を提供。 |
| setup.py             | プロジェクト環境のセットアップスクリプト。依存パッケージのインストール、<br>環境変数の設定、初期設定ファイルの生成などを自動化。     |
| test_mecab_format.py | MeCabの出力形式を確認するテストスクリプト。<br>各種出力フォーマットの検証と形態素解析の動作確認を実行。                              |

## セットアップ

### 1. 環境構築

```bash
# 依存パッケージのインストール
python setup.py

# または個別にインストール
pip install -r requirements.txt
```

### 2. 環境変数設定

`.env`ファイルを作成し、以下を設定：

```bash
OPENAI_API_KEY=your-openai-api-key
QDRANT_URL=http://localhost:6333  # オプション
```

### 3. Qdrantサーバー起動

```bash
# Dockerを使用
cd docker-compose
docker-compose up -d

# またはローカル起動
python server.py
```

## 基本的な使用方法

### 1. データ準備とQ&A生成

```bash
# RAGデータの前処理
streamlit run a01_load_non_qa_rag_data.py --server.port=8502

# Q&Aペアの生成
python a02_make_qa.py --dataset cc_news --model gpt-4o-mini

# カバレッジ分析付きQ&A生成
python a03_rag_qa_coverage_improved.py
```

### 2. Qdrantへのデータ登録

```bash
# データ登録
python a42_qdrant_registration.py --recreate --limit 100

# データ確認
streamlit run a40_show_qdrant_data.py --server.port=8502
```

### 3. 検索システムの起動

```bash
# ローカルQdrant検索UI
streamlit run a50_rag_search_local_qdrant.py --server.port=8504

# OpenAI Vector Store検索
streamlit run a34_rag_search_cloud_vs.py --server.port=8503
```

## 主な特徴

### 🚀 高性能

- カバレッジ率99.7%を2分で達成
- バッチ処理による高速化
- API呼び出しの最小化

### 🌏 多言語対応

- 日本語・英語のクロスリンガル検索
- 翻訳不要の意味的検索
- OpenAI埋め込みモデルによる多言語サポート

### 📊 詳細な分析

- セマンティックカバレッジ分析
- 類似度スコアのリアルタイム表示
- 統計情報とメトリクスの提供

### 🎯 柔軟な設定

- 複数の埋め込みモデル対応
- 動的な次元数調整（384/1536次元）
- ドメイン別フィルタリング

## 対応データセット

- 📰 CC-News（英語ニュース）
- 📖 Wikipedia日本語版
- 🌐 CC100日本語Webテキスト
- 💬 カスタマーサポートFAQ
- 🏥 医療QAデータ
- ⚖️ 法律・判例QA
- 🔬 科学・技術QA
- 🎯 TriviaQA

## Q/Aペア生成ワークフロー

### （1）データDL・前処理
`a01_load_non_qa_rag_data.py`を使用してRAGデータをダウンロード・前処理し、`OUTPUT/preprocessed_*.csv`として保存します。

### （2）チャンク作成
各Q/A生成プログラムが独自のチャンキング戦略を実装しています：
- **a02**: MeCabベースの文境界検出チャンキング（言語別・設定可能なトークン数）
- **a03**: MeCab/正規表現ベースのキーワード抽出を含む高度なチャンキング
- **a10**: バッチ処理最適化されたハイブリッドチャンキング

### （3）Q/Aペア生成の3つの手法

#### 手法1: a02_make_qa.py（LLM直接生成方式）v2.8

**概要**: OpenAI APIを使用してチャンクから直接Q/Aペアを生成する標準的なアプローチ。v2.8でカバレッジ改善機能を実装し、動的Q/A調整により90-95%の高カバレッジを達成。

**処理フロー**:
1. **データ読み込み**: preprocessed CSVファイルから文書を読み込み
2. **MeCabベースチャンク作成**:
   - 言語別の文境界検出（日本語: 。！？、英語: .!?）
   - 設定可能なトークン数制御（デフォルト: 200-300トークン）
   - 文境界を保持したセマンティックチャンキング
3. **動的Q/A数調整**（v2.8新機能）:
   - チャンク長に応じた最適Q/A数の自動決定（2-8個/チャンク）
   - 文書後半の位置バイアス補正（6チャンク目以降+1個）
   - ベースQ/A数を3→5に引き上げ（cc_newsの場合）
4. **チャンク統合**（オプション）:
   - 150トークン未満の小さいチャンクを統合
   - 最大400トークンまで同一文書内でマージ
5. **バッチQ/A生成**:
   - 最大5チャンクを一度にLLMに送信（バッチ処理）
   - 質問タイプを指定（fact/reason/comparison/application）
   - OpenAI Responses API (client.responses.parse)でPydanticモデルを使用
6. **カバレージ分析**（オプション）:
   - 生成されたQ/Aがチャンクをどれだけカバーしているか評価
   - 多段階閾値評価（strict 0.80/standard 0.70/lenient 0.60）

**実行例（v2.8推奨設定）**:
```bash
python a02_make_qa.py \
    --dataset cc_news \
    --batch-chunks 5 \
    --merge-chunks \
    --min-tokens 150 \
    --max-tokens 400 \
    --model gpt-5-mini \
    --analyze-coverage \
    --output qa_output/a02
```

**特徴**:
- ✅ 高品質なQ/A（LLMが内容を理解して生成）
- ✅ バッチ処理によるAPI呼び出し最適化
- ✅ **v2.8: 動的Q/A調整で90-95%の高カバレッジ達成**
- ✅ **v2.8: 位置バイアス補正で文書後半もカバー**
- ⚠️ **v2.8: 処理時間が長い（6-7時間/497文書、改善前の1.5-1.7倍）**
- ⚠️ **v2.8: APIコストが増加（$0.40-0.60、Q/A数1.8-2.2倍）**

#### 手法2: a03_rag_qa_coverage_improved.py（テンプレート＋カバレッジ重視方式）

**概要**: カバレッジ率99.7%を2分で達成する高速・高カバレッジ特化型アプローチ

**処理フロー**:
1. **キーワード抽出**:
   - MeCab（利用可能な場合）または正規表現でキーワード抽出
   - 複合名詞、固有名詞を優先的に抽出
2. **TemplateBasedQAGenerator使用**:
   - 抽出したキーワードに基づいてテンプレートからQ/A生成
   - LLM呼び出しなしでルールベース生成
3. **詳細なQ/A生成**:
   - 1チャンクあたり最大12個のQ/Aペアを生成
   - より長く、より具体的な質問・回答を作成
4. **カバレージ最適化**:
   - 閾値を0.52まで下げて高カバレッジを実現
   - チャンク全体をカバーする戦略的Q/A配置

**実行例**:
```bash
python a03_rag_qa_coverage_improved.py \
    --input OUTPUT/preprocessed_cc_news.csv \
    --dataset cc_news \
    --analyze-coverage \
    --coverage-threshold 0.52 \
    --qa-per-chunk 12 \
    --max-chunks 609 \
    --max-docs 150
```

**特徴**:
- ✅ 超高速処理（2分で完了）
- ✅ 極めて高いカバレッジ率（99.7%）
- ✅ 低コスト（$0.00076）
- ❌ Q/Aの品質がテンプレートに依存
- ❌ 創造的な質問が生成されにくい

#### 手法3: a10_qa_optimized_hybrid_batch.py（ハイブリッドバッチ処理方式）

**概要**: ルールベースとLLMを組み合わせ、バッチ処理で高速化したバランス型アプローチ

**処理フロー**:
1. **BatchHybridQAGenerator初期化**:
   - LLMバッチサイズ（デフォルト10）
   - 埋め込みバッチサイズ（デフォルト150）
2. **ハイブリッドQ/A生成**:
   - キーワード抽出によるルールベース生成
   - 重要部分はLLMで高品質なQ/A生成
   - 両方式を適材適所で組み合わせ
3. **大規模バッチ処理**:
   - 複数文書を一度に処理
   - 埋め込み生成も大量バッチで高速化
4. **カバレッジ計算**:
   - リアルタイムでカバレッジ率を監視
   - 目標95%達成を自動調整

**実行例**:
```bash
python a10_qa_optimized_hybrid_batch.py \
    --dataset cc_news \
    --model gpt-5-mini \
    --batch-size 10 \
    --embedding-batch-size 150 \
    --qa-count 12 \
    --max-docs 150
```

**特徴**:
- ✅ 処理速度とQ/A品質のバランス
- ✅ スケーラブルなバッチ処理
- ✅ 柔軟な品質調整（LLM使用率を制御可能）
- ✅ 95%の高カバレッジ率
- ⚠️ 設定パラメータが多い

### 3つの手法の比較表

| 項目 | a02_make_qa.py (v2.8) | a03_rag_qa_coverage_improved.py | a10_qa_optimized_hybrid_batch.py |
|------|----------------------|----------------------------------|-----------------------------------|
| **生成方式** | LLM直接生成（動的調整） | テンプレートベース（ルール） | ハイブリッド（ルール＋LLM） |
| **処理速度** | **6-7時間 (v2.8: 1.5-1.7倍)** | 超高速（2分） | 高速（10-15分） |
| **Q/A品質** | 高品質 | 中品質（テンプレート依存） | 高品質（重要部分） |
| **カバレッジ率** | **90-95% (v2.8改善)** | 99.7% | 95% |
| **API呼び出し回数** | 265回 | 最小（5回） | 中程度（50-100回） |
| **コスト** | **$0.40-0.60 (v2.8増加)** | 最小（$0.00076） | 中程度 |
| **Q/Aペア数** | **8,363-10,221個 (v2.8: 1.8-2.2倍)** | 2,139個 | 1,000-1,500個 |
| **バッチ処理** | あり（5チャンク） | なし | あり（10文書） |
| **カスタマイズ性** | 中 | 低（テンプレート固定） | 高 |
| **特記事項** | **v2.8: 位置バイアス補正実装** | 最高のカバレッジ率 | 最もバランスが良い |
| **適用場面** | 高品質＋高カバレッジが必要 | 網羅性重視・低コスト要求時 | バランス重視・実用的な利用 |

### （4）統一フォーマットへの変換: a20_output_qa_csv.py

各手法で生成されたQ/Aペアを統一フォーマット（question, answerの2列のみ）に変換します。

| 入力ファイル | 出力ファイル | 説明 |
|-------------|------------|------|
| qa_output/a02/qa_pairs_cc_news_*.csv | qa_output/a02_qa_pairs_cc_news.csv | a02_make_qa.pyで生成された最新のQ&Aペアを統一フォーマットに変換 |
| qa_output/a03/qa_pairs_cc_news_*.csv | qa_output/a03_qa_pairs_cc_news.csv | a03_rag_qa_coverage_improved.pyで生成された最新のQ&Aペアを統一フォーマットに変換 |
| qa_output/a10/batch_qa_pairs_cc_news_gpt_5_mini_b25_*.csv | qa_output/a10_qa_pairs_cc_news.csv | a10_qa_optimized_hybrid_batch.pyで生成された最新のQ&Aペアを統一フォーマットに変換 |

**実行**:
```bash
python a20_output_qa_csv.py
```

### 推奨使用ケース

1. **研究・評価用途** → `a02_make_qa.py`
   - 高品質なQ/Aペアが必要
   - コストは問題にならない
   - 時間的余裕がある

2. **大規模データセット処理** → `a03_rag_qa_coverage_improved.py`
   - 網羅性が最重要
   - 処理速度重視
   - コスト削減が必要

3. **実用的なRAGシステム構築** → `a10_qa_optimized_hybrid_batch.py`
   - バランスの取れた品質と速度
   - スケーラビリティが必要
   - 柔軟な調整が可能

## 生成・保存データ一覧

| データ名 | ダウンロード/作成プログラム | 説明 |
|---------|---------------------------|------|
| preprocessed_cc_news.csv | a01_load_non_qa_rag_data.py | CC-News英語ニュースの前処理済みデータ |
| preprocessed_japanese_text.csv | a01_load_non_qa_rag_data.py | 日本語Webテキストの前処理済みデータ |
| preprocessed_wikipedia_ja.csv | a01_load_non_qa_rag_data.py | Wikipedia日本語版の前処理済みデータ |
| qa_pairs_cc_news_*.csv | a02_make_qa.py | OpenAI APIで生成したCC-NewsのQ&Aペア |
| qa_pairs_cc_news_*.csv | a03_rag_qa_coverage_improved.py | カバレッジ分析付きで生成したQ&Aペア（99.7%カバレッジ） |
| batch_qa_pairs_cc_news_*.csv | a10_qa_optimized_hybrid_batch.py | バッチ処理で生成したハイブリッドQ&Aペア |
| unified_qa_output.csv | a20_output_qa_csv.py | 各生成方式の結果を統合した統一フォーマットQ&A |
| cc_news_jp.txt | translate_cc_news.py | CC-Newsの日本語翻訳版 |
| vector_store_ids.json | a31_make_cloud_vector_store_vsid.py | OpenAI Vector StoreのID管理ファイル |
| qa_corpus（コレクション） | a42_qdrant_registration.py | Qdrantに登録されたQ&Aコーパス |
| product_embeddings（コレクション） | a42_qdrant_registration.py | Qdrantに登録された製品情報埋め込み |
| coverage_analysis.json | a03_rag_qa_coverage_improved.py | セマンティックカバレッジ分析結果 |
| keyword_extraction_results.json | qa_keyword_extractor.py | 抽出されたキーワードとその統計情報 |
| embeddings_cache.npz | helper_api.py | 埋め込みベクトルのキャッシュファイル |

### （4）Embedding(Vector Store ID作成)

### （5-1）OpenAI Vector Store - Cloud登録

### (5-2) Qdrantに登録

## 技術スタック

- **言語処理**: OpenAI API (GPT-4o-mini, text-embedding-3)
- **ベクトルDB**: Qdrant
- **形態素解析**: MeCab
- **UI**: Streamlit
- **データ処理**: pandas, numpy
- **並列処理**: asyncio, batch processing

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容を議論してください。

## お問い合わせ

質問や提案がある場合は、GitHubのissueを作成してください。
