# make_qa.py 詳細設計書

## 1. 概要

### 1.1 目的
`make_qa.py`は、日本語文書から自動的にQ&A（質問・回答）ペアを生成し、文書のセマンティックカバレージを評価・改善するためのツールです。

### 1.2 主要機能
- 文書の意味的チャンク分割
- OpenAI GPTモデルを使用した自動Q&Aペア生成
- セマンティックカバレージ分析
- カバレージ自動改善
- 結果の可視化

## 2. システムアーキテクチャ

### 2.1 全体アーキテクチャ図

```mermaid
graph TB
    subgraph "入力層"
        DOC[文書データ]
        ENV[環境変数/.env]
    end

    subgraph "処理層"
        SC[SemanticCoverage<br/>意味的チャンク分割]
        QG[Q&A Generator<br/>Q&Aペア生成]
        CA[Coverage Analyzer<br/>カバレージ分析]
        CI[Coverage Improver<br/>カバレージ改善]
    end

    subgraph "外部サービス"
        OPENAI[OpenAI API]
        EMBED[Embedding API<br/>text-embedding-3-small]
        GPT[GPT-5-mini<br/>Responses API]
    end

    subgraph "出力層"
        VIZ[可視化<br/>matplotlib/seaborn]
        JSON[JSONファイル出力]
        CONSOLE[コンソール出力]
    end

    DOC --> SC
    ENV --> OPENAI
    SC --> QG
    SC --> CA
    QG --> GPT
    CA --> EMBED
    CA --> CI
    CI --> GPT
    CI --> VIZ
    CI --> JSON
    CI --> CONSOLE
```

### 2.2 データフロー図

```mermaid
sequenceDiagram
    participant User
    participant Main as main()
    participant SC as SemanticCoverage
    participant QAGen as Q/A Generator
    participant OpenAI as OpenAI API
    participant Analyzer as Coverage Analyzer

    User->>Main: 文書入力
    Main->>SC: create_semantic_chunks()
    SC-->>Main: チャンクリスト

    Main->>QAGen: generate_qa_for_all_chunks()
    loop 各チャンクに対して
        QAGen->>OpenAI: GPT-5-mini API呼び出し
        OpenAI-->>QAGen: Q&Aペア(構造化出力)
    end
    QAGen-->>Main: 全Q&Aペア

    Main->>Analyzer: calculate_coverage_matrix()
    loop 埋め込み生成
        Analyzer->>OpenAI: text-embedding-3-small
        OpenAI-->>Analyzer: ベクトル
    end
    Analyzer-->>Main: カバレージマトリックス

    Main->>Analyzer: improve_coverage_with_auto_qa()
    Analyzer->>QAGen: 未カバー領域のQ&A生成
    QAGen->>OpenAI: GPT-5-mini API
    OpenAI-->>QAGen: 追加Q&Aペア
    Analyzer-->>Main: 改善結果

    Main-->>User: 結果出力
```

## 3. モジュール設計

### 3.1 モジュール構成図

```mermaid
graph LR
    subgraph "make_qa.py"
        MAIN[main<br/>メインエントリ]

        subgraph "チャンク処理"
            CHUNK[demonstrate_semantic_coverage<br/>チャンク化デモ]
            SEMANTIC[create_semantic_chunks<br/>意味的分割]
        end

        subgraph "Q&A生成"
            QAGEN[generate_qa_pairs_from_chunk<br/>チャンク単位生成]
            QAALL[generate_qa_for_all_chunks<br/>全チャンク生成]
            TOPIC[generate_topic_based_qa<br/>トピックベース生成]
        end

        subgraph "カバレージ分析"
            CALC[calculate_coverage_matrix<br/>マトリックス計算]
            IDENT[identify_uncovered_chunks<br/>未カバー特定]
            IMPROVE[improve_coverage_with_auto_qa<br/>自動改善]
        end

        subgraph "ユーティリティ"
            KW[extract_keywords<br/>キーワード抽出]
            PRIO[calculate_priority<br/>優先度計算]
            PRED[predict_coverage_improvement<br/>改善予測]
            DET[determine_qa_pairs_count<br/>Q&A数決定]
        end

        subgraph "表示・可視化"
            DISP[display_qa_pairs<br/>Q&A表示]
            VIS[visualize_semantic_coverage<br/>可視化]
            INTER[interpret_results<br/>結果解釈]
        end
    end

    subgraph "外部依存"
        RAG[rag_qa.SemanticCoverage]
        OAI[OpenAI Client]
        PYDANTIC[Pydantic Models]
    end

    MAIN --> CHUNK
    MAIN --> QAALL
    MAIN --> IMPROVE
    CHUNK --> SEMANTIC
    QAALL --> QAGEN
    IMPROVE --> TOPIC
    IMPROVE --> CALC
    CALC --> RAG
    QAGEN --> OAI
    TOPIC --> OAI
    QAGEN --> PYDANTIC
```

## 4. クラス設計

### 4.1 Pydanticモデル

```python
class QAPair(BaseModel):
    question: str        # 質問文
    answer: str         # 回答文
    question_type: str  # 質問タイプ (fact/reason/comparison/application)

class QAPairsResponse(BaseModel):
    qa_pairs: List[QAPair]  # Q&Aペアのリスト
```

### 4.2 データ構造

```mermaid
classDiagram
    class Chunk {
        +str id
        +str text
        +List~str~ sentences
        +int start_sentence_idx
        +int end_sentence_idx
    }

    class QAPairDict {
        +str question
        +str answer
        +str question_type
        +str source_chunk_id
        +bool auto_generated
    }

    class CoverageResult {
        +ndarray coverage_matrix
        +ndarray max_similarities
        +float coverage_rate
        +List uncovered_chunks
    }

    Chunk "1" --> "*" QAPairDict : generates
    Chunk "*" --> "1" CoverageResult : analyzed by
```

## 5. 主要アルゴリズム

### 5.1 セマンティックチャンク分割アルゴリズム

```mermaid
flowchart TD
    START[開始] --> SPLIT[文書を文単位に分割]
    SPLIT --> INIT[空のチャンクを初期化]
    INIT --> LOOP{全文処理済み?}
    LOOP -->|No| ADD[現在の文をチャンクに追加]
    ADD --> CHECK{トークン数 > 200?}
    CHECK -->|Yes| SAVE[チャンクを保存]
    SAVE --> NEW[新しいチャンクを開始]
    NEW --> LOOP
    CHECK -->|No| LOOP
    LOOP -->|Yes| FINAL[最後のチャンクを保存]
    FINAL --> END[終了]
```

### 5.2 Q&Aペア生成アルゴリズム

```mermaid
flowchart TD
    START[チャンク入力] --> CALC[最適Q&A数を計算]
    CALC --> PROMPT[プロンプト生成]
    PROMPT --> API[GPT-5-mini API呼び出し]
    API --> PARSE[構造化出力をパース]
    PARSE --> VALID{検証成功?}
    VALID -->|Yes| ADD_ID[チャンクIDを付与]
    ADD_ID --> RETURN[Q&Aペアを返す]
    VALID -->|No| RETRY{リトライ可能?}
    RETRY -->|Yes| API
    RETRY -->|No| ERROR[空リストを返す]
```

### 5.3 カバレージ改善アルゴリズム

```mermaid
flowchart TD
    START[開始] --> ANALYZE[現在のカバレージ分析]
    ANALYZE --> IDENTIFY[未カバーチャンク特定]
    IDENTIFY --> CHECK{未カバー存在?}
    CHECK -->|No| END[終了]
    CHECK -->|Yes| LOOP[各未カバーチャンクに対して]
    LOOP --> EXTRACT[キーワード抽出]
    EXTRACT --> GENERATE[トピックベースQ&A生成]
    GENERATE --> PREDICT[改善予測]
    PREDICT --> NEXT{次のチャンク?}
    NEXT -->|Yes| LOOP
    NEXT -->|No| MERGE[全Q&Aペア統合]
    MERGE --> RECALC[新カバレージ計算]
    RECALC --> REPORT[改善結果レポート]
    REPORT --> END
```

## 6. API仕様

### 6.1 主要関数インターフェース

| 関数名 | 入力 | 出力 | 説明 |
|--------|------|------|------|
| `create_semantic_chunks(text)` | str | List[Dict] | 文書を意味的チャンクに分割 |
| `generate_qa_pairs_from_chunk(chunk, model)` | Dict, str | List[Dict] | チャンクからQ&Aペア生成 |
| `calculate_coverage_matrix(chunks, qa_pairs, analyzer)` | List, List, obj | ndarray, ndarray | カバレージマトリックス計算 |
| `improve_coverage_with_auto_qa(chunks, qa_pairs, analyzer)` | List, List, obj | List, float, float | カバレージ自動改善 |
| `extract_keywords(text, top_n, use_mecab)` | str, int, bool | List[str] | キーワード抽出 |

### 6.2 OpenAI API使用仕様

```yaml
Embedding API:
  model: text-embedding-3-small
  endpoint: embeddings
  用途: テキストのベクトル化

Responses API:
  model: gpt-5-mini
  endpoint: responses.parse
  用途: 構造化Q&Aペア生成
  出力形式: Pydantic Model (QAPairsResponse)
```

## 7. 処理フロー

### 7.1 メイン処理フロー

```mermaid
stateDiagram-v2
    [*] --> 初期化
    初期化 --> 環境変数確認
    環境変数確認 --> API_KEY検証

    API_KEY検証 --> SemanticCoverage初期化: 成功
    API_KEY検証 --> エラー終了: 失敗

    SemanticCoverage初期化 --> 文書チャンク分割
    文書チャンク分割 --> 統計情報表示
    統計情報表示 --> キーワード抽出

    キーワード抽出 --> QA生成選択
    QA生成選択 --> QAペア生成: Yes
    QA生成選択 --> カバレージ分析選択: No

    QAペア生成 --> QA表示
    QA表示 --> 保存選択
    保存選択 --> JSON保存: Yes
    保存選択 --> カバレージ分析選択: No
    JSON保存 --> カバレージ分析選択

    カバレージ分析選択 --> カバレージ計算: Yes
    カバレージ分析選択 --> 終了: No

    カバレージ計算 --> 可視化選択
    可視化選択 --> グラフ表示: Yes
    可視化選択 --> 改善選択: No
    グラフ表示 --> 改善選択

    改善選択 --> 自動改善実行: Yes
    改善選択 --> 終了: No

    自動改善実行 --> 改善結果表示
    改善結果表示 --> 改善QA保存選択
    改善QA保存選択 --> JSON保存2: Yes
    改善QA保存選択 --> 終了: No
    JSON保存2 --> 終了

    エラー終了 --> [*]
    終了 --> [*]
```

## 8. エラーハンドリング

### 8.1 エラー処理戦略

```mermaid
graph TD
    ERROR[エラー発生] --> TYPE{エラータイプ}
    TYPE -->|API Key Missing| KEY[環境変数チェック<br/>ユーザーへ警告]
    TYPE -->|API Error| RETRY[リトライ機構<br/>指数バックオフ]
    TYPE -->|Parsing Error| FALLBACK[空リスト返却<br/>処理継続]
    TYPE -->|Network Error| TIMEOUT[タイムアウト処理<br/>エラーログ]

    KEY --> EXIT[プログラム終了]
    RETRY --> CHECK{成功?}
    CHECK -->|Yes| CONTINUE[処理継続]
    CHECK -->|No| LOG[エラーログ出力]
    FALLBACK --> CONTINUE
    TIMEOUT --> LOG
    LOG --> CONTINUE
```

### 8.2 リトライ機構

```python
最大リトライ回数: 3
バックオフ戦略: 指数バックオフ (2^attempt秒)
対象エラー:
  - OpenAI APIタイムアウト
  - レート制限エラー
  - 一時的なネットワークエラー
```

## 9. パフォーマンス最適化

### 9.1 最適化戦略

| 最適化項目 | 実装方法 | 効果 |
|------------|----------|------|
| バッチ処理 | 複数チャンクの並列処理 | API呼び出し回数削減 |
| キャッシュ | 埋め込みベクトルのキャッシュ | 再計算回避 |
| トークン数制限 | チャンクサイズ200トークン | API コスト削減 |
| 早期終了 | カバレージ閾値達成時の処理終了 | 不要な処理削減 |

### 9.2 スケーラビリティ

```mermaid
graph LR
    subgraph "小規模文書"
        S1[1-10チャンク]
        S2[10-50 Q&A]
        S3[処理時間: 1-2分]
    end

    subgraph "中規模文書"
        M1[10-50チャンク]
        M2[50-250 Q&A]
        M3[処理時間: 5-10分]
    end

    subgraph "大規模文書"
        L1[50+ チャンク]
        L2[250+ Q&A]
        L3[処理時間: 15+ 分]
        L4[要: バッチ処理]
    end

    S1 --> M1
    M1 --> L1
```

## 10. 設定パラメータ

### 10.1 設定可能パラメータ

```yaml
チャンク分割:
  max_tokens_per_chunk: 200  # チャンクあたり最大トークン数
  tokenizer: cl100k_base      # 使用するトークナイザー

Q&A生成:
  model: gpt-5-mini          # 使用するGPTモデル
  min_qa_pairs: 2            # チャンクあたり最小Q&A数
  max_qa_pairs: 5            # チャンクあたり最大Q&A数
  retry_count: 3             # APIリトライ回数

カバレージ分析:
  similarity_threshold: 0.7   # カバレージ判定閾値
  embedding_model: text-embedding-3-small  # 埋め込みモデル

可視化:
  heatmap_colormap: RdYlGn   # ヒートマップの配色
  figure_size: (12, 10)      # グラフサイズ
```

## 11. 使用例

### 11.1 基本的な使用方法

```python
# 1. 環境設定
export OPENAI_API_KEY="your-api-key"

# 2. スクリプト実行
python make_qa.py

# 3. 対話的オプション選択
# - Q&Aペア生成: y/n
# - カバレージ分析: y/n
# - 結果可視化: y/n
# - 自動改善: y/n
# - 結果保存: y/n
```

### 11.2 出力例

```
【分割結果】
総チャンク数: 3
----------------------------------------
■ チャンク 1 (ID: chunk_0)
  文の数: 2
  トークン数: 45
  内容: 人工知能（AI）は、機械学習と深層学習を基盤として...

【生成されたQ/Aペア】
Q1: トランスフォーマーモデルはどの分野で成果を上げていますか？
A1: 自然言語処理（NLP）の分野で革命的な成果を上げています。
タイプ: fact

【カバレージ改善結果】
初期カバレージ率: 66.7%
最終カバレージ率: 100.0%
改善度: +33.3%
```

## 12. 依存関係

### 12.1 外部ライブラリ

```mermaid
graph TD
    MAKEQA[make_qa.py]
    MAKEQA --> OPENAI[openai>=1.100.2]
    MAKEQA --> NUMPY[numpy]
    MAKEQA --> SKLEARN[scikit-learn]
    MAKEQA --> PYDANTIC[pydantic]
    MAKEQA --> TIKTOKEN[tiktoken]
    MAKEQA --> MATPLOTLIB[matplotlib]
    MAKEQA --> SEABORN[seaborn]
    MAKEQA --> DOTENV[python-dotenv]

    MAKEQA --> RAGQA[rag_qa.py]
    RAGQA --> OPENAI
    RAGQA --> TIKTOKEN
```

### 12.2 内部モジュール依存

- `rag_qa.py`: SemanticCoverageクラス（チャンク分割、埋め込み生成）
- `example_mecab.py`: 日本語形態素解析（オプション）

## 13. テスト戦略

### 13.1 テスト項目

| テストレベル | 対象 | 検証内容 |
|------------|------|---------|
| 単体テスト | 各関数 | 入出力の正確性 |
| 統合テスト | API連携 | OpenAI API応答処理 |
| E2Eテスト | 全体フロー | チャンク分割→Q&A生成→カバレージ分析 |
| 性能テスト | 大規模文書 | 処理時間、メモリ使用量 |

### 13.2 テストケース例

```python
# チャンク分割テスト
def test_semantic_chunks():
    text = "短い文書。これはテストです。"
    chunks = create_semantic_chunks(text)
    assert len(chunks) == 1
    assert chunks[0]['sentences'] == 2

# Q&A生成テスト
def test_qa_generation():
    chunk = {"id": "test", "text": "AIは重要です。"}
    qa_pairs = generate_qa_pairs_from_chunk(chunk)
    assert len(qa_pairs) >= 2
    assert all('question' in qa for qa in qa_pairs)

# カバレージ計算テスト
def test_coverage_calculation():
    chunks = [...]
    qa_pairs = [...]
    matrix, similarities = calculate_coverage_matrix(chunks, qa_pairs, analyzer)
    assert matrix.shape == (len(chunks), len(qa_pairs))
    assert 0 <= similarities.all() <= 1
```

## 14. 今後の拡張予定

### 14.1 機能拡張ロードマップ

```mermaid
gantt
    title 機能拡張ロードマップ
    dateFormat  YYYY-MM
    section Phase 1
    マルチ言語対応        :2024-01, 2M
    バッチ処理最適化      :2024-03, 1M
    section Phase 2
    カスタムプロンプト機能 :2024-04, 2M
    Webインターフェース   :2024-06, 3M
    section Phase 3
    機械学習による品質評価 :2024-09, 3M
    自動チューニング機能   :2024-12, 2M
```

### 14.2 改善項目

1. **パフォーマンス改善**
   - 非同期処理の導入
   - Redis等によるキャッシュ層追加
   - GPUを活用した埋め込み計算

2. **機能拡張**
   - 複数文書の一括処理
   - カスタマイズ可能なQ&Aテンプレート
   - 品質スコアリング機能

3. **ユーザビリティ向上**
   - GUI/Webインターフェース
   - 進捗バー表示
   - 詳細なエラーメッセージ

## 15. ライセンスとコントリビューション

### 15.1 ライセンス
MIT License

### 15.2 コントリビューションガイドライン
- コードスタイル: PEP 8準拠
- ドキュメント: 日本語コメント必須
- テスト: 新機能には単体テスト追加
- プルリクエスト: feature/ブランチから作成

---

*最終更新: 2025年1月*
*バージョン: 1.0.0*