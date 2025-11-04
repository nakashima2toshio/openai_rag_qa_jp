
-----------------------------------
a02_make_qa.py
-----------------------------------
多段階カバレージ分析結果:
- Strict  (閾値0.80): 24.5%
- Standard(閾値0.70): 70.6%
- Lenient (閾値0.60): 94.0%

チャンク特性別カバレージ:
長さ別:
- Short チャンク: 68.0%
- Medium チャンク: 81.4%
- Long チャンク: 68.7%

位置別:
- Beginning (前半): 73.5%
- Middle (中盤): 69.8%
- End (後半): 68.6%

📊 分析インサイト:
2025-11-04 07:55:26,760 - INFO -   • longチャンクのカバレージが低い（68.7%）
2025-11-04 07:55:26,760 - INFO -   • shortチャンクのカバレージが低い（68.0%）
2025-11-04 07:55:26,760 - INFO -   • 文書middle部分のカバレージが低い（69.8%）
2025-11-04 07:55:26,760 - INFO -   • 文書end部分のカバレージが低い（68.6%）

カバレージ分析結果:
- カバレージ率: 70.6%
- カバー済みチャンク: 936/1325
- 未カバーチャンク: 389

=====================================
処理完了
=====================================
生成Q/Aペア数: 4646
保存ファイル:
- Q/A (JSON): qa_output/a02/qa_pairs_cc_news_20251104_075526.json
- Q/A (CSV): qa_output/a02/qa_pairs_cc_news_20251104_075526.csv
- カバレージ: qa_output/a02/coverage_cc_news_20251104_075526.json
- サマリー: qa_output/a02/summary_cc_news_20251104_075526.json


質問タイプ別統計:
  application: 410件
  comparison: 274件
  explanation: 1件
  fact: 3157件
  reason: 804件
----------------------------------------

⏺ （1）カバレージ分析結果の解説

  📊 多段階カバレージ分析（3段階の厳しさで評価）

  | 評価レベル         | 閾値   | カバレージ率 | 意味
   |
  |---------------|------|--------|-------------------------------------------------------------------------------------|
  | Strict (厳格)   | 0.80 | 24.5%  | Q&Aとチャンクの類似度が80%以上で「カバー済み」と判定。非常に高品質なマッチングのみカウント。1325
  チャンク中わずか325チャンクしか高品質カバーされていない。 |
  | Standard (標準) | 0.70 | 70.6%  | 類似度70%以上で判定。実用的な基準。936チャンクがカバー済み、389チャンクが未カバー。
                           |
  | Lenient (寛容)  | 0.60 | 94.0%  | 類似度60%以上で判定。緩い基準で、ほぼ全チャンクがカバーされている状態。
                        |

  解釈: 現状、緩い基準（0.60）では94%達成しているが、実用的な基準（0.70）では70.6%に留まっている。

  ---
  📏 チャンク特性別カバレージ（長さによる分析）

  | チャンク長  | トークン範囲         | カバレージ率   | 課題                                 |
  |--------|----------------|----------|------------------------------------|
  | Short  | < 100 tokens   | 68.0% ⚠️ | 短いチャンクは情報が少なく、LLMが十分なQ&Aを生成できていない  |
  | Medium | 100-200 tokens | 81.4% ✅  | 最も良好。適度な情報量でQ&A生成に最適               |
  | Long   | > 200 tokens   | 68.7% ⚠️ | 長いチャンクは情報が多すぎて、一部の情報がQ&Aで取りこぼされている |

  問題点: Medium以外のチャンクでカバレージが低下している。

  ---
  📍 位置別カバレージ（文書内の位置による分析）

  | 位置             | カバレージ率   | 課題                 |
  |----------------|----------|--------------------|
  | Beginning (前半) | 73.5%    | 最も良好だが、まだ改善の余地あり   |
  | Middle (中盤)    | 69.8% ⚠️ | 文書の中間部分が見落とされがち    |
  | End (後半)       | 68.6% ⚠️ | 文書末尾の情報が最も見落とされている |

  問題点: LLMが文書の前半に注目しがちで、中盤・後半の情報がQ&Aで不足している。

  ---
  📈 全体サマリー

  - 総チャンク数: 1,325個
  - 生成Q&A数: 4,646個（チャンクあたり平均3.5個）
  - カバー済み: 936チャンク（70.6%）
  - 未カバー: 389チャンク（29.4%）

  ---
  （2）カバレージを95%以上にする改善案

  🎯 改善戦略（5つの施策）

  改善案1: チャンク長別のQ&A生成数を動的調整

  現状の問題:
  - Short/Longチャンクのカバレージが68%台と低い
  - determine_qa_count() 関数が画一的な基準

  改善内容:
  def determine_qa_count(chunk: Dict, config: Dict) -> int:
      """改善版: チャンク特性に応じた動的Q&A数決定"""
      base_count = config["qa_per_chunk"]
      tokenizer = tiktoken.get_encoding("cl100k_base")
      token_count = len(tokenizer.encode(chunk['text']))

      # チャンク位置を考慮
      chunk_position = chunk.get('chunk_idx', 0)

      if token_count < 50:
          return 2  # 現行: 1 → 改善: 2（短いチャンクでも最低2個）
      elif token_count < 100:
          qa_count = 3  # 現行: 2 → 改善: 3
      elif token_count < 200:
          qa_count = base_count + 1  # Mediumは現行維持＋1
      elif token_count < 300:
          qa_count = base_count + 2  # Longチャンクは大幅増
      else:
          qa_count = base_count + 3  # 超長文は最多

      # 文書後半は追加生成（位置バイアス補正）
      if chunk_position > 5:  # 6番目以降のチャンク
          qa_count += 1

      return min(qa_count, 8)  # 上限を8に設定

  期待効果: Short/Longチャンクのカバレージが68% → 85%に改善

  ---
  改善案2: 未カバーチャンクへの追加Q&A生成

  実装方法:
  def generate_qa_for_uncovered_chunks(
      uncovered_chunks: List[Dict],
      config: Dict,
      model: str = "gpt-5-mini"
  ) -> List[Dict]:
      """未カバーチャンクに特化したQ&A生成"""
      client = OpenAI()
      additional_qa_pairs = []

      for chunk_info in uncovered_chunks:
          chunk = chunk_info['chunk']
          gap = chunk_info['gap']  # 閾値までの不足度

          # 不足度に応じてQ&A数を決定
          if gap > 0.2:
              num_qa = 4  # 大きなギャップ
          elif gap > 0.1:
              num_qa = 3
          else:
              num_qa = 2

          # より多様な質問タイプを指定
          system_prompt = """未カバー領域を埋めるため、
          以下のチャンクから多様な観点でQ&Aを生成してください:
          - 詳細な事実確認
          - 因果関係の説明
          - 具体例の提示
          - 応用・活用方法"""

          # Q&A生成処理...

      return additional_qa_pairs

  メイン処理への統合:
  # カバレージ分析後に追加
  if coverage_results['coverage_rate'] < 0.95:
      logger.info("未カバーチャンクへの追加Q&A生成開始...")
      additional_qa = generate_qa_for_uncovered_chunks(
          coverage_results['uncovered_chunks'],
          config,
          args.model
      )
      qa_pairs.extend(additional_qa)

      # 再度カバレージ分析
      coverage_results = analyze_coverage(chunks, qa_pairs, args.dataset)

  期待効果: 70.6% → 90%に改善

  ---
  改善案3: 文書位置バイアスの補正プロンプト

  現状の問題: LLMが文書前半に偏ってQ&Aを生成

  改善プロンプト:
  # generate_qa_pairs_for_batch() 内のプロンプトに追加
  position_instruction = """
  重要: 各テキストの全体から均等にQ&Aを生成してください。
  - テキストの前半だけでなく、中盤・後半の情報も必ず含める
  - 後半に重要な情報（結論、結果、影響など）がある場合は特に重視
  """

  user_prompt = f"""以下の{len(chunks)}個のテキストから、合計{total_pairs}個のQ&Aペアを生成してください。

  {position_instruction}

  {combined_text}
  ...
  """

  期待効果: 中盤・後半のカバレージが69%→80%に改善

  ---
  改善案4: バッチサイズの最適化

  現状: --batch-chunks 5（5チャンクを一度に処理）

  問題: バッチサイズが大きすぎると、各チャンクへの注意が散漫になる

  改善案:
  # バッチサイズを3に削減（より丁寧な処理）
  python a02_make_qa.py \
        --dataset cc_news \
        --batch-chunks 3 \  # 5 → 3 に変更
        --merge-chunks \
        --min-tokens 150 \
        --max-tokens 400 \
        --model gpt-5-mini \
        --analyze-coverage

  期待効果: 各チャンクへの注意が向上し、カバレージ+3-5%

  ---
  改善案5: qa_per_chunk設定の引き上げ

  現状設定（DATASET_CONFIGS）:
  "cc_news": {
      "qa_per_chunk": 3,  # 現状
  }

  改善設定:
  "cc_news": {
      "qa_per_chunk": 5,  # 3 → 5 に増加
  }

  期待効果: 単純にQ&A数が増加し、カバレージが向上（+10-15%）

  ---
  🚀 総合改善プラン（95%達成への道筋）

  段階的アプローチ

  | 段階   | 施策                   | 期待カバレージ     | 実装難易度  |
  |------|----------------------|-------------|--------|
  | 第1段階 | 改善案5（qa_per_chunk増加） | 70.6% → 82% | ⭐（簡単）  |
  | 第2段階 | 改善案1（動的Q&A数調整）       | 82% → 88%   | ⭐⭐（中）  |
  | 第3段階 | 改善案3（位置バイアス補正）       | 88% → 92%   | ⭐⭐（中）  |
  | 第4段階 | 改善案2（未カバーチャンク追加生成）   | 92% → 95%+  | ⭐⭐⭐（高） |

  最速での95%達成方法

  # ステップ1: qa_per_chunkを5に変更して再実行
  # (DATASET_CONFIGSを編集)

  python a02_make_qa.py \
        --dataset cc_news \
        --batch-chunks 3 \
        --merge-chunks \
        --min-tokens 100 \  # 150→100に下げて小チャンクを削減
        --max-tokens 350 \  # 400→350に下げて過度な統合を防止
        --model gpt-5-mini \
        --analyze-coverage

  # ステップ2: 改善案1-4を実装して再実行

  ---
  📝 具体的な実装コード例

  コード修正1: DATASET_CONFIGSの変更

⏺ Read(a02_make_qa.py)
  ⎿ Read 30 lines (ctrl+o to expand)

⏺ すぐに実装可能な簡単な改善から提案します：

  即効性のある改善（コード修正なし）

  # 改善実行コマンド
  python a02_make_qa.py \
        --dataset cc_news \
        --batch-chunks 3 \     # 5→3: より丁寧な処理
        --merge-chunks \
        --min-tokens 100 \     # 150→100: 小チャンク削減
        --max-tokens 300 \     # 400→300: 過度な統合防止
        --model gpt-5-mini \
        --analyze-coverage

  この変更だけで 70.6% → 78-82% への改善が期待できます。

  さらに95%を目指す場合のコード修正

  cc_newsのqa_per_chunkを3→5に変更しますか？これで 82-88% まで到達できます。

＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

# カバレージ改善実装完了レポート（2025-11-04）

## ✅ 実装した改善内容

### 改善案1: `determine_qa_count()` の動的調整 (a02_make_qa.py L551-582)

**実装内容**:
- Shortチャンク（<100 tokens）: Q&A数を 2→3 に増加
- Longチャンク（>200 tokens）: base_count + 2〜3 に増加
- 文書後半（chunk_idx >= 5）: 自動的に+1個追加
- 上限を8個に設定

**コード**:
```python
if token_count < 50: qa_count = 2
elif token_count < 100: qa_count = 3
elif token_count < 200: qa_count = base_count + 1
elif token_count < 300: qa_count = base_count + 2
else: qa_count = base_count + 3

if chunk_position >= 5: qa_count += 1
return min(qa_count, 8)
```

---

### 改善案2: ベースQ&A数の引き上げ (a02_make_qa.py L130)

**実装内容**:
```python
"cc_news": {
    "qa_per_chunk": 5,  # 旧: 3 → 新: 5
}
```

**期待効果**: 全体的なQ&A生成数が約1.67倍に増加

---

### 改善案3: 位置バイアス補正（改善案1に統合済み）

**実装箇所**: `determine_qa_count()` 内 (L578-580)
- 文書の6番目以降のチャンクに自動的に+1個のQ&Aを追加
- LLMが文書前半に偏る傾向を補正

---

## 📊 期待される改善効果

### 全体カバレージ予測

| 閾値レベル | 改善前 | 改善後（予測） | 改善幅 |
|---------|--------|--------------|--------|
| Standard (0.70) | 70.6% | **90-95%** | +20-25% |
| Strict (0.80) | 24.5% | 40-50% | +15-25% |
| Lenient (0.60) | 94.0% | 98-99% | +4-5% |

### チャンク特性別の改善予測

| チャンク | 改善前 | 改善後（予測） | 改善幅 |
|---------|--------|--------------|--------|
| Short | 68.0% | **85-90%** | +17-22% |
| Medium | 81.4% | **92-95%** | +11-14% |
| Long | 68.7% | **85-90%** | +16-21% |

### 位置別の改善予測

| 位置 | 改善前 | 改善後（予測） | 改善幅 |
|------|--------|--------------|--------|
| Beginning | 73.5% | **88-92%** | +15-19% |
| Middle | 69.8% | **90-94%** | +20-24% |
| End | 68.6% | **90-94%** | +21-25% |

---

## 🚀 推奨実行コマンド

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

---

## 📋 改善のメカニズム

### 相乗効果

1. **改善案2** がベースQ&A数を3→5に増加
2. **改善案1** がチャンク特性に応じてさらに +1〜+3 調整
3. **改善案3** が文書後半を +1 補正

**結果**: 総合的なQ&A生成数が **1.8-2.2倍** に増加

### トークン別Q&A数の内訳（base=5の場合）

| トークン範囲 | Q&A数 | 計算式 |
|----------|-------|--------|
| < 50 | 2 | 固定 |
| 50-100 | 3 | 固定 |
| 100-200 | 6 | base + 1 |
| 200-300 | 7 | base + 2 |
| 300+ | 8 | base + 3 |
| 後半チャンク | +1 | 上記に加算 |

---

## ✅ 次のステップ

### 実行後の確認事項

1. **カバレージ率の確認**
   - Standard (0.70) が 90%以上達成
   - チャンク特性別がすべて 85%以上

2. **Q&A品質の確認**
   - 生成されたQ&Aの質
   - 質問タイプのバランス

### 95%に届かない場合の追加施策

- `qa_per_chunk` を 5 → 6 に引き上げ
- 未カバーチャンク専用の追加生成パス実装

---

**実装完了日**: 2025年11月4日
**実装者**: Claude Code
**期待カバレージ**: 90-95%
