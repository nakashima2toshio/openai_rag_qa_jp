# PyCharm 緊急対処手順（CPU 450%超え対応）

## 🚨 今すぐ実施すべき3つの手順

### 1️⃣ ヒープメモリを16GBに増量（最重要）

```bash
# PyCharmを起動して実行:
# Help → Edit Custom VM Options...
```

以下を貼り付けて保存:
```properties
-Xmx16384m
-Xms2048m
-XX:ReservedCodeCacheSize=512m
-XX:+UseG1GC
-XX:SoftRefLRUPolicyMSPerMB=50
-XX:MaxMetaspaceSize=512m
-XX:ConcGCThreads=2
-XX:ParallelGCThreads=4
```

### 2️⃣ PyCharmを再起動

```bash
# File → Exit
# その後、PyCharmを再起動
```

### 3️⃣ キャッシュをクリア

PyCharm起動後:
```bash
# File → Invalidate Caches / Restart → Invalidate and Restart
```

---

## ✅ 実施済みの自動修正

- ✅ `.venv`, `datasets`, `OUTPUT`, `old_code`を除外設定済み
- ✅ `.idea/openai_rag_qa_jp.iml`を更新済み

---

## 📊 期待される改善

| 項目 | 現在 | 改善後 |
|------|------|--------|
| CPU使用率 | 450%+ | 100%以下 |
| ヒープメモリ | 98.6% | 60%以下 |
| 合計メモリ | 13.7GB | 8GB以下 |
| スワップ | 4GB+ | 0GB |

---

## 🔍 改善の確認方法

1. **メモリインジケーターを表示**
   - `Help → Diagnostic Tools → Show Memory Indicator`
   - 画面右下に表示される

2. **Activity Monitorで確認**
   - PyCharmのCPU使用率が100%以下になっているか確認

---

## 詳細は PYCHARM_OPTIMIZATION.md を参照
