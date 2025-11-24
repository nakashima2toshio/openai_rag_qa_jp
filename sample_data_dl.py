"""
Hugging Face RAG評価データセットからQ/Aペアデータを作成

入力: data/RAG-Evaluation-Dataset-JA.csv
出力:
  - data/qa_pairs_simple.csv (question, answerのみ)
  - data/qa_pairs_full.csv (メタデータ付き)
  - data/qa_pairs.jsonl (JSON Lines形式)
"""

import pandas as pd
import json
from pathlib import Path


def clean_text(text: str) -> str:
    """テキストのクリーニング処理"""
    if pd.isna(text):
        return ""

    # 文字列に変換
    text = str(text)

    # 前後の空白削除
    text = text.strip()

    # 改行コードの正規化（LFに統一）
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    return text


def generate_qa_id(domain: str, index: int) -> str:
    """Q/A IDの生成"""
    return f"{domain}_{index:03d}"


def create_qa_pairs(input_csv: str, output_dir: str = "data"):
    """
    Q/Aペアデータを作成

    Args:
        input_csv: 入力CSVファイルパス
        output_dir: 出力ディレクトリ
    """

    print("=" * 80)
    print("Q/Aペアデータ作成開始")
    print("=" * 80)

    # データ読み込み
    print(f"\n📂 読み込み中: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"  - 総行数: {len(df)}")
    print(f"  - 総列数: {len(df.columns)}")

    # 必要なカラムのみ抽出
    required_cols = ['question', 'target_answer', 'domain', 'type', 'target_file_name', 'target_page_no']
    df_qa = df[required_cols].copy()

    # データクリーニング
    print("\n🧹 データクリーニング中...")
    df_qa['question'] = df_qa['question'].apply(clean_text)
    df_qa['answer'] = df_qa['target_answer'].apply(clean_text)

    # カラム名変更
    df_qa = df_qa.rename(columns={
        'target_file_name': 'source_file',
        'target_page_no': 'page_no'
    })

    # qa_idの生成
    print("\n🔢 QA IDを生成中...")
    domain_counters = {}
    qa_ids = []

    for idx, row in df_qa.iterrows():
        domain = row['domain']
        if domain not in domain_counters:
            domain_counters[domain] = 1
        else:
            domain_counters[domain] += 1

        qa_id = generate_qa_id(domain, domain_counters[domain])
        qa_ids.append(qa_id)

    df_qa.insert(0, 'qa_id', qa_ids)

    # target_answerカラムを削除
    df_qa = df_qa.drop(columns=['target_answer'])

    # データ統計
    print("\n📊 データ統計:")
    print(f"  - 総Q/Aペア数: {len(df_qa)}")
    print(f"  - ドメイン別:")
    for domain, count in df_qa['domain'].value_counts().items():
        print(f"    - {domain}: {count}件")
    print(f"  - タイプ別:")
    for qtype, count in df_qa['type'].value_counts().items():
        print(f"    - {qtype}: {count}件")

    # 文字数統計
    df_qa['question_len'] = df_qa['question'].str.len()
    df_qa['answer_len'] = df_qa['answer'].str.len()

    print(f"\n📏 文字数統計:")
    print(f"  - 質問文: 平均 {df_qa['question_len'].mean():.1f}文字 (最小{df_qa['question_len'].min()}, 最大{df_qa['question_len'].max()})")
    print(f"  - 回答文: 平均 {df_qa['answer_len'].mean():.1f}文字 (最小{df_qa['answer_len'].min()}, 最大{df_qa['answer_len'].max()})")

    # 統計カラムは出力前に削除
    df_qa = df_qa.drop(columns=['question_len', 'answer_len'])

    # 出力ディレクトリの作成
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 1. シンプル版CSV（question, answerのみ）
    print("\n💾 シンプル版CSV作成中...")
    simple_csv = output_path / "qa_pairs_simple.csv"
    df_simple = df_qa[['question', 'answer']].copy()
    df_simple.to_csv(simple_csv, index=False, encoding='utf-8')
    print(f"  ✅ 保存完了: {simple_csv}")
    print(f"     - 行数: {len(df_simple)}, カラム数: {len(df_simple.columns)}")

    # 2. メタデータ付きCSV
    print("\n💾 メタデータ付きCSV作成中...")
    full_csv = output_path / "qa_pairs_full.csv"
    df_full = df_qa[['qa_id', 'question', 'answer', 'domain', 'type', 'source_file', 'page_no']].copy()
    df_full.to_csv(full_csv, index=False, encoding='utf-8')
    print(f"  ✅ 保存完了: {full_csv}")
    print(f"     - 行数: {len(df_full)}, カラム数: {len(df_full.columns)}")

    # 3. JSON Lines形式
    print("\n💾 JSON Lines形式作成中...")
    jsonl_file = output_path / "qa_pairs.jsonl"
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for _, row in df_full.iterrows():
            json_obj = {
                'qa_id': row['qa_id'],
                'question': row['question'],
                'answer': row['answer'],
                'domain': row['domain'],
                'type': row['type'],
                'source_file': row['source_file'],
                'page_no': int(row['page_no']) if pd.notna(row['page_no']) else None
            }
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

    print(f"  ✅ 保存完了: {jsonl_file}")
    print(f"     - 行数: {len(df_full)}")

    # 4. ドメイン別CSV（オプション）
    print("\n💾 ドメイン別CSV作成中...")
    domain_dir = output_path / "qa_pairs_by_domain"
    domain_dir.mkdir(exist_ok=True)

    for domain in df_qa['domain'].unique():
        df_domain = df_qa[df_qa['domain'] == domain][['qa_id', 'question', 'answer', 'type', 'source_file', 'page_no']].copy()
        domain_csv = domain_dir / f"qa_pairs_{domain}.csv"
        df_domain.to_csv(domain_csv, index=False, encoding='utf-8')
        print(f"  ✅ {domain}: {len(df_domain)}件 → {domain_csv.name}")

    # サンプルデータ表示
    print("\n" + "=" * 80)
    print("サンプルデータ（最初の2件）")
    print("=" * 80)
    for i in range(min(2, len(df_full))):
        row = df_full.iloc[i]
        print(f"\n[{i+1}] QA ID: {row['qa_id']}")
        print(f"    Domain: {row['domain']} | Type: {row['type']}")
        print(f"    Question: {row['question'][:80]}...")
        print(f"    Answer: {row['answer'][:80]}...")

    print("\n" + "=" * 80)
    print("✨ 処理完了！")
    print("=" * 80)
    print(f"\n📁 出力ファイル:")
    print(f"  1. {simple_csv}")
    print(f"  2. {full_csv}")
    print(f"  3. {jsonl_file}")
    print(f"  4. {domain_dir}/*.csv (5ファイル)")


if __name__ == "__main__":
    # 入力ファイルパス
    input_csv = "data/RAG-Evaluation-Dataset-JA.csv"

    # Q/Aペア作成実行
    create_qa_pairs(input_csv)