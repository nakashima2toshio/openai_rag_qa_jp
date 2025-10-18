#!/usr/bin/env python3
"""
ファイルからテキストを読み込み、先頭10センテンス分のチャンクとQ/Aペアを出力
python a03_rag_qa_coverage_file.py
"""

from helper_rag_qa import SemanticCoverage
import os
import re


def load_file(file_path: str, max_sentences: int = 10) -> str:
    """
    指定されたファイルから先頭max_sentences文を読み込む

    Args:
        file_path: 読み込むファイルのパス
        max_sentences: 読み込む最大文数

    Returns:
        読み込んだテキスト
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 日本語文の分割パターン（SemanticCoverageと同じロジック）
    sentence_pattern = r'[。．.!?！？\n]+'
    sentences = re.split(sentence_pattern, text)

    # 空文字列を除外
    sentences = [s.strip() for s in sentences if s.strip()]

    # 先頭max_sentences文を結合
    selected_sentences = sentences[:max_sentences]
    return '。'.join(selected_sentences) + '。' if selected_sentences else ''


def demonstrate_file_coverage(file_path: str, max_sentences: int = 10):
    """
    ファイルからテキストを読み込み、セマンティックカバレッジ分析を実施

    Args:
        file_path: 処理するファイルのパス
        max_sentences: 読み込む最大文数
    """

    print("=" * 80)
    print(f"ファイル: {file_path}")
    print(f"読み込み文数: 先頭{max_sentences}センテンス")
    print("=" * 80)

    # ファイル読み込み
    print("\nファイルを読み込み中...")
    document_text = load_file(file_path, max_sentences=max_sentences)

    print(f"\n読み込んだテキスト ({len(document_text)}文字):")
    print("-" * 80)
    print(document_text)
    print("-" * 80)

    # SemanticCoverageの初期化
    print("\nSemanticCoverageを初期化中...")
    analyzer = SemanticCoverage(embedding_model="text-embedding-3-small")

    # 文書をセマンティックチャンクに分割
    print("\n文書をチャンクに分割中...")
    chunks = analyzer.create_semantic_chunks(document_text, verbose=False)

    print(f"\n✅ {len(chunks)}個のチャンクを作成しました\n")

    # チャンク情報の出力
    print("=" * 80)
    print("チャンク情報")
    print("=" * 80)

    for i, chunk in enumerate(chunks, 1):
        print(f"\n【チャンク {i}】")
        print(f"  ID: {chunk['id']}")
        print(f"  文数: {len(chunk['sentences'])}")
        print(f"  文インデックス: {chunk['start_sentence_idx']} → {chunk['end_sentence_idx']}")
        print(f"  テキスト:")
        print(f"    {chunk['text'][:200]}{'...' if len(chunk['text']) > 200 else ''}")

    # Q/Aペアのサンプル生成（手動）
    print("\n\n" + "=" * 80)
    print("サンプルQ/Aペア（手動作成）")
    print("=" * 80)

    # チャンクから簡易的にQ/Aペアを作成
    qa_pairs = []
    for i, chunk in enumerate(chunks[:3], 1):  # 最初の3チャンクのみ
        # 最初の文をもとに質問を作成
        first_sentence = chunk['sentences'][0] if chunk['sentences'] else chunk['text'][:50]

        qa = {
            "question": f"チャンク{i}について説明してください",
            "answer": first_sentence,
            "chunk_id": chunk['id'],
            "type": "manual"
        }
        qa_pairs.append(qa)

    print(f"\n{len(qa_pairs)}個のQ/Aペアを作成しました:\n")

    for i, qa in enumerate(qa_pairs, 1):
        print(f"【Q/A {i}】")
        print(f"  質問: {qa['question']}")
        print(f"  回答: {qa['answer'][:100]}{'...' if len(qa['answer']) > 100 else ''}")
        print(f"  対応チャンク: {qa['chunk_id']}")
        print()

    # カバレッジ情報
    print("=" * 80)
    print("カバレッジサマリー")
    print("=" * 80)
    print(f"  総チャンク数: {len(chunks)}")
    print(f"  Q/Aペア数: {len(qa_pairs)}")
    print(f"  カバレッジ率（概算）: {len(qa_pairs) / len(chunks) * 100:.1f}%")
    print()


def main():
    """メイン実行関数"""

    # デフォルトのファイルパス
    default_file = "OUTPUT/wikipedia_ja.txt"

    # 環境チェック
    api_key = os.getenv('OPENAI_API_KEY')
    print(f"\n📋 環境チェック:")
    print(f"  OpenAI APIキー: {'✅ 設定済み' if api_key else '❌ 未設定'}")

    if not api_key:
        print("\n⚠️  OPENAI_API_KEYが設定されていません")
        print("   埋め込み生成機能は使用できません")
        print("   チャンク分割のみ実行します\n")

    # ファイルの存在確認
    if not os.path.exists(default_file):
        print(f"\n❌ エラー: ファイルが見つかりません: {default_file}")
        print("\nOUTPUTディレクトリの利用可能なファイル:")
        output_dir = "OUTPUT"
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if file.endswith('.txt'):
                    print(f"  - {os.path.join(output_dir, file)}")
        return

    # セマンティックカバレッジ分析実行
    demonstrate_file_coverage(default_file, max_sentences=10)

    print("\n" + "=" * 80)
    print("処理完了")
    print("=" * 80)


if __name__ == "__main__":
    main()