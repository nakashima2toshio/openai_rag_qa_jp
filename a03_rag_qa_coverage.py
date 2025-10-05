# RAG for QA
import numpy as np
from typing import List, Dict, Tuple
from openai import OpenAI
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
import re

# -----------------------------------
# セマンティックカバレージクラスの完全実装
# -----------------------------------
class SemanticCoverage:
    """意味的な網羅性を測定するクラス"""

    def __init__(self, embedding_model="text-embedding-3-small"):
        self.embedding_model = embedding_model
        self.client = OpenAI()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def create_semantic_chunks(self, document: str, verbose: bool = True) -> List[Dict]:
        """
        文書を意味的に区切られたチャンクに分割

        重要ポイント：
        1. 文の境界で分割（意味の断絶を防ぐ）
        2. トピックの変化を検出
        3. 適切なサイズを維持（埋め込みモデルの制限内）
        """

        # Step 1: 文単位で分割
        sentences = self._split_into_sentences(document)
        if verbose:
            print(f"文の数: {len(sentences)}")

        # Step 2: 意味的に関連する文をグループ化
        chunks = []
        current_chunk = []
        current_tokens = 0
        max_tokens = 200  # チャンクの最大トークン数

        for i, sentence in enumerate(sentences):
            sentence_tokens = len(self.tokenizer.encode(sentence))

            # 現在のチャンクにこの文を追加すべきか判断
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                # チャンクを保存
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "id"                : f"chunk_{len(chunks)}",
                    "text"              : chunk_text,
                    "sentences"         : current_chunk.copy(),
                    "start_sentence_idx": i - len(current_chunk),
                    "end_sentence_idx"  : i - 1
                })
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # 最後のチャンクを追加
        if current_chunk:
            chunks.append({
                "id"                : f"chunk_{len(chunks)}",
                "text"              : " ".join(current_chunk),
                "sentences"         : current_chunk,
                "start_sentence_idx": len(sentences) - len(current_chunk),
                "end_sentence_idx"  : len(sentences) - 1
            })

        # Step 3: トピックの連続性を考慮した再調整
        chunks = self._adjust_chunks_for_topic_continuity(chunks)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """文単位で分割（日本語対応）"""
        # 日本語と英語の文末記号で分割
        sentences = re.split(r'(?<=[。．.!?])\s*', text)
        # 空文字列を除去
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _adjust_chunks_for_topic_continuity(self, chunks: List[Dict]) -> List[Dict]:
        """トピックの連続性を考慮してチャンクを調整"""

        adjusted_chunks = []
        for i, chunk in enumerate(chunks):
            # 隣接チャンクとの意味的類似度を計算
            if i > 0 and len(chunk["sentences"]) < 2:
                # 短すぎるチャンクは前のチャンクとマージを検討
                prev_chunk = adjusted_chunks[-1]
                combined_text = prev_chunk["text"] + " " + chunk["text"]

                if len(self.tokenizer.encode(combined_text)) < 300:
                    # マージ
                    prev_chunk["text"] = combined_text
                    prev_chunk["sentences"].extend(chunk["sentences"])
                    prev_chunk["end_sentence_idx"] = chunk["end_sentence_idx"]
                    continue

            adjusted_chunks.append(chunk)

        return adjusted_chunks

    def generate_embeddings(self, doc_chunks: List[Dict]) -> np.ndarray:
        """
        チャンクのリストから埋め込みベクトルを生成

        重要ポイント：
        1. バッチ処理で効率化
        2. エラーハンドリング
        3. 正規化（コサイン類似度計算の準備）
        """

        embeddings = []
        batch_size = 20  # OpenAI APIの制限を考慮

        for i in range(0, len(doc_chunks), batch_size):
            batch = doc_chunks[i:i + batch_size]
            texts = [chunk["text"] for chunk in batch]

            try:
                # OpenAI APIを呼び出し
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=texts
                )

                # 埋め込みベクトルを取得
                for embedding_data in response.data:
                    embedding = np.array(embedding_data.embedding)
                    # L2正規化（コサイン類似度の計算を高速化）
                    embedding = embedding / np.linalg.norm(embedding)
                    embeddings.append(embedding)

            except Exception as e:
                print(f"埋め込み生成エラー: {e}")
                # エラー時はゼロベクトルを追加
                for _ in batch:
                    embeddings.append(np.zeros(1536))  # embedding dimension

        return np.array(embeddings)

    def generate_embedding(self, text: str) -> np.ndarray:
        """単一テキストの埋め込み生成"""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            embedding = np.array(response.data[0].embedding)
            # 正規化
            return embedding / np.linalg.norm(embedding)
        except Exception as e:
            print(f"埋め込み生成エラー: {e}")
            return np.zeros(1536)

    def cosine_similarity(self, doc_emb: np.ndarray, qa_emb: np.ndarray) -> float:
        """
        コサイン類似度を計算

        重要ポイント：
        1. 事前に正規化済みなら内積で計算可能
        2. 範囲は[-1, 1]、1に近いほど類似
        """

        # ベクトルが正規化済みの場合は内積で計算
        if np.allclose(np.linalg.norm(doc_emb), 1.0) and \
                np.allclose(np.linalg.norm(qa_emb), 1.0):
            return float(np.dot(doc_emb, qa_emb))

        # 正規化されていない場合は完全な計算
        dot_product = np.dot(doc_emb, qa_emb)
        norm_doc = np.linalg.norm(doc_emb)
        norm_qa = np.linalg.norm(qa_emb)

        if norm_doc == 0 or norm_qa == 0:
            return 0.0

        return float(dot_product / (norm_doc * norm_qa))


# ---------------------
#
# ---------------------
class QAGenerationConsiderations:
    """Q/A生成前のチェックリスト"""

    def analyze_document_characteristics(self, document):
        """文書の特性分析"""
        return {
            "document_type   ": self.detect_document_type(document),  # 技術文書、物語、レポート等
            "complexity_level": self.assess_complexity(document),   # 専門性のレベル
            "factual_density ": self.measure_factual_content(document),  # 事実情報の密度
            "structure       ": self.analyze_structure(document),  # 構造化の度合い
            "language        ": self.detect_language(document),     # 言語と文体
            "domain          ": self.identify_domain(document),       # ドメイン特定
            "length          ": len(document.split()),               # 文書長
            "ambiguity_level ": self.assess_ambiguity(document)  # 曖昧さの度合い
        }

    def define_qa_requirements(self):
        """Q/A生成の要件定義"""
        return {
            "purpose          ": ["評価", "学習", "検索テスト", "ユーザー支援"],
            "question_types   ": [
                "事実確認型",    # What/Who/When/Where
                "理解確認型",    # Why/How
                "推論型",       # What if/影響は
                "要約型",       # 要点は何か
                "比較型"        # 違いは何か
            ],
            "difficulty_levels": ["基礎", "中級", "上級", "専門家"],
            "answer_formats   ": ["短答", "説明", "リスト", "段落"],
            "coverage_targets ": {
                "minimum      ": 0.3,  # 最低限のカバレッジ
                "optimal      ": 0.6,  # 最適なカバレッジ
                "comprehensive": 0.8  # 包括的なカバレッジ
            }
        }


# ---------------------
# LLMベースの自動生成
# ---------------------
from openai import OpenAI
import json
from typing import List, Dict


class LLMBasedQAGenerator:
    """LLMを使用したQ/A生成"""

    def __init__(self, model="gpt-5-mini"):
        self.client = OpenAI()
        self.model = model

    def generate_basic_qa(self, text: str, num_pairs: int = 5) -> List[Dict]:
        """基本的なQ/A生成"""

        prompt = f"""
        以下のテキストから{num_pairs}個の質問と回答のペアを生成してください。

        要件：
        1. 質問は具体的で明確にする
        2. 回答はテキストから直接答えられるものにする
        3. 質問の種類を多様にする（What/Why/How/When/Where）
        4. 回答は簡潔かつ正確にする

        テキスト：
        {text[:3000]}  # トークン制限のため切り詰め

        出力形式（JSON）：
        {{
            "qa_pairs": [
                {{
                    "question": "質問文",
                    "answer": "回答文",
                    "question_type": "種類（factual/reasoning/summary等）",
                    "difficulty": "難易度（easy/medium/hard）",
                    "source_span": "回答の根拠となる元テキストの一部"
                }}
            ]
        }}
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )

        return json.loads(response.choices[0].message.content)["qa_pairs"]

    def generate_diverse_qa(self, text: str) -> List[Dict]:
        """多様な種類のQ/A生成"""

        qa_types = {
            "factual"    : "事実確認の質問（Who/What/When/Where）",
            "causal"     : "因果関係の質問（Why/How）",
            "comparative": "比較の質問（違い、類似点）",
            "inferential": "推論が必要な質問",
            "summary"    : "要約を求める質問",
            "application": "応用・活用に関する質問"
        }

        all_qa_pairs = []

        for qa_type, description in qa_types.items():
            prompt = f"""
            以下のテキストから「{description}」を2個生成してください。

            テキスト：{text[:2000]}

            JSON形式で出力：
            {{"qa_pairs": [...]}}
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            qa_pairs = json.loads(response.choices[0].message.content)["qa_pairs"]
            for qa in qa_pairs:
                qa["question_type"] = qa_type
            all_qa_pairs.extend(qa_pairs)

        return all_qa_pairs

# ----------------------------------
# Chain-of-Thoughtを使った高品質Q/A生成
# ----------------------------------
class ChainOfThoughtQAGenerator:
    """思考の連鎖を使った高品質Q/A生成"""

    def generate_with_reasoning(self, text: str) -> List[Dict]:
        """推論過程付きのQ/A生成"""

        prompt = f"""
        以下のテキストから質の高いQ/Aペアを生成します。
        各ステップを踏んで考えてください。

        ステップ1: テキストの主要なトピックと概念を抽出
        ステップ2: 各トピックについて重要な情報を特定
        ステップ3: その情報を問う質問を設計
        ステップ4: テキストから正確な回答を抽出
        ステップ5: 質問と回答の妥当性を検証

        テキスト：
        {text}

        出力形式：
        {{
            "analysis": {{
                "main_topics": ["トピック1", "トピック2"],
                "key_concepts": ["概念1", "概念2"],
                "information_density": "high/medium/low"
            }},
            "qa_pairs": [
                {{
                    "question": "質問",
                    "answer": "回答",
                    "reasoning": "なぜこの質問が重要か",
                    "confidence": 0.95
                }}
            ]
        }}
        """

        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3  # より確定的な出力のため低温
        )

        return json.loads(response.choices[0].message.content)

# ------------------------------
# ルールベース/パターンベース生成
# ------------------------------
import re
import spacy
from typing import List, Tuple


class RuleBasedQAGenerator:
    """ルールベースのQ/A生成"""

    def __init__(self):
        self.nlp = spacy.load("ja_core_news_lg")

    def extract_definition_qa(self, text: str) -> List[Dict]:
        """定義文からQ/A生成"""

        qa_pairs = []

        # パターン1: 「〜とは〜である」
        pattern1 = r'([^。]+)とは([^。]+)(?:である|です)'
        matches = re.findall(pattern1, text)

        for term, definition in matches:
            qa_pairs.append({
                "question"  : f"{term.strip()}とは何ですか？",
                "answer"    : f"{term.strip()}とは{definition.strip()}です。",
                "type"      : "definition",
                "confidence": 0.9
            })

        # パターン2: 「〜は〜と呼ばれる」
        pattern2 = r'([^。]+)は([^。]+)と呼ばれ'
        matches = re.findall(pattern2, text)

        for subject, name in matches:
            qa_pairs.append({
                "question"  : f"{subject.strip()}は何と呼ばれますか？",
                "answer"    : f"{name.strip()}と呼ばれます。",
                "type"      : "terminology",
                "confidence": 0.85
            })

        return qa_pairs

    def extract_fact_qa(self, text: str) -> List[Dict]:
        """事実情報からQ/A生成"""

        doc = self.nlp(text)
        qa_pairs = []

        for sent in doc.sents:
            # 主語と動詞を含む文を対象
            subjects = [token for token in sent if token.dep_ == "nsubj"]
            verbs = [token for token in sent if token.pos_ == "VERB"]

            if subjects and verbs:
                # 日付を含む文
                dates = [ent for ent in sent.ents if ent.label_ == "DATE"]
                if dates:
                    qa_pairs.append({
                        "question"  : f"{subjects[0].text}はいつ{verbs[0].text}ましたか？",
                        "answer"    : f"{dates[0].text}です。",
                        "type"      : "temporal",
                        "confidence": 0.7
                    })

                # 場所を含む文
                locations = [ent for ent in sent.ents if ent.label_ in ["GPE", "LOC"]]
                if locations:
                    qa_pairs.append({
                        "question"  : f"{subjects[0].text}はどこで{verbs[0].text}ますか？",
                        "answer"    : f"{locations[0].text}です。",
                        "type"      : "location",
                        "confidence": 0.7
                    })

        return qa_pairs

    def extract_list_qa(self, text: str) -> List[Dict]:
        """列挙からQ/A生成"""

        qa_pairs = []

        # 「〜には、A、B、Cがある」パターン
        pattern = r'([^。]+)(?:には|に|では)、([^。]+(?:、[^。]+)+)が(?:ある|あります|含まれ|存在)'
        matches = re.findall(pattern, text)

        for topic, items in matches:
            item_list = [item.strip() for item in items.split('、')]
            qa_pairs.append({
                "question"  : f"{topic.strip()}には何がありますか？",
                "answer"    : f"{topic.strip()}には、{'、'.join(item_list)}があります。",
                "type"      : "enumeration",
                "items"     : item_list,
                "confidence": 0.8
            })

        return qa_pairs

# ------------------------------
# テンプレートベース生成
# ------------------------------
class TemplateBasedQAGenerator:
    """テンプレートを使用したQ/A生成"""

    def __init__(self):
        self.templates = self._load_templates()

    def _load_templates(self):
        """質問テンプレートの定義"""
        return {
            "comparison"     : [
                "{A}と{B}の違いは何ですか？",
                "{A}と{B}のどちらが{property}ですか？",
                "{A}と{B}の共通点は何ですか？"
            ],
            "process"        : [
                "{process}のプロセスを説明してください。",
                "{process}にはどのようなステップがありますか？",
                "{process}の最初のステップは何ですか？"
            ],
            "cause_effect"   : [
                "{event}の原因は何ですか？",
                "{cause}の結果として何が起こりますか？",
                "なぜ{phenomenon}が発生するのですか？"
            ],
            "characteristics": [
                "{entity}の特徴は何ですか？",
                "{entity}の主な機能は何ですか？",
                "{entity}はどのように使用されますか？"
            ]
        }

    def generate_from_entities(self, text: str, entities: List[Dict]) -> List[Dict]:
        """抽出されたエンティティからQ/A生成"""

        qa_pairs = []

        for entity in entities:
            entity_type = entity['type']
            entity_text = entity['text']

            # エンティティタイプに応じたテンプレート選択
            if entity_type == "PERSON":
                questions = [
                    f"{entity_text}は誰ですか？",
                    f"{entity_text}は何をしましたか？",
                    f"{entity_text}の役割は何ですか？"
                ]
            elif entity_type == "ORG":
                questions = [
                    f"{entity_text}はどのような組織ですか？",
                    f"{entity_text}の目的は何ですか？",
                    f"{entity_text}はいつ設立されましたか？"
                ]
            elif entity_type == "PRODUCT":
                questions = [
                    f"{entity_text}とは何ですか？",
                    f"{entity_text}の用途は何ですか？",
                    f"{entity_text}の特徴は何ですか？"
                ]
            else:
                continue

            # テキストから回答を探索
            for question in questions:
                answer = self.find_answer_in_text(text, entity_text, question)
                if answer:
                    qa_pairs.append({
                        "question"  : question,
                        "answer"    : answer,
                        "entity"    : entity_text,
                        "type"      : "entity_based",
                        "confidence": 0.75
                    })

        return qa_pairs


# --------------------------------
# 4. ハイブリッド・段階的アプローチ
# --------------------------------
class HybridQAGenerator:
    """複数の手法を組み合わせた高度なQ/A生成"""

    def __init__(self):
        self.llm_generator = LLMBasedQAGenerator()
        self.rule_generator = RuleBasedQAGenerator()
        self.template_generator = TemplateBasedQAGenerator()

    def generate_comprehensive_qa(self, text: str,
                                  target_count: int = 20,
                                  quality_threshold: float = 0.7) -> List[Dict]:
        """包括的なQ/A生成パイプライン"""

        all_qa_pairs = []

        # Phase 1: ルールベースで確実なQ/Aを生成
        print("Phase 1: ルールベース生成...")
        rule_based_qa = []
        rule_based_qa.extend(self.rule_generator.extract_definition_qa(text))
        rule_based_qa.extend(self.rule_generator.extract_fact_qa(text))
        rule_based_qa.extend(self.rule_generator.extract_list_qa(text))

        # 高信頼度のものだけを選択
        rule_based_qa = [qa for qa in rule_based_qa
                         if qa.get('confidence', 0) >= quality_threshold]
        all_qa_pairs.extend(rule_based_qa)
        print(f"  生成数: {len(rule_based_qa)}")

        # Phase 2: テンプレートベースで補完
        print("Phase 2: テンプレートベース生成...")
        entities = self.extract_entities(text)
        template_qa = self.template_generator.generate_from_entities(text, entities)

        # 重複を除去
        template_qa = self.remove_duplicates(template_qa, all_qa_pairs)
        all_qa_pairs.extend(template_qa)
        print(f"  生成数: {len(template_qa)}")

        # Phase 3: LLMで不足分を補完
        remaining_count = target_count - len(all_qa_pairs)
        if remaining_count > 0:
            print(f"Phase 3: LLM生成（残り{remaining_count}個）...")

            # カバーされていない領域を特定
            uncovered_text = self.identify_uncovered_sections(text, all_qa_pairs)

            llm_qa = self.llm_generator.generate_diverse_qa(uncovered_text)
            llm_qa = llm_qa[:remaining_count]

            all_qa_pairs.extend(llm_qa)
            print(f"  生成数: {len(llm_qa)}")

        # Phase 4: 品質検証と改善
        print("Phase 4: 品質検証...")
        validated_qa = self.validate_and_improve_qa(all_qa_pairs, text)

        return validated_qa

    def validate_and_improve_qa(self, qa_pairs: List[Dict],
                                source_text: str) -> List[Dict]:
        """Q/Aペアの品質検証と改善"""

        validated_qa = []

        for qa in qa_pairs:
            # 検証項目
            validations = {
                "answer_found"      : self.verify_answer_in_text(
                    qa['answer'], source_text
                ),
                "question_clear"    : self.check_question_clarity(
                    qa['question']
                ),
                "no_contradiction"  : self.check_no_contradiction(
                    qa, validated_qa
                ),
                "appropriate_length": self.check_length_appropriateness(qa)
            }

            # すべての検証をパスしたものだけを採用
            if all(validations.values()):
                qa['validations'] = validations
                qa['quality_score'] = self.calculate_quality_score(qa)
                validated_qa.append(qa)

        # 品質スコアでソート
        validated_qa.sort(key=lambda x: x['quality_score'], reverse=True)

        return validated_qa


# -------------------------
# 5. 特殊なQ/A生成手法
# -------------------------
class AdvancedQAGenerationTechniques:
    """高度なQ/A生成技術"""

    def generate_adversarial_qa(self, text: str, existing_qa: List[Dict]) -> List[Dict]:
        """敵対的Q/A（システムを混乱させる質問）の生成"""

        adversarial_qa = []

        # 1. 否定質問
        for qa in existing_qa[:5]:
            adversarial_qa.append({
                "question": qa['question'].replace("何ですか", "何ではありませんか"),
                "answer"  : f"{qa['answer']}ではないものを指します。",
                "type"    : "adversarial_negation"
            })

        # 2. 文脈外質問
        adversarial_qa.append({
            "question": "この文書に書かれていない情報は何ですか？",
            "answer"  : "文書には含まれていない情報です。",
            "type"    : "out_of_context"
        })

        # 3. 曖昧な参照
        adversarial_qa.append({
            "question": "それは何を指していますか？",
            "answer"  : "文脈により異なります。",
            "type"    : "ambiguous_reference"
        })

        return adversarial_qa

    def generate_multi_hop_qa(self, text: str) -> List[Dict]:
        """マルチホップ推論が必要なQ/A生成"""

        prompt = f"""
        以下のテキストから、複数の情報を組み合わせて答える必要がある質問を生成してください。

        例：
        - AがBである、BがCである → AとCの関係は？
        - XはYより大きい、YはZより大きい → X、Y、Zの順序は？

        テキスト：{text}

        JSON形式で3つ生成してください。
        """

        # LLM呼び出し（実装略）
        return []

    def generate_counterfactual_qa(self, text: str) -> List[Dict]:
        """反事実的Q/A（もし〜だったら）の生成"""

        counterfactual_templates = [
            "もし{condition}でなかったら、{outcome}はどうなっていましたか？",
            "{event}が起こらなかった場合、何が変わっていましたか？",
            "{factor}が異なっていたら、結果はどう変わりますか？"
        ]

        # テキストから条件と結果を抽出して適用
        return []

#
# 実装上の最適化戦略
#
class QAGenerationOptimizer:
    """Q/A生成の最適化"""

    def optimize_for_coverage(self, text: str, budget: int) -> Dict:
        """カバレッジを最大化する生成戦略"""

        strategy = {
            "phase1": {
                "method"     : "rule_based",
                "target"     : "high_confidence_facts",
                "cost"       : 0,
                "expected_qa": 10
            },
            "phase2": {
                "method"     : "template_based",
                "target"     : "entities_and_concepts",
                "cost"       : 0,
                "expected_qa": 15
            },
            "phase3": {
                "method"     : "llm_cheap",
                "model"      : "gpt-3.5-turbo",
                "target"     : "gap_filling",
                "cost"       : budget * 0.3,
                "expected_qa": 20
            },
            "phase4": {
                "method"     : "llm_quality",
                "model"      : "gpt-4o",
                "target"     : "complex_reasoning",
                "cost"       : budget * 0.5,
                "expected_qa": 10
            },
            "phase5": {
                "method"     : "human_validation",
                "target"     : "quality_assurance",
                "cost"       : budget * 0.2,
                "expected_qa": "validation_only"
            }
        }

        return strategy

    def adaptive_generation(self, text: str,
                            initial_qa: List[Dict]) -> List[Dict]:
        """既存Q/Aを分析して適応的に生成"""

        # カバレッジ分析
        coverage_analysis = self.analyze_coverage(text, initial_qa)

        # 不足している質問タイプを特定
        missing_types = self.identify_missing_question_types(initial_qa)

        # ギャップを埋める新しいQ/A生成
        new_qa = []
        for missing_type in missing_types:
            new_qa.extend(
                self.generate_specific_type(text, missing_type, count=3)
            )

        return new_qa

#
# 考慮事項のまとめ
#
def qa_generation_checklist():
    """Q/A生成時のチェックリスト"""

    return {
        "事前準備"  : [
            "□ 文書の種類と特性を分析",
            "□ 目的（評価/学習/テスト）を明確化",
            "□ 必要なカバレッジレベルを設定",
            "□ 予算とリソースを確認"
        ],
        "品質基準"  : [
            "□ 回答がテキスト内に存在することを確認",
            "□ 質問の明確性と曖昧さの排除",
            "□ 質問タイプの多様性を確保",
            "□ 難易度のバランスを調整"
        ],
        "技術選択"  : [
            "□ ルールベースで基本的なQ/Aを生成",
            "□ LLMで複雑な推論Q/Aを補完",
            "□ ハイブリッドアプローチで最適化",
            "□ 人間のレビューで品質保証"
        ],
        "評価と改善": [
            "□ カバレッジ測定の実施",
            "□ 重複と矛盾の検出",
            "□ ユーザーフィードバックの収集",
            "□ 継続的な改善サイクル"
        ]
    }

