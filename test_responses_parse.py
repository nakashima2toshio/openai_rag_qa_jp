#!/usr/bin/env python3
"""responses.parse APIのテストスクリプト"""

from openai import OpenAI
from pydantic import BaseModel
from typing import List
import json

class QAPair(BaseModel):
    question: str
    answer: str
    question_type: str

class QAPairsResponse(BaseModel):
    qa_pairs: List[QAPair]

# テストコード
def test_responses_parse():
    client = OpenAI()

    # シンプルなテキスト
    test_text = """
    The Eiffel Tower is located in Paris, France.
    It was built in 1889 and is 330 meters tall.
    """

    system_prompt = """You are an expert at creating educational Q&A pairs.
    Generate Q&A pairs from the given text."""

    user_prompt = f"""Generate 2 Q&A pairs from this text:

    {test_text}

    Output format:
    {{
      "qa_pairs": [
        {{
          "question": "question text",
          "answer": "answer text",
          "question_type": "fact"
        }}
      ]
    }}"""

    combined_input = f"{system_prompt}\n\n{user_prompt}"

    try:
        print("Calling responses.parse...")
        response = client.responses.parse(
            input=combined_input,
            model="gpt-4o-mini",  # 確実に動作するモデルを使用
            text_format=QAPairsResponse,
            max_output_tokens=1000
        )

        print("\nResponse type:", type(response))
        print("Response attributes:", dir(response))

        # レスポンスの内容を詳しく調査
        if hasattr(response, 'output'):
            print("\nResponse.output type:", type(response.output))
            print("Response.output:", response.output)

            # outputの各要素を確認
            for i, item in enumerate(response.output):
                print(f"\nOutput[{i}] type:", type(item))
                print(f"Output[{i}] attributes:", dir(item))

                if hasattr(item, 'type'):
                    print(f"Output[{i}].type:", item.type)

                if hasattr(item, 'content'):
                    print(f"Output[{i}].content type:", type(item.content))

                    # contentの各要素を確認
                    for j, content in enumerate(item.content):
                        print(f"\n  Content[{j}] type:", type(content))
                        print(f"  Content[{j}] attributes:", dir(content))

                        if hasattr(content, 'type'):
                            print(f"  Content[{j}].type:", content.type)

                        if hasattr(content, 'text'):
                            print(f"  Content[{j}].text:", content.text)

                        if hasattr(content, 'parsed'):
                            print(f"  Content[{j}].parsed type:", type(content.parsed))
                            print(f"  Content[{j}].parsed:", content.parsed)

                            # parsedがQAPairsResponseの場合
                            if isinstance(content.parsed, QAPairsResponse):
                                print("\n  Successfully parsed QAPairsResponse!")
                                print(f"  Number of Q/A pairs: {len(content.parsed.qa_pairs)}")
                                for qa in content.parsed.qa_pairs:
                                    print(f"    Q: {qa.question}")
                                    print(f"    A: {qa.answer}")
                                    print(f"    Type: {qa.question_type}")

        # 別の可能性：parsed属性が直接存在する場合
        if hasattr(response, 'parsed'):
            print("\nDirect parsed attribute found!")
            print("Response.parsed:", response.parsed)

        print("\n" + "="*50)
        print("Test completed successfully!")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_responses_parse()