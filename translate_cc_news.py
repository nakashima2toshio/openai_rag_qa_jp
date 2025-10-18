"""
Translate OUTPUT/cc_news.txt into Japanese without omission and write
to OUTPUT/cc_news_jp.txt. Uses OpenAI Responses API with chunking.

Usage:
  uv run python translate_cc_news.py \
    --in OUTPUT/cc_news.txt --out OUTPUT/cc_news_jp.txt \
    --model gpt-4o-mini --chunk-tokens 3500

Requires env:
  OPENAI_API_KEY (from .env)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

import tiktoken
from openai import OpenAI
from openai import BadRequestError


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def get_encoding(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        # gpt-4o family uses o200k_base; fallback to cl100k_base if missing
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")


def tokenize_len(encoding, text: str) -> int:
    return len(encoding.encode(text))


def _split_long_text(text: str, encoding, max_tokens: int) -> List[str]:
    """Split a single text into <= max_tokens chunks by binary search on chars."""
    parts: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        lo, hi = 1, n - start
        best = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            segment = text[start:start + mid]
            if tokenize_len(encoding, segment) <= max_tokens:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        segment = text[start:start + best]
        # try to end at a newline if possible
        nl = segment.rfind("\n")
        if nl >= 0 and nl > best * 2 // 3:
            segment = segment[:nl + 1]
        parts.append(segment)
        start += len(segment)
    return parts


def chunk_by_tokens(text: str, encoding, max_tokens: int) -> List[str]:
    """Greedy chunking by lines with fallback to char-based splitting for long lines."""
    chunks: List[str] = []
    cur: List[str] = []
    cur_tokens = 0
    for line in text.splitlines(keepends=True):
        t = tokenize_len(encoding, line)
        if t > max_tokens:
            # flush current
            if cur:
                chunks.append("".join(cur))
                cur, cur_tokens = [], 0
            # split the long line by token budget
            for piece in _split_long_text(line, encoding, max_tokens):
                chunks.append(piece)
        else:
            if cur_tokens + t > max_tokens:
                chunks.append("".join(cur))
                cur, cur_tokens = [], 0
            cur.append(line)
            cur_tokens += t
    if cur:
        chunks.append("".join(cur))
    return chunks


def build_prompt(segment: str) -> str:
    return (
        "以下の英語テキストを、日本語に忠実に翻訳してください。\n"
        "- 省略や要約を行わないこと\n"
        "- 構成・段落・改行は可能な限り保持すること\n"
        "- 意味を変えずに自然な日本語にすること\n\n"
        "テキスト:\n\n" + segment
    )


def translate_segments(client: OpenAI, model: str, segments: List[str]) -> List[str]:
    results: List[str] = []
    for i, seg in enumerate(segments, 1):
        print(f"Translating segment {i}/{len(segments)} (chars={len(seg)})...")
        kwargs = {
            "model": model,
            "input": [
                {"role": "system", "content": "You are a professional English-to-Japanese translator."},
                {"role": "user", "content": build_prompt(seg)},
            ],
            "max_output_tokens": 8000,
        }
        # Some models (e.g., gpt-5) may not support temperature
        if not model.lower().startswith("gpt-5"):
            kwargs["temperature"] = 0.2
        try:
            resp = client.responses.create(**kwargs)
        except BadRequestError as e:
            # Retry without temperature if unsupported
            if "Unsupported parameter" in str(e) or "temperature" in str(e):
                kwargs.pop("temperature", None)
                resp = client.responses.create(**kwargs)
            else:
                raise
        # Extract text content
        out = []
        for item in resp.output_text.split("\n"):
            out.append(item)
        results.append("\n".join(out))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="OUTPUT/cc_news.txt")
    parser.add_argument("--out", dest="out_path", default="OUTPUT/cc_news_jp.txt")
    parser.add_argument("--model", default=os.getenv("TRANSLATE_MODEL", "gpt-4o-mini"))
    parser.add_argument("--chunk-tokens", type=int, default=int(os.getenv("TRANSLATE_CHUNK_TOKENS", "3500")))
    args = parser.parse_args()

    load_dotenv()  # read .env for OPENAI_API_KEY

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    model = args.model
    max_tokens_in = args.chunk_tokens

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    print(f"Reading: {in_path}")
    text = load_text(in_path)

    enc = get_encoding(model)
    segments = chunk_by_tokens(text, enc, max_tokens_in)
    print(f"Total chars: {len(text)}, segments: {len(segments)}")

    client = OpenAI()
    translated_segments = translate_segments(client, model, segments)

    jp = "\n\n".join(translated_segments)
    save_text(out_path, jp)
    print(f"Saved: {out_path} (chars={len(jp)})")


if __name__ == "__main__":
    main()
