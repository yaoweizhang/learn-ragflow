#!/usr/bin/env python3
"""
s01 / 代码 3: RAG pipeline — 完整 RAG 链路：检索 + 拼 prompt + LLM 生成。

保留 代码 2 的词袋模型检索（自包含，不依赖 s04 真 embedding），
加上"按检索结果生成答案"这一步。
若环境变量 LLM_API_KEY 未设置，则打印 prompt 但不真调 LLM，
便于在没有 key 的机器上也能跑通。

运行:
  LLM_API_KEY=sk-xxx python s01_what_is_rag/rag_pipeline.py
  # 或者无 key:
  python s01_what_is_rag/rag_pipeline.py
"""
import math
import os
import re
import urllib.request
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from docx import Document

load_dotenv(override=True)

WORKDIR = Path(__file__).resolve().parents[1]
SAMPLE = WORKDIR / "samples" / "disclosure.docx"

LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE = os.environ.get("LLM_BASE", "https://api.minimaxi.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "MiniMax-Text-01")


def tokenize(text: str) -> list[str]:
    text = re.sub(r"\s+", "", text)
    return [text[i : i + 2] for i in range(len(text) - 1)]


def load_paragraphs(path: Path) -> list[str]:
    return [p.text for p in Document(path).paragraphs if p.text.strip()]


def build_vocab(paragraphs: list[str]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for p in paragraphs:
        for tok in set(tokenize(p)):
            vocab.setdefault(tok, len(vocab))
    return vocab


def vectorize(text: str, vocab: dict[str, int]) -> list[float]:
    counter = Counter(tokenize(text))
    return [float(counter.get(tok, 0)) for tok in vocab]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def retrieve(query: str, paragraphs: list[str], k: int = 3) -> list[str]:
    vocab = build_vocab(paragraphs)
    para_vecs = [vectorize(p, vocab) for p in paragraphs]
    q_vec = vectorize(query, vocab)
    scored = sorted(
        zip(paragraphs, (cosine(q_vec, pv) for pv in para_vecs)),
        key=lambda x: -x[1],
    )
    return [p for p, _ in scored[:k]]


def build_prompt(question: str, hits: list[str]) -> str:
    """对照 docs/reference/ragflow-notes/prompt_templates.md 里的 'You are an AI assistant...' 模板.
    本章用极简版，只保留 [i] (source) text 的渲染."""
    ctx = "\n\n".join(f"[{i + 1}] {h}" for i, h in enumerate(hits))
    return (
        "你只能依据 <context> 标签内的资料回答问题；\n"
        "若资料不足以回答，请回复「我不知道」。\n\n"
        f"<context>\n{ctx}\n</context>\n\n"
        f"问题: {question}\n"
        "回答: "
    )


def call_llm(prompt: str) -> str:
    """最小可用 OpenAI 兼容调用；零 SDK 依赖。"""
    if not LLM_API_KEY:
        return ""

    import json
    req = urllib.request.Request(
        f"{LLM_BASE}/chat/completions",
        data=json.dumps(
            {
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            }
        ).encode(),
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def main() -> None:
    paragraphs = load_paragraphs(SAMPLE)
    q = input("问点啥: ").strip()

    hits = retrieve(q, paragraphs, k=3)
    print(f"\n[retrieve] 召回 {len(hits)} 段")
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] {h[:80].replace(chr(10), ' ')}...")

    prompt = build_prompt(q, hits)
    print(f"\n[prompt]\n{prompt}\n")

    if LLM_API_KEY:
        answer = call_llm(prompt)
        print(f"[llm] {answer}")
    else:
        print("[llm] LLM_API_KEY 未设置，跳过真实生成；如需 LLM 回答:")
        print("      LLM_API_KEY=sk-xxx python s01_what_is_rag/rag_pipeline.py")


if __name__ == "__main__":
    main()