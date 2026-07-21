#!/usr/bin/env python3
"""
s01 / 代码 2: 词袋模型 + 余弦相似度。用最朴素的"段落 → 词频向量"代替 embedding，
概念等价于 s04 的 BGE 向量检索；这里省去模型下载，让 s01 自包含。

运行: python s01_what_is_rag/bag_of_words.py
需要: 无外部依赖
"""
import math
import re
from collections import Counter
from pathlib import Path

from docx import Document

WORKDIR = Path(__file__).resolve().parents[1]
SAMPLE = WORKDIR / "samples" / "disclosure.docx"

# 极简中文 tokenizer —— 1-2 字滑动窗口。不是 jieba，但足够"向量检索"概念演示。
def tokenize(text: str) -> list[str]:
    """2-gram 滑动窗口 tokenize. 不是 jieba, 但足够"向量检索"概念演示."""
    text = re.sub(r"\s+", "", text)
    return [text[i : i + 2] for i in range(len(text) - 1)]


def load_paragraphs(path: Path) -> list[str]:
    return [p.text for p in Document(path).paragraphs if p.text.strip()]


def vectorize(text: str, vocab: dict[str, int]) -> list[float]:
    """把段落转成词频向量, 词表外 token 丢弃."""
    counter = Counter(tokenize(text))
    return [float(counter.get(tok, 0)) for tok in vocab]


def cosine(a: list[float], b: list[float]) -> float:
    """手写余弦相似度，等价 numpy 的 dot / (norm(a) * norm(b))."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def top_k(query: str, paragraphs: list[str], k: int = 3) -> list[tuple[str, float]]:
    """对每段打分，按分排序返回 top-k. 等价 s05 的 chroma.col.query()."""
    # 全局词表：所有段落 token 集合
    vocab: dict[str, int] = {}
    for p in paragraphs:
        for tok in set(tokenize(p)):
            vocab.setdefault(tok, len(vocab))

    para_vecs = [vectorize(p, vocab) for p in paragraphs]
    q_vec = vectorize(query, vocab)

    scored = [(p, cosine(q_vec, pv)) for p, pv in zip(paragraphs, para_vecs)]
    scored.sort(key=lambda x: -x[1])
    return scored[:k]


def main() -> None:
    paragraphs = load_paragraphs(SAMPLE)
    # 默认 query: 演示"有内容时如何工作";stdin closed 时 (CI / pipe) 走同一路径
    try:
        q = input("问点啥: ").strip()
    except EOFError:
        q = "营业收入"
    if not q:
        q = "营业收入"
    print(f"\nTop-3 与你的问题最相关的段落（按向量余弦排序）：")
    for rank, (text, score) in enumerate(top_k(q, paragraphs, k=3), 1):
        snippet = text[:120].replace("\n", " ")
        print(f"\n[{rank}] score={score:.3f}")
        print(f"    {snippet}...")


if __name__ == "__main__":
    main()
