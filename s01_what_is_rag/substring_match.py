#!/usr/bin/env python3
"""
s01 / 代码 1: 子串字面匹配 — 按子串命中把段落拉回来。不调用 LLM。

运行: python s01_what_is_rag/substring_match.py
需要: 无外部依赖；samples/disclosure.docx 存在
"""
from pathlib import Path
from docx import Document

# 把 samples/ 路径写死：上层目录 + samples/disclosure.docx
WORKDIR = Path(__file__).resolve().parents[1]
SAMPLE = WORKDIR / "samples" / "disclosure.docx"


def load_paragraphs(path: Path) -> list[str]:
    """只取非空段落，去掉 Word 文档里那些"占位用的空段"."""
    return [p.text for p in Document(path).paragraphs if p.text.strip()]


def fake_rag(question: str, paragraphs: list[str]) -> str:
    """朴素检索: 拿问题里的每个词去段落里找子串, 命中就返回第一个."""
    for p in paragraphs:
        if any(w.lower() in p.lower() for w in question.split()):
            return p
    return "I don't know."


def main() -> None:
    paragraphs = load_paragraphs(SAMPLE)
    # 默认 query: 演示"有内容时如何工作";stdin closed 时 (CI / pipe) 走同一路径
    try:
        q = input("问点啥: ").strip()
    except EOFError:
        q = "营业收入"
    if not q:
        q = "营业收入"
    print(fake_rag(q, paragraphs))


if __name__ == "__main__":
    main()
