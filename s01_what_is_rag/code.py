#!/usr/bin/env python3
"""
s01 什么是 RAG — 用 30 行做一个"假 RAG"：在文档里找包含问题的段落，作为答案。

运行: python s01_what_is_rag/code.py
需要: 无外部依赖；samples/disclosure.docx 存在
"""
from pathlib import Path
from docx import Document

WORKDIR = Path(__file__).parent.parent
SAMPLE = WORKDIR / "samples" / "disclosure.docx"


def load_paragraphs(path: Path) -> list[str]:
    return [p.text for p in Document(path).paragraphs if p.text.strip()]


def fake_rag(question: str, paragraphs: list[str]) -> str:
    for p in paragraphs:
        if any(w.lower() in p.lower() for w in question.split()):
            return p
    return "I don't know."


def main() -> None:
    paragraphs = load_paragraphs(SAMPLE)
    question = input("问点啥: ").strip()
    print(fake_rag(question, paragraphs))


if __name__ == "__main__":
    main()
