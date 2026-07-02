#!/usr/bin/env python3
"""
s02 文档加载 — 把 PDF/DOCX 读成结构化段落，每段带来源和页码。

运行: python s02_doc_loading/code.py
需要: pip install pypdf python-docx；samples/server_whitepaper.pdf + samples/disclosure.docx
"""
from pathlib import Path
from pypdf import PdfReader
from docx import Document

WORKDIR = Path(__file__).parent.parent
SAMPLES = WORKDIR / "samples"


def load_pdf(path: Path) -> list[dict]:
    out = []
    for i, page in enumerate(PdfReader(path).pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            out.append({"text": text, "page": i, "source": path.name})
    return out


def load_docx(path: Path) -> list[dict]:
    out = []
    for p in Document(path).paragraphs:
        if p.text.strip():
            out.append({"text": p.text, "page": None, "source": path.name})
    return out


def main() -> None:
    pdf = load_pdf(SAMPLES / "server_whitepaper.pdf")
    docx = load_docx(SAMPLES / "disclosure.docx")
    print(f"PDF 段落数: {len(pdf)}, DOCX 段落数: {len(docx)}")
    print("PDF 第一段前 100 字:", pdf[0]["text"][:100])


if __name__ == "__main__":
    main()