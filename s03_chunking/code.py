#!/usr/bin/env python3
"""
s03 文本分块 — 按段落切，超过 max_chars 的段落再按句子切。

运行: python s03_chunking/code.py
需要: 跑通 s02；samples/ 下有真实文件
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from s02_doc_loading.code import load_pdf, load_docx

WORKDIR = Path(__file__).parent.parent
SAMPLES = WORKDIR / "samples"


def split_long_paragraph(text: str, max_chars: int) -> list[str]:
    """把超长段落按中英句子边界切成 <= max_chars 的若干块。

    修复说明: 原示例用 `text.replace("。", ".。").split("。")` 对纯中文文本
    没问题,但对纯英文文本会丢掉以 "." 结尾的句子末尾空白、并在没有中文标
    点的文本上把整段当一句。这里用 lookbehind 正则在 [.。!?！？] 之后切,
    同时覆盖中英文标点。极端兜底: 单句本身超过 max_chars(常见于表格/规格
    表)再按字符硬切,保证最坏情况下也不会输出超过 2*max_chars 的块。
    """
    sentences = re.split(r"(?<=[.。!?！？])\s*", text)
    parts, buf = [], ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # 单句本身超长(无标点表格/规格表)→ 按字符硬切
        if len(sentence) > max_chars:
            if buf:
                parts.append(buf)
                buf = ""
            for i in range(0, len(sentence), max_chars):
                parts.append(sentence[i:i + max_chars])
            continue
        if len(buf) + len(sentence) + 1 > max_chars and buf:
            parts.append(buf)
            buf = sentence
        else:
            buf = (buf + sentence).strip() if buf else sentence
    if buf:
        parts.append(buf)
    return parts


def chunk_by_paragraph(docs: list[dict], max_chars: int = 500) -> list[dict]:
    out = []
    for doc in docs:
        if len(doc["text"]) <= max_chars:
            out.append({**doc, "chunk_id": f"{doc['source']}#{doc.get('page', 0)}#p{len(out)}"})
        else:
            for piece in split_long_paragraph(doc["text"], max_chars):
                out.append({**doc, "text": piece, "chunk_id": f"{doc['source']}#{doc.get('page', 0)}#p{len(out)}"})
    return out


def main() -> None:
    docs = load_pdf(SAMPLES / "server_whitepaper.pdf") + load_docx(SAMPLES / "disclosure.docx")
    chunks = chunk_by_paragraph(docs)
    print(f"输入段落 {len(docs)} → 输出块 {len(chunks)}")
    for c in chunks[:3]:
        print(c["chunk_id"], "|", c["text"][:60])


if __name__ == "__main__":
    main()
