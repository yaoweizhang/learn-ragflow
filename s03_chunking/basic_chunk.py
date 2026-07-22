#!/usr/bin/env python3
"""
s03 / unit 01 — 文本分块最小解法：500 字符 cap + 中英句界切。

短段 (< 500 字符) 整段保留;超长段先按句界切,再贪心装桶,保证每块
≤ max_chars;极长无标点串(规格表)按字符硬切兜底。每个 chunk 带
`chunk_id = {source}#{page}#p{n}` 给 s04+ 引用。

unit 02 会把同一套函数跑在真实样本上,展示 3 类典型失败
(表格切碎 / 父子块缺失 / 跨段引用断裂)。

运行: python s03_chunking/basic_chunk.py
需要: 跑通 s02;samples/{server_whitepaper.pdf,disclosure.docx}

术语速览 (本文件首次出现):
- chunk: 一段被切出来喂给 embedding / 检索的连续文本块
- chunk_id: chunk 的唯一标识,常用 `{source}#{page}#p{n}` 格式
- 句界切: 在 [.。!?！？] 等中英文标点后切分,避免把句子劈成两半
- lookbehind 正则: `(?<=...)` 零宽断言,从右往左看但不消耗字符
- 贪心装桶: 把句子按顺序塞桶,桶满就开新桶,保证块长度 ≤ cap
- max_chars: 每块最大字符数,本教程默认 500
"""
import re
import sys
from pathlib import Path

# 复用 s02 unit 01 的 loader——self-contained 是"文件级自包含",
# 不禁止在同一章节内 import 已经存在的工具层(loader 在 s02
# 里是上游契约,不重新实现)。
import importlib.util

_S02_UNIT01 = Path(__file__).resolve().parents[1] / "s02_doc_loading" / "basic_load.py"
_spec = importlib.util.spec_from_file_location("s02_unit01_basic_load", _S02_UNIT01)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
load_pdf = _mod.load_pdf
load_docx = _mod.load_docx

WORKDIR = Path(__file__).resolve().parents[1]
SAMPLES = WORKDIR / "samples"


def split_long_paragraph(text: str, max_chars: int) -> list[str]:
    """把超长段落按中英句子边界切成 <= max_chars 的若干块。

    用 lookbehind 正则在 [.。!?！？] 之后切,同时覆盖中英文标点。
    单句本身超过 max_chars(常见于表格/规格表)再按字符硬切,
    保证最坏情况下也不会输出超过 2*max_chars 的块。
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
    """短段整段保留,长段按 split_long_paragraph 切成多块。"""
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
    print(f"最大块长度 {max(len(c['text']) for c in chunks)} 字符 (cap=500)")
    for c in chunks[:3]:
        print(c["chunk_id"], "|", c["text"][:60])


if __name__ == "__main__":
    main()