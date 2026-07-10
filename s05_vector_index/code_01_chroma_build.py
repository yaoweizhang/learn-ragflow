#!/usr/bin/env python3
"""
s05 / unit 01 — Chroma 持久化向量索引：把 chunks + embeddings + metadata 写到
`_chroma/`，再用 cosine 距离做相似度检索的"承载器"。

本单元 self-contained：内联 pypdf + python-docx 加载(参考 s04 unit 01 的取舍),
内联 s03 的 500 字符 cap + 中英句界切,内联 s04 unit 01 的 BAAI/bge-small-zh-v1.5
加载;**不跨章节 import**(跟 s04 unit 01 同一套"加载器复刻"取舍)。

unit 02 会用同样的 DB_DIR 重新打开这个 collection,在它上面 query。

运行: python s05_vector_index/code_01_chroma_build.py
需要: 跑通 s02/s03/s04 等价依赖;首次跑会写 s05_vector_index/_chroma/ 目录
      (已加 .gitignore:**/_chroma/,不入库)
"""
import os
import re
import shutil
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path(__file__).resolve().parents[1]
SAMPLES = WORKDIR / "samples"
DB_DIR = WORKDIR / "s05_vector_index" / "_chroma"
COLLECTION_NAME = "docs"

# Windows + urllib3>=2.5 + botocore 旧版兼容补丁;参考 s04 unit 01 的同款补丁。
try:
    import urllib3.util.ssl_ as _ssl
    if not hasattr(_ssl, "DEFAULT_CIPHERS"):
        _ssl.DEFAULT_CIPHERS = "DEFAULT@SECLEVEL=2"
except Exception:
    pass


# ---------- 加载器 (inline pypdf + python-docx,跟 s02 unit 01 等价) ----------

def _pdf(path: Path) -> list[dict]:
    from pypdf import PdfReader
    out = []
    for i, page in enumerate(PdfReader(path).pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            out.append({"text": text, "page": i, "source": path.name})
    return out


def _docx(path: Path) -> list[dict]:
    from docx import Document
    out = []
    for p in Document(path).paragraphs:
        if p.text.strip():
            out.append({"text": p.text, "page": None, "source": path.name})
    return out


# ---------- 切块器 (inline s03 的 500 字符 cap + 中英句界切) ----------

def _split_long(text: str, max_chars: int) -> list[str]:
    """超长段按 [.。!?！？] 句界切,贪心装桶,极长无标点串按字符硬切兜底。"""
    sentences = re.split(r"(?<=[.。!?！？])\s*", text)
    parts, buf = [], ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
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


def _chunk_by_paragraph(docs: list[dict], max_chars: int = 500) -> list[dict]:
    """短段整段保留,长段按句界切;每个 chunk 带 chunk_id = {source}#{page}#p{n}。"""
    out = []
    for doc in docs:
        if len(doc["text"]) <= max_chars:
            out.append({**doc, "chunk_id": f"{doc['source']}#{doc.get('page', 0)}#p{len(out)}"})
        else:
            for piece in _split_long(doc["text"], max_chars):
                out.append({**doc, "text": piece, "chunk_id": f"{doc['source']}#{doc.get('page', 0)}#p{len(out)}"})
    return out


# ---------- Embedding (inline s04 unit 01 的本地 BGE) ----------

@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5"))


def _embed(texts: list[str]) -> list[list[float]]:
    m = _model()
    return [v.tolist() for v in m.encode(texts, normalize_embeddings=True)]


# ---------- 索引核心 ----------

def get_chunks_and_vectors() -> tuple[list[dict], list[list[float]]]:
    """加载 samples → 切块 → embed,返回 (chunks, vectors)。"""
    docs = _pdf(SAMPLES / "server_whitepaper.pdf") + _docx(SAMPLES / "disclosure.docx")
    chunks = _chunk_by_paragraph(docs)
    vectors = _embed([c["text"] for c in chunks])
    return chunks, vectors


def build_index(chunks: list[dict], vectors: list[list[float]]):
    """删旧 _chroma/ → PersistentClient → cosine space → 写入 chunks+vectors+metadata。"""
    import chromadb
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "page": str(c.get("page", ""))} for c in chunks],
    )
    return col


def main() -> None:
    chunks, vectors = get_chunks_and_vectors()
    col = build_index(chunks, vectors)
    print(f"indexed {len(chunks)} chunks into _chroma/ (collection={col.name}, dim={len(vectors[0])})")


if __name__ == "__main__":
    main()