#!/usr/bin/env python3
"""
s05 向量索引 — 用 Chroma 存向量 + 元数据，支持相似度检索。

运行: python s05_vector_index/code.py
需要: 跑通 s04；首次跑会建 s05_vector_index/_chroma/ 目录
"""
import os
import sys
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path(__file__).parent.parent
DB_DIR = WORKDIR / "s05_vector_index" / "_chroma"
SAMPLES = WORKDIR / "samples"


def build_index(chunks: list[dict], vectors: list[list[float]]):
    import chromadb
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col = client.create_collection("docs", metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "page": str(c.get("page", ""))} for c in chunks],
    )
    return col


def search(col, query_vec: list[float], k: int = 5) -> list[dict]:
    res = col.query(query_embeddings=[query_vec], n_results=k)
    hits = []
    for i, doc in enumerate(res["documents"][0]):
        page_val = res["metadatas"][0][i].get("page", "")
        try:
            page_val = int(page_val) if page_val != "" else None
        except (ValueError, TypeError):
            page_val = page_val
        hits.append({
            "text": doc,
            "source": res["metadatas"][0][i]["source"],
            "page": page_val,
            "score": 1 - res["distances"][0][i],  # cosine distance → similarity
        })
    return hits


def main() -> None:
    sys.path.insert(0, str(WORKDIR))
    from s03_chunking.code import load_pdf, load_docx, chunk_by_paragraph
    from s04_embedding.code import embed
    docs = load_pdf(SAMPLES / "server_whitepaper.pdf") + load_docx(SAMPLES / "disclosure.docx")
    chunks = chunk_by_paragraph(docs)
    vectors = embed([c["text"] for c in chunks])
    col = build_index(chunks, vectors)
    query = input("问: ").strip()
    qv = embed([query])[0]
    for hit in search(col, qv, k=3):
        print(f"[{hit['source']}#{hit['page']}] score={hit['score']:.3f} | {hit['text'][:60]}")


if __name__ == "__main__":
    main()
