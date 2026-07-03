#!/usr/bin/env python3
"""
s07 重排序 — 用 BGE-reranker 对召回结果精排。

运行: python s07_rerank/code.py
需要: 跑通 s06；首次跑会下载 BAAI/bge-reranker-base (~1GB)
"""
import sys
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)
WORKDIR = Path(__file__).parent.parent


@lru_cache(maxsize=1)
def _reranker():
    from FlagEmbedding import FlagReranker
    return FlagReranker("BAAI/bge-reranker-base", use_fp16=False)


def rerank(query: str, hits: list[dict], top_k: int = 3) -> list[dict]:
    if not hits:
        return []
    rr = _reranker()
    pairs = [[query, h["text"]] for h in hits]
    scores = rr.compute_score(pairs, normalize=True)
    scored = sorted(zip(hits, scores), key=lambda x: -x[1])
    out = []
    for h, s in scored[:top_k]:
        out.append({**h, "rerank_score": s})
    return out


def main() -> None:
    sys.path.insert(0, str(WORKDIR))
    from s04_embedding.code import embed
    from s06_retrieval.code import hybrid_search
    import chromadb
    col = chromadb.PersistentClient(path=str(WORKDIR / "s05_vector_index" / "_chroma")).get_collection("docs")
    query = input("问: ").strip()
    qv = embed([query])[0]
    candidates = hybrid_search(col, query, qv, k=10, alpha=0.5)
    for hit in rerank(query, candidates, top_k=3):
        print(f"[{hit['source']}#{hit['page']}] rerank={hit['rerank_score']:.3f} vec={hit.get('score', 0):.3f} | {hit['text'][:50]}")


if __name__ == "__main__":
    main()