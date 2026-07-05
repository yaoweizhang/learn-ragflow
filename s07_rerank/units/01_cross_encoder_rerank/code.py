#!/usr/bin/env python3
"""
s07 / unit 01 — Cross-encoder 精排 (BGE-reranker):把 s06 召回的 top-N
命中用 `FlagReranker("BAAI/bge-reranker-base")` 重打分并按新分排序,
打印 rerank 之前 vs 之后的 top-3。

本单元 self-contained:内联 pypdf + python-docx 加载、内联 s03 的 500 字符
cap 切块、内联 s04 unit 01 的本地 BGE embed、内联 s06 unit 02 的 BM25+dense
混合召回,完全 self-contained 跑通 BM25+dense 召回 → cross-encoder 精排
对比。不跨章节 import,也不跨 unit import(直接 importlib 加载同章节的
s06 unit 01 BM25)。

运行: python s07_rerank/units/01_cross_encoder_rerank/code.py
需要: pip install pypdf python-docx sentence-transformers chromadb FlagEmbedding;
samples/{server_whitepaper.pdf,disclosure.docx};首次跑会下载 BAAI/bge-reranker-base (~1GB)
"""
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path(__file__).resolve().parents[3]
SAMPLES = WORKDIR / "samples"
DB_DIR = WORKDIR / "s05_vector_index" / "_chroma"
COLLECTION_NAME = "docs"

# Windows + urllib3>=2.5 + botocore 旧版兼容补丁;参考 s05 unit 01 的同款补丁。
try:
    import urllib3.util.ssl_ as _ssl
    if not hasattr(_ssl, "DEFAULT_CIPHERS"):
        _ssl.DEFAULT_CIPHERS = "DEFAULT@SECLEVEL=2"
except Exception:
    pass

# 复用 s06 unit 01 的 BM25/tokenize/chunker/loader(importlib 加载,不跨章节)。
_UNIT01 = WORKDIR / "s06_retrieval" / "units" / "01_bm25" / "code.py"
_spec01 = importlib.util.spec_from_file_location("s06_unit01_bm25", _UNIT01)
_mod01 = importlib.util.module_from_spec(_spec01)
sys.modules[_spec01.name] = _mod01
_spec01.loader.exec_module(_mod01)
BM25 = _mod01.BM25
_load_chunks = _mod01._load_chunks


# ---------- Embedding (跟 s05 unit 01 同款本地 BGE) ----------

@lru_cache(maxsize=1)
def _embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5"))


def _embed(texts: list[str]) -> list[list[float]]:
    m = _embed_model()
    return [v.tolist() for v in m.encode(texts, normalize_embeddings=True)]


def _cosine(a: list[float], b: list[float]) -> float:
    """两向量的 cosine 相似度,假设已 L2 归一化(BGE normalize_embeddings=True)。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------- Cross-encoder 精排 ----------

@lru_cache(maxsize=1)
def _reranker():
    from FlagEmbedding import FlagReranker
    return FlagReranker("BAAI/bge-reranker-base", use_fp16=False)


def rerank(query: str, hits: list[dict], top_k: int = 3) -> list[dict]:
    """对 s06 召回的 hits 做 cross-encoder 精排,返回按 rerank 分降序的 top-k。

    每个返回项带 `rerank_score`(cross-encoder 的 [0,1] 相关性分,FlagReranker
    normalize=True 归一后的值),同时保留 `score`(s06 的混合召回分)和
    原始的 text/source/page/chunk_id。
    """
    if not hits:
        return []
    rr = _reranker()
    pairs = [[query, h["text"]] for h in hits]
    scores = rr.compute_score(pairs, normalize=True)
    scored = sorted(zip(hits, scores), key=lambda x: -x[1])
    out = []
    for h, s in scored[:top_k]:
        out.append({**h, "rerank_score": float(s)})
    return out


# ---------- 混合召回(内联 BM25 + dense) ----------

def _hybrid_topk(docs, query, query_vec, dense_score_fn, k, alpha):
    """复制 s06 unit 02 的 hybrid_topk 公式(alpha * vec + (1-alpha) * bm25_norm)。"""
    bm = BM25(docs)
    bm_scores = bm.score(query)
    bm_max = max(bm_scores) if any(bm_scores) else 1.0
    combined = []
    for i, d in enumerate(docs):
        v = float(dense_score_fn(d))
        b = bm_scores[i] / bm_max if bm_max > 0 else 0.0
        combined.append({
            "text": d["text"], "source": d["source"],
            "page": d.get("page"), "chunk_id": d.get("chunk_id"),
            "dense": v, "bm25": bm_scores[i],
            "score": alpha * v + (1 - alpha) * b,
        })
    combined.sort(key=lambda x: -x["score"])
    return combined[:k]


# ---------- 入口 ----------

def main() -> None:
    chunks = _load_chunks()
    print(f"loaded {len(chunks)} chunks from samples/")

    # 用 s05 的 chroma 索引拿向量;若不存在则重建一个(便于单机自洽跑通)。
    import chromadb
    if not DB_DIR.exists():
        texts = [c["text"] for c in chunks]
        vectors = _embed(texts)
        client = chromadb.PersistentClient(path=str(DB_DIR))
        col = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        col.add(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=vectors,
            documents=texts,
            metadatas=[{"source": c["source"], "page": str(c.get("page", ""))} for c in chunks],
        )
    else:
        col = chromadb.PersistentClient(path=str(DB_DIR)).get_collection(COLLECTION_NAME)

    try:
        query = input("问: ").strip() or "内存"
    except EOFError:
        query = "内存"
    print(f"query={query!r}, alpha=0.5 (BM25 + dense 等权融合)")

    # 拉 s05 的 dense 向量到内存,inline 一份 dense_score_fn(不绑死 chroma)。
    raw = col.get(include=["embeddings", "metadatas", "documents"])
    vec_by_id = {cid: emb for cid, emb in zip(raw["ids"], raw["embeddings"])}
    qv = _embed([query])[0]

    def _dense_score(chunk):
        return _cosine(qv, vec_by_id[chunk["chunk_id"]])

    candidates = _hybrid_topk(chunks, query, qv, _dense_score, k=10, alpha=0.5)

    # BEFORE rerank: 展示 s06 召回的 top-3(混合分排序)。
    print("\n--- BEFORE rerank (s06 混合召回 top-3) ---")
    for i, h in enumerate(candidates[:3], start=1):
        page = h["page"] if h["page"] is not None else "-"
        snippet = h["text"][:50].replace("\n", " ")
        print(f"  #{i} [{h['source']}#{page}] "
              f"score={h['score']:.3f} (vec={h['dense']:.3f}, bm25={h['bm25']:.3f}) "
              f"| {snippet}")

    # AFTER rerank: cross-encoder 在 top-10 上精排,取 top-3。
    reranked = rerank(query, candidates, top_k=3)
    print("\n--- AFTER rerank (BAAI/bge-reranker-base top-3) ---")
    for i, h in enumerate(reranked, start=1):
        page = h["page"] if h["page"] is not None else "-"
        snippet = h["text"][:50].replace("\n", " ")
        print(f"  #{i} [{h['source']}#{page}] "
              f"rerank={h['rerank_score']:.3f} vec={h.get('dense', 0):.3f} "
              f"| {snippet}")


if __name__ == "__main__":
    main()