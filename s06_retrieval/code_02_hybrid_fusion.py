#!/usr/bin/env python3
"""
s06 / unit 02 — 混合召回 fusion:BM25 + dense cosine 加权融合。

本单元 self-contained:内联 pypdf + python-docx + 500 字符 cap 切块 +
本地 BGE 加载,内联 unit 01 的 BM25/tokenize/bm25_topk,跑一个固定的
混合 query 后打印 BM25 / dense 两路分 + 加权融合后的 top-k。

加权公式:`final = alpha * vec_sim + (1 - alpha) * bm25_norm`,
其中 bm25 分先归一到 [0,1](除以 max),vec_sim 已经是 cosine ∈ [0,1]
(BGE 归一化后)。默认 alpha=0.95 偏向量,mirrors ragflow 的
FusionExpr("weighted_sum", {"weights": "0.05,0.95"})。

运行: python s06_retrieval/code_02_hybrid_fusion.py
需要: pip install pypdf python-docx sentence-transformers chromadb;
samples/{server_whitepaper.pdf,disclosure.docx}
"""
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path(__file__).resolve().parents[1]
SAMPLES = WORKDIR / "samples"

# Windows + urllib3>=2.5 + botocore 旧版兼容补丁。
try:
    import urllib3.util.ssl_ as _ssl
    if not hasattr(_ssl, "DEFAULT_CIPHERS"):
        _ssl.DEFAULT_CIPHERS = "DEFAULT@SECLEVEL=2"
except Exception:
    pass

# 复用 unit 01 的 BM25/tokenize/chunker/loader(importlib 加载,不跨章节)。
_UNIT01 = Path(__file__).resolve().parent / "code_01_bm25.py"
_spec01 = importlib.util.spec_from_file_location("s06_unit01_bm25", _UNIT01)
_mod01 = importlib.util.module_from_spec(_spec01)
sys.modules[_spec01.name] = _mod01
_spec01.loader.exec_module(_mod01)
tokenize = _mod01.tokenize
BM25 = _mod01.BM25
_load_chunks = _mod01._load_chunks


# ---------- Embedding (跟 s05 unit 01 同款本地 BGE) ----------

@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5"))


def _embed(texts: list[str]) -> list[list[float]]:
    m = _model()
    return [v.tolist() for v in m.encode(texts, normalize_embeddings=True)]


def _cosine(a: list[float], b: list[float]) -> float:
    """两向量的 cosine 相似度,假设已 L2 归一化(BGE normalize_embeddings=True)。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------- 混合召回 core ----------

def hybrid_topk(
    docs: list[dict],
    query: str,
    query_vec: list[float],
    dense_score_fn,
    k: int = 5,
    alpha: float = 0.95,
) -> list[dict]:
    """对 docs 算 BM25 分 + dense 分,各自归一 [0,1] 后 alpha 加权融合,返回 top-k。

    `dense_score_fn(chunk) -> float` 由调用方注入(可以是 chroma query、暴力
    遍历、自己实现的近似 KNN 等),保证 unit 02 不绑死任何具体向量库。
    """
    bm = BM25(docs)
    bm_scores = bm.score(query)
    bm_max = max(bm_scores) if any(bm_scores) else 1.0

    combined = []
    for i, d in enumerate(docs):
        v = float(dense_score_fn(d))
        b = bm_scores[i] / bm_max if bm_max > 0 else 0.0
        combined.append({
            "text": d["text"],
            "source": d["source"],
            "page": d.get("page"),
            "chunk_id": d.get("chunk_id"),
            "dense": v,
            "bm25": bm_scores[i],
            "score": alpha * v + (1 - alpha) * b,
        })
    combined.sort(key=lambda x: -x["score"])
    return combined[:k]


# ---------- 入口 ----------

def main() -> None:
    chunks = _load_chunks()
    print(f"loaded {len(chunks)} chunks from samples/")
    texts = [c["text"] for c in chunks]
    vectors = _embed(texts)
    vec_by_id = {c["chunk_id"]: vectors[i] for i, c in enumerate(chunks)}

    query = "应收账款 计提"
    print(f"query={query!r}, alpha=0.95 (dense-dominant)")
    qv = _embed([query])[0]

    def _dense_score(chunk: dict) -> float:
        return _cosine(qv, vec_by_id[chunk["chunk_id"]])

    hits = hybrid_topk(chunks, query, qv, _dense_score, k=5, alpha=0.95)
    print("hybrid top-3 (with both sub-scores visible):")
    for hit in hits[:3]:
        snippet = hit["text"][:60].replace("\n", " ")
        page = hit["page"] if hit["page"] is not None else "-"
        print(f"  [{hit['source']}#{page}] "
              f"final={hit['score']:.3f} = 0.95*vec({hit['dense']:.3f}) + 0.05*bm25({hit['bm25']:.3f})"
              f" | {snippet}")


if __name__ == "__main__":
    main()