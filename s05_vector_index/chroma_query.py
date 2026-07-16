#!/usr/bin/env python3
"""
s05 / unit 02 — Chroma 检索：把 unit 01 持久化的 collection 重新打开，
把 query 文本本地 embed 后做 cosine 相似度 top-k 召回。

本单元 self-contained：**不跨章节 import,也不 import unit 01**(unit 01
的 `_chroma/` 路径是公开契约);BGE 内联加载跟 unit 01 同款,保证
query embedding 跟索引时的向量在同一空间。

运行: python s05_vector_index/chroma_query.py
需要: 先跑过 unit 01 让 _chroma/ 有数据;否则会打印提示并退出。
"""
import os
import sys
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path(__file__).resolve().parents[1]
DB_DIR = WORKDIR / "s05_vector_index" / "_chroma"
COLLECTION_NAME = "docs"

# Windows + urllib3>=2.5 + botocore 旧版兼容补丁;参考 s04 unit 01 的同款补丁。
try:
    import urllib3.util.ssl_ as _ssl
    if not hasattr(_ssl, "DEFAULT_CIPHERS"):
        _ssl.DEFAULT_CIPHERS = "DEFAULT@SECLEVEL=2"
except Exception:
    pass


# ---------- Embedding (跟 unit 01 同款,保证向量空间一致) ----------

@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5"))


def _embed(texts: list[str]) -> list[list[float]]:
    m = _model()
    return [v.tolist() for v in m.encode(texts, normalize_embeddings=True)]


# ---------- 检索核心 ----------

def _open_collection():
    """打开 unit 01 持久化的 collection;不存在就返回 None 让 main 提示。"""
    import chromadb
    if not DB_DIR.exists():
        return None
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return None


def search(col, query_vec: list[float], k: int = 5) -> list[dict]:
    """对已打开的 collection 跑 query,把 cosine distance 翻成 similarity,统一返回结构。"""
    res = col.query(query_embeddings=[query_vec], n_results=k)
    hits = []
    for i, doc in enumerate(res["documents"][0]):
        page_val = res["metadatas"][0][i].get("page", "")
        try:
            page_val = int(page_val) if page_val != "" else None
        except (ValueError, TypeError):
            pass
        hits.append({
            "text": doc,
            "source": res["metadatas"][0][i]["source"],
            "page": page_val,
            "score": 1 - res["distances"][0][i],  # cosine distance ∈ [0,2] → similarity ∈ [-1,1]
        })
    return hits


def main() -> None:
    col = _open_collection()
    if col is None:
        print(f"未发现持久化索引 {DB_DIR}。请先跑:")
        print("  python s05_vector_index/chroma_build.py")
        sys.exit(1)
    try:
        query = input("问: ").strip() or "R3630G5"
    except EOFError:
        query = "R3630G5"
    if not query:
        print("空 query,退出。")
        return
    qv = _embed([query])[0]
    print(f"top-3 hits (query={query!r}):")
    for hit in search(col, qv, k=3):
        snippet = hit["text"][:60].replace("\n", " ")
        print(f"  [{hit['source']}#{hit['page']}] score={hit['score']:.3f} | {snippet}")


if __name__ == "__main__":
    main()