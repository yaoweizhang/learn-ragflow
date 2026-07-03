#!/usr/bin/env python3
"""
s06 检索 — 向量 + BM25 加权混合召回。

运行: python s06_retrieval/code.py
需要: 跑通 s05（_chroma 目录存在）
"""
import math
import re
import sys
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path(__file__).resolve().parent.parent
SAMPLES = WORKDIR / "samples"


def tokenize(text: str) -> list[str]:
    """中文按 1-2 字滑动窗口 + 英文单词 + 数字，整体 lower-case。

    对纯中文（无空格）友好，对中英混排也保留英文 token。
    """
    text = text.lower()
    # 抽取英文 / 数字 token
    en_tokens = re.findall(r"[a-z0-9]+", text)
    # 中文按 1-2 字滑动窗口成 token，BM25 才有 df 命中
    cn_chars = re.findall(r"[一-鿿]", text)
    cn_tokens = []
    for i, c in enumerate(cn_chars):
        cn_tokens.append(c)
        if i + 1 < len(cn_chars):
            cn_tokens.append(c + cn_chars[i + 1])
    return en_tokens + cn_tokens


class BM25:
    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.avgdl = sum(len(tokenize(d["text"])) for d in docs) / max(self.N, 1)
        self.df = Counter()
        self.tf = []
        for d in docs:
            tf = Counter(tokenize(d["text"]))
            self.tf.append(tf)
            for term in tf:
                self.df[term] += 1

    def score(self, query: str) -> list[float]:
        qterms = tokenize(query)
        scores = [0.0] * self.N
        for q in qterms:
            if q not in self.df:
                continue
            idf = math.log((self.N - self.df[q] + 0.5) / (self.df[q] + 0.5) + 1)
            for i, tf in enumerate(self.tf):
                dl = sum(tf.values())
                denom = tf[q] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * tf[q] * (self.k1 + 1) / denom
        return scores


def hybrid_search(col, query: str, query_vec: list[float], k: int = 5, alpha: float = 0.5) -> list[dict]:
    # vector part
    res = col.query(query_embeddings=[query_vec], n_results=20)
    vec_scores = {res["ids"][0][i]: 1 - res["distances"][0][i] for i in range(len(res["ids"][0]))}
    # bm25 part
    all_docs = col.get(include=["documents", "metadatas"])
    docs = [{"text": all_docs["documents"][i], "source": all_docs["metadatas"][i]["source"],
             "page": all_docs["metadatas"][i].get("page"), "id": all_docs["ids"][i]}
            for i in range(len(all_docs["ids"]))]
    bm = BM25(docs)
    bm_scores = bm.score(query)
    bm_norm = {docs[i]["id"]: bm_scores[i] for i in range(len(docs))}
    # normalize each to [0, 1]
    v_max = max(vec_scores.values()) if vec_scores else 1
    b_max = max(bm_norm.values()) if any(bm_norm.values()) else 1
    combined = []
    for d in docs:
        v = vec_scores.get(d["id"], 0) / v_max
        b = bm_norm.get(d["id"], 0) / b_max
        combined.append((d, alpha * v + (1 - alpha) * b))
    combined.sort(key=lambda x: -x[1])
    return [{"text": d["text"], "source": d["source"], "page": d["page"], "score": s} for d, s in combined[:k]]


def main() -> None:
    sys.path.insert(0, str(WORKDIR))
    from s05_vector_index.code import search as vec_search
    from s04_embedding.code import embed
    import chromadb
    col = chromadb.PersistentClient(path=str(WORKDIR / "s05_vector_index" / "_chroma")).get_collection("docs")
    query = input("问: ").strip()
    qv = embed([query])[0]
    for hit in hybrid_search(col, query, qv, k=3):
        print(f"[{hit['source']}#{hit['page']}] score={hit['score']:.3f} | {hit['text'][:60]}")


if __name__ == "__main__":
    main()