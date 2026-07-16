#!/usr/bin/env python3
"""
s06 / unit 01 — BM25 词法召回：hand-written BM25 + 中英分词 + 内存 top-k。

本单元 self-contained:不跨章节 import,也不 import unit 02;内联 pypdf +
python-docx + 500 字符 cap 中英句界切来构造 chunk 集合,在内存里跑 BM25
打分并取 top-k。

unit 02 (hybrid_fusion) 会复用这里的 BM25/tokenize,叠加 dense cosine
做加权融合。

运行: python s06_retrieval/bm25.py
需要: pip install pypdf python-docx;samples/{server_whitepaper.pdf,disclosure.docx}
"""
import math
import re
from collections import Counter
from pathlib import Path


WORKDIR = Path(__file__).resolve().parents[1]
SAMPLES = WORKDIR / "samples"


# ---------- 分词器 ----------

def tokenize(text: str) -> list[str]:
    """中文按 1-2 字滑动窗口 + 英文/数字单词,整体 lower-case。

    对纯中文段落(无空格)也能被 BM25 命中 df,对中英混排也保留英文 token。
    """
    text = text.lower()
    en_tokens = re.findall(r"[a-z0-9]+", text)
    cn_chars = re.findall(r"[一-鿿]", text)
    cn_tokens = []
    for i, c in enumerate(cn_chars):
        cn_tokens.append(c)
        if i + 1 < len(cn_chars):
            cn_tokens.append(c + cn_chars[i + 1])
    return en_tokens + cn_tokens


# ---------- 切块器 (inline s03 的 500 字符 cap + 中英句界切) ----------

def _split_long(text: str, max_chars: int) -> list[str]:
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
    out = []
    for doc in docs:
        if len(doc["text"]) <= max_chars:
            out.append({**doc, "chunk_id": f"{doc['source']}#{doc.get('page', 0)}#p{len(out)}"})
        else:
            for piece in _split_long(doc["text"], max_chars):
                out.append({**doc, "text": piece,
                            "chunk_id": f"{doc['source']}#{doc.get('page', 0)}#p{len(out)}"})
    return out


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


def _load_chunks() -> list[dict]:
    docs = _pdf(SAMPLES / "server_whitepaper.pdf") + _docx(SAMPLES / "disclosure.docx")
    return _chunk_by_paragraph(docs)


# ---------- BM25 核心 ----------

class BM25:
    """Robertson BM25: token-level TF-IDF with TF saturation (k1) + length norm (b)。

    score(q, d) = sum_{t in q} IDF(t) * tf(t,d) * (k1+1) /
                  (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
    """

    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.avgdl = sum(len(tokenize(d["text"])) for d in docs) / max(self.N, 1)
        self.df: Counter = Counter()
        self.tf: list[Counter] = []
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


def bm25_topk(docs: list[dict], query: str, k: int = 5) -> list[dict]:
    """对 docs 跑 BM25,返回按 BM25 分降序的 top-k 命中。"""
    bm = BM25(docs)
    scores = bm.score(query)
    order = sorted(range(len(docs)), key=lambda i: -scores[i])
    return [
        {"text": docs[i]["text"], "source": docs[i]["source"],
         "page": docs[i].get("page"), "chunk_id": docs[i].get("chunk_id"),
         "bm25": scores[i]}
        for i in order[:k] if scores[i] > 0
    ]


# ---------- 入口 ----------

def main() -> None:
    chunks = _load_chunks()
    print(f"loaded {len(chunks)} chunks from samples/")
    query = "应收账款 计提"
    print(f"query={query!r} → BM25 top-5:")
    for hit in bm25_topk(chunks, query, k=5):
        snippet = hit["text"][:60].replace("\n", " ")
        page = hit["page"] if hit["page"] is not None else "-"
        print(f"  [{hit['source']}#{page}] bm25={hit['bm25']:.3f} | {snippet}")


if __name__ == "__main__":
    main()