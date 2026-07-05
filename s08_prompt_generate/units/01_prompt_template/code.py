#!/usr/bin/env python3
"""
s08 / unit 01 — Prompt 模板 + LLM 调用 + 引用解析:把 s07 精排后的 top-3
hits 塞进 `<context>...</context>` 块、调 OpenAI 兼容接口生成答案 + 拒答
兜底,返回 `{text, citations}`。

本单元 self-contained:内联 chroma + s04 unit 01 本地 BGE embed + s06 unit 02
混合召回 + s07 unit 01 cross-encoder 精排(全部 importlib 加载,不走
chapter-root)。LLM 调用走 OpenAI SDK(已在 requirements.txt),无
`LLM_API_KEY` 时优雅降级,返回 `[skipped: LLM_API_KEY not set]`。

运行: python s08_prompt_generate/units/01_prompt_template/code.py
需要: 跑通 s07; .env 里有 LLM_API_KEY(可选,无也能跑)
"""
import importlib.util
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path(__file__).resolve().parents[3]
SAMPLES = WORKDIR / "samples"
DB_DIR = WORKDIR / "s05_vector_index" / "_chroma"
COLLECTION_NAME = "docs"


# ---------- Prompt 模板 ----------
# 硬约束 4 条:
#   1. 只能依据 <context> 回答(防止幻觉 + prompt 注入)
#   2. 没有就拒答(防止硬凑)
#   3. 引用用 [i] 角标对资料编号(可追溯)
#   4. 中文 + 简洁直接(防输出风格漂移)
# ragflow 走得更远:独立的 sufficiency_check / citation_prompt 多 pass,
# 见 ragflow_notes/prompt_templates.md。
PROMPT = """你是严谨的问答助手，只能依据 <context> 里的资料回答问题。
- 如果资料中没有直接回答问题的内容，仅回答"我不知道"，不要附加任何引用或相关但不直接回答问题的信息。
- 引用时用 [1]、[2] 这样的角标对应资料编号。
- 回答用中文，简洁直接。

<context>
{context}
</context>

问题: {question}
"""


def _format_context(hits: list[dict]) -> str:
    """把 hits 渲染成 `[i] (source#page) text` 块,跟 prompt 里 [1][2] 一一对应。"""
    blocks = []
    for i, h in enumerate(hits, start=1):
        loc = f"{h['source']}#{h.get('page', '?')}"
        blocks.append(f"[{i}] ({loc}) {h['text']}")
    return "\n\n".join(blocks)


def answer(question: str, hits: list[dict]) -> dict:
    """调用 OpenAI 兼容 LLM 生成答案,返回 `{text, citations}`。

    无 `LLM_API_KEY` 时降级:返回带 `[skipped: LLM_API_KEY not set]` 的 text,
    citations 仍然回填(让调用方至少能拿到命中的 source/page)。
    """
    citations = [
        {"index": i, "source": h["source"], "page": h.get("page")}
        for i, h in enumerate(hits, 1)
    ]
    if not os.environ.get("LLM_API_KEY"):
        return {
            "text": "[skipped: LLM_API_KEY not set]",
            "citations": citations,
        }

    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    prompt = PROMPT.format(context=_format_context(hits), question=question)
    resp = client.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content
    # 去掉 <think>...</think> 块(MiniMax / DeepSeek R1 类推理模型的中间步骤)。
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return {"text": text, "citations": citations}


# ---------- 入口(self-contained,内联 chroma + embed + hybrid + rerank) ----------


def main() -> None:
    # 复用 s06 unit 01 的 BM25/tokenize/chunker/loader(importlib 加载)。
    _S06_U01 = WORKDIR / "s06_retrieval" / "units" / "01_bm25" / "code.py"
    spec = importlib.util.spec_from_file_location("s06_unit01_bm25", _S06_U01)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    BM25 = m.BM25
    _load_chunks = m._load_chunks

    # 内联 s06 unit 02 的 hybrid_topk 公式 + s07 unit 01 的 cross-encoder 精排。
    from functools import lru_cache

    @lru_cache(maxsize=1)
    def _embed_model():
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5"))

    def _embed(texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in _embed_model().encode(texts, normalize_embeddings=True)]

    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    def _hybrid_topk(docs, query, query_vec, dense_score_fn, k, alpha):
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
        return [{**h, "rerank_score": float(s)} for h, s in scored[:top_k]]

    chunks = _load_chunks()
    print(f"loaded {len(chunks)} chunks from samples/")

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

    question = input("问: ").strip() or "内存"
    raw = col.get(include=["embeddings", "metadatas", "documents"])
    vec_by_id = {cid: emb for cid, emb in zip(raw["ids"], raw["embeddings"])}
    qv = _embed([question])[0]

    def _dense_score(chunk):
        return _cosine(qv, vec_by_id[chunk["chunk_id"]])

    candidates = _hybrid_topk(chunks, question, qv, _dense_score, k=10, alpha=0.5)
    top = rerank(question, candidates, top_k=3)

    print("\n--- top-3 after rerank ---")
    for i, h in enumerate(top, start=1):
        page = h["page"] if h["page"] is not None else "-"
        snippet = h["text"][:50].replace("\n", " ")
        print(f"  #{i} [{h['source']}#{page}] rerank={h['rerank_score']:.3f} | {snippet}")

    result = answer(question, top)
    print(f"\nA: {result['text']}")
    print("引用:", result["citations"])


if __name__ == "__main__":
    main()