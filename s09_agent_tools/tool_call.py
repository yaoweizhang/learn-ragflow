#!/usr/bin/env python3
"""
s09 / unit 01 — 工具调用（单步）：把 2 个工具（retrieve / finish）写进
system prompt,调一次 LLM,正则抠 `Action / ActionInput`,执行工具,展示
Observation。本单元只走 1 轮——证明 LLM **能自己决定** 选哪个工具,
完整循环留给 unit 02。

self-contained: 内联 chroma + s04 unit 01 本地 BGE embed + s06 unit 02
混合召回 + s07 unit 01 cross-encoder 精排(全部 importlib 加载,不走
chapter-root)。LLM 走 OpenAI SDK,无 `LLM_API_KEY` 时打印示例响应让流程
也能复现。

运行: python s09_agent_tools/tool_call.py
需要: 跑通 s08; .env 里有 LLM_API_KEY(可选,无也能跑)
"""
import importlib.util
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path(__file__).resolve().parents[1]
DB_DIR = WORKDIR / "s05_vector_index" / "_chroma"
COLLECTION_NAME = "docs"


# ---------- 工具描述 ----------
# 关键:把"有什么工具 + 怎么回答"全写在 system prompt 里——LLM 没看到
# 函数签名,只能按这段文字的格式输出。这是 MVP 的硬约束,RAGFlow 用
# OpenAI/Anthropic 的 `tool_calls` 字段把这一步结构化了
# (见 agent/component/agent_with_tools.py)。
TOOLS_DESC = """你可以用以下工具:
1. retrieve(query: str) — 从文档库检索相关段落
2. finish(answer: str) — 给出最终答案

按以下格式回答(每轮一步):
Thought: <你的思考>
Action: <retrieve 或 finish>
ActionInput: <JSON 字符串>
"""


# ---------- LLM 客户端 ----------
def _llm(messages: list[dict]) -> str:
    """OpenAI 兼容接口 + 剥 <think>...</think> 推理块(MiniMax / DeepSeek R1)。"""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    resp = client.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=0,
    )
    raw = resp.choices[0].message.content
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


# ---------- 检索工具(内联 s05-s07 管线,跟 s08 unit 01 同款) ----------
# 复用 s06 unit 01 的 BM25/tokenize/chunker/loader(importlib 加载)。
_S06_U01 = WORKDIR / "s06_retrieval" / "bm25.py"
_spec = importlib.util.spec_from_file_location("s06_unit01_bm25", _S06_U01)
_m = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _m
_spec.loader.exec_module(_m)
BM25 = _m.BM25
_load_chunks = _m._load_chunks


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


def _retrieve(query: str) -> str:
    """embed → hybrid_search → rerank 拼成可读字符串。self-contained。"""
    chunks = _load_chunks()
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
    qv = _embed([query])[0]
    raw = col.get(include=["embeddings", "metadatas", "documents"])
    vec_by_id = {cid: emb for cid, emb in zip(raw["ids"], raw["embeddings"])}
    def _dense_score(chunk):
        return _cosine(qv, vec_by_id[chunk["chunk_id"]])
    candidates = _hybrid_topk(chunks, query, qv, _dense_score, k=10, alpha=0.5)
    top = rerank(query, candidates, top_k=3)
    if not top:
        return "(检索无命中)"
    return "\n".join(f"- ({h['source']}#{h.get('page', '?')}) {h['text']}" for h in top)


# ---------- 单步工具调用 ----------
def single_shot(question: str) -> dict:
    """调一次 LLM,解析 Action/ActionInput,执行工具,返回 trace。

    返回 dict 含: `text`(LLM 原话)、`action`(工具名)、`payload`(解析的 JSON)、
    `observation`(工具返回值或提示)。
    """
    import json
    messages = [
        {"role": "system", "content": TOOLS_DESC},
        {"role": "user", "content": question},
    ]
    text = _llm(messages)
    # 兼容多种写法:ActionInput 在新行 / 与 Action 同行 / 带 markdown ```json 围栏
    m = re.search(r"Action:\s*(\w+)\b\s*ActionInput:\s*(.+)", text, re.DOTALL)
    if not m:
        return {"text": text, "action": None, "payload": None,
                "observation": "(no Action line parsed — LLM 直接答了?)"}
    action, raw = m.group(1), m.group(2).strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"text": text, "action": action, "payload": None,
                "observation": f"(JSON 解析失败: {raw[:120]})"}
    if action == "finish":
        obs = payload.get("answer", "(finish 但没给 answer)")
    elif action == "retrieve":
        obs = _retrieve(payload.get("query", ""))
    else:
        obs = f"Unknown action: {action}"
    return {"text": text, "action": action, "payload": payload, "observation": obs}


# ---------- 入口 ----------
def main() -> None:
    try:
        question = input("问: ").strip() or "R3630 G5 的内存插槽数量"
    except EOFError:
        question = "R3630 G5 的内存插槽数量"  # 同 s07/s08/s10/s11:管道/EOF 时回落到默认问题
    print(f"\n[Q] {question}")

    if not os.environ.get("LLM_API_KEY"):
        # 无 key 演示:展示 LLM 假设选了 retrieve 时的 trace 形状,不真跑检索管线
        print("[skipped: LLM_API_KEY not set] — 演示假设 LLM 选了 retrieve:\n")
        fake_text = (
            'Thought: 用户问的是 R3630 G5 的内存插槽数量。\n'
            'Action: retrieve\n'
            f'ActionInput: {{"query": "{question}"}}'
        )
        trace = {
            "text": fake_text,
            "action": "retrieve",
            "payload": {"query": question},
            "observation": "(无 LLM_API_KEY;真实 retrieval 需 chroma + embed,见 _retrieve())",
        }
    else:
        trace = single_shot(question)

    print(f"[LLM raw]\n{trace['text']}\n")
    print(f"[Parsed action] {trace['action']}")
    print(f"[Parsed payload] {trace['payload']}")
    print(f"\n[Observation]\n{trace['observation']}")


if __name__ == "__main__":
    main()