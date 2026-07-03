#!/usr/bin/env python3
"""
s09 Agent 与工具 — ReAct 风格循环：模型决定调 retrieve 还是直接答。

运行: python s09_agent_tools/code.py
需要: 跑通 s08；.env 里有 LLM_API_KEY
"""
import json
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)
WORKDIR = Path(__file__).parent.parent


TOOLS_DESC = """你可以用以下工具:
1. retrieve(query: str) — 从文档库检索相关段落
2. finish(answer: str) — 给出最终答案

按以下格式回答（每轮一步）:
Thought: <你的思考>
Action: <retrieve 或 finish>
ActionInput: <JSON 字符串>
"""


def _llm(messages: list[dict]) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["LLM_API_KEY"],
                    base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"))
    resp = client.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=0,
    )
    raw = resp.choices[0].message.content
    # MiniMax-M3 在 system 消息为空时会在 message.content 里夹 <think>...</think>
    # 推理块；Action 行会被埋在里头，regex 抓不到。剥掉它再交给 run_agent 解析。
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def _retrieve(query: str) -> str:
    sys.path.insert(0, str(WORKDIR))
    from s04_embedding.code import embed
    from s06_retrieval.code import hybrid_search
    from s07_rerank.code import rerank
    import chromadb
    col = chromadb.PersistentClient(path=str(WORKDIR / "s05_vector_index" / "_chroma")).get_collection("docs")
    qv = embed([query])[0]
    candidates = hybrid_search(col, query, qv, k=10)
    top = rerank(query, candidates, top_k=3)
    return "\n".join(f"- ({h['source']}#{h.get('page', '?')}) {h['text']}" for h in top)


def run_agent(question: str, max_steps: int = 5) -> str:
    messages = [{"role": "system", "content": TOOLS_DESC},
                {"role": "user", "content": question}]
    for _ in range(max_steps):
        text = _llm(messages)
        messages.append({"role": "assistant", "content": text})
        m = re.search(r"Action:\s*(\w+)\s*\nActionInput:\s*(.+)", text, re.DOTALL)
        if not m:
            return text
        action, raw = m.group(1), m.group(2).strip()
        if action == "finish":
            return json.loads(raw)["answer"]
        if action == "retrieve":
            q = json.loads(raw)["query"]
            obs = _retrieve(q)
        else:
            obs = f"Unknown action: {action}"
        messages.append({"role": "user", "content": f"Observation: {obs}"})
    return "Max steps reached."


def main() -> None:
    print(run_agent(input("问: ").strip()))


if __name__ == "__main__":
    main()
