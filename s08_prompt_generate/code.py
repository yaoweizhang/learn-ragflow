#!/usr/bin/env python3
"""
s08 Prompt 与生成 — 把检索结果塞进 prompt，让 LLM 引用 + 拒答。

运行: python s08_prompt_generate/code.py
需要: 跑通 s07；.env 里有 LLM_API_KEY
"""
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)
WORKDIR = Path(__file__).parent.parent


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
    blocks = []
    for i, h in enumerate(hits, start=1):
        loc = f"{h['source']}#{h.get('page', '?')}"
        blocks.append(f"[{i}] ({loc}) {h['text']}")
    return "\n\n".join(blocks)


def answer(question: str, hits: list[dict]) -> dict:
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
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return {"text": text, "citations": [
        {"index": i, "source": h["source"], "page": h.get("page")} for i, h in enumerate(hits, 1)
    ]}


def main() -> None:
    sys.path.insert(0, str(WORKDIR))
    from s04_embedding.code import embed
    from s06_retrieval.code import hybrid_search
    from s07_rerank.code import rerank
    import chromadb
    col = chromadb.PersistentClient(path=str(WORKDIR / "s05_vector_index" / "_chroma")).get_collection("docs")
    question = input("问: ").strip()
    qv = embed([question])[0]
    candidates = hybrid_search(col, question, qv, k=10)
    top = rerank(question, candidates, top_k=3)
    result = answer(question, top)
    print(result["text"])
    print("引用:", result["citations"])


if __name__ == "__main__":
    main()
