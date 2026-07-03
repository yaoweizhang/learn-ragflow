#!/usr/bin/env python3
"""
s10 GraphRAG — 用 LLM 抽实体关系，建一个内存里的图，演示 1 跳查询。

运行: python s10_graphrag/code.py
需要: 跑通 s02 + s03 + .env 里有 LLM_API_KEY
"""
import json
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)
WORKDIR = Path(__file__).parent.parent
SAMPLES = WORKDIR / "samples"


EXTRACT_PROMPT = """从下面这段文字中抽取 (实体, 关系, 实体) 三元组。
输出严格 JSON 数组，每项: {{"head": "...", "rel": "...", "tail": "..."}}。
没有就输出 []。

文字: {text}
JSON:"""


def _llm_json(prompt: str) -> list[dict]:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["LLM_API_KEY"],
                    base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"))
    resp = client.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    # MiniMax-M3 在 message.content 里可能夹 <think>...</think> 推理块；剥掉它再交给 JSON 解析。
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # MiniMax-M3 经常把 JSON 用 markdown 代码块包起来；剥掉 ```json ... ``` 围栏。
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    try:
        obj = json.loads(raw)
        # MiniMax-M3 不一定 honor response_format=json_object；model 可能直接返 list。
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return obj.get("triples", [])
        return []
    except json.JSONDecodeError:
        return []


def extract_triples(text: str) -> list[dict]:
    return _llm_json(EXTRACT_PROMPT.format(text=text))


def build_graph(triples_list: list[list[dict]]) -> dict:
    graph: dict[str, set[tuple[str, str]]] = {}
    for triples in triples_list:
        for t in triples:
            for node in (t["head"], t["tail"]):
                graph.setdefault(node, set())
            graph[t["head"]].add((t["rel"], t["tail"]))
    return graph


def query_graph(graph: dict, entity: str) -> list[tuple[str, str]]:
    return sorted(graph.get(entity, set()))


def main() -> None:
    sys.path.insert(0, str(WORKDIR))
    from s02_doc_loading.code import load_pdf, load_docx
    from s03_chunking.code import chunk_by_paragraph
    docs = load_pdf(SAMPLES / "server_whitepaper.pdf") + load_docx(SAMPLES / "disclosure.docx")
    chunks = chunk_by_paragraph(docs)[:8]
    triples_list = [extract_triples(c["text"]) for c in chunks]
    graph = build_graph(triples_list)
    print(f"图节点数: {len(graph)}, 边数: {sum(len(v) for v in graph.values())}")
    entity = input("查哪个实体: ").strip()
    for rel, tail in query_graph(graph, entity):
        print(f"  {entity} --{rel}--> {tail}")


if __name__ == "__main__":
    main()