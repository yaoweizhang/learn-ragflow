#!/usr/bin/env python3
"""
s10 / unit 01 — LLM-based 实体/关系抽取:把每个 chunk 喂给 LLM 让它吐
(head, rel, tail) 三元组,合并成 in-memory 图,持久化到 s10_graphrag/_graph.jsonl
(每行一个 triple,JSON 格式)。

本单元 self-contained:内联 pypdf + python-docx + 简单 chunking(按双换行分段),
不走 s02/s03——unit 02 (1 跳查询) 会读 _graph.jsonl 跑离线查询,不再调 LLM。

运行: python s10_graphrag/code_01_extract.py
需要: 跑通 s02 + s03 + .env 里有 LLM_API_KEY
"""
import json
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path(__file__).resolve().parents[1]
SAMPLES = WORKDIR / "samples"
GRAPH_PATH = Path(__file__).resolve().parents[1] / "_graph.jsonl"


# ---------- LLM 抽取 prompt + 解析 ----------

EXTRACT_PROMPT = """从下面这段文字中抽取 (实体, 关系, 实体) 三元组。
输出严格 JSON 数组,每项: {{"head": "...", "rel": "...", "tail": "..."}}。
没有就输出 []。

文字: {text}
JSON:"""


def _llm_json(prompt: str) -> list[dict]:
    """调 OpenAI 兼容 LLM,期望返回 JSON 数组或带 'triples' 键的 dict。
    失败一律返回 []——让上层 build_graph 当成"没抽到"继续,而不是 crash。
    """
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
    # MiniMax-M3 在 message.content 里可能夹 <think>...</think> 推理块;剥掉它再交给 JSON 解析。
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # MiniMax-M3 经常把 JSON 用 markdown 代码块包起来;剥掉 ```json ... ``` 围栏。
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    try:
        obj = json.loads(raw)
        # MiniMax-M3 不一定 honor response_format=json_object;model 可能直接返 list。
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return obj.get("triples", [])
        return []
    except json.JSONDecodeError:
        return []


def extract_triples(text: str) -> list[dict]:
    """对一段文字跑抽取,返回 [{head, rel, tail}, ...]。"""
    return _llm_json(EXTRACT_PROMPT.format(text=text))


def build_graph(triples_list: list[list[dict]]) -> dict:
    """把所有 chunk 的三元组合并成 dict[head] -> set[(rel, tail)]。"""
    graph: dict[str, set[tuple[str, str]]] = {}
    for triples in triples_list:
        for t in triples:
            for node in (t["head"], t["tail"]):
                graph.setdefault(node, set())
            graph[t["head"]].add((t["rel"], t["tail"]))
    return graph


def save_graph(graph: dict, path: Path) -> None:
    """把图持久化成 JSONL(每行 {head, rel, tail}),set 转 list 便于 JSON 序列化。"""
    with path.open("w", encoding="utf-8") as f:
        for head, edges in graph.items():
            for rel, tail in edges:
                f.write(json.dumps({"head": head, "rel": rel, "tail": tail}, ensure_ascii=False) + "\n")


# ---------- 入口(self-contained,内联 pypdf + python-docx + 简单 chunking) ----------


def main() -> None:
    from pypdf import PdfReader
    from docx import Document

    def _pdf(path: Path) -> list[str]:
        out = []
        for p in PdfReader(path).pages:
            t = (p.extract_text() or "").strip()
            if t:
                out.append(t)
        return out

    def _docx(path: Path) -> list[str]:
        return [p.text.strip() for p in Document(path).paragraphs if p.text.strip()]

    def _chunk(paras: list[str], limit: int = 8) -> list[str]:
        # 简化版按双换行分段 + 500 字 cap,够小;跟 s03 / s04 unit 01 同款"小教学版"。
        chunks = []
        for p in paras:
            for piece in re.split(r"\n\s*\n", p):
                piece = piece.strip()
                if not piece:
                    continue
                if len(piece) > 500:
                    piece = piece[:500]
                chunks.append(piece)
        return chunks[:limit]

    paras = _pdf(SAMPLES / "server_whitepaper.pdf") + _docx(SAMPLES / "disclosure.docx")
    chunks = _chunk(paras, limit=8)
    print(f"chunks: {len(chunks)}")

    triples_list = [extract_triples(c) for c in chunks]
    graph = build_graph(triples_list)
    save_graph(graph, GRAPH_PATH)

    n_nodes = len(graph)
    n_edges = sum(len(v) for v in graph.values())
    print(f"图节点数: {n_nodes}, 边数: {n_edges}")
    print(f"持久化: {GRAPH_PATH.relative_to(WORKDIR)}")


if __name__ == "__main__":
    main()