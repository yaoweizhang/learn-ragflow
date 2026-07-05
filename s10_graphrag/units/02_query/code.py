#!/usr/bin/env python3
"""
s10 / unit 02 — 1 跳图查询:加载 s10_graphrag/_graph.jsonl,在内存里跑 1 跳邻居查询。

本单元 self-contained:不调 LLM,不依赖 s02/s03/s04——只要 unit 01 跑完落盘过 JSONL,
本单元就能离线查询(适合跑多次、调试 prompt、调试实体名)。

运行: python s10_graphrag/units/02_query/code.py
需要: 先跑一次 s10_graphrag/units/01_extract/code.py(生成 _graph.jsonl)
"""
import sys
from pathlib import Path

WORKDIR = Path(__file__).resolve().parents[3]
GRAPH_PATH = Path(__file__).resolve().parents[2] / "_graph.jsonl"


def load_graph(path: Path) -> dict[str, set[tuple[str, str]]]:
    """从 JSONL 读回 dict[head] -> set[(rel, tail)]。缺失 / 空文件返空图。"""
    graph: dict[str, set[tuple[str, str]]] = {}
    if not path.exists():
        return graph
    import json
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            graph.setdefault(t["head"], set()).add((t["rel"], t["tail"]))
            # 顺手把 tail 也注册成节点(就算它没出边,也让它"存在",便于 query 命中)。
            graph.setdefault(t["tail"], set())
    return graph


def query_graph(graph: dict, entity: str) -> list[tuple[str, str]]:
    """从 entity 出发取 1 跳邻居,按 (rel, tail) 字母序排,便于对照。"""
    return sorted(graph.get(entity, set()))


# ---------- 入口(self-contained) ----------


def main() -> None:
    graph = load_graph(GRAPH_PATH)
    if not graph:
        print(f"图为空或缺失: {GRAPH_PATH}")
        print("请先跑: python s10_graphrag/units/01_extract/code.py")
        return

    n_nodes = len(graph)
    n_edges = sum(len(v) for v in graph.values())
    print(f"图节点数: {n_nodes}, 边数: {n_edges}")

    while True:
        entity = input("查哪个实体 (回车退出): ").strip()
        if not entity:
            break
        neighbors = query_graph(graph, entity)
        if not neighbors:
            print(f"  (无结果——'{entity}' 不在图中或没有出边)")
            continue
        for rel, tail in neighbors:
            print(f"  {entity} --{rel}--> {tail}")


if __name__ == "__main__":
    main()