#!/usr/bin/env python3
"""s06 检索 — 聚合入口。实际逻辑在 units/02_hybrid_fusion/code.py。"""
import importlib.util, sys
from pathlib import Path
_UNIT = Path(__file__).resolve().parent / "units" / "02_hybrid_fusion" / "code.py"
_spec = importlib.util.spec_from_file_location("s06_unit02_hybrid_fusion", _UNIT)
_mod = importlib.util.module_from_spec(_spec); sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod); main = _mod.main


def hybrid_search(col, query: str, query_vec: list[float], k: int = 10, alpha: float = 0.95) -> list[dict]:
    """Chroma 召回 top-50 → hybrid_topk 融合 → top-k。

    app.py 的旧接口 (commit 36ceb69) 期望 ``hybrid_search(col, query, qv, k=10)`` 直接对 chroma 集合做混合召回;
    实际 hybrid_topk 需要预取的 docs + 一个 dense_score_fn 注入。本 wrapper 把 chroma 拉取 + dense 分映射到
    一起,保持 app.py 的旧接口形状不变 (Phase C3' fix)。
    """
    # 单元目录以数字开头 (02_hybrid_fusion),普通 `from x.y.02_z import ...` 会被 Python 解析成十进制字面量
    # → 改用 importlib 动态加载,跟 chapter root 的 importlib 模式保持一致。
    import importlib.util as _ilu
    _path = Path(__file__).resolve().parent / "units" / "02_hybrid_fusion" / "code.py"
    _spec = _ilu.spec_from_file_location("s06_unit02_hybrid_fusion_inline", _path)
    _unit = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_unit)
    raw = col.query(query_embeddings=[query_vec], n_results=50,
                    include=["documents", "metadatas", "distances"])
    docs, distances = [], raw["distances"][0]
    for text, meta in zip(raw["documents"][0], raw["metadatas"][0]):
        docs.append({"text": text, "source": meta.get("source"),
                     "page": meta.get("page"), "chunk_id": meta.get("chunk_id")})
    by_id = {id(d): 1.0 - distances[i] for i, d in enumerate(docs)}
    return _unit.hybrid_topk(docs, query, query_vec, lambda d: by_id[id(d)], k=k, alpha=alpha)


if __name__ == "__main__":
    main()