#!/usr/bin/env python3
"""s09 Agent 与工具 — 聚合入口。实际逻辑在 units/02_react_loop/code.py。

章节的核心概念是 ReAct 循环(Thought → Action → Observation),
所以聚合入口委托给 unit 02;unit 01 的"单步工具调用"作为铺垫单独
保留在 `units/01_tool_call/code.py`。

运行: python s09_agent_tools/code.py
需要: 跑通 s08; .env 里有 LLM_API_KEY(可选,无也能跑骨架)
"""
import importlib.util
import sys
from pathlib import Path

_UNIT = Path(__file__).resolve().parent / "units" / "02_react_loop" / "code.py"
_spec = importlib.util.spec_from_file_location("s09_unit02_react_loop", _UNIT)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
main = _mod.main

if __name__ == "__main__":
    main()