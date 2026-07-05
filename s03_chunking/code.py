#!/usr/bin/env python3
"""s03 文本分块 — 聚合入口。实际逻辑在 units/01_basic_chunk/code.py。"""
import importlib.util
import sys
from pathlib import Path

_UNIT = Path(__file__).resolve().parent / "units" / "01_basic_chunk" / "code.py"
_spec = importlib.util.spec_from_file_location("s03_unit01_basic_chunk", _UNIT)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
main = _mod.main

if __name__ == "__main__":
    main()