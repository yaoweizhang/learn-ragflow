#!/usr/bin/env python3
"""
s02 文档加载 — 聚合入口。实际逻辑在 units/01_basic_load/code.py。
本文件保留旧 `python s02_doc_loading/code.py` 启动方式。

运行: python s02_doc_loading/code.py
"""
import importlib.util
import sys
from pathlib import Path

_UNIT_PATH = Path(__file__).resolve().parent / "units" / "01_basic_load" / "code.py"
_spec = importlib.util.spec_from_file_location("s02_unit01_basic_load", _UNIT_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
main = _mod.main

if __name__ == "__main__":
    main()
