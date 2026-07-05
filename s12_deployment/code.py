#!/usr/bin/env python3
"""s12 部署 — 聚合入口。实际逻辑在 units/01_fastapi_docker/code.py。"""
import importlib.util
import sys
from pathlib import Path

_UNIT = Path(__file__).resolve().parent / "units" / "01_fastapi_docker" / "code.py"
_spec = importlib.util.spec_from_file_location("s12_unit01_fastapi_docker", _UNIT)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
main = _mod.main

if __name__ == "__main__":
    main()
