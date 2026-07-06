#!/usr/bin/env python3
"""s04 Embedding — 聚合入口。实际逻辑在 units/01_local_bge/code.py。"""
import importlib.util
import sys
from pathlib import Path

_UNIT = Path(__file__).resolve().parent / "units" / "01_local_bge" / "code.py"
_spec = importlib.util.spec_from_file_location("s04_unit01_local_bge", _UNIT)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
main = _mod.main
embed = _mod.embed_local  # 供 s12_deployment/app.py 复用 (Phase C3' fix)

if __name__ == "__main__":
    main()
