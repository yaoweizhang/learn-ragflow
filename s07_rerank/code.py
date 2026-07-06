#!/usr/bin/env python3
"""s07 重排序 — 聚合入口。实际逻辑在 units/01_cross_encoder_rerank/code.py。"""
import importlib.util, sys
from pathlib import Path
_UNIT = Path(__file__).resolve().parent / "units" / "01_cross_encoder_rerank" / "code.py"
_spec = importlib.util.spec_from_file_location("s07_unit01_cross_encoder_rerank", _UNIT)
_mod = importlib.util.module_from_spec(_spec); sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod); main = _mod.main
rerank = _mod.rerank  # 供 s12_deployment/app.py 复用 (Phase C3' fix)
if __name__ == "__main__":
    main()