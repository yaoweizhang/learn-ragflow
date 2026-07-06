#!/usr/bin/env python3
"""s08 Prompt 与生成 — 聚合入口。实际逻辑在 units/01_prompt_template/code.py。"""
import importlib.util, sys
from pathlib import Path
_UNIT = Path(__file__).resolve().parent / "units" / "01_prompt_template" / "code.py"
_spec = importlib.util.spec_from_file_location("s08_unit01_prompt_template", _UNIT)
_mod = importlib.util.module_from_spec(_spec); sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod); main = _mod.main
answer = _mod.answer  # 供 s12_deployment/app.py 复用 (Phase C3' fix)
if __name__ == "__main__":
    main()