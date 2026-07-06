# s02 Doc Loading — Code Audit

Date: 2026-07-06
Commit: 50de8a09c5618aff478882fd609528d4bc32ff8f (base before changes)

## Criterion 1: README-claimed functions present in code?

Functions the README explicitly mentions/names:

| Function | File | Status |
|---|---|---|
| `load_pdf(path)` | `units/01_basic_load/code.py` | OK present |
| `load_docx(path)` | `units/01_basic_load/code.py` | OK present |
| `main()` (unit 01) | `units/01_basic_load/code.py` | OK present |
| `show_pdf_failure()` | `units/02_failure_modes/code.py` | OK present |
| `show_docx_table_loss()` | `units/02_failure_modes/code.py` | OK present |
| `main()` (unit 02) | `units/02_failure_modes/code.py` | OK present |

All README-mentioned functions are present. The README also references `PdfReader` (imported in unit 01) and `Document` (imported in unit 01, lazily re-imported in unit 02 inside `show_docx_table_loss` to read `Document.tables`) — both imports verified.

**Tier: 对齐**

## Criterion 2: Code's main functions explained in README?

All non-trivial functions documented in §3.3 "核心函数一览" table — name + file + input + output + one-line purpose. Per ambiguity resolution, full re-documentation per function not required; the table satisfies "1-2 lines each".

| Function | Documented? |
|---|---|
| `load_pdf` | OK |
| `load_docx` | OK |
| `main` (unit 01) | OK |
| `show_pdf_failure` | OK |
| `show_docx_table_loss` | OK |
| `main` (unit 02) | OK |

**Tier: 对齐**

## Criterion 3: README sample outputs match live run?

Ran both units. Output snippets from `unit 01` and `unit 02` in the README (§3.4) were cross-checked against the actual runs on the included `samples/`.

Last 3 lines of `python s02_doc_loading/units/01_basic_load/code.py`:

```
一、产品概述
紫光恒越 R3630 G5 是面向企业核心业务、AI 推理与虚拟化负载设计
DOCX 第 1 段前 100 字: 青蓝科技股份有限公司
```

PDF 段落数 = 4, DOCX 段落数 = 27 (matches README). Aggregate entry `python s02_doc_loading/code.py` exits 0 and produces the same output as `unit 01` (it imports unit 01 and calls its `main`).

Last 3 lines of `python s02_doc_loading/units/02_failure_modes/code.py`:

```
[DOCX] paragraphs(非空)=27, tables=3, 表格内总字符=572
  → unit 01 的 load_docx 只读 paragraphs，丢失 572 字符（3 张表）
→ ragflow 的解法: deepdoc/parser/pdf_parser.py 用 XGBoost 版面分析;
```

Both run successfully (exit 0). Numbers in the README's snippet table are consistent with the live output (paragraphs(非空)=27, tables=3, 表格内总字符=572).

**Tier: 对齐**

## Criterion 4: Dead code / orphan import?

All imports grep'd across all 3 code files:

| File | Import | Used? |
|---|---|---|
| `units/01_basic_load/code.py` | `from pathlib import Path` | OK (used for `Path(__file__).resolve().parents[3]`, `SAMPLES`) |
| `units/01_basic_load/code.py` | `from pypdf import PdfReader` | OK (used in `load_pdf`) |
| `units/01_basic_load/code.py` | `from docx import Document` | OK (used in `load_docx`) |
| `code.py` (aggregate) | `import importlib.util` | OK (loads unit 01 module) |
| `code.py` (aggregate) | `import sys` | OK (registers module in sys.modules) |
| `code.py` (aggregate) | `from pathlib import Path` | OK (used to build `_UNIT_PATH`) |
| `units/02_failure_modes/code.py` | `import importlib.util` | OK (loads unit 01) |
| `units/02_failure_modes/code.py` | `import sys` | OK (sys.modules registration) |
| `units/02_failure_modes/code.py` | `from pathlib import Path` | OK (used for paths) |
| `units/02_failure_modes/code.py` | `from docx import Document` (lazy inside `show_docx_table_loss`) | OK (used to read tables) |

No orphan imports, no unused variables, no dead branches. The `importlib` indirection in `s02/code.py` and `unit 02/code.py` is intentional (directory starts with digit, so plain `import` would SyntaxError).

**Tier: 对齐**

## Summary

No small fixes needed (no `小修` triggered). No large fixes needed (no `大修` triggered). All 4 criteria are `对齐` — README rewrite absorbed previous minor inconsistencies (e.g. aggregate entry vs unit 01 print format) without code changes.

0 small fixes applied.
0 big fixes needing user sign-off.

## Big fixes needing user sign-off

None.
