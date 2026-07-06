# s03 Text Chunking — Code Audit

Date: 2026-07-06
Commit: c79f76b (base) → see Phase A Task 2 report for final hash
Auditor: Phase A Task 2 (12-step pattern)

## Criterion 1: README-claimed functions present in code?

Functions named in §3.3 of the new README:

| Function | File | Present? |
|---|---|---|
| `split_long_paragraph(text, max_chars)` | `units/01_basic_chunk/code.py` | ✓ (line 36) |
| `chunk_by_paragraph(docs, max_chars=500)` | `units/01_basic_chunk/code.py` | ✓ (line 67) |
| `main()` (unit 01) | `units/01_basic_chunk/code.py` | ✓ (line 79) |
| `demo_table_split()` | `units/02_chunk_failures/code.py` | ✓ (line 43) |
| `demo_parent_child()` | `units/02_chunk_failures/code.py` | ✓ (line 65) |
| `demo_cross_ref()` | `units/02_chunk_failures/code.py` | ✓ (line 98) |
| `main()` (unit 02) | `units/02_chunk_failures/code.py` | ✓ (line 123) |

All 7 functions accounted for. **Tier: 对齐**

## Criterion 2: Code's main functions explained in README?

| Function | Documented in README? | Where |
|---|---|---|
| `split_long_paragraph` | ✓ | §3.3 row 1, §3.4 ("句界正则") |
| `chunk_by_paragraph` | ✓ | §3.3 row 2, §3.4 ("短段整段保留 / 长段调 split_long_paragraph"), §3.5 schema example |
| `main` (unit 01) | ✓ | §3.3 row 3, §3.6 "跑出来是什么样" |
| `demo_table_split` | ✓ | §3.3 row 4, §3.6 expected output [a] |
| `demo_parent_child` | ✓ | §3.3 row 5, §3.6 expected output [b] |
| `demo_cross_ref` | ✓ | §3.3 row 6, §3.6 expected output [c] |
| `main` (unit 02) | ✓ | §3.3 row 7, §3.6 |
| `_hr` (helper) | not separately listed | only used internally by unit 02 demos; not user-facing; OK to skip |

All user-facing functions documented. The internal `_hr` helper is a print decorator (one screen-width line + title) used only inside unit 02 to make demo boundaries visible — omission from README is intentional (not in user-facing surface). **Tier: 对齐**

## Criterion 3: README sample outputs match live run?

### Unit 01 (`python s03_chunking/units/01_basic_chunk/code.py`)

Last 5 lines of actual run:
```
server_whitepaper.pdf#1#p0 | 紫光恒越 R3630 G5 双路机架式服
务器
产品白皮书  ·  v1.0  ·  仅用于 RAG 教程测试
一、产品
server_whitepaper.pdf#1#p1 | 二、关键特性
```

README §3.6 says: "输入段落 31 → 输出块 34, 最大块长度 452 字符 (cap=500), chunk_id = `server_whitepaper.pdf#1#p0` ..."

Actual run prints exactly `输入段落 31 → 输出块 34` and `最大块长度 452 字符 (cap=500)`. The chunk IDs and first 60 chars of each chunk match verbatim. Exit 0.

### Unit 02 (`python s03_chunking/units/02_chunk_failures/code.py`)

Last 5 lines of actual run:
```
  id=disclosure.docx#None#p30 len=  8 | '第八节 未来展望'

AFTER  (DOCX 原文: '按业务板块划分,2024 年公司收入结构如下:...'):
  '按业务板块划分，2024 年公司收入结构如下：智能算力基础设施业务收入 12.86 亿元，占比 44.7%，同比增长 28.4%；工业互联网平台业务收入 8.43 亿元，占比 29.3%，同比增长 38.1%；...'
  (后续'智能算力 ...' 等具体数字段都另起段落,被 chunker 切成独立 chunk,
   '如下' 这个指代词所在的 chunk 完全不知道它指的是谁)
```

README §3.6 lists demo [a] / [b] / [c] expected outputs that match the actual run (the 562-char 整机规格 table splits into 2 chunks, 节标题 header-only chunks include `disclosure.docx#None#p18 len=11 text='第四节 分季度财务数据'` and similar). Exit 0.

### Aggregate (`python s03_chunking/code.py`)

Equivalent to unit 01 (imports unit 01's `main`). Exit 0, same output. **Tier: 对齐**

## Criterion 4: Dead code / orphan import?

All imports grep'd in 3 code files:

**`s03_chunking/code.py`** (3 imports):
- `importlib.util` — used in `spec_from_file_location` (line 7)
- `sys` — used in `sys.modules` (line 10)
- `pathlib.Path` — used for `_UNIT` path construction (line 7)

**`s03_chunking/units/01_basic_chunk/code.py`** (5 imports):
- `re` — used in `re.split(...)` (line 43)
- `sys` — used in `sys.modules` (line 27)
- `pathlib.Path` — used in path construction
- `importlib.util` — used in spec construction
- (Re-exports from s02 loader via `_mod.load_pdf` / `_mod.load_docx`)

**`s03_chunking/units/02_chunk_failures/code.py`** (4 imports + s03 unit01 + s02 unit01 + python-docx):
- `importlib.util`, `sys`, `pathlib.Path` — all used in spec loading
- `from docx import Document` (inside `demo_parent_child`) — used for table introspection fallback when header chunk has no data body

No orphan imports. No dead code paths.

The `importlib` indirection in all three files is **intentional** — `s03_chunking/code.py` and `s03_chunking/units/02_chunk_failures/code.py` need to import from digit-prefixed dir names (`01_basic_chunk`, `02_chunk_failures`) which Python's normal `import` system can't resolve without making them packages. Same pattern used in s02.

**Tier: 对齐**

## Summary

All 4 criteria pass without code changes — the previous README / code were already in good shape; the rewrite brought structure to 4-段式 and added the "schema 设计取舍" / "实际 schema 形状" / "主流工具速览" / "选型速记" sub-sections for learning value. The 3 unit 02 failure demos (table-split, parent-child, cross-ref) match exactly what `code.py` runs and what the new README documents.

**Small fixes applied**: 0 (no criterion triggered 小修)

**Big fixes needing user sign-off**: 0 (none flagged)

## Notes for downstream tasks

- `chunk_id` format `{source}#{page}#p{n}` — `p{n}` is **output index**, not input index. This is stable across re-runs of the same input set, but **changes if any chunker input is reordered**. Downstream embedding (s04) and indexing (s05) should treat `chunk_id` as opaque identifier, not as a positional key.
- `chunk_by_paragraph` is **not idempotent under reorder** — if you re-run on a shuffled input list, `p{n}` shifts. If your pipeline needs stable IDs across edits, hash the chunk text into the ID (not done here for readability).
- `max_chars=500` is **deliberately over BGE `max_seq_len=512` token** — s04 will demonstrate truncation. Don't "fix" this without coordinating with s04.

## Forbidden content check

`git grep -nE '\[\^[0-9]|RAG 已死|参考文献' s03_chunking/README.md s03_chunking/AUDIT.md` → empty.