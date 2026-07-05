# s02-s12 Units Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor every chapter s02–s12 from one flat `code.py` into a `units/01_xxx/` (and optional `units/02_xxx/`) structure, with a 4-段式 README per unit (这是什么 / 跑起来 / 它做对了什么 / 它做错了什么 / 对照 ragflow / 思考题). Chapter-root `code.py` becomes a thin聚合 entry that imports from `units/`. s01 is already done (3 units) and stays as the pilot.

**Architecture:** A fixed meta-pattern is applied per chapter. The hard part is per-chapter unit decomposition (1 or 2 units; "by depth" preferred over "by format"). s02 is the full pilot with detailed code; s03–s12 are mechanical applications of the template.

**Tech Stack:** Python 3.10+, `python-docx`, `pypdf`, `chromadb`, `transformers<5.0`, `xgboost`, `pdfplumber`, `pytesseract`, `fastapi`, `docker-compose`. Embedding model: `BAAI/bge-small-zh-v1.5`. LLM via OpenAI-compatible (`LLM_API_KEY` / `LLM_BASE` / `LLM_MODEL`).

## Global Constraints

These apply to every task below. Carried verbatim from the brainstorm session:

1. **Unit count per chapter: ≤ 2** (s01 is the 3-unit exception; do not touch).
2. **Unit directory shape:** `sXX_topic/units/NN_<verb_noun>/{code.py,README.md}` with `NN` zero-padded to 2 digits.
3. **Each unit's `code.py` must be self-contained runnable:** `python sXX_topic/units/NN_xxx/code.py` works alone, no internal imports from other units in the same chapter unless explicitly stated in that task.
4. **Each unit's `README.md` is 4-段式:**
   - 这是什么 (1-2 paragraphs: what + why now)
   - 跑起来 (copy-pastable `python ...` command + sample output)
   - 它做对了什么 / 它做错了什么 (2 bullets each, the "失败" sets up the next unit or chapter)
   - 对照 ragflow 怎么做的 (link to `ragflow_notes/<topic>.md`, cite 1-2 lines of source)
   - 思考题 (1 question with hints, answer file untouched)
5. **Chapter-root `code.py` becomes a聚合 entry:** imports the unit's main function and re-runs it. Old `python sXX/code.py` invocation still works. **Aggregate file must not duplicate logic** — if unit was 1 unit, the chapter-root code.py is just `from units.01_xxx.code import main; if __name__ == "__main__": main()`.
6. **Chapter-root `README.md` gets a new top section** listing the units (table) + the chapter's `ragflow_notes/` paths. Old sections preserved.
7. **Naming convention for unit directory:** `01_<verb_or_topic_lowercase_snake>` — examples from s01: `01_naive_keyword`, `02_vector_basics`, `03_augmented_llm`; planned for s02: `01_basic_load`, `02_failure_modes`.
8. **Decomposition preference: by depth, not by format.** Don't split "PDF vs DOCX"; split "minimal vs failure modes".
9. **No new dependencies.** Use what's already in `requirements.txt`.
10. **No new tests added.** Verify each unit by running it and visually comparing output to the previous chapter-root `code.py` output.
11. **Frequent commits:** one commit per chapter (1 chapter = 1 commit); commit message `sNN: split into units/<summary>`.
12. **Push to master after each chapter.** Use the established `GITHUB_PAT` retry pattern (up to 5 attempts, expect `https://github.com:443` to be flaky).

## File Structure Overview

```
learn-ragflow/
├── docs/00_introduction/01_what_is_rag.md           (existing, untouched)
├── ragflow_notes/                                    (existing, untouched)
└── s02_doc_loading/                                  (refactored)
    ├── README.md                                     (modified: +units nav + ragflow paths)
    ├── README.en.md                                  (untouched)
    ├── thinking_answers.md                           (untouched)
    ├── code.py                                       (rewritten: 薄聚合)
    └── units/
        ├── 01_basic_load/{code.py,README.md}
        └── 02_failure_modes/{code.py,README.md}

# Same pattern applied to s03-s12 (each chapter):
s03_chunking/
s04_embedding/
s05_vector_index/
s06_retrieval/
s07_rerank/
s08_prompt_generate/
s09_agent_tools/
s10_graphrag/
s11_multimodal/
s12_deployment/    (Dockerfile, docker-compose.yml, app.py untouched)
```

---

## Task 0: Create the 4-段式 README template (shared across chapters)

**Files:**
- Create: `docs/superpowers/templates/unit_readme.md`

**Why:** Avoids drift across 11 chapters × 2 units = 22 README files. Single source of truth for the 4-段式 structure.

- [ ] **Step 1: Write the template file**

```markdown
# <CHAPTER_NAME> / Unit <NN> — <TITLE>

> 由浅入深第 <N> 步：<ONE_LINE_GOAL>.  
> <CONTEXT_LINE_2>

## 这是什么

<PARAGRAPH_1: WHAT_THE_CODE_DOES>

<PARAGRAPH_2: WHY_THIS_UNIT_NOW>

## 跑起来

```bash
python <EXACT_PATH_TO_CODE_PY>
```

输出示例：

```
<SAMPLE_OUTPUT>
```

## 它做对了什么

- <BULLET_1>
- <BULLET_2>

## 它做错了什么

- <BULLET_1_LEADS_TO_NEXT>
- <BULLET_2>

## 对照 ragflow 怎么做的

<RAGFLOW_NOTE_PARAGRAPH>

参考：[`ragflow_notes/<FILE>.md`](../../../../ragflow_notes/<FILE>.md)

## 思考题

<QUESTION>

提示：<HINT>
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/templates/unit_readme.md
git commit -m "s02-s12 plan: add 4-段式 README template"
```

---

## Task 1: Refactor s02 (full pilot — proves the template)

**Files:**
- Create: `s02_doc_loading/units/01_basic_load/code.py`
- Create: `s02_doc_loading/units/01_basic_load/README.md`
- Create: `s02_doc_loading/units/02_failure_modes/code.py`
- Create: `s02_doc_loading/units/02_failure_modes/README.md`
- Modify: `s02_doc_loading/code.py` (rewrite to聚合)
- Modify: `s02_doc_loading/README.md` (add units nav + ragflow paths)

**Unit decomposition (per brainstorm):**
- `01_basic_load`: minimal PDF + DOCX loaders using `pypdf` + `python-docx`, output unified `list[{text, page, source}]`.
- `02_failure_modes`: same loaders run against the real `samples/` data; demonstrates (a) PDF page ordering breakage on the whitepaper, (b) DOCX tables silently dropped. Cites ragflow's `deepdoc/parser/{pdf.py,docx.py}` and the Vision / multi-Parser dispatcher.

- [ ] **Step 1: Create unit 01 `code.py`**

Write `s02_doc_loading/units/01_basic_load/code.py`:

```python
#!/usr/bin/env python3
"""
s02 / unit 01 — 最小可跑加载：pypdf + python-docx → list[{text, page, source}]。

对照 s02 的最小解法 code.py；本单元作为聚合入口的最底层，
unit 02 会复用这里的 load_pdf / load_docx 并展示真实样本上的失败。

运行: python s02_doc_loading/units/01_basic_load/code.py
需要: pip install pypdf python-docx；samples/{server_whitepaper.pdf,disclosure.docx}
"""
from pathlib import Path
from pypdf import PdfReader
from docx import Document

WORKDIR = Path(__file__).resolve().parents[3]
SAMPLES = WORKDIR / "samples"


def load_pdf(path: Path) -> list[dict]:
    out = []
    for i, page in enumerate(PdfReader(path).pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            out.append({"text": text, "page": i, "source": path.name})
    return out


def load_docx(path: Path) -> list[dict]:
    out = []
    for p in Document(path).paragraphs:
        if p.text.strip():
            out.append({"text": p.text, "page": None, "source": path.name})
    return out


def main() -> None:
    pdf = load_pdf(SAMPLES / "server_whitepaper.pdf")
    docx = load_docx(SAMPLES / "disclosure.docx")
    print(f"PDF 段落数: {len(pdf)}, DOCX 段落数: {len(docx)}")
    print("PDF 第 1 段前 100 字:", pdf[0]["text"][:100])
    print("DOCX 第 1 段前 100 字:", docx[0]["text"][:100])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create unit 01 `README.md`**

Use the template. Body (write to `s02_doc_loading/units/01_basic_load/README.md`):

```markdown
# s02 / Unit 01 — 最小可跑加载：PDF + DOCX → 统一 schema

> 由浅入深第 1 步：把 PDF 和 DOCX 都读成 `list[{text, page, source}]`，作为后续章节的输入契约。  
> unit 02 会跑同一套函数到真实样本上，演示哪些情况它会崩。

## 这是什么

1. `load_pdf(path)` — `pypdf.PdfReader` 逐页 `extract_text()`，page 从 1 开始；
2. `load_docx(path)` — `python-docx` 按 `paragraphs` 顺序读，仅保留非空段；
3. 输出统一 `{text, page, source}` schema，page 在 DOCX 时为 `None`。

## 跑起来

\`\`\`bash
python s02_doc_loading/units/01_basic_load/code.py
\`\`\`

输出：

```
PDF 段落数: 4, DOCX 段落数: 27
PDF 第 1 段前 100 字: 紫光恒越 R3630 G5 双路机架式服务器 产品白皮书 ...
DOCX 第 1 段前 100 字: 青蓝科技股份有限公司 ...
```

## 它做对了什么

- **零依赖外的最小化**：`pypdf` + `python-docx` 覆盖最常见两种格式；
- **schema 一致**：PDF 和 DOCX 喂给下游切块器时是同一种形状。

## 它做错了什么

- **DOCX 表格被吃掉**：`Document.paragraphs` 不含 `tables`，所有表内文字都丢；
- **PDF 多栏排版错位**：双栏 PDF 抽出来的文本会把左栏底部接到右栏顶部；
- **扫描件完全没救**：`extract_text()` 对图片型 PDF 返回空字符串。

## 对照 ragflow 怎么做的

RAGFlow 的 `deepdoc/parser/` 是按文件类型 dispatcher：

- `deepdoc/parser/pdf_parser.py` 用 `pdfplumber` + 自训练 XGBoost 模型（30 特征，详见 `ragflow_notes/deepdoc_chunking.md`）做版面分析，能识别多栏 / 表格 / 图；
- `deepdoc/parser/docx_parser.py` 同时处理 `paragraphs` 和 `tables`；
- `deepdoc/parser/utils.py` 里有 VisionParser 兜底扫描件。

参考：[`ragflow_notes/deepdoc_chunking.md`](../../../../ragflow_notes/deepdoc_chunking.md)

## 思考题

**为什么 PDF 输出会有 4 段（4 页）但 DOCX 输出 27 段？这是"段落"的语义不同吗？**

提示：PDF 的"段" = 页（page 是结构边界）；DOCX 的"段" = `\n` 切分的人工段落。统一 schema 时要不要把"页"硬塞给 DOCX？
```

- [ ] **Step 3: Create unit 02 `code.py`**

Write `s02_doc_loading/units/02_failure_modes/code.py`:

```python
#!/usr/bin/env python3
"""
s02 / unit 02 — 失败模式：把 unit 01 的 loader 跑在真实样本上,
展示 (a) PDF 多栏错位 (b) DOCX 表格被吞。

对照 ragflow 的 deepdoc/parser/：用 XGBoost 版面分析 + table-aware 解析修这些问题。

运行: python s02_doc_loading/units/02_failure_modes/code.py
需要: 同 unit 01 + samples/{server_whitepaper.pdf,disclosure.docx}
"""
import sys
from pathlib import Path

# 复用 unit 01 的 loader（章节内 import 是允许的——这就是为什么要拆 unit）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from units.01_basic_load.code import load_pdf, load_docx

WORKDIR = Path(__file__).resolve().parents[3]
SAMPLES = WORKDIR / "samples"


def show_pdf_failure() -> None:
    pdf = load_pdf(SAMPLES / "server_whitepaper.pdf")
    print(f"[PDF] {len(pdf)} 页抽出的段落 (page, len, first 60 字):")
    for seg in pdf:
        print(f"  page={seg['page']:>2} len={len(seg['text']):>4} | {seg['text'][:60].replace(chr(10), ' ')}")


def show_docx_table_loss() -> None:
    from docx import Document
    path = SAMPLES / "disclosure.docx"
    doc = Document(path)
    para_count = sum(1 for p in doc.paragraphs if p.text.strip())
    table_count = len(doc.tables)
    table_text_len = sum(
        len(cell.text)
        for tbl in doc.tables
        for row in tbl.rows
        for cell in row.cells
    )
    print(f"\n[DOCX] paragraphs(非空)={para_count}, tables={table_count}, 表格内总字符={table_text_len}")
    print(f"  → unit 01 的 load_docx 只读 paragraphs，丢失 {table_text_len} 字符（{table_count} 张表）")


def main() -> None:
    show_pdf_failure()
    show_docx_table_loss()
    print("\n→ ragflow 的解法: deepdoc/parser/pdf_parser.py 用 XGBoost 版面分析;")
    print("  deepdoc/parser/docx_parser.py 同时遍历 paragraphs + tables")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create unit 02 `README.md`**

Write to `s02_doc_loading/units/02_failure_modes/README.md`:

```markdown
# s02 / Unit 02 — 真实样本上的失败模式

> 由浅入深第 2 步：unit 01 在 toy 上能跑；放到真实 `samples/` 上会崩在哪？  
> 本单元定位问题 + 引出 ragflow 的工业解法。

## 这是什么

把 unit 01 的 `load_pdf` / `load_docx` 喂给真实样本 (`samples/server_whitepaper.pdf` 4 页 + `samples/disclosure.docx` 27 段)，把"看不见的损失"暴露出来：

1. **PDF 多栏错位**——`pypdf.extract_text()` 在双栏 PDF 上按字符位置扫，左右栏会交错；
2. **DOCX 表格丢失**——`python-docx.Document.paragraphs` 不含 `Document.tables`，表内所有文字静默丢弃。

## 跑起来

\`\`\`bash
python s02_doc_loading/units/02_failure_modes/code.py
\`\`\`

输出片段：

```
[PDF] 4 页抽出的段落 ...
  page= 1 len= 612 | 紫光恒越 R3630 G5 双路机架式服务器 产品白皮书 ...
  page= 2 len=1024 | 产品型号 产品白皮书 文档版本 ...
  ...

[DOCX] paragraphs(非空)=27, tables=3, 表格内总字符=1456
  → unit 01 的 load_docx 只读 paragraphs，丢失 1456 字符（3 张表）
```

## 它做对了什么

- **暴露问题**：列出真实样本上的"丢失字符数"和"页段错位"，给后续章节优化提供量化目标；
- **解法对照**：每个失败都点名 ragflow 的对应模块。

## 它做错了什么

- 暂时什么也没"做对"——它的目的就是展示 unit 01 在 prod 上的失败。下一步要么换 loader (s11 表格抽取) 要么换格式（structured extraction）。

## 对照 ragflow 怎么做的

`deepdoc/parser/` 下的两个核心模块直接对应本单元的两个失败：

- **`deepdoc/parser/pdf_parser.py`** —— `RAGFlowPdfParser` 内部用 `pdfplumber` + XGBoost 30 特征模型识别多栏 / 表格 / 图，详见 [`ragflow_notes/deepdoc_chunking.md`](../../../../ragflow_notes/deepdoc_chunking.md)；
- **`deepdoc/parser/docx_parser.py`** —— `DocxParser` 同步遍历 `paragraphs` + `tables`，每段打 `section` 标签（`text` / `table` / `image`）；
- **`deepdoc/parser/utils.py`** —— VisionParser 用 OCR 模型兜底扫描件（s11 会更细讲）。

参考：[`ragflow_notes/multimodal_parsing.md`](../../../../ragflow_notes/multimodal_parsing.md)

## 思考题

**如果你的真实语料里 80% 是扫描件 PDF，unit 01 的链路对它们返回空字符串。要不要给 unit 01 加一层 OCR 兜底？还是放到 s11 单独做？**

提示：放 unit 01 会让所有用户都跑 OCR 模型，下载 1GB+；放 s11 是"按需启用"，更工程化。答案见 [`../../thinking_answers.md`](../../thinking_answers.md)。
```

- [ ] **Step 5: Rewrite chapter-root `code.py` as聚合 entry**

Write `s02_doc_loading/code.py`:

```python
#!/usr/bin/env python3
"""
s02 文档加载 — 聚合入口。实际逻辑在 units/01_basic_load/code.py。
本文件保留旧 `python s02_doc_loading/code.py` 启动方式。

运行: python s02_doc_loading/code.py
"""
from units.01_basic_load.code import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Update chapter-root `README.md`**

Prepend a units nav section to `s02_doc_loading/README.md`. Insert at line 1 (before existing `# s02 — 文档加载` heading):

```markdown
# s02 文档加载 — 章节导航

| Unit | 主题 | 它解决什么 | 对照 RAGFlow |
|---|---|---|---|
| [01_basic_load](./units/01_basic_load/README.md) | 最小可跑加载 (PDF + DOCX) | "统一 schema 是什么样" | `deepdoc/parser/{pdf,docx}_parser.py` |
| [02_failure_modes](./units/02_failure_modes/README.md) | 真实样本上的失败模式 | "为什么 unit 01 在 prod 不够" | `deepdoc/parser/utils.py` VisionParser |

跑：

\`\`\`bash
python s02_doc_loading/units/01_basic_load/code.py
python s02_doc_loading/units/02_failure_modes/code.py
# 旧路径仍可用:
python s02_doc_loading/code.py
\`\`\`

---

```

The original `# s02 — 文档加载 (PDF + DOCX)` heading follows unchanged.

- [ ] **Step 7: Verify all three entry points run**

Run:
```bash
python s02_doc_loading/code.py
python s02_doc_loading/units/01_basic_load/code.py
python s02_doc_loading/units/02_failure_modes/code.py
```

Expected: all three succeed. The聚合 and unit 01 print identical output (`PDF 段落数: 4, DOCX 段落数: 27`); unit 02 prints the failure-mode table.

- [ ] **Step 8: Commit and push**

```bash
git add s02_doc_loading/
git commit -m "s02: split into 2 units (01_basic_load, 02_failure_modes)"
# Push with retry:
for i in 1 2 3 4 5; do
  PAT=$(grep GITHUB_PAT /home/bibdr/projects/ai_agent/.env | cut -d= -f2)
  git push https://x-access-token:${PAT}@github.com/yaoweizhang/learn-ragflow.git master && break
  sleep 4
done
```

---

## Task 2: Refactor s03 (chunking)

**Files:**
- Create: `s03_chunking/units/01_basic_chunk/{code.py,README.md}`
- Create: `s03_chunking/units/02_chunk_failures/{code.py,README.md}`
- Modify: `s03_chunking/code.py` (聚合)
- Modify: `s03_chunking/README.md` (+units nav)

**Unit decomposition:**
- `01_basic_chunk`: 500-char cap + 句界切（`.。!?！？`）；reads from `samples/`, outputs chunks with `text` + page.
- `02_chunk_failures`: three named failure modes (a) 表格被切碎 (b) 父子块 (c) 跨段引用 — demonstrate each, cite ragflow's `_concat_downward` + `naive_merge` + `hierarchical_merge`.

**Steps (apply template):**
1. Read current `s03_chunking/code.py` (72 lines) → split functions into unit 01 (basic) and unit 02 (failure demos).
2. Write unit 01 `code.py` (copy core logic, add module docstring + `WORKDIR = parents[3]` fix).
3. Write unit 01 `README.md` using template; reference `ragflow_notes/deepdoc_chunking.md`.
4. Write unit 02 `code.py` with three functions (`demo_table_split`, `demo_parent_child`, `demo_cross_ref`); each prints "before/after" snippets.
5. Write unit 02 `README.md` referencing `deepdoc_chunking.md`.
6. Rewrite chapter `code.py` as `from units.01_basic_chunk.code import main`.
7. Update chapter `README.md` (insert nav table).
8. Verify all three entry points run.
9. Commit + push with retry.

- [ ] Commit message: `s03: split into 2 units (01_basic_chunk, 02_chunk_failures)`

---

## Task 3: Refactor s04 (embedding)

**Files:**
- Create: `s04_embedding/units/01_local_bge/{code.py,README.md}`
- Create: `s04_embedding/units/02_provider_routing/{code.py,README.md}`
- Modify: `s04_embedding/code.py` (聚合)
- Modify: `s04_embedding/README.md` (+units nav)

**Unit decomposition:**
- `01_local_bge`: BAAI/bge-small-zh-v1.5, 512 维, 归一化. Self-contained sentence-transformers load.
- `02_provider_routing`: OpenAI / Ollama provider switching; shows how `LLM_BASE` env selects backend; cite ragflow's `rag/llm/embedding_model.py`.

- [ ] Commit message: `s04: split into 2 units (01_local_bge, 02_provider_routing)`

---

## Task 4: Refactor s05 (vector index)

**Files:**
- Create: `s05_vector_index/units/01_chroma_build/{code.py,README.md}`
- Create: `s05_vector_index/units/02_chroma_query/{code.py,README.md}`
- Modify: `s05_vector_index/code.py` (聚合)
- Modify: `s05_vector_index/README.md` (+units nav)

**Unit decomposition:**
- `01_chroma_build`: persist chunks + embeddings + metadata into `_chroma/`; uses BGE from s04 via simple inline load (no chapter cross-import).
- `02_chroma_query`: load persisted collection, query by text embedding, print top-k with `text`/`page`/`source`.

- [ ] Commit message: `s05: split into 2 units (01_chroma_build, 02_chroma_query)`

---

## Task 5: Refactor s06 (retrieval — hybrid)

**Files:**
- Create: `s06_retrieval/units/01_bm25/{code.py,README.md}`
- Create: `s06_retrieval/units/02_hybrid_fusion/{code.py,README.md}`
- Modify: `s06_retrieval/code.py` (聚合)
- Modify: `s06_retrieval/README.md` (+units nav)

**Unit decomposition:**
- `01_bm25`: BM25 over chunks via `rank_bm25`; returns top-k with BM25 score.
- `02_hybrid_fusion`: combine BM25 + dense scores with weighted sum (default α=0.95 dense / 0.05 BM25, mirroring ragflow's `FusionExpr("weighted_sum", {"weights": "0.05,0.95"})`).

- [ ] Commit message: `s06: split into 2 units (01_bm25, 02_hybrid_fusion)`

---

## Task 6: Refactor s07 (rerank — single unit)

**Files:**
- Create: `s07_rerank/units/01_cross_encoder_rerank/{code.py,README.md}`
- Modify: `s07_rerank/code.py` (聚合)
- Modify: `s07_rerank/README.md` (+units nav)

**Unit decomposition (1 unit only):**
- `01_cross_encoder_rerank`: load BGE reranker, rerank top-N retrieval hits, print before/after ordering; cite ragflow's `_rerank_window` block/page alignment.

- [ ] Commit message: `s07: split into 1 unit (01_cross_encoder_rerank)`

---

## Task 7: Refactor s08 (prompt + generate — single unit)

**Files:**
- Create: `s08_prompt_generate/units/01_prompt_template/{code.py,README.md}`
- Modify: `s08_prompt_generate/code.py` (聚合)
- Modify: `s08_prompt_generate/README.md` (+units nav)

**Unit decomposition (1 unit only):**
- `01_prompt_template`: assemble `[i] (source#page) text` chunks into prompt, call LLM with `LLM_API_KEY` (graceful fallback when missing); cite `ragflow_notes/prompt_templates.md` for production prompt with `<|COMPLETE|>` sentinel.

- [ ] Commit message: `s08: split into 1 unit (01_prompt_template)`

---

## Task 8: Refactor s09 (agent + tools)

**Files:**
- Create: `s09_agent_tools/units/01_tool_call/{code.py,README.md}`
- Create: `s09_agent_tools/units/02_react_loop/{code.py,README.md}`
- Modify: `s09_agent_tools/code.py` (聚合)
- Modify: `s09_agent_tools/README.md` (+units nav)

**Unit decomposition:**
- `01_tool_call`: define 1-2 tools (e.g., `search_docs`, `get_page`), show LLM choosing which to call; cite `agent/component/`.
- `02_react_loop`: full Reason+Act loop with trace; cite `agent/canvas.py` DAG.

- [ ] Commit message: `s09: split into 2 units (01_tool_call, 02_react_loop)`

---

## Task 9: Refactor s10 (GraphRAG)

**Files:**
- Create: `s10_graphrag/units/01_extract/{code.py,README.md}`
- Create: `s10_graphrag/units/02_query/{code.py,README.md}`
- Modify: `s10_graphrag/code.py` (聚合)
- Modify: `s10_graphrag/README.md` (+units nav)

**Unit decomposition:**
- `01_extract`: LLM-based entity/relation extraction on `samples/`, persist to JSONL (toy graph).
- `02_query`: load toy graph, support 1-hop neighborhood retrieval; cite `ragflow_notes/graph_extraction.md` for LightRAG prompt + `entity_resolution.py` 2-stage pipeline.

- [ ] Commit message: `s10: split into 2 units (01_extract, 02_query)`

---

## Task 10: Refactor s11 (multimodal)

**Files:**
- Create: `s11_multimodal/units/01_table_extract/{code.py,README.md}`
- Create: `s11_multimodal/units/02_ocr/{code.py,README.md}`
- Modify: `s11_multimodal/code.py` (聚合)
- Modify: `s11_multimodal/README.md` (+units nav)

**Unit decomposition:**
- `01_table_extract`: `pdfplumber` extract tables from whitepaper, output as DataFrame-like dicts.
- `02_ocr`: `pytesseract` + `pdf2image` for image-based pages; cite `deepdoc/parser/utils.py` VisionParser.

- [ ] Commit message: `s11: split into 2 units (01_table_extract, 02_ocr)`

---

## Task 11: Refactor s12 (deployment — single unit)

**Files:**
- Create: `s12_deployment/units/01_fastapi_docker/{code.py,README.md}`
- Modify: `s12_deployment/code.py` (聚合)
- Modify: `s12_deployment/README.md` (+units nav)
- **Untouched:** `s12_deployment/{app.py,Dockerfile,docker-compose.yml}`

**Unit decomposition (1 unit only):**
- `01_fastapi_docker`: build the FastAPI wrapper (`app.py`) + show `docker-compose up` workflow; same end-state as current chapter but unit README teaches the why.

- [ ] Commit message: `s12: split into 1 unit (01_fastapi_docker)`

---

## Task 12: Update root README to reflect new units structure

**Files:**
- Modify: `README.md` (add a "每章结构" section after 学习路径)

- [ ] **Step 1: Add a "每章结构" section**

Insert after the 学习路径 section (around line 213):

```markdown
## 每章结构

每章 (`sXX_topic/`) 内部统一形状：

\`\`\`
sXX_topic/
├── README.md              # 章节入口：units 导航表 + 本章对照 ragflow_notes
├── README.en.md
├── thinking_answers.md
├── code.py                # 聚合入口：import units/, 保留旧启动方式
└── units/
    ├── 01_xxx/code.py     # unit 1（必有）
    ├── 01_xxx/README.md   # 4 段式：这是什么/跑起来/对照 ragflow/思考题
    └── 02_xxx/...         # unit 2（按需；≤ 2 unit/章）
\`\`\`

每个 unit 独立可跑：

\`\`\`bash
python sXX_topic/units/01_xxx/code.py
\`\`\`

旧入口仍可用：

\`\`\`bash
python sXX_topic/code.py   # 等价于跑 unit 01
\`\`\`
```

- [ ] **Step 2: Commit and push**

```bash
git add README.md
git commit -m "docs: root README - add 每章结构 section describing units/ shape"
# Push with retry (same pattern as Task 1).
```

---

## Self-Review Checklist

After completing all 12 tasks, before declaring done:

1. **Spec coverage:** Each chapter s02-s12 has been refactored; root README updated.
2. **No placeholders:** No `TODO`/`TBD` in any unit README; all code blocks are complete and runnable.
3. **Type consistency:** Every unit uses the same `WORKDIR = Path(__file__).resolve().parents[3]` pattern (verified by grep).
4. **聚合 correctness:** Each chapter-root `code.py` does `from units.NN.code import main` (verified by inspection).
5. **Push success:** Run `git fetch` and confirm `master` is at the latest local commit.