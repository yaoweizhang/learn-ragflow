# Chapter Content Audit — Phase C2 (s10_graphrag + s11_multimodal) Design

> **Status:** Drafted 2026-07-06 (continuation from Phase C1). Awaiting user review of this written spec before transitioning to writing-plans.

## Goal

Rewrite chapter-root `README.md` for **s10_graphrag** and **s11_multimodal** to match the depth and structure of all-in-rag's corresponding chapters, while running a code audit that ensures `code.py` + `units/*/code.py` deliver what each README promises. Mirror Phase C1 + Phase A + B + B' pattern (all shipped).

## Why this exists

- **Current state:** s10 README = 96 lines, s11 README = 112 lines. Both informal narrative — no explicit 是什么/为什么/怎么做/对照 RAGFlow+思考题 4-段式, no MVP function table. Below 200-line target.
- **Code state:** Both chapters have 2 units (`units/01_extract` + `units/02_query` for s10; `units/01_table_extract` + `units/02_ocr` for s11) + chapter `code.py` aggregate = 3 code files each. AUDIT.md does not exist for either.
- **Carryover state:** Neither chapter has the self-referential all-in-rag blockquote (s09-s12 start with `# ...` directly); Phase B' cleanup does not apply. Both start with `## Units` heading.
- **Reference URL issue (from C1 lesson):** all-in-rag upstream chapter numbering has shifted since our reference was set; C1's URL `chapter6/19_agent.md` returned 404 and we used loose-borrow fallback successfully. Apply same fallback preemptively here.

## Architecture

- **Phased rollout (continued):** Phase A (s02-s06) + Phase B (s07-s08) + Phase B' (carryovers) + Phase C1 (s09) all shipped; Phase C2 = this spec; Phase C3 (s12_deployment) = separate spec per user direction.
- **Per-chapter workflow (mirrors Phase A + B + C1):** one task per chapter does README rewrite + code audit + (small) fixes in one commit. 2 chapters in this sub-phase.
- **Borrowed structure DNA:** all-in-rag's 4-段式 (是什么 / 为什么 / 怎么做 / 对照 RAGFlow + 思考题) + comparison tables + decision paths + MVP patterns.
- **Forbidden overlay (per `rag-intro-writing-style.md`):** no `[^N]` academic footnotes, no 参考文献 section, no 思辨/辩论 chapters, no "X is dead" sections, no verbatim copying of all-in-rag sentences.

## Tech Stack

Same as parent project + Phase A + B + C1 — Python 3.10+, no new dependencies. Plan execution via superpowers:subagent-driven-development (sub-agent implementer) + sub-agent reviewer (per C1 pattern, user-approved).

## Global Constraints

These apply to both s10 + s11 tasks:

1. **Direct master push approved.** One commit per chapter. No PR. 2 commits total in this sub-phase.
2. **Chapter README target:** 200-350 lines after rewrite. If a chapter naturally stays < 200, that's fine — quality over quantity — but every 段 must be present.
3. **Unit READMEs untouched.** Phase C2 does not modify `units/01_*/README.md` or `units/02_*/README.md`. (Note: this is a stricter rule than C1 — C2 forbids unit README edits even for "obvious bug" fixes; that's a Phase B' cleanup if needed.)
4. **Code changes ≤ 5 lines per chapter.** Larger changes require user sign-off (out of Phase C2 scope). Phase C2 budget: ≤ 10 lines cumulative across both chapters.
5. **No new dependencies.** `requirements.txt` is unchanged.
6. **No `[^N]` footnotes / 参考文献 sections / 思辨 chapters.** If reviewer flags any, treat as Critical.
7. **No verbatim copying from all-in-rag.** Allow ≤ 1 sentence of direct quote per chapter (terms of art / definitions are OK); everything else must be rewritten in our project's voice.
8. **Project-specific content preserved:**
   - **For s10 (GraphRAG)**: entity extraction via LLM, knowledge graph schema (`head / relation / tail`), query via community-detection or path-traversal, 2-unit progression (extract → query), `@lru_cache` model caching pattern (carried from s08/s09), `MiniMax-M3 over minimaxi.com` LLM provider example, references to `thinking_answers.md`, existing units nav table, link to `ragflow_notes/graph_extraction.md` (verify it exists).
   - **For s11 (multimodal)**: pdfplumber for table extraction, pytesseract/PIL for OCR (Phase A sweep already fixed import order in unit 02), 2-unit progression (tables → OCR), graceful-skip pattern when tesseract binary is missing, `samples/` file references, references to `thinking_answers.md`, existing units nav table, link to `ragflow_notes/multimodal_parsing.md` (verify it exists).
9. **Implementer must `fetch` all-in-rag's equivalent sub-file** before writing. Reference mapping:
   - s10 ↔ all-in-rag `chapter4/13_graph_rag.md` (loose borrow — **expected 404 from C1 lesson**; implementer must use loose-borrow fallback: borrow 4-段式 from `docs/00_introduction/01_what_is_rag.md` + s09 chapter pattern, with project-specific substitution for hand-rolled LLM-based entity extraction).
   - s11 ↔ all-in-rag `chapter5/15_multimodal.md` (loose borrow — **expected 404**; same loose-borrow fallback pattern).
   - If reference URL 404s, **document this in the report** (don't fabricate content) and use the fallback. C1 established this pattern.
10. **Both chapters produce AUDIT.md** — one-page audit report per chapter (对齐 / 小修 / 大修 per criterion).
11. **Carryovers from prior phases** (all addressed in their respective phases; no new carryovers for C2):
    - s09 unit 01 EOFError guard (Phase C1' carryover; out of C2 scope)
    - All Phase A + B + B' carryovers (all addressed)

---

## Per-chapter 4-段式 template (Phase C2 — same as Phase A + B + C1)

Every chapter README must contain all four sections in this order:

```
# sNN <中文主题> (<English topic>)

> 一句话定位:本章节解决什么、给出什么。

## 一、是什么
   - 概念定义(≤ 5 行)
   - 在 RAG 全链路中的位置(可链 01_what_is_rag.md)
   - 本章 unit 拆解的 rationale (1-2 段)

## 二、为什么 / 现实问题
   - 不用这个技术会崩在哪(2-4 条具体场景,借鉴 all-in-rag)
   - 选型对比表(若适用 —— s10: 实体抽取 LLM-based vs rule-based vs NER; s11: pdfplumber vs camelot vs PyMuPDF; tesseract vs PaddleOCR)
   - 引用 all-in-rag 对应章节的核心要点 (1-2 段, 改写不照抄)

## 三、怎么做 (MVP)
   - 本项目最小实现思路(代码架构 + 关键函数签名)
   - 跑起来:命令 + 期望输出片段
   - 真实世界会遇到的问题 (2-4 条)

## 四、对照 RAGFlow 怎么做的 + 思考题
   - 引用 ragflow_notes/<topic>.md 的关键模块
   - 工业实现 vs MVP 的差距 (2-3 条)
   - 思考题 2-3 个(指向 thinking_answers.md)
```

---

## Code audit criteria (Phase C2 — same 4 criteria as Phase A + B + C1)

Each chapter audit (`s10_graphrag/AUDIT.md` + `s11_multimodal/AUDIT.md`) must report on all 4 criteria, applied to the 3 in-scope code files per chapter (`code.py` + 2 unit `code.py` files):

| Criterion | Question | Result tier |
|---|---|---|
| **1. README 声称的函数/输出 → 代码里有吗？** | Does the 3-file in-scope code set export every function/API the README documents? | 对齐 / 小修 / 大修 |
| **2. 代码里的主要函数 → README 解释了吗？** | Does the README explain input/output/purpose of every non-trivial function in the 3 files? | 对齐 / 小修 / 大修 |
| **3. README 里的运行示例 → 真能跑出那个输出吗？** | Are sample outputs in README still accurate (post-run) for both unit files? | 对齐 / 小修 / 大修 |
| **4. Dead code / orphan import** | Any obvious lint issues in the 3 files (unused imports, dead vars)? | 对齐 / 小修 / 大修 |

**Tier definitions (unchanged from Phase A + B + C1):**
- **对齐** — no change needed
- **小修** — ≤ 5 lines code or doc edit
- **大修** — > 5 lines; requires user sign-off before commit

All 小修 are in-scope for the implementer. All 大修 are reported back to the user (not committed).

---

## Task structure (Phase C2 = 2 chapters + 1 final review)

### Task 1: s10 graphrag
- **Files to rewrite:** `s10_graphrag/README.md`
- **Files to audit:** `s10_graphrag/code.py`, `s10_graphrag/units/01_extract/code.py`, `s10_graphrag/units/02_query/code.py`
- **Files to produce:** `s10_graphrag/AUDIT.md`
- **Reference fetch:** `https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter4/13_graph_rag.md` (expected 404; use loose-borrow fallback)
- **Commit message:** `s10: rewrite chapter README + code audit`

### Task 2: s11 multimodal
- **Files to rewrite:** `s11_multimodal/README.md`
- **Files to audit:** `s11_multimodal/code.py`, `s11_multimodal/units/01_table_extract/code.py`, `s11_multimodal/units/02_ocr/code.py`
- **Files to produce:** `s11_multimodal/AUDIT.md`
- **Reference fetch:** `https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter5/15_multimodal.md` (expected 404; use loose-borrow fallback)
- **Commit message:** `s11: rewrite chapter README + code audit`

### Task 3: Whole-Phase C2 review
After Tasks 1 + 2, dispatch final review (inline, per Phase C1 pattern) to confirm:
- Both chapters follow 4-段式
- No forbidden content
- Both AUDIT.md files exist and are not no-ops
- No regressions: every `units/*/code.py` for both chapters still runs
- Master IN SYNC at final commit

---

## Execution pattern (Phase C2)

Carried from Phase C1 working pattern:
- **Implementer sub-agent (sonnet)** dispatched per task — same per-task brief format, same forbidden-content guard, same credential-helper push pattern
- **Reviewer sub-agent (sonnet)** dispatched per task — spec compliance + task quality review (C1 pattern; user explicitly chose "Full SDD w/ auto-classifier dispatch")
- **Final review inline by controller** (Phase C1 pattern — read brief + report + spot-read new README + grep checks + commit-stat review)
- **Cumulative dispatch re-authorization:** user re-authorized this dispatch pattern for Phase C ("Full SDD w/ auto-classifier dispatch"). Phase C2 dispatches fall under the same re-authorization.

---

## Acceptance criteria

Phase C2 complete when ALL of:

1. ✅ `s10` and `s11` chapter-root READMEs each rewritten to ≥ 200 lines (or natural shorter with all 4 段 present)
2. ✅ Every chapter README contains all 4 sections in order (是什么 / 为什么 / 怎么做 / 对照 RAGFlow + 思考题)
3. ✅ Every chapter has `AUDIT.md` with all 4 criteria reported
4. ✅ Every 小修 has been committed + pushed
5. ✅ Every 大修 has been reported back to user with sign-off decision
6. ✅ All `units/*/code.py` for both chapters still run (exit 0)
7. ✅ `git grep -E '\[\^[0-9]|RAG 已死|参考文献'` returns 0 matches across `s10_*` + `s11_*` README + AUDIT.md
8. ✅ Master IN SYNC at final commit

---

## Out of scope (Phase C2)

- Phase C1 (s09_agent_tools) — shipped at commit 0a772bd
- Phase C3 (s12_deployment) — separate spec
- `ragflow_notes/*.md` files (existing content; referenced but not modified)
- Unit-level READMEs (`units/01_*/README.md`, `units/02_*/README.md`)
- `README.en.md` English versions (out of scope for all phases)
- Adding new units to s10 or s11 (feature addition, not audit)
- New features / new dependencies
- Any code change > 5 lines in s10 or s11
- s09 unit 01 EOFError guard (Phase C1' carryover)
- All Phase A + B + B' + C1 carryovers (all addressed in their respective phases)

---

## Self-review (spec)

1. **Placeholder scan:** No TBDs. Every § has content. Concrete reference URLs. ✓
2. **Internal consistency:** §1 scope matches §5 acceptance criteria. Task list matches reference mappings in Global Constraint 9. ✓
3. **Scope check:** 2 chapters × 3 files = 6 files touched + 2 AUDIT.md created. Bounded, mirrors Phase B's 4-file total (per chapter). ✓
4. **Ambiguity check:**
   - "Audit" defined via 4 criteria with tier definitions (same as Phase A + B + C1). ✓
   - "Forbidden content" listed explicitly. ✓
   - "Borrow loosely" for s10/s11 expected 404 — fallback pattern from C1 explicitly stated. ✓
   - "2 units per chapter" audit scope explicit (3 files: chapter + 2 units). ✓
5. **Phase A + B + C1 execution pattern carried forward:** Sub-agent implementer + sub-agent reviewer per task, inline final review, credential-helper push. ✓
6. **Carryovers from prior phases documented as out-of-scope** so they don't surprise user mid-execution. ✓
7. **Phase C split (C1/C2/C3) is per user direction** (chose Three sub-phases option in brainstorm AskUserQuestion 2026-07-06). ✓
8. **Reference URL 404 expected** (C1 precedent; fallback pattern documented). ✓

## Open questions for user review

- 是否同意 s10 + s11 的 all-in-rag 参考 URL 走 loose borrow fallback(C1 已确立,直接采用)?
- 是否同意 Phase C2 沿用 Phase C1 的 sub-agent implementer + sub-agent reviewer 模式(每章 2 dispatch)?
- 是否同意 Phase C2 沿用 "≤ 5 行 per chapter + ≤ 10 行 cumulative" 修复预算?
- 是否同意 s09 unit 01 EOFError guard 不进 Phase C2(预算独立,C1' followup 时处理)?