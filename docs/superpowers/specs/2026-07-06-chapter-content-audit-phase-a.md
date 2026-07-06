# Chapter Content Audit — Phase A (s02-s06) Design

> **Status:** Approved by user 2026-07-06 (brainstorming turn). Awaiting user review of this written spec before transitioning to writing-plans.

## Goal

Rewrite chapter-root `README.md` for **s02_doc_loading / s03_chunking / s04_embedding / s05_vector_index / s06_retrieval** to match the depth and structure of all-in-rag's corresponding chapters, while running a code audit that ensures `code.py` + `units/*/code.py` deliver what each README promises. Unit-level READMEs stay as-is (4-段式 原状).

## Why this exists

- **Current state:** chapter-root READMEs are 60-118 lines (mostly tables and nav). all-in-rag equivalent sub-files are 10-22KB per sub-file. The 5-20× depth gap makes our chapters feel like API docs rather than a tutorial.
- **Spot issues:** `s04/README.md` has a leaked Windows path (`cd D:/study/...`), no clear "what is embedding / why / how / RAGFlow 对照" narrative; s05 and s06 are similarly thin.
- **Code drift:** the s02 unit 02 README "1456 chars" sample was actually 572 (recent fix in `5c44084`) — same class of drift likely exists elsewhere.

## Architecture

- **Phased rollout:** Phase A (s02-s06) now; Phase B (s07-s08) and Phase C (s09-s12) brainstormed separately later.
- **Per-chapter workflow:** one task per chapter, each does README rewrite + code audit + (small) fixes in one PR.
- **Borrowed structure DNA:** all-in-rag's 4-段式 (是什么 / 为什么 / 怎么做 / RAGFlow 对照 + 思考题) + comparison tables + decision paths + MVP 4-步 patterns.
- **Forbidden overlay (per `rag-intro-writing-style.md`):** no `[^N]` academic footnotes, no 参考文献 section, no 思辨/辩论 chapters, no "X is dead" sections, no verbatim copying of all-in-rag sentences.

## Tech Stack

Same as parent project — Python 3.10+, no new dependencies. Plan execution via superpowers:subagent-driven-development.

## Global Constraints

These apply to every per-chapter task:

1. **Direct master push approved.** Each chapter = one commit. No PR.
2. **Chapter README target:** 200-350 lines after rewrite. If a chapter naturally stays < 200, that's fine — quality over quantity — but every 段 must be present.
3. **Unit READMEs untouched.** Phase A does not modify `units/NN_xxx/README.md`.
4. **Code changes ≤ 5 lines per chapter.** Larger changes require user sign-off (out of Phase A scope).
5. **No new dependencies.** `requirements.txt` is unchanged.
6. **No `[^N]` footnotes / 参考文献 sections / 思辨 chapters.** If reviewer flags any, treat as Critical.
7. **No verbatim copying from all-in-rag.** Allow ≤ 1 sentence of direct quote per chapter (terms of art / definitions are OK); everything else must be rewritten in our project's voice.
8. **Project-specific content preserved:** ASCII pipeline diagrams (where they exist), `samples/` file references, `units/` navigation tables, links to `ragflow_notes/<topic>.md`.
9. **Each chapter's implementer must `fetch` all-in-rag's equivalent sub-file** before writing. Reference mapping:
   - s02 ↔ all-in-rag `chapter2/04_data_load.md`
   - s03 ↔ all-in-rag `chapter2/05_text_chunking.md`
   - s04 ↔ all-in-rag `chapter3/06_vector_embedding.md`
   - s05 ↔ all-in-rag `chapter3/08_vector_db.md` (+ `09_milvus.md` only if it adds value)
   - s06 ↔ all-in-rag `chapter4/11_hybrid_search.md`
10. **Each chapter produces `sNN_topic/AUDIT.md`** — a one-page audit report (对齐 / 小修 / 大修 per criterion).

---

## Per-chapter 4-段式 template

Every chapter README must contain all four sections in this order:

```
# sNN <中文主题> (<English topic>)

> 一句话定位:本章节解决什么、给出什么。

## 一、是什么
   - 概念定义(≤ 5 行)
   - 在 RAG 全链路中的位置(可链 01_what_is_rag.md)
   - 本章 unit 拆解的 rationale (1-2 段)

## 二、为什么 / 现实问题
   - 不用这个技术会崩在哪(2-4 条具体场景, 借鉴 all-in-rag)
   - 选型对比表(若适用,如 embedding 模型维度/语言, 或 vector DB 取舍)
   - 引用 all-in-rag 对应章节的核心要点 (1-2 段, 改写不照抄)

## 三、怎么做 (MVP)
   - 本项目最小实现思路 (代码架构 + 关键函数签名)
   - 跑起来: 命令 + 期望输出片段
   - 真实世界会遇到的问题 (2-4 条, 衔接 unit 02 失败模式 或 s11 对应内容)

## 四、对照 RAGFlow 怎么做的 + 思考题
   - 引用 ragflow_notes/<topic>.md 的关键模块
   - 工业实现 vs MVP 的差距 (2-3 条)
   - 思考题 2-3 个 (指向 thinking_answers.md)
```

---

## Code audit criteria

Each chapter audit (`sNN_topic/AUDIT.md`) must report on all 4 criteria:

| Criterion | Question | Result tier |
|---|---|---|
| **1. README 声称的函数/输出 → 代码里有吗？** | Does `code.py` (or unit code) export every function/API the README documents? | 对齐 / 小修 / 大修 |
| **2. 代码里的主要函数 → README 解释了吗？** | Does the README explain input/output/purpose of every non-trivial function? | 对齐 / 小修 / 大修 |
| **3. README 里的运行示例 → 真能跑出那个输出吗？** | Are sample outputs in README still accurate (post-run)? | 对齐 / 小修 / 大修 |
| **4. Dead code / orphan import** | Any obvious lint issues like the s04 WORKDIR + Path import? | 对齐 / 小修 / 大修 |

**Tier definitions:**
- **对齐** — no change needed
- **小修** — ≤ 5 lines code or doc edit
- **大修** — > 5 lines; requires user sign-off before commit

All 小修 are in-scope for the implementer. All 大修 are reported back to the user (not committed).

---

## Task structure (Phase A = 5 chapters)

### Task 1: s02 doc loading
- **Files to rewrite:** `s02_doc_loading/README.md`
- **Files to audit:** `s02_doc_loading/code.py`, `s02_doc_loading/units/01_basic_load/code.py`, `s02_doc_loading/units/02_failure_modes/code.py`
- **Files to produce:** `s02_doc_loading/AUDIT.md`
- **Reference fetch:** `https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter2/04_data_load.md`
- **Commit message:** `s02: rewrite chapter README + code audit`

### Task 2: s03 chunking
- **Files to rewrite:** `s03_chunking/README.md`
- **Files to audit:** `s03_chunking/code.py`, `s03_chunking/units/01_basic_chunk/code.py`, `s03_chunking/units/02_chunk_failures/code.py`
- **Files to produce:** `s03_chunking/AUDIT.md`
- **Reference fetch:** `https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter2/05_text_chunking.md`
- **Commit message:** `s03: rewrite chapter README + code audit`

### Task 3: s04 embedding
- **Files to rewrite:** `s04_embedding/README.md` (also fix leaked Windows path `cd D:/study/...`)
- **Files to audit:** `s04_embedding/code.py`, `s04_embedding/units/01_local_bge/code.py`, `s04_embedding/units/02_provider_routing/code.py`
- **Files to produce:** `s04_embedding/AUDIT.md`
- **Reference fetch:** `https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter3/06_vector_embedding.md`
- **Commit message:** `s04: rewrite chapter README + code audit (also drop leaked Windows path)`

### Task 4: s05 vector index
- **Files to rewrite:** `s05_vector_index/README.md`
- **Files to audit:** `s05_vector_index/code.py`, `s05_vector_index/units/01_chroma_build/code.py`, `s05_vector_index/units/02_chroma_query/code.py`
- **Files to produce:** `s05_vector_index/AUDIT.md`
- **Reference fetch:** `https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter3/08_vector_db.md`
- **Commit message:** `s05: rewrite chapter README + code audit`

### Task 5: s06 retrieval
- **Files to rewrite:** `s06_retrieval/README.md`
- **Files to audit:** `s06_retrieval/code.py`, `s06_retrieval/units/01_bm25/code.py`, `s06_retrieval/units/02_hybrid_fusion/code.py`
- **Files to produce:** `s06_retrieval/AUDIT.md`
- **Reference fetch:** `https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter4/11_hybrid_search.md`
- **Commit message:** `s06: rewrite chapter README + code audit`

### Phase A final: whole-Phase review
After all 5 tasks, dispatch final reviewer subagent to confirm:
- All 5 chapters follow 4-段式
- No forbidden content (academic citations, 思辨, verbatim copies)
- All AUDIT.md files exist and have ≥ 1 fix in each criterion (else the audit was a no-op)
- No regressions: every unit code still runs

---

## Acceptance criteria

Phase A complete when ALL of:

1. ✅ `s02`–`s06` chapter-root READMEs all rewritten to ≥ 200 lines (or natural shorter with all 4 段 present)
2. ✅ Every chapter README contains all 4 sections in order (是什么 / 为什么 / 怎么做 / 对照 RAGFlow + 思考题)
3. ✅ Every chapter has `AUDIT.md` with all 4 criteria reported
4. ✅ Every 小修 has been committed + pushed
5. ✅ Every 大修 has been reported back to user with sign-off decision
6. ✅ Every `units/NN_xxx/code.py` still runs (no regressions)
7. ✅ `git grep -E '\[\^[0-9]|RAG 已死|参考文献'` returns 0 matches
8. ✅ Master IN SYNC at final commit

---

## Out of scope (Phase A)

- s07-s12 (Phase B and C, brainstormed separately)
- `ragflow_notes/*.md` files
- Unit-level READMEs (`units/NN_xxx/README.md`)
- New features / new dependencies
- `README.en.md` English versions
- Any code change > 5 lines per chapter

---

## Self-review (spec)

1. **Placeholder scan:** No TBDs. Every § has content. Concrete reference URLs. ✓
2. **Internal consistency:** §1 scope matches §5 acceptance criteria. Task list in §-matches reference mappings in Global Constraint 9. ✓
3. **Scope check:** 5 chapters with 2-3 files each = 10-15 files touched. Bounded. ✓
4. **Ambiguity check:** "audit" defined via 4 criteria with tier definitions. "Forbidden content" listed explicitly. "Rewrite voice" constrained by writing-style memory. ✓

## Open questions for user review

- 是否同意每章写之前 implementer 先 fetch 一份 all-in-rag 对应 sub-file？(已写入 Global Constraint 9)
- 是否同意 Phase A 不重写 `README.en.md`？(已写入 Out of scope)
- 是否同意大修需 user sign-off？(已写入 §Code audit criteria)
- 是否同意 unit README 不动？(已写入 Global Constraint 3)