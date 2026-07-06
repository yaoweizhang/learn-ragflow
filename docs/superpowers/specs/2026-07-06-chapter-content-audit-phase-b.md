# Chapter Content Audit — Phase B (s07-s08) Design

> **Status:** Approved by user 2026-07-06 (brainstorming turn). Awaiting user review of this written spec before transitioning to writing-plans.

## Goal

Rewrite chapter-root `README.md` for **s07_rerank / s08_prompt_generate** to match the depth and structure of all-in-rag's corresponding chapters, while running a code audit that ensures `code.py` + `units/*/code.py` deliver what each README promises. Mirror Phase A's shape (chapters s02-s06 already shipped).

## Why this exists

- **Current state:** s07 README = 91 lines, s08 README = 94 lines. Both are informal narrative — problem/solution/output/concerns/RAGFlow对照 — but lack Phase A's explicit 是什么/为什么/怎么做/对照 RAGFlow+思考题 4-段式 structure, no 选型对比表, no MVP function table. Lighter than Phase A chapters (~210-260 lines).
- **Code state:** Both chapters have only 1 unit each (`units/01_*`) — no failure-mode unit. Phase A's 3-file audit scope reduces to 2 files (chapter `code.py` + `units/01_*/code.py`).
- **Carryover drift to watch:** s07's existing content is already pretty good; the rewrite should preserve its concrete insight (rerank 分 vs vec 分的不同步例子) rather than rewriting it away. s08's content is similar — preserve the 拒绝 vs 引用 example.

## Architecture

- **Phased rollout (continued from Phase A):** Phase A (s02-s06) shipped 2026-07-06; Phase B (s07-s08) now; Phase C (s09-s12) separate later.
- **Per-chapter workflow (mirrors Phase A):** one task per chapter, each does README rewrite + code audit + (small) fixes in one commit.
- **Borrowed structure DNA:** all-in-rag's 4-段式 (是什么 / 为什么 / 怎么做 / 对照 RAGFlow + 思考题) + comparison tables + decision paths + MVP patterns.
- **Forbidden overlay (per `rag-intro-writing-style.md`):** no `[^N]` academic footnotes, no 参考文献 section, no 思辨/辩论 chapters, no "X is dead" sections, no verbatim copying of all-in-rag sentences.

## Tech Stack

Same as parent project + Phase A — Python 3.10+, no new dependencies. Plan execution via superpowers:subagent-driven-development (sub-agent implementer) + inline review (per Phase A's working pattern).

## Global Constraints

These apply to every per-chapter task:

1. **Direct master push approved.** Each chapter = one commit. No PR.
2. **Chapter README target:** 200-350 lines after rewrite. If a chapter naturally stays < 200, that's fine — quality over quantity — but every 段 must be present.
3. **Unit READMEs untouched.** Phase B does not modify `units/01_*/README.md`.
4. **Code changes ≤ 5 lines per chapter.** Larger changes require user sign-off (out of Phase B scope).
5. **No new dependencies.** `requirements.txt` is unchanged.
6. **No `[^N]` footnotes / 参考文献 sections / 思辨 chapters.** If reviewer flags any, treat as Critical.
7. **No verbatim copying from all-in-rag.** Allow ≤ 1 sentence of direct quote per chapter (terms of art / definitions are OK); everything else must be rewritten in our project's voice.
8. **Project-specific content preserved:**
   - For s07: rerank_score vs vec_score different-scoring example (rerank=0.954 vs vec=0.905), `@lru_cache` model caching, BAAI/bge-reranker-base mention, references to `thinking_answers.md` for "100 pairs" question
   - For s08: `<context>` 定界符 pattern, `[i] (source#page) text` rendering, `MiniMax-M3 over minimaxi.com` example output, refusal-flow example, references to `thinking_answers.md` for "why model doesn't cite #5" question
   - Both: links to `ragflow_notes/rerank.md` / `ragflow_notes/prompt_templates.md`, links to unit-level READMEs, the existing units nav table.
9. **Each chapter's implementer must `fetch` all-in-rag's equivalent sub-file** before writing. Reference mapping:
   - s07 ↔ all-in-rag `chapter4/15_advanced_retrieval_techniques.md` (Re-ranking subsection; ≈21KB; **strong scope match**)
   - s08 ↔ all-in-rag `chapter5/16_formatted_generation.md` (≈12KB; **scope mismatch accepted** — their focus is JSON output / Function Calling; ours is prompt template + citation parsing. Borrow structural arc + project-specific substitution.)
10. **Each chapter produces `sNN_topic/AUDIT.md`** — a one-page audit report (对齐 / 小修 / 大修 per criterion).
11. **Carryovers from Phase A remain out of scope** for Phase B: cross-chapter self-referential all-in-rag blockquote cleanup (across all 5+2=7 chapters), s05 README HF_HUB_OFFLINE env vars doc polish, s05 unit 02 README "~0.95" → "~0.50" score drift text. Phase B's chapter rewrites naturally will not introduce the blockquote at s07:4 / s08:4; existing blockquotes in s07:1 area and s08:1 area remain.

---

## Per-chapter 4-段式 template (Phase B — adapted from Phase A spec §Per-chapter 4-段式)

Every chapter README must contain all four sections in this order. Phase B-specific emphasis is shown in `[Phase B]` annotations:

```
# sNN <中文主题> (<English topic>)

> 一句话定位:本章节解决什么、给出什么。

## 一、是什么
   - 概念定义(≤ 5 行)
   - 在 RAG 全链路中的位置(可链 01_what_is_rag.md)
   - 本章 unit 拆解的 rationale (1-2 段)

## 二、为什么 / 现实问题
   - 不用这个技术会崩在哪(2-4 条具体场景,借鉴 all-in-rag 对应章节)
   - 选型对比表(若适用 —— [Phase B] s07 应有 rerank 模型选型表; s08 应有 LLM-provider 选型表)
   - 引用 all-in-rag 对应章节的核心要点 (1-2 段, 改写不照抄)

## 三、怎么做 (MVP)
   - 本项目最小实现思路(代码架构 + 关键函数签名)
   - 跑起来:命令 + 期望输出片段
   - 真实世界会遇到的问题 (2-4 条) [Phase B] s08 强调 prompt injection / token overflow / citation misalignment 三类
   - 代码归档与 schema 形态(若适用)

## 四、对照 RAGFlow 怎么做的 + 思考题
   - 引用 ragflow_notes/<topic>.md 的关键模块
   - 工业实现 vs MVP 的差距 (2-3 条) [Phase B] s07 强调 cross-encoder → LLM-rerank 二阶段跳级; s08 强调 RAGFlow 的 sufficiency-check + citation-plus 双 pass
   - 思考题 2-3 个(指向 thinking_answers.md)
```

---

## Code audit criteria (Phase B — same 4 criteria as Phase A)

Each chapter audit (`sNN_topic/AUDIT.md`) must report on all 4 criteria, applied to the 2 in-scope code files per chapter (chapter `code.py` + `units/01_*/code.py`):

| Criterion | Question | Result tier |
|---|---|---|
| **1. README 声称的函数/输出 → 代码里有吗？** | Does the 2-file in-scope code set export every function/API the README documents? | 对齐 / 小修 / 大修 |
| **2. 代码里的主要函数 → README 解释了吗？** | Does the README explain input/output/purpose of every non-trivial function in the 2 files? | 对齐 / 小修 / 大修 |
| **3. README 里的运行示例 → 真能跑出那个输出吗？** | Are sample outputs in README still accurate (post-run) for both files? | 对齐 / 小修 / 大修 |
| **4. Dead code / orphan import** | Any obvious lint issues in the 2 files (unused imports, dead vars)? | 对齐 / 小修 / 大修 |

**Tier definitions (unchanged from Phase A):**
- **对齐** — no change needed
- **小修** — ≤ 5 lines code or doc edit
- **大修** — > 5 lines; requires user sign-off before commit

All 小修 are in-scope for the implementer. All 大修 are reported back to the user (not committed).

**Phase B-specific audit note:** Because Phase B chapters have only 1 unit each (no `units/02_*` failure-mode code), criterion 3 evidence comes primarily from the `units/01/code.py` exit-0 run, not from a separate failure-mode demonstration. This is expected, not a defect.

---

## Task structure (Phase B = 2 chapters + 1 final review)

### Task 1: s07 rerank
- **Files to rewrite:** `s07_rerank/README.md`
- **Files to audit:** `s07_rerank/code.py`, `s07_rerank/units/01_cross_encoder_rerank/code.py`
- **Files to produce:** `s07_rerank/AUDIT.md`
- **Reference fetch:** `https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter4/15_advanced_retrieval_techniques.md` (Re-ranking subsection)
- **Commit message:** `s07: rewrite chapter README + code audit`

### Task 2: s08 prompt_generate
- **Files to rewrite:** `s08_prompt_generate/README.md`
- **Files to audit:** `s08_prompt_generate/code.py`, `s08_prompt_generate/units/01_prompt_template/code.py`
- **Files to produce:** `s08_prompt_generate/AUDIT.md`
- **Reference fetch:** `https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter5/16_formatted_generation.md`
- **Commit message:** `s08: rewrite chapter README + code audit`

### Task 3: Whole-Phase B review
After both tasks, dispatch final review (inline, per Phase A's pattern) to confirm:
- Both chapters follow 4-段式
- No forbidden content
- Both AUDIT.md files exist and are not no-ops
- No regressions: every `units/*/code.py` for both chapters still runs
- Master IN SYNC at final commit

---

## Execution pattern (Phase B)

Carried from Phase A's working pattern:
- **Implementer sub-agent (sonnet)** dispatched per chapter — same per-task brief format, same forbidden-content guard, same credential-helper push pattern
- **Review inline by controller** (Phase A pattern — read brief + report + spot-read new README + grep checks + commit-stat review)
- **Cumulative dispatch re-authorization:** user re-authorized this dispatch pattern with "Continue SDD with re-authorization note (Recommended)" in Phase A's last AskUserQuestion. Phase B dispatches will fall under the same re-authorization.

---

## Acceptance criteria

Phase B complete when ALL of:

1. ✅ `s07` and `s08` chapter-root READMEs each rewritten to ≥ 200 lines (or natural shorter with all 4 段 present)
2. ✅ Every chapter README contains all 4 sections in order (是什么 / 为什么 / 怎么做 / 对照 RAGFlow + 思考题)
3. ✅ Every chapter has `AUDIT.md` with all 4 criteria reported
4. ✅ Every 小修 has been committed + pushed
5. ✅ Every 大修 has been reported back to user with sign-off decision
6. ✅ `s07_rerank/units/01_cross_encoder_rerank/code.py` and `s08_prompt_generate/units/01_prompt_template/code.py` still run (exit 0)
7. ✅ `git grep -E '\[\^[0-9]|RAG 已死|参考文献'` returns 0 matches across `s07_*` + `s08_*` README/AUDIT.md
8. ✅ Master IN SYNC at final commit

---

## Out of scope (Phase B)

- s09-s12 (Phase C, separate brainstorm later)
- `ragflow_notes/*.md` files
- Unit-level READMEs (`units/01_*/README.md`)
- Adding new units (e.g., `units/02_failure_modes` for s07/s08) — feature addition, not audit
- New features / new dependencies
- `README.en.md` English versions
- Any code change > 5 lines per chapter
- Carryovers from Phase A (cross-chapter blockquote cleanup, s05 env-var doc, s05 unit 02 README score drift text) — out of Phase B scope; can become a Phase B' follow-up or ship with Phase C cleanup

---

## Self-review (spec)

1. **Placeholder scan:** No TBDs. Every § has content. Concrete reference URLs. ✓
2. **Internal consistency:** §1 scope matches §5 acceptance criteria. Task list in §matches reference mappings in Global Constraint 9. ✓
3. **Scope check:** 2 chapters with 2 files each = 4 files touched + 2 AUDIT.md created. Bounded, mirrors Phase A's 14-file total. ✓
4. **Ambiguity check:**
   - "Audit" defined via 4 criteria with tier definitions (same as Phase A). ✓
   - "Forbidden content" listed explicitly. ✓
   - "Borrow loosely" for s08 → chapter5/16 scope mismatch explicitly stated (user picked option 1, expected mismatch). ✓
   - "1 unit per chapter" audit scope explicit (no unit 02). ✓
5. **Phase-A execution pattern carried forward:** Sub-agent implementer + inline review, credential-helper push. ✓
6. **Carryovers from Phase A documented as out-of-scope** so they don't surprise user mid-execution. ✓

## Open questions for user review

- 是否同意每章 implementer 先 fetch 一份 all-in-rag 对应 sub-file？(已写入 Global Constraint 9)
- 是否同意 s08 ↔ chapter5/16 是 scope-mismatched borrow，按用户选项 1 仅借结构不借内容？(已写入 Global Constraint 9)
- 是否同意 Phase B 不沿用 Phase A 已发的 sub-agent-driven 模式做 Phase B dispatch？(本次已在 AskUserQuestion 明确批准)
- 是否同意 Phase B 沿用 Phase A 的 "≤ 5 行 per chapter + ≤ 10 行 cumulative" 修复预算？(已写入 Global Constraint 4)
- 是否同意 Phase A 后续 followups (跨章节 blockquote 清理、s05 README env-var 文档、s05 unit 02 README score 文本) 留到 Phase B' 或与 Phase C 合并, 不进入本 Phase B？(已写入 Out of scope)
