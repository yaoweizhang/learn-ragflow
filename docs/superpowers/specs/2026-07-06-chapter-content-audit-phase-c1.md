# Chapter Content Audit — Phase C1 (s09_agent_tools) Design

> **Status:** Approved by user 2026-07-06 (brainstorming turn). Awaiting user review of this written spec before transitioning to writing-plans.

## Goal

Rewrite chapter-root `README.md` for **s09_agent_tools** to match the depth and structure of all-in-rag's corresponding chapter, while running a code audit that ensures `code.py` + `units/*/code.py` deliver what each README promises. Mirror Phase A + B + B' pattern (Phases A s02-s06, B s07-s08, B' carryovers all shipped).

## Why this exists

- **Current state:** s09 README = 108 lines. Informal narrative — no explicit 是什么/为什么/怎么做/对照 RAGFlow+思考题 4-段式 structure, no 选型对比表, no MVP function table. Below the 200-line target used by Phase A + B chapters (210-290 lines).
- **Code state:** s09 has 2 units (`units/01_tool_call`, `units/02_react_loop`) + chapter `code.py` aggregate shim = 3 code files. AUDIT.md does not exist.
- **Carryover state:** Unlike s02-s08 (which had self-referential all-in-rag blockquote), s09 starts with no meta-attribution — Phase B' cleanup does not apply here. README opens with `# s09 ...` and goes straight to `## Units`.

## Architecture

- **Phased rollout (continued from Phase A + B + B'):** Phase A (s02-s06) shipped 2026-07-06; Phase B (s07-s08) shipped; Phase B' carryovers shipped (commit 72d33d8); Phase C now (split into C1 / C2 / C3 per user direction 2026-07-06).
- **Phase C split:**
  - **Phase C1** = s09_agent_tools (this spec)
  - **Phase C2** = s10_graphrag + s11_multimodal (separate spec)
  - **Phase C3** = s12_deployment (separate spec, with §四 改名为「对照生产部署实践」)
- **Per-chapter workflow (mirrors Phase A + B):** one task per chapter does README rewrite + code audit + (small) fixes in one commit.
- **Borrowed structure DNA:** all-in-rag's 4-段式 (是什么 / 为什么 / 怎么做 / 对照 RAGFlow + 思考题) + comparison tables + decision paths + MVP patterns.
- **Forbidden overlay (per `rag-intro-writing-style.md`):** no `[^N]` academic footnotes, no 参考文献 section, no 思辨/辩论 chapters, no "X is dead" sections, no verbatim copying of all-in-rag sentences.

## Tech Stack

Same as parent project + Phase A + B — Python 3.10+, no new dependencies. Plan execution via superpowers:subagent-driven-development (sub-agent implementer) + inline review (per Phase A + B working pattern).

## Global Constraints

These apply to the s09 task:

1. **Direct master push approved.** One commit per chapter. No PR.
2. **Chapter README target:** 200-350 lines after rewrite. If a chapter naturally stays < 200, that's fine — quality over quantity — but every 段 must be present.
3. **Unit READMEs untouched.** Phase C1 does not modify `units/01_tool_call/README.md` or `units/02_react_loop/README.md`.
4. **Code changes ≤ 5 lines per chapter.** Larger changes require user sign-off (out of Phase C1 scope). Phase C1 budget: ≤ 10 lines cumulative (only one chapter so this is just the 5-line chapter budget; the 10-line is for when other chapters come online).
5. **No new dependencies.** `requirements.txt` is unchanged.
6. **No `[^N]` footnotes / 参考文献 sections / 思辨 chapters.** If reviewer flags any, treat as Critical.
7. **No verbatim copying from all-in-rag.** Allow ≤ 1 sentence of direct quote per chapter (terms of art / definitions are OK); everything else must be rewritten in our project's voice.
8. **Project-specific content preserved:**
   - For s09: ReAct loop pattern (`thought → action → observation`), `retrieve` / `finish` tool names, 2-unit progression (basic tool call → full ReAct loop), `@lru_cache` model caching pattern (carried from s08 unit 01), `MiniMax-M3 over minimaxi.com` LLM provider example, references to `thinking_answers.md`, the existing units nav table.
9. **Implementer must `fetch` all-in-rag's equivalent sub-file** before writing. Reference mapping:
   - s09 ↔ all-in-rag `chapter6/19_agent.md` (ReAct / Tool Calling section; **loose borrow accepted** — their focus is on framework-based agents like LangChain AgentExecutor; ours is a hand-rolled ReAct loop with 2 tools. Borrow 4-段式 arc + project-specific substitution.)
10. **s09 produces `s09_agent_tools/AUDIT.md`** — a one-page audit report (对齐 / 小修 / 大修 per criterion).
11. **Carryovers from Phase A + B + B' remain out of scope** for Phase C1: cross-chapter self-referential all-in-rag blockquote cleanup (Phase B' addressed this for s02-s08; s09-s12 don't have such blockquotes — out-of-scope by no-op), s05 README HF_HUB_OFFLINE env vars doc polish (Phase B' noted moot), s05 unit 02 README score drift text (Phase B' fixed). s09 introduces no new carryovers.

---

## Per-chapter 4-段式 template (Phase C1 — same as Phase A + B)

Every chapter README must contain all four sections in this order:

```
# s09 <中文主题> (<English topic>)

> 一句话定位:本章节解决什么、给出什么。

## 一、是什么
   - 概念定义(≤ 5 行)
   - 在 RAG 全链路中的位置(可链 01_what_is_rag.md)
   - 本章 unit 拆解的 rationale (1-2 段)

## 二、为什么 / 现实问题
   - 不用这个技术会崩在哪(2-4 条具体场景,借鉴 all-in-rag 对应章节)
   - 选型对比表(若适用 —— s09: ReAct vs function-calling vs plan-and-execute)
   - 引用 all-in-rag 对应章节的核心要点 (1-2 段, 改写不照抄)

## 三、怎么做 (MVP)
   - 本项目最小实现思路(代码架构 + 关键函数签名)
   - 跑起来:命令 + 期望输出片段
   - 真实世界会遇到的问题 (2-4 条)

## 四、对照 RAGFlow 怎么做的 + 思考题
   - 引用 ragflow_notes/agent.md 的关键模块 (verify file exists; if not, note in report)
   - 工业实现 vs MVP 的差距 (2-3 条)
   - 思考题 2-3 个(指向 thinking_answers.md)
```

---

## Code audit criteria (Phase C1 — same 4 criteria as Phase A + B)

Chapter audit (`s09_agent_tools/AUDIT.md`) must report on all 4 criteria, applied to the 3 in-scope code files (`code.py` + `units/01_tool_call/code.py` + `units/02_react_loop/code.py`):

| Criterion | Question | Result tier |
|---|---|---|
| **1. README 声称的函数/输出 → 代码里有吗？** | Does the 3-file in-scope code set export every function/API the README documents? | 对齐 / 小修 / 大修 |
| **2. 代码里的主要函数 → README 解释了吗？** | Does the README explain input/output/purpose of every non-trivial function in the 3 files? | 对齐 / 小修 / 大修 |
| **3. README 里的运行示例 → 真能跑出那个输出吗？** | Are sample outputs in README still accurate (post-run) for both unit files? | 对齐 / 小修 / 大修 |
| **4. Dead code / orphan import** | Any obvious lint issues in the 3 files (unused imports, dead vars)? | 对齐 / 小修 / 大修 |

**Tier definitions (unchanged from Phase A + B):**
- **对齐** — no change needed
- **小修** — ≤ 5 lines code or doc edit
- **大修** — > 5 lines; requires user sign-off before commit

All 小修 are in-scope for the implementer. All 大修 are reported back to the user (not committed).

---

## Task structure (Phase C1 = 1 chapter + 1 final review)

### Task 1: s09 agent_tools
- **Files to rewrite:** `s09_agent_tools/README.md`
- **Files to audit:** `s09_agent_tools/code.py`, `s09_agent_tools/units/01_tool_call/code.py`, `s09_agent_tools/units/02_react_loop/code.py`
- **Files to produce:** `s09_agent_tools/AUDIT.md`
- **Reference fetch:** `https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter6/19_agent.md` (loose borrow — ReAct / Tool Calling section)
- **Commit message:** `s09: rewrite chapter README + code audit`

### Task 2: Whole-Phase C1 review
After Task 1, dispatch final review (inline, per Phase A + B pattern) to confirm:
- s09 follows 4-段式
- No forbidden content
- AUDIT.md exists and is not a no-op
- No regressions: every `units/*/code.py` for s09 still runs
- Master IN SYNC at final commit

---

## Execution pattern (Phase C1)

Carried from Phase A + B + B' working pattern:
- **Implementer sub-agent (sonnet)** dispatched for Task 1 — same per-task brief format, same forbidden-content guard, same credential-helper push pattern
- **Review inline by controller** (Phase A + B pattern — read brief + report + spot-read new README + grep checks + commit-stat review)
- **Cumulative dispatch re-authorization:** user re-authorized this dispatch pattern with "Continue SDD with re-authorization note (Recommended)" in Phase A's last AskUserQuestion. Phase C1 dispatches fall under the same re-authorization.

---

## Acceptance criteria

Phase C1 complete when ALL of:

1. ✅ `s09` chapter-root README rewritten to ≥ 200 lines (or natural shorter with all 4 段 present)
2. ✅ s09 README contains all 4 sections in order (是什么 / 为什么 / 怎么做 / 对照 RAGFlow + 思考题)
3. ✅ s09 AUDIT.md exists with all 4 criteria reported
4. ✅ Every 小修 has been committed + pushed
5. ✅ Every 大修 has been reported back to user with sign-off decision
6. ✅ `s09_agent_tools/units/01_tool_call/code.py` and `s09_agent_tools/units/02_react_loop/code.py` still run (exit 0)
7. ✅ `git grep -E '\[\^[0-9]|RAG 已死|参考文献'` returns 0 matches across `s09_*` README + AUDIT.md
8. ✅ Master IN SYNC at final commit

---

## Out of scope (Phase C1)

- Phase C2 (s10_graphrag + s11_multimodal) — separate spec
- Phase C3 (s12_deployment) — separate spec
- `ragflow_notes/*.md` files
- Unit-level READMEs (`units/01_tool_call/README.md`, `units/02_react_loop/README.md`)
- Adding new units to s09 (e.g., `units/03_planning`) — feature addition, not audit
- New features / new dependencies
- `README.en.md` English versions
- Any code change > 5 lines in s09
- Carryovers from Phase A + B + B' (all addressed in their respective phases)

---

## Self-review (spec)

1. **Placeholder scan:** No TBDs. Every § has content. Concrete reference URLs. ✓
2. **Internal consistency:** §1 scope matches §5 acceptance criteria. Task list matches reference mappings in Global Constraint 9. ✓
3. **Scope check:** 1 chapter with 3 files = 3 files touched + 1 AUDIT.md created. Bounded, mirrors Phase B's 4-file total. ✓
4. **Ambiguity check:**
   - "Audit" defined via 4 criteria with tier definitions (same as Phase A + B). ✓
   - "Forbidden content" listed explicitly. ✓
   - "Borrow loosely" for s09 → chapter6/19_agent scope mismatch explicitly stated (user picked option A, accepted mismatch). ✓
   - "2 units per chapter" audit scope explicit (3 files: chapter + 2 units). ✓
5. **Phase A + B + B' execution pattern carried forward:** Sub-agent implementer + inline review, credential-helper push. ✓
6. **Carryovers from Phase A + B + B' documented as out-of-scope** so they don't surprise user mid-execution. ✓
7. **Phase C split (C1/C2/C3) is per user direction** (chose Three sub-phases option in brainstorm AskUserQuestion 2026-07-06). ✓
8. **Reference mapping choice A (borrow loosely) is per user direction** (chose A in brainstorm AskUserQuestion 2026-07-06). ✓

## Open questions for user review

- 是否同意 s09 ↔ chapter6/19_agent 是 loose borrow(行业关注 LangChain AgentExecutor,我们是手写 ReAct + 2 tools),按用户选项 A 仅借结构不借内容？(已写入 Global Constraint 9)
- 是否同意 Phase C1 沿用 Phase A + B + B' 的 "≤ 5 行 per chapter" 修复预算？(已写入 Global Constraint 4)
- 是否同意 Phase C1 不沿用 Phase A 的 sub-agent-driven 模式做 dispatch？(本次已在 AskUserQuestion 明确批准)
- 是否同意 Phase A + B + B' 后续 followups(已无新增;Phase B' 4 项已清 3 项 + 1 moot)不进入本 Phase C1？(已写入 Out of scope)