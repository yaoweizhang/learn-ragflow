# Phase C1 Chapter Content Audit Implementation Plan (s09_agent_tools)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite chapter-root `README.md` for s09 to match all-in-rag's depth (200-350 lines, 4-段式), run a code audit covering 3 files (chapter `code.py` + 2 unit `code.py`), fix small mismatches (≤ 5 lines), and produce `AUDIT.md`.

**Architecture:** Mirror Phase A + B pattern. 2 tasks: Task 1 s09 + Task 2 final review. Each task: subagent implementer (sonnet) reads current state + fetches all-in-rag reference → rewrites README → runs code audit → applies small fixes → commits + pushes; controller reviews inline. Direct master push (user-approved).

**Tech Stack:** Python 3.10+, Markdown, Git. No new dependencies. Implements spec at `docs/superpowers/specs/2026-07-06-chapter-content-audit-phase-c1.md`.

## Global Constraints

These apply to every task. **Verify all are met before marking task complete.**

1. **Direct master push approved.** One commit for Task 1. Subject format: `s09: rewrite chapter README + code audit`. Use credential-helper pattern (per-task brief contains the canonical command).
2. **Chapter README target: 200-350 lines after rewrite.** Quality over quantity — natural shorter is OK if all 4 段 are present.
3. **Unit READMEs untouched** (`units/01_tool_call/README.md` and `units/02_react_loop/README.md` not modified). Code in `units/*/code.py` is in audit scope.
4. **Code changes ≤ 5 lines per chapter.** Larger changes require user sign-off (report, don't commit). Phase C1 budget: ≤ 10 lines cumulative (only one chapter so this is just the 5-line chapter budget; the 10-line is for when other chapters come online).
5. **No new dependencies.** `requirements.txt` is unchanged.
6. **No `[^N]` footnotes / 参考文献 sections / 思辨 chapters.** If reviewer flags any, treat as Critical.
7. **No verbatim copying from all-in-rag.** Allow ≤ 1 sentence of direct quote per chapter (terms of art / definitions OK). Everything else rewritten in our project's voice.
8. **Project-specific content preserved:** For s09 — ReAct loop pattern (`thought → action → observation`), `retrieve` / `finish` tool names, 2-unit progression (basic tool call → full ReAct loop), `@lru_cache` model caching pattern (carried from s08 unit 01), `MiniMax-M3 over minimaxi.com` LLM provider example, references to `thinking_answers.md`, existing units nav table, link to `ragflow_notes/agent.md` (verify it exists; if not, say so explicitly in the report).
9. **Implementer must `fetch` all-in-rag's equivalent sub-file before writing** (per chapter mapping below).
10. **Chapter produces `s09_agent_tools/AUDIT.md`** — 4-criterion report (对齐 / 小修 / 大修 for each).
11. **Forbidden content self-check** at end of every task: `git grep -E '\[\^[0-9]|RAG 已死|参考文献' <changed files>` returns 0 matches.
12. **No regressions**: every `units/*/code.py` for s09 still runs (verified by implementer before commit).

### Reference mapping (per spec Global Constraint 9)

| Chapter | all-in-rag reference URL |
|---|---|
| s09 | `https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter6/19_agent.md` (Agent / ReAct section; loose borrow — scope mismatch accepted; mostly substitute OUR content) |

### The 4-段式 template (every chapter README must contain all 4)

```
# s09 <中文主题> (<English topic>)

> 一句话定位:本章节解决什么、给出什么。

## 一、是什么
   - 概念定义(≤ 5 行)
   - 在 RAG 全链路中的位置(可链 01_what_is_rag.md)
   - 本章 unit 拆解的 rationale (1-2 段)

## 二、为什么 / 现实问题
   - 不用这个技术会崩在哪(2-4 条具体场景,借鉴 all-in-rag)
   - 选型对比表(若适用 —— s09: ReAct vs function-calling vs plan-and-execute)
   - 引用 all-in-rag 对应章节的核心要点(1-2 段,改写不照抄)

## 三、怎么做 (MVP)
   - 本项目最小实现思路(代码架构 + 关键函数签名)
   - 跑起来:命令 + 期望输出片段
   - 真实世界会遇到的问题 (2-4 条)

## 四、对照 RAGFlow 怎么做的 + 思考题
   - 引用 ragflow_notes/agent.md 的关键模块
   - 工业实现 vs MVP 的差距 (2-3 条)
   - 思考题 2-3 个(指向 thinking_answers.md)
```

### The 4-criterion audit (every chapter must produce `s09_agent_tools/AUDIT.md`)

| # | Criterion | Question to answer |
|---|---|---|
| 1 | README 声称的函数/输出 → 代码里有吗？ | Does `code.py` + `units/01/code.py` + `units/02/code.py` export every function/API the README documents? |
| 2 | 代码里的主要函数 → README 解释了吗？ | Does the README explain input/output/purpose of every non-trivial function? |
| 3 | README 里的运行示例 → 真能跑出那个输出吗？ | Are sample outputs in README still accurate (post-run)? |
| 4 | Dead code / orphan import | Any obvious lint issues like unused vars or orphan imports? |

**Tier per criterion:** 对齐 / 小修 / 大修. **Phase C1-specific note:** s09 has 2 units (`01_tool_call`, `02_react_loop`); criterion 3 evidence comes from BOTH units' exit-0 + matching README's expected output (acceptable, not a defect).

---

### Task 1: s09 agent_tools

**Files:**
- Rewrite: `s09_agent_tools/README.md`
- Audit (read + 小修 ≤ 5 lines): `s09_agent_tools/code.py`, `s09_agent_tools/units/01_tool_call/code.py`, `s09_agent_tools/units/02_react_loop/code.py`
- Create: `s09_agent_tools/AUDIT.md`

**Reference fetch (mandatory):**
```bash
curl -sL --max-time 15 "https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter6/19_agent.md" > /tmp/air-s09.md
```

**Interfaces:**
- Consumes: nothing (first task)
- Produces: new s09 README, s09/AUDIT.md, possibly patched code (≤ 5 lines total)

- [ ] **Step 1: Capture base SHA**

```bash
git rev-parse HEAD | tee /tmp/s09-base.txt
```

- [ ] **Step 2: Read current state**

```bash
sed -n '1,200p' s09_agent_tools/README.md
echo "---unit READMEs (do not edit)---"
ls s09_agent_tools/units/01_tool_call/
ls s09_agent_tools/units/02_react_loop/
echo "---code files---"
cat s09_agent_tools/code.py
cat s09_agent_tools/units/01_tool_call/code.py
cat s09_agent_tools/units/02_react_loop/code.py
```

- [ ] **Step 3: Fetch all-in-rag reference and extract structural pattern**

```bash
curl -sL --max-time 15 "https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter6/19_agent.md" > /tmp/air-s09.md
wc -l /tmp/air-s09.md
# Read /tmp/air-s09.md, focus on the ReAct / Tool Calling sections. Capture: (a) ReAct definition, (b) tool calling flow, (c) agent framework comparison (LangChain AgentExecutor vs hand-rolled). Write notes to scratch — NOT into the project.
```

- [ ] **Step 4: Draft new `s09_agent_tools/README.md`**

Use the 4-段式 template. 200-350 lines. Project-specific content to preserve:
- ReAct loop pattern: `thought → action → observation` cycle
- `retrieve` / `finish` tool names (chapter-specific tool set)
- 2-unit progression: unit 01 = basic tool call, unit 02 = full ReAct loop
- `@lru_cache` model caching (carried from s08 unit 01 pattern)
- `MiniMax-M3 over minimaxi.com` LLM provider example output
- Existing units nav table
- Link to `ragflow_notes/agent.md` (verify it exists; if not, say so explicitly in the report)
- References to `thinking_answers.md` for "ReAct vs function-calling" question

Forbidden self-check before saving: no `[^N]`, no "RAG 已死", no "参考文献" section.

- [ ] **Step 5: Run code audit — 4 criteria**

Read each code file. For each criterion, decide 对齐 / 小修 / 大修. Record in `s09_agent_tools/AUDIT.md`:

```markdown
# s09 Agent — Code Audit

Date: <YYYY-MM-DD>
Commit: <HEAD>

## Criterion 1: README-claimed functions present in code?
Tier: 对齐 | 小修 | 大修
...

## Criterion 2: Code's main functions explained in README?
...
## Criterion 3: README sample outputs match live run?
...
## Criterion 4: Dead code / orphan import?
...

## Summary
<small fixes applied (≤ 5 lines) | big fixes reported to user>

## Big fixes needing user sign-off
<none if no big fixes>
```

- [ ] **Step 6: Apply 小修 (≤ 5 lines total)**

If any criterion is 小修, edit the corresponding code or doc line. Verify the diff is ≤ 5 lines:

```bash
git diff --stat s09_agent_tools/
```

If diff > 5 lines, revert and report as 大修 instead.

- [ ] **Step 7: Verify unit code still runs**

```bash
timeout 30 python s09_agent_tools/units/01_tool_call/code.py 2>&1 | tail -5
timeout 30 python s09_agent_tools/units/02_react_loop/code.py 2>&1 | tail -5
```

Both must exit 0.

- [ ] **Step 8: Forbidden-content self-check**

```bash
git grep -nE '\[\^[0-9]|RAG 已死|参考文献' s09_agent_tools/README.md s09_agent_tools/AUDIT.md
```

Expected: empty output. If non-empty, fix the offending line.

- [ ] **Step 9: Commit**

```bash
git add s09_agent_tools/README.md s09_agent_tools/AUDIT.md s09_agent_tools/code.py s09_agent_tools/units/01_tool_call/code.py s09_agent_tools/units/02_react_loop/code.py
git commit -m "s09: rewrite chapter README + code audit"
```

- [ ] **Step 10: Push to master (credential-helper pattern)**

```bash
set -a && source /home/bibdr/projects/ai_agent/.env && set +a && _STORE=$(mktemp) && \
  printf 'https://x-access-token:%s@github.com\n' "$GITHUB_PAT" > "$_STORE" && chmod 600 "$_STORE" && \
  for i in 1 2 3 4 5; do
    if GIT_TERMINAL_PROMPT=0 git -c "credential.helper=store --file=$_STORE" push https://github.com/yaoweizhang/learn-ragflow.git master; then
      echo "PUSH OK iter=$i"; break
    fi
    echo "PUSH FAIL iter=$i, retrying in 8s..."; sleep 8
  done
  _RC=$?
  rm -f "$_STORE"; unset GITHUB_PAT
  exit $_RC
```

Verify with `git fetch origin master` + `[LOCAL == REMOTE]` (sandbox TLS may fail; trust `git push` output line in that case). Local fetch retry loop up to 5 iterations.

- [ ] **Step 11: Update local ledger (gitignored)**

Append to `.superpowers/sdd/progress.md`:

```
Task 1 (phase C1): complete (commits <base9>..<head9>)
  s09 agent_tools: README rewritten to 4-段式 (XXX lines); AUDIT.md emitted; N small fixes.
```

- [ ] **Step 12: Write report to `.superpowers/sdd/task-1-report.md`**

Include:
- New README line count
- Audit summary (each criterion tier)
- Big fixes list (if any)
- Commit hash
- Verification command output (last 5 lines of each unit run)
- Concerns

Return ≤ 15 lines to controller with status DONE / DONE_WITH_CONCERNS / BLOCKED.

---

### Task 2 (Final): Whole-Phase C1 review

**Files:**
- Read: `s09_agent_tools/AUDIT.md`
- Read: `s09_agent_tools/README.md`

- [ ] **Step 1: Capture base range**

```bash
git rev-parse HEAD | tee /tmp/phase-c1-base.txt
# Note: this is HEAD after Task 1's commit
```

- [ ] **Step 2: Verify s09 meets acceptance criteria**

For s09:
1. README ≥ 200 lines OR all 4 段 present
2. AUDIT.md exists with 4 criteria reported
3. No forbidden content

- [ ] **Step 3: Verify no regressions**

```bash
for unit in s09_agent_tools/units/*/code.py; do
  [ -f "$unit" ] && echo "checking $unit"
  timeout 15 python "$unit" 2>&1 | tail -3
done
```

- [ ] **Step 4: Verify push status**

```bash
git fetch origin master
LOCAL=$(git rev-parse HEAD); REMOTE=$(git rev-parse origin/master)
[ "$LOCAL" = "$REMOTE" ] && echo "IN SYNC @ $LOCAL" || echo "DRIFT"
```

- [ ] **Step 5: Append final ledger entry**

```
Phase C1: complete (commits <base9>..<head9>)
  - s09 rewritten + audited
  - 1 commit, IN SYNC (per `git push` output)
  - <N> small fixes total, <N> big fixes reported
  - 0 forbidden content matches
```

- [ ] **Step 6: Write final Phase C1 review report** (inline by controller)

Inline review (read new README + AUDIT file + check forbidden grep + confirm push status). No sub-agent dispatch needed (Phase A + B working pattern).

Return review verdict to user for Phase C1 closeout.

---

## Self-Review (plan)

1. **Spec coverage:** Phase C1 spec covers 1 chapter + 8 acceptance criteria + reference mapping + 4-段式 + 4 audit criteria. Plan covers each via Task 1 + Task 2. ✓
2. **Placeholder scan:** No TBDs. Every Step has concrete commands or file paths. ✓
3. **Type consistency:** Audit file format consistent with Phase A + B. Commit message format consistent. Reference fetch pattern consistent. ✓
4. **Forbidden content check** is explicit in Step 8 (per task) and Step 2 (final review). ✓
5. **Big-fix handling** is explicit: report back to user, don't commit. ✓
6. **Per-chapter granularity** allows failure isolation. ✓
7. **Audit scope adapted for Phase C1** (3 files: chapter + 2 units). ✓
8. **No placeholders** like "similar to Task N" — Task 1 explicitly enumerates its substitutions. ✓

**Adjustments made during self-review:**
- Task 2 changed from sub-agent dispatch to inline review (auto-classifier pattern from Phase A + B confirmed conservative on cumulative sub-agent dispatch).
- Phase C1 budget for fixes is ≤ 5 lines per chapter, ≤ 10 lines cumulative (per spec §Global Constraint 4).
- Reference mapping: chapter6/19_agent (loose borrow — scope mismatch accepted per user brainstorm choice A).

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-06-chapter-content-audit-phase-c1.md`.

Executing via Subagent-Driven Development (per user's direct invocation in prior phases). Phase A + B + B' working pattern carried: sub-agent implementer per task + inline review by controller + credential-helper push.