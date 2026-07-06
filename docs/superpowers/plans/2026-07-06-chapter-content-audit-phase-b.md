# Phase B Chapter Content Audit Implementation Plan (s07-s08)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite chapter-root `README.md` for s07 / s08 to match all-in-rag's depth (200-350 lines, 4-段式), run a code audit per chapter, fix small mismatches (≤ 5 lines), and produce `AUDIT.md` for each.

**Architecture:** Mirror Phase A's pattern. 3 tasks: Task 1 s07 + Task 2 s08 + Task 3 final review. Each task: subagent implementer (sonnet) reads current state + fetches all-in-rag reference → rewrites README → runs code audit → applies small fixes → commits + pushes; controller reviews inline. Direct master push (user-approved).

**Tech Stack:** Python 3.10+, Markdown, Git. No new dependencies. Implements spec at `docs/superpowers/specs/2026-07-06-chapter-content-audit-phase-b.md`.

## Global Constraints

These apply to every task. **Verify all are met before marking task complete.**

1. **Direct master push approved.** One commit per chapter. Subject format: `sNN: rewrite chapter README + code audit`. Use credential-helper pattern (per-task brief contains the canonical command).
2. **Chapter README target: 200-350 lines after rewrite.** Quality over quantity — natural shorter is OK if all 4 段 are present.
3. **Unit READMEs untouched** (`units/01_*/README.md` not modified). Code in `units/01_*/code.py` is in audit scope.
4. **Code changes ≤ 5 lines per chapter.** Larger changes require user sign-off (report, don't commit). Phase B cumulative budget: ≤ 10 lines.
5. **No new dependencies.** `requirements.txt` is unchanged.
6. **No `[^N]` footnotes / 参考文献 sections / 思辨 chapters.** If reviewer flags any, treat as Critical.
7. **No verbatim copying from all-in-rag.** Allow ≤ 1 sentence of direct quote per chapter (terms of art / definitions OK). Everything else rewritten in our project's voice.
8. **Project-specific content preserved:** For s07 — rerank_score vs vec_score 不同步 example, `@lru_cache` model caching, `BAAI/bge-reranker-base` mention. For s08 — `<context>` 定界符 pattern, `[i] (source#page) text` rendering, `MiniMax-M3 over minimaxi.com` example output. Both: existing units nav table, `thinking_answers.md` references, `ragflow_notes/<topic>.md` links.
9. **Implementer must `fetch` all-in-rag's equivalent sub-file before writing** (per chapter mapping below).
10. **Each chapter produces `sNN_topic/AUDIT.md`** — 4-criterion report (对齐 / 小修 / 大修 for each).
11. **Forbidden content self-check** at end of every task: `git grep -E '\[\^[0-9]|RAG 已死|参考文献' <changed files>` returns 0 matches.
12. **No regressions**: every `units/01_*/code.py` for that chapter still runs (verified by implementer before commit).

### Reference mapping (per spec Global Constraint 9)

| Chapter | all-in-rag reference URL |
|---|---|
| s07 | `https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter4/15_advanced_retrieval_techniques.md` (Re-ranking subsection; strong scope match) |
| s08 | `https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter5/16_formatted_generation.md` (loose borrow — scope mismatch accepted; mostly substitute OUR content) |

### The 4-段式 template (every chapter README must contain all 4)

```
# sNN <中文主题> (<English topic>)

> 一句话定位:本章节解决什么、给出什么。

## 一、是什么
   - 概念定义(≤ 5 行)
   - 在 RAG 全链路中的位置(可链 01_what_is_rag.md)
   - 本章 unit 拆解的 rationale (1-2 段)

## 二、为什么 / 现实问题
   - 不用这个技术会崩在哪(2-4 条具体场景,借鉴 all-in-rag)
   - 选型对比表(若适用 —— s07: rerank 模型; s08: LLM provider)
   - 引用 all-in-rag 对应章节的核心要点(1-2 段,改写不照抄)

## 三、怎么做 (MVP)
   - 本项目最小实现思路(代码架构 + 关键函数签名)
   - 跑起来:命令 + 期望输出片段
   - 真实世界会遇到的问题 (2-4 条)

## 四、对照 RAGFlow 怎么做的 + 思考题
   - 引用 ragflow_notes/<topic>.md 的关键模块
   - 工业实现 vs MVP 的差距 (2-3 条)
   - 思考题 2-3 个(指向 thinking_answers.md)
```

### The 4-criterion audit (every chapter must produce `sNN_topic/AUDIT.md`)

| # | Criterion | Question to answer |
|---|---|---|
| 1 | README 声称的函数/输出 → 代码里有吗？ | Does `code.py` + `units/01/code.py` export every function/API the README documents? |
| 2 | 代码里的主要函数 → README 解释了吗？ | Does the README explain input/output/purpose of every non-trivial function? |
| 3 | README 里的运行示例 → 真能跑出那个输出吗？ | Are sample outputs in README still accurate (post-run)? |
| 4 | Dead code / orphan import | Any obvious lint issues like unused vars or orphan imports? |

**Tier per criterion:** 对齐 / 小修 / 大修. **Phase B-specific note:** no `units/02/*` failure-mode code; criterion 3 evidence comes from `units/01/code.py` exit-0 + matching README's expected output (acceptable, not a defect).

---

### Task 1: s07 rerank

**Files:**
- Rewrite: `s07_rerank/README.md`
- Audit (read + 小修 ≤ 5 lines): `s07_rerank/code.py`, `s07_rerank/units/01_cross_encoder_rerank/code.py`
- Create: `s07_rerank/AUDIT.md`

**Reference fetch (mandatory):**
```bash
curl -sL --max-time 15 "https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter4/15_advanced_retrieval_techniques.md" > /tmp/air-s07.md
```

**Interfaces:**
- Consumes: nothing (first task)
- Produces: new s07 README, s07/AUDIT.md, possibly patched code (≤ 5 lines total)

- [ ] **Step 1: Capture base SHA**

```bash
git rev-parse HEAD | tee /tmp/s07-base.txt
```

- [ ] **Step 2: Read current state**

```bash
sed -n '1,200p' s07_rerank/README.md
echo "---unit READMEs (do not edit)---"
ls s07_rerank/units/01_cross_encoder_rerank/
echo "---code files---"
cat s07_rerank/code.py
cat s07_rerank/units/01_cross_encoder_rerank/code.py
```

- [ ] **Step 3: Fetch all-in-rag reference and extract structural pattern**

```bash
curl -sL --max-time 15 "https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter4/15_advanced_retrieval_techniques.md" > /tmp/air-s07.md
wc -l /tmp/air-s07.md
# Read /tmp/air-s07.md, focus on the "## 一、重排序 (Re-ranking)" subsection. Capture: (a) cross-encoder vs bi-encoder definition, (b) typical cross-encoder cost info, (c) rerank model selection table. Write notes to scratch — NOT into the project.
```

- [ ] **Step 4: Draft new `s07_rerank/README.md`**

Use the 4-段式 template. 200-350 lines. Project-specific content to preserve:
- Concrete example: `[server_whitepaper.pdf#1] rerank=0.954 vec=0.905 | ... 内存、10 个 PCIe 4.0 扩展槽位 ...` (rerank vs vec 不同步)
- `@lru_cache(maxsize=1)` for model loading
- `BAAI/bge-reranker-base` mention
- Existing units nav table
- Link to `ragflow_notes/rerank.md` (verify it exists; if not, say so explicitly in the report)

Forbidden self-check before saving: no `[^N]`, no "RAG 已死", no "参考文献" section.

- [ ] **Step 5: Run code audit — 4 criteria**

Read each code file. For each criterion, decide 对齐 / 小修 / 大修. Record in `s07_rerank/AUDIT.md`:

```markdown
# s07 Rerank — Code Audit

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
git diff --stat s07_rerank/
```

If diff > 5 lines, revert and report as 大修 instead.

- [ ] **Step 7: Verify unit code still runs**

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # if reranker loads BGE
timeout 20 python s07_rerank/units/01_cross_encoder_rerank/code.py 2>&1 | tail -5
```

Must exit 0.

- [ ] **Step 8: Forbidden-content self-check**

```bash
git grep -nE '\[\^[0-9]|RAG 已死|参考文献' s07_rerank/README.md s07_rerank/AUDIT.md
```

Expected: empty output. If non-empty, fix the offending line.

- [ ] **Step 9: Commit**

```bash
git add s07_rerank/README.md s07_rerank/AUDIT.md s07_rerank/code.py s07_rerank/units/01_cross_encoder_rerank/code.py
git commit -m "s07: rewrite chapter README + code audit"
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
Task 1 (phase B): complete (commits <base7>..<head7>)
  s07 rerank: README rewritten to 4-段式 (XXX lines); AUDIT.md emitted; N small fixes.
```

- [ ] **Step 12: Write report to `.superpowers/sdd/task-1-report.md`**

Include:
- New README line count
- Audit summary (each criterion tier)
- Big fixes list (if any)
- Commit hash
- Verification command output (last 5 lines of unit run)
- Concerns

Return ≤ 15 lines to controller with status DONE / DONE_WITH_CONCERNS / BLOCKED.

---

### Task 2: s08 prompt_generate

**Files:**
- Rewrite: `s08_prompt_generate/README.md`
- Audit (read + 小修 ≤ 5 lines): `s08_prompt_generate/code.py`, `s08_prompt_generate/units/01_prompt_template/code.py`
- Create: `s08_prompt_generate/AUDIT.md`

**Reference fetch:**
```bash
curl -sL --max-time 15 "https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter5/16_formatted_generation.md" > /tmp/air-s08.md
```

**Same workflow as Task 1, with these substitutions:**
- 4-段式 template (s08 specifics: prompt template + citation parsing + sufficiency check)
- s08 reference scope mismatch already noted (chapter5/16 is JSON output / Function Calling, our focus is prompt template) — borrow 4-段式 arc + project-specific substitution
- s08 unit 02 failure categories: prompt injection / token overflow / citation misalignment (the README §三 should explicitly surface these — Phase B-specific guidance per spec)
- Reference: all-in-rag `chapter5/16_formatted_generation.md` (≈12KB; sparse; mostly substitute OUR content)
- Project-specific: link to `ragflow_notes/prompt_templates.md`, mention `<context>` 定界符 + `[i] (source#page) text` rendering, `MiniMax-M3 over minimaxi.com` example output, refusal-flow example
- Commit message: `s08: rewrite chapter README + code audit`

Steps 1-12 identical structure to Task 1.

---

### Task 3 (Final): Whole-Phase B review

**Files:**
- Read: `s07_rerank/AUDIT.md`, `s08_prompt_generate/AUDIT.md`
- Read: `s07_rerank/README.md`, `s08_prompt_generate/README.md`

- [ ] **Step 1: Capture base range**

```bash
git rev-parse HEAD | tee /tmp/phase-b-base.txt
# Note: this is HEAD after Task 2's commit
```

- [ ] **Step 2: Verify both chapters meet acceptance criteria**

For each chapter s07-s08:
1. README ≥ 200 lines OR all 4 段 present
2. AUDIT.md exists with 4 criteria reported
3. No forbidden content

- [ ] **Step 3: Verify no regressions**

```bash
for ch in s07_rerank s08_prompt_generate; do
  for unit in $ch/units/*/code.py; do
    [ -f "$unit" ] && echo "checking $unit"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    timeout 15 python "$unit" 2>&1 | tail -3
  done
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
Phase B: complete (commits <base7>..<head7>)
  - s07 / s08 all rewritten + audited
  - 2 commits, both IN SYNC (per `git push` output)
  - <N> small fixes total, <N> big fixes reported
  - 0 forbidden content matches
```

- [ ] **Step 6: Write final Phase B review report** (inline by controller)

Inline review (read new READMEs + AUDIT files + check forbidden grep + confirm push status). No sub-agent dispatch needed (Phase A's working pattern).

Return review verdict to user for Phase B closeout.

---

## Self-Review (plan)

1. **Spec coverage:** Phase B spec covers 2 chapters + 8 acceptance criteria + reference mapping + per-chapter 4-段式 + 4 audit criteria. Plan covers each via Tasks 1 + 2 + 3. ✓
2. **Placeholder scan:** No TBDs. Every Step has concrete commands or file paths. ✓
3. **Type consistency:** Audit file format consistent across both tasks. Commit message format consistent. Reference fetch pattern consistent. ✓
4. **Forbidden content check** is explicit in Step 8 (per task) and Step 2 (final review). ✓
5. **Big-fix handling** is explicit: report back to user, don't commit. ✓
6. **Per-chapter granularity** allows failure isolation. ✓
7. **Audit scope adapted for Phase B** (2 files per chapter instead of 3). ✓
8. **No placeholders** like "similar to Task N" — each task explicitly enumerates its substitutions. ✓

**Adjustments made during self-review:**
- Originally planned Task 3 to dispatch a final reviewer sub-agent. Switched to inline review (controller) since Phase A confirmed the auto-classifier is conservative about cumulative sub-agent dispatch.
- Phase B budget for fixes is ≤ 5 lines per chapter, ≤ 10 lines cumulative (per spec §Acceptance criteria).

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-06-chapter-content-audit-phase-b.md`.

Executing via Subagent-Driven Development (per user's direct invocation). Phase A's working pattern carried: sub-agent implementer per task + inline review by controller + credential-helper push.
