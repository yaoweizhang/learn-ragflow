# Phase C2 Chapter Content Audit Implementation Plan (s10_graphrag + s11_multimodal)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite chapter-root `README.md` for s10_graphrag + s11_multimodal to match all-in-rag's depth (200-350 lines, 4-段式), run code audits covering 3 files each (chapter `code.py` + 2 unit `code.py`), fix small mismatches (≤ 5 lines per chapter), and produce `AUDIT.md` per chapter.

**Architecture:** Mirror Phase C1 pattern. 3 tasks: Task 1 s10 + Task 2 s11 + Task 3 final review. Each task: subagent implementer (sonnet) reads current state + fetches all-in-rag reference (expected 404, fallback used) → rewrites README → runs code audit → applies small fixes → commits + pushes; subagent reviewer (sonnet) reviews spec compliance + task quality; controller reviews final review inline. Direct master push (user-approved).

**Tech Stack:** Python 3.10+, Markdown, Git. No new dependencies. Implements spec at `docs/superpowers/specs/2026-07-06-chapter-content-audit-phase-c2.md`.

## Global Constraints

These apply to every task. **Verify all are met before marking task complete.**

1. **Direct master push approved.** One commit per chapter (2 commits in Tasks 1 + 2). Subject format: `sNN: rewrite chapter README + code audit`. Use credential-helper pattern (per-task brief contains the canonical command).
2. **Chapter README target: 200-350 lines after rewrite.** Quality over quantity — natural shorter is OK if all 4 段 are present.
3. **Unit READMEs untouched** (`units/01_*/README.md` and `units/02_*/README.md` not modified). Code in `units/*/code.py` is in audit scope.
4. **Code changes ≤ 5 lines per chapter.** Larger changes require user sign-off (report, don't commit). Phase C2 budget: ≤ 10 lines cumulative across both chapters.
5. **No new dependencies.** `requirements.txt` is unchanged.
6. **No `[^N]` footnotes / 参考文献 sections / 思辨 chapters.** If reviewer flags any, treat as Critical.
7. **No verbatim copying from all-in-rag.** Allow ≤ 1 sentence of direct quote per chapter (terms of art / definitions OK). Everything else rewritten in our project's voice.
8. **Project-specific content preserved:** Per spec Global Constraint 8 (s10: GraphRAG specifics; s11: multimodal specifics).
9. **Implementer must `fetch` all-in-rag's equivalent sub-file before writing** (per chapter mapping below). **C1 lesson: URLs expected 404; use loose-borrow fallback.**
10. **Each chapter produces `sNN_topic/AUDIT.md`** — 4-criterion report (对齐 / 小修 / 大修 for each).
11. **Forbidden content self-check** at end of every task: `git grep -E '\[\^[0-9]|RAG 已死|参考文献' <changed files>` returns 0 matches.
12. **No regressions**: every `units/*/code.py` for that chapter still runs (verified by implementer before commit).

### Reference mapping (per spec Global Constraint 9; C1 lesson applied)

| Chapter | all-in-rag reference URL | Status (C1 lesson) |
|---|---|---|
| s10 | `https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter4/13_graph_rag.md` | Expected 404 (C1 precedent); use loose-borrow fallback |
| s11 | `https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter5/15_multimodal.md` | Expected 404 (C1 precedent); use loose-borrow fallback |

**Fallback pattern (from C1):** When URL returns 404, borrow 4-段式 structural DNA from `docs/00_introduction/01_what_is_rag.md` + s09 chapter pattern + project-specific substitution. Document the 404 in the report; do not fabricate content.

### The 4-段式 template (every chapter README must contain all 4)

Same template as Phase C1. Reference: `docs/superpowers/specs/2026-07-06-chapter-content-audit-phase-c2.md` §Per-chapter 4-段式 template.

### The 4-criterion audit (every chapter must produce `sNN_topic/AUDIT.md`)

Same as Phase C1. Reference: spec §Code audit criteria.

---

### Task 1: s10 graphrag

**Files:**
- Rewrite: `s10_graphrag/README.md`
- Audit (read + 小修 ≤ 5 lines): `s10_graphrag/code.py`, `s10_graphrag/units/01_extract/code.py`, `s10_graphrag/units/02_query/code.py`
- Create: `s10_graphrag/AUDIT.md`

**Reference fetch (expected 404):**
```bash
curl -sL --max-time 15 "https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter4/13_graph_rag.md" > /tmp/air-s10.md
# If 404 (likely), use loose-borrow fallback per C1 lesson:
# - Borrow 4-段式 from docs/00_introduction/01_what_is_rag.md
# - Borrow s09 chapter pattern (ReAct + 2 tools) for tool/extract analogy
# - Substitute project-specific content (hand-rolled LLM-based entity extraction)
```

**Interfaces:**
- Consumes: nothing (Phase C2 first task; Phase C1 s09 already shipped — implementer may reference s09 README for pattern consistency)
- Produces: new s10 README, s10/AUDIT.md, possibly patched code (≤ 5 lines)

- [ ] **Step 1: Capture base SHA**

```bash
git rev-parse HEAD | tee /tmp/s10-base.txt
```

- [ ] **Step 2: Read current state**

```bash
sed -n '1,200p' s10_graphrag/README.md
echo "---unit READMEs (do not edit)---"
ls s10_graphrag/units/01_extract/
ls s10_graphrag/units/02_query/
echo "---code files---"
cat s10_graphrag/code.py
cat s10_graphrag/units/01_extract/code.py
cat s10_graphrag/units/02_query/code.py
```

- [ ] **Step 3: Fetch all-in-rag reference (expect 404)**

```bash
curl -sL --max-time 15 "https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter4/13_graph_rag.md" > /tmp/air-s10.md
wc -l /tmp/air-s10.md
# If 404 (likely per C1), document in report and proceed with loose-borrow fallback.
# Fallback references:
#   - docs/00_introduction/01_what_is_rag.md (4-段式 DNA)
#   - s09_agent_tools/README.md (chapter pattern)
```

- [ ] **Step 4: Draft new `s10_graphrag/README.md`**

Use the 4-段式 template. 200-350 lines. Project-specific content to preserve (per spec §Global Constraint 8):
- LLM-based entity extraction
- Knowledge graph schema (`head / relation / tail`)
- Query via community-detection or path-traversal
- 2-unit progression (extract → query)
- `@lru_cache` model caching pattern
- `MiniMax-M3 over minimaxi.com` LLM provider example
- Existing units nav table
- Link to `ragflow_notes/graph_extraction.md` (verify it exists)
- References to `thinking_answers.md`

Forbidden self-check before saving: no `[^N]`, no "RAG 已死", no "参考文献" section.

- [ ] **Step 5: Run code audit — 4 criteria**

Read each code file. For each criterion, decide 对齐 / 小修 / 大修. Record in `s10_graphrag/AUDIT.md`:

```markdown
# s10 GraphRAG — Code Audit

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

```bash
git diff --stat s10_graphrag/
```

If diff > 5 lines, revert and report as 大修 instead.

- [ ] **Step 7: Verify unit code still runs**

```bash
timeout 30 python s10_graphrag/units/01_extract/code.py 2>&1 | tail -5
timeout 30 python s10_graphrag/units/02_query/code.py 2>&1 | tail -5
```

Both must exit 0. (If running into HF Hub timeout without cached models, set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.)

- [ ] **Step 8: Forbidden-content self-check**

```bash
git grep -nE '\[\^[0-9]|RAG 已死|参考文献' s10_graphrag/README.md s10_graphrag/AUDIT.md
```

Expected: empty output.

- [ ] **Step 9: Commit**

```bash
git add s10_graphrag/README.md s10_graphrag/AUDIT.md s10_graphrag/code.py s10_graphrag/units/01_extract/code.py s10_graphrag/units/02_query/code.py
git commit -m "s10: rewrite chapter README + code audit"
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

Verify with `git fetch origin master` + `[LOCAL == REMOTE]` (sandbox TLS may fail; trust `git push` output line).

- [ ] **Step 11: Update local ledger**

Append to `.superpowers/sdd/progress.md`:

```
Task 1 (phase C2): complete (commits <base10>..<head10>)
  s10 graphrag: README rewritten to 4-段式 (XXX lines); AUDIT.md emitted; N small fixes.
```

- [ ] **Step 12: Write report to `.superpowers/sdd/task-1-report.md`**

Include status, README line count, audit summary (each criterion tier), commit hash, verification output (last 5 lines), concerns.

Return ≤ 15 lines to controller with status DONE / DONE_WITH_CONCERNS / BLOCKED.

---

### Task 2: s11 multimodal

**Files:**
- Rewrite: `s11_multimodal/README.md`
- Audit (read + 小修 ≤ 5 lines): `s11_multimodal/code.py`, `s11_multimodal/units/01_table_extract/code.py`, `s11_multimodal/units/02_ocr/code.py`
- Create: `s11_multimodal/AUDIT.md`

**Reference fetch (expected 404):**
```bash
curl -sL --max-time 15 "https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter5/15_multimodal.md" > /tmp/air-s11.md
# Same 404 fallback as Task 1.
```

**Same workflow as Task 1, with these substitutions:**
- 4-段式 template (s11 specifics: pdfplumber + pytesseract/PIL)
- Project-specific content (per spec §Global Constraint 8):
  - pdfplumber for table extraction (s11 unit 01)
  - pytesseract/PIL for OCR (s11 unit 02; Phase A sweep already fixed import order — verify)
  - 2-unit progression (tables → OCR)
  - Graceful-skip pattern when tesseract binary is missing
  - `samples/` file references
  - References to `thinking_answers.md`
  - Existing units nav table
  - Link to `ragflow_notes/multimodal_parsing.md` (verify it exists)
- Commit message: `s11: rewrite chapter README + code audit`
- Document any pre-existing Phase A sweep fixes in audit (don't re-fix)

Steps 1-12 identical structure to Task 1.

---

### Task 3 (Final): Whole-Phase C2 review

**Files:**
- Read: `s10_graphrag/AUDIT.md`, `s11_multimodal/AUDIT.md`
- Read: `s10_graphrag/README.md`, `s11_multimodal/README.md`

- [ ] **Step 1: Capture base range**

```bash
git rev-parse HEAD | tee /tmp/phase-c2-base.txt
```

- [ ] **Step 2: Verify both chapters meet acceptance criteria**

For each chapter s10 + s11:
1. README ≥ 200 lines OR all 4 段 present
2. AUDIT.md exists with 4 criteria reported
3. No forbidden content

- [ ] **Step 3: Verify no regressions**

```bash
for ch in s10_graphrag s11_multimodal; do
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
Phase C2: complete (commits <base>..<head>)
  - s10 / s11 all rewritten + audited
  - 2 commits, both IN SYNC
  - <N> small fixes total, <N> big fixes reported
  - 0 forbidden content matches
```

- [ ] **Step 6: Write final Phase C2 review report** (inline by controller)

Inline review (read new READMEs + AUDIT files + check forbidden grep + confirm push status). Return review verdict to user for Phase C2 closeout.

---

## Self-Review (plan)

1. **Spec coverage:** Phase C2 spec covers 2 chapters + 8 acceptance criteria + reference mapping (with C1 404 lesson) + per-chapter 4-段式 + 4 audit criteria. Plan covers each via Tasks 1 + 2 + 3. ✓
2. **Placeholder scan:** No TBDs. Every Step has concrete commands or file paths. ✓
3. **Type consistency:** Audit file format consistent with C1. Commit message format consistent. Reference fetch pattern consistent (with C1 404 lesson). ✓
4. **Forbidden content check** is explicit in Step 8 (per task) and Step 2 (final review). ✓
5. **Big-fix handling** is explicit: report back to user, don't commit. ✓
6. **Per-chapter granularity** allows failure isolation. ✓
7. **Audit scope adapted for Phase C2** (3 files per chapter: chapter + 2 units). ✓
8. **No placeholders** like "similar to Task N" — Task 2 explicitly enumerates its substitutions. ✓

**Adjustments made during self-review:**
- Reference mapping carries C1 404 lesson: implementer expected to use loose-borrow fallback when URL returns 404.
- Phase C2 budget for fixes is ≤ 5 lines per chapter, ≤ 10 lines cumulative (per spec §Global Constraint 4).

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-06-chapter-content-audit-phase-c2.md`.

Executing via Subagent-Driven Development (per user's "Full SDD w/ auto-classifier dispatch" choice in C1). Phase C1 working pattern carried: sub-agent implementer + sub-agent reviewer per task + inline final review + credential-helper push.