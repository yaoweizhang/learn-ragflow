# Phase C3 Chapter Content Audit Implementation Plan (s12_deployment)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite chapter-root `README.md` for s12_deployment to 4-段式 (with §四 renamed to "对照生产部署实践"), run a code audit on 3 in-scope Python files (chapter `code.py` + `app.py` + `units/01_fastapi_docker/code.py`), fix small mismatches (≤ 5 lines), and produce `AUDIT.md`.

**Architecture:** Mirror Phase C1 + C2 pattern. 2 tasks: Task 1 s12 + Task 2 final review. Each task: subagent implementer (sonnet) reads current state → rewrites README (no all-in-rag fetch, use loose-borrow from intro doc + s10/s11 chapter pattern + project-specific) → runs code audit on 3 Python files → applies small fixes → commits + pushes; subagent reviewer (sonnet) reviews spec compliance + task quality; controller reviews final review inline. Direct master push (user-approved). **One** new commit (s12) plus optional followup(s) for review fixes.

**Tech Stack:** Python 3.10+, FastAPI, Docker, Markdown, Git. No new dependencies. Implements spec at `docs/superpowers/specs/2026-07-06-chapter-content-audit-phase-c3.md`.

## Global Constraints

These apply to every task. **Verify all are met before marking task complete.**

1. **Direct master push approved.** One primary commit for Task 1 (`s12: rewrite chapter README + code audit`). Followup commits for reviewer-flagged Critical/Important findings are allowed (Phase C2 precedent: 1 followup for AUDIT.md). Use credential-helper pattern (per-task brief contains the canonical command).
2. **Chapter README target: 200-350 lines after rewrite.** Quality over quantity — natural shorter is OK if all 4 段 are present.
3. **Unit READMEs untouched** (`units/01_fastapi_docker/README.md` not modified). Code in `units/01_fastapi_docker/code.py` is in audit scope.
4. **Code changes ≤ 5 lines total (single chapter).** Larger changes require user sign-off (report, don't commit). Phase C3 budget: ≤ 5 lines cumulative.
5. **No new dependencies.** `requirements.txt` is unchanged.
6. **No `[^N]` footnotes / 参考文献 sections / 思辨 chapters.** If reviewer flags any, treat as Critical.
7. **No verbatim copying from all-in-rag.** Allow ≤ 1 sentence of direct quote per chapter (terms of art / definitions OK). N/A here — no all-in-rag reference URL exists for s12.
8. **Project-specific content preserved:** Per spec §Global Constraint 8 (FastAPI wrapper, Dockerfile, docker-compose, code.py aggregator, 1-unit progression, graceful-skip / gating pattern, no LLM call, references to `thinking_answers.md`, link to `ragflow_notes/deployment.md`).
9. **Implementer must NOT `fetch` any all-in-rag equivalent** (per Phase C decision). Use loose-borrow from `docs/00_introduction/01_what_is_rag.md` + s10 / s11 chapter pattern + project-specific substitution for FastAPI + Docker + production adaptation.
10. **§四 renamed to "对照生产部署实践"** (per Phase C brainstorm decision). Reference frame for s12 is "production deployment practice" broadly, not just RAGFlow specifically.
11. **Single chapter produces `s12_deployment/AUDIT.md`** — 4-criterion report (对齐 / 小修 / 大修 for each), covering the **3 in-scope Python files** (`code.py` + `app.py` + `units/01_fastapi_docker/code.py`).
12. **Audit scope: 3 Python files only.** `Dockerfile` + `docker-compose.yml` are referenced in README but NOT audited for Python lint (config, not Python).
13. **Forbidden content self-check** at end of every task: `git grep -E '\[\^[0-9]|RAG 已死|参考文献' <changed files>` returns 0 matches.
14. **No regressions**: every `units/*/code.py` for that chapter still runs (verified by implementer before commit). `app.py` imports cleanly.

### Reference mapping (per spec Global Constraint 9)

| Chapter | all-in-rag reference URL | Status |
|---|---|---|
| s12 | N/A (no all-in-rag equivalent) | Use loose-borrow from `docs/00_introduction/01_what_is_rag.md` + s10/s11 chapter pattern + project-specific content |

**Fallback pattern (carried from C1 + C2):** Borrow 4-段式 structural DNA from `docs/00_introduction/01_what_is_rag.md` + s10 / s11 chapter pattern + project-specific substitution for FastAPI wrapper + Dockerfile + docker-compose.

### The 4-段式 template (s12 — ADAPTED per Phase C decision)

```
# s12 <中文主题> (<English topic>)

> 一句话定位:本章节解决什么、给出什么。

## 一、是什么
   - 概念定义(≤ 5 行)
   - 在 RAG 全链路中的位置(可链 01_what_is_rag.md)
   - 本章 unit 拆解的 rationale (1-2 段)

## 二、为什么 / 现实问题
   - 不用这个会崩在哪(2-4 条具体场景)
   - 选型对比表(若适用 — s12: FastAPI vs Flask vs Django; docker compose vs k8s; python:slim vs alpine vs distroless)
   - 引用 ragflow_notes/deployment.md 的核心要点 (1-2 段, 改写不照抄)

## 三、怎么做 (MVP)
   - 本项目最小实现思路(代码架构 + 关键函数签名)
   - 跑起来:命令 + 期望输出片段
   - 真实世界会遇到的问题 (2-4 条)

## 四、对照生产部署实践 + 思考题
   - 引用 ragflow_notes/deployment.md 的关键模块 (10+ 容器拆分)
   - 工业实现 vs MVP 的差距 (2-3 条)
   - 思考题 2-3 个(指向 thinking_answers.md)
```

> **§四 naming convention:** Use "**对照生产部署实践**" (not "对照 RAGFlow 怎么做的").

### The 4-criterion audit (Phase C3 — same 4 criteria as Phase A + B + C1 + C2)

| Criterion | Question | Result tier |
|---|---|---|
| **1. README 声称的函数/输出 → 代码里有吗？** | Does the 3-file in-scope code set export every function/API the README documents? | 对齐 / 小修 / 大修 |
| **2. 代码里的主要函数 → README 解释了吗？** | Does the README explain input/output/purpose of every non-trivial function in the 3 files? | 对齐 / 小修 / 大修 |
| **3. README 里的运行示例 → 真能跑出那个输出吗？** | Are sample outputs in README still accurate (post-run) for the 3 files? | 对齐 / 小修 / 大修 |
| **4. Dead code / orphan import** | Any obvious lint issues in the 3 files (unused imports, dead vars)? | 对齐 / 小修 / 大修 |

**Tier definitions:** 对齐 (no change), 小修 (≤ 5 lines), 大修 (> 5 lines, requires user sign-off).

---

### Task 1: s12 deployment

**Files:**
- Rewrite: `s12_deployment/README.md`
- Audit (read + 小修 ≤ 5 lines): `s12_deployment/code.py` (15 lines) + `s12_deployment/app.py` (51 lines) + `s12_deployment/units/01_fastapi_docker/code.py` (37 lines)
- Reference only (NOT audited): `s12_deployment/Dockerfile` + `s12_deployment/docker-compose.yml` (config, not Python)
- Create: `s12_deployment/AUDIT.md`

**Reference fetch (N/A per Phase C decision):**
- No all-in-rag equivalent for s12 deployment
- Use loose-borrow fallback:
  - Borrow 4-段式 from `docs/00_introduction/01_what_is_rag.md`
  - Borrow s10 / s11 chapter pattern (closest project neighbors with code)
  - Substitute project-specific content: FastAPI wrapper + Dockerfile + docker-compose + 1-unit progression

**Interfaces:**
- Consumes: nothing (Phase C3 first task; Phase C2 s10 + s11 already shipped — implementer may reference s10/s11 README for pattern consistency)
- Produces: new s12 README, s12/AUDIT.md, possibly patched code (≤ 5 lines)

- [ ] **Step 1: Capture base SHA**

```bash
git rev-parse HEAD | tee /tmp/s12-base.txt
```

- [ ] **Step 2: Read current state**

```bash
sed -n '1,200p' s12_deployment/README.md
echo "---unit READMEs (do not edit)---"
ls s12_deployment/units/01_fastapi_docker/
echo "---code files (3 in-scope)---"
cat s12_deployment/code.py
cat s12_deployment/app.py
cat s12_deployment/units/01_fastapi_docker/code.py
echo "---config (referenced but not audited)---"
cat s12_deployment/Dockerfile
cat s12_deployment/docker-compose.yml
```

- [ ] **Step 3: Borrow 4-段式 DNA (no fetch — Phase C decision)**

No `curl`. Borrow structure from:
- `docs/00_introduction/01_what_is_rag.md` (4-段式 reference)
- `s10_graphrag/README.md` (closest project chapter pattern, 270 lines)
- `s11_multimodal/README.md` (other recent chapter, 233 lines)

Substitute project-specific content for FastAPI + Docker + production adaptation.

- [ ] **Step 4: Draft new `s12_deployment/README.md`**

Use the adapted 4-段式 template (with §四 = "对照生产部署实践"). 200-350 lines. Project-specific content to preserve (per spec §Global Constraint 8):
- FastAPI wrapper (`app.py` — `POST /qa` endpoint, chains s04 → s06 → s07 → s08)
- Dockerfile (python:3.11-slim + tesseract-ocr + build-essential for chroma-hnswlib)
- docker-compose.yml (single `rag` service, build context = project root, port 8000, mounted `.env` / `samples` / `_chroma`)
- `code.py` (15-line aggregator, importlib delegation to unit 01)
- 1-unit progression (fastapi_docker)
- Graceful-skip / gating pattern in unit 01 (`.env` gating + 索引 gating + clear error messages)
- **No LLM API call** in this chapter (deployment only) — `@lru_cache` model caching pattern from s08/s10 **not applicable** to s12; document this explicitly
- References to `thinking_answers.md`
- Existing units nav table (extended to cover 4 supporting files)
- Link to `ragflow_notes/deployment.md` (verified exists, 4093 bytes)
- §四 "对照生产部署实践" — references ragflow_notes/deployment.md's 10+ container split, MVP vs production gap analysis

Forbidden self-check before saving: no `[^N]`, no "RAG 已死", no "参考文献" section.

- [ ] **Step 5: Run code audit — 4 criteria**

Read each of the 3 in-scope code files. For each criterion, decide 对齐 / 小修 / 大修. Record in `s12_deployment/AUDIT.md`:

```markdown
# s12 Deployment — Code Audit

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

The 3 in-scope code files export these functions/objects (verify and document):
- `s12_deployment/code.py`: `main` (re-exported from unit 01)
- `s12_deployment/app.py`: `app` (FastAPI), `QARequest` (pydantic model), `_get_col` (helper), `qa` (POST /qa handler)
- `s12_deployment/units/01_fastapi_docker/code.py`: `main` (gating + subprocess docker compose up)

- [ ] **Step 6: Apply 小修 (≤ 5 lines total)**

```bash
git diff --stat s12_deployment/
```

If diff > 5 lines, revert and report as 大修 instead.

- [ ] **Step 7: Verify code still runs**

```bash
# Test 1: aggregator (code.py) — should delegate to unit 01
# Skip if Docker not installed; if not, expect gating message + clean exit
python s12_deployment/code.py 2>&1 | tail -5

# Test 2: app.py imports cleanly (FastAPI / uvicorn / pydantic in env)
python -c "from s12_deployment.app import app, qa, _get_col, QARequest; print('app.py imports OK')" 2>&1 | tail -3

# Test 3: unit 01 main() — expect gating message (no .env or no _chroma in sandbox)
python -c "from s12_deployment.units.01_fastapi_docker.code import main; main()" 2>&1 | tail -5
```

All 3 should exit 0 or print a graceful-skip message. Document which paths hit gating vs which complete the actual flow.

- [ ] **Step 8: Forbidden-content self-check**

```bash
git grep -nE '\[\^[0-9]|RAG 已死|参考文献' s12_deployment/README.md s12_deployment/AUDIT.md
```

Expected: empty output.

- [ ] **Step 9: Commit**

```bash
git add s12_deployment/README.md s12_deployment/AUDIT.md s12_deployment/code.py s12_deployment/app.py s12_deployment/units/01_fastapi_docker/code.py
git commit -m "s12: rewrite chapter README + code audit"
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
Task 1 (phase C3): complete (commits <base>..<head>)
  s12 deployment: README rewritten to 4-段式 (XXX lines); AUDIT.md emitted; N small fixes.
```

- [ ] **Step 12: Write report to `.superpowers/sdd/task-1-report.md`**

Include status, README line count, audit summary (each criterion tier), commit hash, verification output (last 5 lines), concerns.

Return ≤ 15 lines to controller with status DONE / DONE_WITH_CONCERNS / BLOCKED.

---

### Task 2: Whole-Phase C3 review

**Files:**
- Read: `s12_deployment/AUDIT.md`
- Read: `s12_deployment/README.md`
- Read: `.superpowers/sdd/task-1-report.md`

- [ ] **Step 1: Capture base range**

```bash
git rev-parse HEAD | tee /tmp/phase-c3-base.txt
```

- [ ] **Step 2: Verify chapter meets acceptance criteria**

For s12:
1. README ≥ 200 lines OR all 4 段 present
2. AUDIT.md exists with 4 criteria reported
3. No forbidden content
4. §四 titled "对照生产部署实践" (NOT "对照 RAGFlow 怎么做的")

- [ ] **Step 3: Verify no regressions**

```bash
# Aggregator + unit + app.py imports
python -c "from s12_deployment.app import app; print('OK')" 2>&1 | tail -3
python s12_deployment/code.py 2>&1 | tail -3
python -c "from s12_deployment.units.01_fastapi_docker.code import main; main()" 2>&1 | tail -3
```

- [ ] **Step 4: Verify push status**

```bash
git fetch origin master
LOCAL=$(git rev-parse HEAD); REMOTE=$(git rev-parse origin/master)
[ "$LOCAL" = "$REMOTE" ] && echo "IN SYNC @ $LOCAL" || echo "DRIFT"
```

- [ ] **Step 5: Append final ledger entry**

```
Phase C3: complete (commits <base>..<head>)
  - s12 rewritten + audited
  - 1 primary commit (+ N followup commits if reviewer found Critical/Important issues)
  - <N> small fixes total, <N> big fixes reported
  - 0 forbidden content matches
```

- [ ] **Step 6: Write final Phase C3 review report** (inline by controller)

Inline review (read new README + AUDIT file + check forbidden grep + confirm push status). Return review verdict to user for Phase C3 closeout.

---

## Self-Review (plan)

1. **Spec coverage:** Phase C3 spec covers 1 chapter + 8 acceptance criteria + reference mapping (N/A explicit) + per-chapter 4-段式 (with §四 renamed) + 4 audit criteria. Plan covers each via Tasks 1 + 2. ✓
2. **Placeholder scan:** No TBDs. Every Step has concrete commands or file paths. ✓
3. **Type consistency:** Audit file format consistent with C1 + C2. Commit message format consistent. ✓
4. **Forbidden content check** is explicit in Step 8 (per task) and Step 2 (final review). ✓
5. **Big-fix handling** is explicit: report back to user, don't commit. ✓
6. **Audit scope adapted for Phase C3** (3 Python files: code.py + app.py + units/01_fastapi_docker/code.py; Dockerfile + compose referenced but not audited). ✓
7. **§四 renaming** explicit in template and Task 1 acceptance criteria. ✓
8. **No placeholders** like "similar to Task N" — Task 2 references Task 1's report by name. ✓
9. **No all-in-rag fetch** — explicit in Step 3 of Task 1; loose-borrow fallback carried from C1 + C2. ✓
10. **Verification paths** for 3 in-scope code files (aggregator, app.py imports, unit 01 main). ✓

**Adjustments made during self-review:**
- §四 renamed to "对照生产部署实践" per Phase C decision (explicit in template and Task 2 Step 2 acceptance check).
- Audit scope explicit: 3 Python files only, Dockerfile + compose referenced but not audited.
- No curl in Step 3 (no all-in-rag equivalent for s12).
- Verification paths cover all 3 in-scope code files individually.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-06-chapter-content-audit-phase-c3.md`.

Executing via Subagent-Driven Development (per user's "Full SDD w/ auto-classifier dispatch" choice in Phase C). Phase C1 + C2 working pattern carried: sub-agent implementer + sub-agent reviewer per task + inline final review + credential-helper push.
