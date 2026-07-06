# Chapter Content Audit — Phase C3 (s12_deployment) Design

> **Status:** Drafted 2026-07-06 (Phase C2 closed at commit 9bccf26). Awaiting user review before transitioning to writing-plans.

## Goal

Rewrite chapter-root `README.md` for **s12_deployment** to match the depth and structure of all-in-rag's corresponding chapters, while running a code audit that ensures the in-scope Python code delivers what the README promises. Mirror Phase C1 + C2 pattern (all shipped) — but with one **adaptation** specific to s12.

## Why this exists

- **Current state:** s12 README = 102 lines, no 是什么/为什么/怎么做/对照 RAGFlow 4-段式 structure, no explicit MVP function table. Below 200-line target.
- **Code state:** s12 has 1 unit (`units/01_fastapi_docker`) + chapter `code.py` (15-line aggregator) + `app.py` (FastAPI wrapper) + `Dockerfile` + `docker-compose.yml` (config). AUDIT.md does not exist.
- **Reference URL:** **No all-in-rag equivalent chapter** — s12 (Docker deployment) does not exist in upstream's chapter map. Per Phase C brainstorm, user chose "A: Adapt the template" (no all-in-rag ref, use loose-borrow fallback + adapt §四).
- **§四 adaptation:** Per Phase C brainstorm, user chose to rename §四 to **"对照生产部署实践"** (instead of "对照 RAGFlow 怎么做的") — because s12's comparison target is "production deployment practice in general," not a specific RAGFlow module.

## Architecture

- **Phased rollout (continued):** Phase A (s02-s06) + Phase B (s07-s08) + Phase B' (carryovers) + Phase C1 (s09) + Phase C2 (s10-s11) all shipped; Phase C3 = this spec.
- **Per-chapter workflow (mirrors Phase C1 + C2):** one task does README rewrite + code audit + (small) fixes in one commit. **1 chapter** in this sub-phase.
- **Borrowed structure DNA:** all-in-rag's 4-段式 (是什么 / 为什么 / 怎么做 / 对照 + 思考题) + comparison tables + decision paths + MVP patterns.
- **§四 adaptation:** "对照生产部署实践" — focus on production deployment best practices (multi-container orchestration, model weight management, monitoring, auth) as a comparison frame, not just RAGFlow internals.
- **Forbidden overlay (per `rag-intro-writing-style.md`):** no `[^N]` academic footnotes, no 参考文献 section, no 思辨/辩论 chapters, no "X is dead" sections, no verbatim copying of all-in-rag sentences.

## Tech Stack

Same as parent project + Phase A + B + C1 + C2 — Python 3.10+, FastAPI, Docker. **No new dependencies.** Plan execution via superpowers:subagent-driven-development (sub-agent implementer) + sub-agent reviewer (per C1 + C2 pattern, user-approved).

## Global Constraints

These apply to s12:

1. **Direct master push approved.** One commit for the chapter rewrite + audit. Followup commits for review fixes OK (C2 had 1 followup for AUDIT.md Important finding). No PR.
2. **Chapter README target:** 200-350 lines after rewrite. Quality over quantity — natural shorter is OK if all 4 段 are present.
3. **Unit READMEs untouched.** Phase C3 does not modify `units/01_*/README.md`.
4. **Code changes ≤ 5 lines per chapter.** Larger changes require user sign-off (out of Phase C3 scope). Phase C3 budget: ≤ 5 lines cumulative (single chapter).
5. **No new dependencies.** `requirements.txt` is unchanged. FastAPI / uvicorn / pydantic already in there (verified in app.py imports).
6. **No `[^N]` footnotes / 参考文献 sections / 思辨 chapters.** If reviewer flags any, treat as Critical.
7. **No verbatim copying from all-in-rag.** Allow ≤ 1 sentence of direct quote per chapter (terms of art / definitions OK). N/A here — no all-in-rag reference URL exists for s12.
8. **Project-specific content preserved (s12):**
   - FastAPI wrapper (`app.py` — `POST /qa` endpoint that chains s04 → s06 → s07 → s08)
   - Dockerfile (python:3.11-slim + tesseract-ocr + build-essential for chroma-hnswlib)
   - docker-compose.yml (single `rag` service, build context = project root, port 8000, mounted `.env` / `samples` / `_chroma`)
   - `code.py` (15-line aggregator, importlib delegation to unit 01)
   - 1-unit progression (fastapi_docker)
   - **Graceful-skip / gating pattern** in unit 01 (`.env` gating + 索引 gating + clear error messages)
   - **No LLM API call** in this chapter (deployment only) — `@lru_cache` model caching pattern from s08/s10 **not applicable** to s12
   - References to `thinking_answers.md`
   - Existing units nav table (extended to cover 4 supporting files)
   - Link to `ragflow_notes/deployment.md` (verified exists, 4093 bytes, comprehensive)
9. **Implementer must NOT `fetch` any all-in-rag equivalent** (per Phase C decision). Reference mapping: **N/A**. Borrow 4-段式 from `docs/00_introduction/01_what_is_rag.md` + s10 / s11 chapter pattern (closest project neighbors) + project-specific substitution for FastAPI + Docker + production adaptation.
10. **Single AUDIT.md** — one-page audit report for s12 (对齐 / 小修 / 大修 per criterion).
11. **Carryovers from prior phases (all addressed in their respective phases; no new carryovers for C3):**
    - s09 unit 01 EOFError guard (Phase C1' carryover; out of C3 scope)
    - All Phase A + B + B' + C1 + C2 carryovers (all addressed)

---

## Per-chapter 4-段式 template (Phase C3 — ADAPTED for s12)

Every chapter README must contain all four sections in this order. **§四 is renamed** per Phase C brainstorm decision:

```
# s12 <中文主题> (<English topic>)

> 一句话定位:本章节解决什么、给出什么。

## 一、是什么
   - 概念定义(≤ 5 行) — s12: 把 MVP 流水线"包成可分发服务"的工程抽象
   - 在 RAG 全链路中的位置(可链 01_what_is_rag.md) — s12: 第 12 章,最后一公里
   - 本章 unit 拆解的 rationale (1-2 段) — s12: 单 unit (fastapi_docker) 覆盖 FastAPI + Docker + compose 三件套

## 二、为什么 / 现实问题
   - 不用这个会崩在哪(2-4 条具体场景,借鉴 all-in-rag) — s12: 命令行脚本分享难 / 环境不一致 / 无法水平扩展 / 没有鉴权
   - 选型对比表(若适用 —— s12: FastAPI vs Flask vs Django; docker compose vs k8s; python:slim vs alpine vs distroless)
   - 引用 all-in-rag 对应章节的核心要点 (1-2 段, 改写不照抄) — N/A (no all-in-rag ref); 引用 ragflow_notes/deployment.md

## 三、怎么做 (MVP)
   - 本项目最小实现思路(代码架构 + 关键函数签名) — s12: app.py 3 函数 + Dockerfile 7 行 + compose 12 行 + unit 01 3-gate 启动器
   - 跑起来:命令 + 期望输出片段 — s12: docker compose up --build + curl /qa
   - 真实世界会遇到的问题 (2-4 条) — s12: 镜像大 / 冷启动慢 / 监控缺失 / 鉴权缺失

## 四、对照生产部署实践 + 思考题
   - 引用 ragflow_notes/deployment.md 的关键模块 (10+ 容器拆分)
   - 工业实现 vs MVP 的差距 (2-3 条) — s12: 单容器 vs 多容器 / 进程内 vs ES+MySQL+MinIO
   - 思考题 2-3 个(指向 thinking_answers.md)
```

> **§四 naming convention:** Use "**对照生产部署实践**" (not "对照 RAGFlow 怎么做的") — per Phase C brainstorm user decision. The reference frame for s12 is "production deployment practice" broadly, not just RAGFlow specifically.

---

## Code audit criteria (Phase C3 — same 4 criteria as Phase A + B + C1 + C2)

The chapter audit (`s12_deployment/AUDIT.md`) must report on all 4 criteria, applied to the **3 in-scope Python code files** (chapter `code.py` + 1 unit `code.py` + `app.py` FastAPI wrapper):

| Criterion | Question | Result tier |
|---|---|---|
| **1. README 声称的函数/输出 → 代码里有吗？** | Does the 3-file in-scope code set export every function/API the README documents? | 对齐 / 小修 / 大修 |
| **2. 代码里的主要函数 → README 解释了吗？** | Does the README explain input/output/purpose of every non-trivial function in the 3 files? | 对齐 / 小修 / 大修 |
| **3. README 里的运行示例 → 真能跑出那个输出吗？** | Are sample outputs in README still accurate (post-run) for the 3 files? | 对齐 / 小修 / 大修 |
| **4. Dead code / orphan import** | Any obvious lint issues in the 3 files (unused imports, dead vars)? | 对齐 / 小修 / 大修 |

> **Audit scope clarification:** s12 has 5 supporting files: `code.py` (15 lines) + `app.py` (51 lines) + `Dockerfile` (11 lines) + `docker-compose.yml` (13 lines) + `units/01_fastapi_docker/code.py` (37 lines). **In-scope for the Python audit:** `code.py` + `app.py` + `units/01_fastapi_docker/code.py` (3 files, 103 lines total). **Out of audit scope (config, not Python):** `Dockerfile` + `docker-compose.yml` (referenced in README but not audited for Python lint).

**Tier definitions (unchanged from Phase A + B + C1 + C2):**
- **对齐** — no change needed
- **小修** — ≤ 5 lines code or doc edit
- **大修** — > 5 lines; requires user sign-off before commit

All 小修 are in-scope for the implementer. All 大修 are reported back to the user (not committed).

---

## Task structure (Phase C3 = 1 chapter + 1 final review)

### Task 1: s12 deployment
- **Files to rewrite:** `s12_deployment/README.md`
- **Files to audit (3 Python files):** `s12_deployment/code.py` (15 lines) + `s12_deployment/app.py` (51 lines) + `s12_deployment/units/01_fastapi_docker/code.py` (37 lines)
- **Files referenced but not audited:** `s12_deployment/Dockerfile` + `s12_deployment/docker-compose.yml` (config, not Python)
- **Files to produce:** `s12_deployment/AUDIT.md`
- **Reference fetch:** **N/A** (per Phase C decision; no all-in-rag equivalent). Use loose-borrow from `docs/00_introduction/01_what_is_rag.md` + s10/s11 chapter pattern + project-specific content.
- **Commit message:** `s12: rewrite chapter README + code audit`

### Task 2: Whole-Phase C3 review
After Task 1, dispatch final review (inline, per C1 + C2 pattern) to confirm:
- Chapter follows 4-段式 with §四 renamed to "对照生产部署实践"
- No forbidden content
- AUDIT.md exists and is not a no-op
- No regressions: `code.py` (aggregator) and `units/01_fastapi_docker/code.py` still run; `app.py` still imports cleanly
- Master IN SYNC at final commit

---

## Execution pattern (Phase C3)

Carried from Phase C1 + C2 working pattern:
- **Implementer sub-agent (sonnet)** dispatched per task — same per-task brief format, same forbidden-content guard, same credential-helper push pattern
- **Reviewer sub-agent (sonnet)** dispatched per task — spec compliance + task quality review (C1 + C2 pattern; user explicitly chose "Full SDD w/ auto-classifier dispatch")
- **Final review inline by controller** (C1 + C2 pattern — read brief + report + spot-read new README + grep checks + commit-stat review)
- **Cumulative dispatch re-authorization:** user re-authorized this dispatch pattern for Phase C ("Full SDD w/ auto-classifier dispatch"). Phase C3 dispatches fall under the same re-authorization.

---

## Acceptance criteria

Phase C3 complete when ALL of:

1. ✅ s12 chapter-root README rewritten to ≥ 200 lines (or natural shorter with all 4 段 present)
2. ✅ README contains all 4 sections in order (是什么 / 为什么 / 怎么做 / **对照生产部署实践** + 思考题)
3. ✅ s12/AUDIT.md exists with all 4 criteria reported
4. ✅ Every 小修 has been committed + pushed
5. ✅ Every 大修 has been reported back to user with sign-off decision
6. ✅ `s12_deployment/code.py` (aggregator) and `units/01_fastapi_docker/code.py` still run; `app.py` imports cleanly
7. ✅ `git grep -E '\[\^[0-9]|RAG 已死|参考文献'` returns 0 matches across `s12_deployment` README + AUDIT.md
8. ✅ Master IN SYNC at final commit

---

## Out of scope (Phase C3)

- Phase C1 (s09_agent_tools) — shipped at commit 0a772bd
- Phase C2 (s10_graphrag + s11_multimodal) — shipped at commit 9bccf26
- `ragflow_notes/deployment.md` (existing content; referenced but not modified)
- Unit-level README (`units/01_fastapi_docker/README.md`)
- `README.en.md` English version (out of scope for all phases)
- Adding new units to s12 (feature addition, not audit)
- New features / new dependencies
- Any code change > 5 lines in s12
- `Dockerfile` + `docker-compose.yml` lint (config, not Python; referenced but not audited)
- s09 unit 01 EOFError guard (Phase C1' carryover)
- All Phase A + B + B' + C1 + C2 carryovers (all addressed in their respective phases)

---

## Self-review (spec)

1. **Placeholder scan:** No TBDs. Every § has content. Reference URLs explicit (N/A for s12). ✓
2. **Internal consistency:** §1 scope matches §5 acceptance criteria. Task list matches constraints in §6. ✓
3. **Scope check:** 1 chapter × 3 Python files = 3 files touched + 1 AUDIT.md created. Bounded, mirrors C2's 6-file total (per chapter). ✓
4. **Ambiguity check:**
   - "Audit" defined via 4 criteria with tier definitions (same as C1 + C2). ✓
   - "Forbidden content" listed explicitly. ✓
   - "§四 renaming to 对照生产部署实践" — explicit and explained (per user decision in Phase C brainstorm). ✓
   - "Audit scope = 3 Python files (excluding Dockerfile + compose)" — explicit. ✓
   - "No all-in-rag reference" — explicit (N/A in §1 + §6 + §8). ✓
5. **Phase A + B + C1 + C2 execution pattern carried forward:** Sub-agent implementer + sub-agent reviewer per task, inline final review, credential-helper push. ✓
6. **Carryovers from prior phases documented as out-of-scope** so they don't surprise user mid-execution. ✓
7. **Phase C split (C1/C2/C3) is per user direction** (chose Three sub-phases option in brainstorm AskUserQuestion 2026-07-06). ✓

## Open questions for user review

- 是否同意 s12 §四 改名为"对照生产部署实践"(放弃"对照 RAGFlow 怎么做的"沿用)?
- 是否同意 audit scope 只覆盖 3 个 Python 文件(Dockerfile + compose 引用但不审)?
- 是否同意 Phase C3 沿用 Phase C1+C2 的 sub-agent implementer + sub-agent reviewer 模式?
- 是否同意 Phase C3 沿用 "≤ 5 行 per chapter" 修复预算?
- 是否同意 s09 unit 01 EOFError guard 不进 Phase C3(预算独立,C1' followup 时处理)?
