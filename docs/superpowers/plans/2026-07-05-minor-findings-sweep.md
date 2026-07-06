# Minor Findings Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address the 3 actionable minor findings left over from the s02-s12 units restructure (Tasks 1-12 of the prior plan) — verified against current `master` HEAD `5c44084`.

**Architecture:** Each task is a per-chapter, single-file (or 2-file) edit with no new dependencies. No new tests (project is print-based MVP). Each task ends with running the affected unit and confirming behavior is unchanged.

**Tech Stack:** Python 3.10+; same deps as parent project (no additions).

## Global Constraints

These apply to every task:

- **Direct master push approved.** No PR. Use `git push` with the credential-helper pattern from `.superpowers/sdd/progress.md` Task 12.
- **Working tree must start clean** at the beginning of every task. Run `git status` first; if dirty, stop.
- **Verification per task**: re-run the affected `units/NN/code.py` and confirm output is unchanged (modulo the fix).
- **No new dependencies.** `requirements.txt` is unchanged.
- **Existing importlib 聚合 pattern at chapter-root is untouched.** Only fix the specific lines flagged.
- **Commit message format**: `<chapter>: <one-line description>` (e.g. `s04: remove dead WORKDIR + unnecessary f-string`).
- **Ledger update**: append one line to `.superpowers/sdd/progress.md` at the end of every task.
- **Findings explicitly NOT in scope** (verified as false positives or intentional tradeoffs — DO NOT re-fix):
  - s07 unit 01 README "32 chunks" claim — no such string in current README
  - s10 unit 01 README "8 nodes" claim — no such string in current README
  - s04 chapter README old `embed()` reference — README shows correct current signature
  - Trailing newlines across all new files — verified all files end with `\n`
  - s05 unit 02 expected scores (~0.95 vs ~0.50) — actual top-hit IS 0.950; finding is stale
  - s05 unit 01 docstring "不跨章节 import" wording — accurate; unit 02 only shares DB_DIR path, not Python import
  - s04 unit 02 self-containment (inlines pypdf/python-docx) — intentional per dispatch-independence docstring
  - s03 hardcoded narrative / forward-looking (d)(e)(f) — sections are (a)(b)(c), not (d)(e)(f); no scope-creep

---

## File Structure

Three independent edits, one per task. No new files. Each task touches exactly one `code.py`:

```
s04_embedding/units/02_provider_routing/code.py   # Task 1 (2 edits)
s07_rerank/units/01_cross_encoder_rerank/code.py  # Task 2 (1 edit)
s11_multimodal/units/02_ocr/code.py               # Task 3 (1 edit)
.superpowers/sdd/progress.md                      # Tasks 1-3 ledger append
```

---

## Task 1: s04 unit 02 — remove dead `WORKDIR` + unnecessary f-string

**Files:**
- Modify: `s04_embedding/units/02_provider_routing/code.py:25` (dead var)
- Modify: `s04_embedding/units/02_provider_routing/code.py:92` (unnecessary f-string)

**Rationale (from prior review ledger):**
- "dead WORKDIR var in unit 02" — `WORKDIR = Path(__file__).resolve().parents[3]` is defined on line 25 but never referenced in the file.
- "unnecessary f-string prefixes" — line 92 has no interpolation, the `f` prefix is dead.

- [ ] **Step 1: Verify the file state**

```bash
grep -n "WORKDIR" s04_embedding/units/02_provider_routing/code.py
grep -nE '^[^#]*f"' s04_embedding/units/02_provider_routing/code.py
```

Expected: line 25 shows `WORKDIR = ...`; line 92 shows the f-string with no `{}` interpolation.

- [ ] **Step 2: Remove the dead `WORKDIR` line**

Edit `s04_embedding/units/02_provider_routing/code.py`:

```python
load_dotenv(override=True)

DEMOS = [
```

(That is: delete line 25 entirely — `WORKDIR = Path(__file__).resolve().parents[3]` — and the blank line 26.)

- [ ] **Step 3: Strip unnecessary `f` prefix on line 92**

Change:

```python
        print(f"[openai] skipped, set LLM_API_KEY (and LLM_BASE_URL) to enable")
```

To:

```python
        print("[openai] skipped, set LLM_API_KEY (and LLM_BASE_URL) to enable")
```

Leave the other f-strings on lines 89, 95, 98, 101 alone — they have interpolation.

- [ ] **Step 4: Verify by running the unit (EMBED_PROVIDER=local)**

```bash
EMBED_PROVIDER=local python s04_embedding/units/02_provider_routing/code.py
```

Expected: prints `provider: local, dim: 512, count: 3` (or similar), plus the two `[openai] skipped` / `[ollama] skipped` lines. No exception, no NameError on `WORKDIR`.

- [ ] **Step 5: Commit + push**

```bash
git add s04_embedding/units/02_provider_routing/code.py
git commit -m "s04: remove dead WORKDIR + unnecessary f-string in unit 02"
```

Then push using the credential-helper pattern (see Global Constraints). Retry loop: up to 5 attempts with 8s sleep on github 443 block. Verify `git fetch origin master && git rev-parse HEAD = git rev-parse origin/master`.

- [ ] **Step 6: Append ledger entry**

Append to `.superpowers/sdd/progress.md`:

```
Task 1 (sweep): complete (commits <base7>..<head7>)
  Removed dead WORKDIR var (line 25) and unnecessary f-string prefix (line 92) in s04 unit 02.
  Verified unit still runs with EMBED_PROVIDER=local.
```

---

## Task 2: s07 unit 01 — wrap `input()` in try/except EOFError

**Files:**
- Modify: `s07_rerank/units/01_cross_encoder_rerank/code.py:142`

**Rationale (from prior review ledger):**
- "input() lacks EOFError" — `query = input("问: ").strip() or "内存"` on line 142 raises `EOFError` if stdin is closed (e.g. piped input ends, or running in a non-interactive shell with no TTY).

- [ ] **Step 1: Locate the offending line**

```bash
grep -n "input(" s07_rerank/units/01_cross_encoder_rerank/code.py
```

Expected: line 142 matches `query = input("问: ").strip() or "内存"`.

- [ ] **Step 2: Wrap the `input()` call**

Replace line 142:

```python
    query = input("问: ").strip() or "内存"
```

With:

```python
    try:
        query = input("问: ").strip() or "内存"
    except EOFError:
        query = "内存"
```

The default `"内存"` fallback is the same fallback the `or "内存"` clause already provides for an empty input, so behavior on TTY is unchanged. On EOF (e.g. `echo "" | python ...`), `query` now lands on `"内存"` instead of crashing.

- [ ] **Step 3: Verify by simulating EOF and TTY**

```bash
# EOF path — should print rerank results for "内存", not crash
echo "" | python s07_rerank/units/01_cross_encoder_rerank/code.py 2>&1 | tail -5

# TTY path — pipe the literal query
echo "R3630G5" | python s07_rerank/units/01_cross_encoder_rerank/code.py 2>&1 | tail -5
```

Expected: both commands run to completion without `EOFError` traceback. First command prints results for "内存"; second for "R3630G5".

- [ ] **Step 4: Commit + push**

```bash
git add s07_rerank/units/01_cross_encoder_rerank/code.py
git commit -m "s07: handle EOFError on input() in unit 01"
```

Push with credential-helper pattern. Verify IN SYNC.

- [ ] **Step 5: Append ledger entry**

```
Task 2 (sweep): complete (commits <base7>..<head7>)
  Wrapped input() in try/except EOFError; default falls back to "内存" on EOF.
  Verified EOF path and TTY path both run to completion.
```

---

## Task 3: s11 unit 02 — make `except ImportError` reachable

**Files:**
- Modify: `s11_multimodal/units/02_ocr/code.py:19-21`

**Rationale (from prior review ledger):**
- "unit 02 exception-order latent edge case (NameError if pytesseract missing AND tesseract call fires)" — currently `import pytesseract` and `from PIL import Image` are on lines 19-20, BEFORE the `try:` block on line 30. If pytesseract is missing, line 19 raises `ImportError` at function entry, never reaching the `except ImportError` on line 32. The current branch is unreachable. Worse, if `from PIL import Image` succeeds but `import pytesseract` succeeds and *only* `image_to_string` triggers a `TesseractNotFoundError` mid-call, the `except pytesseract.TesseractNotFoundError` references `pytesseract` in the except clause — if `pytesseract` was never bound (because import failed), this `except` line itself raises `NameError` at handler-lookup time.

The fix: move both imports inside the `try` block so `except ImportError` and `except pytesseract.TesseractNotFoundError` are both reachable, and the except clause's name resolution only happens after a successful import.

- [ ] **Step 1: Read the current function**

Confirm lines 17-46 look like:

```python
def ocr_image(image_path: Path, lang: str = "chi_sim+eng") -> str:
    """Pillow 打开图片，pytesseract 调系统 tesseract 二进制做 OCR。"""
    import pytesseract
    from PIL import Image
    return pytesseract.image_to_string(Image.open(image_path), lang=lang)


def main() -> None:
    # 默认无图：演示 tesseract 不可用 / 输入缺失时的优雅跳过路径
    img_path = input("可选: 输入图片路径跑 OCR (回车跳过): ").strip()
    if not img_path:
        print("OCR skipped: 未提供图片路径")
        return
    try:
        text = ocr_image(Path(img_path))
    except ImportError:
        print("OCR skipped: pytesseract 未安装，请 `pip install pytesseract Pillow`")
        return
    except pytesseract.TesseractNotFoundError:
        ...
```

- [ ] **Step 2: Restructure the function**

Replace the `ocr_image` body so imports live inside the try:

```python
def ocr_image(image_path: Path, lang: str = "chi_sim+eng") -> str:
    """Pillow 打开图片，pytesseract 调系统 tesseract 二进制做 OCR。

    pytesseract / Pillow 都在函数体内 import,这样 main() 里的 except 分支
    才能正确捕获 ImportError + pytesseract.TesseractNotFoundError。
    """
    import pytesseract
    from PIL import Image
    return pytesseract.image_to_string(Image.open(image_path), lang=lang)
```

Wait — that's the existing body. The actual fix is to move `ocr_image`'s body INSIDE the try in main, OR restructure ocr_image to do its own try. The cleanest fix is option B: move the OCR call from `ocr_image` into `main`'s try, OR have `ocr_image` itself catch. Let me reconsider.

Cleanest pattern: drop the `ocr_image` helper entirely and inline the try in `main`. Replace lines 17-46 with:

```python
def main() -> None:
    # 默认无图：演示 tesseract 不可用 / 输入缺失时的优雅跳过路径
    img_path = input("可选: 输入图片路径跑 OCR (回车跳过): ").strip()
    if not img_path:
        print("OCR skipped: 未提供图片路径")
        return
    try:
        import pytesseract
        from PIL import Image
        text = pytesseract.image_to_string(Image.open(Path(img_path)), lang="chi_sim+eng")
    except ImportError:
        print("OCR skipped: pytesseract 未安装，请 `pip install pytesseract Pillow`")
        return
    except pytesseract.TesseractNotFoundError:
        print(
            "OCR skipped: 系统未找到 tesseract 二进制。"
            "Windows: 安装 https://github.com/UB-Mannheim/tesseract/wiki 并加 PATH；"
            "macOS: brew install tesseract tesseract-lang；"
            "Linux: sudo apt install tesseract-ocr tesseract-ocr-chi-sim"
        )
        return
    except FileNotFoundError:
        print(f"OCR skipped: 图片不存在: {img_path}")
        return
    print(text)


if __name__ == "__main__":
    main()
```

This removes the `ocr_image` helper (it's only called once, in main's try), puts both imports INSIDE the try so `except ImportError` and `except pytesseract.TesseractNotFoundError` are both reachable, and the `except pytesseract.TesseractNotFoundError` clause's `pytesseract` name lookup only happens after a successful `import pytesseract`.

- [ ] **Step 3: Verify**

```bash
# Empty input — should print skip message, no traceback
echo "" | python s11_multimodal/units/02_ocr/code.py
# Expected: "OCR skipped: 未提供图片路径"

# Nonexistent path — should hit FileNotFoundError branch
echo "/no/such/file.png" | python s11_multimodal/units/02_ocr/code.py
# Expected: "OCR skipped: 图片不存在: /no/such/file.png"
```

- [ ] **Step 4: Commit + push**

```bash
git add s11_multimodal/units/02_ocr/code.py
git commit -m "s11: move pytesseract/PIL imports into try block in unit 02"
```

Push with credential-helper pattern. Verify IN SYNC.

- [ ] **Step 5: Append ledger entry**

```
Task 3 (sweep): complete (commits <base7>..<head7>)
  Removed unreachable except ImportError + NameError risk in s11 unit 02 by
  inlining imports inside the try block; dropped redundant ocr_image helper.
```

---

## Final: Whole-Sweep Review

After all 3 tasks:

- [ ] **Step 1: Run all 3 affected units in sequence**

```bash
EMBED_PROVIDER=local python s04_embedding/units/02_provider_routing/code.py
echo "" | python s07_rerank/units/01_cross_encoder_rerank/code.py
echo "" | python s11_multimodal/units/02_ocr/code.py
```

Expected: all three exit 0 with no traceback.

- [ ] **Step 2: Append final ledger entry**

```
Sweep: complete (commits <first-base7>..<last-head7>)
  All 3 actionable minor findings from prior restructure addressed.
  Ledger reconciled against current master.
```

- [ ] **Step 3: Hand back to user for branch closeout (Option 3: keep as-is — direct master push was approved)**

---

## Self-Review

**1. Spec coverage:** Three findings (s04 WORKDIR/f-string, s07 EOFError, s11 import order) each have a dedicated task. Coverage: complete.

**2. Placeholder scan:** No TBDs. Every Step shows the exact code to use. Commit commands are spelled out. Verification commands are spelled out. No "see above" references.

**3. Type consistency:** The plan doesn't introduce new functions or change signatures beyond removing the `ocr_image` helper. No cross-task name drift.

**4. Findings explicitly excluded** (and why): listed in Global Constraints. Reviewers should not re-flag excluded items.

**5. Real false-positive risk:** The s11 fix restructures the function body. Reviewer should confirm behavior is preserved (the `if not img_path` early-return path is unchanged, both error message paths preserved verbatim).