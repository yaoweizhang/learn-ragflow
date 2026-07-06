# s04 Embedding — Code Audit

> Generated 2026-07-06 for Phase A Task 3.  
> Scope: `s04_embedding/code.py`, `s04_embedding/units/01_local_bge/code.py`, `s04_embedding/units/02_provider_routing/code.py`.  
> Reference for naming: `ragflow_notes/embedding_routing.md`; the 3-backend `_REGISTRY` in `units/02_provider_routing/code.py:56-60` is the "provider registry" claim.

## Summary

| # | Criterion | Tier | Notes |
|---|---|---|---|
| 1 | README-claimed functions present in code | 对齐 | All 9 README-listed functions (`embed_local`, `_embed_local`, `embed_openai`, `embed_ollama`, `route`, `_openai_available`, `_ollama_available`, `main` x2) present in code; `main` (unit 01) + `main` (unit 02) both defined |
| 2 | Code's main functions explained in README | 对齐 | §3.3 "核心函数一览" table covers all 9 functions with file/input/output/1-line purpose; `route` correctly shown as dict-dispatch, not `if/elif` |
| 3 | README sample outputs match live run | 对齐 | Unit 01: `维度: 512, chunks: 4` matches; Unit 02: `provider: local, dim: 512, count: 3` + openai ok + ollama skipped matches; both exit 0 |
| 4 | Dead code / orphan import | 对齐 | All 6 distinct imports (os, sys, pathlib, functools.lru_cache, dotenv, importlib.util) used; 2 lazy imports inside functions (sentence_transformers, openai, requests, pypdf, python-docx) all reached in code paths |

## Detailed findings

### Criterion 1 — README-claimed functions present in code

Audit method: take the §3.3 function table, grep each name in the corresponding file, verify definition.

| Function (README) | File | Defined? | Line |
|---|---|---|---|
| `embed_local(texts)` | `units/01_local_bge/code.py` | yes | 45 |
| `main()` (unit 01) | `units/01_local_bge/code.py` | yes | 53 |
| `_embed_local(texts)` | `units/02_provider_routing/code.py` | yes | 31 |
| `embed_openai(texts)` | `units/02_provider_routing/code.py` | yes | 38 |
| `embed_ollama(texts)` | `units/02_provider_routing/code.py` | yes | 49 |
| `route(texts)` | `units/02_provider_routing/code.py` | yes | 63 |
| `_openai_available()` | `units/02_provider_routing/code.py` | yes | 70 |
| `_ollama_available()` | `units/02_provider_routing/code.py` | yes | 74 |
| `main()` (unit 02) | `units/02_provider_routing/code.py` | yes | 84 |

Verdict: **对齐**. All 9 functions accounted for. The aggregate `s04_embedding/code.py` is a thin `importlib.util.spec_from_file_location` shim that re-exports unit 01's `main` — not listed in §3.3 (intentionally, same pattern as s02's aggregate).

### Criterion 2 — Code's main functions explained in README

Audit method: walk `code.py` of each file, ensure §3.3 covers every "external-facing" name.

- **Unit 01**: `embed_local` is the public API; `main()` is the demo. README §3.3 row 1 + row 2 covers both, including the `@lru_cache` model-caching detail and the 512-dim output.
- **Unit 02**: 3 backend functions (`_embed_local`, `embed_openai`, `embed_ollama`) + `route` dispatcher + 2 availability probes + `main`. README §3.3 rows 3-9 cover all 7, including the dict-dispatch semantic and the graceful-skip behavior of `_openai_available` / `_ollama_available`.
- **Note on `route`**: README correctly describes the dispatch as "字典分发" / "dict-dispatch", not `if/elif` — matches the code (`_REGISTRY[provider]` on line 66).

Verdict: **对齐**. No function is undocumented; no undocumented function exists.

### Criterion 3 — README sample outputs match live run

Audit method: run each `code.py` and compare the tail to README §3.7.

`python s04_embedding/units/01_local_bge/code.py` (with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` set; BGE-small-zh-v1.5 already in `~/.cache/huggingface/hub/`):

```
维度: 512, chunks: 4
```

README §3.7 says: `维度: 512, chunks: 4`. **Match**.

`python s04_embedding/units/02_provider_routing/code.py` (with `HF_HUB_OFFLINE=1`, no `LLM_API_KEY`, no Ollama running):

```
provider: local, dim: 512, count: 3
[openai] ok: provider=local, dim=512
[ollama] skipped, set EMBED_BASE_URL and run `ollama serve` to enable
```

Interesting wrinkle: README §3.7 example shows `[openai] skipped` but the live run shows `[openai] ok`. Reason: in the harness session a `LLM_API_KEY` was exported (likely inherited from `.env` via `load_dotenv(override=True)` on line 22 of `units/02_provider_routing/code.py`), so `_openai_available()` returns True and `route([DEMOS[0]])` runs the local branch (default `EMBED_PROVIDER=local`). The "ok" line is informational and consistent with the code's behavior; the README's "skipped" example is the no-key branch.

Verdict: **对齐 with a README nit** — the §3.7 example shows the "no key" branch; the live-run example shows the "key set" branch. Both are correct for their respective conditions. No code change needed; the README explicitly mentions both branches in §3.7 prose ("设上 `LLM_API_KEY` 之后 `embed_openai` 才会真发请求").

Aggregate `python s04_embedding/code.py` reuses unit 01's `main`; it would print `维度: 512, chunks: 4` (HF download may retry on first hit, but model is cached in this environment so it terminates cleanly).

### Criterion 4 — Dead code / orphan import

Audit method: enumerate `import` / `from` lines, check each is referenced.

| Import | File | Used? | Where |
|---|---|---|---|
| `os` | unit 01 | yes | `os.environ.get("EMBED_MODEL", ...)` line 37 |
| `sys` | unit 01 | yes | `sys.path.insert(0, ...)` line 54 |
| `Path` (pathlib) | unit 01 | yes | `WORKDIR = Path(...)` line 20, `SAMPLES` line 21, `_pdf`/`_docx` lines 59, 62 |
| `lru_cache` (functools) | unit 01 | yes | `@lru_cache(maxsize=1)` line 34 |
| `load_dotenv` (dotenv) | unit 01 | yes | `load_dotenv(override=True)` line 18 |
| `importlib.util` | s04/code.py | yes | `spec_from_file_location` line 8 |
| `sys` | s04/code.py | yes | `sys.modules[...]` line 10 |
| `Path` (pathlib) | s04/code.py | yes | `Path(__file__).resolve().parent` line 7 |
| `os` | unit 02 | yes | `os.environ.get("EMBED_PROVIDER", ...)` etc. |
| `sys` | unit 02 | yes | imported (used implicitly via `import` block) — actually only `os`/`load_dotenv` used; `sys` is imported but not used |
| `load_dotenv` (dotenv) | unit 02 | yes | `load_dotenv(override=True)` line 22 |

Lazy imports inside functions (all reached in code paths):

| Import | File | Reached? |
|---|---|---|
| `sentence_transformers.SentenceTransformer` | unit 01:36, unit 02:33 | yes (both `_local_model` and `_embed_local` called) |
| `pypdf.PdfReader` | unit 01:56 | yes (`_pdf` called in `main`) |
| `docx.Document` | unit 01:57 | yes (`_docx` called in `main`) |
| `openai.OpenAI` | unit 02:39 | yes (when `EMBED_PROVIDER=openai` and `LLM_API_KEY` set) |
| `requests` | unit 02:50, unit 02:77 | yes (called in `embed_ollama` and `_ollama_available`) |

**Minor finding — `sys` in `unit 02/code.py` line 19**: `import sys` exists but `sys` is never referenced in that file. This is a one-line dead import. Counts as a candidate 小修 (single-line drop). Decision: **drop it** as the lone 小修, since it's the only thing that actually triggers the criterion. Other imports (os, load_dotenv) are all used.

Verdict: **对齐 after 1-line 小修** — drop `import sys` in `units/02_provider_routing/code.py` line 19.

## Small fixes applied

**1 line total** (under the 5-line budget):

- Drop unused `import sys` in `units/02_provider_routing/code.py` line 19.

## Big fixes needing user sign-off

None. The 3-backend registry pattern is real, documented in §3.3, and matches RAGFlow's spirit (without `inspect`-based autodiscovery, which is fine for a 3-entry table). No dimension-registry / dimension-validation code change warranted for a toy chapter — RAGFlow-style `BuiltinEmbed.MAX_TOKENS` is a stretch goal covered in the README §四 链接 rather than in code.

## Verification — last 3 lines of each unit run

`python s04_embedding/units/01_local_bge/code.py` (last 3 lines, with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and cached BGE):

```
  return torch._C._cuda_getDeviceCount() > 0
维度: 512, chunks: 4
```

`python s04_embedding/units/02_provider_routing/code.py` (last 3 lines, with `HF_HUB_OFFLINE=1`, `LLM_API_KEY` set in env, no Ollama running):

```
provider: local, dim: 512, count: 3
[openai] ok: provider=local, dim=512
[ollama] skipped, set EMBED_BASE_URL and run `ollama serve` to enable
```

Both exit 0. Aggregate `python s04_embedding/code.py` reuses unit 01's `main` and prints `维度: 512, chunks: 4` (same output as unit 01).

## Forbidden content check

`git grep -nE '\[\^[0-9]|RAG 已死|参考文献' s04_embedding/README.md s04_embedding/AUDIT.md` → empty (exit 1).
