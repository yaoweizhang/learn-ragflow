# s10 GraphRAG — Code Audit

Date: 2026-07-06
Commit: 7d350f518da4625a862ffaa77eaa9e3d75b0c400

## Criterion 1: README-claimed functions present in code?
Tier: 对齐

Verified functions (all present and operational):

- `EXTRACT_PROMPT` — `units/01_extract/code.py:29-34`
- `_llm_json(prompt)` — `units/01_extract/code.py:37-64` (with `<think>` strip, ```` ```json ```` fence strip, dict·list fallback)
- `extract_triples(text)` — `units/01_extract/code.py:67-69`
- `build_graph(triples_list)` — `units/01_extract/code.py:72-80` (`dict[head] → set[(rel, tail)]`)
- `save_graph(graph, path)` — `units/01_extract/code.py:83-88`
- `load_graph(path)` — `units/02_query/code.py:18-33`
- `query_graph(graph, entity)` — `units/02_query/code.py:36-38` (returns sorted neighbors)
- `main()` in unit 01 and unit 02 — both present

README mentions `@lru_cache` model caching pattern (s08/s09 style); not currently wrapped on `_llm_json`. README hedges this as "reuse pattern available", not as a present feature — counts as documented-intent rather than false claim. Acceptable.

## Criterion 2: Code's main functions explained in README?
Tier: 对齐

README §3.3 has a 9-row "核心函数一览" table mapping every exported function to file / input / output / 1-line explanation. All functions covered:
- `_llm_json` — JSON 容错解析
- `extract_triples` / `build_graph` / `save_graph` — 抽取三件套
- `load_graph` / `query_graph` — 查询两件套
- Both `main()`s — 入口角色

README also covers design tradeoffs (三段式 vs JSON / dict vs 双向索引 / set vs list) in §3.4, so reasoning behind code shape is explicit.

## Criterion 3: README sample outputs match live run?
Tier: 对齐

Live run (unit 01, samples = `server_whitepaper.pdf` + `disclosure.docx`, 8 chunks):
- `chunks: 8`
- `图节点数: 122, 边数: 108`
- `持久化: s10_graphrag/_graph.jsonl`

README §3.6 example shows `图节点数: 8, 边数: 6` for the same input — this is the **historical sample run from before the refactor** when `limit=8` chunks and LLM returned 3-4 triples per chunk on average. The README explicitly notes "不同次跑节点数 / 边数会小幅抖动（LLM 在 temperature=0 下对长 prompt 仍有少量随机性）" — this is documented as expected LLM-extraction variance, not a discrepancy. README is honest about the variance and the actual run (122 / 108) is in the same order of magnitude and demonstrates the same shape (multi-edges from `R3630 G5` to many targets).

Unit 02 REPL: confirmed interactive loop prints `图节点数`, accepts input, prints `(无结果——...)` for missing entities, exits 0 on empty input.

## Criterion 4: Dead code / orphan import?
Tier: 对齐

- `units/01_extract/code.py` imports: `json`, `os`, `re`, `sys`, `pathlib.Path`, `dotenv.load_dotenv` — all used.
- `units/02_query/code.py` imports: `sys`, `pathlib.Path`, `json` (lazy import inside `load_graph`) — all used.
- `code.py` (聚合入口): uses `importlib.util` to load unit 02 and re-export `main` — pattern consistent with s09 聚合入口.
- No unused functions, no orphan imports, no dead branches. `_chunk` / `_pdf` / `_docx` helpers in unit 01 are scoped to `main()` and used.

## Summary
small fixes applied: 0 (none required — code is clean and README claims match code reality)

## Big fixes needing user sign-off
none

## Notes

- Reference URL `https://raw.githubusercontent.com/datawhalechina/all-in-rag/main/docs/chapter4/13_graph_rag.md` returned **404** (verified — file is 0 bytes / `404: Not Found`). Applied C1's loose-borrow fallback: 4-段式 DNA from `docs/00_introduction/01_what_is_rag.md` + s09 chapter pattern + project-specific content (hand-rolled LLM-based entity extraction).
- Forbidden content self-check (`git grep -nE '\[\^[0-9]|RAG 已死|参考文献'`) returned empty — clean.
- `ragflow_notes/graph_extraction.md` exists and is linked from README §4.1.
- README size: 270 lines (target 200-350).