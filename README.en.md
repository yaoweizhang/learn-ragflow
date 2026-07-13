# Learn RAGFlow — A 12-Chapter RAG Tutorial, Zero to Deployment

> **Canonical text is in [`README.md`](./README.md) (Chinese).** This file is the EN mirror — same depth, same coverage, translated for non-Chinese readers.

## What this is

A hands-on, engineering-focused RAG (Retrieval-Augmented Generation) tutorial aimed at LLM application developers. The goal is not to "read theory" but to **write 12 minimal MVP toys of 30–80 lines each**, then explain what each segment of an RAG system does, why it does it, and what production looks like.

Two layers (双层):

- **Main chapters**: each chapter ships one `README.md` plus N descriptive `code_NN_<topic>.py` files at the chapter root (30–80 lines each); modify one line, see the output change, and validate "does a higher alpha give better recall?" in 5 minutes.
- **Source reference**: [`docs/reference/ragflow-notes/`](./docs/reference/ragflow-notes/) collects [RAGFlow](https://github.com/infiniflow/ragflow)'s 5–15 line production excerpts with line numbers and "why this design" commentary, organized by chapter at the repository top — read when you want to go deep, not duplicated in each chapter README.
- **Industrial reference (default reading path)**: [RAGFlow](https://github.com/infiniflow/ragflow) is one of GitHub's highest-star open-source RAG engines, with each RAG-pipeline stage engineered as an independent, swappable module. This tutorial uses RAGFlow as its **sole** industrial reference — every chapter ends with a `## RAGFlow 实现` section excerpting RAGFlow's implementation at that layer, with detailed source analysis in [`docs/reference/ragflow-notes/`](docs/reference/ragflow-notes/). **ragflow-notes is not supplementary material — it's the second half of the learning path**.

Every chapter contains:

- A **self-written MVP** (30–80 lines of Python, single file, minimal dependencies)
- **Reproducible experiments** that run on two shared sample files: `samples/server_whitepaper.pdf` and `samples/disclosure.docx`

**Target reader:** developers comfortable calling OpenAI-compatible LLM APIs in Python, who want to understand RAG from an engineering angle rather than from theory. No prior RAGFlow exposure required. Each chapter takes 5–10 minutes to read and run.

## Quick start

```bash
git clone <repo-url>
cd learn-ragflow
pip install -r requirements.txt
cp .env.example .env       # then edit .env and set LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
# Chapter 0 (preface) is prose-only — read s00_concepts/README.md to build the RAG mental model
python s01_what_is_rag/c01_naive_keyword.py
```

Requires Python 3.10+ and at least 8 GB RAM (16 GB recommended for BGE embeddings). GPU optional.

The LLM endpoint defaults to OpenAI-compatible protocol (`LLM_BASE_URL` + `LLM_MODEL`); point it at any OpenAI-compatible service (OpenAI, DeepSeek, Zhipu, MiniMax, vLLM self-hosted, etc.). Set `LLM_MODEL` to the actual chat model name supported by your service.

## Why this project exists

RAG has moved from "research prototype" to "production default." But learners usually hit three walls:

1. **Theory is too abstract.** Papers give formal definitions of "chunking strategy" or "hybrid retrieval" without 50 lines of runnable code.
2. **Engineering is too fragmented.** To build RAG you must glue together: document parsing, vector DB, BM25, reranker, prompt engineering, agent orchestration — each layer requires a selection, each selection has 5 candidates.
3. **Industry practice is invisible.** Frameworks like LangChain / LlamaIndex run, but you can't see **why** they're designed that way; extending them means starting from scratch.

This tutorial tears down the first two walls and builds a ladder to the third:

- 12 chapters × 12 MVPs give you the minimum skeleton for **each** layer — every chapter runs, every chapter invites modification, every chapter shows output change.
- [`docs/reference/ragflow-notes/`](./docs/reference/ragflow-notes/) gives you **production code's real design** — [RAGFlow](https://github.com/infiniflow/ragflow) is one of GitHub's highest-star open-source RAG engines, has engineered every layer above, and reads cleanly.

After working through this, you'll have **x-ray vision into LangChain / LlamaIndex / Dify** source — knowing what they abstract, why, when to trust them, and when to bypass and write your own.

**Why focus on RAGFlow** — RAGFlow solves the core tension of industrial-grade RAG systems: every layer needs selection + every layer needs engineering. It abstracts "connect an embedding provider" into a factory pattern + provider registry; splits "PDF parsing" into per-file-type parsers + a dispatcher; turns "agent routing" into a visualizable DAG. These engineering insights are what every chapter's `## RAGFlow 实现` section is about — after running a 30-line toy, see what RAGFlow's 1000-line production code does to handle "invisible engineering".

## Audience

**This tutorial fits:**

- Developers fluent in Python + LLM APIs, wanting to understand the full RAG pipeline from an engineering angle.
- Engineers who have built LangChain / LlamaIndex demos and want to peek inside the implementation.
- AI application engineers preparing for RAG selection, in-house development, or second-tier customization.
- Algorithm engineers interested in retrieval, recommendation, or knowledge-graph domains.

**Prerequisites:**

- Python basics + `pip install` + ability to run `python script.py`.
- Can read Python ≤ 50 lines, can debug with `print()`.
- Has an OpenAI-compatible LLM API key.

**Not required:**

- Prior RAGFlow source reading — focus is what this chapter builds, not the reference.
- Deep learning / Transformer background — mentioned briefly when relevant.
- GPU — BGE embedding finishes in 1–2 minutes on CPU.

## Highlights

1. **Minimum runnable**: every chapter is "first write a 30-line toy" — running is the first goal; for production look at [`docs/reference/ragflow-notes/`](./docs/reference/ragflow-notes/) yourself.
2. **Shared sample files**: the repo ships two fictional samples (`samples/server_whitepaper.pdf`, `samples/disclosure.docx`), reused across chapters for easy output comparison.
3. **Env vars + single dep list**: 12 chapters share one `requirements.txt` and one `.env.example` — no setup pain.
4. **Questions separated from answers**: each chapter's "思考题" sits at the bottom of its `README.md`; try first, peek later.

## Learning paths

The tutorial has two tracks — pick by your time budget:

**Fast path (2–3 hours):** s01 → s06 → s08 → s12.

- Get the minimum RAG running (s01–s06), see the chat side answering questions from the documents (s08), then glance at deployment (s12).
- Focus is **end-to-end pipeline fluency**; suitable for building a mental model first.

**Full path (10–12 hours):** s01 → s02 → ... → s12, one chapter at a time.

- Each chapter takes 30–60 minutes: run `code_NN_<topic>.py` + modify `code_NN_<topic>.py` to see the change + read the matching `docs/reference/ragflow-notes/<topic>.md`.
- Focus is **design tradeoffs in each layer**; suitable for engineers planning to build, select, or extend a RAG system.

## Detailed outline

The tutorial is split into **5 parts and 12 chapters**, each chapter shipping one `README.md` plus N descriptive `code_NN_<topic>.py` files at the chapter root.

### Part 0 — Concept primer

**Chapter 0 — What is RAG / Why RAG / How RAG evolved** &nbsp;[chapter details](./s00_concepts/)

- Concept map: RAG = retrieve + augment + generate (parametric vs. non-parametric knowledge)
- Selection comparison: vs. long-context / vs. fine-tune (token cost + knowledge freshness + controllability)
- Evolution arc: Naive RAG → Advanced RAG → Modular RAG, and which of the 12 chapters map to each stage

Read Part 0 first; then Chapter 1's "substr → vectors → retrieve + LLM" line will make sense.

### Part 1 — RAG fundamentals

**Chapter 1 — RAG fundamentals** &nbsp;[chapter details](./s01_what_is_rag/)

- 3 units progressing from naive substring matching → bag-of-words vectors → retrieve + LLM end-to-end
- Three failure modes of LLMs: training cutoff, private data, hallucination
- RAG workflow: retrieve → augment → generate

### Part 2 — Data and indexing

**Chapter 2 — Document loading** &nbsp;[chapter details](./s02_doc_loading/)

- PDF / DOCX parsing into a unified `list[{text, page, source}]`
- Three real-world problems: scanned PDFs, tables, headers / footers
- Production reference: [`docs/reference/ragflow-notes/deepdoc_pdf_parsing.md`](./docs/reference/ragflow-notes/deepdoc_pdf_parsing.md) (`deepdoc/parser/` + VisionParser)
- → [RAGFlow implementation](./docs/reference/ragflow-notes/deepdoc_pdf_parsing.md)

**Chapter 3 — Text chunking** &nbsp;[chapter details](./s03_chunking/)

- Fixed character cap + sentence-boundary split (spec: ≤ 500 chars; sentence boundaries `(.。!?！？)`)
- Three failure modes: tables, parent-child chunks, cross-paragraph references
- Production reference: [`docs/reference/ragflow-notes/deepdoc_chunking.md`](./docs/reference/ragflow-notes/deepdoc_chunking.md) (`_concat_downward` / `naive_merge` / `hierarchical_merge`)
- → [RAGFlow implementation](./docs/reference/ragflow-notes/deepdoc_chunking.md)

**Chapter 4 — Embedding** &nbsp;[chapter details](./s04_embedding/)

- BGE local embedding (BAAI/bge-small-zh-v1.5, 512 dim, normalized)
- OpenAI / Ollama providers optional
- Production reference: [`docs/reference/ragflow-notes/embedding_routing.md`](./docs/reference/ragflow-notes/embedding_routing.md)
- → [RAGFlow implementation](./docs/reference/ragflow-notes/embedding_routing.md)

**Chapter 5 — Vector indexing** &nbsp;[chapter details](./s05_vector_index/)

- Chroma persistent (`PersistentClient` + HNSW cosine)
- Metadata filtering: `where={"source": "..."}`
- Production reference: [`docs/reference/ragflow-notes/vector_indexing.md`](./docs/reference/ragflow-notes/vector_indexing.md) (ES / Infinity / OceanBase)
- → [RAGFlow implementation](./docs/reference/ragflow-notes/vector_indexing.md)

### Part 3 — Retrieval and generation

**Chapter 6 — Hybrid retrieval** &nbsp;[chapter details](./s06_retrieval/)

- Self-implemented BM25 + dense vectors + `alpha * vec + (1-α) * bm25` weighted fusion
- `alpha` is a configurable knob (fact-style questions skew BM25; concept-style skew vector)
- Production reference: [`docs/reference/ragflow-notes/hybrid_retrieval.md`](./docs/reference/ragflow-notes/hybrid_retrieval.md) (`FusionExpr` + `rerank_with_knn` + `rank_fea`)
- → [RAGFlow implementation](./docs/reference/ragflow-notes/hybrid_retrieval.md)

**Chapter 7 — Reranking** &nbsp;[chapter details](./s07_rerank/)

- BGE cross-encoder (`bge-reranker-base`) for precision rerank
- `top_k` controls cross-encoder pair count (O(n) not O(n²))
- Production reference: [`docs/reference/ragflow-notes/rerank.md`](./docs/reference/ragflow-notes/rerank.md) (`RerankModel.Base` multi-provider abstraction)
- → [RAGFlow implementation](./docs/reference/ragflow-notes/rerank.md)

**Chapter 8 — Prompt & generation** &nbsp;[chapter details](./s08_prompt_generate/)

- Prompt templates for citation `[i]`, refusal, footnote alignment
- `_format_context` renders hits as `[i] (source#page) text`
- Production reference: [`docs/reference/ragflow-notes/prompt_templates.md`](./docs/reference/ragflow-notes/prompt_templates.md) (dual-pass + multi-template)
- → [RAGFlow implementation](./docs/reference/ragflow-notes/prompt_templates.md)

**Chapter 9 — Agent & tools** &nbsp;[chapter details](./s09_agent_tools/)

- ReAct loop: `Thought` / `Action` / `ActionInput` parsing
- Two tools: `retrieve(query)` + `finish(answer)`
- Parsing fragility: `max_steps` + markdown-fence stripping + JSON retry
- Production reference: [`docs/reference/ragflow-notes/agent_tools.md`](./docs/reference/ragflow-notes/agent_tools.md) (Canvas DAG + `bind_tools()`)
- → [RAGFlow implementation](./docs/reference/ragflow-notes/agent_tools.md)

### Part 4 — Advanced RAG

**Chapter 10 — GraphRAG** &nbsp;[chapter details](./s10_graphrag/)

- LLM-extracted `(head, rel, tail)` triples
- `dict[head] → set[(rel, tail)]` 1-hop query
- Production reference: [`docs/reference/ragflow-notes/graph_extraction.md`](./docs/reference/ragflow-notes/graph_extraction.md) (light path + `entity_resolution` two-stage pipeline)
- → [RAGFlow implementation](./docs/reference/ragflow-notes/graph_extraction.md)

**Chapter 11 — Multimodal** &nbsp;[chapter details](./s11_multimodal/)

- pdfplumber for table extraction (row × column structure)
- pytesseract OCR (`chi_sim+eng`)
- Production reference: [`docs/reference/ragflow-notes/multimodal_parsing.md`](./docs/reference/ragflow-notes/multimodal_parsing.md) (vision model + multi-OCR backend)
- → [RAGFlow implementation](./docs/reference/ragflow-notes/multimodal_parsing.md)

### Part 5 — Deployment and shipping

**Chapter 12 — Deployment** &nbsp;[chapter details](./s12_deployment/)

- FastAPI wrapper (`/qa` endpoint) + pydantic input validation
- docker-compose (api + chroma persistent volume)
- 503 fallback: clear error when index is missing, not a raw exception
- Production reference: [`docs/reference/ragflow-notes/deployment.md`](./docs/reference/ragflow-notes/deployment.md)
- → [RAGFlow implementation](./docs/reference/ragflow-notes/deployment.md)

### Further reading (project-level reference, not part of the tutorial runtime — referenced from s01-s12)

- [RAGFlow source-reading index](./docs/reference/ragflow-notes/README.md) — production-code excerpts (read when you want to go deep)
- [`docs/` directory guide](./docs/) — navigation for `docs/reference/ragflow-notes/` and other reference material

## Chapter index

1. [s01](./s01_what_is_rag/) — **RAG fundamentals.** Substring → bag-of-words → retrieve + LLM end-to-end.
2. [s02](./s02_doc_loading/) — **Document loading.** PDF / DOCX parsing; metadata preservation.
3. [s03](./s03_chunking/) — **Text chunking.** Fixed-size vs. structure-aware splitting.
4. [s04](./s04_embedding/) — **Embedding.** BGE local; OpenAI / Ollama switches.
5. [s05](./s05_vector_index/) — **Vector index.** Chroma persistent; metadata filtering.
6. [s06](./s06_retrieval/) — **Hybrid retrieval.** BM25 + dense vectors; weighted fusion.
7. [s07](./s07_rerank/) — **Reranking.** BGE cross-encoder precision.
8. [s08](./s08_prompt_generate/) — **Prompt & generation.** Citation tracing; hallucination control.
9. [s09](./s09_agent_tools/) — **Agent & tools.** ReAct loop; retrieve / finish tools.
10. [s10](./s10_graphrag/) — **GraphRAG.** LLM-based entity-relation extraction; 1-hop query.
11. [s11](./s11_multimodal/) — **Multimodal.** pdfplumber table extraction; pytesseract OCR.
12. [s12](./s12_deployment/) — **Deployment.** FastAPI + docker compose.

All 12 chapters are in place.

## Project structure

```
learn-ragflow/
├── README.md                    # Chinese canonical readme
├── README.en.md                 # This file (English mirror)
├── LICENSE                      # MIT
├── .env.example                 # Env template (LLM / Embedding / Reranker)
├── requirements.txt             # Aggregated deps across all chapters
├── samples/
│   ├── README.md
│   ├── server_whitepaper.pdf
│   └── disclosure.docx
├── docs/                        # Design docs (not part of the tutorial runtime)
│   └── reference/
│       └── ragflow-notes/       # RAGFlow source excerpts (one per chapter)
├── s00_concepts/                # Chapter 0 (preface; 5-H2 prose, no code files)
│   └── README.md
├── s01_what_is_rag/             # Chapter 1
│   ├── README.md
│   ├── c01_naive_keyword.py
│   ├── c02_vector_basics.py
│   └── c03_augmented_llm.py
├── s02_doc_loading/             # Chapter 2
├── ...
└── s12_deployment/              # Chapter 12
```

### Per-chapter layout

Every chapter (`sXX_topic/`) follows the same shape — one `README.md` plus N descriptive `code_NN_<topic>.py` files at the chapter root:

```
sXX_topic/
├── README.md                # Chapter entry: navigation + one H2 per code file
├── code_01_<topic>.py       # always present, 30–80 lines
├── code_02_<topic>.py       # as needed (most chapters have 2)
└── code_03_<topic>.py       # as needed; s01 / s03 have 3
```

Each chapter's `README.md` uses a uniform H2 skeleton that adapts to the code file count:

- **一、章节介绍** — what this chapter solves, why it's designed this way, where it sits in the RAG pipeline.
- **二、<code_01 topic>：`code_NN_xxx.py`** — full coverage of one code file (H3 children: 这是什么 / 跑起来 / 它做对了什么 / 它做错了什么 / 思考题). Each code file gets its own H2 so the "what it is" and "how to use it" stay bundled — after reading one section, you know what the code does, how to run it, and where it breaks.
- **三、<code_02 topic>：`code_NN_xxx.py`** (as needed) — same shape.
- **四、<code_03 topic>：`code_NN_xxx.py`** (as needed) — same shape.
- **+1、其他 / 整体设计取舍** — cross-file design decisions, schema shape, troubleshooting.
- **+1、RAGFlow 实现** — excerpted RAGFlow production implementation for this layer (5–15 line key code + line numbers + design intent), contrasted with the toy.
- **+1、其他 / 选型与思考题** — mainstream approach quick-look + selection memo + this chapter's thinking exercises.
- **思考题答案** (last H2) — all Q&A consolidated.

- 1-code-file chapters (s07 / s08 / s12): 一 + 二 (code_01) + 三 (RAGFlow 实现) + 四 (选型) + 五 (思考题答案) = 6 H2 sections.
- 2-code-file chapters: 一 + 二 + 三 + 四 (RAGFlow 实现) + 五 (选型) + 六 (思考题答案) = 7 H2 sections.
- 3-code-file chapters (s01 / s03): 一 + 二 + 三 + 四 + 五 (RAGFlow 实现) + 六 (选型) + 七 (思考题答案) = 7 H2 sections.
- Chapter 0 (preface, no code files): 一/二/三/四/五 + 思考题答案 = 5 H2 sections.

Each code file is independently runnable:

```bash
python sXX_topic/code_01_<topic>.py
```

The 思考题 (thinking exercises) and answers are appended to the bottom of each chapter's `README.md`; no separate `thinking_answers.md`.

## Where to go next

After working through all 12 chapters, the natural next steps are:

- **Swap embedding**: change `EMBED_PROVIDER` in `.env` from `local` to `openai` or `ollama` and observe retrieval quality changes.
- **Swap chunker**: s03's `chunk_by_paragraph` is paragraph-based; try sentence + sliding-window or Markdown-heading-based splitting.
- **Tune retrieval weights**: s06's `alpha` controls vector vs. BM25 weighting; build a 5–10 question eval set to pick the best value.
- **Go to production**: s12 ships FastAPI + docker compose. Add Prometheus monitoring, Sentry error tracking, and an independent model server (vLLM / TGI) for scale.
- **Read RAGFlow source**: see [`docs/reference/ragflow-notes/`](./docs/reference/ragflow-notes/) for chapter-aligned source excerpts; focus on **design intent**, not line-by-line memorization.

## Acknowledgements

- [**RAGFlow**](https://github.com/infiniflow/ragflow) — primary industrial reference; see [`docs/reference/ragflow-notes/`](./docs/reference/ragflow-notes/) for chapter-aligned source excerpts. RAGFlow evolves, so excerpts may go stale — focus on **design intent**, not line-by-line memorization.
- [**all-in-rag**](https://github.com/datawhalechina/all-in-rag) — sibling tutorial. The "project intro / motivation / audience / highlights / detailed outline" layout of this README borrows from its "content outline" organization.
- [**learn-claude-code**](https://github.com/shareAI-lab/learn-claude-code) — inspiration for the "minimal runnable MVP per chapter" format.
- [**BGE models**](https://github.com/FlagOpen/FlagEmbedding) (BAAI) — default embedding and reranker.

## License

MIT — see [LICENSE](./LICENSE).
