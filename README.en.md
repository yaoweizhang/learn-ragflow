# Learn RAGFlow — A 12-Chapter RAG Tutorial, Zero to Deployment

> **This is a Chinese-first tutorial.** Each chapter ships a 30–80 line self-written MVP in Chinese, paired with a guided reading of [RAGFlow](https://github.com/infiniflow/ragflow)'s industrial-grade source. The canonical text is in `README.md`; this file is an English summary so non-Chinese readers can follow the progression.

This repository is a hands-on, engineering-focused RAG (Retrieval-Augmented Generation) tutorial. Every chapter contains:

- A **self-written MVP** (30–80 lines of Python, single file, minimal dependencies)
- **Reproducible experiments** that run on two shared sample files: `samples/server_whitepaper.pdf` and `samples/disclosure.docx`

**Target reader:** comfortable calling LLM APIs in Python; wants to understand the full RAG pipeline from an engineering angle rather than from pure theory. No prior RAGFlow exposure required.

## Quick start

```bash
git clone <repo-url>
cd learn-ragflow
pip install -r requirements.txt
cp .env.example .env       # then edit .env and set LLM_API_KEY
python s01_what_is_rag/code.py
```

Requires Python 3.10+ and at least 8 GB RAM (16 GB recommended for BGE embeddings). GPU optional.

The LLM endpoint defaults to OpenAI-compatible protocol (`LLM_BASE_URL` + `LLM_MODEL`); point it at any OpenAI-compatible service (OpenAI, DeepSeek, Zhipu, MiniMax, etc.).

## Chapter index

1. [s01](./s01_what_is_rag/) — **What is RAG?** Naive RAG vs. long-context LLMs; minimal end-to-end demo.
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
├── README.en.md                 # This file (English summary)
├── LICENSE                      # MIT
├── .env.example                 # Env template (LLM / Embedding / Reranker)
├── requirements.txt             # Aggregated deps across all chapters
├── samples/
│   ├── README.md
│   ├── server_whitepaper.pdf
│   └── disclosure.docx
├── docs/                        # Design docs (not part of the tutorial runtime)
│   ├── 00-intro/                # RAG primer + docs/ usage
│   └── reference/
│       └── ragflow-notes/       # RAGFlow source excerpts (one per chapter)
├── s01_what_is_rag/             # Chapter 1
├── s02_doc_loading/             # Chapter 2
├── ...
├── s12_deployment/              # Chapter 12
└── samples/                     # Shared samples
```

## Where to go next

After working through all 12 chapters, the natural next steps are:

- **Swap embedding**: change `EMBED_PROVIDER` in `.env` from `local` to `openai` or `ollama` and observe retrieval quality changes.
- **Swap chunker**: s03's `chunk_by_paragraph` is paragraph-based; try sentence + sliding-window or Markdown-heading-based splitting.
- **Tune retrieval weights**: s06's `alpha` controls vector vs. BM25 weighting; build a 5-10 question eval set to pick the best value.
- **Go to production**: s12 ships FastAPI + docker compose. Add Prometheus monitoring, Sentry error tracking, and an independent model server (vLLM / TGI) for scale.
- **Read RAGFlow source (optional)**: see [`docs/reference/ragflow-notes/`](./docs/reference/ragflow-notes/) for chapter-aligned source excerpts; not on the default reading path.

## Acknowledgements

- [**RAGFlow**](https://github.com/infiniflow/ragflow) — primary industrial reference; see [`docs/reference/ragflow-notes/`](./docs/reference/ragflow-notes/) for chapter-aligned source excerpts. RAGFlow evolves, so excerpts may go stale.
- [**learn-claude-code**](https://github.com/shareAI-lab/learn-claude-code) — inspiration for the "minimal runnable MVP per chapter" format.
- [**BGE models**](https://github.com/FlagOpen/FlagEmbedding) (BAAI) — default embedding and reranker.

## License

MIT — see [LICENSE](./LICENSE).
