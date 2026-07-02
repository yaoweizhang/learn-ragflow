# Learn RAGFlow — A 12-Chapter RAG Tutorial, Zero to Deployment

> **This is a Chinese-first tutorial.** Each chapter ships a 30–80 line self-written MVP in Chinese, paired with a guided reading of [RAGFlow](https://github.com/infiniflow/ragflow)'s industrial-grade source. English summaries per chapter will be added on request — the canonical text is in `README.md`. Below is the chapter index with one-line English titles so non-Chinese readers can follow the progression.

This repository is a hands-on, engineering-focused RAG (Retrieval-Augmented Generation) tutorial. Every chapter contains:

- A **self-written MVP** (30–80 lines of Python, single file, minimal dependencies)
- A **RAGFlow reference reading** (key source snippets excerpted into `ragflow_notes/`)
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

## Chapter index

1. [s01](./s01_what_is_rag/) — **What is RAG?** Naive RAG vs. long-context LLMs; minimal end-to-end demo.
2. [s02](./s02_document_loading/) — **Document loading.** PDF / DOCX / OCR parsing; metadata preservation.
3. [s03](./s03_chunking/) — **Text chunking.** Fixed-size vs. structure-aware splitting.
4. [s04](./s04_embedding/) — **Embedding.** BGE / M3E model selection; batching and normalization.
5. [s05](./s05_vector_index/) — **Vector index.** Chroma / in-memory HNSW; distance metrics.
6. [s06](./s06_hybrid_retrieval/) — **Hybrid retrieval.** BM25 + dense vectors; reciprocal rank fusion.
7. [s07](./s07_reranking/) — **Reranking.** Cross-encoder precision; Top-K trade-offs.
8. [s08](./s08_prompt_and_generation/) — **Prompt & generation.** Citation tracing; hallucination control.
9. [s09](./s09_agent_and_tools/) — **Agent & tools.** Function calling; multi-turn retrieval.
10. [s10](./s10_graphrag/) — **GraphRAG.** Entity-relation extraction; graph merging.
11. [s11](./s11_multimodal/) — **Multimodal.** Image-text interleaving; table understanding.
12. [s12](./s12_deployment/) — **Deployment.** FastAPI + Docker; offline evaluation.

> Chapter folders are populated sequentially by Tasks 1–12.

## Project structure

```
learn-ragflow/
├── README.md                    # Chinese canonical readme
├── README.en.md                 # This file (English summary)
├── .env.example                 # Env template (LLM / Embedding / Reranker)
├── requirements.txt             # Aggregated deps across all chapters
├── samples/
│   ├── README.md
│   ├── server_whitepaper.pdf
│   └── disclosure.docx
├── ragflow_notes/               # RAGFlow source excerpts, cited per chapter
│   └── README.md
├── s01_what_is_rag/             # Placeholders — filled by Tasks 1–12
├── ...
└── s12_deployment/
```

## Acknowledgements

- [**RAGFlow**](https://github.com/infiniflow/ragflow) — primary industrial reference; each chapter's section 5 cites excerpts in `ragflow_notes/`.
- [**learn-claude-code**](https://github.com/shareAI-lab/learn-claude-code) — inspiration for the "minimal runnable MVP per chapter" format.
- [**BGE models**](https://github.com/FlagOpen/FlagEmbedding) (BAAI) — default embedding and reranker.

## License

TBD (to be decided by the user; MIT or CC-BY-SA 4.0 suggested).
