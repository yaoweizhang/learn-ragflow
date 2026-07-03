# s07 Reranking — BGE-reranker for precise reordering

A cross-encoder reranker (BAAI/bge-reranker-base) is dropped on top of
s06's hybrid top-K: every `(query, chunk_text)` pair is fed together
through a BERT-style model that does token-level cross-attention and
emits a `[0, 1]` relevance score, after which the candidates are
re-sorted and the top-3 are returned. The model is loaded once via
`@lru_cache(maxsize=1)` so subsequent calls in the same process only
pay the inference cost (~100-300 ms for top-10). Compared to s06's
bi-encoder scoring, the cross-encoder can see exact word matches and
pushes literal-term hits (e.g. "内存") above paraphrased chunks that
bi-encoder cosine alone would rank higher. RAGFlow generalises this
second stage through `RerankModel.Base` in `rag/llm/rerank_model.py`
— Jina / Cohere / Voyage / Qwen / local HuggingFace cross-encoder —
all normalised to `[0, 1]` and blended with `tksim` via
`tkweight * tksim + vtweight * vtsim`; whether to even invoke the
second stage is gated by `if rerank_mdl and sres.total > 0` in
`Dealer.retrieval`, keeping cost opt-in.