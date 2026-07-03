# s04 Embedding

Three-way embedding switch (local BGE default, OpenAI, Ollama). The
default `local` path downloads `BAAI/bge-small-zh-v1.5` (~100MB on first
run) and produces 512-dim L2-normalised vectors via `sentence-transformers`;
subsequent runs hit an `@lru_cache` and finish in <1s. Set
`EMBED_PROVIDER=openai` to route through the OpenAI-compatible API using
`LLM_API_KEY` / `LLM_BASE_URL`; set `EMBED_PROVIDER=ollama` to POST to a
local Ollama server. RAGFlow's `embedding_routing.md` shows the production
pattern: validate dimensions per model in a dict (e.g.
`BuiltinEmbed.MAX_TOKENS`), declare providers as classes with a
`_FACTORY_NAME`, register them by `inspect` rather than `if/elif`, and
funnel every failure through a single `EmbeddingError` so retry / fallback
behaves uniformly regardless of which SDK raised.