# s05 Vector Indexing

Chroma persistent index with cosine distance and metadata filtering.
`s05_vector_index/code.py` ships two reusable functions: `build_index(chunks,
vectors) -> chromadb.Collection` writes the chunks to `s05_vector_index/_chroma/`
(added to `.gitignore` as `**/_chroma/`) using `chromadb.PersistentClient`
with `metadata={"hnsw:space": "cosine"}` so HNSW ranks by cosine distance;
`search(col, query_vec, k=5) -> list[dict]` calls `col.query(...)` and
converts the raw cosine distance back to a similarity score with
`score = 1 - distance`, returning a uniform `{text, source, page, score}`
shape. End-to-end: 2 sample docs (PDF + docx) → 901 chunks → 901 512-dim
BGE vectors → index → interactive query (`应收账款` returns 3 disclosure
hits with scores 0.950 / 0.926 / 0.850). RAGFlow's `vector_indexing.md`
explains why production picks Elasticsearch or Infinity over Chroma:
multi-tenant sharded indices, native BM25 + vector fusion in a single
`bool` query, and `terms / range / rank_feature` metadata filtering that
Chroma's `$eq / $in / $and / $or` only-`where` clause cannot match.
