# s06 Retrieval — Vector + BM25 + Hybrid

Vector + BM25 hybrid with cosine similarity on the s05 Chroma index, fused
by an `alpha`-weighted sum (`alpha * vec + (1 - alpha) * bm25`); BM25 is
implemented inline as a teaching tool (Okapi BM25, k1=1.5, b=0.75) with
a tokenizer that handles Chinese (1-2 char sliding window) plus English
word tokens. RAGFlow uses `FusionExpr("weighted_sum", ...)` at the DB
level and `rerank_with_knn(tkweight, vtweight)` at the application
level, with `vector_similarity_weight` exposed as a per-query-type
parameter rather than a constant — the MVP here collapses those three
layers into a single `alpha` knob for clarity.