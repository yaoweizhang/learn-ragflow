# s03 Text Chunking

Naive paragraph-based chunker with hard cap: paragraphs shorter than
`max_chars` are kept as one chunk; longer paragraphs are split at Chinese /
English sentence boundaries (`[.。!?！？]`); oversized runs without any
delimiter (e.g. table specs) are hard-cut by characters as a fallback. Each
output chunk carries a `chunk_id = {source}#{page}#p{n}` for downstream
references. RAGFlow extends this with hierarchical parent-child chunks and
token-aware splitting — see `../ragflow_notes/deepdoc_chunking.md` for the
production design.
