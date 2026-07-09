# s08 Prompt & Generation

After s07 reranks the top-K chunks, the LLM still has no idea which piece of context
backs which claim. This chapter wraps the reranked hits into a single `<context>` block,
prepends a rule-bound prompt that forces citation markers like `[1]`, `[2]`, and adds
an explicit "I don't know" refusal instruction so the model never fabricates an answer
when the retrieved chunks don't actually cover the question. The `answer()` function
returns both the model's `text` and a `citations` list (source + page per hit) so the
upstream UI can render clickable references. The production hard parts — prompt
injection defence, long-context truncation, and citation number drift — are addressed
in the chapter README; RAGFlow's multi-template approach (separating sufficiency check,
query rewrite, and citation re-pass) is summarised in
[docs/reference/ragflow-notes/prompt_templates.md](../docs/reference/ragflow-notes/prompt_templates.md).
