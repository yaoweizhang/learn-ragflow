# s01 — What is RAG (English summary)

This chapter introduces Retrieval-Augmented Generation with a 30-line Python toy
that does naive substring search over `samples/disclosure.docx`: the user types a
question, the program returns the first paragraph containing any word from the
question, or `"I don't know."` if nothing matches. It demonstrates the core
"retrieve-then-answer" shape of RAG without any vector database or LLM call,
sets up the two failure modes (no match vs. wrong match) that motivate the rest
of the tutorial, and previews why we will study [RAGFlow](https://github.com/infiniflow/ragflow)
in later chapters.
