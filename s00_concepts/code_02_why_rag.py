"""s00 / Unit 02 — 为什么用 RAG:vs 长上下文 vs 微调

Offline demo. Computes approximate token counts for 3 strategies:
- Long-context: every query ships the full corpus in the prompt.
- RAG: every query ships only the top-k retrieved chunks.
- Fine-tune: every query ships the question; knowledge lives in weights.

Uses tiktoken if available; falls back to char/4 heuristic if not installed.
"""
from __future__ import annotations

# Toy corpus size: 10k chunks × ~400 chars ≈ 4MB raw text.
CORPUS_CHARS = 10_000 * 400
TOP_K_CHARS = 3 * 400  # 3 retrieved chunks per RAG query
QUESTION_CHARS = 50

def count_tokens(chars: int) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode("x" * chars))  # proxy: tokenize placeholder
    except ImportError:
        return chars // 4  # rough rule of thumb

def main() -> None:
    long_ctx = count_tokens(CORPUS_CHARS + QUESTION_CHARS)
    rag      = count_tokens(TOP_K_CHARS + QUESTION_CHARS)
    ft       = count_tokens(QUESTION_CHARS)
    print("策略     每次 query 喂给 LLM 的 token 数(粗估)")
    print("-" * 50)
    print(f"长上下文  {long_ctx:>8} tokens  (全语料每次都塞,贵)")
    print(f"RAG       {rag:>8} tokens  (只塞 top-3 命中,便宜 {long_ctx / rag:.0f}x)")
    print(f"微调      {ft:>8} tokens  (知识在权重里,但改不动)")
    print()
    print("代价矩阵:")
    print("- 长上下文: 改知识秒级,但 token 钱按 query × 语料烧")
    print("- RAG:      改知识秒级,且只塞相关段,钱按 query × top-k 烧")
    print("- 微调:     改知识按小时/天,改完后查询最便宜,但失去了'热拔插'")
    print()
    print("经验法则:能不动模型就别动模型(RAG > 微调);")
    print("知识实时性高 + 体积大 → RAG;模型行为/风格定制 → 微调。")

if __name__ == "__main__":
    main()