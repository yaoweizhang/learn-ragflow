"""s00 / Unit 01 — RAG = 检索 + 增强 + 生成

Offline demo. Shows what the LLM's prompt looks like
- without RAG (only the user's question, drawing on parametric memory)
- with RAG (the question + retrieved context, drawing on non-parametric memory)
"""


# Toy "retrieved context" — what a RAG pipeline would inject
RETRIEVED_CHUNK = """\
[1] (server_whitepaper.pdf#2) 三、整机规格 内存:32 × DDR4 DIMM 插槽,最大 4TB 容量。
[2] (server_whitepaper.pdf#4) 五、可靠性 内存支持镜像、备用与纠错码(ECC)三种数据保护模式.\
"""

def without_rag(question: str) -> str:
    return f"[without RAG]\nQ: {question}\nA: (LLM draws on parametric memory only)\n"

def with_rag(question: str) -> str:
    context_block = f"<context>\n{RETRIEVED_CHUNK}\n</context>\n"
    return f"[with RAG]\n{context_block}Q: {question}\nA: (LLM grounded by non-parametric context)\n"

def main() -> None:
    q = "R3630 G5 配备多少内存插槽?"
    print("RAG = 检索 + 增强 + 生成\n" + "=" * 40)
    print("\n关键词:参数化知识 = 模型权重;非参数化知识 = 外部索引。")
    print("RAG 让 LLM 同时用两类知识:训练时学的 + 实时查的。\n")
    print(without_rag(q))
    print(with_rag(q))
    print("差异:同一个问题,带 RAG 时 prompt 里多了 <context> 块,")
    print("LLM 可以'引用' [1][2] 给答案背书 — 这就是增强(Augmented)。")

if __name__ == "__main__":
    main()