# s07 思考题

## 如果召回了 100 个、rerank 要跑多少对？

**100 对** (1 query × 100 candidates)。Cross-encoder 的 query+chunk
是 1 对 1，不是 1 对 N。100 个 chunk 就是 100 对，O(n) 不是 O(n²)。

注意原 plan/brief 这里写的是 10000 对，那是因为把 BM25 + 向量那种
双塔检索的笛卡尔积混淆进来了——cross-encoder 没有 N×N 那回事。

时间粗算：单对 cross-encoder forward 在 CPU 上 ~3-5ms，GPU 上 ~1-2ms。
- top-10：~30-100ms，可以接受；
- top-100：~300ms-1s，还能撑；
- top-1000：~3-10s，**线上不可接受**。

这就是为什么生产 RAG 系统都把"召回量"压到 cross-encoder 能吃的范围
（一般 50-100），再多就让粗召回用**更便宜的近似**顶住：

- 向量召回用 **IVF / HNSW 索引**（s05 的 Chroma 就是 HNSW），把
  O(N) 全扫变成 O(log N) 近邻；
- BM25 用**倒排表 + 跳表**而不是线性扫；
- 召回完了再用 cross-encoder 在小池子上精排。

RAGFlow 把这条原则硬编码进了 `_rerank_window`
（[ragflow/rag/nlp/search.py:548-571](ragflow/rag/nlp/search.py#L548-L571)）：

```python
window = math.ceil(64 / page_size) * page_size
if top > 0:
    window = min(window, math.ceil(top / page_size) * page_size)
```

——粗召回池子永远卡在 ~64 候选，配 `rerank_mdl` 时更小（`top` 参数
封顶）。这是"召回 1000 个再 rerank"和"召回 64 个再 rerank"的工程差距。

## 那为什么不直接让 LLM rerank？

理论上 GPT-4 / Claude 看 100 个 chunk 给出 top-3 是终极方案，实际上：

- **延迟**：100 个 chunk + 一个 query 进 GPT-4-128k 一次 ~2-10s；
- **钱**：输入 token × 100 chunk 平均 500 token ≈ 50K token / 次
  ≈ $0.5-$1（GPT-4o 价位）；
- **不稳**：LLM 输出的"前 3 名"格式需要后处理解析，且对长 chunk
  容易"前部偏置"（chunk 开头被看得多、结尾被看得少）。

所以 RAGFlow 把 LLM 风格的 rerank 做成**多 provider 抽象**
（Jina / Cohere / Voyage / Qwen / 本地 HF cross-encoder，见
`ragflow/rag/llm/rerank_model.py`），让租户按预算选；cross-encoder
的"快 + 准 + 便宜"组合通常是默认推荐——我们的 MVP 也是这个选择。