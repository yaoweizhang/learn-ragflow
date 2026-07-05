# s07 重排序 — BGE-reranker 精排

## Units

| Unit | 标题 | 入口 |
| --- | --- | --- |
| 01 | Cross-encoder 精排 (BGE-reranker 重打分, BEFORE/AFTER 对比) | [`units/01_cross_encoder_rerank/code.py`](units/01_cross_encoder_rerank/code.py) |

## 问题

召回了 10 个 chunk，前 3 个可能不是最好的。s06 的混合召回是**双塔**
（bi-encoder）——query 和 chunk 各自独立编码再算相似度，快但只看到向量
层面的语义接近度，没办法捕捉"查询词"和"chunk 里某个具体词"的精确匹配。
所以字面命中"内存"的 chunk 经常被语义近但字面不沾边的"7.存货"挤到
队尾。

## 最小解法

把 s06 召回的 top-K（这里 K=10）再过一道 **cross-encoder**
（BAAI/bge-reranker-base）：把 `(query, chunk_text)` 拼一起让 BERT
一次性看两端、做 token 级 cross-attention，输出一个 `[0,1]` 的相关性分。
然后按这个分排序、取 top-3。

跑：

```bash
python s07_rerank/code.py
# 问: 内存
```

`rerank(query, hits, top_k=3)` 函数签名：吃 s06 的 hits list，
吐 `[0, top_k)` 个按 rerank 分排好序的 dict，每个带
`rerank_score`（cross-encoder 的分）加上 s06 的 `score`
（混合召回分）。模型用 `@lru_cache(maxsize=1)` 缓存住，
同一进程只下载、加载一次。

## 跑起来

预期（实测，top_k=3 over 10 candidates from s06）：

```
[server_whitepaper.pdf#1]   rerank=0.954 vec=0.905 | ... 内存、10 个 PCIe 4.0 扩展槽位 ...
[disclosure.docx#None]      rerank=0.913 vec=0.575 | 7. 存货 ...
[server_whitepaper.pdf#2]   rerank=0.644 vec=0.976 | 内存 32 × DDR4 3200 ECC RDIMM ...
```

注意 rerank 分数和原 vec 分数**不同步**——s06 把 vec=#1 的混合表
chunk（`#2 内存 32 × DDR4 3200 ECC RDIMM`，vec=0.976）排到第一，但 cross-encoder
觉得它只有 0.644（因为正文是配置表，`内存`只是表里一行），而纯
"2 内存"章节虽然 vec 只有 0.905，rerank 却给到 0.954。这就是
cross-encoder 比 bi-encoder 准的地方：它能看到具体词而不是被一
个向量平均值糊弄。

## 真实世界的问题

1. **rerank 慢**——cross-encoder 是 O(n) 次 BERT forward。
   `top_k=10` 大概 100-300ms；`top_k=100` 直接 1-3s。
   生产上两阶段：先用便宜的双塔召回 ~100-200 候选，再让 cross-encoder
   在小池子上精排；**绝不在 top-1000 上跑 cross-encoder**。
2. **rerank 模型语言错配**——`bge-reranker-base` 主要在英文上训，
   中文任务用 `bge-reranker-v2-m3` 或 `bge-reranker-large` 更稳。
   这次默认 base 是因为模型小、下载快；换中文重排序模型只需改
   `_reranker()` 里那行字符串。
3. **LLM rerank 太贵**——还能再叠一层：拿 rerank 后的 top-N
   让 GPT-4 / Claude 做"哪个最相关"判断。RAGFlow 把这条线
   抽象成 `RerankModel.Base` 的多 provider（Cohere / Voyage / Qwen），
   准但每千次调用都要花 token 钱。本次 MVP 不接——教学仓库只要
   把"cross-encoder 比 bi-encoder 准"这件事讲清楚就够了。

## ragflow 怎么做的

见 [ragflow_notes/rerank.md](../ragflow_notes/rerank.md)。
要点：RAGFlow 在 `Dealer.retrieval` 里把"是否走 cross-encoder / LLM
rerank"做成开关（`if rerank_mdl and sres.total > 0`），配
`rerank_mdl` 才走第二阶段；走的时候还要再和 `tksim`（term similarity）
线性加权一次（`tkweight * tksim + vtweight * vtsim`），不是单看
rerank 分。`RerankModel.Base.similarity` 把十几个云 provider
（Cohere / Jina / Voyage / Qwen / 本地 HF）的输出都归一到 `[0,1]`
再喂进同一个公式。

## 思考题

- **如果召回了 100 个、rerank 要跑多少对？**
  答：**100 对** (1 query × 100 candidates)。Cross-encoder 的
  query+chunk 是 1 对 1，不是 1 对 N。100 个 chunk 就是 100 对，
  O(n) 不是 O(n²)。注意原 plan/brief 这里写的是 10000 对，那是
  因为把 BM25 + 向量那种双塔检索的笛卡尔积混淆进来了——cross-encoder
  没有 N×N 那回事。
  100 对 × ~3ms/对 ≈ 300ms-1s；如果召回 1000 个直接 3-10s，
  线上不可接受。所以**召回量要压到 cross-encoder 能吃的范围**
  （一般 50-100），再多就让粗召回用更便宜的近似（量化向量、IVF 索引）
  顶住。详见 `thinking_answers.md`。