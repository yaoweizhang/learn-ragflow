# s06 / Unit 02 — 混合召回 fusion (BM25 + dense cosine, α-weighted)

> 由浅入深第 2 步：把 unit 01 的 BM25 字面分和 dense cosine 相似度做加权融合，  
> 让"完全一致关键词"和"语义近邻"两条信号同时参与排序。

## 这是什么

`code.py` 实现一个可注入的 `hybrid_topk(docs, query, query_vec, dense_score_fn, k, alpha)`：

- `dense_score_fn(chunk) -> float` —— 由调用方注入，本单元就地实现"对每条 chunk 暴力算 cosine"（chunks 都在内存里、量级小）。生产里换成 chroma `col.query` 或 ES `knn`；
- `BM25(docs).score(query)` —— 复用 unit 01 的 hand-written BM25，拿到每个 chunk 的字面分；
- **归一 + 加权**：BM25 分除以 `max(bm25_scores)` 落到 [0,1]，dense cosine 已经是 [0,1]（BGE `normalize_embeddings=True`），按 `final = alpha * vec + (1 - alpha) * bm25_norm` 融合；默认 `alpha=0.95` 偏向量，mirrors ragflow `FusionExpr("weighted_sum", {"weights": "0.05,0.95"})` 的权重大小关系（dense=0.95, keyword=0.05）；
- `main()` —— 内联 pypdf + python-docx + 500 字符 cap 切块 + 本地 BGE + unit 01 的 BM25，跑固定 query `"应收账款 计提"`，打印 hybrid top-3，**两个子分都可见**（不只是融合分）。

`hybrid_topk` 是这一章的核心 API：unit 02 就是把"怎么把两路召回合在一起"封装成一个独立函数，下一章 (s07) 会在这上面叠 `rerank` / `rerank_with_knn`。

## 跑起来

```bash
python s06_retrieval/units/02_hybrid_fusion/code.py
```

输出示例（实测，alpha=0.95）：

```
loaded 34 chunks from samples/
query='应收账款 计提', alpha=0.95 (dense-dominant)
hybrid top-3 (with both sub-scores visible):
  [disclosure.docx#-] final=0.946 = 0.95*vec(0.957) + 0.05*bm25(1.413) | ...
  [disclosure.docx#-] final=0.910 = 0.95*vec(0.922) + 0.05*bm25(1.310) | ...
  ...
```

每个 hit 同时打出 `vec` / `bm25` / `final`，方便看"是 dense 拉上去的还是 BM25 拉上去的"。

## 它做对了什么

- **同时吃下两路信号**：dense cosine 解决"同义 / 改写"，BM25 解决"完全一致关键词"；`alpha=0.95` 时向量主导但 BM25 仍能在 dense 几乎打平时把字面命中顶上来；
- **score 归一**：两路分先各自归一到 [0,1]（dense 已经是 cosine ∈ [0,1]，BM25 除以 max）→ 加权时不会出现"一边量纲 0~100、另一边 0~1"的偏置；
- **API 解耦**：`hybrid_topk` 接 `dense_score_fn` 而不是绑死 chroma / ES / numpy，单元测试可以直接喂 list[dict] + 暴力 cosine；生产替换为 `col.query` / `knn_search` 一行 import 替换；
- **可解释**：每个 hit 的 `dense` / `bm25` / `score` 都打印出来，bad case 立刻能看出是哪一路偏了；
- **chunk_id 可追溯**：fusion 后命中结构带 `chunk_id`，跟 s05 的 `chunk_id` 命名一致，下一章 (s07) 做 rerank / chunk 拉取直接复用。

## 它做错了什么

- **没有 per-query alpha 调度**：写死 `alpha=0.95` 对所有 query 同权；事实型查询（"应付账款多少"）应该更偏 BM25（alpha 低），概念型查询（"营收下滑的原因"）更偏 dense（alpha 高）——RAGFlow 把 `vector_similarity_weight` 做成检索接口的入参，由调用方按 query 类型传；
- **假设两路信号都存在**：如果 docs 是空、或 dense_score_fn 全 0，hybrid_topk 仍然返回结构但都是 0 分；没有"任一路缺失就退化成另一路"的兜底——生产里 `Dealer.search` 在 ES 全文召回失败时会跳过 fusion 单独走向量；
- **没有第三层信号**：RAGFlow 的 `sim = tkweight*tksim + vtweight*vtsim + rank_fea` 里 `rank_fea` 是 PageRank + tag boost，本单元完全没有——权威文档没被抬高，带标签的 chunk 也没特殊对待；
- **没有精排**：`hybrid_topk` 直接返回粗排 top-k；RAGFlow 在粗排后再 `rerank_with_knn(tkweight, vtweight, rank_fea)` 用 cross-encoder 重打分。本章 MVP 不做精排，s07 会接上；
- **没有分页 / 候选窗口对齐**：固定 top-5，没有 `_rerank_window(page_size, top)` 这种"块大小 = page_size 整数倍"的工程细节——本节不深陷这块，s07+ 才会碰到。

## 对照 ragflow 怎么做的

RAGFlow 的 `Dealer.search` 在 ES / Infinity 内部用 `FusionExpr("weighted_sum", {"weights": "0.05,0.95"})` 做粗召回（`ragflow/rag/nlp/search.py:189-196`），dense=0.95 / keyword=0.05 跟本单元 `alpha=0.95` 权重大小关系一致；之后 `Dealer.rerank_with_knn`（`search.py:443-472`）再用 `tkweight * tksim + vtweight * vtsim + rank_fea` 做精排。**两阶段都是线性加权**：DB 内 fusion 粗排 + 应用侧 rerank 精排，链路简单、可解释、能叠加第三层信号。RAGFlow 的 alpha 不是常数 —— `vector_similarity_weight` 是检索接口的入参，事实型查询可调低、概念型查询可调高。本单元把 alpha 写成 `hybrid_topk` 的参数就是给这条路径留入口。

参考：[`docs/reference/ragflow-notes/hybrid_retrieval.md`](../../../../docs/reference/ragflow-notes/hybrid_retrieval.md)

## 思考题

**`alpha=0.95` 和 `alpha=0.5` 在 query `"内存"` 上有什么差别？能不能用同一份 docs 跑两组对比，看 top-5 命中集合是不是变了？**

提示：α 越大，dense cosine 的相对权重越高 → `RAM` 这种同义词的命中越靠前；α=0.5 时 BM25 和 dense 各占一半，`内存`字面命中会更强，但 `RAM` / `存储器` 这种同义词会被稀释。在 unit 02 的 `main()` 里改 `alpha=0.5` 跑一次、再改 `alpha=0.05` 跑一次，比较三组 top-3 的 chunk_id 是否重叠——这就是最简单的"per-query alpha sweep"实验。