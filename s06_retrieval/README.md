# s06 检索 — 向量 + BM25 + 混合

## 问题

纯向量召回：把查询 embedding 和 chunk embedding 比余弦。问题——同义词/改写
能找到，但**完全一致的关键词**反而排不到第一（embedding 把"内存"和"RAM"
拉得近，但语料里"内存"是字面词）。纯 BM25：能精准匹配"应收账款"这种
字面术语，但**完全找不到同义词**（"营收" vs "营业收入"）。

## 最小解法

跑 `python s06_retrieval/code.py`，输入查询，函数
`hybrid_search(col, query, query_vec, k, alpha)` 会同时：

1. 用 s05 的 Chroma 索引做向量近邻（拿 top-20 余弦相似度）；
2. 自己实现一个微型 BM25 在所有 chunk 上重打分；
3. 把两路分各自归一到 [0,1]，按 `alpha * vec + (1 - alpha) * bm25` 合并。

`tokenize()` 对中文按 1-2 字滑动窗口 + 英文单词拆分，这样纯中文段落也
能被 BM25 命中。

## 跑起来

```bash
python s06_retrieval/code.py
# 问: 内存
```

预期（实测，alpha=0.5）：

```
[server_whitepaper.pdf#24] score=0.976 | 图12 内存标识 ...
[server_whitepaper.pdf#2]  score=0.905 | 2 内存 ...
[server_whitepaper.pdf#24] score=0.869 | 4.2 内存 ...
```

`alpha=0` 退化成纯 BM25（同上、纯字面）；`alpha=1` 退化成纯向量（语义
近但字面不一定出现）。alpha=0.5 是兜底默认。

## 真实世界的问题

1. **BM25 自己实现太慢**——每查一次都要重新分词 + 重新算 df/tf/IDF。
   真实系统把 inverted index 离线建好，查询时只算 `tf` 部分；这次 MVP
   因为 chunk 数小就无所谓。
2. **不同 query 应该用不同 alpha**——事实型查询（"应付账款"）应
   `alpha=0.2`（重 BM25），概念型查询（"为什么营收下滑"）应 `alpha=0.8`
   （重向量）。生产系统把 alpha 做成检索接口的可配参数，而不是常数。
3. **chunk_id 排序影响后续生成**——LLM 看到的是 top-K 拼接顺序，BM25 分
   和向量分同样重要时要保证高 BM25 的命中不会因为低向量分被压到队尾。
   这次用 `alpha * v + (1 - alpha) * b` 加权融合简单处理；RAGFlow
   做了 DB 侧 `weighted_sum` + 应用侧 `rerank_with_knn` 两级。

## ragflow 怎么做的

见 [ragflow_notes/hybrid_retrieval.md](../ragflow_notes/hybrid_retrieval.md)。
要点：DB 侧 `FusionExpr("weighted_sum", ...)` 做粗召回 + 应用侧
`rerank_with_knn(tkweight=0.3, vtweight=0.7)` 做精排，权重由调用方
按 query 类型传入，而不是写死的常数。

## 思考题

- **如果 alpha=0 应该退化成什么？alpha=1 呢？**
  alpha=0 → 完全忽略向量分，只看 BM25 字面打分；适合精确术语检索
  （"应收账款金额前 5 名"）。alpha=1 → 完全忽略 BM25，只看余弦相似度；
  适合改写/同义词多的查询（"内存不足怎么办"→ 找到"内存故障处理"）。
  实际工程上需要交叉验证集（人工标注的相关 chunk）来挑 alpha——
  经验上 0.3~0.7 之间都合理，极端值（0 或 1）只在特定场景才合适。
  本章 MVP 把 alpha 写成可配参数就是为了能快速 sweep 出最优点。