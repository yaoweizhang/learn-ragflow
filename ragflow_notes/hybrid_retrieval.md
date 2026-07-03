# RAGFlow 混合检索：`weighted_sum` + 可配权重

来源：`ragflow/rag/nlp/search.py`（pinned commit `828c5789f651d4c4ebe4645190b8b8d244144fe0`）

## 关键代码段：`weighted_sum` fusion + 本地 `rerank` 加权

`Dealer.search` 阶段把全文召回和向量召回一起送给 ES/Infinity，由底层的
`FusionExpr` 用 weighted_sum 合并（[search.py:189-196](ragflow/rag/nlp/search.py#L189-L196)）：

```python
if settings.DOC_ENGINE_OCEANBASE:
    src.append(f"q_{len(q_vec)}_vec")

fusionExpr = FusionExpr("weighted_sum", topk, {"weights": "0.05,0.95"})
matchExprs = [matchText, matchDense, fusionExpr]

res = await thread_pool_exec(self.dataStore.search, src, highlightFields, filters,
                            matchExprs, orderBy, offset, limit,
                            idx_names, kb_ids, rank_feature=rank_feature)
```

`Dealer.retrieval` 阶段，权重由 API 入参 `vector_similarity_weight` 决定，并
**互补地** 算全文权重（[search.py:582-637](ragflow/rag/nlp/search.py#L582-L637)）：

```python
            vector_similarity_weight=0.3, ...
...
term_similarity_weight = 1 - vector_similarity_weight
logging.debug(
    "[Search] retrieval weights: trace_id=%s kb_count=%s similarity_threshold=%s "
    "vector_similarity_weight=%s full_text_weight=%s rerank_enabled=%s",
    ...
)
```

`Dealer.rerank_with_knn` 把 ES 出来的 KNN 余弦分和本地算的 term 相似度
再线性加权一次（[search.py:443-472](ragflow/rag/nlp/search.py#L443-L472)）：

```python
def rerank_with_knn(self, sres, query, knn_scores, tkweight=0.3, vtweight=0.7, ...):
    ...
    tksim = np.array(self.qryr.token_similarity(keywords, ins_tw), dtype=np.float64)
    vtsim = np.array([knn_scores.get(chunk_id, 0.0) for chunk_id in sres.ids],
                     dtype=np.float64)
    rank_fea = self._rank_feature_scores(rank_feature, sres)
    sim = tkweight * tksim + vtweight * vtsim + rank_fea
    return sim, tksim, vtsim
```

## 为什么这样写（3 个 bullet）

- **为什么用 weighted_sum 而不是 RRF？**
  `weighted_sum` 直接用各通道归一化后的分做线性加权，结果区间直观、可解释、
  也能叠加 `rank_feature`（PageRank、tag boost）这种**额外信号**；
  RRF 只看排名不看分，加 rank_fea 时还得另起一条管线。RAGFlow 在 DB 侧
  做一次 fusion、应用侧再做一次 rerank，二者都用线性加权，链路简单。
  代价是必须先把两路分数归一到同一量纲（ES / Infinity 内部完成），
  否则两个 0-1 量纲不同的分直接相加会偏。

- **为什么 alpha 不是常数？**
  `vector_similarity_weight` 是 **检索接口的入参**，由调用方（API、agent、
  graphrag）按场景传不同值。事实型查询（"应付账款多少"）偏 keyword、
  概念型查询（"营收下滑的原因"）偏向量，所以做成可调。
  同一份 chunk 在不同 query 下，权重组合可能完全不同——一个全局常数
  alpha 没办法同时让两种 query 都跑出 top-1 准确。

- **查询类型怎么影响权重？**
  没有写死的"query-type → alpha"映射；权重来自三层叠加：
  (1) 调用方传入 `vector_similarity_weight`（最大粒度控制）；
  (2) 全文/向量两路在 DB 内 `weighted_sum` fusion，权重 `"0.05,0.95"`
      偏向向量（粗召回阶段要召回更多候选）；
  (3) `rerank_with_knn` 用 `tkweight=0.3, vtweight=0.7`（精排阶段同样偏
      向量，但全文信号仍占 30% 兜底"完全一致关键词"的查询）。
  这种"DB fusion + app rerank + rank_fea"的级联，比一个全局 alpha
  灵活得多，但代价是工程复杂度——本次 MVP 的 `alpha=0.5` 就是把这三层
  压扁成一个旋钮的简化版。