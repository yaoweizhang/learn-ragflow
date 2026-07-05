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

## 第三层信号：PageRank + tag boost `_rank_feature_scores`

`Dealer.rerank_with_knn` 的最终公式是 `sim = tkweight*tksim + vtweight*vtsim + rank_fea`，
rank_fea 来自 [`_rank_feature_scores`](https://github.com/infiniflow/ragflow/blob/828c5789/rag/nlp/search.py#L330-L361)：

```python
def _rank_feature_scores(self, query_rfea, search_res):
    rank_fea = []
    pageranks = []
    for chunk_id in search_res.ids:
        pageranks.append(search_res.field[chunk_id].get(PAGERANK_FLD, 0))
    pageranks = np.array(pageranks, dtype=float)

    if not query_rfea:
        return np.array([0 for _ in range(len(search_res.ids))]) + pageranks

    q_denor = np.sqrt(np.sum([s * s for t, s in query_rfea.items() if t != PAGERANK_FLD]))
    if q_denor == 0:
        return np.array([0 for _ in range(len(search_res.ids))]) + pageranks

    for i in search_res.ids:
        nor, denor = 0, 0
        if not search_res.field[i].get(TAG_FLD):
            rank_fea.append(0)
            continue
        tag_feas = parse_tag_features(search_res.field[i].get(TAG_FLD), allow_json_string=True,
                                      allow_python_literal=True)
        if not tag_feas:
            rank_fea.append(0)
            continue
        for t, sc in tag_feas.items():
            if t in query_rfea:
                nor += query_rfea[t] * sc
            denor += sc * sc
        if denor == 0:
            rank_fea.append(0)
        else:
            rank_fea.append(nor / np.sqrt(denor) / q_denor)

    return np.array(rank_fea) * 10.0 + pageranks
```

**两层叠加**：
- **PageRank（`PAGERANK_FLD`）**：建索引时图遍历出来的文档权威度，每条 chunk 自动继承——
  不用手工设定。权威文档的 chunk 天然排前。
- **Tag cosine（`TAG_FLD`）**：用户给 chunk 打标签（向量 / 权重字典），`TAG_FLD` 存；
  查询时如果 query 也带 `query_rfea`（同款 tag 字典），按 cosine 相似度算 boost；
  没有 tag 的 chunk 直接退化成纯 PageRank 加权。

`rank_feature=rank_fea` 直接加进 `sim`（见上面 `rerank_with_knn` 第 46 行）。
MVP 完全没这一层——所有 chunk 一视同仁，权威文档没被抬高。

## 分页与候选窗口对齐 `_rerank_window`

[search.py:525-547](https://github.com/infiniflow/ragflow/blob/828c5789/rag/nlp/search.py#L525-L547)
解决"分页和块拉取不对齐"这个真实的生产 bug：

```python
def _rerank_window(page_size: int, top: int = 0) -> int:
    """Candidate-window size shared by retrieval's block fetch and slice.

    retrieval reuses this value BOTH as the backend block size and as
    the modulus for extracting a single page from a (re)ranked block::

        req["page"] = global_offset // window   # which block to fetch
        begin       = global_offset %  window   # where the page starts

    For those two to agree the window MUST be an exact multiple of
    page_size; otherwise blocks and pages drift apart and deep
    pagination silently drops results and returns short pages.

    The window targets a provider-friendly pool of ~64 candidates, bounded
    by top when given (i.e. when an external reranker is active), and is
    always rounded UP to a whole number of pages to preserve the invariant.
    """
    if page_size <= 1:
        return min(30, top) if top > 0 else 30
    window = math.ceil(64 / page_size) * page_size
    ...
```

调用点（[search.py:576-590](https://github.com/infiniflow/ragflow/blob/828c5789/rag/nlp/search.py#L576-L590)）：

```python
RERANK_LIMIT = self._rerank_window(page_size, top if rerank_mdl else 0)
...
"page": global_offset // RERANK_LIMIT + 1,
"size": RERANK_LIMIT,
...
begin = global_offset % RERANK_LIMIT
```

**为什么有这个**：后端（ES / Infinity）只给"整个 block"打分；前端要"第 page_size 条"。
如果用 `global_offset // page_size` 算 block 索引、用 `(global_offset % page_size)` 切 block 内部，
page_size 不是 block 大小因子时，跨块的 1-page 切片会和原文位置错位——
深分页要么漏结果要么返回短页。

**RAGFlow 的做法**：拉块时**强制把窗口调大到 page_size 的整数倍**（向上取整到 ~64 候选），
后端一次返回 `RERANK_LIMIT` 大小的 block，前端在内存里切片：
`begin = global_offset % RERANK_LIMIT`、`end = begin + page_size`。
一个窗口公式同时控制两个偏移量，永远对齐。

MVP 是"固定 topk=10"——根本没有分页概念。生产里如果一页 5 条、第 100 页，
没对齐就会出问题。MVP 不做分页不是因为概念难，是因为我们的 topk=10 默认就是"一屏看完"。