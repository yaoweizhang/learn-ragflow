# s05 / Unit 02 — Chroma 检索:cosine distance → similarity top-k

> 由浅入深第 2 步：在 unit 01 持久化的 `_chroma/` 上做查询，把"嵌入 + cosine 召回 + 分数翻一下"这一闭环跑通。  
> 本单元是后续 s06+ 混合检索的"dense-only"前置切片。

## 这是什么

`code.py` 把上一单元留在磁盘的 collection 重新打开，对 query 文本做本地 BGE 嵌入（跟 unit 01 同款 `BAAI/bge-small-zh-v1.5`），然后用 cosine 距离取 top-k。

- `_open_collection()` — `chromadb.PersistentClient(path=DB_DIR).get_collection("docs")`，如果 `_chroma/` 不存在或 collection 还没建出来，返回 `None` 让 `main()` 提示"先跑 unit 01"；
- `_embed([query])` — 内联本地 BGE（跟 unit 01 同款，保证 query 向量落在跟索引同款空间）；
- `search(col, query_vec, k=5) -> list[dict]` — `col.query(query_embeddings=[...], n_results=k)` 拿到 `documents / metadatas / distances` 三列，把 `page` 从字符串还原回 `int` 或 `None`，把 `1 - cosine_distance` 翻成 `score ∈ [-1, 1]`（BGE 归一化后实际落在 `[0, 1]`），统一返回 `{text, source, page, score}`；
- `main()` — 拿输入 query → embed → search(k=3) → 打印 `[source#page] score=... | snippet`。

## 跑起来

**前置**：必须先跑过 unit 01，否则 `_chroma/` 不存在。

```bash
python s05_vector_index/units/01_chroma_build/code.py   # 先建索引
python s05_vector_index/units/02_chroma_query/code.py    # 再查
```

期望输出（输入 `应收账款`）：

```
top-3 hits (query='应收账款'):
  [disclosure.docx#None] score=0.499 | 报告期内，公司实现营业收入人民币 28.74 亿元...
  [disclosure.docx#None] score=0.487 | 第二节 主要财务数据
  [disclosure.docx#None] score=0.449 | 第四节 分季度财务数据
```

分数严格递减，3 条都来自 `disclosure.docx`，`page=None` 是 docx 加载器没编页号的契约（`s02_doc_loading/code.py:29` 故意设的），不是 bug。

## 它做对了什么

- **cosine 而不是 L2**：索引建在 cosine space，查询也按 cosine 取，符合 BGE 训练时的相似度目标；
- **metadata 跟结果一起回**：`source` / `page` 不用二次查，直接在 `res["metadatas"][0][i]` 拿到；下游可以按 `where={"source": "x.pdf"}` 在 query 时再做一次过滤；
- **本地 embed + 持久化索引 = 离线闭环**：模型、向量库全在本地，不需要 API key 也不需要起服务；演示 / 调试 / 单机 demo 都够用；
- **cosine distance → similarity 翻一下**：`score = 1 - distance` 让下游用统一 `[-1, 1]` / `[0, 1]` 量纲比较，UI 显示更直观。

## 它做错了什么

- **单进程锁 + 不支持并发写**：`PersistentClient` 同进程多线程读 OK，但写会触发 SQLite/HNSW 文件互斥；多 worker 并发写会撞锁——生产上要走专用服务进程 + 读写分离；
- **没有 score 阈值**：top-k 强制返回 k 条，哪怕最差的一条 score 接近 0 也照样返回；应该加一个 `where_score >= 0.5` 之类的下限，把噪音砍掉；
- **dense-only 召回**：本单元只走向量通道，没有 BM25 / 全文召回兜底；事实型 query（精确词、专业术语）容易输给"近义但错"的 chunk——这是 s06 混合检索要补的洞；
- **没分页**：固定 top-k=3，不支持 `offset / page_size`；生产 API 要按页拉，得自己 wrap 一层 `_rerank_window`（参考 RAGFlow `search.py:525-547`）。

## 对照 ragflow 怎么做的

RAGFlow 在这一层把全文召回（BM25）和向量召回（dense）**用同一个 DB query 一起发**，在 ES / Infinity 侧用 `FusionExpr("weighted_sum", topk, {"weights": "0.05,0.95"})` 融合，再按 API 入参 `vector_similarity_weight` 互补地算全文权重（`search.py:582-637`）；最后 `rerank_with_knn` 用 `tkweight=0.3 / vtweight=0.7` 再做一次线性加权叠 `rank_feature`。**本单元就是 RAGFlow 链路里"dense-only + DB 内部 fusion 前"那个最朴素的切片**——只走了 `MatchDenseExpr` 这一路，没 BM25 兜底也没 PageRank / tag boost。粗召回偏向量（`0.05, 0.95`），精排也偏向量（`0.3, 0.7`），但全文信号从不缺席。

参考：[`docs/reference/ragflow-notes/hybrid_retrieval.md`](../../../../docs/reference/ragflow-notes/hybrid_retrieval.md)  
另见 [`docs/reference/ragflow-notes/vector_indexing.md`](../../../../docs/reference/ragflow-notes/vector_indexing.md)（"为什么 Chroma 没进选型"——分片 / 副本 / 多租户 / BM25 都没原生）

## 思考题

**`score = 1 - cosine_distance` 在向量未归一化时会得到什么？为什么 BGE 配 cosine 比配 L2 更"准"？**

提示：cosine = `1 - cos_sim`，`cos_sim = (a·b) / (||a||·||b||)`；向量归一化后 `||a|| = ||b|| = 1`，cosine 距离只剩 `1 - a·b`，跟内积（点积）等价。未归一化时短文本向量天然小、长文本向量天然大，L2 距离会被向量长度主导而不是语义主导——RAGFlow 的 `_rank_feature_scores` 也是同样的"先归一再加权"逻辑（`sqrt(sum(s*s))` 归一）。