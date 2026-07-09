# s05 / Unit 01 — Chroma 持久化索引构建 (PersistentClient + cosine)

> 由浅入深第 1 步：把"飘在内存里"的 embedding 落到磁盘上，绑上 source / page 元数据，形成可重复检索的索引。  
> unit 02 会在同一份 `_chroma/` 上做 query，演示 cosine distance → similarity 的转换。

## 这是什么

`code.py` 干一件事：**把 chunk 列表 + 向量列表 + metadata 一起写进 Chroma 的 `PersistentClient`**。

- `get_chunks_and_vectors()` — 加载 samples 下的 PDF/DOCX、内联 500 字符 cap + 句界切、内联本地 BGE（`BAAI/bge-small-zh-v1.5`，`normalize_embeddings=True`），返回 `(chunks, vectors)`；
- `build_index(chunks, vectors)` — 先 `shutil.rmtree(DB_DIR)` 确保干净（重建而非增量），再用 `chromadb.PersistentClient(path=DB_DIR)` 建出 collection `docs`，`metadata={"hnsw:space": "cosine"}` 指定 HNSW 用 cosine 距离，最后 `col.add(ids=, embeddings=, documents=, metadatas=)` 一次性写入；
- metadata 维度只有两列：`source`（文件名）和 `page`（PDF 才有，DOCX 留空字符串）；这两列就是后续 `where` 过滤能用的字段。

## 跑起来

```bash
python s05_vector_index/units/01_chroma_build/code.py
```

输出：

```
indexed 34 chunks into _chroma/ (collection=docs, dim=512)
```

首次跑会建 `s05_vector_index/_chroma/chroma.sqlite3` + HNSW 索引目录；再次跑会先 `rmtree` 整个目录保证"重建而非合并"，这是当前最简单的正确性取舍。

## 它做对了什么

- **持久化**：`_chroma/` 落到磁盘，关掉 Python 不丢；下次启动直接 `PersistentClient(path=...)` 打开；
- **cosine 距离空间**：`metadata={"hnsw:space": "cosine"}` 显式指定 HNSW 的距离度量，跟 BGE 训练时的余弦目标对齐；
- **元数据跟向量绑在一起**：`source` / `page` 走 SQLite 单独存，检索时可以 `where={"source": "x.pdf"}` 过滤；
- **删旧重建而非合并 upsert**：保证幂等——跑两次结果一样，不会因为重跑塞重复 `chunk_id`。

## 它做错了什么

- **重建 = rm 整棵目录树**：每次都把整个 `_chroma/` 删掉重建，几千 chunk 还行，几十万 chunk 时每次重建要重算 HNSW 图，分钟级成本；下一步要学 ES 的 `_bulk` 增量 upsert；
- **没有远程 Chroma**：服务端、客户端、单机、HA（高可用）四件事 Chroma 都没原生成熟方案，单点故障 + 不能水平扩展——生产里这条路走不通；
- **维度硬编码**：`vectors[0]` 是 512，因为 BGE-small-zh 是 512 维；换模型（比如 `bge-large-zh-v1.5` 是 1024）整个 collection schema 要重建，**没有 schema 迁移**；
- **metadata 太薄**：只有 `source` / `page` 两列；想做"按部门 / 时间段 / 标签"切片，得在 schema 里加更多列，**目前不在设计里**。

## 对照 ragflow 怎么做的

RAGFlow 直接弃用 Chroma，选了 `Elasticsearch` 或 `Infinity` 这两个**"BM25 + 向量二合一"**的引擎：原生支持 `dense_vector` 字段、跨节点分片副本、可水平扩到亿级，**多租户硬隔离**靠 `ragflow_<tenant_id>_<kb_id>` 索引命名规则。Chroma 在 RAGFlow 眼里只是"开发态玩具"，详细对照见 `docs/reference/ragflow-notes/vector_indexing.md`，里面 ES `create_idx` 的 `mapping.json` 模板把 `settings + mappings`（分片数、副本数、`dense_vector` 字段类型、HNSW 参数）一份文件管全集群。

参考：[`docs/reference/ragflow-notes/vector_indexing.md`](../../../../docs/reference/ragflow-notes/vector_indexing.md)  
另见 [`docs/reference/ragflow-notes/embedding_routing.md`](../../../../docs/reference/ragflow-notes/embedding_routing.md)（多 provider 路由的设计取舍，间接影响"换模型 → 重建 collection"的痛感）

## 思考题

**重建索引时 `shutil.rmtree(DB_DIR)` 是不是最干净的取舍？如果换成 `col.upsert(...)` 增量更新，会有什么副作用？**

提示：rm 重建 = 幂等但贵；upsert 增量 = 快但要小心 `chunk_id` 重用（旧 chunk 文本变了但 id 不变，HNSW 里就指向"过期向量"）；生产环境通常两边都要：upsert 主路径 + 定期 full rebuild 兜底。