# s05 向量索引 (Vector Indexing)

## 问题
s04 已经把 chunk 变成 512 维向量了,但这些向量还"飘"在内存里 —— 程序
一关,几万条 embedding 就丢了;重新跑又要喂一遍 BGE 算几秒到几分钟。
需要一个**持久化的、能按相似度召回的、带元数据过滤**的容器,把
"chunk ↔ 向量 ↔ source/page"三者绑在一起存到磁盘。

## 最小解法
`code.py` 实现了两个函数 + 一个 `main`:
- `build_index(chunks, vectors) -> chromadb.Collection` —— 用
  `chromadb.PersistentClient` 把集合写到 `s05_vector_index/_chroma/`,
  `metadata={"hnsw:space": "cosine"}` 指定余弦距离;HNSW 索引
  (近似最近邻) 走 `hnswlib`,元数据走 SQLite。
- `search(col, query_vec, k=5) -> list[dict]` —— 把 `col.query` 返回
  的 `distances` 翻一下: `score = 1 - distance`(cosine 距离 ∈ [0,2]
  转成相似度 ∈ [-1,1],我们 BGE 归一化后实际落在 [0,1]),统一返回
  `{text, source, page, score}`。
- `main` —— 加载 PDF+docx → `chunk_by_paragraph` → `embed` →
  `build_index` → 等用户输入 query → `embed([query])` →
  `search(k=3)` → 打印。

```bash
cd D:/study/rag_study/learn-ragflow
python s05_vector_index/code.py
```

## 跑起来
首次跑会:
1. 删旧 `_chroma/`(确保干净);
2. 加载 2 份样本 → 901 个 chunk → 901 条 512 维向量;
3. 写入 `s05_vector_index/_chroma/chroma.sqlite3` + 一个 HNSW 索引目录;
4. 提示 `问: ` → 输入中文问句(已加 .gitignore:`**/_chroma/`,不入库)。

期望输出(`应收账款`):

```
[disclosure.docx#None] score=0.950 | 1. 应收账款
[disclosure.docx#None] score=0.926 | 3. 应收账款
[disclosure.docx#None] score=0.850 | 24. 应收账款
```

分数严格递减,3 个都来自 `disclosure.docx`,符合预期。`page=None`
是 docx 加载器没编页号(`s02_doc_loading/code.py:29` 故意设的),
不是 bug。

## 真实世界的问题
1. **数据增长到百万级 Chroma 撑不住**。`PersistentClient` 把 HNSW 索
   引和 SQLite 放单台机器,**单点故障 + 不能水平扩展**;RAGFlow 直接
   弃用 Chroma,选了 `Elasticsearch` 或 `Infinity` 这两个既支持
   `dense_vector` 字段又支持 BM25 全文的"二合一"引擎,详见
   `../ragflow_notes/vector_indexing.md`。
2. **元数据过滤不灵活**。Chroma 的 `where` 只支持 `$eq / $in / $and / $or`
   几样,**做不了 `range` 数值范围、`rank_feature` 加权、按权限标签桶
   聚合**;RAGFlow 在 ES 上跑 `bool + knn + terms + range + rank_feature`
   一锅炖,业务侧要做"按部门 / 时间段 / 标签"切片时不必换系统。
3. **没有副本 / 分片**。Chroma 文件级持久化意味着备份只能整个拷贝;
   ES 默认 1 副本 + 后续 `number_of_replicas` 拉起来,Infinity
   `ConnectionPool(max_size=4)` (`infinity_conn_pool.py:48-58`) 多
   worker 共享一个池,**生产环境的"挂了别丢数据"是底线**。

## RAGFlow 怎么做的
详见 `../ragflow_notes/vector_indexing.md`。一句话总结:**选 ES / Infinity
这种"BM25 + 向量二合一"的引擎,牺牲单点便利,换多租户分片、混合检索、
可观测聚合**。ES 的 `create_idx` 模板 (`es_conn_base.py:128-141`) 用一
份 `mapping.json` 统一管 `settings + mappings`(分片数、副本数、
`dense_vector` 字段类型、HNSW 构造参数),改一个文件全集群生效。

## 思考题
**元数据过滤 `source = server_whitepaper.pdf` 怎么写?**

答: 见 `thinking_answers.md`。
