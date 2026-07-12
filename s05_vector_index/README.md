# s05 向量索引 (Vector Indexing) — 把向量 + 元数据写进可查询的库

> **本章定位**：s05 是 RAG 全景的"向量承载层"——把 s04 输出的 512 维 embedding 向量绑回 chunk 文本 + 来源元数据，落盘成可重复查询的索引；在线时由 s06+ 把用户 query 投影到同空间做 top-k 召回。详细定位见 s00 §1.4；RAGFlow 实现见本章末"## RAGFlow 实现"。

---

## 一、章节介绍

### 1.1 核心定义：什么是向量索引？

**向量索引 (Vector Index)** 是一个把 *embedding 向量 + 原始文本 + 元数据* 三者绑在一起、落到磁盘、并提供 *近似最近邻 (ANN) 查询 + 元数据过滤* 的容器。它的上游是 s04 的 embedding，下游是 s06+ 的召回 / rerank / prompt 拼装。

```
   s04 chunk + vec                Chroma collection("docs")
   {text, source, page}            ┌─ HNSW 索引 (hnswlib, cosine)
   [vec₁, vec₂, ...]               │   百万级毫秒级 ANN
        │                          ├─ SQLite: documents + metadata (source/page)
        │  col.add(embeddings=...) │
        ▼                          └─ PersistentClient → _chroma/ 目录
   chroma.PersistentClient
   ─────────▶  query(q_vec, where={...})
                                          │
                                          ▼
                                s06 召回: top-k hits 带 (text, source, page, score)
```

把它放进 RAG 全景看：**s05 是把"飘在内存里的向量"变成"可持久化、可检索的数据结构"**。如果只把 s04 跑出来扔进 Python `list`、每次启进程重算一次，几万条 chunk 还行，几十万条就要算几分钟；更糟的是 —— 你**根本没法在线查**。s05 解决两件事：**持久化**（进程关掉不丢）和 **ANN 查询**（百万级向量毫秒级 top-k）。

#### 三个核心部件

s05 的代码把所有事情塞在 Chroma 一个 collection 里，但拆开看是三个独立部件：

1. **ANN 索引 (HNSW)** —— 一种基于图的近似最近邻算法（`hierarchical navigable small world`）。多层邻近图让搜索从 O(N) 降到 O(log N），代价是 *近似* 而非 *精确*（返回的 top-k 不一定是全局最近，只是大概率最近）。Chroma 0.5 用的是封装好的 `hnswlib`，距离度量由 `metadata={"hnsw:space": "cosine"}` 选 cosine。
2. **元数据存储 (SQLite)** —— 每条向量附带几列 *标量字段*（我们这里两列：`source` / `page`），走 SQLite 单文件存。查询时可以 `where={"source": "x.pdf"}` 这种"先按元数据过滤、再在子集上跑向量近邻"的两段式召回。
3. **持久化层 (PersistentClient)** —— Chroma 把 ANN 索引 + SQLite 数据库 + 文档原文一起写到本地目录（`s05_vector_index/_chroma/`），下次启动直接 `PersistentClient(path=...)` 打开，**不丢数据、不重 embed**。

把这三件事拼回 RAG 全景，跟传统数据库的对比一目了然：

| 向量索引 (Chroma / HNSW) | 传统 RDBMS (MySQL / Postgres) | 共同目标 |
|---|---|---|
| 高维稠密向量 (512 维 float) | 结构化标量 (text / int / date) | 把数据存起来、按条件拉 |
| 相似性搜索 (ANN, 近似) | 精确匹配 (B-Tree,精确) | 都是"找"的入口 |
| `HNSW` / `IVF` / `PQ` | `B-Tree` / `Hash Index` | 索引机制 |
| AI / RAG / 推荐 | 业务系统 / 报表 | 适用场景 |

向量库和传统数据库不是替代关系，**是互补关系** —— 同一份业务里，向量库管"语义召回"，RDBMS 管"用户/订单/权限"。

#### cosine vs L2 vs inner_product —— 距离度量选哪个

HNSW 的距离空间选错了，再贵的 ANN 索引都是"近朱者赤近墨者黑"反着来。本节挑三种最常见的距离度量对比：

| 度量 | 公式 | 受向量长度影响 | 何时用 |
|---|---|---|---|
| **L2 (欧氏距离)** | `sqrt(Σ(a_i - b_i)²)` | 是 —— 长向量天然距离大 | 原始未归一化特征、图像像素 |
| **cosine (余弦距离)** | `1 - (a·b) / (||a||·||b||)` | 否 —— 比较"方向" | 文本 embedding (BGE / OpenAI text-embedding),`normalize=True` 之后 = 内积 |
| **inner_product (内积)** | `a·b` | 是 —— 长向量天然内积大 | 已经归一化到单位球的向量(此时内积 = cosine) |

s04 的 BGE 输出 `normalize_embeddings=True`，所有向量都落在单位球面（`||v|| = 1`）。此时 `cosine 距离 = 1 - 内积`，**选 cosine / inner_product 等价，但 cosine 量纲更直观** —— `0 = 完全相同、1 = 完全无关`。本教程固定选 `cosine`，跟 BGE 训练时的相似度目标对齐。

> Chroma 在 SQLite 里存 `documents` 字段（即 chunk 的完整 text），HNSW 索引只存向量 + id 映射；query 时通过 id 反查 SQLite 拿到原文。

#### 元数据过滤：pre-filter vs post-filter

向量库经常被问到"在某个子集里找最像的"，比如"只在 server_whitepaper.pdf 的第 2 页之后找"。这个 "子集筛选" 在索引层面有两条实现路径：

- **post-filter（后过滤）** —— 先向量召回 top-k × N（比如 k×10），再按元数据过滤，N 不够多时就漏召回。简单但召回率打折；
- **pre-filter（预过滤）** —— 先按 `where` 把候选集砍到子集，再在子集上跑向量近邻。**召回率不打折**，但元数据列没有 ANN 索引时，SQLite 全扫一遍的延迟可能盖过向量召回；

Chroma 走的是 **per-collection 元数据 + 简单 SQLite WHERE**(`where` 解析成 SQL `WHERE ...`），过滤完再在子集上跑 HNSW；**元数据列本身没有 ANN 索引**，百万级 + 多列同时过滤时会慢。要做"大规模按权限 / 时间段 / 多标签"过滤，得换 ES / Infinity（见 §四）。

### 1.2 真实世界的问题：为什么单独写一章

`chromadb.PersistentClient(...).create_collection(...)` 几行就能跑，看起来不值得单独一章。但把它接进真实 RAG 链路就会发现，"跑通"和"在 prod 不爆"之间也隔着一道悬崖 —— 这道悬崖由几类典型问题堆起来。

#### 真实世界的问题

1. **维度对不上会爆索引**（从 s04 §2.1 延续）。同一 collection 混用 512 维 BGE 和 1536 维 OpenAI 会让 Chroma 在 insert 阶段直接 `Dimension mismatch` 报错；生产代码在 `BuiltinEmbed.MAX_TOKENS` 用字典显式登记维度上限，s05 这一层只能用"一 collection 一维度 + 重建"的笨办法保证不混。`vectors[0]` 硬编码 512 是 BGE-small-zh 的输出维度；换模型（`bge-large-zh-v1.5` 是 1024 维）整张 collection 必须删掉重建，**没有 schema 迁移**。
2. **元数据过滤做不深**。Chroma 的 `where` 只支持 `$eq / $ne / $in / $nin / $gt / $gte / $lt / $lte` 数值比较 + `$and / $or` 复合，**做不了全文 BM25、做不了 `rank_feature` 加权、做不了聚合桶**(`terms` / `date_histogram`）；业务侧要做"按部门 / 时间段 / 标签"切片时，`where` 撑不住。生产代码在 ES 上跑 `bool + knn + terms + range + rank_feature` 一锅炖，把全文打分和向量打分在同一轮查询里融合。
3. **top-k 召回 vs 精度的取舍**。ANN 是 *近似* 算法 —— `n_results=k` 返回的 k 条不一定是全局最像的 k 条，只是大概率最像。HNSW 通过 `ef / M` 两个参数控制"图搜得多深"，值越大越接近暴力搜索但越慢；Chroma 0.5 默认 `M=16, ef_construction=100, ef=10`，对几十条 chunk 够用，百万级要调。更糟的是 **top-k 强制返回 k 条，哪怕最差一条 score 接近 0 也照样返回** —— 应该加一个 `where_score >= 0.5` 之类的下限把噪音砍掉，但 Chroma 没有原生 `where` 分数过滤，得在后处理里加。

#### 这些问题为什么必须显式面对

每条都对应不同的工业级解法 —— schema 注册、混合检索、分数阈值。这些是 s06+ 的主题。**s05 的目标不是解决它们，而是把它们显式暴露出来，让你看到 toy 方案的边界**。

这也是为什么本章用两个代码文件递进：

- **code_01** —— 跑通最小骨架（`PersistentClient` + `create_collection` + `add`）；删旧 `_chroma/` 重建而非增量 upsert，保证幂等；
- **code_02** —— 重新打开 `_chroma/`，对 query 做本地 embed + cosine top-k 召回，把 `1 - distance` 翻成 `[0, 1]` 区间的 `score`；演示"cosine + 归一化向量 = 内积"这套距离翻分的工程闭环。

这也是为什么我们不直接用 FAISS / Milvus / Pinecone 这些更"生产"的库 —— FAISS 没有 `where` 元数据过滤，Pinecone 是托管服务要钱，Milvus 是分布式集群单机器跑不起来。Chroma 是"最小可用 toy"，**看得到每一步**。先见边界，再看生产，比直接用封装库学到的多。

---

## 二、Chroma 持久化索引构建 (PersistentClient + cosine)：[code_01_chroma_build.py](code_01_chroma_build.py)

入口：[`code_01_chroma_build.py`](code_01_chroma_build.py)

把"飘在内存里"的 embedding 落到磁盘上，绑上 source / page 元数据，形成可重复检索的索引。code_02 会在同一份 `_chroma/` 上做 query，演示 cosine distance → similarity 的转换。

### 这是什么

`code.py` 干一件事：**把 chunk 列表 + 向量列表 + metadata 一起写进 Chroma 的 `PersistentClient`**。

- `get_chunks_and_vectors()` — 加载 samples 下的 PDF/DOCX、内联 500 字符 cap + 句界切、内联本地 BGE(`BAAI/bge-small-zh-v1.5`，`normalize_embeddings=True`），返回 `(chunks, vectors)`；
- `build_index(chunks, vectors)` — 先 `shutil.rmtree(DB_DIR)` 确保干净（重建而非增量），再用 `chromadb.PersistentClient(path=DB_DIR)` 建出 collection `docs`，`metadata={"hnsw:space": "cosine"}` 指定 HNSW 用 cosine 距离，最后 `col.add(ids=, embeddings=, documents=, metadatas=)` 一次性写入；
- metadata 维度只有两列：`source`（文件名）和 `page`(PDF 才有，DOCX 留空字符串）；这两列就是后续 `where` 过滤能用的字段。

### 跑起来

```bash
python s05_vector_index/code_01_chroma_build.py
```

输出：

```
indexed 34 chunks into _chroma/ (collection=docs, dim=512)
```

首次跑会建 `s05_vector_index/_chroma/chroma.sqlite3` + HNSW 索引目录；再次跑会先 `rmtree` 整个目录保证"重建而非合并"，这是当前最简单的正确性取舍。

### 实际输出

把 code_01 跑在仓库自带的 `samples/` 上，磁盘上长这样：

```
s05_vector_index/_chroma/
├── chroma.sqlite3                              # SQLite: 文档原文 + metadata + collection 元数据
└── <uuid>/                                     # HNSW 索引目录:hnswlib 二进制文件
    ├── data_level0.bin
    ├── header.bin
    ├── link_lists.bin
    └── length.bin
```

内存中的 collection 形状（用 `peek()` 看到的）：

```python
collection.peek()["metadatas"][:2]
# [
#   {"source": "server_whitepaper.pdf", "page": "1"},   # 注意 page 是 str 不是 int
#   {"source": "disclosure.docx",       "page": ""},    # DOCX 没有 page,空串
# ]

collection.peek()["embeddings"][:1][0][:5]
# [-0.028, 0.041, -0.013, -0.057, 0.009]                # 512 维,前 5 个分量
```

下游 s06+ 拿到 `search()` 返回的 `list[{text, source, page, score}]` 时，**不需要知道背后是 Chroma / Milvus / FAISS** —— 它只关心四个字段。这是 s05 把"向量库选型"封装掉的价值：**后续章节按统一接口消费即可**，换底层只改 `build_index` / `search` 两个函数。

### 它做对了什么

- **持久化**：`_chroma/` 落到磁盘，关掉 Python 不丢；下次启动直接 `PersistentClient(path=...)` 打开；
- **cosine 距离空间**：`metadata={"hnsw:space": "cosine"}` 显式指定 HNSW 的距离度量，跟 BGE 训练时的余弦目标对齐；
- **元数据跟向量绑在一起**：`source` / `page` 走 SQLite 单独存，检索时可以 `where={"source": "x.pdf"}` 过滤；
- **删旧重建而非合并 upsert**：保证幂等——跑两次结果一样，不会因为重跑塞重复 `chunk_id`。

### 它做错了什么

- **重建 = rm 整棵目录树**：每次都把整个 `_chroma/` 删掉重建，几千 chunk 还行，几十万 chunk 时每次重建要重算 HNSW 图，分钟级成本；下一步要学 ES 的 `_bulk` 增量 upsert；
- **没有远程 Chroma**：服务端、客户端、单机、HA（高可用）四件事 Chroma 都没原生成熟方案，单点故障 + 不能水平扩展——生产里这条路走不通；
- **维度硬编码**：`vectors[0]` 是 512，因为 BGE-small-zh 是 512 维；换模型（比如 `bge-large-zh-v1.5` 是 1024）整个 collection schema 要重建，**没有 schema 迁移**；
- **metadata 太薄**：只有 `source` / `page` 两列；想做"按部门 / 时间段 / 标签"切片，得在 schema 里加更多列，**目前不在设计里**。

### troubleshooting

- `ModuleNotFoundError: No module named 'chromadb'`： `pip install chromadb`（同时拉 `onnxruntime` / `hnswlib` / `pysqlite3-binary` 等传递依赖）；Windows 下 `pysqlite3-binary` 是关键依赖，不带它 chromadb 加载 SQLite 会崩。
- `Could not import google.protobuf` / `onnxruntime` 加载报错：首次跑 BGE 时 chromadb 内部 `all-MiniLM-L6-v2` 默认 embedding 函数会触发 protobuf 解析，可以 `pip install protobuf` 兜底；**我们用 `col.add(embeddings=...)` 传预算好的向量绕过**。
- `Dimension mismatch` / `[InvalidArgumentError]`： `vectors[0]` 维度跟 collection 期望不一致 —— 多半是中途换了 `EMBED_MODEL`(512 → 1024），旧 collection 留着但新向量塞不进去。**删 `_chroma/` + 重跑 code_01**。
- 第一次跑 code_02 报 "未发现持久化索引"：先跑 code_01 让 `_chroma/` 有数据。
- Windows GBK 控制台打印中文报 `UnicodeEncodeError`：控制台编码问题不是代码 bug，跑前 `set PYTHONIOENCODING=utf-8`。
- `_chroma/` 占了 1-2 GB 但 git 不收： `.gitignore` 里有 `**/_chroma/`，是预期行为；**清缓存就 `rm -rf s05_vector_index/_chroma/`**。

---

## 三、Chroma 检索：cosine distance → similarity top-k：[code_02_chroma_query.py](code_02_chroma_query.py)

入口：[`code_02_chroma_query.py`](code_02_chroma_query.py)

在 code_01 持久化的 `_chroma/` 上做查询，把"嵌入 + cosine 召回 + 分数翻一下"这一闭环跑通。
本节是后续 s06+ 混合检索的"dense-only"前置切片。

### 这是什么

`code.py` 把上一节留在磁盘的 collection 重新打开，对 query 文本做本地 BGE 嵌入（跟 code_01 同款 `BAAI/bge-small-zh-v1.5`），然后用 cosine 距离取 top-k。

- `_open_collection()` — `chromadb.PersistentClient(path=DB_DIR).get_collection("docs")`，如果 `_chroma/` 不存在或 collection 还没建出来，返回 `None` 让 `main()` 提示"先跑 code_01"；
- `_embed([query])` — 内联本地 BGE（跟 code_01 同款，保证 query 向量落在跟索引同款空间）；
- `search(col, query_vec, k=5) -> list[dict]` — `col.query(query_embeddings=[...], n_results=k)` 拿到 `documents / metadatas / distances` 三列，把 `page` 从字符串还原回 `int` 或 `None`，把 `1 - cosine_distance` 翻成 `score ∈ [-1, 1]`(BGE 归一化后实际落在 `[0, 1]`），统一返回 `{text, source, page, score}`；
- `main()` — 拿输入 query → embed → search(k=3) → 打印 `[source#page] score=... | snippet`。

### 跑起来

**前置**：必须先跑过 code_01，否则 `_chroma/` 不存在。

```bash
python s05_vector_index/code_01_chroma_build.py   # 先建索引
python s05_vector_index/code_02_chroma_query.py    # 再查
```

期望输出（输入 `应收账款`）：

```
top-3 hits (query='应收账款'):
  [disclosure.docx#None] score=0.499 | 报告期内,公司实现营业收入人民币 28.74 亿元...
  [disclosure.docx#None] score=0.487 | 第二节 主要财务数据
  [disclosure.docx#None] score=0.449 | 第四节 分季度财务数据
```

分数严格递减，3 条都来自 `disclosure.docx`，`page=None` 是 docx 加载器没编页号的契约（`s02_doc_loading/code_01_basic_load.py` 故意设的），不是 bug。

### 实际输出

Code_01 的预期输出（具体数字由 `samples/` 决定）：

```
indexed 34 chunks into _chroma/ (collection=docs, dim=512)
```

34 是 4 页白皮书 + 27 段披露报告 → 34 个 chunk；512 是 BGE-small-zh 的输出维度。**首次跑会从 HF Hub 下载 ~100MB 的 BGE 模型**，后续跑复用本地缓存，只算 embedding。

Code_02 的预期输出（输入 `应收账款`）：

```
top-3 hits (query='应收账款'):
  [disclosure.docx#None] score=0.950 | 1. 应收账款
  [disclosure.docx#None] score=0.926 | 3. 应收账款
  [disclosure.docx#None] score=0.850 | 24. 应收账款
```

分数严格递减，3 条都来自 `disclosure.docx`（因为 "应收账款" 真的只出现在 DOCX 里）；`page=None` 是 docx 加载器没编页号的契约（`s02_doc_loading/code_01_basic_load.py` 故意设的），不是 bug。**分数 ~0.95 而不是 ~1.00 的根因是 cosine 距离被翻成相似度后还有语义噪声** —— "1。 应收账款" 和 "24。 应收账款" 标题下的内容不同，BGE 编码后向量微差。

### 它做对了什么

- **cosine 而不是 L2**：索引建在 cosine space，查询也按 cosine 取，符合 BGE 训练时的相似度目标；
- **metadata 跟结果一起回**：`source` / `page` 不用二次查，直接在 `res["metadatas"][0][i]` 拿到；下游可以按 `where={"source": "x.pdf"}` 在 query 时再做一次过滤；
- **本地 embed + 持久化索引 = 离线闭环**：模型、向量库全在本地，不需要 API key 也不需要起服务；演示 / 调试 / 单机 demo 都够用；
- **cosine distance → similarity 翻一下**：`score = 1 - distance` 让下游用统一 `[-1, 1]` / `[0, 1]` 量纲比较，UI 显示更直观。

### 它做错了什么

- **单进程锁 + 不支持并发写**：`PersistentClient` 同进程多线程读 OK，但写会触发 SQLite/HNSW 文件互斥；多 worker 并发写会撞锁——生产上要走专用服务进程 + 读写分离；
- **没有 score 阈值**：top-k 强制返回 k 条，哪怕最差的一条 score 接近 0 也照样返回；应该加一个 `where_score >= 0.5` 之类的下限，把噪音砍掉；
- **dense-only 召回**：本节只走向量通道，没有 BM25 / 全文召回兜底；事实型 query（精确词、专业术语）容易输给"近义但错"的 chunk——这是 s06 混合检索要补的洞；
- **没分页**：固定 top-k=3，不支持 `offset / page_size`；生产 API 要按页拉，得自己 wrap 一层 `_rerank_window`（参考 RAGFlow `search.py:525-547`）。

### troubleshooting

- `Collection docs does not exist`：code_01 没跑过 / `_chroma/` 目录删了；按"前置"步骤先跑 code_01。
- `_open_collection()` 返 None：第一件事是 `main()` 提示"先跑 code_01"——属于预期行为，不算 bug。
- `UnicodeEncodeError: 'gbk' codec can't encode character`：Windows 控制台编码问题，跑前 `set PYTHONIOENCODING=utf-8`(s05 / s06 / s07 同问题）。
- Chroma 0.5。x 在 Windows 上可能 `Cannot convert None to MetadataValue`：`page=None` 触发；`code_01` 第 33 行做了 `str(c.get("page", ""))` 转空串兜底，如果还报错就是 Python 版本问题，降到 3.11。
- 维度对不上导致 `Dimension mismatch`：见 code_01 troubleshooting 的对应条目；删除旧 `_chroma/` 重跑。

---

## 四、核心函数一览

| 函数 | 文件 | 输入 | 输出 | 一句话解释 |
|---|---|---|---|---|
| `_pdf(path)` / `_docx(path)` | `code_01_chroma_build.py` | `Path` | `list[{text, page, source}]` | 复制 s02 的 `load_pdf` / `load_docx`;走同一份 `{text, page, source}` schema |
| `_split_long(text, max_chars)` / `_chunk_by_paragraph(docs, max_chars=500)` | `code_01_chroma_build.py` | `str` / `list[dict]` | `list[str]` / `list[dict]` | 复制 s03 的 chunker:500 字符 cap + 句界切 + `chunk_id` |
| `_model()` / `_embed(texts)` | `code_01_chroma_build.py` | `list[str]` | `list[list[float]]` | `@lru_cache(maxsize=1)` 加载 BGE-small-zh-v1.5;`normalize_embeddings=True` |
| `get_chunks_and_vectors()` | `code_01_chroma_build.py` | — | `tuple(list[dict], list[list[float]])` | 把 `samples/` 走 s02 → s03 → s04 链路 → `(chunks, vectors)`;在 01 内自包含 |
| `build_index(chunks, vectors)` | `code_01_chroma_build.py` | `list[dict], list[list[float]]` | — | `shutil.rmtree(_chroma)` + `PersistentClient` + `cosine` collection + `add(embeddings=, metadatas=, ids=)` |
| `main()` (01) | `code_01_chroma_build.py` | — | 交互输入 query | 01 演示入口,`build_index` 后调用 `search()` 验证一次 |
| `_open_collection()` | `code_02_chroma_query.py` | — | `chromadb.Collection` | 打开 `_chroma/` 里 `chunks` collection;不存在 → `sys.exit(1)` |
| `search(col, query_vec, k=5)` | `code_02_chroma_query.py` | `(Collection, list[float], int)` | `list[{text, source, page, chunk_id, score}]` | `col.query(query_embeddings=[vec], n_results=k)`;`distance → similarity = 1 - distance` |
| `main()` (02) | `code_02_chroma_query.py` | — | 打印 top-k hits | 02 演示入口:加载 01 的 collection + 交互输入 query |

## 五、跨代码 schema 设计取舍

为什么 collection 的 metadata 字段只有 `source` / `page` 两列，而不是"作者 / 创建时间 / 部门 / 标签"全塞进去？几个常见取舍的折中：

- **多列 vs 单列 JSON**：Chroma 0.5 的 `where` 要求 metadata 字段是 *强类型 scalar*，塞 JSON 进去会被强制 `str()` 序列化、`$gt` 之类的运算符全部失效。MVP 阶段只放两列能让 code_02 直接演示 `where={"source": "x.pdf"}`；生产想加列，得想清楚每列是不是要走 `WHERE` 过滤（过滤列才值得占 metadata 位）。
- **删旧重建 vs 增量 upsert**：`build_index` 走 `shutil.rmtree(DB_DIR)`，粗暴但幂等 —— 跑两次结果完全一样，不会因为重跑塞重复 `chunk_id`。生产里走 `col.upsert(...)` 增量更新更快，但要小心 `chunk_id` 重用（旧 chunk 文本变了但 id 不变，HNSW 里就指向"过期向量"）；通常两边都要：**upsert 主路径 + 定期 full rebuild 兜底**。
- **HNSW 参数让 Chroma 默认 vs 显式指定**：`metadata={"hnsw:space": "cosine"}` 是我们**显式指定**的（Chroma 默认 `l2` —— 不显式就翻车）。`M / ef_construction / ef` 三个 HNSW 图参数走 Chroma 默认值（M=16 / ef_construction=100 / ef=10），三十几条 chunk 测不出来；百万级要改成 `M=32, ef_construction=200, ef=50`，在 `create_collection` 的 metadata 里加 `"hnsw:M": 32` 之类。
- **`page` 转字符串存，查询时再翻回 int**：`add` 阶段 `metadatas=[{"source": c["source"], "page": str(c.get("page", ""))} for c in chunks]` —— Chroma 0.5。x 拒绝 int + string 混合（会 `Cannot convert None to MetadataValue` / native segfault）。`search` 阶段用 `try: int(page_val) else None` 翻回来。**类型抖动被吸收在 s05 层，下游 s06+ 拿到的 `page` 又是 `int / None`**。
- **不做 server mode / 不做 embedding 内置到 collection**：`col.add(embeddings=...)` 显式传入已经算好的向量，**不让 Chroma 内部调 embedding**。这样 s05 跟 s04 是显式数据流（embeding 函数由调用方控制，不是 Chroma 黑盒），换 embedding 模型时只改 s04，不动 s05。如果用 `col.add(documents=[...])` 让 Chroma 自动 embed，就得在创建 collection 时配 `embedding_function=...`，把模型绑死。

如果你的语料需要"按权限 / 时间段 / 标签"过滤，就在 metadata 里加字段 —— 但**保持字段是 scalar(`str / int / float / bool`)**，别塞 JSON / list。

## 六、跨代码文件集成

01 写索引 → 02 读查询；两者通过 `_chroma/` 目录解耦。**没有 01 持久化的索引，02 没有任何数据可查**（因为 02 是"读"半边，不是"嵌入 + 写入"半边）。生产里也走这个模式——离线 pipeline 跑完 01 之后就停了，在线 service 只跑 02（或更上层的 s06+），持久化文件被独立 service 持有、SIGKILL 不丢数据。

**整体拓扑**：(1) 01 走 `samples/ → load → chunk → embed → build_index` 写 `_chroma/` → (2) 02 启动时 `_open_collection()` 拿 collection → (3) 接收 query → (4) `_embed(query)` 算 query 向量 → (5) `col.query(query_embeddings=...)` 拿 top-k hits。**生产差异**：RAGFlow 把这层抽成 `rag/vector_store/` 接口,具体实现三选一(Elasticsearch / Infinity / OceanBase),`doc_engine` 字段从 `.env` 读决定用哪个;s05 toy 锁死 Chroma,生产换库需改 import。


## RAGFlow 实现

RAGFlow 的向量索引层是抽象接口 `rag/vector_store/`，具体实现分三种：`Elasticsearch`（生产首选，支持完整 metadata 过滤）、`Infinity`（自研数据库，向量 + 全文 + 结构化混合查询）、`OceanBase`（阿里分布式数据库）。`doc_engine` 字段从 `.env` 读决定用哪个。

**设计取舍**：不绑死单一向量库；让用户按"团队技术栈 + 文档规模 + 是否需要全文混查"三选一。Chroma 这种单机玩具不进生产候选。

详细摘录与 5-15 行 "为什么这样写" 的分析见 [`docs/reference/ragflow-notes/vector_indexing.md`](../docs/reference/ragflow-notes/vector_indexing.md)。

---

## 选型速记

### 主流向量库速览

下面这张表把社区常用的几类向量方案按"是否纯本地 / 是否支持 BM25 / 部署形态 / ANN 算法"列出来，方便选型时快速对照：

| 库 | 本地 / 服务 | 是否支持 BM25 | ANN 算法 | 多租户隔离 | 适用场景 |
|---|---|---|---|---|---|
| **ChromaDB**(本教程 demo) | 本地 | 否(只 `where`) | HNSW (`hnswlib`) | 单 collection | 原型 / 教学 / 几百万级单租户 |
| **FAISS** (`faiss-cpu`) | 本地(算法库) | 否 | IVF / HNSW / PQ | 无(纯文件) | 离线实验 / 评测 / 不带元数据 |
| **Milvus** | 服务 / 集群 | 间接(`BM25` 需外置) | HNSW / IVF / PQ + GPU | 命名 + 权限库 | 十万到百亿级、生产 |
| **Qdrant** | 服务 / 集群 | 是(原生 sparse + dense) | HNSW(Rust 自研) | collection-level | 中大规模、Rust 性能 |
| **Weaviate** | 服务 / 集群 | 是(原生 hybrid) | HNSW | 多 tenant 字段 | GraphQL + 多模态 AI 应用 |
| **Pinecone** | 托管 Serverless | 否(只 dense) | 自研(基于 SPANN) | namespace 隔离 | 不想运维、要 SLA 保证 |
| **Elasticsearch / OpenSearch** | 服务 / 集群 | 是(原生 BM25 + knn) | HNSW | index / alias | 已有 ES 栈、想要 BM25 + dense 一锅炖 |
| **Infinity** | 服务 / 集群 | 是(原生 fulltext + vector) | HNSW / IVF | database / table | 亿级 + 多列过滤 |

我们的 toy Chroma 在"本地 / 服务 + BM25"上只占第一行 —— 能跑，但 1) 不原生支持 BM25 全文检索（只是元数据 `where`）；2) 没有副本/分片/多租户物理隔离；3) ANN 算法只暴露 HNSW 三参数，不暴露 IVF / PQ。生产规模上通常切到 `Elasticsearch`(BM25 + 向量二合一，运维熟悉）或 `Milvus`（分片副本 + 完整 IVF/PQ + 多租户命名）而不是单进程 Chroma。

- **教学 / 快速原型** → Chroma（本教程）；`PersistentClient` 几行跑通，看得到每一步；
- **中等规模 / 不想运维 / 接受 dense-only** → Qdrant / Weaviate，单二进制部署，带 hybrid；
- **已有 ES 栈 / 想要 BM25 + dense 一锅炖** → Elasticsearch / OpenSearch 的 `dense_vector` 字段；
- **十万到百亿级 / 多租户硬隔离** → Milvus + 分片副本，或 Pinecone（托管）；
- **算法研究 / 离线评测 / 不要 metadata** → FAISS，把向量当 numpy 处理；
- **要先看清每个边界再选** → 用本章 code_02 把 query 跑一遍，看清楚"小数过滤 / 分数阈值 / metadata 切片"在 Chroma 上的体感，再决定要不要换。

### 扩展指南

加一个新 vector store 后端（FAISS / Qdrant / Milvus）只要三步：

1. 写一个 `class FaissIndex` 或 `class QdrantIndex`，**对外暴露和 `code_01_chroma_build.py` 里 `get_collection()` / `col.add()` / `col.query()` 同形的 `upsert(vectors, metadatas, ids)` / `search(query_vec, top_k, where)` 方法**，下游 s06 / s12 不用改一行；
2. 把当前 `_chroma/` 目录的持久化逻辑剥成 `VectorBackend` 抽象类，`main()` 按 `VECTOR_BACKEND` env 选 `ChromaBackend()` / `FaissBackend()` 实例化，不要在 `main()` 里写 `if backend == "faiss": import faiss; ...`；
3. 给代码文件 README 加一段"它跟 Chroma 比，赢在哪 / 输在哪"的对照（FAISS：算法全 / 没有 metadata；Qdrant：原生 hybrid / 单二进制；Milvus：分片副本 / 运维重）。

不要把后端判断塞进 `code_02_chroma_query.py` 的 `main()`——它只懂 Chroma `col.query()`。本章 MVP 只跑 Chroma，但 `VectorBackend` 接口留好了，挂 3 个后端都不动调度逻辑。

---

## 思考题

1. **元数据过滤 `where={"source": "server_whitepaper.pdf"}` 怎么写？Chroma 的 `where` 还支持哪些 operator？如果要"只在第 5 页之后找"怎么写？**
2. **重建索引时 `shutil.rmtree(DB_DIR)` 是不是最干净的取舍？如果换成 `col.upsert(...)` 增量更新，会有什么副作用？**
3. **`score = 1 - cosine_distance` 在向量未归一化时会得到什么？为什么 BGE 配 cosine 比配 L2 更"准"？**

（答案见文末「思考题答案」）

---

## 思考题答案

### Q1. 元数据过滤 `source = server_whitepaper.pdf` 怎么写？

**答：在 `col.query(...)` 里加一个 `where={"source": "server_whitepaper.pdf"}` 参数，Chroma 会把它转成"先按 source 过滤，再在过滤后的子集上跑向量近邻"，SQL 语义等价于 `SELECT ... WHERE source = ? ORDER BY cosine LIMIT k`。**

#### 1. 最小写法

```python
hits = col.query(
    query_embeddings=[query_vec],
    n_results=3,
    where={"source": "server_whitepaper.pdf"},
)
```

就这么一行。Chroma 0.5。x 的 `where` 支持 `{"字段": "值"}` 这种最简形（隐式 `$eq`），也会翻译成 `$eq` 内部表示。

#### 2. 想要"只搜 PDF，排除扫描件"怎么写？

`where` 的"值"也可以是 dict，显式带 operator：

```python
hits = col.query(
    query_embeddings=[query_vec],
    n_results=3,
    where={
        "source": "server_whitepaper.pdf",
        "page": {"$gte": 5},          # 第 5 页之后
        "$or": [
            {"format": "text"},
            {"ocr_done": True},
        ],
    },
)
```

支持的 operator 有 `$eq / $ne / $gt / $gte / $lt / $lte / $in / $nin` 数值比较，以及 `$and / $or` 复合；但**不支持 BM25 全文匹配**（那是 Elasticsearch 的事，见 [`docs/reference/ragflow-notes/vector_indexing.md`](../../docs/reference/ragflow-notes/vector_indexing.md) 第二节"为什么不用 Chroma"）。

#### 3. 元数据 schema 约束

加进去的 `metadatas=[{"source": ..., "page": ...}]` 字段在 Chroma 里是**强类型 + 全 string** —— 我们 `code.py` 第 33 行把 `page` 转成 `str(c.get("page", ""))` 后再塞，因为 Chroma 1.5 拒绝 int+string 混合，0.5。x 拒绝 None（直接抛 `Cannot convert None to MetadataValue`）。如果忘了这步，`col.add` 会当场 segfault（在 Windows + chroma-hnswlib 0.7.6 上是真的 native crash，见 commit message 备注）。

#### 4. 一句话总结

**`where={"source": "..."}` 就够了 —— 把"按业务维度切"和"按相似度排"两步合并成一次原子操作，是 Chroma 比手写 FAISS + 自己维护 dict 的唯一胜场**；真要做权限 / 时间段 / 多标签过滤，Chroma 就得换成 ES 或 Infinity（见 README.md 第二节"真实世界的问题"）。

### Q2. 重建索引时 `shutil.rmtree(DB_DIR)` 是不是最干净的取舍？

`shutil.rmtree` 重建**粗暴但幂等** —— 跑两次结果完全一样，不会因为重跑塞重复 `chunk_id`。

如果换成 `col.upsert(...)` 增量更新会更快（尤其在百万 chunk 时），但要小心：
- `chunk_id` 重用（旧 chunk 文本变了但 id 不变，HNSW 里就指向"过期向量"）；
- metadata 只更新部分字段时的覆盖行为（upsert 是整行覆盖，不是 field-level merge）。

生产环境通常两边都要：**upsert 主路径 + 定期 full rebuild 兜底**。

### Q3. `score = 1 - cosine_distance` 在向量未归一化时

cosine = `1 - cos_sim`，`cos_sim = (a·b) / (||a|| · ||b||)`；向量归一化后 `||a|| = ||b|| = 1`，cosine 距离只剩 `1 - a·b`，跟内积（点积）等价。

未归一化时短文本向量天然小、长文本向量天然大，**L2 距离会被向量长度主导而不是语义主导** —— 一条短查询和一条长文档的 cosine 距离，既有"长度差"也有"语义差"，分不清哪部分是长度贡献、哪部分是语义贡献。生产代码的 `_rank_feature_scores` 也是同样的"先归一再加权"逻辑（`sqrt(sum(s*s))` 归一）。

所以 BGE 配 cosine 比配 L2 更"准"的根本原因：**BGE 训练目标就是 cosine similarity，推理用 cosine 等于和训练目标对齐**；而 L2 距离把"长度信号"当噪声掺进来，召回率打折。
