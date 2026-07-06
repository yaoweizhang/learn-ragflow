# s05 向量索引 (Vector Indexing) — 章节总览

> **章节定位**: RAG 离线流水线的"向量承载层" —— 把 s04 输出的 512 维稠密向量绑回 chunk 文本 + 来源元数据,落盘成可重复查询的索引;在线时由 s06+ 把用户 query 投影到同空间做 top-k 召回。  
> **章节结构** 借鉴 [all-in-rag 第三章 向量数据库](https://github.com/datawhalechina/all-in-rag/blob/main/docs/chapter3/08_vector_db.md) 的"是什么 / 为什么 / 怎么做 / 对照 RAGFlow"叙述弧,只取对**单 backend + HNSW + 元数据过滤**这一节有用的部分 —— 不引入 Pinecone / Milvus 全表对比(那些是综述性质、留到延伸阅读),也不展开 IVF / LSH / PQ 等其他 ANN 算法(本教程只跑 HNSW)。

---

## 章节导航 (聚合入口保留)

| Unit | 标题 | 入口 |
| --- | --- | --- |
| 01 | Chroma 持久化索引构建 (PersistentClient + cosine) | [`units/01_chroma_build/code.py`](units/01_chroma_build/code.py) |
| 02 | Chroma 检索 (cosine distance → similarity top-k) | [`units/02_chroma_query/code.py`](units/02_chroma_query/code.py) |

跑法:

```bash
python s05_vector_index/units/01_chroma_build/code.py    # 先建索引
python s05_vector_index/units/02_chroma_query/code.py    # 再查
# 旧路径仍可用 (聚合入口,等价于 unit 01):
python s05_vector_index/code.py
```

依赖: `chromadb`(同时拉 `onnxruntime` / `hnswlib` / `pysqlite3-binary` 等传递依赖)。把 s02 / s03 / s04 跑通,s05 才能跑。

---

## 一、什么是向量索引?

### 1.1 核心定义

**向量索引 (Vector Index)** 是一个把 *稠密向量 + 原始文本 + 元数据* 三者绑在一起、落到磁盘、并提供 *近似最近邻 (ANN) 查询 + 元数据过滤* 的容器。它的上游是 s04 的 embedding,下游是 s06+ 的召回 / rerank / prompt 拼装。

把它放进 RAG 全景看:**s05 是把"飘在内存里的向量"变成"可持久化、可检索的数据结构"**。如果只把 s04 跑出来扔进 Python `list`、每次启进程重算一次,几万条 chunk 还行,几十万条就要算几分钟;更糟的是 —— 你**根本没法在线查**。s05 解决两件事:**持久化**(进程关掉不丢)和 **ANN 查询**(百万级向量毫秒级 top-k)。

### 1.2 三个核心部件

s05 的代码把所有事情塞在 Chroma 一个 collection 里,但拆开看是三个独立部件:

1. **ANN 索引 (HNSW)** —— 一种基于图的近似最近邻算法(`hierarchical navigable small world`)。多层邻近图让搜索从 O(N) 降到 O(log N),代价是 *近似* 而非 *精确*(返回的 top-k 不一定是全局最近,只是大概率最近)。Chroma 0.5 用的是封装好的 `hnswlib`,距离度量由 `metadata={"hnsw:space": "cosine"}` 选 cosine。
2. **元数据存储 (SQLite)** —— 每条向量附带几列 *标量字段*(我们这里两列:`source` / `page`),走 SQLite 单文件存。查询时可以 `where={"source": "x.pdf"}` 这种"先按元数据过滤、再在子集上跑向量近邻"的两段式召回。
3. **持久化层 (PersistentClient)** —— Chroma 把 ANN 索引 + SQLite 数据库 + 文档原文一起写到本地目录(`s05_vector_index/_chroma/`),下次启动直接 `PersistentClient(path=...)` 打开,**不丢数据、不重 embed**。

把这三件事拼回 RAG 全景,跟传统数据库的对比一目了然:

| 向量索引 (Chroma / HNSW) | 传统 RDBMS (MySQL / Postgres) | 共同目标 |
|---|---|---|
| 高维稠密向量 (512 维 float) | 结构化标量 (text / int / date) | 把数据存起来、按条件拉 |
| 相似性搜索 (ANN, 近似) | 精确匹配 (B-Tree,精确) | 都是"找"的入口 |
| `HNSW` / `IVF` / `PQ` | `B-Tree` / `Hash Index` | 索引机制 |
| AI / RAG / 推荐 | 业务系统 / 报表 | 适用场景 |

向量库和传统数据库不是替代关系,**是互补关系** —— 同一份业务里,向量库管"语义召回",RDBMS 管"用户/订单/权限"。

### 1.3 cosine vs L2 vs inner_product —— 距离度量选哪个

HNSW 的距离空间选错了,再贵的 ANN 索引都是"近朱者赤近墨者黑"反着来。本节挑三种最常见的距离度量对比:

| 度量 | 公式 | 受向量长度影响 | 何时用 |
|---|---|---|---|
| **L2 (欧氏距离)** | `sqrt(Σ(a_i - b_i)²)` | 是 —— 长向量天然距离大 | 原始未归一化特征、图像像素 |
| **cosine (余弦距离)** | `1 - (a·b) / (||a||·||b||)` | 否 —— 比较"方向" | 文本 embedding (BGE / OpenAI text-embedding),`normalize=True` 之后 = 内积 |
| **inner_product (内积)** | `a·b` | 是 —— 长向量天然内积大 | 已经归一化到单位球的向量(此时内积 = cosine) |

s04 的 BGE 输出 `normalize_embeddings=True`,所有向量都落在单位球面(`||v|| = 1`)。此时 `cosine 距离 = 1 - 内积`,**选 cosine / inner_product 等价,但 cosine 量纲更直观** —— `0 = 完全相同、1 = 完全无关`。本教程固定选 `cosine`,跟 BGE 训练时的相似度目标对齐。

### 1.4 元数据过滤: pre-filter vs post-filter

向量库经常被问到"在某个子集里找最像的",比如"只在 server_whitepaper.pdf 的第 2 页之后找"。这个 "子集筛选" 在索引层面有两条实现路径:

- **post-filter(后过滤)** —— 先向量召回 top-k × N(比如 k×10),再按元数据过滤,N 不够多时就漏召回。简单但召回率打折;
- **pre-filter(预过滤)** —— 先按 `where` 把候选集砍到子集,再在子集上跑向量近邻。**召回率不打折**,但元数据列没有 ANN 索引时,SQLite 全扫一遍的延迟可能盖过向量召回;

Chroma 走的是 **per-collection 元数据 + 简单 SQLite WHERE**(`where` 解析成 SQL `WHERE ...`),过滤完再在子集上跑 HNSW;**元数据列本身没有 ANN 索引**,百万级 + 多列同时过滤时会慢。要做"大规模按权限 / 时间段 / 多标签"过滤,得换 ES / Infinity(见 §四)。

---

## 二、为什么要单独写一章向量索引?

`chromadb.PersistentClient(...).create_collection(...)` 几行就能跑,看起来不值得单独一章。但把它接进真实 RAG 链路就会发现,"跑通"和"在 prod 不爆"之间也隔着一道悬崖 —— 这道悬崖由几类典型问题堆起来。

### 2.1 真实世界的问题 (3 条典型)

1. **维度对不上会爆索引**(从 s04 §2.1 延续)。同一 collection 混用 512 维 BGE 和 1536 维 OpenAI 会让 Chroma 在 insert 阶段直接 `Dimension mismatch` 报错;RAGFlow 在 `BuiltinEmbed.MAX_TOKENS`(`embedding_model.py:222`)用字典显式登记维度上限,s05 这一层只能用"一 collection 一维度 + 重建"的笨办法保证不混。`vectors[0]` 硬编码 512 是 BGE-small-zh 的输出维度;换模型(`bge-large-zh-v1.5` 是 1024 维)整张 collection 必须删掉重建,**没有 schema 迁移**。
2. **元数据过滤做不深**。Chroma 的 `where` 只支持 `$eq / $ne / $in / $nin / $gt / $gte / $lt / $lte` 数值比较 + `$and / $or` 复合,**做不了全文 BM25、做不了 `rank_feature` 加权、做不了聚合桶**(`terms` / `date_histogram`);业务侧要做"按部门 / 时间段 / 标签"切片时,`where` 撑不住。RAGFlow 在 ES 上跑 `bool + knn + terms + range + rank_feature` 一锅炖(`rag/utils/es_conn.py:141-230`),把全文打分和向量打分在同一轮查询里融合。
3. **top-k 召回 vs 精度的取舍**。ANN 是 *近似* 算法 —— `n_results=k` 返回的 k 条不一定是全局最像的 k 条,只是大概率最像。HNSW 通过 `ef / M` 两个参数控制"图搜得多深",值越大越接近暴力搜索但越慢;Chroma 0.5 默认 `M=16, ef_construction=100, ef=10`,对几十条 chunk 够用,百万级要调。更糟的是 **top-k 强制返回 k 条,哪怕最差一条 score 接近 0 也照样返回** —— 应该加一个 `where_score >= 0.5` 之类的下限把噪音砍掉,但 Chroma 没有原生 `where` 分数过滤,得在后处理里加。

### 2.2 这些问题为什么必须显式面对

每条都对应不同的工业级解法 —— schema 注册、混合检索、分数阈值。这些是 s06+ 的主题。**s05 的目标不是解决它们,而是把它们显式暴露出来,让你看到 toy 方案的边界**。

这也是为什么本章用两个 unit 递进:

- **unit 01** —— 跑通最小骨架(`PersistentClient` + `create_collection` + `add`);删旧 `_chroma/` 重建而非增量 upsert,保证幂等;
- **unit 02** —— 重新打开 `_chroma/`,对 query 做本地 embed + cosine top-k 召回,把 `1 - distance` 翻成 `[0, 1]` 区间的 `score`;演示"cosine + 归一化向量 = 内积"这套距离翻分的工程闭环。

这也是为什么我们不直接用 FAISS / Milvus / Pinecone 这些更"生产"的库 —— FAISS 没有 `where` 元数据过滤,Pinecone 是托管服务要钱,Milvus 是分布式集群单机器跑不起来。Chroma 是"最小可用 toy",**看得到每一步**。先见边界,再看生产,比直接用封装库学到的多。

---

## 三、怎么做?

### 3.1 章节导航

| Unit | 主题 | 它解决什么 | 对照 RAGFlow |
|---|---|---|---|
| [01_chroma_build](./units/01_chroma_build/README.md) | Chroma 持久化索引构建 (PersistentClient + cosine) | "embedding 怎么从内存落到磁盘,绑上 source/page" | `create_idx` 模板 + `ragflow_<tenant>_<kb>` 命名 |
| [02_chroma_query](./units/02_chroma_query/README.md) | Chroma 检索 (cosine distance → similarity top-k) | "按 query 向量找 top-k,把分数翻回人类能理解的相似度" | `MatchDenseExpr + FusionExpr` DB 侧融合 |

### 3.2 跑起来

```bash
python s05_vector_index/units/01_chroma_build/code.py          # 建索引 (~数秒,首次含 BGE 下载)
python s05_vector_index/units/02_chroma_query/code.py           # 提示 "问: " 后输入中文 query
# 旧路径仍可用 (聚合入口,等价于 unit 01):
python s05_vector_index/code.py
```

首次跑 unit 01 会:

1. 删旧 `_chroma/`(确保干净);
2. 加载 `samples/` 下的 PDF + DOCX → 34 个 chunk → 34 条 512 维向量;
3. 写入 `s05_vector_index/_chroma/chroma.sqlite3` + 一个 HNSW 索引目录;
4. 提示 `问: ` → 输入中文 query(加 `.gitignore: **/_chroma/`,不入库)。

Unit 02 必须**先**跑过 unit 01;否则 `_chroma/` 不存在,`main()` 会打印提示并 `sys.exit(1)`。

### 3.3 核心函数一览

s05 的代码薄,但每个函数都对应一种"向量承载"的能力:

| 函数 | 文件 | 输入 | 输出 | 一句话解释 |
|---|---|---|---|---|
| `get_chunks_and_vectors()` | `units/01_chroma_build/code.py` | — | `(chunks, vectors)` | 加载 samples → 切块 → embed(本地 BGE),返回 `(list[dict], list[list[float]])` |
| `build_index(chunks, vectors)` | `units/01_chroma_build/code.py` | `(chunks, vectors)` | `chromadb.Collection` | `rmtree` + `PersistentClient` + `create_collection(metadata={"hnsw:space": "cosine"})` + `col.add(...)` |
| `main()` (unit 01) | `units/01_chroma_build/code.py` | — | 打印索引统计 | 演示入口;加载 → 切 → embed → build → 打印 `indexed N chunks into _chroma/` |
| `_open_collection()` | `units/02_chroma_query/code.py` | — | `Collection / None` | `PersistentClient.get_collection("docs")`;不存在返 `None` 让 `main()` 提示"先跑 unit 01" |
| `search(col, query_vec, k=5)` | `units/02_chroma_query/code.py` | `(Collection, list[float], int)` | `list[{text, source, page, score}]` | `col.query(...)` 拿 `documents / metadatas / distances`,`1 - distance` 翻成 `[0, 1]` 相似度 |
| `main()` (unit 02) | `units/02_chroma_query/code.py` | — | 打印 top-3 hits | 演示入口;`input("问: ...")` → embed → `search(k=3)` → 打印 `[source#page] score=... \| snippet` |

### 3.4 schema 设计取舍

为什么 collection 的 metadata 字段只有 `source` / `page` 两列,而不是"作者 / 创建时间 / 部门 / 标签"全塞进去?几个常见取舍的折中:

- **多列 vs 单列 JSON**:Chroma 0.5 的 `where` 要求 metadata 字段是 *强类型 scalar*,塞 JSON 进去会被强制 `str()` 序列化、`$gt` 之类的运算符全部失效。MVP 阶段只放两列能让 unit 02 直接演示 `where={"source": "x.pdf"}`;生产想加列,得想清楚每列是不是要走 `WHERE` 过滤(过滤列才值得占 metadata 位)。
- **删旧重建 vs 增量 upsert**:`build_index` 走 `shutil.rmtree(DB_DIR)`,粗暴但幂等 —— 跑两次结果完全一样,不会因为重跑塞重复 `chunk_id`。生产里走 `col.upsert(...)` 增量更新更快,但要小心 `chunk_id` 重用(旧 chunk 文本变了但 id 不变,HNSW 里就指向"过期向量");通常两边都要:**upsert 主路径 + 定期 full rebuild 兜底**。
- **HNSW 参数让 Chroma 默认 vs 显式指定**:`metadata={"hnsw:space": "cosine"}` 是我们**显式指定**的(Chroma 默认 `l2` —— 不显式就翻车)。`M / ef_construction / ef` 三个 HNSW 图参数走 Chroma 默认值(M=16 / ef_construction=100 / ef=10),三十几条 chunk 测不出来;百万级要改成 `M=32, ef_construction=200, ef=50`,在 `create_collection` 的 metadata 里加 `"hnsw:M": 32` 之类。
- **`page` 转字符串存,查询时再翻回 int**:`add` 阶段 `metadatas=[{"source": c["source"], "page": str(c.get("page", ""))} for c in chunks]` —— Chroma 0.5.x 拒绝 int + string 混合(会 `Cannot convert None to MetadataValue` / native segfault)。`search` 阶段用 `try: int(page_val) else None` 翻回来。**类型抖动被吸收在 s05 层,下游 s06+ 拿到的 `page` 又是 `int / None`**。
- **不做 server mode / 不做 embedding 内置到 collection**:`col.add(embeddings=...)` 显式传入已经算好的向量,**不让 Chroma 内部调 embedding**。这样 s05 跟 s04 是显式数据流(embeding 函数由调用方控制,不是 Chroma 黑盒),换 embedding 模型时只改 s04,不动 s05。如果用 `col.add(documents=[...])` 让 Chroma 自动 embed,就得在创建 collection 时配 `embedding_function=...`,把模型绑死。

如果你的语料需要"按权限 / 时间段 / 标签"过滤,就在 metadata 里加字段 —— 但**保持字段是 scalar(`str / int / float / bool`)**,别塞 JSON / list。

### 3.5 如何扩展更多向量库

换一个向量库(比如 Milvus / Weaviate / Qdrant / ES 的 `dense_vector`)只要三步:

1. 写一个 `build_xxx(chunks, vectors) -> xxx.Collection`,签名和 `build_index` 一致(吃 chunks 列表 + 向量列表);
2. 写一个 `search_xxx(col, query_vec, k=5) -> list[{text, source, page, score}]`,签名和 `search` 一致;
3. 给单元 README 加一段"它跟 Chroma 比,赢在哪 / 输在哪"的对照(参考 §四的工具速览)。

不要在 `build_index` 里写 `if backend == "chroma": ... elif backend == "milvus": ...` 之类分发 —— 它会污染单一职责。`build_index` 只懂 Chroma,`main()` 懂全 backend。s05 是 toy,只跑 Chroma,但接口形状留好了。

### 3.6 实际跑出来的索引形状

把 unit 01 跑在仓库自带的 `samples/` 上,磁盘上长这样:

```
s05_vector_index/_chroma/
├── chroma.sqlite3                              # SQLite: 文档原文 + metadata + collection 元数据
└── <uuid>/                                     # HNSW 索引目录:hnswlib 二进制文件
    ├── data_level0.bin
    ├── header.bin
    ├── link_lists.bin
    └── length.bin
```

内存中的 collection 形状(用 `peek()` 看到的):

```python
collection.peek()["metadatas"][:2]
# [
#   {"source": "server_whitepaper.pdf", "page": "1"},   # 注意 page 是 str 不是 int
#   {"source": "disclosure.docx",       "page": ""},    # DOCX 没有 page,空串
# ]

collection.peek()["embeddings"][:1][0][:5]
# [-0.028, 0.041, -0.013, -0.057, 0.009]                # 512 维,前 5 个分量
```

下游 s06+ 拿到 `search()` 返回的 `list[{text, source, page, score}]` 时,**不需要知道背后是 Chroma / Milvus / FAISS** —— 它只关心四个字段。这是 s05 把"向量库选型"封装掉的价值:**后续章节按统一接口消费即可**,换底层只改 `build_index` / `search` 两个函数。

### 3.7 跑出来是什么样

Unit 01 的预期输出(具体数字由 `samples/` 决定):

```
indexed 34 chunks into _chroma/ (collection=docs, dim=512)
```

34 是 4 页白皮书 + 27 段披露报告 → 34 个 chunk;512 是 BGE-small-zh 的输出维度。**首次跑会从 HF Hub 下载 ~100MB 的 BGE 模型**,后续跑复用本地缓存,只算 embedding。

Unit 02 的预期输出(输入 `应收账款`):

```
top-3 hits (query='应收账款'):
  [disclosure.docx#None] score=0.950 | 1. 应收账款
  [disclosure.docx#None] score=0.926 | 3. 应收账款
  [disclosure.docx#None] score=0.850 | 24. 应收账款
```

分数严格递减,3 条都来自 `disclosure.docx`(因为 "应收账款" 真的只出现在 DOCX 里);`page=None` 是 docx 加载器没编页号的契约(`s02_doc_loading/code.py:29` 故意设的),不是 bug。**分数 ~0.95 而不是 ~1.00 的根因是 cosine 距离被翻成相似度后还有语义噪声** —— "1. 应收账款" 和 "24. 应收账款" 标题下的内容不同,BGE 编码后向量微差。

**Troubleshooting**:

- `ModuleNotFoundError: No module named 'chromadb'`:`pip install chromadb`(同时拉 `onnxruntime` / `hnswlib` / `pysqlite3-binary` 等传递依赖);Windows 下 `pysqlite3-binary` 是关键依赖,不带它 chromadb 加载 SQLite 会崩。
- `Could not import google.protobuf` / `onnxruntime` 加载报错:首次跑 BGE 时 chromadb 内部 `all-MiniLM-L6-v2` 默认 embedding 函数会触发 protobuf 解析,可以 `pip install protobuf` 兜底;**我们用 `col.add(embeddings=...)` 传预算好的向量绕过**。
- `Dimension mismatch` / `[InvalidArgumentError]`:`vectors[0]` 维度跟 collection 期望不一致 —— 多半是中途换了 `EMBED_MODEL`(512 → 1024),旧 collection 留着但新向量塞不进去。**删 `_chroma/` + 重跑 unit 01**。
- 第一次跑 unit 02 报 "未发现持久化索引":先跑 unit 01 让 `_chroma/` 有数据。
- Windows GBK 控制台打印中文报 `UnicodeEncodeError`:控制台编码问题不是代码 bug,跑前 `set PYTHONIOENCODING=utf-8`。
- `_chroma/` 占了 1-2 GB 但 git 不收:`.gitignore` 里有 `**/_chroma/`,是预期行为;**清缓存就 `rm -rf s05_vector_index/_chroma/`**。

---

## 四、对照 RAGFlow + 思考题

### 4.1 ragflow 怎么做的

RAGFlow 在这一层把全文召回 (BM25) 和向量召回 (dense) **用同一个 DB query 一起发**,在 ES / Infinity 侧用 `FusionExpr("weighted_sum", topk, {"weights": "0.05,0.95"})` 融合,再按 API 入参 `vector_similarity_weight` 互补地算全文权重(`rag/utils/es_conn.py:582-637`);最后 `rerank_with_knn` 用 `tkweight=0.3 / vtweight=0.7` 再做一次线性加权叠 `rank_feature`(`rag/utils/es_conn.py:201-230`)。**本章就是 RAGFlow 链路里"dense-only + DB 内部 fusion 前"那个最朴素的切片** —— 只走了 `MatchDenseExpr` 这一路,没 BM25 兜底也没 PageRank / tag boost。粗召回偏向量 (`0.05, 0.95`),精排也偏向量 (`0.3, 0.7`),但全文信号从不缺席。完整摘录见 [`ragflow_notes/vector_indexing.md`](../ragflow_notes/vector_indexing.md) 和 [`ragflow_notes/hybrid_retrieval.md`](../ragflow_notes/hybrid_retrieval.md)。

一句话对比:RAGFlow 把向量库做成"BM25 + 向量二合一 + 分片副本 + 多租户命名"的统一封装,牺牲单点便利,换多租户分片、混合检索、可观测聚合;ES 的 `create_idx` 模板(`common/doc_store/es_conn_base.py:128-141`)用一份 `mapping.json` 统一管 `settings + mappings`(分片数、副本数、`dense_vector` 字段类型、HNSW 构造参数),改一个文件全集群生效。Chroma 在 RAGFlow 眼里只是"开发态玩具",`persistent` 是单文件 SQLite + 本地 `hnswlib`,做不到多租户物理隔离 + 副本 + 分片 —— 生产场景一上百个团队 / 几亿文档就崩。

### 4.2 主流向量库速览

下面这张表把社区常用的几类向量方案按"是否纯本地 / 是否支持 BM25 / 部署形态 / ANN 算法"列出来,方便选型时快速对照:

| 库 | 本地 / 服务 | 是否支持 BM25 | ANN 算法 | 多租户隔离 | 适用场景 |
|---|---|---|---|---|---|
| **ChromaDB**(本教程 demo) | 本地 | 否(只 `where`) | HNSW (`hnswlib`) | 单 collection | 原型 / 教学 / 几百万级单租户 |
| **FAISS** (`faiss-cpu`) | 本地(算法库) | 否 | IVF / HNSW / PQ | 无(纯文件) | 离线实验 / 评测 / 不带元数据 |
| **Milvus** | 服务 / 集群 | 间接(`BM25` 需外置) | HNSW / IVF / PQ + GPU | 命名 + 权限库 | 十万到百亿级、生产 |
| **Qdrant** | 服务 / 集群 | 是(原生 sparse + dense) | HNSW(Rust 自研) | collection-level | 中大规模、Rust 性能 |
| **Weaviate** | 服务 / 集群 | 是(原生 hybrid) | HNSW | 多 tenant 字段 | GraphQL + 多模态 AI 应用 |
| **Pinecone** | 托管 Serverless | 否(只 dense) | 自研(基于 SPANN) | namespace 隔离 | 不想运维、要 SLA 保证 |
| **Elasticsearch / OpenSearch** | 服务 / 集群 | 是(原生 BM25 + knn) | HNSW | index / alias | RAGFlow 选型之一,已有 ES 栈 |
| **Infinity** | 服务 / 集群 | 是(原生 fulltext + vector) | HNSW / IVF | database / table | RAGFlow 选型之二,亿级 + 多列过滤 |

我们的 toy Chroma 在"本地 / 服务 + BM25"上只占第一行 —— 能跑,但 1) 不原生支持 BM25 全文检索(只是元数据 `where`);2) 没有副本/分片/多租户物理隔离;3) ANN 算法只暴露 HNSW 三参数,不暴露 IVF / PQ。RAGFlow 选 ES / Infinity 而不是 Chroma 的全部原因见 [`ragflow_notes/vector_indexing.md`](../ragflow_notes/vector_indexing.md) 第二节"为什么不用 Chroma"。

### 4.3 选型速记

- **教学 / 快速原型** → Chroma(本教程);`PersistentClient` 几行跑通,看得到每一步;
- **中等规模 / 不想运维 / 接受 dense-only** → Qdrant / Weaviate,单二进制部署,带 hybrid;
- **已有 ES 栈 / 想要 BM25 + dense 一锅炖** → Elasticsearch / OpenSearch 的 `dense_vector` 字段,RAGFlow 走的路;
- **十万到百亿级 / 多租户硬隔离** → Milvus + 分片副本,或 Pinecone(托管);
- **算法研究 / 离线评测 / 不要 metadata** → FAISS,把向量当 numpy 处理;
- **要先看清每个边界再选** → 用本章 `unit 02` 把 query 跑一遍,看清楚"小数过滤 / 分数阈值 / metadata 切片"在 Chroma 上的体感,再决定要不要换。

### 4.4 思考题

**元数据过滤 `where={"source": "server_whitepaper.pdf"}` 怎么写?Chroma 的 `where` 还支持哪些 operator?如果要"只在第 5 页之后找"怎么写?**

参考答案见 [`thinking_answers.md`](./thinking_answers.md)。
