# s05 向量索引 — 把向量 + 元数据写进可查询的库

[上一章 s04 → · 下一章 s06 → ... → s12]

> *"s04 把 chunk 变成了向量，但那些向量还飘在内存里 —— 进程一关就没，几十万条要重算几分钟，更糟的是根本没法在线查。s05 用 Chroma 把它们落盘 + 建 ANN 索引，'持久化 + 毫秒级 top-k' 一次跑通"*
>
> **链路位置**: 离线索引链路最后一步 (s02 → s03 → s04 → **s05**)，在线由 s06+ 消费
> **代码文件**: chroma_build.py · chroma_query.py

> 环境准备: 见 root README §快速开始 — `pip install chromadb sentence-transformers`；首次跑会下载 ~100MB 的 BGE 模型，之后复用本地缓存

---

## 问题

`chromadb.PersistentClient(...).create_collection(...)` 几行就能跑，看起来不值得单独一章。但把它接进真实 RAG 链路就会发现，"跑通"和"在 prod 不爆"之间隔着一道悬崖 —— 而在此之前，得先想清楚：s04 已经把 chunk 编码成 512 维向量了，为什么还需要一个"库"来装它们？

**因为向量飘在内存里等于没存。** 如果只把 s04 跑出来的向量扔进 Python `list`、每次启进程重算一次，几万条 chunk 还能忍，几十万条就要算几分钟；更致命的是 —— 你**根本没法在线查**。一次用户 query 进来，难道要把全部向量重新 embed 一遍、再逐条算余弦？s05 要解决的第一件事就是**持久化**（进程关掉不丢）和 **ANN 查询**（百万级向量毫秒级 top-k），把"飘在内存里的向量"变成"可持久化、可检索的数据结构"。

而向量索引本身不是一个黑盒，拆开看是三个独立部件：**ANN 索引 (HNSW)** —— 一种基于图的近似最近邻算法，多层邻近图让搜索从 O(N) 降到 O(log N)，代价是 *近似* 而非 *精确*；**元数据存储 (SQLite)** —— 每条向量附带几列标量字段 (`source` / `page`)，查询时可以 `where={"source": "x.pdf"}` 先过滤再近邻；**持久化层 (PersistentClient)** —— 把 ANN 索引 + SQLite + 文档原文一起写到本地目录，下次 `PersistentClient(path=...)` 直接打开，不丢数据、不重 embed。

但真正接进链路后，几类典型问题会一个个冒出来：

**第一，维度对不上会爆索引**。同一 collection 混用 512 维 BGE 和 1536 维 OpenAI，Chroma 在 insert 阶段直接 `Dimension mismatch` 报错。s05 这一层只能用"一 collection 一维度 + 重建"的笨办法保证不混 —— `vectors[0]` 硬编码 512 是 BGE-small-zh 的维度，换模型 (`bge-large-zh-v1.5` 是 1024 维) 整张 collection 必须删掉重建，**没有 schema 迁移**。

**第二，元数据过滤做不深**。Chroma 的 `where` 只支持 `$eq / $ne / $in / $nin / $gt / $gte / $lt / $lte` 数值比较 + `$and / $or` 复合，**做不了全文 BM25、做不了 `rank_feature` 加权、做不了聚合桶**。业务侧要做"按部门 / 时间段 / 标签"切片时，`where` 撑不住。

**第三，top-k 召回 vs 精度的取舍**。ANN 是 *近似* 算法 —— `n_results=k` 返回的 k 条不一定是全局最像的，只是大概率最像；更糟的是 **top-k 强制返回 k 条，哪怕最差一条 score 接近 0 也照样返回**，应该加一个分数下限把噪音砍掉，但 Chroma 没有原生 `where` 分数过滤，得在后处理里加。

这三条每一条都对应不同的工业级解法 —— schema 注册、混合检索、分数阈值，都是 s06+ 的主题。**s05 的目标不是解决它们，而是把它们显式暴露出来，让你看到 toy 方案的边界**。这也是为什么不直接用 FAISS / Milvus / Pinecone —— FAISS 没有 `where` 元数据过滤，Pinecone 是托管服务要钱，Milvus 是分布式集群单机跑不起来。Chroma 是"最小可用 toy"，**看得到每一步**：先见边界，再看生产，比直接用封装库学到的多。

---

## 解决方案

s05 用 **两个递进的脚本** 把"持久化 + ANN 检索"跑起来：一个负责写 (`build`)，一个负责读 (`query`)，两者通过 `_chroma/` 目录解耦。每一步解决前一步的局限，但也留下新的脆弱性。

```
代码 1 (build)                     代码 2 (query)
┌──────────────────┐            ┌──────────────────────┐
│ samples/         │            │ 重新打开 _chroma/    │
│  → load → chunk  │            │       │              │
│  → BGE embed     │            │       ▼              │
│       │          │  ───────▶  │ query 同空间 embed   │
│       ▼          │  _chroma/  │       │              │
│ PersistentClient │            │       ▼              │
│ + cosine + add   │            │ col.query top-k      │
│       │          │            │ 1-distance → score   │
│       ▼          │            │       │              │
│ 落盘 34 chunks   │            │       ▼              │
└──────────────────┘            │ {text,source,page,   │
   持久化 (写)                   │  score}              │
                                └──────────────────────┘
                                   ANN 检索 (读)
```

| 脚本 | 解决什么 | 留下什么局限 | 何时用 |
|---|---|---|---|
| `chroma_build.py` | 512 维向量 + 文本 + `source`/`page` → 落盘 Chroma collection (cosine) | 删旧重建 (无增量); 单机无 HA; 维度硬编码; metadata 只两列 | toy / 教学 / 单租户几百万级 |
| `chroma_query.py` | 重开 collection + 同空间 embed + cosine top-k + `1-distance` 翻分 | dense-only (无 BM25); 无 score 阈值; 单进程锁; 无分页 | dense-only 召回前置切片 |

两脚本的关系是一条**离线→在线主干**: 代码 1 把"samples → load → chunk → embed → 落盘"做出来，暴露"重建成本 + 单机瓶颈 + 维度硬编码"的局限；代码 2 把"重开 collection → query 同空间 embed → cosine top-k → 翻分"做出来，暴露"dense-only + 无阈值"的局限 —— 事实型 query 容易输给"近义但错"的 chunk。生产里也走这个模式：离线 pipeline 跑完 `chroma_build.py` 就停，在线 service 只跑 `chroma_query.py` (或更上层的 s06+)。**每一章的局限，都是下一章要解决的入口**。

---

## 代码 1: Chroma 持久化索引构建 ([chroma_build.py](chroma_build.py))

### 工作原理

**做一件事**: 把 chunk 列表 + 向量列表 + metadata 一起写进 Chroma 的 `PersistentClient`，落盘成可重复查询的 collection。

**5 步**:
1. `_pdf` / `_docx` 复刻 s02 加载器读 `samples/` (`server_whitepaper.pdf` + `disclosure.docx`)，走同一份 `{text, page, source}` schema
2. `_chunk_by_paragraph` 复刻 s03：短段整段保留，长段按句界切 (500 字符 cap)，每 chunk 带 `chunk_id`
3. `_embed` 复刻 s04：本地 BGE (`BAAI/bge-small-zh-v1.5`, `normalize_embeddings=True`)，把 chunk 文本编码成 512 维单位向量
4. `build_index` 先 `shutil.rmtree(DB_DIR)` 确保干净 (重建而非增量)，再建 collection `docs`，`metadata={"hnsw:space": "cosine"}` 指定 HNSW 用 cosine 距离
5. `col.add(ids=, embeddings=, documents=, metadatas=)` 一次性写入；metadata 只两列 `source` / `page`，`page` 转 `str` 存 (DOCX 留空串) —— 这就是后续 `where` 能过滤的字段

```python
# 中间片段: rmtree 重建 + cosine collection + 一次性 add
if DB_DIR.exists():
    shutil.rmtree(DB_DIR)                       # 重建而非增量，保证幂等
client = chromadb.PersistentClient(path=str(DB_DIR))
col = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
col.add(
    ids=[c["chunk_id"] for c in chunks],
    embeddings=vectors,
    documents=[c["text"] for c in chunks],
    metadatas=[{"source": c["source"], "page": str(c.get("page", ""))} for c in chunks],
)
```

**完整函数**:

```python
def build_index(chunks: list[dict], vectors: list[list[float]]):
    """删旧 _chroma/ → PersistentClient → cosine space → 写入 chunks+vectors+metadata。"""
    import chromadb
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "page": str(c.get("page", ""))} for c in chunks],
    )
    return col


def main() -> None:
    chunks, vectors = get_chunks_and_vectors()
    col = build_index(chunks, vectors)
    print(f"indexed {len(chunks)} chunks into _chroma/ (collection={col.name}, dim={len(vectors[0])})")
```

### 试一下

```bash
python s05_vector_index/chroma_build.py
```

首次跑会从 HF Hub 下载 ~100MB 的 BGE 模型，然后 embed 34 个 chunk (4 页白皮书 + 27 段披露报告 = 34)，落盘到 `_chroma/`：

```
indexed 34 chunks into _chroma/ (collection=docs, dim=512)
```

磁盘上留下 `s05_vector_index/_chroma/chroma.sqlite3` (文档原文 + metadata) + 一个 `<uuid>/` 目录 (HNSW 索引的 `data_level0.bin` / `header.bin` / `link_lists.bin` / `length.bin` 二进制)。

**观察**: 再次跑会先 `rmtree` 整个目录再重建 —— **重建而非合并**是当前最简单的正确性取舍，跑两次结果完全一样，不会塞重复 `chunk_id`。metadata 里 `page` 是 `str` 不是 `int` (`"1"` / `""`)，因为 Chroma 0.5.x 拒绝 int+string 混合、拒绝 `None`；类型抖动被吸收在 s05 层，下游 s06+ 拿到的 `page` 又会在 `chroma_query.py` 里翻回 `int / None`。

### 为什么不只写这一种

``chroma_build.py`` 只把向量落盘、能查，但**重建 = rm 整棵目录树** (几十万 chunk 时每次重算 HNSW 图，分钟级成本)、**没有远程 / HA** (单点故障，不能水平扩展)、**维度硬编码 512** (换模型整张 collection 要重建，无 schema 迁移)、**metadata 只两列** (想按部门/时间段切片得加列)。而且落盘只是半条链路 —— 怎么在这份 `_chroma/` 上查 query，是**代码 2** 的事。

---

## 代码 2: Chroma 检索 — cosine distance → similarity top-k ([chroma_query.py](chroma_query.py))

### 工作原理

**做一件事**: 把代码 1 持久化的 collection 重新打开，对 query 文本做同空间 BGE embed，用 cosine 距离取 top-k，把 `1 - distance` 翻成 `[0, 1]` 区间的 `score`。

**4 步**:
1. `_open_collection()` — `PersistentClient(path=DB_DIR).get_collection("docs")`；`_chroma/` 不存在或 collection 没建出来就返 `None`，让 `main()` 提示"先跑 chroma_build.py"
2. `_embed([query])` — 内联本地 BGE (跟 `chroma_build.py` 同款)，保证 query 向量落在跟索引**同款空间** (否则余弦无意义)
3. `search(col, query_vec, k)` — `col.query(query_embeddings=[...], n_results=k)` 拿 `documents / metadatas / distances` 三列，`page` 从字符串还原回 `int` 或 `None`，`1 - cosine_distance` 翻成 `score`
4. 统一返回 `list[{text, source, page, score}]` —— 下游 s06+ 不需要知道背后是 Chroma / Milvus / FAISS，只关心这四个字段

```python
# 中间片段: cosine distance → similarity，page 翻回 int/None
page_val = res["metadatas"][0][i].get("page", "")
try:
    page_val = int(page_val) if page_val != "" else None
except (ValueError, TypeError):
    pass
hits.append({
    "text": doc,
    "source": res["metadatas"][0][i]["source"],
    "page": page_val,
    "score": 1 - res["distances"][0][i],  # cosine distance ∈ [0,2] → similarity ∈ [-1,1]
})
```

**完整函数**:

```python
def _open_collection():
    """打开 chroma_build.py 持久化的 collection;不存在就返回 None 让 main 提示。"""
    import chromadb
    if not DB_DIR.exists():
        return None
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return None


def search(col, query_vec: list[float], k: int = 5) -> list[dict]:
    """对已打开的 collection 跑 query,把 cosine distance 翻成 similarity,统一返回结构。"""
    res = col.query(query_embeddings=[query_vec], n_results=k)
    hits = []
    for i, doc in enumerate(res["documents"][0]):
        page_val = res["metadatas"][0][i].get("page", "")
        try:
            page_val = int(page_val) if page_val != "" else None
        except (ValueError, TypeError):
            pass
        hits.append({
            "text": doc,
            "source": res["metadatas"][0][i]["source"],
            "page": page_val,
            "score": 1 - res["distances"][0][i],
        })
    return hits
```

### 试一下

**前置**: 必须先跑过代码 1，否则 `_chroma/` 不存在。

```bash
python s05_vector_index/chroma_build.py   # 先建索引
python s05_vector_index/chroma_query.py    # 再查
```

交互输入 `应收账款`，看 top-3 hits (按相似度递减)：

```
top-3 hits (query='应收账款'):
  [disclosure.docx#None] score=0.499 | 报告期内，公司实现营业收入人民币 28.74 亿元，同比增长 31.6%；归属于上市公司股东的净利润 3.92 亿元，同
  [disclosure.docx#None] score=0.487 | 第二节 主要财务数据
  [disclosure.docx#None] score=0.449 | 第四节 分季度财务数据
```

**观察**: 分数严格递减，3 条都来自 `disclosure.docx` (因为"应收账款"这类财务概念真的只在 DOCX 里)；`page=None` 是 docx 加载器没编页号的契约 (`s02_doc_loading/basic_load.py` 故意设的)，不是 bug。分数落在 ~0.5 而不是 ~1.0，是因为 cosine 距离翻成相似度后还有语义噪声 —— query 和命中段落"相关但不等价"，BGE 编码后向量有微差。

### 为什么不只写这一种

``chroma_query.py`` 只走**单条 dense 向量通道**，没有 BM25 / 全文召回兜底 —— 事实型 query (精确词、专业术语) 容易输给"近义但错"的 chunk；再加上 **top-k 强制返回 k 条无 score 阈值** (最差一条接近 0 也照返)、**单进程锁不支持并发写**、**没分页**。这条单通道召回就是 s06 混合检索要补的洞。

---

## 接下来

s05 把 s04 算出的 512 维向量绑回 chunk 文本 + `source` / `page` 元数据，落到 Chroma `PersistentClient`，关掉 Python 不丢、重启直接打开继续查 —— 实现了"s05 = 持久化 + ANN 检索"的最小闭环。``chroma_build.py`` 把"写"做出来，``chroma_query.py`` 把"读"做出来，两者通过 `_chroma/` 目录解耦。但这个 toy 方案的每一处边界，都是后续章节的填空入口：

- **重建成本 + 单机瓶颈** — `shutil.rmtree` 整棵目录树重建，几十万 chunk 时分钟级；Chroma 单点故障、不能水平扩展。生产要走 ES 的 `_bulk` 增量 upsert + 专用服务进程读写分离。
- **维度硬编码 + metadata 太薄** — `vectors[0]` 锁死 512，换模型整张 collection 重建；metadata 只 `source` / `page` 两列，做不了"按部门 / 时间段 / 多标签"切片。生产在 ES 上跑 `bool + knn + terms + range + rank_feature` 一锅炖。
- **dense-only + 无 score 阈值** — 单向量通道漏掉"精确词 / 专业术语"query，top-k 强制返回噪音。这是 s06 混合检索 + s07 rerank 要解决的召回质量问题。

s06 **混合检索**: 在 dense 召回旁边开一条 BM25 词法通道，两条结果按 α-weighted 融合 —— 既救回 dense 漏的"精确词 / 专业术语"query，又保留 dense 的语义兜底，single-channel 召回升级成 hybrid，把 `chroma_query.py` 留下的"近义但错"漏召回补上。

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

就这么一行。Chroma 0.5.x 的 `where` 支持 `{"字段": "值"}` 这种最简形（隐式 `$eq`），也会翻译成 `$eq` 内部表示。

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

支持的 operator 有 `$eq / $ne / $gt / $gte / $lt / $lte / $in / $nin` 数值比较，以及 `$and / $or` 复合；但**不支持 BM25 全文匹配**（那是 Elasticsearch 的事，见 [`docs/reference/ragflow-notes/vector_indexing.md`](../docs/reference/ragflow-notes/vector_indexing.md) 第二节"为什么不用 Chroma"）。

#### 3. 元数据 schema 约束

加进去的 `metadatas=[{"source": ..., "page": ...}]` 字段在 Chroma 里是**强类型 + 全 string** —— 我们 `chroma_build.py` 第 33 行把 `page` 转成 `str(c.get("page", ""))` 后再塞，因为 Chroma 0.5.x 拒绝 int+string 混合、拒绝 `None`（直接抛 `Cannot convert None to MetadataValue`）。如果忘了这步，`col.add` 会当场 segfault（在 Windows + chroma-hnswlib 0.7.6 上是真的 native crash，见 commit message 备注）。

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
