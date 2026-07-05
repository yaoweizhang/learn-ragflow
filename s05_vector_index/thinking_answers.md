# 思考题答案

## 问题: 元数据过滤 `source = server_whitepaper.pdf` 怎么写?

**答: 在 `col.query(...)` 里加一个 `where={"source": "server_whitepaper.pdf"}`
参数,Chroma 会把它转成"先按 source 过滤,再在过滤后的子集上跑向量
近邻",SQL 语义等价于 `SELECT ... WHERE source = ? ORDER BY cosine
LIMIT k`。**

### 1. 最小写法

```python
hits = col.query(
    query_embeddings=[query_vec],
    n_results=3,
    where={"source": "server_whitepaper.pdf"},
)
```

就这么一行。Chroma 0.5.x 的 `where` 支持 `{"字段": "值"}` 这种最简形
(隐式 `$eq`),也会翻译成 `$eq` 内部表示。

### 2. 想要"只搜 PDF,排除扫描件"怎么写?

`where` 的"值"也可以是 dict,显式带 operator:

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

支持的 operator 有 `$eq / $ne / $gt / $gte / $lt / $lte / $in / $nin` 数值
比较,以及 `$and / $or` 复合;但**不支持 BM25 全文匹配**(那是
Elasticsearch 的事,见 `../ragflow_notes/vector_indexing.md` 第二节
"为什么不用 Chroma")。

### 3. 元数据 schema 约束

加进去的 `metadatas=[{"source": ..., "page": ...}]` 字段在 Chroma 里
是**强类型 + 全 string** —— 我们 `code.py` 第 33 行把 `page` 转成
`str(c.get("page", ""))` 后再塞,因为 Chroma 1.5 拒绝 int+string 混
合,0.5.x 拒绝 None(直接抛 `Cannot convert None to MetadataValue`)。
如果忘了这步,`col.add` 会当场 segfault(在 Windows + chroma-hnswlib
0.7.6 上是真的 native crash,见 commit message 备注)。

### 4. 一句话总结
**`where={"source": "..."}` 就够了 —— 把"按业务维度切"和"按相似度排"
两步合并成一次原子操作,是 Chroma 比手写 FAISS + 自己维护 dict 的唯
一胜场**;真要做权限 / 时间段 / 多标签过滤,Chroma 就得换成 ES 或
Infinity(见 README.md 第二节"真实世界的问题")。
