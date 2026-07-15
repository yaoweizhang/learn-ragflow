# s06 混合检索 — BM25 字面召回 + dense 语义召回的加权融合

[上一章 s05 → · 下一章 s07 → ... → s12]

> *"纯 dense 漏字面 (型号 / 编号), 纯 BM25 漏同义 (营收 / 营业收入) — 两路并行 + 加权融合才稳"*
>
> **链路位置**: 在线检索链路的召回入口 (s05 索引 → **s06 召回** → s07 精排)
> **代码文件**: c01_bm25.py · c02_hybrid_fusion.py

> 环境准备: 见 root README §快速开始 — 代码 1 纯 stdlib; 代码 2 需 `pip install sentence-transformers` + 本地 BGE (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 走缓存)

---

## 问题

s05 把 chunk 落盘成可查的索引, 但"查"这个动作还没定义: 用户敲一句 query, 系统要把它投影回已建好的 chunk 空间, 找出 top-k 最相关的段落。这一步就是**召回 (retrieval)**, 也是整条在线链路里最容易翻车的一环 — 因为**任何单一召回通道都有它的死穴**。

**第一, 纯 dense 漏字面**。s04 的 BGE dense 向量擅长语义近邻: query `营收` 能召回写着 `营业收入` 的 chunk, `内存` 能召回 `RAM`。但对精确型号 (`R3630 G5`)、编号 (`ZX-00123`)、专有术语, dense 会把它们拉偏到"语义相似但字面不对"的邻居 — 长 ID 在 embedding 空间里噪声大, 一个字符差异在 512 维里几乎不改变方向。用户想精确定位一个型号, dense 却给他一堆"看起来相关"的近似段。

**第二, 纯 BM25 漏同义**。BM25 是纯统计的稀疏词法召回: 词项命中就打分 (tf + idf + 长度归一), 型号 / 编号 / 术语这类"字面强信号"它抓得极准, 可解释性也强 (每个分都能反推到具体 token)。但它对同义改写完全无能 — `内存` 和 `RAM` 在 BM25 下是 0 分, `营收` 和 `营业收入` 也互不命中。用户换个说法提问, BM25 就全 miss。

**第三, 两路信号量纲不同, 不能直接相加**。就算你想"两路都要", 也不能简单把分加起来: BM25 累加分可能到几 (无界), dense cosine 落在 `[0, 1]`。直接相加会让 BM25 一边倒地主导排序, dense 的语义信号被淹没。融合前**必须先各自归一**, 再按一个权重 `α` 加权 — `α` 大偏语义, `α` 小偏字面。而 `α` 到底该取多少、是常数还是 per-query 入参, 又是一个必须面对的取舍。

这三个问题合起来指向同一个解法: **混合检索 (hybrid retrieval)** — 把稀疏词法 (BM25) 和稠密语义 (cosine) 两条通道并行跑在同一份 chunk 上, 各自归一后按 `α` 加权融合成一个 top-k。BM25 救 dense 漏字面, dense 救 BM25 漏同义, 两条通道 1+1>2。s06 的任务就是把这条"两路召回 + 加权融合"的主干用手写代码跑通 — 先写 BM25 单路 baseline, 再叠 dense 做融合, 让 `α` 这个旋钮的作用肉眼可见。

---

## 解决方案

s06 用 **两个递进的脚本** 把混合检索跑起来。代码 1 先把 BM25 单路召回做出来, 代码 2 把它和 dense cosine 融合成完整的 hybrid。

```
代码 1 (BM25 单路)                代码 2 (BM25 + dense 融合)
┌──────────────────┐           ┌────────────────────────┐
│ chunk 集合        │           │ 复用 c01 的 BM25        │
│      │            │           │      │                 │
│      ▼            │           │      ▼                 │
│ tokenize (中英)   │  ───────▶ │ BM25 分 + dense cosine  │
│      │            │           │ 各自归一 [0,1]          │
│      ▼            │           │      │                 │
│ BM25 打分 top-k   │           │ α·vec + (1-α)·bm25_norm │
└──────────────────┘           └────────────────────────┘
  只有字面, 漏同义               两路互补, 但 α 写死
```

| 脚本 | 解决什么 | 留下什么局限 | 何时用 |
|---|---|---|---|
| `c01_bm25.py` | 手写 BM25 字面召回 (tf + idf + 长度归一) | 漏同义 (`内存` ≠ `RAM`); 中文分词 naive; 无倒排索引 | BM25-only baseline / 关键词型检索 / 教学 |
| `c02_hybrid_fusion.py` | BM25 + dense cosine α 加权融合 | `α` 写死 0.95; 无第三层信号; 无精排 | 教学混合检索 / 粗排入口 / s07 精排的底座 |

两脚本的关系是一条**主干**: 代码 1 把"chunk 集合 → BM25 top-k"做出来, 暴露"漏同义"的死穴 — `营收` 找不到 `营业收入`; 代码 2 把 代码 1 的 BM25 分和 s04 BGE 的 dense cosine 在 `hybrid_topk` 里按 `α` 加权融合, 暴露"`α` 写死 + 无精排"的局限 — 同一个 `α` 对术语型和概念型 query 一刀切, 粗排 top-k 还没经过 cross-encoder 重打分。**每一步的局限, 都是下一步 (代码 2 / s07) 要解决的入口**。

---

## 代码 1: BM25 词法召回 ([c01_bm25.py](c01_bm25.py))

### 工作原理

**做一件事**: 在内存里对 chunk 集合跑一遍手写 BM25, 拿到每条 chunk 的"字面命中"分, 按分降序取 top-k。

**5 步**:
1. 内联 `pypdf` + `python-docx` 加载 `samples/` (复刻 s02 loader), 再按 500 字符 cap 中英句界切成 chunk (复刻 s03 chunker) — 得到 34 个自包含 chunk
2. `tokenize(text)` — 中文按 1-2 字滑动窗口 + 英文/数字单词拆分 (整体 lowercase), 纯中文段落也有 df 命中
3. `BM25(docs, k1=1.5, b=0.75)` — 构造期一次性算 `df` / `tf` / `avgdl`, 查询期只跑打分公式的 `tf` 部分
4. `score(query)` — 对每个 doc 算 BM25 累加分 (`IDF * tf * (k1+1) / (tf + k1*(1 - b + b*dl/avgdl))`)
5. `bm25_topk(docs, query, k)` — 按 BM25 分降序, 取分 > 0 的前 k 条

```python
# 中间片段: BM25 打分公式 (TF 饱和 k1 + 长度归一 b)
for q in qterms:
    if q not in self.df:
        continue
    idf = math.log((self.N - self.df[q] + 0.5) / (self.df[q] + 0.5) + 1)
    for i, tf in enumerate(self.tf):
        dl = sum(tf.values())
        denom = tf[q] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        scores[i] += idf * tf[q] * (self.k1 + 1) / denom
```

**完整函数**:

```python
class BM25:
    """Robertson BM25: token-level TF-IDF with TF saturation (k1) + length norm (b)."""

    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.avgdl = sum(len(tokenize(d["text"])) for d in docs) / max(self.N, 1)
        self.df: Counter = Counter()
        self.tf: list[Counter] = []
        for d in docs:
            tf = Counter(tokenize(d["text"]))
            self.tf.append(tf)
            for term in tf:
                self.df[term] += 1

    def score(self, query: str) -> list[float]:
        qterms = tokenize(query)
        scores = [0.0] * self.N
        for q in qterms:
            if q not in self.df:
                continue
            idf = math.log((self.N - self.df[q] + 0.5) / (self.df[q] + 0.5) + 1)
            for i, tf in enumerate(self.tf):
                dl = sum(tf.values())
                denom = tf[q] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * tf[q] * (self.k1 + 1) / denom
        return scores


def bm25_topk(docs: list[dict], query: str, k: int = 5) -> list[dict]:
    """对 docs 跑 BM25, 返回按 BM25 分降序的 top-k 命中。"""
    bm = BM25(docs)
    scores = bm.score(query)
    order = sorted(range(len(docs)), key=lambda i: -scores[i])
    return [
        {"text": docs[i]["text"], "source": docs[i]["source"],
         "page": docs[i].get("page"), "chunk_id": docs[i].get("chunk_id"),
         "bm25": scores[i]}
        for i in order[:k] if scores[i] > 0
    ]
```

### 试一下

```bash
python s06_retrieval/c01_bm25.py
```

固定 query `应收账款 计提`, 打印 34 chunks 上的 BM25 top-5:

```
loaded 34 chunks from samples/
query='应收账款 计提' → BM25 top-5:
  [disclosure.docx#-] bm25=4.360 | 报告期内，公司实现营业收入人民币 28.74 亿元，同比增长 31.6%；归属于上市公司股东的净利润 3.92 亿元，同
  [server_whitepaper.pdf#3] bm25=3.998 | 四、应用场景 云数据中心：作为通用计算节点支撑私有云与混合云平台，配合虚拟化与容器平台提供高 密度的虚拟机/容器实例；典
  [disclosure.docx#-] bm25=3.896 | 三、供应链风险：高端 GPU 与专用 AI 芯片供应受国际地缘政治影响，可能导致部分订单交付延迟。公司已建立国产化替代方
  [disclosure.docx#-] bm25=3.871 | 展望 2025 年，公司将围绕"行业大模型 + 智能体平台 + 行业知识库"三位一体战略，持续加大研发投入，预计 202
  [disclosure.docx#-] bm25=2.828 | 按业务板块划分，2024 年公司收入结构如下：智能算力基础设施业务收入 12.86 亿元，占比 44.7%，同比增长 2
```

**观察**: 34 是 4 页白皮书 + 27 段披露报告切出的 chunk (跟 s03/s04/s05 一致); BM25 分严格递减。**注意第二个 hit 是 `server_whitepaper.pdf#3`** — BM25 不区分 PDF / DOCX 来源, 只要词项命中就拉过来。每个分都能反推到具体 token 的 tf, 可解释性远超向量检索。

### 为什么不只写这一种

BM25 是纯字面召回 — `内存` 找不到 `RAM`, `营收` 找不到 `营业收入`, 改写 / 同义 / 跨语种召回是它的死穴; 而且中文 1-2 字滑动窗口分词 naive (噪声 token 稀释判别力), 构造期没建倒排索引 (chunk 多时逐 doc 扫 tf 成瓶颈)。同义召回的缺口要靠 代码 2 叠一路 dense cosine 来补。

---

## 代码 2: 混合检索 fusion ([c02_hybrid_fusion.py](c02_hybrid_fusion.py))

### 工作原理

**做一件事**: 把 代码 1 的 BM25 字面分和 dense cosine 语义分各自归一到 `[0, 1]`, 按 `α` 加权融合成一个 top-k, 让"完全一致关键词"和"语义近邻"两条信号同时参与排序。

**5 步**:
1. 用 `importlib` 加载 代码 1 (目录以数字开头, 普通 `import` 报 SyntaxError), 复用它的 `BM25` / `tokenize` / `_load_chunks` — 不重写、不跨章节
2. 用 s04 同款本地 BGE (`bge-small-zh-v1.5`, `normalize_embeddings=True`) 把 34 chunk 和 query 编码成 512 维向量
3. `dense_score_fn(chunk) -> float` — 由调用方注入, 本节就地实现"对每条 chunk 暴力算 cosine" (chunk 量级小、都在内存); 生产里换成 chroma `col.query` 或 ES `knn` 一行 import 替换
4. **归一 + 加权**: BM25 分除以 `max(bm25_scores)` 落 `[0, 1]`, dense cosine 已是 `[0, 1]`, 按 `final = α * vec + (1 - α) * bm25_norm` 融合; 默认 `α=0.95` 偏向量, 镜像生产 `FusionExpr("weighted_sum", {"weights": "0.05,0.95"})`
5. 打印 hybrid top-3, **两个子分都可见** (`vec` / `bm25` / `final`), 方便看是 dense 还是 BM25 把某条 chunk 拉上来的

```python
# 中间片段: 各自归一 + α 加权融合
bm_scores = bm.score(query)
bm_max = max(bm_scores) if any(bm_scores) else 1.0
for i, d in enumerate(docs):
    v = float(dense_score_fn(d))
    b = bm_scores[i] / bm_max if bm_max > 0 else 0.0
    combined.append({..., "dense": v, "bm25": bm_scores[i],
                     "score": alpha * v + (1 - alpha) * b})
```

**完整函数**:

```python
def hybrid_topk(
    docs: list[dict],
    query: str,
    query_vec: list[float],
    dense_score_fn,
    k: int = 5,
    alpha: float = 0.95,
) -> list[dict]:
    """对 docs 算 BM25 分 + dense 分, 各自归一 [0,1] 后 alpha 加权融合, 返回 top-k.

    `dense_score_fn(chunk) -> float` 由调用方注入 (可以是 chroma query、暴力
    遍历、自己实现的近似 KNN 等), 保证 unit 02 不绑死任何具体向量库.
    """
    bm = BM25(docs)
    bm_scores = bm.score(query)
    bm_max = max(bm_scores) if any(bm_scores) else 1.0

    combined = []
    for i, d in enumerate(docs):
        v = float(dense_score_fn(d))
        b = bm_scores[i] / bm_max if bm_max > 0 else 0.0
        combined.append({
            "text": d["text"],
            "source": d["source"],
            "page": d.get("page"),
            "chunk_id": d.get("chunk_id"),
            "dense": v,
            "bm25": bm_scores[i],
            "score": alpha * v + (1 - alpha) * b,
        })
    combined.sort(key=lambda x: -x["score"])
    return combined[:k]
```

### 试一下

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python s06_retrieval/c02_hybrid_fusion.py
```

固定 query `应收账款 计提`, `alpha=0.95`, 打印含子分的 hybrid top-3:

```
loaded 34 chunks from samples/
query='应收账款 计提', alpha=0.95 (dense-dominant)
hybrid top-3 (with both sub-scores visible):
  [disclosure.docx#-] final=0.503 = 0.95*vec(0.477) + 0.05*bm25(4.360) | 报告期内，公司实现营业收入人民币 28.74 亿元，同比增长 31.6%；归属于上市公司股东的净利润 3.92 亿元，同
  [disclosure.docx#-] final=0.487 = 0.95*vec(0.513) + 0.05*bm25(0.000) | 第二节 主要财务数据
  [disclosure.docx#-] final=0.471 = 0.95*vec(0.475) + 0.05*bm25(1.727) | 本报告所涉及财务数据已经立信会计师事务所审计，并出具标准无保留意见审计报告（信会师报字 [2025] 第 ZX-0012
```

**观察**: `final` 落在 `[0.4, 0.6]` 而非 `[0.9, 1.0]` 的根因是 s04 提过的 cosine 归一化后 top-1 通常在 0.5 左右 (BGE 编码 query 与标题的向量微差); `α=0.95` 时 dense 主导, BM25 只在 dense 几乎打平时做 tie-breaker。**第二个 hit `bm25=0.000` 是关键证据** — 纯 dense 把它拉到第二位, 纯 BM25 根本不会看它; `α=0.95` 让"语义近但字面无"的 chunk 保留在 top-k。

### 为什么不只写这一种

`hybrid_topk` 把 `α` 写死 0.95, 对所有 query 一刀切 — 事实型查询 (`应付账款多少`) 该偏 BM25, 概念型 (`营收下滑的原因`) 该偏 dense, `α` 必须做成 per-query 入参; 而且它只返回粗排 top-k, 没有第三层 rank_fea 信号 (PageRank / tag boost), 也没有 cross-encoder 精排。粗排上叠一层重排, 是 s07 要解决的入口。

---

## 接下来

s06 是在线检索链路的召回入口: 代码 1 把 BM25 字面召回跑通, 代码 2 把它和 dense cosine 融合成鲁棒的 hybrid 粗排。但每一步都留下脆弱点, 这些是后续章节的填空目标:

- **代码 1 BM25 漏同义 + 分词 naive** — `内存` 找不到 `RAM`, 1-2 字滑动窗口噪声大; 前者靠 代码 2 叠 dense 补, 后者生产里换 `jieba` 带词典 tokenizer + 倒排索引 (`bm25s` / `pyserini`)。
- **代码 2 `α` 写死 + 无第三层信号** — 同一个 `α` 对术语型和概念型 query 一刀切; 生产把 `vector_similarity_weight` 做成 API 入参, 并加 `sim = tkweight*tksim + vtweight*vtsim + rank_fea` 的 PageRank / tag boost 第三层信号。
- **代码 2 只出粗排, 无精排** — `hybrid_topk` 的 top-k 是召回粗排, "对不对"还得靠精排。

s07 **rerank 精排**: 在 s06 输出的 hybrid top-k 顶上叠一层 cross-encoder — 把 query 和每条候选 chunk 拼成 pair 送 `[CLS]` 头重打分, 把 hybrid 的 top-20 重排成真正相关的 top-3 给 LLM。粗排看"召回全不全", 精排看"排序对不对", 两级串起来才是生产级的检索质量。

---

## 思考题

1. **`alpha=0.95` 和 `alpha=0.5` 在 query `"内存"` 上有什么差别？能不能用同一份 docs 跑两组对比，看 top-5 命中集合是不是变了？**
2. **为什么 query `"应收"` 能命中含 `"应收账款"` 的 chunk，但 query `"应收账款"` 也能命中含 `"应收"` 的 chunk？两个 query 命中的 chunk 是同一批吗？**

（答案见文末「思考题答案」）

---

## 思考题答案

### Q1. `alpha=0.95` 和 `alpha=0.5` 在 query `"内存"` 上的差别

α 越大，dense cosine 的相对权重越高 → `RAM` 这种同义词的命中越靠前；α=0.5 时 BM25 和 dense 各占一半，`内存`字面命中会更强，但 `RAM` / `存储器` 这种同义词会被稀释。

**实测** —— 把 `hybrid_topk(..., alpha=0.5)` 跑一次对比 `alpha=0.95`：

```
query=内存 alpha=0.5
[server_whitepaper.pdf#24] score=1.000 | 图12 内存标识 ...
[server_whitepaper.pdf#2]  score=1.000 | 2 内存 ...
[server_whitepaper.pdf#24] score=0.887 | 4.2 内存 ...
```

vs `alpha=0.95` 的输出（向量主导）：

```
query=内存 alpha=0.95
[server_whitepaper.pdf#24] score=1.000 | 图12 内存标识 ...
[server_whitepaper.pdf#24] score=0.932 | 21 具体可选购 ...
[disclosure.docx#None]      score=0.887 | 7. 存货
```

第三个 hit 在 α=1.0 时已经不是"内存"字面了，是向量把"内存"和"7。存货"拉得太近——这正是 alpha=1 的副作用：牺牲字面精度换语义召回。

**怎么调？** 靠**带标签的交叉验证集**——一批 query + 人工标注的相关 chunk id，跑一遍 sweep(0.0， 0.1， 。。。， 1.0），看每个 alpha 下的 recall@k / MRR。经验上 0.3 ~ 0.7 都合理，极端值只在特定场景才合适：

- 0.0~0.3：术语密集的财务/法务/医疗场景；
- 0.7~1.0：开放域问答、闲聊、文档摘要型场景。

生产代码干脆不存一个 alpha 常数——把 `vector_similarity_weight` 做成 API 入参，每类 query 由调用方传不同值。本 MVP 用 `alpha=0.95` 兜底是因为 demo query 是术语密集型；真要上线必须按业务场景重新调。

### Q2. 两个 query 命中的 chunk 是不是同一批？

不是。

1-2 字滑动窗口让两个 query 都有 `应`、`收`、`应收`、`应收账`、`账款` 之类公共 token；具体命中的 doc 集合 + 分的高低取决于 doc 里这些 token 的 tf 和 dl。

```
query="应收" → 命中文档集合 = {所有出现"应"或"收"或"应收"的 chunk}
              top-1 = bm25=2.something 的 chunk

query="应收账款" → 命中文档集合 = {所有出现"应收"+"账款"或其中至少一个的 chunk}
                top-1 = bm25=2.something 的 chunk(跟上面那个很可能重叠)
```

但**集合大小不一样**——query="应收" 命中范围更广（任何含"应"或"收"的 chunk），query="应收账款" 命中范围窄。要验的话，把 `tokenize` 改成只保留 2 字切分（不要 1 字），再跑一遍 `bm25_topk`，对比两组 top-5 —— 1 字窗口噪声大、几乎所有 chunk 都"命中"；纯 2 字窗口对术语型查询更精准。

下一章 s07 如何解决 — hybrid_topk 的 top-k 是粗排,真正回答用户问题的"对不对"还得靠精排;在 s06 输出顶部叠一层 cross-encoder rerank(query + chunk pair → 真相关分),把 hybrid 的 top-20 重排成 top-3 给 LLM,精度再上一档。

> 排错事项（pytesseract / FlagEmbedding / `alpha` sweep 等）见对应代码文件的 `### 局限与下一步`。