# s06 检索 (Retrieval) — BM25 + 向量召回的加权融合

> **本章定位**：s06 是 RAG 在线链路的"召回入口"——把用户 query 投影到 s04/s05 已建好的 embedding 空间，先做 **dense 近邻**，再做 **BM25 字面打分**，最后按 `α` 加权融合成 top-k。详细定位见 s00 §1.4；RAGFlow 实现见本章末"## RAGFlow 实现"。

---

## 一、章节介绍

### 1.1 核心定义：什么是混合检索？

**混合检索 (Hybrid Retrieval)** 是把两条异构的召回通道——**稀疏词法 (BM25)** 和 **embedding 语义 (cosine similarity)**——并行跑在同一份 chunk 集合上，按某种融合策略 (weighted_sum / RRF) 合并成一个 top-k 排序。它要解决的痛点是：**任何一路单独召回都会在某些 query 类型上翻车，两路互补才能稳定**。

```
                    query: "应收账款 计提"
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
        BM25 (稀疏/字面)          dense cosine (稠密/语义)
        tf + idf + 长度归一       BGE → 512-d vec → cosine
        强: 型号 / 编号 / 术语    强: 同义词 / 改写 / 语义近邻
                │                       │
                │ 各自归一 [0, 1]        │
                └───────────┬───────────┘
                            ▼
                  score = α · vec + (1−α) · bm25_norm
                            │
                            ▼
                    top-k hits (送 s07 rerank)
```

把它放进 RAG 全景看：**s06 是把"用户问题"投影回"已索引 chunk 空间"的第一步**。s05 把 chunks 落盘成可查的索引，但查询的入口 (ann 召回 / bm25 召回 / 融合权重） 都是 s06 的事。如果只用 dense，精确型号（`"R3630 G5"`）会被拉偏到语义近邻；如果只用 BM25，改写（`"营收" vs "营业收入"`）永远找不到。**两路并行 + 加权融合 = 鲁棒召回**。

#### 稀疏 vs 稠密：两条通道的本质差异

s06 的代码把所有事都写在一个文件里，但拆开看是两种**性质相反**的检索通道：

| 维度 | 稀疏 / BM25 | 稠密 / dense cosine |
|---|---|---|
| 表示 | 高维 0-1 稀疏向量(词袋) | 低维稠密向量(BGE 512 维) |
| 核心信号 | 词项命中 (tf + idf + 长度归一) | 语义方向(内积 / cosine) |
| 命中同义词 | **不能** (`内存` ≠ `RAM`) | **能** (embedding 把它们拉到近) |
| 命中型号 / 编号 | **强** (字面匹配) | **弱** (长 ID 在 embedding 里噪声大) |
| 训练成本 | 无 (纯统计) | 高 (Transformer + 千万级句对) |
| 可解释性 | **强** (反推到具体 token) | **弱** (512 维每个分量无字面含义) |
| 量化分范围 | 0 ~ 几十 (无界) | 0 ~ 1 (cosine,归一化后等价于内积) |

**关键 takeaway**：两路信号**量纲不同**——BM25 累加分可能到几，cosine ∈ [0，1]。直接相加会偏 BM25 一边。**融合前必须归一**——本教程 BM25 除以 `max(bm_scores)` 落 [0，1]，dense 已经是 cosine ∈ [0，1]。

#### 与传统信息检索的对应

混合检索和传统 IR 的演进关系正好对应 RAG 召回链路的"从单通道到多通道"：

| 检索形态 | 通道数 | 典型融合 | 适用阶段 |
|---|---|---|---|
| BM25-only (Lucene / ES `match`) | 1 | 单分排序 | 全文搜索、关键词型 FAQ |
| dense-only (FAISS / Chroma `query`) | 1 | cosine / 内积 | 语义近邻、KNN 推荐 |
| **weighted_sum fusion** (s06 MVP) | 2 | `α·v + (1-α)·b_norm` | 教学 / 简单混合 |
| RRF (Reciprocal Rank Fusion) | 2 | `Σ 1/(rank_i + c)` | Milvus / ES 多通道融合 |
| 生产 3-layer | 3+ | DB `weighted_sum` + app `rerank_with_knn` + `rank_fea` | 生产 |

s06 的 MVP 是这张表的第三行——**两通道 + 加权融合**，α 是唯一的旋钮。第四章要学的 RRF / 三层融合是它的扩展版本。

### 1.2 真实世界的问题

3 类典型失败（纯 dense 漏字面 / 纯 BM25 漏同义 / α 是常数还是入参）详见 §三 "做错了什么"。

---

## 二、BM25 词法召回 (hand-written BM25 + 中英分词）：[code_01_bm25.py](code_01_bm25.py)

入口：[`code_01_bm25.py`](code_01_bm25.py)

在内存里对 chunk 集合跑一遍 BM25，拿到"字面命中"分。
code_02 会把这里的 BM25 分和 dense cosine 一起做加权融合，进入混合检索。

### 概念

`code.py` 实现一个 Robertson BM25 召回器：拿到 `docs` 后离线算 df/tf/IDF，query 来时按词项打分、按 BM25 分降序取 top-k。

- `tokenize(text)` —— 中文按 1-2 字滑动窗口 + 英文/数字单词拆分（lowercase），纯中文段落也有 df 命中；
- `BM25(docs, k1=1.5, b=0.75)` —— 构造期一次性算 `df` / `tf` / `avgdl`，查询期只跑 score 公式的 `tf` 部分；
- `score(query)` —— 对每个 doc 算 BM25 累加分（`IDF * tf * (k1+1) / (tf + k1*(1 - b + b*dl/avgdl))`）；
- `bm25_topk(docs, query, k)` —— 按 BM25 分排序，取分 > 0 的前 k 条；
- `main()` —— 内联 pypdf + python-docx + 500 字符 cap 句界切，跟 s02 code_01 / s03 同一套"加载器复刻"取舍，跑固定 query `"应收账款 计提"` 打印 BM25 top-5。

### 跑一遍

```bash
python s06_retrieval/code_01_bm25.py
```

输出示例：

```
loaded 34 chunks from samples/
query='应收账款 计提' → BM25 top-5:
  [disclosure.docx#-] bm25=... | ...
```

chunk 数与 s05 code_01 一致（34），但走的不是向量空间，是字面 token 命中。

### 看输出

Code_01 的预期输出（`query='应收账款 计提'`）：

```
loaded 34 chunks from samples/
query='应收账款 计提' → BM25 top-5:
  [disclosure.docx#-] bm25=4.360 | 报告期内,公司实现营业收入人民币 28.74 亿元,同比增长 31.6%;归属于上市公司股东的净利润 3.92 亿元,同
  [server_whitepaper.pdf#3] bm25=3.998 | 四、应用场景 云数据中心:作为通用计算节点支撑私有云与混合云平台...
  [disclosure.docx#-] bm25=3.896 | 三、供应链风险:高端 GPU 与专用 AI 芯片供应受国际地缘政治影响...
  [disclosure.docx#-] bm25=3.871 | 展望 2025 年,公司将围绕"行业大模型 + 智能体平台 + 行业知识库"三位一体战略...
  [disclosure.docx#-] bm25=2.828 | 按业务板块划分,2024 年公司收入结构如下:智能算力基础设施业务收入 12.86 亿元...
```

34 是 4 页白皮书 + 27 段披露报告 → 34 个 chunk（跟 s03 / s04 / s05 一致）；BM25 分严格递减。**注意第二个 hit 是 `server_whitepaper.pdf#3`**——BM25 不区分 PDF / DOCX 来源，只要词项命中就拉过来。

### 局限与下一步

本段做对了什么 — 用 ~100 行手写 BM25 + 中英 1-2 字滑动窗口,把"字面 token 命中 → 分"这条路跑通了,纯 stdlib、无外部 embedding 依赖,k1 / b 两个超参可调,每个命中的分都能反推到具体 token tf,可解释性远超向量检索。


- **没语义匹配**：`"内存"`和 `"RAM"`、`"营收"`和 `"营业收入"`在 BM25 下是 0 分；改写 / 同义 / 跨语种召回是 BM25 的死穴——这是 code_02 必须叠 dense cosine 的根本原因；
- **中文分词太 naive**：1-2 字滑动窗口会产生大量噪声 token(`"应收"`、`"账款"`、`"应"`、`"收"`），df 散在海量 1-字 token 上 → 真正有判别力的 2-字词被稀释；生产应该走 `jieba` 之类带词典的 tokenizer；
- **没字段加权 / 没 BM25F**：所有 token 同权；title / heading / body 的 boost 要手工实现，生产代码把这部分挪到了 `rank_feature` 第三层信号；
- **构造期没有倒排索引**：每次查询还是要扫所有 doc 的 tf 才能算分，chunk 数大时会成瓶颈；生产里 `bm25s` / `rank_bm25` / `pyserini` 都建倒排表，查询 O(|q| * avg_postings) 而不是 O(|q| * N）；
- **没有 query expansion / 同义词扩展**：`"AI"` 查不到 `"人工智能"`；生产里靠 query analyzer 做改写、拼写纠正、term drop。


- 中文 chunk 都 0 分：确认 `tokenize()` 的 1-2 字滑动窗口没被改成空；中文段落的分词质量完全靠它。
- 想验 1 字 vs 2 字切分的差别：把 `tokenize` 改成只保留 2 字切分（不要 1 字），再跑一遍 `bm25_topk`，对比两组 top-5。
- `ModuleNotFoundError`：本节无外部 embedding 依赖；只靠 Python stdlib 跑。如果 BGE 还没装好不会影响本节。

下一章 code_02 如何解决 — 把这段 BM25 跟 s05 的 dense cosine 在 `hybrid_topk` 里按 `α` 加权融合,BM25 救 dense 漏字面,dense 救 BM25 漏同义,两条通道 1+1>2;alpha 走默认 0.95(dense 主导),生产里改成 per-query 调。

---

## 三、混合检索 fusion (BM25 + dense cosine， α-weighted)：[code_02_hybrid_fusion.py](code_02_hybrid_fusion.py)

入口：[`code_02_hybrid_fusion.py`](code_02_hybrid_fusion.py)

把 code_01 的 BM25 字面分和 dense cosine 相似度做加权融合，
让"完全一致关键词"和"语义近邻"两条信号同时参与排序。

### 概念

`code.py` 实现一个可注入的 `hybrid_topk(docs, query, query_vec, dense_score_fn, k, alpha)`：

- `dense_score_fn(chunk) -> float` —— 由调用方注入，本节就地实现"对每条 chunk 暴力算 cosine"(chunks 都在内存里、量级小）。生产里换成 chroma `col.query` 或 ES `knn`；
- `BM25(docs).score(query)` —— 复用 code_01 的 hand-written BM25，拿到每个 chunk 的字面分；
- **归一 + 加权**：BM25 分除以 `max(bm25_scores)` 落到 [0，1]，dense cosine 已经是 [0，1](BGE `normalize_embeddings=True`)，按 `final = alpha * vec + (1 - alpha) * bm25_norm` 融合；默认 `alpha=0.95` 偏向量，mirrors 生产代码 `FusionExpr("weighted_sum", {"weights": "0.05,0.95"})` 的权重大小关系（dense=0.95， keyword=0.05）；
- `main()` —— 内联 pypdf + python-docx + 500 字符 cap 切块 + 本地 BGE + code_01 的 BM25，跑固定 query `"应收账款 计提"`，打印 hybrid top-3，**两个子分都可见**（不只是融合分）。

`hybrid_topk` 是这一章的核心 API：code_02 就是把"怎么把两路召回合在一起"封装成一个独立函数，下一章（s07）会在这上面叠 `rerank` / `rerank_with_knn`。

### 跑一遍

```bash
python s06_retrieval/code_02_hybrid_fusion.py
```

输出示例（实测，alpha=0.95）：

```
loaded 34 chunks from samples/
query='应收账款 计提', alpha=0.95 (dense-dominant)
hybrid top-3 (with both sub-scores visible):
  [disclosure.docx#-] final=0.946 = 0.95*vec(0.957) + 0.05*bm25(1.413) | ...
  [disclosure.docx#-] final=0.910 = 0.95*vec(0.922) + 0.05*bm25(1.310) | ...
  ...
```

每个 hit 同时打出 `vec` / `bm25` / `final`，方便看"是 dense 拉上去的还是 BM25 拉上去的"。

### 看输出

Code_02 的预期输出（`query='应收账款 计提'`，`alpha=0.95`）：

```
loaded 34 chunks from samples/
query='应收账款 计提', alpha=0.95 (dense-dominant)
hybrid top-3 (with both sub-scores visible):
  [disclosure.docx#-] final=0.503 = 0.95*vec(0.477) + 0.05*bm25(4.360) | 报告期内,公司实现营业收入人民币 28.74 亿元,同比增长 31.6%...
  [disclosure.docx#-] final=0.487 = 0.95*vec(0.513) + 0.05*bm25(0.000) | 第二节 主要财务数据
  [disclosure.docx#-] final=0.471 = 0.95*vec(0.475) + 0.05*bm25(1.727) | 本报告所涉及财务数据已经立信会计师事务所审计...
```

`final` 范围 [0.4， 0.6] 而不是 [0.9， 1.0] 的根因是 s04 §2.1 提的 cosine 归一化后 top-1 score 通常在 0.5 而不是 1.0(BGE 编码"应收账款 计提"和"1。应收账款"标题的向量微差）；`α=0.95` 时 dense cosine 主导，BM25 只在 dense 几乎打平时（差距 < 5%）做 tie-breaker。**第二个 hit BM25=0.000 是关键 evidence**——纯 dense 召回把它拉到第二位，纯 BM25 召回根本不会看它；`α=0.95` 让"语义近但字面无"的 chunk 保留在 top-k。

### 局限与下一步

本段做对了什么 — 把 01 的 BM25 跟 s05 的 dense cosine 在 `hybrid_topk(docs, query, query_vec, dense_score_fn, k, alpha)` 里按 α 加权融合,两路先各自归一 [0, 1] 再相加;API 解耦(`dense_score_fn` 注入)让单元测试可以直接喂 list[dict] + 暴力 cosine,production 换 chroma / ES 一行 import 替换;`α = 0.95` 时 dense 主导但 BM25 仍在 dense 几乎打平时把字面命中顶上来,默认权重与 RAGFlow `weighted_sum` 0.95/0.05 同序。


- **没有 per-query alpha 调度**：写死 `alpha=0.95` 对所有 query 同权；事实型查询（"应付账款多少"）应该更偏 BM25(alpha 低），概念型查询（"营收下滑的原因"）更偏 dense(alpha 高）——生产代码把 `vector_similarity_weight` 做成检索接口的入参，由调用方按 query 类型传；
- **假设两路信号都存在**：如果 docs 是空、或 dense_score_fn 全 0，hybrid_topk 仍然返回结构但都是 0 分；没有"任一路缺失就退化成另一路"的兜底——生产里 `Dealer.search` 在 ES 全文召回失败时会跳过 fusion 单独走向量；
- **没有第三层信号**：生产代码的 `sim = tkweight*tksim + vtweight*vtsim + rank_fea` 里 `rank_fea` 是 PageRank + tag boost，本节完全没有——权威文档没被抬高，带标签的 chunk 也没特殊对待；
- **没有精排**：`hybrid_topk` 直接返回粗排 top-k；生产代码在粗排后再 `rerank_with_knn(tkweight, vtweight, rank_fea)` 用 cross-encoder 重打分。本章 MVP 不做重排序，s07 会接上；
- **没有分页 / 候选窗口对齐**：固定 top-5，没有 `_rerank_window(page_size, top)` 这种"块大小 = page_size 整数倍"的工程细节——本节不深陷这块，s07+ 才会碰到。


- `ModuleNotFoundError: No module named 'sentence_transformers'`：BGE 依赖；`pip install sentence-transformers` 兜底；离线环境先 `pip download` 到本地、`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 强制走本地缓存。
- `OSError: [E050] Can't find model 'BAAI/bge-small-zh-v1.5'`：HuggingFace Hub 不可达；构建镜像时预下载模型到 `~/.cache/huggingface/hub/`，或 `HF_ENDPOINT=https://hf-mirror.com` 走国内镜像。
- `UnicodeEncodeError: 'gbk' codec can't encode character`：Windows 控制台编码问题，跑前 `set PYTHONIOENCODING=utf-8`(s05 / s06 / s07 同问题）。
- `alpha=0` 或 `alpha=1` 跑出意料的结果：把 α 调到极端值时，dense 几乎不参与排序（`α=0`）或 BM25 不参与（`α=1`）；见下方"思考题答案"的两条实测对比。
- 想看 `α` sweep：把 `hybrid_topk` 调三次（`α=0.05 / 0.5 / 0.95`），对比三组 top-3 的 `chunk_id` 是否重叠——这是最简单的 per-query alpha sweep 实验。

---

## 四、核心函数一览

| 函数 | 文件 | 输入 | 输出 | 一句话解释 |
|---|---|---|---|---|
| `tokenize(text)` / `_split_long(text, max_chars)` / `_chunk_by_paragraph(docs, max_chars=500)` | `code_01_bm25.py` | `str` / `list[dict]` | `list[str]` / `list[dict]` | 复制 s03 chunker;中英分词 + 句界切 + 500 字符 cap |
| `_pdf(path)` / `_docx(path)` / `_load_chunks()` | `code_01_bm25.py` | `Path` | `list[{text, page, source, chunk_id}]` | 复制 s02 loader + s03 chunker;在 01 内自包含 |
| `BM25` (class) | `code_01_bm25.py` | `list[list[str]]` | — | 手写 BM25:`__init__(corpus)` 算 IDF + avgdl;`score(query_tokens, doc_tokens)` 算每 doc 的 BM25 分 |
| `bm25_topk(docs, query, k=5)` | `code_01_bm25.py` | `list[dict], str, int` | `list[{chunk, bm25}]` | 用 `BM25` 给每个 chunk 打 BM25 分,降序取 top-k |
| `main()` (01) | `code_01_bm25.py` | — | 打印 BM25 top-k | 01 演示入口:BM25-only baseline |
| `_model()` / `_embed(texts)` / `_cosine(a, b)` | `code_02_hybrid_fusion.py` | `list[str]` / `(list, list)` | `list[list[float]]` / `float` | 复制 s04 BGE encode;cosine 用内积(已 L2 归一化) |
| `hybrid_topk(docs, query, query_vec, dense_score_fn, k, alpha)` | `code_02_hybrid_fusion.py` | `(list, str, list, callable, int, float)` | `list[{chunk, bm25, dense, score}]` | 核心公式:`final = α * dense_norm + (1-α) * bm25_norm`;dense 注入而非硬写 |
| `main()` (02) | `code_02_hybrid_fusion.py` | — | 打印 BM25 vs dense vs hybrid 三组 top-k | 02 演示入口;离线镜像环境 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 跑 |

## 五、跨代码 schema 设计取舍

为什么 fusion 公式是 `α * vec + (1-α) * bm25_norm` 而不是 RRF / concat / max？几个常见取舍的折中：

- **weighted_sum vs RRF**：`weighted_sum` 直接用各通道归一化后的分做线性加权，结果区间直观（`[0,1]`）、可解释、能叠加第三层信号；RRF(`Σ 1/(rank_i + 60)`）只看排名不看分，加 PageRank / tag boost 时得另起一条管线。**生产代码的两阶段都是线性加权**——DB `FusionExpr("weighted_sum", {"weights": "0.05,0.95"})` + app `tkweight*tksim + vtweight*vtsim + rank_fea`，链路简单、可扩展。本教程选 weighted_sum 是为了演示"分相加"这个最朴素的信号叠加。
- **α=0.95 vs α=0.5**：`α=0.5` 是"six-of-one-half-dozen-of-the-other"的折中默认值，本教程选 `α=0.95` 是因为 demo 用的 query(`"应收账款 计提"`）是**术语型密集场景**——dense cosine 在 BGE 编码后已经把相关 chunk 拉得很近，BM25 在 `1 - α = 0.05` 的权重下还能在 dense 几乎打平时把字面命中顶上来。**调小 α 到 0.5 的副作用**：`"营收"` / `"营业收入"` 这种同义场景里 dense 主导会让 BM25 的字面优势稀释。**通用 API 必须把 α 做成人参**。
- **dense_score_fn 注入 vs 内部硬写**：`hybrid_topk` 接 `dense_score_fn(chunk) -> float` 而不是绑死 chroma / ES / numpy，**让单元测试可以直接喂 `list[dict]` + 暴力 cosine**。生产替换为 `col.query` / `knn_search` 一行 import 替换，s06 的 MVP 不绑死任何具体向量库。
- **不归一 vs 各自归一 [0，1]**：BM25 累加分可能到几，cosine ∈ [0，1]。**直接相加 → BM25 一边倒**。必须先各自归一——本教程 BM25 除以 `max(bm_scores)`、dense 已经是 cosine ∈ [0，1](BGE `normalize_embeddings=True`)。
- **不存 chunk_id vs 保留 chunk_id**：fusion 后命中结构带 `chunk_id = {source}#{page}#p{n}`，跟 s05 的 `chunk_id` 命名一致，下一章（s07）做 rerank / chunk 拉取直接复用——**全链路 ID 守恒**是 RAG pipeline 的关键约定。

如果你的场景需要**第三层信号**（权威文档 / 标签 boost），就把 `hybrid_topk` 扩成 `final = α·v + (1-α)·b + γ·rank_fea`，γ 是 PageRank / tag cosine 的权重——生产代码的 `sim = tkweight*tksim + vtweight*vtsim + rank_fea` 就是这个思路。

## 六、跨代码文件集成

01 跑 BM25（纯字面，无外部依赖），02 把 01 的 BM25 + s04 BGE 的 dense cosine 在 `hybrid_topk` 里拼成加权融合。两者通过同一份 chunk 列表对齐——01 输出 `{chunk, bm25}`，02 拿 `dense_score_fn(chunk)` 算 dense 分，两者按 `chunk_id` 配对后做加权融合，得到最终排序。**s07 拿到 s06 的输出后，直接做 rerank 精排 + chunk 拉取，不需要重算 BM25 或 dense**——这是把"sparse 算过 / dense 算过"两件事固化在 s06 输出的红利。

## RAGFlow 实现

RAGFlow 的混合检索在 `rag/search.py` 的 `search()` 函数里：`FusionExpr` 类负责"BM25 + dense 加权融合"，公式是 `alpha * vec_score + (1 - alpha) * bm25_score`，alpha 可按 query 类型动态调（事实型偏 BM25，概念型偏 dense）。

**设计取舍**：RAGFlow 把"alpha 调度"提到运行时（per-query 调），不是写死。这跟 s06 toy 的 `alpha=0.5` 固定值差一档——alpha 应该在 query routing 阶段根据 query 类别（factual / conceptual / multi-hop）选不同值。

**整体拓扑**：(1) 01 走 `samples/ → load → chunk → BM25(corpus)` 拿 BM25 分 → (2) 02 拿 query → `s04._embed(query)` 拿 query 向量 → (3) `hybrid_topk(docs, query, query_vec, dense_score_fn, k, alpha)` 按 `α * dense + (1-α) * bm25_norm` 加权融合 → (4) 输出 `{chunk, bm25, dense, score}` 列表给 s07 精排。**生产差异**：RAGFlow 把 α 提到运行时 per-query 调(事实型偏 BM25、概念型偏 dense),s06 toy 写死 `α=0.95`(术语型密集场景);另外 RAGFlow 加第三层 PageRank / tag boost 信号(`sim = tkweight*tksim + vtweight*vtsim + rank_fea`),s06 只两层。

详细摘录与 5-15 行 "为什么这样写" 的分析见 [`docs/reference/ragflow-notes/hybrid_retrieval.md`](../docs/reference/ragflow-notes/hybrid_retrieval.md)。

---

## 选型速记

### 主流融合策略速览

下面这张表把社区常用的几类 fusion 策略按"信号维度 / 融合方式 / 是否需要训练 / 适用场景"列出来：

| 策略 | 信号维度 | 融合方式 | 训练成本 | 适用场景 |
|---|---|---|---|---|
| **weighted_sum**(本教程) | 2 (dense + sparse) | `α·v + (1-α)·b_norm` | 0 | 教学 / 简单混合 / 权重可解释 |
| **RRF (Reciprocal Rank Fusion)** | 2+ | `Σ 1/(rank_i + c)`,c=60 | 0 | Milvus `RRFRanker` / 不依赖分数量纲 |
| **Convex Combination** | 2+ | `Σ α_i · s_i`,Σ α_i = 1 | 0 | 多通道、加权约束凸 |
| **Borda Count** | 2+ | `Σ (N - rank_i + 1)` | 0 | 投票式融合、社交选择 |
| **生产 3-layer** | 3+ (dense + sparse + rank_fea) | DB `weighted_sum` + app `rerank_with_knn` + `rank_fea` | 0 (linear) | 生产 / per-query α |
| **Cross-encoder rerank** | 1 (query-doc pair) | `[CLS]` 头 + softmax | 训练 cross-encoder | s07 主题、精排 |

我们的 toy `hybrid_topk` 在信号维度上只占第一行——**两通道 + 加权**；生产代码把它扩到三通道并加了 PageRank / tag boost，s07 会叠 cross-encoder 重排序 做精排。

- **教学 / 快速原型 / 权重可解释** → `weighted_sum`（本教程）；α 直接调，看得见每个 hit 的子分贡献；
- **不依赖分数量纲 / 多通道** → RRF(`Σ 1/(rank_i + 60)`），Milvus `RRFRanker` 原生支持；
- **生产 / per-query α / 多层信号** → 生产 3-layer 流水线，DB fusion + app rerank + `rank_feature`；
- **追求 top-1 精度** → 加 cross-encoder 重排序（s07），把 `hybrid_topk` 的 top-20 喂给 `[CLS]` 模型重打分；
- **要先看清每个边界再选** → 用本章 code_02 把 query `"应收账款 计提"` 和 `"内存"` 各跑一次，看清楚"BM25 / dense 谁拉上去的、α 在什么范围最稳"，再决定要不要换 fusion。

### 扩展指南

加一种新 fusion 策略（RRF / convex / 加权）只要三步：

1. 写一个 `rrf_fusion(rank_lists: list[list[Hit]], k=60) -> list[Hit]` 或 `convex_fusion(score_lists, weights=[0.5, 0.5])`，**签名 / 返回 hit dict 形状和 `hybrid_topk` 一致**（`{chunk_id, text, dense, bm25, score}`），下游 s07 rerank 不用改一行；
2. 在 `code_02_hybrid_fusion.py` 的 `main()` 里按 `FUSION_MODE` env 选 fusion 函数（默认 `weighted_sum`），不要在 `hybrid_topk` 里写 `if mode == "rrf": ... elif mode == "convex": ...` 之类分发——它只懂 α 加权；
3. 给代码文件 README 加一段"它跟 weighted_sum 比，赢在哪 / 输在哪"的对照（RRF：不依赖分数量纲 / 多通道融合简单；convex：凸约束 / 数学性质好）。

不要把多 fusion 的判断塞进 `hybrid_topk`——它只懂"BM25 + dense α 加权"这一件事。`main()` 懂全 fusion 模式。本章 MVP 只跑 weighted_sum，但接口形状留好了。

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

### troubleshooting

- `ModuleNotFoundError: No module named 'sentence_transformers'`：BGE 依赖；`pip install sentence-transformers` 兜底；离线环境先 `pip download` 到本地、`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 强制走本地缓存。
- `OSError: [E050] Can't find model 'BAAI/bge-small-zh-v1.5'`：HuggingFace Hub 不可达；构建镜像时预下载模型到 `~/.cache/huggingface/hub/`，或 `HF_ENDPOINT=https://hf-mirror.com` 走国内镜像。
- `UnicodeEncodeError: 'gbk' codec can't encode character`：Windows 控制台编码问题，跑前 `set PYTHONIOENCODING=utf-8`(s05 / s06 / s07 同问题）。
- `alpha=0` 或 `alpha=1` 跑出意料的结果：把 α 调到极端值时，dense 几乎不参与排序（`α=0`）或 BM25 不参与（`α=1`）；见下方"思考题答案"的两条实测对比。
- 想看 `α` sweep：把 `hybrid_topk` 调三次（`α=0.05 / 0.5 / 0.95`），对比三组 top-3 的 `chunk_id` 是否重叠——这是最简单的 per-query alpha sweep 实验。
