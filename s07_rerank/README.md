# s07 重排序 (Rerank) — 章节总览

> **章节定位**: RAG 在线链路的"精排器"——把 s06 召回的 top-N 候选再过一次 **cross-encoder** (BAAI/bge-reranker-base),按 token 级 cross-attention 的相关性分重排,挑出真正沾边的 top-k。
> **章节定位**:本章节围绕 *BGE-reranker 本地精排* 这一层给出概念 / 问题 / MVP / 工业对照的完整弧线 —— 不引入 RankLLM 的 prompt 工程细节(那是另一种 LLM-as-rerank 路线),也不展开 ColBERT 的后期交互机制(那是另一种精度/效率折中)。

---

## 章节导航 (聚合入口保留)

| Unit | 标题 | 入口 |
| --- | --- | --- |
| 01 | Cross-encoder 精排 (BGE-reranker 重打分, BEFORE/AFTER 对比) | [`units/01_cross_encoder_rerank/code.py`](units/01_cross_encoder_rerank/code.py) |

跑法:

```bash
python s07_rerank/units/01_cross_encoder_rerank/code.py    # 跑 BEFORE/AFTER rerank 对比
# 旧路径仍可用 (聚合入口,等价于 unit 01):
python s07_rerank/code.py
```

依赖: `pypdf` + `python-docx` + `sentence-transformers` + `chromadb` + `FlagEmbedding` (BGE-reranker)。把 s02 / s03 / s04 / s05 / s06 跑通,s07 才能跑;首次跑会下载 `BAAI/bge-reranker-base` (~1GB)。

---

## 一、什么是重排序 (Re-ranking)?

### 1.1 核心定义

**重排序 (Re-ranking)** 是 RAG 在线链路的**第二阶段**——把第一阶段 (s06) 召回的 top-N 候选 (~10-100 条) 喂给一个**精排模型**,让模型对每个 `(query, chunk_text)` 对独立打分,再按新分取 top-k。s06 的混合召回 (BM25 + dense cosine) 是**双塔 (bi-encoder)**——query 和 chunk 各自独立编码再算相似度,**快但只看到向量层面的语义接近度**,没办法捕捉"查询词"和"chunk 里某个具体词"的精确匹配。重排序就是为了补这一刀:**慢一点,但看得更准**。

把它放进 RAG 全景看:**s07 是把 s06 的 top-N 命中再过一道 cross-attention,挑出真正相关的 top-k**。s05 落盘索引、s06 拉回候选、s07 在小池子上精排、s08 拼 Prompt 喂 LLM 生成。**没有 rerank 的 RAG 通常 top-1 精度只有 60-70%,加上 cross-encoder rerank 能顶到 80-90%**——这是工业 RAG 几乎必加 rerank 的根因。

### 1.2 Cross-encoder vs Bi-encoder:两条性质相反的通道

s06 / s07 的代码把所有事都写在一个文件里,但拆开看是两种**结构相反**的编码方式:

| 维度 | Bi-encoder (s06) | Cross-encoder (s07) |
|---|---|---|
| 编码方式 | query 和 chunk **独立**过同一个 BERT | query 和 chunk **拼接**成一个序列 `[CLS] query [SEP] chunk [SEP]` 一起过 BERT |
| 输出 | 两个向量,各自 L2 归一化后算 cosine | 一个 `[0,1]` 相关性分 (FlagReranker `normalize=True`) |
| 注意力范围 | 只能做**段级**语义对齐 (平均到一坨向量) | 做**token 级** cross-attention,query 的每个词和 chunk 的每个词都互相看见 |
| 速度 | 快——chunk 编码可离线预计算,query 编码 O(1),cosine O(d) | 慢——每对都要做一次完整 BERT forward,**O(n) 次推理** |
| 精度 | 粗,会被"向量平均"糊弄 | 精,能看到具体词的精确匹配 |
| 命中"内存"等具体词 | 弱 (向量里"内存"和"RAM"被平均到一起) | 强 (token 级 attention 直接命中"内存"两个字) |

**关键 takeaway**:cross-encoder 的高延迟不是 bug 而是 feature——它慢是因为**真的在算 query-doc 的 token 级交互**,不是用一个向量代表整段。这是为什么生产 RAG 几乎都把"双塔召回 + cross-encoder 精排"做成两阶段:**用便宜的 bi-encoder 拉宽,用贵的 cross-encoder 收紧**。

### 1.3 主流 rerank 方法对比

社区常用的几类 rerank 策略可以按"信号维度 / 推理成本 / 是否需要训练 / 适用场景"列成一张表:

| 策略 | 核心机制 | 推理成本 | 是否需要训练 | 适用场景 |
|---|---|---|---|---|
| **RRF (Reciprocal Rank Fusion)** | 融合多个检索器的排名,`Σ 1/(rank_i + c)`,c=60 | 极低 (纯排名计算) | 否 | Milvus `RRFRanker` / 多通道排名融合 (s06 末段提过) |
| **Cross-Encoder** (本章) | 把 `(query, doc)` 拼接过 BERT,输出 1 个 `[0,1]` 相关分 | 高 (N 次 BERT forward) | 是 (训练 cross-encoder) | Top-K 精排、本教程 MVP |
| **ColBERT** | 独立编码,后期 token 级 MaxSim | 中 (向量点积,不拼接) | 是 (训练 ColBERT) | 精度/效率折中、大规模检索 |
| **LLM-as-rerank (RankLLM)** | 把候选摘要塞进 prompt,LLM 输出排序 + 分 | 高 (按 token 计费 + 远端调用) | 否 (prompt 工程) | 高价值语义理解场景、多语言 |

本章 MVP 只用第二行——**BGE-reranker 本地 cross-encoder**。ColBERT / RankLLM 留作扩展,RAGFlow 把这四类统一抽象成 `RerankModel.Base` 的多 provider (Jina / Cohere / Voyage / Qwen / 本地 HF)。

---

## 二、为什么要单独写一章 rerank?

`rerank(query, hits, top_k=3)` 调起来不到 20 行——`_reranker()` 一次、`compute_score` 一次、按分排序取前 k。看起来不值得单独一章。但把它扔进真实样本就会发现,**"bi-encoder 召回了对的 chunk"和"排序把对的 chunk 顶到第一"之间隔着一道悬崖**——这道悬崖由几类典型问题堆起来。

### 2.1 真实世界的问题 (3 条典型)

1. **rerank 慢的 latency cliff**——cross-encoder 是 O(n) 次 BERT forward,**每一次都对 (query, chunk) 拼好的完整序列跑一次完整 BERT**。`top_k=10` 大概 100-300ms (CPU) / 30-50ms (GPU);`top_k=100` 直接 300ms-1s (CPU) / 100-300ms (GPU);`top_k=1000` 直接 3-10s,**线上不可接受**。生产上的标准解法就是两阶段——先用便宜的双塔召回 ~100-200 候选,再让 cross-encoder 在小池子上精排;**绝不在 top-1000 上跑 cross-encoder**。RAGFlow 把这条原则硬编码进 `_rerank_window`(`ceil(64/page_size) * page_size`,`rag/nlp/search.py:548-571`)。
2. **rerank 模型语言错配**——`bge-reranker-base` 主要在**英文 MS-MARCO + 英文 Wikipedia** 上训,中文任务上经常"准头不对"——它会把"内存"和"RAM"按英文语料里的共现模式打分,中文特有的术语对不上。**中文场景换 `bge-reranker-v2-m3` 或 `bge-reranker-large`** 更稳;切换只需改 `_reranker()` 里那行字符串。RAGFlow 默认就配 `bge-reranker-v2-m3` 做本地 rerank(`rerank_model.py:447-488`)。
3. **LLM-as-rerank 的成本 tier**——还能再叠一层:拿 rerank 后的 top-N 让 GPT-4 / Claude 做"哪个最相关"判断。RAGFlow 把这条线抽象成 `RerankModel.Base` 的多 provider (Cohere / Jina / Voyage / Qwen / NVIDIA / 百度千帆),准但每千次调用都要花 token 钱 + 远端 HTTP 延迟。本次 MVP 不接——教学仓库只要把"cross-encoder 比 bi-encoder 准"这件事讲清楚就够了;**生产系统需要按"成本 vs 精度"做 tier 选型**(本地 BGE → 远端 BGE-v2-m3 → Cohere Rerank → GPT-4 zero-shot)。

### 2.2 这些问题为什么必须显式面对

每条都对应不同的工业级解法——两阶段窗口控制、语言适配、多 provider 成本 tier。**s07 的目标不是解决它们,而是把它们显式暴露出来,让你看到 toy rerank 的边界**。

这也是为什么本章只有一个 unit:

- **unit 01**——跑通最小骨架(`_reranker()` + `rerank()` + BEFORE/AFTER 对比),演示"rerank 把字面命中顶到 top-1、向量近邻被排到第二位"的反面 case;每个 hit 同时打出 `rerank_score` 和 `vec` 两个分,方便看出"是 rerank 拉上去的还是 dense 拉上去的"。

也是为什么我们不直接接 LLM-as-rerank——它把"精排这件事到底在算什么"封装得非常好,但你看不到**为什么 cross-encoder 能看到 bi-encoder 看不到的精确匹配**。先见 cross-encoder 的 token 级 attention,再看 RankLLM 的 prompt 设计,比直接用云 API 学到的多。

---

## 三、怎么做?

### 3.1 章节导航

| Unit | 主题 | 它解决什么 | 对照 RAGFlow |
|---|---|---|---|
| [01_cross_encoder_rerank](./units/01_cross_encoder_rerank/README.md) | BGE-reranker cross-encoder 精排 + BEFORE/AFTER 对比 | "bi-encoder 召回了对的 chunk,排序没顶上去" | `RerankModel.Base.similarity` 多 provider 抽象 + `if rerank_mdl and sres.total > 0` 两阶段开关 |

### 3.2 跑起来

```bash
pip install FlagEmbedding            # BGE-reranker 依赖
python s07_rerank/units/01_cross_encoder_rerank/code.py    # 跑 BEFORE/AFTER rerank (要本地 BGE-reranker ~1GB,首次下载)
# 旧路径仍可用 (聚合入口,等价于 unit 01):
python s07_rerank/code.py
```

离线 / 镜像环境跑 unit 01:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python s07_rerank/units/01_cross_encoder_rerank/code.py
```

### 3.3 核心函数一览

s07 的代码同样薄,每个函数都对应一种"加载 / 精排"的角色:

| 函数 | 文件 | 输入 | 输出 | 一句话解释 |
|---|---|---|---|---|
| `_embed_model()` / `_embed(texts)` | `units/01_cross_encoder_rerank/code.py` | `list[str]` | `list[list[float]]` | `@lru_cache(maxsize=1)` 加载本地 BGE-small-zh-v1.5;`normalize_embeddings=True` (跟 s04 / s05 / s06 同款) |
| `_cosine(a, b)` | `units/01_cross_encoder_rerank/code.py` | `(list[float], list[float])` | `float` | 两个已 L2 归一化向量的内积 (cosine ≡ inner_product) |
| `_reranker()` | `units/01_cross_encoder_rerank/code.py` | — | `FlagReranker` | `@lru_cache(maxsize=1)` 加载 `BAAI/bge-reranker-base`,`use_fp16=False`;同一进程只下载、加载一次 |
| `rerank(query, hits, top_k=3)` | `units/01_cross_encoder_rerank/code.py` | `(str, list[dict], int)` | `list[{text, source, page, chunk_id, dense, bm25, score, rerank_score}]` | 把 s06 的 hits 喂给 cross-encoder 重打分,按 `rerank_score` 降序取前 k,保留 `score` (s06 混合分) 供前后对比 |
| `_hybrid_topk(docs, query, query_vec, dense_score_fn, k, alpha)` | `units/01_cross_encoder_rerank/code.py` | `(list, str, list, callable, int, float)` | `list[{text, source, page, chunk_id, dense, bm25, score}]` | 复制 s06 unit 02 的 hybrid_topk 公式 (`α * vec + (1-α) * bm25_norm`);self-contained 跑通 BM25+dense 召回 → rerank 对比 |
| `main()` (unit 01) | `units/01_cross_encoder_rerank/code.py` | — | 打印 BEFORE/AFTER rerank top-3 | unit 01 演示入口,默认 query `"内存"` (EOFError 时兜底) |

### 3.4 rerank 设计取舍

为什么 rerank 公式是 `_reranker().compute_score(pairs, normalize=True)` + 按分排序取前 k,而不是 token 级 attention mask / 多任务学习 / ColBERT 后期交互?几个常见取舍的折中:

- **cross-encoder vs ColBERT vs LLM-as-rerank**:cross-encoder 把 query 和 chunk 拼接,精度最高但 latency 最高;ColBERT 独立编码 + 后期 MaxSim,精度/效率居中;LLM-as-rerank 把候选塞进 prompt,精度最高但成本最高。**RAGFlow 用 `RerankModel.Base` 把这四类统一抽象成 `.similarity(query, docs) -> [score]`,输出都归一到 `[0,1]`**——`tkweight * tksim + vtweight * vtsim` 公式对 provider 无感知。本教程选 cross-encoder 是因为它**离线可复现 + 不需要 API key + 精度足够 demo**;BAAI/bge-reranker-base ~1GB 一次下载完就一直在本地。
- **`top_k=3` (精排后保留几个) vs `top_k=10` (召回多少)**:前者是"喂给 LLM 的最终候选数"——s08 会把这 k 个 hit 拼进 prompt,**越大越费 token 但越准**;后者是"喂给 cross-encoder 的候选数"——**越大越费 latency**。生产上一般召回 50-100,精排取 3-5。本教程选 `top_k=3` 是因为 s05 / s06 的 `samples/` 只有 34 个 chunk,10 召回 3 精排已经能看出"rerank 把对的顶上去"的效果;**生产请按 query 复杂度调**。
- **`@lru_cache(maxsize=1)` 缓存 vs 每次重载**:`FlagReranker` 加载要 ~5-10s (CPU) / ~2-3s (GPU),rerank 一对 query-chunk 本身只要 3-5ms (CPU) / 1-2ms (GPU)。**没有缓存,每次 rerank 都白白浪费 5s 加载**;有缓存,同一进程内 rerank 任意次只加载一次。`@lru_cache(maxsize=1)` 是 Python 标准库最简单的"单例"装饰器,跟 s04 / s05 / s06 的 `_embed_model()` 同款做法。
- **`use_fp16=False` vs `use_fp16=True`**:`FlagReranker` 默认 fp32,精度最高但显存占用也最高 (~2GB)。**GPU + 显存紧** 时改 `use_fp16=True`,推理速度 ~2x 但精度损失 ~1%。本教程选 fp32 是因为 demo 在 CPU 上跑,fp16 反而更慢;**生产 GPU 请开 fp16**。
- **`normalize=True` vs `normalize=False`**:`FlagReranker.compute_score` 默认输出是**原始 sigmoid logits**,范围 `[0, +∞)`;设 `normalize=True` 后会被 FlagEmbedding 内部映射到 `[0,1]` (具体公式是 sigmoid + 校准),跟 s06 的 cosine ∈ [0,1] 同一个量纲。**如果不归一,print 出来的 `rerank_score` 会显示 10+ 这样的数字**,看起来吓人但其实是 logits。

### 3.5 如何切换到 RAGFlow 风格 rerank

加一种 rerank 策略 (ColBERT / RankLLM / Cohere API) 只要三步:

1. 写一个 `colbert_rerank(query, hits, top_k=3)` 或 `cohere_rerank(query, hits, top_k=3, api_key=...)`,签名和 `rerank` 一致;
2. 在 `main()` 里按 `RERANK_MODE` env 选 rerank 函数;
3. 给 unit README 加一段"它跟 BGE-reranker 比,赢在哪 / 输在哪"的对照。

不要在 `rerank` 里写 `if mode == "bge": ... elif mode == "colbert": ...` 之类分发——它会污染单一职责。`rerank` 只懂 BGE,`main()` 懂全 rerank 模式。本章 MVP 只跑 BGE,但接口形状留好了。

### 3.6 实际跑出来的 rerank 形状

把 unit 01 跑在仓库自带的 `samples/` 上,`rerank` 返回的命中结构长这样:

```python
# query='内存', top_k=3 over 10 candidates from s06 hybrid_topk
[
  {
    "text": "五、可靠性与可维护性 冗余设计:电源、风扇、Boot 盘、PCIe 控制器均支持 N+1 冗余;内存 ...",
    "source": "server_whitepaper.pdf",
    "page": 4,
    "chunk_id": "server_whitepaper.pdf#4#p?",
    "dense": 0.590,        # s06 dense cosine (BGE 归一化)
    "bm25": 5.411,         # s06 BM25 累加分
    "score": 0.795,        # s06 α=0.5 混合分 = 0.5*0.590 + 0.5*(5.411/max_bm25)
    "rerank_score": 0.527  # s07 BGE-reranker-base [0,1] 相关性分
  },
  ...
]
```

**关键现象**:rerank 分和 s06 的 vec 分**不同步**——s06 把 BM25 字面命中冠军 (`#4` 可靠性章节里顺带提到"内存",`score=0.795`) 排到第一,但 cross-encoder 觉得它只有 `rerank_score=0.527`(因为正文主题是"可靠性","内存"只是一行配置);而纯"四、应用场景"章节虽然 `dense=0.545`,`rerank_score` 却给到 **0.664**——cross-encoder 看到它把"高密度计算"作为主题,跟 query "内存"的计算密度语义最沾边。这就是 **cross-encoder 比 bi-encoder 准的地方:它能看到具体词而不是被一个向量平均值糊弄**——bi-encoder 只看 `dense=0.590 vs 0.545` 这种几乎打平的数字,排不出"哪个才是真沾边"。

### 3.7 跑出来是什么样

Unit 01 的实测输出(`query='内存'`,`EOFError` 自动兜底时实际等价于 stdin 空输入 → `query='内存'`):

```
loaded 34 chunks from samples/
query='内存', alpha=0.5 (BM25 + dense 等权融合)

--- BEFORE rerank (s06 混合召回 top-3) ---
  #1 [server_whitepaper.pdf#4] score=0.795 (vec=0.590, bm25=5.411) | 五、可靠性与可维护性 冗余设计：电源、风扇、Boot 盘、PCIe 控制器均支持 N+1 冗余；内存
  #2 [server_whitepaper.pdf#2] score=0.736 (vec=0.552, bm25=4.974) | 三、整机规格 组件 规格 说明 处理器 2 × 第三代 Intel Xeon 可 扩展处理器 最高
  #3 [server_whitepaper.pdf#3] score=0.726 (vec=0.545, bm25=4.909) | 四、应用场景 云数据中心：作为通用计算节点支撑私有云与混合云平台

--- AFTER rerank (BAAI/bge-reranker-base top-3) ---
  #1 [server_whitepaper.pdf#3] rerank=0.664 vec=0.545 | 四、应用场景 云数据中心：作为通用计算节点支撑私有云与混合云平台，配合虚拟化与容器平台提供高 密度的
  #2 [server_whitepaper.pdf#1] rerank=0.550 vec=0.559 | 二、关键特性 计算密度：单台 2U 机箱内集成两颗处理器、32 条内存 DIMM 与 10 个 PC
  #3 [server_whitepaper.pdf#4] rerank=0.527 vec=0.590 | 五、可靠性与可维护性 冗余设计：电源、风扇、Boot 盘、PCIe 控制器均支持 N+1 冗余；内存
```

`rerank_score` 范围 [0, 1] (`FlagReranker` `normalize=True` 归一化后);`#3 [server_whitepaper.pdf#4]` 在 BEFORE 和 AFTER 都出现但排序微调——`score=0.795` (vec=0.590 + bm25 词面命中) 和 `rerank=0.527` (cross-encoder 看到它是可靠性章节里顺带提到内存) **信号不一致**:BM25 字面命中把它顶到第一,rerank 觉得它不是"内存"主题段落。**这正是 rerank 的价值**——bi-encoder 召回了对的 chunk (BEFORE 也有它),但 cross-encoder 在 token 级 attention 上看出"内存"在 #4 章节里只是顺带提一句,该把它从第一压到第三。

**Troubleshooting**:

- `ModuleNotFoundError: No module named 'FlagEmbedding'`:BGE-reranker 依赖;`pip install FlagEmbedding` 兜底;离线环境先 `pip download FlagEmbedding` 到本地、`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 强制走本地缓存。
- `OSError: [E050] Can't find model 'BAAI/bge-reranker-base'`:HuggingFace Hub 不可达;构建镜像时预下载模型到 `~/.cache/huggingface/hub/`,或 `HF_ENDPOINT=https://hf-mirror.com` 走国内镜像。**模型 ~1GB,首次下载慢**。
- `UnicodeEncodeError: 'gbk' codec can't encode character`:Windows 控制台编码问题,跑前 `set PYTHONIOENCODING=utf-8`(s05 / s06 / s07 同问题)。
- `rerank_score > 1` 或 `< 0`:`FlagReranker(..., normalize=False)` 默认输出是 logits 不是概率,设 `normalize=True` 让它映射到 `[0,1]`。
- `rerank 跑 5 分钟不出结果`:`_reranker()` 没缓存住 / 每次新进程——检查 `@lru_cache(maxsize=1)` 是否还在,或者是不是被多线程调用 (lru_cache 不跨进程)。

---

## 四、对照 RAGFlow + 思考题

### 4.1 ragflow 怎么做的

RAGFlow 在 `Dealer.retrieval` 里把"是否走 cross-encoder / LLM rerank"做成开关(`if rerank_mdl and sres.total > 0`),配 `rerank_mdl` 才走第二阶段;走的时候还要再和 `tksim` (term similarity) 线性加权一次(`tkweight * tksim + vtweight * vtsim`),不是单看 rerank 分。`RerankModel.Base.similarity` 把十几个云 provider (Cohere / Jina / Voyage / Qwen / NVIDIA / 百度千帆) 和**本地 HuggingFace cross-encoder** (`BAAI/bge-reranker-v2-m3`) 的输出都归一到 `[0,1]` 再喂进同一个公式——所以你**配不同的 provider,链路其他部分一行不用改**。完整摘录与 3 条"为什么这样写"的分析见 [`ragflow_notes/rerank.md`](../ragflow_notes/rerank.md)。

一句话对比:RAGFlow 把"rerank"做成了**三级流水线**——DB 侧 `weighted_sum` fusion (粗召回,偏向量) → app 侧 `rerank_by_model` (cross-encoder 精排,**只在配了 `rerank_mdl` 时走**) → `rank_feature` (PageRank + tag cosine,权威文档 + 标签加权)。**cross-encoder rerank 是可选项**,不是必走,这是把"开销留给愿意付钱的人"的开关。本章 MVP 走的就是中间那一步,但**没接云 provider**(教学聚焦 + 离线可复现 + 成本与延迟),用本地 `BAAI/bge-reranker-base` 替代 RAGFlow 的 `BAAI/bge-reranker-v2-m3`。

### 4.2 主流 rerank 策略速览

下面这张表把社区常用的几类 rerank 策略按"信号维度 / 推理成本 / 是否需要训练 / 适用场景"列出来:

| 策略 | 信号维度 | 推理成本 | 是否需要训练 | 适用场景 |
|---|---|---|---|---|
| **RRF** (s06 末段) | 2+ (排名倒数) | 极低 | 否 | Milvus `RRFRanker` / 多通道排名融合 |
| **Cross-Encoder** (本章) | 1 (query-doc pair) | 高 (N 次 BERT forward) | 是 (训练 cross-encoder) | Top-K 精排、本教程 MVP |
| **ColBERT** | token 级 MaxSim | 中 (向量点积,不拼接) | 是 (训练 ColBERT) | 精度/效率折中、大规模检索 |
| **LLM-as-rerank (RankLLM)** | prompt + LLM 推理 | 高 (token 计费 + 远端) | 否 (prompt 工程) | 高价值语义理解、多语言 |
| **RAGFlow 多 provider** | 1 (统一归一到 [0,1]) | 高 (依 provider) | 否 (provider 接管) | 生产 / per-query 成本 tier |

我们的 toy `rerank` 在信号维度上只占第二行——**cross-encoder 精排**;RAGFlow 用 `RerankModel.Base` 把第二到第四行都包成同一个接口,租户按"成本 vs 精度"选 provider。

### 4.3 选型速记

- **教学 / 快速原型 / 离线可复现** → 本地 cross-encoder (本教程,`BAAI/bge-reranker-base`),无 API key、~1GB 一次下载、CPU 可跑;
- **生产中文场景** → `BAAI/bge-reranker-v2-m3` 或 `bge-reranker-large`,v2-m3 多语言、large 中文更准;
- **生产英文场景 + 预算足** → Cohere Rerank / Voyage Rerank,远端 API、付费、按 query 计费;
- **生产 + 极致精度 + 接受 LLM 成本** → RankLLM (GPT-4 / Claude zero-shot rerank),按 token 计费、latency 高;
- **要先看清每个边界再选** → 用本章 unit 01 把 `BAAI/bge-reranker-base` 和 `BAAI/bge-reranker-v2-m3` 各跑一次,对比 top-3 的 `chunk_id` 是否重叠——这是最简单的"rerank model A/B"实验。

### 4.4 思考题

1. **如果召回了 100 个、rerank 要跑多少对?**  
   答:**100 对** (1 query × 100 candidates)。Cross-encoder 的 query+chunk 是 1 对 1,不是 1 对 N。100 个 chunk 就是 100 对,O(n) 不是 O(n²)。100 对 × ~3ms/对 ≈ 300ms-1s;如果召回 1000 个直接 3-10s,**线上不可接受**。所以**召回量要压到 cross-encoder 能吃的范围** (一般 50-100),再多就让粗召回用更便宜的近似 (量化向量、IVF 索引) 顶住。详见 [`thinking_answers.md`](./thinking_answers.md)。

2. **为什么 s07 没有 unit 02 failure mode 而 s02/s03 有?**  
   答:s07 的失败模式不是"代码跑不通"而是"rerank 在某些边界条件下退化成 bi-encoder"——比如 (a) 召回量太小 (<5) 时 rerank 跟 dense 排序几乎重合,看不出差别;(b) `bge-reranker-base` 配中文 query 时精度比 `v2-m3` 低 ~10%;(c) `top_k` 大于召回量时直接返回原序。这些"边界模式"已经在 README §2.1 用文字讲清楚,不需要单独的 failure-mode unit 跑——s02 (loader edge cases) / s03 (chunker 边界) 是"代码逻辑分支",s07 是"超参/模型选择",**叙述载体不一样**。

3. **如果改用 LLM-as-reranker,prompt 设计要点是什么?**  
   答:三条 — (a) **候选摘要 + 编号**,不要塞完整 chunk (LLM 上下文有限);(b) **明确输出格式**,如 `Doc: 9, Relevance: 7`、`Doc: 3, Relevance: 4`,强结构化输出便于解析;(c) **给 "无关文档" 留位**,提示词里写明"请不要包含与问题无关的文档",LLM 才不会硬凑够 k 个返回。完整 prompt 模板见 all-in-rag 第四章 §1.2 RankLLM 的示例。