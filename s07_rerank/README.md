# s07 重排序 (Rerank) — Cross-encoder 把 top-N 重排成 top-K

> **本章定位**：s07 是 RAG 在线链路的"精排器"——把 s06 召回的 top-N 候选再过一次 **cross-encoder**（BAAI/bge-reranker-base），按 token 级 cross-attention 的相关性分重排，挑出真正沾边的 top-k。详细定位见 s00 §1.4；RAGFlow 实现见本章末"## RAGFlow 实现"。

---

## 一、章节介绍

### 1.1 核心定义：什么是重排序 (Re-ranking)？cross-encoder vs bi-encoder

**重排序 (Re-ranking)** 是 RAG 在线链路的**第二阶段**——把第一阶段（s06）召回的 top-N 候选（~10-100 条）喂给一个**精排模型**，让模型对每个 `(query, chunk_text)` 对独立打分，再按新分取 top-k。s06 的混合召回（BM25 + dense cosine）是**双塔 (bi-encoder)**——query 和 chunk 各自独立编码再算相似度，**快但只看到向量层面的语义接近度**，没办法捕捉"查询词"和"chunk 里某个具体词"的精确匹配。重排序就是为了补这一刀：**慢一点，但看得更准**。

```
        s06 hits (top-N)                            s08 hits (top-K)
        BM25 + dense 融合后                         BGE-reranker 重打分后
        ┌─ #1 [bm25冠军] score=0.795 ─┐             ┌─ #1 [主题真沾边] rerank=0.664 ─┐
        ├─ #2 [vec=0.552]    score=0.736  ─▶  cross-encoder N 次 BERT forward   ─▶  ├─ #2 ...        rerank=0.550 ─┤
        ├─ #3 ...                score=0.726        (query, chunk) 拼接 → [0,1]    ├─ #3 ...        rerank=0.527 ─┘
        └─ ... top-N ...                              按 rerank_score 降序取 K
            段级语义对齐(粗)                          token 级 cross-attention(精)
```

把它放进 RAG 全景看：**s07 是把 s06 的 top-N 命中再过一道 cross-attention，挑出真正相关的 top-k**。s05 落盘索引、s06 拉回候选、s07 在小池子上精排、s08 拼 Prompt 喂 LLM 生成。**没有 rerank 的 RAG 通常 top-1 精度只有 60-70%，加上 cross-encoder 重排序 能顶到 80-90%**——这是工业 RAG 几乎必加 rerank 的根因。

#### Cross-encoder vs Bi-encoder：两条性质相反的通道

s06 / s07 的代码把所有事都写在一个文件里，但拆开看是两种**结构相反**的编码方式：

| 维度 | Bi-encoder (s06) | Cross-encoder (s07) |
|---|---|---|
| 编码方式 | query 和 chunk **独立**过同一个 BERT | query 和 chunk **拼接**成一个序列 `[CLS] query [SEP] chunk [SEP]` 一起过 BERT |
| 输出 | 两个向量，各自 L2 归一化后算 cosine | 一个 `[0,1]` 相关性分（FlagReranker `normalize=True`） |
| 注意力范围 | 只能做**段级**语义对齐（平均到一坨向量） | 做**token 级** cross-attention，query 的每个词和 chunk 的每个词都互相看见 |
| 速度 | 快——chunk 编码可离线预计算，query 编码 O(1)，cosine O(d) | 慢——每对都要做一次完整 BERT forward，**O(n) 次推理** |
| 精度 | 粗，会被"向量平均"糊弄 | 精，能看到具体词的精确匹配 |
| 命中"内存"等具体词 | 弱（向量里"内存"和"RAM"被平均到一起） | 强（token 级 attention 直接命中"内存"两个字） |

**关键 takeaway**：cross-encoder 的高延迟不是 bug 而是 feature——它慢是因为**真的在算 query-doc 的 token 级交互**，不是用一个向量代表整段。这是为什么生产 RAG 几乎都把"双塔召回 + cross-encoder 精排"做成两阶段：**用便宜的 bi-encoder 拉宽，用贵的 cross-encoder 收紧**。

#### 主流 rerank 方法对比

社区常用的几类 rerank 策略可以按"信号维度 / 推理成本 / 是否需要训练 / 适用场景"列成一张表：

| 策略 | 核心机制 | 推理成本 | 是否需要训练 | 适用场景 |
|---|---|---|---|---|
| **RRF (Reciprocal Rank Fusion)** | 融合多个检索器的排名，`Σ 1/(rank_i + c)`，c=60 | 极低（纯排名计算） | 否 | Milvus `RRFRanker` / 多通道排名融合（s06 末段提过） |
| **Cross-Encoder**（本章） | 把 `(query, doc)` 拼接过 BERT，输出 1 个 `[0,1]` 相关分 | 高（N 次 BERT forward） | 是（训练 cross-encoder） | Top-K 精排、本教程 MVP |
| **ColBERT** | 独立编码，后期 token 级 MaxSim | 中（向量点积，不拼接） | 是（训练 ColBERT） | 精度/效率折中、大规模检索 |
| **LLM-as-rerank (RankLLM)** | 把候选摘要塞进 prompt，LLM 输出排序 + 分 | 高（按 token 计费 + 远端调用） | 否（prompt 工程） | 高价值语义理解场景、多语言 |

本章 MVP 只用第二行——**BGE-reranker 本地 cross-encoder**。ColBERT / RankLLM 留作扩展，生产代码把这四类统一抽象成 `RerankModel.Base` 的多 provider（Jina / Cohere / Voyage / Qwen / 本地 HF）。

### 1.2 真实世界的问题：latency cliff / 语言错配 / LLM-as-rerank 成本 tier

`rerank(query, hits, top_k=3)` 调起来不到 20 行——`_reranker()` 一次、`compute_score` 一次、按分排序取前 k。看起来不值得单独一章。但把它扔进真实样本就会发现，**"bi-encoder 召回了对的 chunk"和"排序把对的 chunk 顶到第一"之间隔着一道悬崖**——这道悬崖由几类典型问题堆起来。

#### 真实世界的问题

1. **rerank 慢的 latency cliff**——cross-encoder 是 O(n) 次 BERT forward，**每一次都对 (query， chunk) 拼好的完整序列跑一次完整 BERT**。`top_k=10` 大概 100-300ms (CPU) / 30-50ms (GPU）；`top_k=100` 直接 300ms-1s (CPU) / 100-300ms (GPU）；`top_k=1000` 直接 3-10s，**线上不可接受**。生产上的标准解法就是两阶段——先用便宜的双塔召回 ~100-200 候选，再让 cross-encoder 在小池子上精排；**绝不在 top-1000 上跑 cross-encoder**。生产代码把这条原则硬编码进 `_rerank_window`（`ceil(64/page_size) * page_size`）。
2. **rerank 模型语言错配**——`bge-reranker-base` 主要在**英文 MS-MARCO + 英文 Wikipedia** 上训，中文任务上经常"准头不对"——它会把"内存"和"RAM"按英文语料里的共现模式打分，中文特有的术语对不上。**中文场景换 `bge-reranker-v2-m3` 或 `bge-reranker-large`** 更稳；切换只需改 `_reranker()` 里那行字符串。生产代码默认就配 `bge-reranker-v2-m3` 做本地 rerank。
3. **LLM-as-rerank 的成本 tier**——还能再叠一层：拿 rerank 后的 top-N 让 GPT-4 / Claude 做"哪个最相关"判断。生产代码把这条线抽象成 `RerankModel.Base` 的多 provider（Cohere / Jina / Voyage / Qwen / NVIDIA / 百度千帆），准但每千次调用都要花 token 钱 + 远端 HTTP 延迟。本次 MVP 不接——教学仓库只要把"cross-encoder 比 bi-encoder 准"这件事讲清楚就够了；**生产系统需要按"成本 vs 精度"做 tier 选型**（本地 BGE → 远端 BGE-v2-m3 → Cohere Rerank → GPT-4 zero-shot）。

#### 这些问题为什么必须显式面对

每条都对应不同的工业级解法——两阶段窗口控制、语言适配、多 provider 成本 tier。**s07 的目标不是解决它们，而是把它们显式暴露出来，让你看到 toy rerank 的边界**。

这也是为什么本章只有一个代码文件：

- **code_01**——跑通最小骨架（`_reranker()` + `rerank()` + BEFORE/AFTER 对比），演示"rerank 把字面命中顶到 top-1、向量近邻被排到第二位"的反面 case；每个 hit 同时打出 `rerank_score` 和 `vec` 两个分，方便看出"是 rerank 拉上去的还是 dense 拉上去的"。

也是为什么我们不直接接 LLM-as-rerank——它把"精排这件事到底在算什么"封装得非常好，但你看不到**为什么 cross-encoder 能看到 bi-encoder 看不到的精确匹配**。先见 cross-encoder 的 token 级 attention，再看 RankLLM 的 prompt 设计，比直接用云 API 学到的多。

---

## 二、Cross-encoder 重打分 (BGE-reranker）：[c01_cross_encoder_rerank.py](c01_cross_encoder_rerank.py)

入口：[`c01_cross_encoder_rerank.py`](c01_cross_encoder_rerank.py)

把 s06 召回的 top-N 候选再过一道 cross-encoder，按"query+chunk 拼起来看"的相关性重排序。
这是 bi-encoder（双塔）召回之后的两阶段精排：精排贵但只对小池子跑，所以准。

### 2.1 代码干了什么：FlagReranker + query+chunk 拼对 + rerank() + BEFORE/AFTER 对比

`code.py` 把 s06 混合召回吐出的 top-N（默认 N=10）候选，跟原始 query 拼成 `[query, chunk_text]` 对，喂给 `FlagReranker("BAAI/bge-reranker-base")` ——一个 BERT 类 cross-encoder 模型。它对每对 `(query, chunk)` 做一次完整 forward，让 BERT 的 self-attention 同时看到两端、做 token 级 cross-attention，输出一个归一化到 `[0,1]` 的相关性分。我们按这个分降序取 top-3。

`rerank(query, hits, top_k=3)` 函数吃 s06 风格的 hits list（每条带 `text/source/page/chunk_id/score`），吐按 `rerank_score` 降序的前 K 条；每个返回项保留原始 `score`（混合召回分），新增 `rerank_score`（cross-encoder 分）。模型用 `@lru_cache(maxsize=1)` 缓存，同进程只下载、加载一次。

`main()` 跑一个完整的对比：BEFORE 是 s06 混合召回的 top-3（按 `alpha*vec + (1-alpha)*bm25_norm` 排），AFTER 是 cross-encoder 精排后的 top-3 —— 你会看到排序变化，因为 cross-encoder 看到的"查询词 vs 文档词"的精确匹配信号，比双塔向量平均值敏锐得多。

### 2.2 跑一遍：单条命令与首次 ~1GB 模型下载 + BEFORE/AFTER 输出

```bash
python s07_rerank/c01_cross_encoder_rerank.py
# 问: 内存
```

输出示例（首次跑会下载 BAAI/bge-reranker-base ~1GB）：

```
loaded 34 chunks from samples/
query='内存', alpha=0.5 (BM25 + dense 等权融合)

--- BEFORE rerank (s06 混合召回 top-3) ---
  #1 [server_whitepaper.pdf#2] score=0.976 (vec=0.976, bm25=0.000) | 内存 32 × DDR4 3200 ECC RDIMM ...
  #2 [server_whitepaper.pdf#1] score=0.905 (vec=0.905, bm25=0.000) | ... 内存、10 个 PCIe 4.0 扩展槽位 ...
  #3 [server_whitepaper.pdf#4] score=0.869 (vec=0.869, bm25=0.000) | 内存支持镜像、备用与纠错码（ECC）三种数据保护模式 ...

--- AFTER rerank (BAAI/bge-reranker-base top-3) ---
  #1 [server_whitepaper.pdf#1] rerank=0.954 vec=0.905 | ... 内存、10 个 PCIe 4.0 扩展槽位 ...
  #2 [server_whitepaper.pdf#2] rerank=0.644 vec=0.976 | 内存 32 × DDR4 3200 ECC RDIMM ...
  #3 [server_whitepaper.pdf#4] rerank=0.870 vec=0.869 | 内存支持镜像、备用与纠错码（ECC）三种数据保护模式 ...
```

注意第 1 条 vs 第 2 条：混合召回把 vec=#1 的"内存 32 × DDR4 。。。"排第一（配置表，纯字面）但 cross-encoder 觉得它只有 0.644（因为正文是配置表，"内存"只是表里一行），而"2 内存"章节虽然 vec 只有 0.905，rerank 却给到 0.954。这就是 cross-encoder 比 bi-encoder 准的地方：它能看到具体词而不是被一个向量平均值糊弄。

### 2.3 实测输出：rerank 返回结构 + 关键现象 rerank 分 ≠ vec 分

把 code_01 跑在仓库自带的 `samples/` 上，`rerank` 返回的命中结构长这样：

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

**关键现象**：rerank 分和 s06 的 vec 分**不同步**——s06 把 BM25 字面命中冠军（`#4` 可靠性章节里顺带提到"内存"，`score=0.795`）排到第一，但 cross-encoder 觉得它只有 `rerank_score=0.527`（因为正文主题是"可靠性"，"内存"只是一行配置）；而纯"四、应用场景"章节虽然 `dense=0.545`，`rerank_score` 却给到 **0.664**——cross-encoder 看到它把"高密度计算"作为主题，跟 query "内存"的计算密度语义最沾边。这就是 **cross-encoder 比 bi-encoder 准的地方：它能看到具体词而不是被一个向量平均值糊弄**——bi-encoder 只看 `dense=0.590 vs 0.545` 这种几乎打平的数字，排不出"哪个才是真沾边"。

Code_01 的实测输出（`query='内存'`，`EOFError` 自动兜底时实际等价于 stdin 空输入 → `query='内存'`）：

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

`rerank_score` 范围 [0， 1]（`FlagReranker` `normalize=True` 归一化后）；`#3 [server_whitepaper.pdf#4]` 在 BEFORE 和 AFTER 都出现但排序微调——`score=0.795`（vec=0.590 + bm25 词面命中）和 `rerank=0.527`（cross-encoder 看到它是可靠性章节里顺带提到内存）**信号不一致**：BM25 字面命中把它顶到第一，rerank 觉得它不是"内存"主题段落。**这正是 rerank 的价值**——bi-encoder 召回了对的 chunk（BEFORE 也有它），但 cross-encoder 在 token 级 attention 上看出"内存"在 #4 章节里只是顺带提一句，该把它从第一压到第三。

### 2.4 局限与下一步：必须先有 top-N、模型 ~1GB、O(N) per-pair 成本、小池子天花板

本段做对了什么 — 用 BGE-reranker-base cross-encoder 把 s06 召回的 top-N 做 token 级精排,在 bi-encoder 的"向量平均"糊弄之上补一层"query+chunk 同看"的精确打分,排序变化肉眼可见。


- **必须先有 top-N 召回**：cross-encoder 不能直接对百万级文档跑（O(N) BERT forward 太贵）。生产里典型流程是 bi-encoder 召回 ~200 候选 → cross-encoder 精排 → 取 top-5 给 LLM；本节只演示精排这一步。
- **模型文件 ~1GB**：BGE-reranker-base 第一次跑会从 HuggingFace 下载约 1GB 模型权重；网络慢的话要等几分钟。生产部署通常提前 `huggingface-cli download` 或用模型仓库的 CDN。
- **O(N) per-pair 成本**：cross-encoder 一次只看 1 个 `(query, chunk)` 对，不复用任何计算。N 个候选 = N 次 BERT forward ≈ N × 3ms；N=100 大概 300ms-1s，N=1000 直接 3-10s 不可接受。和 bi-encoder 的"一次编码、千万次 ANN"完全相反。
- **小池子的天花板**：如果 bi-encoder 召回阶段就漏了真正相关的 chunk，cross-encoder 也救不回来 —— 精排只能重排已有候选。所以召回（recall）必须先高，再谈精排（precision）。


- `ModuleNotFoundError: No module named 'FlagEmbedding'`：BGE-reranker 依赖；`pip install FlagEmbedding` 兜底；离线环境先 `pip download FlagEmbedding` 到本地、`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 强制走本地缓存。
- `OSError: [E050] Can't find model 'BAAI/bge-reranker-base'`：HuggingFace Hub 不可达；构建镜像时预下载模型到 `~/.cache/huggingface/hub/`，或 `HF_ENDPOINT=https://hf-mirror.com` 走国内镜像。**模型 ~1GB，首次下载慢**。
- `UnicodeEncodeError: 'gbk' codec can't encode character`：Windows 控制台编码问题，跑前 `set PYTHONIOENCODING=utf-8`（s05 / s06 / s07 同问题）。
- `rerank_score > 1` 或 `< 0`：`FlagReranker(..., normalize=False)` 默认输出是 logits 不是概率，设 `normalize=True` 让它映射到 `[0,1]`。
- `rerank 跑 5 分钟不出结果`：`_reranker()` 没缓存住 / 每次新进程——检查 `@lru_cache(maxsize=1)` 是否还在，或者是不是被多线程调用（lru_cache 不跨进程）。

---

## 三、核心函数一览

| 函数 | 文件 | 输入 | 输出 | 一句话解释 |
|---|---|---|---|---|
| `_embed_model()` / `_embed(texts)` | `c01_cross_encoder_rerank.py` | `list[str]` | `list[list[float]]` | `@lru_cache(maxsize=1)` 加载本地 BGE-small-zh-v1.5;`normalize_embeddings=True`(跟 s04 / s05 / s06 同款) |
| `_cosine(a, b)` | `c01_cross_encoder_rerank.py` | `(list[float], list[float])` | `float` | 两个已 L2 归一化向量的内积(cosine ≡ inner_product) |
| `_reranker()` | `c01_cross_encoder_rerank.py` | — | `FlagReranker` | `@lru_cache(maxsize=1)` 加载 `BAAI/bge-reranker-base`,`use_fp16=False`;同一进程只下载、加载一次 |
| `rerank(query, hits, top_k=3)` | `c01_cross_encoder_rerank.py` | `(str, list[dict], int)` | `list[{text, source, page, chunk_id, dense, bm25, score, rerank_score}]` | 把 s06 的 hits 喂给 cross-encoder 重打分,按 `rerank_score` 降序取前 k,保留 `score`(s06 混合分)供前后对比 |
| `_hybrid_topk(docs, query, query_vec, dense_score_fn, k, alpha)` | `c01_cross_encoder_rerank.py` | `(list, str, list, callable, int, float)` | `list[{text, source, page, chunk_id, dense, bm25, score}]` | 复制 s06 code_02 的 hybrid_topk 公式(`α * vec + (1-α) * bm25_norm`);self-contained 跑通 BM25+dense 召回 → rerank 对比 |
| `main()` (code_01) | `c01_cross_encoder_rerank.py` | — | 打印 BEFORE/AFTER rerank top-3 | code_01 演示入口,默认 query `"内存"`(EOFError 时兜底) |

---

## RAGFlow 实现

RAGFlow 的重排序在 `rag/llm/rerank_model.py`：抽象 `RerankModel.Base`，provider 包括 BGE-reranker / Cohere / Jina / 自部署 cross-encoder，统一签名 `rerank(query: str, docs: list[str], top_k: int) -> list[(doc, score)]`。`.env` 的 `RERANK_PROVIDER` 决定用哪个。

**设计取舍**：与 embedding 路由同样的 provider 抽象，避免散弹式判断。同时 `top_k` 是 rerank 后的最终返回数（不是 cross-encoder pair 数）——cross-encoder 要对 query + N 个 doc 算 N 次相似度（O(N）），不是 O(N²）。

详细摘录与 5-15 行 "为什么这样写" 的分析见 [`docs/reference/ragflow-notes/rerank.md`](../docs/reference/ragflow-notes/rerank.md)。

---

## 选型速记

### 主流 rerank 策略速览

下面这张表把社区常用的几类 rerank 策略按"信号维度 / 推理成本 / 是否需要训练 / 适用场景"列出来：

| 策略 | 信号维度 | 推理成本 | 是否需要训练 | 适用场景 |
|---|---|---|---|---|
| **RRF**（s06 末段） | 2+（排名倒数） | 极低 | 否 | Milvus `RRFRanker` / 多通道排名融合 |
| **Cross-Encoder**（本章） | 1（query-doc pair） | 高（N 次 BERT forward） | 是（训练 cross-encoder） | Top-K 精排、本教程 MVP |
| **ColBERT** | token 级 MaxSim | 中（向量点积，不拼接） | 是（训练 ColBERT） | 精度/效率折中、大规模检索 |
| **LLM-as-rerank (RankLLM)** | prompt + LLM 推理 | 高（token 计费 + 远端） | 否（prompt 工程） | 高价值语义理解、多语言 |
| **RAGFlow 多 provider** | 1（统一归一到 [0,1]） | 高（依 provider） | 否（provider 接管） | 生产 / per-query 成本 tier |

我们的 toy `rerank` 在信号维度上只占第二行——**cross-encoder 精排**；生产代码用 `RerankModel.Base` 把第二到第四行都包成同一个接口，租户按"成本 vs 精度"选 provider。

- **教学 / 快速原型 / 离线可复现** → 本地 cross-encoder（本教程，`BAAI/bge-reranker-base`），无 API key、~1GB 一次下载、CPU 可跑；
- **生产中文场景** → `BAAI/bge-reranker-v2-m3` 或 `bge-reranker-large`，v2-m3 多语言、large 中文更准；
- **生产英文场景 + 预算足** → Cohere Rerank / Voyage Rerank，远端 API、付费、按 query 计费；
- **生产 + 极致精度 + 接受 LLM 成本** → RankLLM（GPT-4 / Claude zero-shot rerank），按 token 计费、latency 高；
- **要先看清每个边界再选** → 用本章 code_01 把 `BAAI/bge-reranker-base` 和 `BAAI/bge-reranker-v2-m3` 各跑一次，对比 top-3 的 `chunk_id` 是否重叠——这是最简单的"rerank model A/B"实验。

### 扩展指南

加一种 rerank 策略（ColBERT / RankLLM / Cohere API）只要三步：

1. 写一个 `colbert_rerank(query, hits, top_k=3)` 或 `cohere_rerank(query, hits, top_k=3, api_key=...)`，签名和 `rerank` 一致；
2. 在 `main()` 里按 `RERANK_MODE` env 选 rerank 函数；
3. 给代码文件 README 加一段"它跟 BGE-reranker 比，赢在哪 / 输在哪"的对照。

不要在 `rerank` 里写 `if mode == "bge": ... elif mode == "colbert": ...` 之类分发——它会污染单一职责。`rerank` 只懂 BGE，`main()` 懂全 rerank 模式。本章 MVP 只跑 BGE，但接口形状留好了。

---

## 思考题

1. **如果召回了 100 个、rerank 要跑多少对？**
2. **为什么 s07 没有第二段 failure mode 而 s02/s03 有？**
3. **如果改用 LLM-as-reranker，prompt 设计要点是什么？**

（答案见文末「思考题答案」）

---

## 思考题答案

### Q1. 如果召回了 100 个、rerank 要跑多少对？

**100 对**（1 query × 100 candidates）。Cross-encoder 的 query+chunk 是 1 对 1，不是 1 对 N。100 个 chunk 就是 100 对，O(n) 不是 O(n²）。

注意原 plan/brief 这里写的是 10000 对，那是因为把 BM25 + 向量那种双塔检索的笛卡尔积混淆进来了——cross-encoder 没有 N×N 那回事。

时间粗算：单对 cross-encoder forward 在 CPU 上 ~3-5ms，GPU 上 ~1-2ms。

- top-10：~30-100ms，可以接受；
- top-100：~300ms-1s，还能撑；
- top-1000：~3-10s，**线上不可接受**。

这就是为什么生产 RAG 系统都把"召回量"压到 cross-encoder 能吃的范围（一般 50-100），再多就让粗召回用**更便宜的近似**顶住：

- 向量召回用 **IVF / HNSW 索引**（s05 的 Chroma 就是 HNSW），把 O(N) 全扫变成 O(log N) 近邻；
- BM25 用**倒排表 + 跳表**而不是线性扫；
- 召回完了再用 cross-encoder 在小池子上精排。

生产代码把这条原则硬编码进了 `_rerank_window`：

```python
window = math.ceil(64 / page_size) * page_size
if top > 0:
    window = min(window, math.ceil(top / page_size) * page_size)
```

——粗召回池子永远卡在 ~64 候选，配 `rerank_mdl` 时更小（`top` 参数封顶）。这是"召回 1000 个再 rerank"和"召回 64 个再 rerank"的工程差距。

### Q2. 为什么 s07 没有第二段 failure mode 而 s02/s03 有？

答：s07 的失败模式不是"代码跑不通"而是"rerank 在某些边界条件下退化成 bi-encoder"——比如 (a) 召回量太小（<5）时 rerank 跟 dense 排序几乎重合，看不出差别；（b) `bge-reranker-base` 配中文 query 时精度比 `v2-m3` 低 ~10%；（c) `top_k` 大于召回量时直接返回原序。这些"边界模式"已经在 §一 用文字讲清楚，不需要单独的 failure-mode 段跑——s02（loader edge cases）/ s03（chunker 边界）是"代码逻辑分支"，s07 是"超参/模型选择"，**叙述载体不一样**。

### Q3. 如果改用 LLM-as-reranker，prompt 设计要点是什么？

三条 — (a) **候选摘要 + 编号**，不要塞完整 chunk（LLM 上下文有限）；（b) **明确输出格式**，如 `Doc: 9, Relevance: 7`、`Doc: 3, Relevance: 4`，强结构化输出便于解析；（c) **给"无关文档"留位**，提示词里写明"请不要包含与问题无关的文档"，LLM 才不会硬凑够 k 个返回。

### 那为什么不直接让 LLM rerank？

理论上 GPT-4 / Claude 看 100 个 chunk 给出 top-3 是终极方案，实际上：

- **延迟**：100 个 chunk + 一个 query 进 GPT-4-128k 一次 ~2-10s；
- **钱**：输入 token × 100 chunk 平均 500 token ≈ 50K token / 次 ≈ $0.5-$1（GPT-4o 价位）；
- **不稳**：LLM 输出的"前 3 名"格式需要后处理解析，且对长 chunk 容易"前部偏置"（chunk 开头被看得多、结尾被看得少）。

所以生产代码把 LLM 风格的 rerank 做成**多 provider 抽象**（Jina / Cohere / Voyage / Qwen / 本地 HF cross-encoder），让租户按预算选；cross-encoder 的"快 + 准 + 便宜"组合通常是默认推荐——我们的 MVP 也是这个选择。
下一章 — 这一节把"召回 → 排序 → 生成 → 服务化"中的某一环跑通,留下 +1 章填下一档的实现;每加一档,缺失上层就越明显,直到 s12 把所有环节收敛到 FastAPI 服务。

> 排错事项（`ModuleNotFoundError` / OSError / `rerank_score` 取值 等）见 `c01` / `c02` 的 `### 局限与下一步`。