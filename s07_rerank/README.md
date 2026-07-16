# s07 重排序 (Rerank) — Cross-encoder 把 s06 的 top-N 重排成 top-K

[上一章 s06 → · 下一章 s08 → ... → s12]

> *"bi-encoder 召回了对的 chunk"和"排序把对的 chunk 顶到第一"之间隔着一道悬崖 — 这道悬崖由 latency cliff / 语言错配 / LLM-as-rerank 成本 tier 三类问题堆起来"*
>
> **链路位置**: 在线检索链路的精排器 (s06 召回 → **s07 精排** → s08 生成)
> **代码文件**: cross_encoder_rerank.py

> 环境准备: 见 root README §快速开始 — `pip install FlagEmbedding` + 首次跑下载 `BAAI/bge-reranker-base` (~1GB, 本地缓存走 `~/.cache/huggingface/hub/`)

---

## 问题

s06 用 BM25 + dense cosine 拉回 top-N 候选 — 但"召回"和"排序准"之间隔着一道悬崖:**bi-encoder 把 query 和 chunk 各自独立编码成一个向量,再做 cosine**, 这种"双塔"结构**快但只看到向量层面的语义接近度**, 没办法捕捉"查询词"和"chunk 里某个具体词"的精确匹配。这是为什么工业 RAG 几乎都把"双塔召回 + cross-encoder 精排"做成两阶段 — 便宜的 bi-encoder 拉宽,贵的 cross-encoder 收紧。

真实样本上,这条悬崖由三类典型问题堆起来:

**第一, latency cliff**。cross-encoder 是 O(n) 次 BERT forward — 每一次都对 `(query, chunk)` 拼好的完整序列跑一次完整 BERT。`top_k=10` 大约 100-300ms (CPU) / 30-50ms (GPU); `top_k=100` 直接 300ms-1s (CPU); `top_k=1000` 直接 3-10s,**线上不可接受**。生产上的标准解法就是两阶段 — 先用便宜的双塔召回 ~100-200 候选,再让 cross-encoder 在小池子上精排;**绝不在 top-1000 上跑 cross-encoder**。

**第二, 语言错配**。`bge-reranker-base` 主要在**英文 MS-MARCO + 英文 Wikipedia** 上训,中文任务上经常"准头不对" — 它会把"内存"和"RAM"按英文语料里的共现模式打分,中文特有的术语对不上。中文场景换 `bge-reranker-v2-m3` 或 `bge-reranker-large` 更稳; 切换只需改 `_reranker()` 里那行字符串。

**第三, LLM-as-rerank 的成本 tier**。还能再叠一层: 拿 rerank 后的 top-N 让 GPT-4 / Claude 做"哪个最相关"判断。每次调用都花 token 钱 + 远端 HTTP 延迟,latency 高得吓人。本次 MVP 不接 — 教学仓库只要把"cross-encoder 比 bi-encoder 准"这件事讲清楚就够了;**生产系统需要按"成本 vs 精度"做 tier 选型**(本地 BGE → 远端 BGE-v2-m3 → Cohere Rerank → GPT-4 zero-shot)。

把这三类问题合起来看,**`rerank(query, hits, top_k=3)` 调起来不到 20 行,但把它扔进真实样本就会发现"bi-encoder 召回了对的 chunk"和"排序把对的 chunk 顶到第一"之间隔着一道悬崖**。s07 的目标不是解决这道悬崖,而是把它的边界显式暴露出来,让你看到 toy rerank 的脆弱性 — 它只能在小池子上精排,且只能排 top-N 召回里"已经被召回"的 chunk,**如果 bi-encoder 召回阶段就漏了真正相关的 chunk,cross-encoder 也救不回来**。

---

## 解决方案

s07 用 **一个脚本** 把 cross-encoder 精排跑通,演示"召回没变但排序变了"的对照效果。

```
                s06 hits (top-N)                          s08 hits (top-K)
                BM25 + dense 融合后                       BGE-reranker 重打分后
                ┌─ #1 [bm25冠军] score=0.795 ─┐           ┌─ #1 [主题真沾边] rerank=0.664 ─┐
                ├─ #2 [vec=0.552]    score=0.736 ─▶  cross-encoder N 次 BERT forward  ─▶ ├─ #2 ...        rerank=0.550 ─┤
                ├─ #3 ...                score=0.726      (query, chunk) 拼 → [0,1]    ├─ #3 ...        rerank=0.527 ─┘
                └─ ... top-N ...                          按 rerank_score 降序取 K
                    段级语义对齐(粗)                       token 级 cross-attention(精)
```

| 脚本 | 解决什么 | 留下什么局限 | 何时用 |
|---|---|---|---|
| `cross_encoder_rerank.py` | FlagReranker 把 s06 召回的 top-N 重打分,按 token 级 cross-attention 取 top-K | 必须先有 top-N 召回;~1GB 模型下载;O(N) per-pair;小池子天花板 | 在线检索链路精排 / 提升 top-1 精度 / s08 喂 LLM 前的最后过滤 |

脚本的关系是一条**主干**:``cross_encoder_rerank.py`` 把"s06 top-N → cross-encoder → 重排 top-K"做出来,在双塔的"向量平均"糊弄之上补一层"query+chunk 同看"的精确打分,**BEFORE/AFTER 对比肉眼可见 — 排序变了,但召回的那批 chunk 没变**。**每一步的局限,都是后续章节(s08 起)要解决的入口**: s08 把精排后的 top-K 拼进 prompt 喂 LLM;s12 在 FastAPI 服务里把整个链路固化成 production endpoint。

---

## 代码 1: Cross-encoder 精排 (BGE-reranker) ([cross_encoder_rerank.py](cross_encoder_rerank.py))

入口:[`cross_encoder_rerank.py`](cross_encoder_rerank.py)

把 s06 召回的 top-N 候选再过一道 cross-encoder,按"query+chunk 拼起来看"的相关性重排序。
这是 bi-encoder(双塔)召回之后的两阶段精排: 精排贵但只对小池子跑,所以准。

### 工作原理

**做一件事**: 把 s06 召回的 top-N 命中,用 `FlagReranker("BAAI/bge-reranker-base")` 对每条 `(query, chunk)` pair 做一次完整 BERT forward,输出归一化的 `[0,1]` 相关性分,按新分降序取 top-k。

**N 步**:
1. 内联 `pypdf` + `python-docx` 加载 `samples/`,再按 s03 同款 500 字符 cap 中英句界切块 — 拿到 34 个 chunk
2. 用 s05 的 chroma 索引拿向量(reuse 已有 / 不存在则重建),把 query 编码成 BGE 512 维向量
3. `_hybrid_topk` (内联 s06 hybrid_fusion.py 的 `α*vec + (1-α)*bm25_norm` 公式) 取 top-10 候选 — `α=0.5` 等权融合,演示 dense + BM25 都能进精排
4. `_reranker()` 用 `@lru_cache(maxsize=1)` 加载 `FlagReranker("BAAI/bge-reranker-base", use_fp16=False)` — 同一进程只下载加载一次,首次跑会下 ~1GB
5. `compute_score(pairs, normalize=True)` 对每个 `[query, chunk_text]` 对跑一次 BERT forward,输出 `[0,1]` 相关性分 — **O(n) 次推理,不是 O(n²)**
6. 按 `rerank_score` 降序取 top-3,每条命中同时保留原始 `score`(s06 混合分)、`dense`、`bm25`、`rerank_score`,方便 BEFORE/AFTER 对比
7. 打印 BEFORE(混合召回 top-3) vs AFTER(cross-encoder 精排 top-3),**两条榜的 chunk 集合可能完全一致,但排序变了**

```python
# 中间片段: FlagReranker compute_score — 一次 BERT forward / 对
@lru_cache(maxsize=1)
def _reranker():
    from FlagEmbedding import FlagReranker
    return FlagReranker("BAAI/bge-reranker-base", use_fp16=False)


def rerank(query: str, hits: list[dict], top_k: int = 3) -> list[dict]:
    """对 s06 召回的 hits 做 cross-encoder 精排,返回按 rerank 分降序的 top-k."""
    if not hits:
        return []
    rr = _reranker()
    pairs = [[query, h["text"]] for h in hits]
    scores = rr.compute_score(pairs, normalize=True)
    scored = sorted(zip(hits, scores), key=lambda x: -x[1])
    out = []
    for h, s in scored[:top_k]:
        out.append({**h, "rerank_score": float(s)}  # cross-encoder [0,1] 相关分
    return out
```

**完整函数**:

```python
@lru_cache(maxsize=1)
def _reranker():
    """@lru_cache(maxsize=1) 加载 BAAI/bge-reranker-base, use_fp16=False; 同一进程只下载、加载一次."""
    from FlagEmbedding import FlagReranker
    return FlagReranker("BAAI/bge-reranker-base", use_fp16=False)


def rerank(query: str, hits: list[dict], top_k: int = 3) -> list[dict]:
    """对 s06 召回的 hits 做 cross-encoder 精排, 返回按 rerank 分降序的 top-k。

    每个返回项带 `rerank_score` (cross-encoder 的 [0,1] 相关性分, FlagReranker
    normalize=True 归一后的值), 同时保留 `score` (s06 的混合召回分) 和
    原始的 text/source/page/chunk_id。
    """
    if not hits:
        return []
    rr = _reranker()
    pairs = [[query, h["text"]] for h in hits]
    scores = rr.compute_score(pairs, normalize=True)
    scored = sorted(zip(hits, scores), key=lambda x: -x[1])
    out = []
    for h, s in scored[:top_k]:
        out.append({**h, "rerank_score": float(s)})
    return out


def _hybrid_topk(docs, query, query_vec, dense_score_fn, k, alpha):
    """复制 s06 hybrid_fusion.py 的 _hybrid_topk 公式 (alpha * vec + (1-alpha) * bm25_norm)."""
    bm = BM25(docs)
    bm_scores = bm.score(query)
    bm_max = max(bm_scores) if any(bm_scores) else 1.0
    combined = []
    for i, d in enumerate(docs):
        v = float(dense_score_fn(d))
        b = bm_scores[i] / bm_max if bm_max > 0 else 0.0
        combined.append({
            "text": d["text"], "source": d["source"],
            "page": d.get("page"), "chunk_id": d.get("chunk_id"),
            "dense": v, "bm25": bm_scores[i],
            "score": alpha * v + (1 - alpha) * b,
        })
    combined.sort(key=lambda x: -x["score"])
    return combined[:k]
```

### 试一下

```bash
python s07_rerank/cross_encoder_rerank.py
# 问: 内存
```

首次跑会下载 `BAAI/bge-reranker-base` ~1GB,从 HuggingFace Hub 拉模型权重;走 hf-mirror 镜像,正常情况下 5-15 分钟。`EOFError` 自动兜底为 `query='内存'`。30 chunks 上 BEFORE/AFTER 对比:

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

**观察**: BEFORE 第 1 是 `#4 可靠性`(BM25 字面命中,vec=0.590),AFTER 第 1 变成 `#3 应用场景`(rerank=0.664,vec=0.545) — **rerank 分和 s06 的 vec 分不同步**。BM25 把"可靠性章节里顺带提到内存"顶到第一,但 cross-encoder 看到它只有 `rerank_score=0.527`(因为正文主题是"可靠性","内存"只是配置表里一行); 而"应用场景"章节虽然 `dense=0.545`,`rerank_score` 却给到 **0.664** — cross-encoder 看到它把"高密度计算"作为主题,跟 query "内存"的计算密度语义最沾边。**bi-encoder 召回了对的 chunk(BEFORE 也有它), 但 cross-encoder 在 token 级 attention 上看出"#4 章节里只是顺带提一句内存", 该把它从第一压到第三** — 这正是 rerank 的价值。

### 为什么不只写这一种

``cross_encoder_rerank.py`` 用 **cross-encoder 精排** 解决了"排序准"的问题,但**它只在已有召回池子上重排**; 如果 bi-encoder 召回阶段就漏了真正相关的 chunk,cross-encoder 也救不回来 — **召回率是精排的天花板**。其次,`bge-reranker-base` 中文场景精度有限(bge-reranker-v2-m3 或 bge-reranker-large 更稳); 第三,**LLM-as-rerank + 远端 Cohere Rerank 都是 production tier 选项**,本次 MVP 没接。粗排看"召回全不全",精排看"排序对不对",两级串起来才是生产级的检索质量 — s08 把精排后的 top-K 喂给 LLM,看 prompt 模板 + 引用跟踪如何把"召回 → 排序 → 生成"的全链路跑通。

---

## 接下来

s07 是在线检索链路的精排器: ``cross_encoder_rerank.py`` 把 BM25 字面 + dense 语义的 hybrid top-N 用 cross-encoder 重排成 top-K。但每一步都留下脆弱点, 这些是后续章节的填空目标:

- **必须先有 top-N 召回** — cross-encoder 不能直接对百万级文档跑 (O(N) BERT forward 太贵)。生产里典型流程是 bi-encoder 召回 ~200 候选 → cross-encoder 精排 → 取 top-5 给 LLM; 本节只演示精排这一步。
- **模型文件 ~1GB** — BGE-reranker-base 第一次跑会从 HuggingFace 下载约 1GB 模型权重; 网络慢的话要等几分钟。生产部署通常提前 `huggingface-cli download` 或用模型仓库的 CDN。
- **O(N) per-pair 成本** — cross-encoder 一次只看 1 个 `(query, chunk)` 对, 不复用任何计算。N 个候选 = N 次 BERT forward ≈ N × 3ms; N=100 大概 300ms-1s, N=1000 直接 3-10s 不可接受。和 bi-encoder 的"一次编码、千万次 ANN"完全相反。
- **小池子的天花板** — 如果 bi-encoder 召回阶段就漏了真正相关的 chunk, cross-encoder 也救不回来 — 精排只能重排已有候选。所以召回 (recall) 必须先高, 再谈精排 (precision)。

s08 **prompt 生成**: 把 s07 重排后的 top-K 喂进 prompt, 让 LLM 依据这些段落作答 — 召回 + 精排 + 生成 = RAG 全链路的最后一环; 同时 prompt 模板 + 引用跟踪是 LLM 输出"对不对"的关键防线。

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

> 排错事项（`ModuleNotFoundError` / OSError / `rerank_score` 取值 等）见 `cross_encoder_rerank.py` 的 `### 局限与下一步`。