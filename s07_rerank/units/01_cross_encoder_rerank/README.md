# s07 / Unit 01 — Cross-encoder 重打分（BGE-reranker）

> 由浅入深第 1 步：把 s06 召回的 top-N 候选再过一道 cross-encoder，按"query+chunk 拼起来看"的相关性重排序。  
> 这是 bi-encoder（双塔）召回之后的两阶段精排：精排贵但只对小池子跑，所以准。

## 这是什么

`s07_rerank/units/01_cross_encoder_rerank/code.py` 把 s06 混合召回吐出的 top-N（默认 N=10）候选，跟原始 query 拼成 `[query, chunk_text]` 对，喂给 `FlagReranker("BAAI/bge-reranker-base")` ——一个 BERT 类 cross-encoder 模型。它对每对 `(query, chunk)` 做一次完整 forward，让 BERT 的 self-attention 同时看到两端、做 token 级 cross-attention，输出一个归一化到 `[0,1]` 的相关性分。我们按这个分降序取 top-3。

`rerank(query, hits, top_k=3)` 函数吃 s06 风格的 hits list（每条带 `text/source/page/chunk_id/score`），吐按 `rerank_score` 降序的前 K 条；每个返回项保留原始 `score`（混合召回分），新增 `rerank_score`（cross-encoder 分）。模型用 `@lru_cache(maxsize=1)` 缓存，同进程只下载、加载一次。

`main()` 跑一个完整的对比：BEFORE 是 s06 混合召回的 top-3（按 `alpha*vec + (1-alpha)*bm25_norm` 排），AFTER 是 cross-encoder 精排后的 top-3 —— 你会看到排序变化，因为 cross-encoder 看到的"查询词 vs 文档词"的精确匹配信号，比双塔向量平均值敏锐得多。

## 跑起来

```bash
python s07_rerank/units/01_cross_encoder_rerank/code.py
# 问: 内存
```

输出示例（首次跑会下载 BAAI/bge-reranker-base ~1GB）：

```
loaded 28 chunks from samples/
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

注意第 1 条 vs 第 2 条：混合召回把 vec=#1 的"内存 32 × DDR4 ..."排第一（配置表，纯字面）但 cross-encoder 觉得它只有 0.644（因为正文是配置表，"内存"只是表里一行），而"2 内存"章节虽然 vec 只有 0.905，rerank 却给到 0.954。这就是 cross-encoder 比 bi-encoder 准的地方：它能看到具体词而不是被一个向量平均值糊弄。

## 它做对了什么

- **cross-encoder 看到 query+chunk 联合信号**：bi-encoder 把 query 和 chunk 各自编成向量再算相似度，丢失了 token 级对齐；cross-encoder 把 `(query, chunk)` 当一个序列让 BERT 一次性 cross-attend，能直接判别"这个词是不是在响应那个查询"。
- **bi-encoder 召回 + cross-encoder 精排 = 两阶段漏斗**：bi-encoder 编码一次、向量化、ANN 召回千级候选 O(log N) 廉价；cross-encoder 在小池子（50-100）上跑 O(N) 精排准但贵。组合起来既快又准。
- **不重编码**：rerank 不重新生成向量，只是在已有候选上重打分 —— 整个精排阶段不需要 GPU 重跑 embed。

## 它做错了什么

- **必须先有 top-N 召回**：cross-encoder 不能直接对百万级文档跑（O(N) BERT forward 太贵）。生产里典型流程是 bi-encoder 召回 ~200 候选 → cross-encoder 精排 → 取 top-5 给 LLM；本单元只演示精排这一步。
- **模型文件 ~1GB**：BGE-reranker-base 第一次跑会从 HuggingFace 下载约 1GB 模型权重；网络慢的话要等几分钟。生产部署通常提前 `huggingface-cli download` 或用模型仓库的 CDN。
- **O(N) per-pair 成本**：cross-encoder 一次只看 1 个 `(query, chunk)` 对，不复用任何计算。N 个候选 = N 次 BERT forward ≈ N × 3ms；N=100 大概 300ms-1s，N=1000 直接 3-10s 不可接受。和 bi-encoder 的"一次编码、千万次 ANN"完全相反。
- **小池子的天花板**：如果 bi-encoder 召回阶段就漏了真正相关的 chunk，cross-encoder 也救不回来 —— 精排只能重排已有候选。所以召回（recall）必须先高，再谈精排（precision）。

## 对照 ragflow 怎么做的

ragflow 的 `_rerank_window(page_size, top)`（见 [`ragflow_notes/hybrid_retrieval.md`](../../../../ragflow_notes/hybrid_retrieval.md)）解决的是"分页和块拉取不对齐"这个真实生产 bug：

```python
window = math.ceil(64 / page_size) * page_size   # 向上取整到 page_size 整数倍
```

它强制把候选窗口**向上取整到 page_size 的整数倍**，让后端一次返回 `RERANK_LIMIT` 大小的 block、前端在内存里切片 `begin = global_offset % RERANK_LIMIT`。一个窗口公式同时控制 block fetch 和 page slice 两个偏移量，永远对齐。

对照本 MVP：MVP 是"固定 topk=10"，根本没有分页概念。但本单元演示的"cross-encoder 在 ~50-100 候选上跑"的精排模式，正是 ragflow `_rerank_window` 把 64 候选作为目标池子的来源 —— 它把 ~64 当成 cross-encoder / LLM rerank 能吃的最大池子，超过就要再向上取整到 page_size 的整数倍保证分页对齐。本单元的 top-10 是 64 池子的子集，符合 ragflow 同样的"小池子精排"原则。

参考：[`ragflow_notes/hybrid_retrieval.md`](../../../../ragflow_notes/hybrid_retrieval.md)

## 思考题

- **如果召回了 100 个候选，rerank 要跑多少对？**
  答：100 对（1 query × 100 candidates）。Cross-encoder 的 query+chunk 是 1 对 1，不是 1 对 N。100 个 chunk 就是 100 对 BERT forward，O(N) 不是 O(N²)。100 对 × ~3ms/对 ≈ 300ms-1s；如果召回 1000 个直接 3-10s，线上不可接受。所以**召回量要压到 cross-encoder 能吃的范围**（一般 50-100），再多就让粗召回用更便宜的近似（量化向量、IVF 索引）顶住。

- **cross-encoder 为什么能比 bi-encoder 准？**
  答：bi-encoder 把 query 和 chunk 各自独立编码成单个向量，查询时只能用余弦相似度比较；这意味着 query 里"内存"和 chunk 里"内存"虽然字面一致，但 bi-encoder 也只看到它们各自 768 维向量的某种平均，无法判断"内存"这个词在 query 和 chunk 里的"重要性"是否一致。Cross-encoder 把 `[CLS] query [SEP] chunk]` 当一个序列送进 BERT，self-attention 能直接让 query token 关注 chunk 里所有 token，判别出 chunk 里"内存"只是配置表里的一个属性 vs. 是章节标题 —— 这就是 0.954 vs 0.644 的差距来源。