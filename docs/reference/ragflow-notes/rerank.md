# RAGFlow 重排序

## 一句话
RAGFlow 的 rerank 是两阶段流水线：DB 内 `weighted_sum` 粗召回 → 应用侧 `rerank_by_model` 跨编码器精排；可选走多家 provider（Jina / Cohere / Voyage / Qwen / 本地 BGE cross-encoder）。MVP s07 只接 BGE 本地。

## 来源
- 仓库：https://github.com/infiniflow/ragflow
- 模块：`rag/nlp/search.py`（`Dealer.retrieval` / `rerank_by_model` / `rerank_with_knn`）、`rag/llm/rerank_model.py`（provider 抽象）
- 关联：本仓库 s07 `cross_encoder_rerank.py`

## 两阶段切换

RAGFlow 的"重排序"实际上是**两阶段**流水线——DB 内 `weighted_sum` 做粗召回（见 hybrid_retrieval.md），然后**可选地**用跨编码器（cross-encoder）或者云 LLM rerank 做精排；只有配了 `rerank_mdl` 才会走第二阶段：

```python
        if rerank_mdl and sres.total > 0:
            sim, tsim, vsim = self.rerank_by_model(
                rerank_mdl,
                sres,
                question,
                term_similarity_weight,
                vector_similarity_weight,
                rank_feature=rank_feature,
            )
        else:
            if settings.DOC_ENGINE_INFINITY:
                # Don't need rerank here since Infinity normalizes each way score before fusion.
                sim = [sres.field[id].get("_score", 0.0) for id in sres.ids]
                sim = [s if s is not None else 0.0 for s in sim]
                tsim = sim
                vsim = sim
            elif settings.DOC_ENGINE_OCEANBASE:
                # OceanBase still returns chunk vectors in the result; use
                # the historical local rerank that depends on them.
                sim, tsim, vsim = self.rerank(
                    sres,
                    question,
                    term_similarity_weight,
                    vector_similarity_weight,
                    rank_feature=rank_feature,
                )
```

`rerank_by_model` 内部把**跨编码器分数** `vtsim` 与**本地词项相似度** `tksim` 再做一次线性加权：

```python
        tksim = self.qryr.token_similarity(keywords, ins_tw)
        # rerank_mdl.similarity() returns scores normalized to [0, 1] for every
        # provider (see RerankModel.Base.similarity), so the blend below stays
        # on a single scale regardless of the configured reranker.
        vtsim, _ = rerank_mdl.similarity(query, docs)
        ## For rank feature(tag_fea) scores.
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        return tkweight * np.array(tksim) + vtweight * vtsim + rank_fea, tksim, vtsim
```

`rerank_mdl` 是一个 `RerankModel.Base` 子类，按用户配置可以是 Jina / Cohere / Voyage / NVIDIA / Qwen / **本地 HuggingFace 跨编码器**（`BAAI/bge-reranker-v2-m3`）——所以"RAGFlow 接入了 LLM-as-rerank"这一说准确说成"接入了多种 rerank provider"，其中云端大模型类（Cohere / Voyage / Qwen）本质上调用远程 LLM 风格的端点，本地类（BGE）才是真正的 cross-encoder。

## 为什么这样写（3 个 bullet）

- **为什么需要两阶段？**
  粗召回（向量+BM25）是**双塔（bi-encoder）**架构——query 和 chunk 各自独立编码再做相似度，**快但粗**；精排（cross-encoder）把 query 和 chunk **拼在一起**让 BERT/Transformer 一次性看两端、做 token 级的 cross attention，准但慢（O(n) 而非向量检索的 O(log n)）。两阶段把"召回广"和"排序准"分开：先用便宜的 bi-encoder 拉 ~64 候选（`RERANK_LIMIT = ceil(64/page_size) * page_size`，见 hybrid_retrieval.md），再让 cross-encoder 在小池子上精排。`rerank_mdl` 配不配决定是否走第二阶段（`if rerank_mdl and sres.total > 0`），这是把"开销留给愿意付钱的人"的开关。

- **为什么 RAGFlow 还接入了 LLM-as-rerank（多 provider 抽象）？**
  `rag/llm/rerank_model.py` 列了 Jina / Cohere / Voyage / NVIDIA / Qwen / 百度千帆 / 本地 HuggingFace 等十几个 provider，这是**业务妥协**而不是技术必须：cross-encoder 在通用英文基准强，但**多语言/垂直领域**（中文金融、中文法律、医疗）未必有开源模型能打；接多家云就能让租户按自己 KB 的语种/领域挑最合适的。云端那一路本质是 **LLM 风格的 rerank 端点**（`documents=[...]`、`relevance_score` 返回），跟本地 cross-encoder 接口同形，所以 `RerankModel.Base.similarity` 能用同一套归一化把它们拍到 `[0,1]` 上，混到 `tkweight * tksim + vtweight * vtsim` 公式里——这是它"既能本地又能云端"的核心技巧。

- **我们这版为什么不接？**
  MVP 只接 BGE 本地 cross-encoder，原因有三：(1) **教学聚焦**——重排序的"为什么"是 cross-encoder 比 bi-encoder 准这一个点，加多家 cloud provider 会冲淡主题；(2) **离线可复现**——BAAI/bge-reranker-base ~1GB 一次下载完就一直在本地，**不需要 API key、不需要网络**，CI / 学习者都能跑；(3) **成本与延迟**——`rerank(query, docs)` 一次推理 N 对就是 N 次 BERT forward，top-10 大概 100-300ms；top-100 就直接 1-3s 了。再叠一层云 LLM rerank 又多一次 HTTP + 按 token 计费，对一个教学仓库不值。RAGFlow 接多 provider 是因为它是**生产框架**，要服务各语种/各预算的租户；我们这是**学习最小集**，一套 cross-encoder 走到底反而更清晰。