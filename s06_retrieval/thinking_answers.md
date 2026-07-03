# s06 思考题

## 如果 alpha=0 应该退化成什么？alpha=1 呢？

`alpha` 在 `hybrid_search` 里是向量分的权重，所以：

- **alpha=0** → 完全忽略向量分，只用 BM25 打分排序。退化成**纯关键词
  召回**。适合"应付账款"、"内存容量"这种字面术语查询；对同义词/改写
  无能为力（"营收"找不到"营业收入"）。实测：

  ```
  query=内存 alpha=0.0
  [server_whitepaper.pdf#24] score=1.000 | 图12 内存标识 ...
  [server_whitepaper.pdf#2]  score=1.000 | 2 内存 ...
  [server_whitepaper.pdf#24] score=0.887 | 4.2 内存 ...
  ```

- **alpha=1** → 完全忽略 BM25，只用余弦相似度排序。退化成**纯向量
  召回**。适合改写/口语化查询（"机器为啥跑得慢"→ 找到"内存故障处理"）；
  对完全一致的关键词反而容易排不到第一。实测：

  ```
  query=内存 alpha=1.0
  [server_whitepaper.pdf#24] score=1.000 | 图12 内存标识 ...
  [server_whitepaper.pdf#24] score=0.932 | 21 具体可选购 ...
  [disclosure.docx#None]      score=0.887 | 7. 存货
  ```

  （第三个 hit 已经不是"内存"字面了，是向量把"内存"和"7.存货"拉得太近
  ——这正是 alpha=1 的副作用：牺牲字面精度换语义召回。）

## 实际怎么挑 alpha？

靠**带标签的交叉验证集**——一批 query + 人工标注的相关 chunk id，跑一
遍 sweep（0.0, 0.1, ..., 1.0），看每个 alpha 下的 recall@k / MRR。
经验上 0.3 ~ 0.7 都合理，极端值只在特定场景才合适：
- 0.0~0.3：术语密集的财务/法务/医疗场景；
- 0.7~1.0：开放域问答、闲聊、文档摘要型场景。

RAGFlow 干脆不存一个 alpha 常数——把 `vector_similarity_weight` 做
成 API 入参，每类 query 由调用方传不同值。本 MVP 用 `alpha=0.5` 兜底
是因为样本小，没法 sweep；真要上线必须按业务场景重新调。