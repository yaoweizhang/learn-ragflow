# s10 GraphRAG — 关系型问题的最小图查询

## 问题

s06–s09 都把"找相关段落"当终点——但有一类问题，**段落相关性不够**：

- "X 和 Y 有什么关系？"
- "提到 Z 的产品都有哪些？"
- "A 公司投资了谁？被谁投资？"

这些问题的答案**散落在多个段落**里，单独的"最相似段"答不全。
向量检索给你"包含 X 的段"和"包含 Y 的段"，但**不会告诉你 X 和 Y 之间
的那条边**。

经典 RAG 的解法：把段落里**实体之间的指向关系**抽出来，建一张图；查的时候
"先定位起点实体 → 沿着边走 1 跳 / N 跳"。

## 最小解法

`s10_graphrag/code.py`：

1. `extract_triples(text)` 喂一段文字给 LLM，让它吐 `(head, rel, tail)` 三元组列表；
2. `build_graph(triples_list)` 把所有段的三元组合并成 `dict[head] → set[(rel, tail)]`；
3. `query_graph(graph, entity)` 返回这个 entity 出发的所有 (rel, tail) 对——1 跳邻居。

```python
graph = {
    "紫光恒越技术有限公司": {("版权所有", "UNIS Server R3630 G5 服务器技术白皮书"),
                             ("拥有版权", "UNIS Server R3630 G5 服务器技术白皮书")},
    "3.1 技术规格": {("属于", "3 产品规格")},
    ...
}
query_graph(graph, "紫光恒越技术有限公司")
# → [("版权所有", "UNIS Server R3630 G5 服务器技术白皮书"),
#    ("拥有版权", "UNIS Server R3630 G5 服务器技术白皮书")]
```

## 跑起来

```bash
python s10_graphrag/code.py
# 查哪个实体: 紫光恒越技术有限公司
```

实测（MiniMax-M3 over minimaxi.com，samples = server_whitepaper.pdf + disclosure.docx，
只取前 8 个 chunk）：

```
图节点数: 8, 边数: 6

查: '紫光恒越技术有限公司'
  紫光恒越技术有限公司 --版权所有--> UNIS Server R3630 G5 服务器技术白皮书

查: '不存在的实体xyz'
  (无结果)
```

不同次跑节点数 / 边数会小幅抖动（LLM 在 temperature=0 下对长 prompt 仍有少量随机性；
chunk 0/1/2 是封面 + 目录，信息密度低，模型决定抽不抽也有差异）——这是 LLM 抽取的
固有现象，不是 bug。

## 真实世界的问题

1. **实体歧义 / 一物多名**——"海光"和"Hygon"在白皮书里指同一家公司，但 LLM
   会当成两个节点建两条边。MVP 的 `dict` 不会合并。生产里要做 **entity
   resolution / entity linking**：用 embedding 余弦相似度聚类 + LLM 判断
   "这两个名字是不是同一实体"。RAGFlow 的 `entity_resolution.py` 就是这一步。
2. **图爆炸**——一段 500 字里如果模型太"勤快"，能抽 30+ 个三元组；上千段文
   档的图节点轻松上万。治理：① prompt 限定 `entity_types` 白名单（人/公司/产品
   这几类才要）；② 设置"每个段最多抽 N 个三元组"上限；③ 低频节点做剪枝。
   RAGFlow 的 `DEFAULT_ENTITY_TYPES = ["organization", "person", "geo",
   "event", "category"]`（`general/extractor.py:46`）就是这个白名单。
3. **多跳查询慢 / 答案拼接难**——"X 的竞争对手的合作伙伴"是 3 跳。MVP 的
   `query_graph` 只 1 跳；多跳要 BFS 自己写，并且 1 跳图 + LLM 组合就够用，
   3 跳以上必须上 community summary（RAGFlow 的 Leiden 社区检测 + 摘要
   就是干这事的：把"某一组相关实体"压成一段文字，让 LLM 答宏观问题）。

## ragflow 怎么做的

见 [ragflow_notes/graph_extraction.md](../ragflow_notes/graph_extraction.md)。
要点：RAGFlow 把图谱**当 chunk 存进倒排索引**（`knowledge_graph_kwd` 区分
entity / relation / community_report），查询时既可以走"向量召回实体 →
读 `n_hop_with_weight` 字段扩展多跳"（`rag/graphrag/search.py` 的 `KGSearch`），
也可以先跑 hierarchical Leiden 社区检测、给每个社区生成一段 summary 用来答
"文档集主要在讲什么"这种宏观问题。

## 思考题

- **如果两段文字里同一实体名字不同（"产品 A" vs "A 型"）怎么办？**
  答：见 `thinking_answers.md`。