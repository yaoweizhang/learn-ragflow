# s01 / Unit 01 — 朴素关键词检索

> 由浅入深第 1 步：**先知道"检索"是什么意思**——不用向量库，不用 LLM，只用最简单的子串匹配。  
> 对应 all-in-rag 章节 1 第一节"什么是 RAG"中的核心直觉：让 LLM "开卷考试"，先得有个"卷"。

## 这是什么

最朴素的检索策略：

1. 把文档读成段落列表；
2. 把用户问题拆词；
3. 找第一个段落里有任意一个词的；
4. 把那段返回。

30 行代码，零外部依赖。

## 跑起来

```bash
python s01_what_is_rag/units/01_naive_keyword/code.py
```

输入对照（用 `samples/disclosure.docx` 实测）：

| 输入 | 输出 |
|---|---|
| `披露` | `相关信息披露详见财务报表附注三(二十五)、五 (二)1 及十五(二)。` |
| `外星人` | `I don't know.` |

## 它做对了什么

- **零依赖**：只用 `python-docx`，适合"我想先看 RAG 在干啥"的入门。
- **结构同构**：返回一段文本给 LLM 这一步，是真 RAG 永远不能省的——后面章节只是把"这段文本"换得更准。

## 它做错了什么（这就是后面章节要解决的）

- **找不到同义词就完蛋**。问"营收"找不到"营业收入"。
- **找到关键词不一定是答案**。段落里出现"应收账款"，但讲的是会计科目列表，不是用户想问的"如何计提坏账"。
- **没有评分**。第一个命中就返回，多个相关时不能排序。

这两个问题分别对应 RAG 系统的两大难题：
- **召回（recall）** → s04 embedding + s06 混合检索
- **精排（precision）** → s07 rerank + s08 prompt

## 对照 ragflow 怎么做的

RAGFlow 在 `rag/nlp/search.py:Dealer.search` 阶段就已经不是朴素子串，而是 **DB 内部 `FusionExpr("weighted_sum", topk, {"weights": "0.05,0.95"})`**——把 BM25 和向量分加权融合，再走应用层 `rerank_with_knn` 叠加 PageRank tag。详见 [`docs/reference/ragflow-notes/hybrid_retrieval.md`](../../../../docs/reference/ragflow-notes/hybrid_retrieval.md)。

教程 s06 会从这 30 行的 naive 一步步演进到这一层。

## 思考题

**如何改成返回 Top-3 候选段？最简单的打分怎么算？**

提示：朴素方案 = 数"命中的关键词数量"。答案见 [`../../thinking_answers.md`](../../thinking_answers.md)。
