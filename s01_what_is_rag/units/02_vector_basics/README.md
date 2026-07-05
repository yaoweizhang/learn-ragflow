# s01 / Unit 02 — 词袋向量 + 余弦相似度

> 由浅入深第 2 步：**向量检索的概念**——把段落和问题都转成"向量"，按相似度排序。  
> 本单元用词袋 (bag-of-2-grams) + 手写余弦，省去 embedding 模型下载，让 s01 自包含。
> 后面 s04 用 BGE 真向量替代这套玩具；s05 用 Chroma 持久化索引。

## 这是什么

1. 把每段切成 2-gram（中文每 2 字 1 个 token）；
2. 全部 token 组成词表 `vocab: {token: index}`;
3. 每段转成词频向量 `vec = [词频 in vocab]`；
4. 问题转同样形状的向量；
5. 余弦相似度 = "问题向量" 与 "段落向量" 的夹角；
6. 按分排序返回 Top-3。

## 跑起来

```bash
python s01_what_is_rag/units/02_vector_basics/code.py
# 问点啥: 披露
```

输出示例（按相似度分排序的前 3 段）：

```
Top-3 与你的问题最相关的段落（按向量余弦排序）：
[1] score=0.342
    相关信息披露详见财务报表附注三(二十五)、五 (二)1 及十五(二)...
[2] score=0.215
    ...
```

## 与 unit 01 的差别

- **能排了**：Top-3 而不是"第一个命中"。
- **可量化**：分数范围 [0, 1]，可以选阈值（虽然这一版没做）。
- **仍然朴素**：词袋向量维度爆炸（每段可能 100+ unique token）、sparse；不像 BGE 是 dense 512 维真语义向量。这就是为什么要换模型。

## 手写余弦 = 真余弦

为了避免 NumPy 依赖（chapter 1 应零依赖），我们展开公式手算：

```
cosine(a, b) = dot(a, b) / (norm(a) * norm(b))
             = sum(a[i]*b[i]) / sqrt(sum(a[i]^2)) / sqrt(sum(b[i]^2))
```

这跟 NumPy 的 `np.dot / (np.linalg.norm(a) * np.linalg.norm(b))` 在数值上一致。生产里只是用 NumPy / torch 利用 SIMD 加速。

## 对照 ragflow 怎么做的

RAGFlow 的向量检索走：
- `Embedding Model → Dense Vector Index` (`ragflow_notes/embedding_routing.md`)
- `Dealer.search` 里 `FusionExpr("weighted_sum", ..., {"weights": "0.05,0.95"})` 把 BM25 和向量加权融合
- （详见 [`ragflow_notes/hybrid_retrieval.md`](../../../../ragflow_notes/hybrid_retrieval.md)）

本单元对应的是 **Dense Vector** 部分（不带 BM25 fusion，不带 rerank）。完整的"双塔"在 s06。

## 思考题

**如果两段都包含"披露"两次，词袋向量会怎么算？它分得开"摘要式披露"和"详细披露"吗？**

提示：词袋丢弃了**位置信息**和**上下文**。生产里要解决就是 s04 (真语义 embedding) + s07 (cross-encoder 精排)。
