# s01 什么是 RAG

> 本章用 **3 个 unit** 把 RAG 从"零"搭到"能跑"：  
> 朴素关键词 → 词袋向量检索 → 完整 RAG 链路（检索 + LLM）。  
> 每个 unit 独立可跑、自包含；s02-s12 会把这些小玩具逐步换成工业实现。

## 为什么分 3 个 unit

一个真实 RAG 系统 = 文档解析 → 切块 → embedding → 索引 → 检索 → 精排 → prompt → LLM。  
一上来全讲会"只见森林不见树"。s01 的目标是**先把"开卷考试"的直觉讲清楚**，所以拆 3 步：

| Unit | 主题 | 它解决什么 | 对照 RAGFlow |
|---|---|---|---|
| [01_naive_keyword](./units/01_naive_keyword/README.md) | 朴素子串匹配 | "检索"是什么意思 | — |
| [02_vector_basics](./units/02_vector_basics/README.md) | 词袋向量 + 余弦 | "排序"怎么算 | s04 (BGE 真 embedding) |
| [03_augmented_llm](./units/03_augmented_llm/README.md) | 检索 + Prompt + LLM | "生成"是怎么基于资料 | s08 (工业 prompt + 拒答) |

跑完 3 个 unit，你就有了 RAG **完整链路**的最小版——后面 11 章的工作是把这 3 个 unit 里的"玩具"逐个替换成工业级实现。

## 跑起来

```bash
cd learn-ragflow

# unit 1：子串匹配
python s01_what_is_rag/units/01_naive_keyword/code.py

# unit 2：词袋向量 + cosine
python s01_what_is_rag/units/02_vector_basics/code.py

# unit 3：完整链路（无 key 时只打印 prompt；有 key 时真调 LLM）
python s01_what_is_rag/units/03_augmented_llm/code.py
LLM_API_KEY=sk-xxx python s01_what_is_rag/units/03_augmented_llm/code.py
```

每个 unit 的 README 里都有 **跑起来 / 它做对了什么 / 它做错了什么 / 对照 ragflow** 四个固定段落。

## 与 RAG 综述的对接

如果你对 RAG 的整体地图还不熟，先看 [`../docs/00_introduction/01_what_is_rag.md`](../docs/00_introduction/01_what_is_rag.md) —— 它先讲 RAG 是什么、为什么、和长上下文/微调的取舍，再串到 12 章大纲。

## 三个 unit 的递进关系

```
unit 01 (fake_rag)              unit 02 (cosine top-k)              unit 03 (RAG pipeline)
┌────────────────┐              ┌───────────────────────┐           ┌─────────────────────┐
│ 段落列表       │              │ 段落 → 词频向量       │           │ 同 unit 02 检索     │
│       │        │              │       │               │           │       │             │
│       ▼        │              │       ▼               │           │       ▼             │
│ 子串匹配第一段  │  ──演进──▶   │ cosine 排序 top-k     │ ──演进──▶ │ top-k 拼 prompt     │
│       │        │              │       │               │           │       │             │
│       ▼        │              │       ▼               │           │       ▼             │
│ 直接返回       │              │ 直接返回              │           │ LLM 生成答案        │
└────────────────┘              └───────────────────────┘           └─────────────────────┘
   没排序                       有分数                                真正"开卷"
   没语义                       sparse 语义                          完整链路
```

两条主轴线：

- **检索质量** —— unit 01 (子串) → unit 02 (sparse 向量) → s04 (dense BGE) → s06 (BM25+dense 融合)；
- **生成质量** —— unit 01 (直接返回) → unit 03 (prompt+LLM) → s07 (rerank 后喂 LLM) → s08 (工业 prompt 模板)。

## 对照 ragflow

每 unit 的 README 第 5 节都对应 `ragflow_notes/` 里相应章节：

| Unit | 引用 |
|---|---|
| unit 01 | [`ragflow_notes/hybrid_retrieval.md`](../ragflow_notes/hybrid_retrieval.md) |
| unit 02 | [`ragflow_notes/embedding_routing.md`](../ragflow_notes/embedding_routing.md) |
| unit 03 | [`ragflow_notes/prompt_templates.md`](../ragflow_notes/prompt_templates.md) |

## 思考题

完成 3 个 unit 之后试着回答：

**把 unit 02 的词袋换成真 BGE embedding 后，"披露" vs "关联方披露" 的检索顺序会变吗？为什么？**

提示：词袋靠 token 重叠，BGE 靠语义接近。后者会区分"披露"是泛指还是特指关联方，因为 embedding 见过大量金融语料。详细答案见 [`./thinking_answers.md`](./thinking_answers.md)。