# s00 概念速览 — RAG 是什么、为什么、怎么演进

> **章节定位**：全书的概念地图。读完能用 5 分钟向同事讲清楚 RAG 是什么、解决什么问题、什么时候不该用、RAG 这个标签会不会过时。
>
> **章节结构**：本章 3 个 unit 都是概念级，**不需要写代码、不需要 LLM key**；每 unit 配一个 ≤ 50 行的迷你 demo 把概念数值化展示。
> 落到代码层面的细节留到 s01-s12。

---

## 为什么拆 3 个 unit

| 概念问题 | 对应 unit |
|---|---|
| RAG = 检索 + 增强 + 生成？参数化 vs 非参数化知识？ | [unit 01](./units/01_what_is_rag/) |
| 为什么不直接喂长上下文给 LLM？为什么不微调？ | [unit 02](./units/02_why_rag/) |
| RAG 这 6 年演进了什么？教程的 12 章对应演进哪一阶段？ | [unit 03](./units/03_evolution/) |

读完这 3 个 unit，下面 12 章的代码、选型、trade-off 都有了锚点。

---

## Unit 导航

| Unit | 主题 | 入口 | Code |
| --- | --- | --- | --- |
| 01 | RAG 是什么：参数化 vs 非参数化 | [units/01_what_is_rag/README.md](./units/01_what_is_rag/README.md) | [code.py](./units/01_what_is_rag/code.py) |
| 02 | 为什么用 RAG：vs 长上下文、vs 微调 | [units/02_why_rag/README.md](./units/02_why_rag/README.md) | [code.py](./units/02_why_rag/code.py) |
| 03 | 三代演进：初级 / 高级 / 模块化 | [units/03_evolution/README.md](./units/03_evolution/README.md) | [code.py](./units/03_evolution/code.py) |

跑法：

```bash
python s00_concepts/units/01_what_is_rag/code.py
python s00_concepts/units/02_why_rag/code.py
python s00_concepts/units/03_evolution/code.py
```

依赖：`requirements.txt` 已经在仓库根，3 个 unit 都用 Python 标准库 + （可选）`tiktoken`。

---

## 一、为什么先读这一章

RAG 教程铺天盖地，但绝大多数上来就讲"怎么切 chunk"、"怎么选 embedding"——读者看完会写代码，但**讲不清楚为什么这样写**。这一章的目的是先建立心智模型：
- **unit 01**：用 1 句话定义 RAG，用一张 ASCII 图把"离线索引 / 在线问答"两条线拆开。
- **unit 02**：用一张风险分级表 + 一个 token 成本对照告诉你"为什么不是所有问题都该用 RAG"。
- **unit 03**：用三代 RAG 的对照表告诉你"教程的 12 章对应演进哪一阶段"，后面读到 s09 / s10 / s12 不会再一头雾水。

## 二、上手顺序

```bash
cd learn-ragflow
python s00_concepts/units/01_what_is_rag/code.py    # 1 分钟：什么是 RAG
python s00_concepts/units/02_why_rag/code.py        # 1 分钟：为什么用 RAG
python s00_concepts/units/03_evolution/code.py      # 1 分钟：三代演进
```

3 分钟跑完 3 个 unit，再去 s01 开始写第一个 MVP。

## 三、本教程的展开方式

| 章 | 主题 | 在 RAG 工作流中的位置 |
|---|---|---|
| s00 | 概念速览（本文件） | 心智模型 |
| s01 | 什么是 RAG（最小可跑） | 总览 + 玩具实现 |
| s02 | 文档加载 | 离线 → 文档解析 |
| s03 | 文本分块 | 离线 → 切块 |
| s04 | Embedding | 离线 / 在线 → Embedding |
| s05 | 向量索引 | 离线 → 向量数据库 |
| s06 | 混合检索 | 在线 → 召回 + 融合 |
| s07 | 重排序 | 在线 → Rerank |
| s08 | Prompt 与生成 | 在线 → 拼 Prompt + LLM |
| s09 | Agent 与工具 | 在线 → 模块化路由 |
| s10 | GraphRAG | 高级 → 实体关系抽取 |
| s11 | 多模态 | 离线 / 在线 → 表格 / OCR |
| s12 | 部署 | 上线 → FastAPI + Docker |

每一章配 [`docs/reference/ragflow-notes/<topic>.md`](./docs/reference/ragflow-notes/README.md) 对照工业级实现——同一个组件，30 行玩具和 1000 行生产代码的差距在哪里一目了然。

## 四、延伸阅读

- 同款综述：[all-in-rag 教程](https://github.com/datawhalechina/all-in-rag)
- 工业参考：[RAGFlow](https://github.com/infiniflow/ragflow)
- 评估工具：[RAGAS](https://github.com/explodinggradients/ragas)、[TruLens](https://github.com/truera/trulens)
