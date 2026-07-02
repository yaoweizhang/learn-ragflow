# Learn RAGFlow：12 章从零到部署的 RAG 教程

> **一句话定位**：12 章从零到部署的 RAG 教程，每章自写 30–80 行 MVP + 对照 RAGFlow 工业级实现。

本仓库是一份**中文优先**的检索增强生成（RAG）实战教程。每章包含：

- **自写 MVP**（30–80 行 Python，单文件可运行，零或极简依赖）
- **RAGFlow 对照**（阅读 [infiniflow/RAGFlow](https://github.com/infiniflow/ragflow) 工业级源码，摘录到 `ragflow_notes/`）
- **可复现实验**（跑在两个共享样本文件上：`samples/server_whitepaper.pdf` + `samples/disclosure.docx`）

读者画像：会用 Python 调 LLM API，想从工程角度理解 RAG 全链路；不需要先看过 RAGFlow 源码。

## 目录

| # | 章节 | 一句话简介 |
|---|---|---|
| [s01](./s01_what_is_rag/) | 什么是 RAG | 朴素 RAG vs. 长上下文；端到端最小 demo |
| [s02](./s02_document_loading/) | 文档加载 | PDF / DOCX / OCR 解析；元数据保留 |
| [s03](./s03_chunking/) | 文本分块 | 固定切分 vs. 结构感知切分 |
| [s04](./s04_embedding/) | Embedding | BGE / M3E 模型选型；批处理与归一化 |
| [s05](./s05_vector_index/) | 向量索引 | Chroma / 内存 HNSW；距离度量 |
| [s06](./s06_hybrid_retrieval/) | 混合检索 | BM25 + 向量；RRF 融合 |
| [s07](./s07_reranking/) | 重排序 | Cross-Encoder 精排；Top-K 权衡 |
| [s08](./s08_prompt_and_generation/) | Prompt 与生成 | 引用溯源；幻觉抑制 |
| [s09](./s09_agent_and_tools/) | Agent 与工具 | Function call；多轮检索 |
| [s10](./s10_graphrag/) | GraphRAG | 实体-关系抽取；图谱合并 |
| [s11](./s11_multimodal/) | 多模态 | 图文混排；表格理解 |
| [s12](./s12_deployment/) | 部署 | FastAPI + Docker；离线评估 |

> 章节文件夹尚未填充，按 Task 1–12 顺序解锁。

## 快速开始

```bash
git clone <repo-url>
cd learn-ragflow
pip install -r requirements.txt
cp .env.example .env       # 编辑 .env，填入 LLM_API_KEY
python s01_what_is_rag/code.py
```

环境要求：Python 3.10+，8GB+ 内存（跑 BGE embedding 推荐 16GB）。GPU 可选。

## 目录结构

```
learn-ragflow/
├── README.md                    # 本文件（中文）
├── README.en.md                 # 英文版摘要
├── .env.example                 # 环境变量模板（LLM / Embedding / Reranker）
├── requirements.txt             # 全部章节依赖
├── samples/
│   ├── README.md                # 说明：读者自行准备样本
│   ├── server_whitepaper.pdf    # 样本 1（中文 PDF + 表格）
│   └── disclosure.docx          # 样本 2（中文 DOCX + 段落 + 表格）
├── ragflow_notes/               # RAGFlow 源码摘录（每章引用）
│   └── README.md
├── s01_what_is_rag/             # 章节 1（占位，Task 1 创建）
├── s02_document_loading/        # 章节 2
├── ...
├── s12_deployment/              # 章节 12
└── docs/                        # 设计文档（不参与教程运行）
```

## 引用与致谢

本教程的工业级对照来自以下开源项目：

- [**RAGFlow**](https://github.com/infiniflow/ragflow) — 主力参考实现。每章 README 第 5 节引用 `ragflow_notes/` 中的源码片段。
- [**learn-claude-code**](https://github.com/shareAI-lab/learn-claude-code) — 本教程的"最小可运行 MVP"风格借鉴此仓库。
- [**BGE 系列模型**](https://github.com/FlagOpen/FlagEmbedding)（BAAI）— 默认 embedding + reranker。

## License

TBD（待用户决定，建议 MIT 或 CC-BY-SA 4.0）。
