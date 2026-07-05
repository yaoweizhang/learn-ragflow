# Learn RAGFlow：12 章从零到部署的 RAG 教程

> **一句话定位**：12 章从零到部署的 RAG 教程，每章自写 30–80 行 MVP + 对照 RAGFlow 工业级实现。

本仓库是一份**中文优先**的检索增强生成（RAG）实战教程。每章包含：

- **自写 MVP**（30–80 行 Python，单文件可运行）
- **RAGFlow 对照**（阅读 [infiniflow/RAGFlow](https://github.com/infiniflow/ragflow) 工业级源码，摘录到 `ragflow_notes/`）
- **可复现实验**（跑在两个共享样本文件上：`samples/server_whitepaper.pdf` + `samples/disclosure.docx`）

读者画像：会用 Python 调 LLM API，想从工程角度理解 RAG 全链路；不需要先看过 RAGFlow 源码。

## 目录

| # | 章节 | 一句话简介 |
|---|---|---|
| [s01](./s01_what_is_rag/) | 什么是 RAG | 朴素 RAG vs. 长上下文；端到端最小 demo |
| [s02](./s02_doc_loading/) | 文档加载 | PDF / DOCX 解析；元数据保留 |
| [s03](./s03_chunking/) | 文本分块 | 固定切分 vs. 结构感知切分 |
| [s04](./s04_embedding/) | Embedding | BGE 本地 + OpenAI / Ollama 可选 |
| [s05](./s05_vector_index/) | 向量索引 | Chroma 持久化；metadata 过滤 |
| [s06](./s06_retrieval/) | 混合检索 | BM25 + 向量；加权融合 |
| [s07](./s07_rerank/) | 重排序 | BGE cross-encoder 精排 |
| [s08](./s08_prompt_generate/) | Prompt 与生成 | 引用溯源；幻觉抑制 |
| [s09](./s09_agent_tools/) | Agent 与工具 | ReAct 循环；retrieve / finish 工具 |
| [s10](./s10_graphrag/) | GraphRAG | LLM 抽实体关系；1 跳查询 |
| [s11](./s11_multimodal/) | 多模态 | pdfplumber 抽表格；pytesseract OCR |
| [s12](./s12_deployment/) | 部署 | FastAPI + docker compose |

12 章全部就绪。

## 快速开始

```bash
git clone <repo-url>
cd learn-ragflow
pip install -r requirements.txt
cp .env.example .env       # 编辑 .env，填入 LLM_API_KEY
# 可选：如果 huggingface.co 拉不到 BGE 模型 / 系统 libstdc++ 太老
source env.sh
python s01_what_is_rag/code.py
```

环境要求：Python 3.10+，8GB+ 内存（跑 BGE embedding 推荐 16GB）。GPU 可选。

LLM 端点默认配置 OpenAI 兼容协议（`LLM_BASE_URL` + `LLM_MODEL`），用户可指向任意 OpenAI 兼容服务（OpenAI / DeepSeek / 智谱 / MiniMax 等）。`LLM_MODEL` 填你所用服务实际支持的 chat 模型名即可。

## 目录结构

```
learn-ragflow/
├── README.md                    # 本文件（中文）
├── README.en.md                 # 英文版摘要
├── LICENSE                      # MIT
├── .env.example                 # 环境变量模板（LLM / Embedding / Reranker）
├── requirements.txt             # 全部章节依赖
├── samples/
│   ├── README.md                # 样本说明
│   ├── server_whitepaper.pdf    # 样本 1（中文 PDF + 表格）
│   └── disclosure.docx          # 样本 2（中文 DOCX + 段落 + 表格）
├── ragflow_notes/               # RAGFlow 源码摘录（每章引用）
│   ├── README.md                # 版本 pin + 免责说明
│   ├── deepdoc_pdf_parsing.md
│   ├── deepdoc_chunking.md
│   ├── embedding_routing.md
│   ├── vector_indexing.md
│   ├── hybrid_retrieval.md
│   ├── rerank.md
│   ├── prompt_templates.md
│   ├── agent_tools.md
│   ├── graph_extraction.md
│   ├── multimodal_parsing.md
│   └── deployment.md
├── s01_what_is_rag/             # 章节 1
├── s02_doc_loading/             # 章节 2
├── ...
├── s12_deployment/              # 章节 12
└── docs/                        # 设计文档（不参与教程运行）
```

## 下一步（reader 的延伸练习）

跑完 12 章后，可以从以下任一方向继续：

- **换 embedding**：把 `.env` 里的 `EMBED_PROVIDER` 从 `local` 改成 `openai` 或 `ollama`，对照检索质量差异。
- **换 chunker**：s03 的 `chunk_by_paragraph` 是按段落切分；换成"按句子 + 滑动窗口"或"按 Markdown 标题"试试。
- **换检索权重**：s06 的 `alpha` 控制向量 vs. BM25，构造 5-10 题的评估集挑最优值。
- **上生产**：s12 给了 FastAPI + docker compose；进一步可加 Prometheus 监控、Sentry 错误追踪、模型独立部署（vLLM / TGI）。
- **接 RAGFlow 源码**：每章 `ragflow_notes/<topic>.md` 都摘录了 RAGFlow 关键 5-10 行；从 s01 的 `rag/nlp/search.py` 一直读到 s12 的 `docker/docker-compose.yml`，看工业级 RAG 系统怎么从 30 行 MVP 演化出来。

## 引用与致谢

本教程的工业级对照来自以下开源项目：

- [**RAGFlow**](https://github.com/infiniflow/ragflow) — 主力参考实现。每章 README 第 5 节引用 `ragflow_notes/` 中的源码片段。RAGFlow 摘录 pin 在 commit `828c5789f`（2026-07-01）；RAGFlow 持续演化，摘录可能过时。
- [**learn-claude-code**](https://github.com/shareAI-lab/learn-claude-code) — 本教程的"最小可运行 MVP"风格借鉴此仓库。
- [**BGE 系列模型**](https://github.com/FlagOpen/FlagEmbedding)（BAAI）— 默认 embedding + reranker。

## License

MIT — 见 [LICENSE](./LICENSE)。
