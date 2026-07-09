# Learn RAGFlow — RAG 技术全栈指南（中文版）

<div align="center">

**12 章从零到部署的 RAG 教程**

[English](./README.en.md)

</div>

> **一句话定位**：用最小可运行 MVP 跑通 RAG 全链路，同时对照 [RAGFlow](https://github.com/infiniflow/ragflow) 工业级源码，看懂每一行为什么这样写。

---

## 项目简介（中文｜[English](./README.en.md)）

本项目是一个面向大模型应用开发者的 **RAG（检索增强生成）实战教程**。它的目标不是让你"看完理论"，而是让你**写完 12 个 30-80 行的 MVP 玩具**之后，能讲清楚 RAG 系统每一段在做什么、为什么这样做、生产里又会变成什么样。

教程的与众不同之处在于**双线并行**：

- **左线 —— 动手做**：每章一个 `sNN_topic/code.py`，30-80 行可单独运行；你可以改一行、看输出变化，把"调大 alpha 是不是召回更好"这类问题在 5 分钟内验证。
- **右线 —— 看源码**：每章 [`docs/reference/ragflow-notes/<topic>.md`](./docs/reference/ragflow-notes/) 摘录 RAGFlow 工业级实现的 5-15 行关键代码 + 行号 + commit pin + 为什么这样写的注释。

读者画像：会用 Python 调 OpenAI 兼容 LLM API，希望从工程角度弄懂 RAG 全链路；不需要先看过 RAGFlow 源码。每章 5-10 分钟可读完 + 跑通。

## 项目意义

RAG 已经从"研究原型"变成"生产标配"。然而读者学 RAG 时通常遇到三座墙：

1. **理论太抽象**。论文讲 "chunking strategy" "hybrid retrieval" 时给了形式化定义，没给 50 行能跑的代码。
2. **工程太碎**。要做 RAG 通常得拼：文档解析、向量库、BM25、reranker、Prompt 工程、Agent 编排……每一环都要选型，每一选型都有 5 个候选。
3. **工业级看不见**。拿 LangChain / LlamaIndex 现成接口能跑，但**看不到底层**为什么这么设计；想自己扩就无从下手。

本教程的目的是**拆掉前两座、给第三座搭梯子**：

- 用 12 章 12 个 MVP 给你**每一环的最小骨架**，每章都能跑、改、看变化；
- 用 [`docs/reference/ragflow-notes/`](./docs/reference/ragflow-notes/) 给你**生产代码的真实设计**——RAGFlow 是 GitHub 上 star 数最高的开源 RAG 引擎之一、把上面每一环都工程化了、代码可读性高，正好做教材。

学完这一遍，你**再去看 LangChain / LlamaIndex / Dify** 的源码会有穿透感——知道它们在抽象什么、为什么这样抽象、什么场景能信任它、什么场景要绕过去自己写。

## 项目受众

**适合以下人群**：

- 会 Python + LLM API，想从工程角度弄懂 RAG 全链路的开发者；
- 用 LangChain / LlamaIndex 做过 demo，想拆开看看内部实现的工程师；
- 准备做 RAG 选型 / 自研 / 二次开发的 AI 应用工程师；
- 对检索 / 推荐 / 知识图谱相关领域感兴趣的算法工程师。

**前置要求**：

- 掌握 Python 基础 + `pip install` + 跑得起 `python script.py`；
- 能读懂 50 行以内的 Python，会用 `print()` debug；
- 会从终端调 OpenAI 兼容的 LLM API（已有 key）。

**不需要**：

- 不需要先看过 RAGFlow 源码——重点是本章做的东西，不是参照系；
- 不需要深度学习/Transformer 背景——提到时会点到为止；
- 不需要 GPU——BGE embedding 在 CPU 上 1-2 分钟跑完。

## 项目亮点

1. **最小可运行**：每章都是"先写 30 行的玩具"，能跑通是第一目标——生产怎么长什么样去 [`docs/reference/ragflow-notes/`](./docs/reference/ragflow-notes/) 自己翻。
2. **共享样本文件**：仓库随附两个虚构样本（`server_whitepaper.pdf`、`disclosure.docx`），跨章复用，方便对照各章输出。
3. **环境变量 + 单一依赖清单**：12 章共用一份 `requirements.txt`、一份 `.env.example`，跑通不踩坑。
4. **思考题 + 答案分离**：每章末尾的"思考题"在 `thinking_answers.md` 单独放，鼓励先自己想再看。

## 内容大纲（详细）

教程分 **5 部分、12 章**，每章一个独立可跑的 `sXX_topic/code.py`：

### 第 0 部分：概念入门

**第 0 章 — RAG 是什么 / 为什么 / 怎么演进**　[📖 章节详情](./s00_concepts/)
- [x] 概念地图：RAG = 检索 + 增强 + 生成（参数化 vs 非参数化知识）
- [x] 选型对比：vs 长上下文 / vs 微调（token 成本 + 知识新鲜度 + 可控性）
- [x] 演进主线：初级 RAG → 高级 RAG → 模块化 RAG，12 章对应演进哪个阶段
- [x] 3 个概念级 mini demo（无需 LLM key，3 分钟跑完）

读完第 0 章再进第 1 章，"朴素子串 → 词袋向量 → 检索 + LLM" 这条线讲的是什么都心里有数。

### 第一部分：RAG 入门

**第 1 章 — 什么是 RAG**　[📖 章节详情](./s01_what_is_rag/)
- [x] 3 个 unit 递进：朴素子串 → 词袋向量 → 检索 + LLM 完整链路
- [x] LLM 的三种失败场景：训练截止、私有数据、幻觉
- [x] RAG 工作流：retrieve → augment → generate

### 第二部分：数据与索引

**第 2 章 — 文档加载**　[📖 章节详情](./s02_doc_loading/)
- [x] PDF / DOCX 解析为统一 `list[{text, page, source}]`
- [x] 真实世界三个问题：扫描件 / 表格 / 页眉页脚
- [x] RAGFlow `deepdoc/parser/` 对照：VisionParser + 多 Parser 调度

**第 3 章 — 文本分块**　[📖 章节详情](./s03_chunking/)
- [x] 固定字符 cap + 句界切（spec：≤ 500 字符 / 句界为 `(.。!?！？)`）
- [x] 表格 / 父子块 / 跨段落引用三个失败模式
- [x] RAGFlow `_concat_downward` + XGBoost 30 特征 + `naive_merge` tiktoken 对照

**第 4 章 — Embedding**　[📖 章节详情](./s04_embedding/)
- [x] BGE 本地 embedding（BAAI/bge-small-zh-v1.5，512 维，归一化）
- [x] OpenAI / Ollama provider 可选
- [x] RAGFlow `embedding_model.py` 多 provider 路由对照

**第 5 章 — 向量索引**　[📖 章节详情](./s05_vector_index/)
- [x] Chroma 持久化（`PersistentClient` + HNSW cosine）
- [x] metadata 过滤：`where={"source": "..."}`
- [x] RAGFlow ES / Infinity / OceanBase 三选一对照

### 第三部分：检索与生成

**第 6 章 — 混合检索**　[📖 章节详情](./s06_retrieval/)
- [x] 自实现 BM25 + 向量召回 + `alpha * vec + (1-α) * bm25` 加权融合
- [x] `alpha` 是可配旋钮（事实型 → 偏 BM25；概念型 → 偏向量）
- [x] RAGFlow 三层融合：DB `FusionExpr` + `rerank_with_knn` + PageRank/tag `rank_fea` 对照

**第 7 章 — 重排序**　[📖 章节详情](./s07_rerank/)
- [x] BGE cross-encoder（`bge-reranker-base`）精排
- [x] `top_k` 控制 cross-encoder pair 数（O(n) 不是 O(n²)）
- [x] RAGFlow `RerankModel.Base` 多 provider 抽象对照

**第 8 章 — Prompt 与生成**　[📖 章节详情](./s08_prompt_generate/)
- [x] 引用 [i] / 拒答 / 角标对齐的 prompt 模板
- [x] `_format_context` 把 hits 渲染成 `[i] (source#page) text`
- [x] RAGFlow `citation_prompt` 双 pass + 多 prompt 模板对照

**第 9 章 — Agent 与工具**　[📖 章节详情](./s09_agent_tools/)
- [x] ReAct 循环：`Thought` / `Action` / `ActionInput` 解析
- [x] 两个工具：`retrieve(query)` + `finish(answer)`
- [x] 解析脆弱性：`max_steps` + markdown 围栏剥离 + JSON retry
- [x] RAGFlow `agent/canvas.py` DAG + `bind_tools()` OpenAI tool_calls 对照

### 第四部分：高级 RAG

**第 10 章 — GraphRAG**　[📖 章节详情](./s10_graphrag/)
- [x] LLM 抽 `(head, rel, tail)` 三元组
- [x] `dict[head] → set[(rel, tail)]` 1 跳查询
- [x] RAGFlow light 路径 + `entity_resolution` 两阶段管线对照

**第 11 章 — 多模态**　[📖 章节详情](./s11_multimodal/)
- [x] pdfplumber 抽表格（行 × 列结构）
- [x] pytesseract OCR（`chi_sim+eng`）
- [x] RAGFlow `TableStructureRecognizer` 视觉模型 + 多 OCR 后端对照

### 第五部分：部署与上线

**第 12 章 — 部署**　[📖 章节详情](./s12_deployment/)
- [x] FastAPI 包装（`/qa` 端点）+ pydantic 入参校验
- [x] docker-compose（api + chroma 持久化目录）
- [x] 503 fallback：索引缺失时给清晰错误而非裸抛异常

### 补充阅读（项目级参考，不参与教程运行，但会在 s01-s12 里反复引用）

- [RAGFlow 源码阅读索引](./docs/reference/ragflow-notes/README.md) — 工业级实现的源码摘录（想做深读时翻）
- [`docs/` 目录说明](./docs/) — `docs/reference/ragflow-notes/` 等参考材料的导航

## 快速开始

```bash
git clone <repo-url>
cd learn-ragflow
pip install -r requirements.txt
cp .env.example .env              # 编辑 .env，填入 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
source env.sh                     # 可选：HF_ENDPOINT=https://hf-mirror.com + LD_PRELOAD 修正
python s00_concepts/units/01_what_is_rag/code.py        # 第 0 章 unit 1：什么是 RAG（建议先跑，建立心智模型）
python s01_what_is_rag/units/01_naive_keyword/code.py    # 第 1 章 unit 1：朴素检索
```

LLM 端点默认配置 OpenAI 兼容协议（`LLM_BASE_URL` + `LLM_MODEL`），用户可指向任意 OpenAI 兼容服务（OpenAI / DeepSeek / 智谱 / MiniMax / vLLM 自部署等）。`LLM_MODEL` 填你所用服务实际支持的 chat 模型名即可。

## 目录结构

```
learn-ragflow/
├── README.md                     # 本文件（项目总览，含详细大纲）
├── README.en.md                  # 英文版摘要
├── LICENSE                       # MIT
├── .env.example                  # 环境变量模板（LLM / Embedding / Reranker）
├── env.sh                        # 一键环境变量（HF 镜像 + libstdc++ 修补）
├── requirements.txt              # 全部章节依赖（pypdf, python-docx, chromadb, ...）
├── samples/
│   ├── README.md                 # 样本说明
│   ├── server_whitepaper.pdf     # 样本 1（中文 PDF，4 页，1 张 13×3 表）
│   └── disclosure.docx           # 样本 2（中文 DOCX，27 段 + 3 张表）
├── docs/                         # ← 工业级源码阅读（深读时翻，补充材料）
│   └── reference/
│       └── ragflow-notes/        # RAGFlow 源码摘录（按章对应）
│           ├── README.md         # 摘录索引 + 阅读建议
│           ├── deepdoc_pdf_parsing.md    # s02
│           ├── deepdoc_chunking.md       # s03
│           ├── embedding_routing.md      # s04
│           ├── vector_indexing.md        # s05
│           ├── hybrid_retrieval.md       # s06
│           ├── rerank.md                 # s07
│           ├── prompt_templates.md       # s08
│           ├── agent_tools.md            # s09
│           ├── graph_extraction.md       # s10
│           ├── multimodal_parsing.md     # s11
│           └── deployment.md             # s12
├── s00_concepts/                 # ← 第 0 章：概念速览（3 unit，每个 unit 一个迷你 demo）
│   ├── README.md
│   └── units/
│       ├── 01_what_is_rag/       # RAG 是什么
│       ├── 02_why_rag/           # 为什么用 RAG
│       └── 03_evolution/         # 三代演进
└── s01_what_is_rag/              # 章节 1（README + 3 units + README.en + 思考题答案）
│   └── units/
│       ├── 01_naive_keyword/      # unit 1：朴素子串
│       ├── 02_vector_basics/      # unit 2：词袋向量 + cosine
│       └── 03_augmented_llm/      # unit 3：检索 + prompt + LLM
    ...
└── s12_deployment/               # 章节 12
```

## 学习路径

教程有两种走法，按时间预算选：

**快路径（2-3 小时）**：s01 → s06 → s08 → s12。
- 跑通最小 RAG（s01-s06），看到 chat 端能答资料里的问题（s08），再看一眼部署（s12）。
- 重点是**全链路打通**，适合先建立心智模型。

**完整路径（10-12 小时）**：s01 → s02 → ... → s12，按章跑。
- 每章 30-60 分钟：跑 `code.py` + 改 `code.py` 看变化 + 读对应 `docs/reference/ragflow-notes/<topic>.md`。
- 重点是**每一环的设计取舍**，适合要自研 / 选型的人。

## 每章结构

每章 (`sXX_topic/`) 内部统一形状：

```
sXX_topic/
├── README.md              # 章节入口：units 导航表 + 本章对照 ragflow_notes
├── README.en.md
├── thinking_answers.md
├── code.py                # 聚合入口：importlib 加载 units/NN/code.py（保留旧启动方式）
└── units/
    ├── 01_xxx/code.py     # unit 1（必有）
    ├── 01_xxx/README.md   # 4 段式：这是什么/跑起来/对照 ragflow/思考题
    └── 02_xxx/...         # unit 2（按需；≤ 2 unit/章）
```

每个 unit 独立可跑：

```bash
python sXX_topic/units/01_xxx/code.py
```

旧入口仍可用：

```bash
python sXX_topic/code.py   # 等价于跑 unit 01（importlib 委托）
```

> Python 模块标识符不能以数字开头，所以 chapter-root `code.py` 用 `importlib.util.spec_from_file_location` 加载 units/ 里的文件，**不**用 `from units.NN_xxx.code import main`（那是 `SyntaxError`）。

## 下一步

跑完 12 章后，可以从以下方向继续：

- **换 embedding**：把 `.env` 里的 `EMBED_PROVIDER` 从 `local` 改成 `openai` 或 `ollama`，对照检索质量差异。
- **换 chunker**：s03 的 `chunk_by_paragraph` 是按段落切；试试"按句子 + 滑动窗口"或"按 Markdown 标题"。
- **换检索权重**：s06 的 `alpha` 控制向量 vs. BM25，构造 5-10 题的评估集挑最优值。
- **上生产**：s12 给了 FastAPI + docker compose；进一步可加 Prometheus 监控、Sentry 错误追踪、模型独立部署（vLLM / TGI）。
- **接 RAGFlow 源码**：在 [`docs/reference/ragflow-notes/`](./docs/reference/ragflow-notes/) 里按章对应了 RAGFlow 源码摘录；想做深读再翻，不是默认阅读路径。

## 引用与致谢

本教程的工业级对照来自以下开源项目：

- [**RAGFlow**](https://github.com/infiniflow/ragflow) — 主力参考实现。源码摘录见 [`docs/reference/ragflow-notes/`](./docs/reference/ragflow-notes/)。RAGFlow 持续演化，摘录可能与最新版不一致。
- [**all-in-rag**](https://github.com/datawhalechina/all-in-rag) — 本教程 README 的"项目简介 / 意义 / 受众 / 亮点 / 详细大纲"结构借鉴此项目的"内容大纲"组织方式。
- [**learn-claude-code**](https://github.com/shareAI-lab/learn-claude-code) — 本教程的"最小可运行 MVP"风格借鉴此仓库；每章 README 的"问题 / 最小解法 / 跑起来 / 思考题"结构也来自此。
- [**BGE 系列模型**](https://github.com/FlagOpen/FlagEmbedding)（BAAI）— 默认 embedding + reranker。

## License

MIT — 见 [LICENSE](./LICENSE)。
