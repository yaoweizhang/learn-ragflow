# s04 文本嵌入 (Embedding) — 章节总览

> **章节定位**：RAG 流水线的"文本 → 向量"翻译层——把 s03 切好的 chunk 投影到语义空间，让 s05 的向量库和 s06 的召回能找到"意思相近"的段落。  
> **章节定位**：本章节围绕 *单 backend + 多后端路由* 这一层给出概念 / 选型 / MVP / 工业对照的完整弧线,**不引入 MTEB 榜单全表 / 自监督训练细节**(那些是综述性质、留到延伸阅读)。

---

## 一、什么是 Embedding？

### 1.1 核心定义

**Embedding（嵌入）** 是把任意长度的字符串（词、句、段）映射成**固定维度的稠密向量**——一串能放进 `numpy` 数组的浮点数。语义相近的文本，映射出来的向量在空间里**距离更近**；语义不相关的，距离更远。

```
    text (s03 chunk)                          vec (512-d, L2-normalized)
    "二、关键特性 计算密度..."                  [-0.028, 0.041, -0.013, -0.057, ...]
        │                                              │
        │  BGE-small-zh-v1.5                           │
        │  ────────────────────▶                       │
        │  normalize_embeddings=True                   │
        ▼                                              ▼
   离线: 全部 chunk → vec 写到 s05 chroma        在线: query → 同空间做 s06 ANN 召回
```

把这个映射装回 RAG 全景看：**s04 是离线 / 在线两条线共用的翻译层**。离线时它把 s03 切出的 chunk 翻译成向量、写进 s05 的向量库；在线时它把用户问题翻译成同一空间里的向量，让 s06 能用"找最近的邻居"代替关键词匹配。这一层翻译的**模型选型一旦定下来，整条索引链就锁死**——换模型意味着全部 chunk 重新 embed，索引重建。

### 1.2 三个核心任务

一个 Embedding 模型在 RAG 流水线里做三件事：

1. **编码（Encode）**——把字符串送进一个 Transformer 编码器（BGE、BERT、SBERT 都是典型），输出 `list[float]`，长度 = 模型的"维度"（dimension）；维度越高，能编码的语义细节越丰富，但存储和检索成本也线性上涨。
2. **归一化（Normalize）**——把输出向量 L2 归一化到单位球面。归一化之后**内积 ≡ 余弦相似度**，下游向量库（FAISS / Chroma / Milvus）选 `inner_product` 就等同于选 `cosine`，省一次 sqrt、还消除"长向量天然内积大"的隐性偏差。
3. **空间对齐（Space alignment）**——同一个语料库的**所有 chunk** 和**用户问题**必须用**同一个模型** embed。否则"问题向量"和"chunk 向量"根本不在同一个空间里，距离再近也没意义——这是 99% 的"明明语义相关但召回不到"的根因。

### 1.3 与传统 IR 的对应

Embedding 检索和传统关键词检索的对比，正好对应 RAG 里的"语义召回 vs 词法召回"分工：

| Embedding 检索 | 传统 BM25 / TF-IDF | 共同目标 |
|---|---|---|
| 稠密向量（Dense） | 稀疏词项（Sparse） | 从语料里挑出"和问题相关的"段落 |
| 语义相似度 | 词项重合度 | 用 Top-K 喂给 LLM |
| 模型驱动（黑盒） | 词频统计（白盒） | 都是"找"的手段 |
| "苹果" ↔ "iPhone" 近 | "苹果" ≠ "iPhone" | 同一个查、两种召回方式 |

BM25 处理"原词命中"，Embedding 处理"意思命中"——s06 会把两条路拼起来做混合检索，s04 只管其中一条。

---

## 二、为什么要单独写一章 Embedding？

`sentence-transformers` 三行就能跑——`model = SentenceTransformer(...); model.encode(texts)`。看起来不值得单独一章。但把它接上真实 RAG 链路就会发现，"跑通"和"在 prod 不爆"之间也隔着一道悬崖——这道悬崖由几类典型问题堆起来：

### 2.1 真实世界的问题（3 条典型）

1. **模型维度不一致会爆索引**。同一向量库混用 512 维（BGE-small-zh）和 1536 维（text-embedding-3-small）会让 Chroma / Milvus 在 insert 阶段直接 `Dimension mismatch` 报错；RAGFlow 在 `BuiltinEmbed.MAX_TOKENS`（`embedding_model.py:222`）用字典显式登记每个模型的维度上限，避免运行时才发现。
2. **中英文混排选错模型**。BGE 系列按语种分（`bge-small-zh` 中文、`bge-small-en` 英文、`bge-m3` 多语言），混排文档不区分语种直接 embed 会让中英向量在同一空间失真；中文文档用 `bge-small-en` 算出来的东西是噪声。RAGFlow 同样用 `BuiltinEmbed.MAX_TOKENS` 把 token 上限和模型绑定，避免错配。
3. **Embedding 只解决"召回"一半**。向量相似度召回 top-k 之后，还需要 rerank 精排（s07）和 prompt 拼装（s08）才能交给 LLM。**Embedding 模型再强，也只是把"对的内容"找出来——它不能解决"对的内容里挑最相关"，更不能"把相关内容总结成答案"**。把 Embedding 当 RAG 全套是常见误解。

### 2.2 这些问题为什么必须显式面对

每条都对应着不同的工业级解法——维度注册、语种路由、rerank 精排。**s04 的目标不是解决它们，而是把它们显式暴露出来，让你看到 toy 方案的边界**。

这也是为什么本章用两个 unit 递进：

- **unit 01**——跑通最小骨架（`sentence-transformers` 加载 BGE → 512 维归一化向量，免 key、能跑就行）；
- **unit 02**——把同一思路扩成 provider 路由（`EMBED_PROVIDER` env 在 local / openai / ollama 之间分发），让你看到"切后端"在工程上具体是几个 if。

这也是为什么我们不直接用 LangChain 的 `OpenAIEmbeddings` / `HuggingFaceEmbeddings` 这类更"省心"的封装——它在底层解决了 provider 切换，但你看不到**每个后端要哪个 env、哪段代码、哪个边界条件会崩**。先见问题，再看封装，比直接用学到的多。

---

## 三、怎么做？

### 3.1 章节导航

| Unit | 主题 | 它解决什么 |
|---|---|---|
| [01_local_bge](./units/01_local_bge/README.md) | 最小可跑 Embedding（BGE 本地） | "免 key 的中文向量长什么样" |
| [02_provider_routing](./units/02_provider_routing/README.md) | `EMBED_PROVIDER` env 三后端分发 | "切 OpenAI / Ollama 改哪几行" |

### 3.2 跑起来

```bash
pip install sentence-transformers
python s04_embedding/units/01_local_bge/code.py          # 免 key, 首次下 ~100MB
python s04_embedding/units/02_provider_routing/code.py   # 默认 local, 缺 key 时打印 skipped
# 旧路径仍可用（聚合入口）:
python s04_embedding/code.py
```

切 OpenAI：

```bash
EMBED_PROVIDER=openai LLM_API_KEY=sk-... python s04_embedding/units/02_provider_routing/code.py
```

切 Ollama（先 `ollama pull bge-m3` + `ollama serve`）：

```bash
EMBED_PROVIDER=ollama EMBED_BASE_URL=http://localhost:11434 \
  python s04_embedding/units/02_provider_routing/code.py
```

### 3.3 核心函数一览

s04 的代码同样薄，每个函数都对应一个"后端能力 → 统一接口"的桥接：

| 函数 | 文件 | 输入 | 输出 | 一句话解释 |
|---|---|---|---|---|
| `embed_local(texts)` | `units/01_local_bge/code.py` | `list[str]` | `list[list[float]]` (512 维) | `sentence-transformers` 加载 BGE，输出已 L2 归一化的向量；模型用 `@lru_cache` 缓存 |
| `main()` (unit 01) | `units/01_local_bge/code.py` | — | 打印维度 + 块数 | 演示入口；读 samples/ 取 4 个 chunk 跑一遍 |
| `_embed_local(texts)` | `units/02_provider_routing/code.py` | `list[str]` | `list[list[float]]` (512 维) | unit 02 独立实现的 local 分支——和 unit 01 同款，故意不复用以便单测 |
| `embed_openai(texts)` | `units/02_provider_routing/code.py` | `list[str]` | `list[list[float]]` (1536 维) | `openai.OpenAI` 走 OpenAI 兼容协议，读 `LLM_API_KEY` / `LLM_BASE_URL` |
| `embed_ollama(texts)` | `units/02_provider_routing/code.py` | `list[str]` | `list[list[float]]` (依模型) | `requests` POST 到 `EMBED_BASE_URL/api/embeddings` |
| `route(texts)` | `units/02_provider_routing/code.py` | `list[str]` | `(provider_name, vectors)` | 字典分发入口，按 `EMBED_PROVIDER` 选 backend |
| `_openai_available()` | `units/02_provider_routing/code.py` | — | `bool` | 检测 `LLM_API_KEY` 是否存在；缺则 graceful skip |
| `_ollama_available()` | `units/02_provider_routing/code.py` | — | `bool` | `GET /api/tags` 探活；缺则 graceful skip |
| `main()` (unit 02) | `units/02_provider_routing/code.py` | — | 打印三条后端的 status | 演示入口；逐一跑三个 provider 探活 |

### 3.4 schema 设计取舍

为什么 `embed()` 返回 `list[list[float]]` 而不是 `np.ndarray` 或 Pydantic 模型？几个常见取舍的折中：

- **`list[list[float]]` vs `np.ndarray`**：我们用 Python 列表。好处是 JSON 可序列化、和 OpenAI / Ollama 的 SDK 返回结构一致，坏处是大 batch 时内存和速度都不如 ndarray。s05 写 Chroma 时再做 `np.asarray(...)` 转换就够了——把"array 还是 list"的决定推迟到下游。
- **每条输入一行 vs 整批一块**：本教程每次调用都是"一段文本对应一行向量"，和 s03 的 chunk 一一对应，下游不需要 flatten / reshape。如果你想一次传 100 条 sentence，sentence-transformers 支持 batch，但 OpenAI 的 `embeddings.create(input=[...])` 也接受 list——两套接口在这点上一致。
- **不存 `model_name` 字段**：我们返回纯向量、不附带"我是哪个模型 embed 出来的"元数据。这意味着**调用方必须自己记住用的哪个模型**——RAGFlow 把它写进 `tenant_id` + 集合名，本教程不引入这套命名约定（s05 再加）。
- **归一化是默认开启不是可选**：unit 01 / unit 02 都强制 `normalize_embeddings=True`，理由见 [thinking_answers.md](./thinking_answers.md) 的三条（内积 = 余弦、距离度量统一、和训练目标对齐）。想要"未归一化"向量的场景是少数，按需改 `_embed_local` 一行即可。

如果你的场景需要"每次返回带元数据"（比如 `[{vec, model, dim, took_ms}, ...]`），就在外层加一个 wrapper——但**保持 `embed()` 的签名是 `list[str] → list[list[float]]`**，不要把它升成 Pydantic 那种重型接口。toy 阶段越简单越好。

### 3.5 如何扩展更多 provider

加一个后端（比如 Cohere / Voyage / 智谱）只要三步：

1. 写一个 `embed_xxx(texts) -> list[list[float]]`，签名和 `embed_openai` / `embed_ollama` 一致；
2. 在 `_REGISTRY` 字典里加一行 `"xxx": embed_xxx`；
3. 在 `_xxx_available()` 里写探活逻辑（多数是检查某个 env 或 HTTP 探活），在 `main()` 里加一段"skipped / ok"打印。

不要在 `route()` 里写 `if/elif` 分发——它会污染字典分发的简洁性，新加 provider 时 diff 噪声大、容易漏 case。RAGFlow 的 `inspect.getmembers` 模式更进一步——连那行注册都省了，只在 import 时扫一遍类变量 `_FACTORY_NAME`。s04 用最朴素的字面量字典是同等思路的最小版。

### 3.6 实际跑出来的向量形状

把 `unit 01` 跑在仓库自带的 `samples/` 上，得到的真实片段长这样（用于对照"归一化后的向量是单位球面上的点"）：

```python
# 输入 4 个 chunk（PDF 前 2 段 + DOCX 前 2 段）
chunks = [
  "紫光恒越 R3630 G5 是面向企业核心业务、AI 推理与虚拟化负载设计...",
  "产品白皮书  ·  v1.0  ·  仅用于 RAG 教程测试...",
  "青蓝科技股份有限公司\n2024 年度财务信息披露报告...",
  "一、公司基本情况\n...",
]

# 输出 4 × 512 的 list[list[float]]
vecs = embed_local(chunks)
# vecs[0][0:5] ≈ [-0.028, 0.041, -0.013, -0.057, 0.009]  (值会因 BGE 版本浮动)
# len(vecs) == 4, len(vecs[0]) == 512
```

下游向量库（s05）拿到这 4×512 时，**不需要知道来源是 BGE 还是 OpenAI**——它只关心"每个 chunk 对应一行固定维度的浮点数"。这就是 schema 对齐的价值：**模型差异被吸收在 Embedding 层，后续章节不用再分情况处理**——只在你**重新切回另一个模型 embed**时，s05 的索引会告诉你"模型换了，要重建"。

### 3.7 跑出来是什么样

`unit 01` 的预期输出（具体数字由 `samples/` 决定）：

```
维度: 512, chunks: 4
```

512 是 BGE-small-zh-v1.5 的输出维度；4 是从 samples 抽出的非空 chunk 数（PDF 前 2 段 + DOCX 前 2 段）。**首次跑会从 HF Hub 下载 ~100MB**——网络不通时直接 `ConnectionError`，见 §二.1 第 1 条。

`unit 02` 的预期输出（本机无 OpenAI key 且无 Ollama）：

```
provider: local, dim: 512, count: 3
[openai] skipped, set LLM_API_KEY (and LLM_BASE_URL) to enable
[ollama] skipped, set EMBED_BASE_URL and run `ollama serve` to enable
```

第二、三行的 "skipped" 是 **graceful fallback**——unit 02 故意不抛异常，让你能在 demo 机器上跑通整条 demo 链，缺哪个后端就只缺哪个。设上 `LLM_API_KEY` 之后 `embed_openai` 才会真发请求，输出从 `skipped` 变成 `[openai] ok: provider=openai, dim=1536`。

**Troubleshooting**：

- `ConnectionError` / `HF Hub unreachable`：内网 / 离线环境跑 unit 01 会失败；构建镜像时预下载 `HF_HOME=/path/to/cache`，或 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 强制走本地缓存。
- `openai.AuthenticationError`：unit 02 走 openai 分支时 `LLM_API_KEY` 没设或已过期；检查 `.env`。
- `ollama._ollama_available()` 一直 False：Ollama 没起 / `EMBED_BASE_URL` 配错；先 `curl http://localhost:11434/api/tags` 探活。
- 维度对不上（s05 写 Chroma 时报 `Dimension mismatch`）：多半是中途换了 `EMBED_MODEL`，旧 chunk 是 512 维、新 chunk 是 1536 维——**重 embed 一遍** 或 **新建一个 collection**。

---

## 四、选型与思考题

### 4.1 主流 Embedding 工具速览

下面这张表把社区常用的几类 Embedding 方案按"维度 / 是否需 key / 是否本地 / 典型尺寸"列出来，方便选型时快速对照：

| 模型 / 方案 | 维度 | API key | 部署 | 适用场景 |
|---|---|---|---|---|
| **BGE-small-zh-v1.5**（本教程 demo） | 512 | 不需要 | 本地（~100MB） | 中文文本、低资源起步 |
| **BGE-m3** | 1024 | 不需要 | 本地（~2GB） | 多语言、混合检索（dense + sparse + multi-vec） |
| **OpenAI text-embedding-3-small** | 1536 | 需要 | 商业 API | 通用英文 / 中文、零运维 |
| **OpenAI text-embedding-3-large** | 3072 | 需要 | 商业 API | 追求高语义质量、接受高成本 |
| **Cohere embed-multilingual-v3** | 1024 | 需要 | 商业 API | 100+ 语言、强多语言 |
| **M3E** | 384 | 不需要 | 本地 | 中文轻量备选、社区维护 |

我们的 toy 方案（BGE-small-zh-v1.5）在"维度 / 是否本地 / 是否需 key"上只占第一行——能跑但不抗多语言。RAGFlow 在生产里会选 BGE-m3 / text-embedding-3-large 之一作为 1024 / 3072 维的 default，**rerank 友好的 1024 维变体**在 MTEB 检索任务上常常不输 3072 维而省一半存储。

### 4.2 选型速记

- **要快、只要中文、零成本起步** → `bge-small-zh-v1.5`（本教程 demo）；
- **要多语言、混合检索** → `bge-m3`，1024 维有 rerank-friendly 变体；
- **要省运维、英文为主** → `text-embedding-3-small`，1536 维，OpenAI 兼容；
- **要 rerank 配套** → 选 RAGFlow 用的 1024 维 BGE-m3 或 1024 维 Cohere multilingual，和 s07 的 BGE-reranker-large 维度对齐；
- **要先看清每个后端边界条件再选** → 用本章 `unit 02` 把 openai / ollama 都跑一遍，看清楚"什么 env 缺了就 graceful skip"。

### 4.3 思考题

**为什么 BGE 输出的向量需要 `normalize_embeddings=True`？如果忘了归一化会怎样？**

参考答案见 [`thinking_answers.md`](./thinking_answers.md)——三条理由：内积 = 余弦、距离度量统一、和训练目标对齐。
