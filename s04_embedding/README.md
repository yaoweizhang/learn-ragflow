# s04 文本嵌入 (Embedding) — 把 chunk 转成 512 维 embedding 向量

> **本章定位**：s04 是 RAG 链路的"文本 → 向量"翻译层——把 s03 切好的 chunk 投影到语义空间，让 s05 的向量库和 s06 的召回能找到"意思相近"的段落。详细定位见 s00 §1.4；RAGFlow 实现见本章末"## RAGFlow 实现"。

---

## 一、章节介绍

`sentence-transformers` 三行就能跑——`model = SentenceTransformer(...); model.encode(texts)`。看起来不值得单独一章。但把它接上真实 RAG 链路就会发现，"跑通"和"在 prod 不爆"之间也隔着一道悬崖——这道悬崖由几类典型问题堆起来。

### 1.1 核心定义

**Embedding（嵌入）** 是把任意长度的字符串（词、句、段）映射成**固定维度的 embedding 向量**——一串能放进 `numpy` 数组的浮点数。语义相近的文本，映射出来的向量在空间里**距离更近**；语义不相关的，距离更远。

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

一个 Embedding 模型在 RAG 链路里做三件事：

1. **编码（Encode)**——把字符串送进一个 Transformer 编码器（BGE、BERT、SBERT 都是典型），输出 `list[float]`，长度 = 模型的"维度"(dimension）；维度越高，能编码的语义细节越丰富，但存储和检索成本也线性上涨。
2. **归一化（Normalize)**——把输出向量 L2 归一化到单位球面。归一化之后**内积 ≡ 余弦相似度**，下游向量库（FAISS / Chroma / Milvus）选 `inner_product` 就等同于选 `cosine`，省一次 sqrt、还消除"长向量天然内积大"的隐性偏差。
3. **空间对齐（Space alignment)**——同一个语料库的**所有 chunk** 和**用户问题**必须用**同一个模型** embed。否则"问题向量"和"chunk 向量"根本不在同一个空间里，距离再近也没意义——这是 99% 的"明明语义相关但召回不到"的根因。

### 1.3 真实世界的问题

1. **模型维度不一致会爆索引**。同一向量库混用 512 维（BGE-small-zh）和 1536 维（text-embedding-3-small）会让 Chroma / Milvus 在 insert 阶段直接 `Dimension mismatch` 报错；生产代码在 `BuiltinEmbed.MAX_TOKENS` 用字典显式登记每个模型的维度上限，避免运行时才发现。
2. **中英文混排选错模型**。BGE 系列按语种分（`bge-small-zh` 中文、`bge-small-en` 英文、`bge-m3` 多语言），混排文档不区分语种直接 embed 会让中英向量在同一空间失真；中文文档用 `bge-small-en` 算出来的东西是噪声。同样用 `MAX_TOKENS` 把 token 上限和模型绑定，避免错配。
3. **Embedding 只解决"召回"一半**。向量相似度召回 top-k 之后，还需要 rerank 重排序（s07）和 prompt 拼装（s08）才能交给 LLM。**Embedding 模型再强，也只是把"对的内容"找出来——它不能解决"对的内容里挑最相关"，更不能"把相关内容总结成答案"**。把 Embedding 当 RAG 全套是常见误解。

### 1.4 与传统 IR 的对应

Embedding 检索和传统关键词检索的对比，正好对应 RAG 里的"语义召回 vs 词法召回"分工：

| Embedding 检索 | 传统 BM25 / TF-IDF | 共同目标 |
|---|---|---|
| 稠密向量(Dense) | 稀疏词项(Sparse) | 从语料里挑出"和问题相关的"段落 |
| 语义相似度 | 词项重合度 | 用 Top-K 喂给 LLM |
| 模型驱动(黑盒) | 词频统计(白盒) | 都是"找"的手段 |
| "苹果" ↔ "iPhone" 近 | "苹果" ≠ "iPhone" | 同一个查、两种召回方式 |

BM25 处理"原词命中"，Embedding 处理"意思命中"——s06 会把两条路拼起来做混合检索，s04 只管其中一条。

每条都对应着不同的工业级解法——维度注册、语种路由、rerank 重排序。**s04 的目标不是解决它们，而是把它们显式暴露出来，让你看到 toy 方案的边界**。

这也是为什么本章用两个脚本递进：

- **01**——跑通最小骨架（`sentence-transformers` 加载 BGE → 512 维归一化向量，免 key、能跑就行）；
- **02**——把同一思路扩成 provider 路由（`EMBED_PROVIDER` env 在 local / openai / ollama 之间分发），让你看到"切后端"在工程上具体是几个 if。

这也是为什么我们不直接用 LangChain 的 `OpenAIEmbeddings` / `HuggingFaceEmbeddings` 这类更"省心"的封装——它在底层解决了 provider 切换，但你看不到**每个后端要哪个 env、哪段代码、哪个边界条件会崩**。先见问题，再看封装，比直接用学到的多。

---

## 二、本地 BGE Embedding (BAAI/bge-small-zh-v1.5)：[c01_local_bge.py](c01_local_bge.py)

> 02 会基于同样的 `EMBED_PROVIDER` 字典分发思路，把 openai / ollama 串进同一套接口。

### 概念

`code.py` 提供一个 `embed_local(texts)` 函数，把任意字符串列表送进 `sentence-transformers` 加载的 `BAAI/bge-small-zh-v1.5`（默认模型名，可用 `EMBED_MODEL` 覆盖），模型跑完 `model.encode(..., normalize_embeddings=True)` 返回 `list[list[float]]`，每行是 512 维、长度 1 的单位向量。模型用 `@lru_cache(maxsize=1)` 缓存，第二次跑同一进程不重载。

入口：[`c01_local_bge.py`](c01_local_bge.py)

### 跑一遍

```bash
python s04_embedding/c01_local_bge.py
```

输出：

```
维度: 512, chunks: 4
```

首次跑 local 会从 HF Hub 下载 ~100MB 模型到 `~/.cache/huggingface/hub/`；后续运行靠 `lru_cache` 直接命中内存模型，秒回。

### 看输出

把 `01` 跑在仓库自带的 `samples/` 上，得到的真实片段长这样（用于对照"归一化后的向量是单位球面上的点"）：

```python
# 输入 4 个 chunk (PDF 前 2 段 + DOCX 前 2 段)
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

### 局限与下一步

本段做对了什么 — 用 `sentence-transformers` 加载 BGE-small-zh,32 行代码里跑通"chunks → 512 维归一化向量"的最小翻译层,免 key、`lru_cache` 不重载,schema 是统一 `list[list[float]]`,s05+ 不需要分模型分情况处理。

- **只对中文友好**：`bge-small-zh-v1.5` 是中文专用模型，英文 / 代码 / 长文档混合输入时向量空间会失真——生产环境应按语种切模型、把每个 provider 的维度上限显式登记，避免错配；
- **大 batch 缺 GPU**：CPU 上一次塞 1000+ 句会跑分钟级，需要 GPU 或 ONNX 量化才快；
- **首次依赖网络**：HF Hub 下载 100MB+，离线 / 内网环境直接报错；生产通常在构建镜像时预下载并把 `HF_HOME` 指到挂载卷。

- `ConnectionError` / `HF Hub unreachable`：内网 / 离线环境跑 01 会失败；构建镜像时预下载 `HF_HOME=/path/to/cache`，或 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 强制走本地缓存。
- `OSError: [E050] Can't find model 'BAAI/bge-small-zh-v1.5'`：HuggingFace Hub 不可达；构建镜像时预下载模型到 `~/.cache/huggingface/hub/`，或 `HF_ENDPOINT=https://hf-mirror.com` 走国内镜像。
- `UnicodeEncodeError: 'gbk' codec can't encode character`：Windows 控制台编码问题，跑前 `set PYTHONIOENCODING=utf-8`(s04 / s05 / s06 同问题）。
- 维度对不上（s05 写 Chroma 时报 `Dimension mismatch`）：多半是中途换了 `EMBED_MODEL`，旧 chunk 是 512 维、新 chunk 是 1536 维——**重 embed 一遍** 或 **新建一个 collection**。

下一章 s05 如何解决 — 把这些 `list[list[float]]` 持久化到 Chroma,带上 chunk_id / text / page / source 元数据,s06 的召回才能在数十万 chunk 上跑 ANN 检索而不只是顺序扫。

---

## 三、Provider 路由：EMBED_PROVIDER 字典分发：[c02_provider_routing.py](c02_provider_routing.py)

> 把 01 的本地 BGE 思路扩展成"env-driven dispatcher"——同一接口后挂多个后端。
> 与 01 不同，本脚本不 import 任何前序脚本，分发层独立。

### 概念

`code.py` 暴露一个 `route(texts)`，按 `EMBED_PROVIDER` 环境变量选：

- `local`（默认）— `sentence-transformers` 跑 `BAAI/bge-small-zh-v1.5`，512 维；
- `openai` — `openai.OpenAI` 走 OpenAI 兼容协议，模型默认 `text-embedding-3-small`，读 `LLM_API_KEY` / `LLM_BASE_URL`；
- `ollama` — `requests` POST 到 `EMBED_BASE_URL/api/embeddings`（默认 `http://localhost:11434`），模型默认 `bge-m3`。

注册表 `_REGISTRY = {"local": ..., "openai": ..., "ollama": ...}` 是和 RAGFlow `EmbeddingModel` 字典同思路的最小版本——新增 provider 只要写一个函数 + 注册一行。

入口：[`c02_provider_routing.py`](c02_provider_routing.py)

### 跑一遍

```bash
# 默认 local,免 key
python s04_embedding/c02_provider_routing.py
```

切换 provider / 改 base URL：在 `.env` 里设 `EMBED_PROVIDER=openai|ollama`、`LLM_API_KEY`、`EMBED_BASE_URL` 等，code 顶部 `load_dotenv(override=True)` 会读到。`EMBED_PROVIDER` 默认 `local`。

输出示例（本机无 ollama / 无 key）：

```
provider: local, dim: 512, count: 3
[openai] skipped, set LLM_API_KEY (and LLM_BASE_URL) to enable
[ollama] skipped, set EMBED_BASE_URL and run `ollama serve` to enable
```

### 看输出

`02` 的预期输出（本机无 OpenAI key 且无 Ollama）：

```
provider: local, dim: 512, count: 3
[openai] skipped, set LLM_API_KEY (and LLM_BASE_URL) to enable
[ollama] skipped, set EMBED_BASE_URL and run `ollama serve` to enable
```

第二、三行的 "skipped" 是 **graceful fallback**——02 故意不抛异常，让你能在 demo 机器上跑通整条 demo 链，缺哪个后端就只缺哪个。设上 `LLM_API_KEY` 之后 `embed_openai` 才会真发请求，输出从 `skipped` 变成 `[openai] ok: provider=openai, dim=1536`。

切 OpenAI：

```bash
EMBED_PROVIDER=openai LLM_API_KEY=sk-... python s04_embedding/c02_provider_routing.py
```

切 Ollama（先 `ollama pull bge-m3` + `ollama serve`）：

```bash
EMBED_PROVIDER=ollama EMBED_BASE_URL=http://localhost:11434 \
  python s04_embedding/c02_provider_routing.py
```

### 局限与下一步

本段做对了什么 — 把 01 的本地 BGE 思路扩成 env-driven 三后端 dispatcher(`local / openai / ollama`),`_REGISTRY` 字典让加 provider = 一行注册,graceful fallback 让 demo 机不会因为缺 key 全崩,接口形状仍锁在 `list[list[float]]`。

- **没 retry / 没 rate-limit**：openai 偶发 5xx / ollama 长连接 timeout 直接抛，单元外的 retry 还得自己写——生产环境应把所有异常包成统一的 `EmbeddingError`，调用方只看一种类型就能重试或换 provider；
- **没 batched ollama fallback**：本实现逐文本 POST，N 个句子 = N 次请求；Ollama 原生支持 `inputs=[...]` 一把提交，缺批处理在大 batch 时延迟成倍放大；
- **本地 BGE 仍依赖联网**：第一次 `SentenceTransformer(...)` 还会触发模型下载，路由层假设离线就废了。

- `openai.AuthenticationError`：02 走 openai 分支时 `LLM_API_KEY` 没设或已过期；检查 `.env`。
- `ollama._ollama_available()` 一直 False：Ollama 没起 / `EMBED_BASE_URL` 配错；先 `curl http://localhost:11434/api/tags` 探活。
- `requests.exceptions.ConnectionError` 接 ollama：`EMBED_BASE_URL` 错或 ollama serve 没启；`ps aux | grep ollama` 验进程。
- 网络层 timeout 没设：`requests.post(..., timeout=10)` 是 MVP 默认，但生产要 `tenacity` + 指数退避。

下一章 s05 如何解决 — 把任一 provider 输出的 `list[list[float]]` 持久化到 Chroma,索引存储不再是"调一次重算一次"的 in-memory 列表,s06 召回可以遍历百万级 chunk。provider 路由与存储层的边界在此处定型:换 provider 不冲索引,换索引库不动 provider。

---

## 四、核心函数一览

| 函数 | 文件 | 输入 | 输出 | 一句话解释 |
|---|---|---|---|---|
| `_local_model()` | `c01_local_bge.py` | — | `SentenceTransformer(BAAI/bge-small-zh-v1.5)` | `@lru_cache(maxsize=1)` 加载本地 BGE-small-zh-v1.5;同进程只下载加载一次 |
| `_embed_local(texts)` | `c01_local_bge.py` | `list[str]` | `list[list[float]]` | 内部 helper:直接走本地 BGE `encode(..., normalize_embeddings=True)` |
| `embed_local(texts)` | `c01_local_bge.py` | `list[str]` | `list[list[float]]` | 公开 API:`texts → list[list[float]]`;`device` 自动选 cuda / cpu;主入口 |
| `main()` (01) | `c01_local_bge.py` | — | 打印句子 + 余弦相似度 | 01 演示入口;4 句中文 + cosine 矩阵 |
| `_embed_local(texts)` | `c02_provider_routing.py` | `list[str]` | `list[list[float]]` | 02 内部 helper:同样走本地 BGE(独立 import,不依赖 01);EMBED_PROVIDER=local 路径 |
| `embed_openai(texts)` | `c02_provider_routing.py` | `list[str]` | `list[list[float]]` | OpenAI 兼容 `/v1/embeddings`;`LLM_API_KEY` + `EMBED_MODEL` 控制 |
| `embed_ollama(texts)` | `c02_provider_routing.py` | `list[str]` | `list[list[float]]` | Ollama `/api/embeddings`;默认 `bge-m3`;走 `EMBED_BASE_URL` |
| `route(texts)` | `c02_provider_routing.py` | `list[str]` | `tuple(provider, list[list[float]])` | 按 `EMBED_PROVIDER` env 选后端;返回 `(provider_name, vectors)` |
| `_openai_available()` / `_ollama_available()` | `c02_provider_routing.py` | — | `bool` | 检测 key / endpoint 可用性;不可用时 `route()` 走 graceful fallback 提示 |
| `main()` (02) | `c02_provider_routing.py` | — | 打印 query + 命中的 provider + vectors | 02 演示入口;按 env 路由 + 给出 friendly hint |

## 五、跨代码协同

为什么 `embed()` 返回 `list[list[float]]` 而不是 `np.ndarray` 或 Pydantic 模型？几个常见取舍的折中：

- **`list[list[float]]` vs `np.ndarray`**：我们用 Python 列表。好处是 JSON 可序列化、和 OpenAI / Ollama 的 SDK 返回结构一致，坏处是大 batch 时内存和速度都不如 ndarray。s05 写 Chroma 时再做 `np.asarray(...)` 转换就够了——把"array 还是 list"的决定推迟到下游。
- **每条输入一行 vs 整批一块**：本教程每次调用都是"一段文本对应一行向量"，和 s03 的 chunk 一一对应，下游不需要 flatten / reshape。如果你想一次传 100 条 sentence，sentence-transformers 支持 batch，但 OpenAI 的 `embeddings.create(input=[...])` 也接受 list——两套接口在这点上一致。
- **不存 `model_name` 字段**：我们返回纯向量、不附带"我是哪个模型 embed 出来的"元数据。这意味着**调用方必须自己记住用的哪个模型**——生产代码把它写进 `tenant_id` + 集合名，本教程不引入这套命名约定（s05 再加）。
- **归一化是默认开启不是可选**：01 / 02 都强制 `normalize_embeddings=True`，理由见下方"思考题答案"的三条（内积 = 余弦、距离度量统一、和训练目标对齐）。想要"未归一化"向量的场景是少数，按需改 `_embed_local` 一行即可。

如果你的场景需要"每次返回带元数据"（比如 `[{vec, model, dim, took_ms}, ...]`），就在外层加一个 wrapper——但**保持 `embed()` 的签名是 `list[str] → list[list[float]]`**，不要把它升成 Pydantic 那种重型接口。toy 阶段越简单越好。

01 / 02 都签同一个 schema：`embed(texts: list[str]) -> list[list[float]]`。01 是本地 BGE 直跑，02 是 env-driven dispatcher 选后端。**两者不能串行**——02 不 import 01，自己处理所有后端；调用方按 `EMBED_PROVIDER` env 决定跑谁。结果集被同样的 schema 锁住，s05/s06 拿到不论来自哪个 provider 的 `list[list[float]]` 都不需要分支判断。这是把"模型选型"封装掉的价值：**后续章节按统一接口消费**，换底层只改 `c02_provider_routing.py` 的 `_REGISTRY`。

## RAGFlow 实现

RAGFlow 的 embedding 路由在 `rag/llm/embedding_model.py`：抽象出 `EmbeddingModel.Base` 接口，本地 BGE / OpenAI / Cohere / Voyage / 自部署 都走统一签名 `embed(texts: list[str]) -> list[list[float]]`。`provider` 字段从 `.env` 的 `EMBED_PROVIDER` 读，调度时按 provider 实例化对应类。

**设计取舍**：provider 抽象避免"业务代码里 if/elif provider = 'openai' 。。。" 的散弹式判断；新接一个 provider 只需要写一个 `OpenAIEmbed` 类 + 在 `EMBEDDING_FACTORY` 注册一行。

**整体拓扑**：(1) 调用方持 `list[str]` → (2) 读 `EMBED_PROVIDER` env → (3) `code_02_provider_routing.route()` 选 01(本地 BGE)/ 02(OpenAI) / Ollama 后端 → (4) 拿到统一形状 `list[list[float]]` → (5) 下游 s05 写 Chroma 索引 / s06 做 dense cosine。**生产差异**：RAGFlow 把这段抽成 `EmbeddingModel.Base` 抽象类 + `EMBEDDING_FACTORY` 注册表,新接一个 provider(比如 Cohere)只需写一个 `CohereEmbed` 类 + 一行注册,s04 toy 走 dict-based dispatch 已经够 MVP。

详细摘录与 5-15 行 "为什么这样写" 的分析见 [`docs/reference/ragflow-notes/embedding_routing.md`](../docs/reference/ragflow-notes/embedding_routing.md)。

---

## 选型速记

### 主流 Embedding 工具速览

下面这张表把社区常用的几类 Embedding 方案按"维度 / 是否需 key / 是否本地 / 典型尺寸"列出来，方便选型时快速对照：

| 模型 / 方案 | 维度 | API key | 部署 | 适用场景 |
|---|---|---|---|---|
| **BGE-small-zh-v1.5**(本教程 demo) | 512 | 不需要 | 本地(~100MB) | 中文文本、低资源起步 |
| **BGE-m3** | 1024 | 不需要 | 本地(~2GB) | 多语言、混合检索(dense + sparse + multi-vec) |
| **OpenAI text-embedding-3-small** | 1536 | 需要 | 商业 API | 通用英文 / 中文、零运维 |
| **OpenAI text-embedding-3-large** | 3072 | 需要 | 商业 API | 追求高语义质量、接受高成本 |
| **Cohere embed-multilingual-v3** | 1024 | 需要 | 商业 API | 100+ 语言、强多语言 |
| **M3E** | 384 | 不需要 | 本地 | 中文轻量备选、社区维护 |

我们的 toy 方案（BGE-small-zh-v1.5）在"维度 / 是否本地 / 是否需 key"上只占第一行——能跑但不抗多语言。生产代码在生产里会选 BGE-m3 / text-embedding-3-large 之一作为 1024 / 3072 维的 default，**rerank 友好的 1024 维变体**在 MTEB 检索任务上常常不输 3072 维而省一半存储。

- **要快、只要中文、零成本起步** → `bge-small-zh-v1.5`（本教程 demo）；
- **要多语言、混合检索** → `bge-m3`，1024 维有 rerank-friendly 变体；
- **要省运维、英文为主** → `text-embedding-3-small`，1536 维，OpenAI 兼容；
- **要 rerank 配套** → 选 1024 维 BGE-m3 或 1024 维 Cohere multilingual，和 s07 的 BGE-reranker-large 维度对齐；
- **要先看清每个后端边界条件再选** → 用本章 `02` 把 openai / ollama 都跑一遍，看清楚"什么 env 缺了就 graceful skip"。

### 扩展指南

加一个新 embedding provider（cohere / jina / 本地 sentence-transformers 换模型）只要三步：

1. 写一个 `embed_cohere(texts) -> list[list[float]]` 或 `embed_jina(texts, api_key=...) -> list[list[float]]`，**签名必须和 `embed_openai` / `embed_ollama` 一致**——返回 `list[list[float]]`，dim 写到 `_REGISTRY` 里；
2. 在 `c02_provider_routing.py` 顶部加一行 `"cohere": (embed_cohere, 1024)`，`route()` 自动通过 `_REGISTRY[EMBED_PROVIDER]` 拿到函数，不要在 `route()` 里写 `if provider == "cohere": ...`；
3. 给代码文件 README 加一段"它跟 BGE-small 比，赢在哪 / 输在哪"的对照（cohere：100+ 语言 / 需 key + 按 token 计费；jina：claude 生态友好 / 8K context）。

不要把新 provider 的判断塞进 `route()`——它只懂"按 env 查 `_REGISTRY`"这一件事。本章 MVP 只跑 local + openai + ollama，但 `_REGISTRY` 是开放表，加 provider 不动调度逻辑。

---

## 思考题

1. **为什么 BGE 输出的向量需要 `normalize_embeddings=True`？如果忘了归一化会怎样？**
2. **为什么 `_REGISTRY` 用字面量字典而不是 `if/elif` 链？RAGFlow 用 `inspect.getmembers` 自动扫的目的是什么？**

（答案见文末「思考题答案」）

---

## 思考题答案

### Q1. 为什么 BGE 输出的向量需要 `normalize_embeddings=True`?

**核心原因：归一化后内积 ≡ 余弦相似度，可以直接当相似度用；同时让距离度量统一，无论向量长度。**

#### 1. 内积 = 余弦相似度（数学）

余弦相似度的定义是：

```
cos(A, B) = (A · B) / (||A|| · ||B||)
```

如果 A、B 都先 L2 归一化到单位向量（`||A|| = ||B|| = 1`）：

```
cos(A, B) = A · B
```

也就是说，归一化之后，**两个向量的内积就是它们的余弦相似度**。下游向量检索（Chroma / FAISS / Milvus）选 `inner_product` 度量就等同于选 `cosine`，不用再算范数，检索速度也更快（少一次 sqrt）。

#### 2. 距离度量统一（工程）

向量库通常提供三种距离：

- `inner_product`（内积，越大越相似）
- `cosine`（余弦，越大越相似）
- `L2`（欧氏距离，越小越相似）

没归一化的向量：**内积受向量长度影响** —— 一条长文档的向量 norm 天然大，内积天然高，会被错误地判为"更相似"。归一化后所有向量长度=1，这三种度量在单位球面上**等价**，你选哪个都不会错。

#### 3. 与训练目标对齐（语义）

BGE 这类 Embedding 模型在训练时用的就是**余弦相似度**做对比学习（InfoNCE 之类的损失函数）。归一化之后再算相似度，推理和训练的目标一致；不归一化算内积会偏向"长向量"，召回率和训练时报告的 benchmark 对不上。

#### 4. 一句话总结

`normalize_embeddings=True` 把"任意长度向量"变成"单位球面上的点"，让内积、余弦、L2 距离在排序上等价，既快又准，**和 BGE 训练时的相似度目标一致** —— 所以 s04 默认开启，而不是可选。

### Q2. 为什么 `_REGISTRY` 用字面量字典而不是 `if/elif` 链？

新增一个 provider 时，"字典版"只在文件底部加一行 `"Xxx": fn_xxx`；`if/elif` 链要改 dispatch 函数——后者每次加 provider 都动调度代码，diff 噪声大、自动测试也容易漏；`inspect` 自动扫更进一步，连注册那行也省了。

生产代码用 `inspect.getmembers` 模式更进一步——连那行注册都省了，只在 import 时扫一遍类变量 `_FACTORY_NAME`。s04 用最朴素的字面量字典是同等思路的最小版。
