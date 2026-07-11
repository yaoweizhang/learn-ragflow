# s04 文本嵌入 (Embedding) — 把 chunk 转成 512 维稠密向量

> **章节定位**：RAG 流水线的"文本 → 向量"翻译层——把 s03 切好的 chunk 投影到语义空间，让 s05 的向量库和 s06 的召回能找到"意思相近"的段落。
>
> **章节结构**：2 个脚本。01 用 BAAI/bge-small-zh-v1.5 在本地 CPU 跑 embedding；02 加 provider 路由层，让 `EMBED_PROVIDER` 在 local / openai / ollama 之间切。
>
> **scope 注意**：本章节围绕 *单 backend + 多后端路由* 这一层给出概念 / 选型 / MVP / 工业对照的完整弧线——不引入 MTEB 榜单全表 / 自监督训练细节（那些是综述性质、留到延伸阅读）。

---

## 章节导航

| 序号 | 标题 | 入口 |
| --- | --- | --- |
| 01 | 本地 BGE Embedding (BAAI/bge-small-zh-v1.5) | [`code_01_local_bge.py`](code_01_local_bge.py) |
| 02 | Provider 路由：EMBED_PROVIDER 字典分发 | [`code_02_provider_routing.py`](code_02_provider_routing.py) |

跑法：

```bash
python s04_embedding/code_01_local_bge.py        # CPU 跑 BGE 本地 embedding
python s04_embedding/code_02_provider_routing.py # 按 EMBED_PROVIDER 在 local/openai/ollama 之间切
```

依赖：`sentence-transformers`（已在 `requirements.txt`）；`openai`（provider=openai 时）、`requests`（provider=ollama 时）。

样本输入：s03 chunker 输出（4 个 chunk，PDF 前 2 段 + DOCX 前 2 段）。

---

## 一、章节介绍

`sentence-transformers` 三行就能跑——`model = SentenceTransformer(...); model.encode(texts)`。看起来不值得单独一章。但把它接上真实 RAG 链路就会发现，"跑通"和"在 prod 不爆"之间也隔着一道悬崖——这道悬崖由几类典型问题堆起来。

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

### 1.3 真实世界的问题

1. **模型维度不一致会爆索引**。同一向量库混用 512 维（BGE-small-zh）和 1536 维（text-embedding-3-small）会让 Chroma / Milvus 在 insert 阶段直接 `Dimension mismatch` 报错；生产代码在 `BuiltinEmbed.MAX_TOKENS` 用字典显式登记每个模型的维度上限，避免运行时才发现。
2. **中英文混排选错模型**。BGE 系列按语种分（`bge-small-zh` 中文、`bge-small-en` 英文、`bge-m3` 多语言），混排文档不区分语种直接 embed 会让中英向量在同一空间失真；中文文档用 `bge-small-en` 算出来的东西是噪声。同样用 `MAX_TOKENS` 把 token 上限和模型绑定，避免错配。
3. **Embedding 只解决"召回"一半**。向量相似度召回 top-k 之后，还需要 rerank 精排（s07）和 prompt 拼装（s08）才能交给 LLM。**Embedding 模型再强，也只是把"对的内容"找出来——它不能解决"对的内容里挑最相关"，更不能"把相关内容总结成答案"**。把 Embedding 当 RAG 全套是常见误解。

### 1.4 与传统 IR 的对应

Embedding 检索和传统关键词检索的对比，正好对应 RAG 里的"语义召回 vs 词法召回"分工：

| Embedding 检索 | 传统 BM25 / TF-IDF | 共同目标 |
|---|---|---|
| 稠密向量（Dense） | 稀疏词项（Sparse） | 从语料里挑出"和问题相关的"段落 |
| 语义相似度 | 词项重合度 | 用 Top-K 喂给 LLM |
| 模型驱动（黑盒） | 词频统计（白盒） | 都是"找"的手段 |
| "苹果" ↔ "iPhone" 近 | "苹果" ≠ "iPhone" | 同一个查、两种召回方式 |

BM25 处理"原词命中"，Embedding 处理"意思命中"——s06 会把两条路拼起来做混合检索，s04 只管其中一条。

每条都对应着不同的工业级解法——维度注册、语种路由、rerank 精排。**s04 的目标不是解决它们，而是把它们显式暴露出来，让你看到 toy 方案的边界**。

这也是为什么本章用两个脚本递进：

- **01**——跑通最小骨架（`sentence-transformers` 加载 BGE → 512 维归一化向量，免 key、能跑就行）；
- **02**——把同一思路扩成 provider 路由（`EMBED_PROVIDER` env 在 local / openai / ollama 之间分发），让你看到"切后端"在工程上具体是几个 if。

这也是为什么我们不直接用 LangChain 的 `OpenAIEmbeddings` / `HuggingFaceEmbeddings` 这类更"省心"的封装——它在底层解决了 provider 切换，但你看不到**每个后端要哪个 env、哪段代码、哪个边界条件会崩**。先见问题，再看封装，比直接用学到的多。

---

## 二、详细解说

### 2.1 本地 BGE Embedding (BAAI/bge-small-zh-v1.5)

> 02 会基于同样的 `EMBED_PROVIDER` 字典分发思路，把 openai / ollama 串进同一套接口。

#### 这是什么

`code.py` 提供一个 `embed_local(texts)` 函数，把任意字符串列表送进 `sentence-transformers` 加载的 `BAAI/bge-small-zh-v1.5`（默认模型名，可用 `EMBED_MODEL` 覆盖），模型跑完 `model.encode(..., normalize_embeddings=True)` 返回 `list[list[float]]`，每行是 512 维、长度 1 的单位向量。模型用 `@lru_cache(maxsize=1)` 缓存，第二次跑同一进程不重载。

入口：[`code_01_local_bge.py`](code_01_local_bge.py)

#### 跑起来

```bash
python s04_embedding/code_01_local_bge.py
```

输出：

```
维度: 512, chunks: 4
```

首次跑 local 会从 HF Hub 下载 ~100MB 模型到 `~/.cache/huggingface/hub/`；后续运行靠 `lru_cache` 直接命中内存模型，秒回。

#### 它做对了什么

- **离线 / 免 key**：不需要任何外部 API，第一行就能跑通；
- **归一化**：所有向量落在单位球面上 → 内积 ≡ 余弦相似度，下游选点积 / L2 / cosine 哪种度量都对；
- **模型小**：单文件 ~100MB，嵌入 4 个 chunk 在 CPU 上 < 1s，重跑靠 `lru_cache` 几乎瞬时。

#### 它做错了什么

- **只对中文友好**：`bge-small-zh-v1.5` 是中文专用模型，英文 / 代码 / 长文档混合输入时向量空间会失真——生产环境应按语种切模型、把每个 provider 的维度上限显式登记，避免错配；
- **大 batch 缺 GPU**：CPU 上一次塞 1000+ 句会跑分钟级，需要 GPU 或 ONNX 量化才快；
- **首次依赖网络**：HF Hub 下载 100MB+，离线 / 内网环境直接报错；生产通常在构建镜像时预下载并把 `HF_HOME` 指到挂载卷。

#### 思考题

**为什么 BGE 输出的向量需要 `normalize_embeddings=True`？如果忘了归一化会怎样？**

提示：归一化让"内积"和"余弦相似度"在数值上等价，下游用点积或 L2 都能直接比较；不归一化时短文本向量天然小、长文本向量天然大，会让检索结果被"长度"而非"语义"主导。BGE 训练时也按余弦相似度优化，忘了归一化相当于和训练目标错位。

### 2.2 Provider 路由：EMBED_PROVIDER 字典分发

> 把 01 的本地 BGE 思路扩展成"env-driven dispatcher"——同一接口后挂多个后端。
> 与 01 不同，本脚本不 import 任何前序脚本，分发层独立。

#### 这是什么

`code.py` 暴露一个 `route(texts)`，按 `EMBED_PROVIDER` 环境变量选：

- `local`（默认）— `sentence-transformers` 跑 `BAAI/bge-small-zh-v1.5`，512 维；
- `openai` — `openai.OpenAI` 走 OpenAI 兼容协议，模型默认 `text-embedding-3-small`，读 `LLM_API_KEY` / `LLM_BASE_URL`；
- `ollama` — `requests` POST 到 `EMBED_BASE_URL/api/embeddings`（默认 `http://localhost:11434`），模型默认 `bge-m3`。

注册表 `_REGISTRY = {"local": ..., "openai": ..., "ollama": ...}` 是和 RAGFlow `EmbeddingModel` 字典同思路的最小版本——新增 provider 只要写一个函数 + 注册一行。

入口：[`code_02_provider_routing.py`](code_02_provider_routing.py)

#### 跑起来

```bash
# 默认 local,免 key
python s04_embedding/code_02_provider_routing.py

# 切 openai
EMBED_PROVIDER=openai LLM_API_KEY=sk-... python s04_embedding/code_02_provider_routing.py

# 切 ollama
EMBED_PROVIDER=ollama EMBED_BASE_URL=http://localhost:11434 python s04_embedding/code_02_provider_routing.py
```

输出示例（本机无 ollama / 无 key）：

```
provider: local, dim: 512, count: 3
[openai] skipped, set LLM_API_KEY (and LLM_BASE_URL) to enable
[ollama] skipped, set EMBED_BASE_URL and run `ollama serve` to enable
```

#### 它做对了什么

- **同一接口三个后端**：调用方只 `route(texts)`，后端切换零代码改动；
- **graceful fallback**：缺 key / ollama 没起时打印 `skipped, set env to enable`，不会让本地 demo 崩；
- **env-only 配置**：切换 = 改一个 env 变量，不需要重新打包。

#### 它做错了什么

- **没 retry / 没 rate-limit**：openai 偶发 5xx / ollama 长连接 timeout 直接抛，单元外的 retry 还得自己写——生产环境应把所有异常包成统一的 `EmbeddingError`，调用方只看一种类型就能重试或换 provider；
- **没 batched ollama fallback**：本实现逐文本 POST，N 个句子 = N 次请求；Ollama 原生支持 `inputs=[...]` 一把提交，缺批处理在大 batch 时延迟成倍放大；
- **本地 BGE 仍依赖联网**：第一次 `SentenceTransformer(...)` 还会触发模型下载，路由层假设离线就废了。

#### 思考题

**为什么 `_REGISTRY` 用字面量字典而不是 `if/elif` 链？RAGFlow 用 `inspect.getmembers` 自动扫的目的是什么？**

提示：新增一个 provider 时，"字典版"只在文件底部加一行 `"Xxx": fn_xxx`；`if/elif` 链要改 dispatch 函数——后者每次加 provider 都动调度代码，diff 噪声大、自动测试也容易漏；`inspect` 自动扫更进一步，连注册那行也省了。

---

## 三、怎么做？

### 3.1 跑起来

```bash
pip install sentence-transformers
python s04_embedding/code_01_local_bge.py          # 免 key, 首次下 ~100MB
python s04_embedding/code_02_provider_routing.py   # 默认 local, 缺 key 时打印 skipped
```

切 OpenAI：

```bash
EMBED_PROVIDER=openai LLM_API_KEY=sk-... python s04_embedding/code_02_provider_routing.py
```

切 Ollama（先 `ollama pull bge-m3` + `ollama serve`）：

```bash
EMBED_PROVIDER=ollama EMBED_BASE_URL=http://localhost:11434 \
  python s04_embedding/code_02_provider_routing.py
```

### 3.2 核心函数一览

| 函数 | 文件 | 输入 | 输出 | 一句话解释 |
|---|---|---|---|---|
| `embed_local(texts)` | `code_01_local_bge.py` | `list[str]` | `list[list[float]]` (512 维) | `sentence-transformers` 加载 BGE，输出已 L2 归一化的向量；模型用 `@lru_cache` 缓存 |
| `main()` (01) | `code_01_local_bge.py` | — | 打印维度 + 块数 | 演示入口；读 samples/ 取 4 个 chunk 跑一遍 |
| `_embed_local(texts)` | `code_02_provider_routing.py` | `list[str]` | `list[list[float]]` (512 维) | 02 独立实现的 local 分支——和 01 同款，故意不复用以便单测 |
| `embed_openai(texts)` | `code_02_provider_routing.py` | `list[str]` | `list[list[float]]` (1536 维) | `openai.OpenAI` 走 OpenAI 兼容协议，读 `LLM_API_KEY` / `LLM_BASE_URL` |
| `embed_ollama(texts)` | `code_02_provider_routing.py` | `list[str]` | `list[list[float]]` (依模型) | `requests` POST 到 `EMBED_BASE_URL/api/embeddings` |
| `route(texts)` | `code_02_provider_routing.py` | `list[str]` | `(provider_name, vectors)` | 字典分发入口，按 `EMBED_PROVIDER` 选 backend |
| `_openai_available()` | `code_02_provider_routing.py` | — | `bool` | 检测 `LLM_API_KEY` 是否存在；缺则 graceful skip |
| `_ollama_available()` | `code_02_provider_routing.py` | — | `bool` | `GET /api/tags` 探活；缺则 graceful skip |
| `main()` (02) | `code_02_provider_routing.py` | — | 打印三条后端的 status | 演示入口；逐一跑三个 provider 探活 |

### 3.3 schema 设计取舍

为什么 `embed()` 返回 `list[list[float]]` 而不是 `np.ndarray` 或 Pydantic 模型？几个常见取舍的折中：

- **`list[list[float]]` vs `np.ndarray`**：我们用 Python 列表。好处是 JSON 可序列化、和 OpenAI / Ollama 的 SDK 返回结构一致，坏处是大 batch 时内存和速度都不如 ndarray。s05 写 Chroma 时再做 `np.asarray(...)` 转换就够了——把"array 还是 list"的决定推迟到下游。
- **每条输入一行 vs 整批一块**：本教程每次调用都是"一段文本对应一行向量"，和 s03 的 chunk 一一对应，下游不需要 flatten / reshape。如果你想一次传 100 条 sentence，sentence-transformers 支持 batch，但 OpenAI 的 `embeddings.create(input=[...])` 也接受 list——两套接口在这点上一致。
- **不存 `model_name` 字段**：我们返回纯向量、不附带"我是哪个模型 embed 出来的"元数据。这意味着**调用方必须自己记住用的哪个模型**——生产代码把它写进 `tenant_id` + 集合名，本教程不引入这套命名约定（s05 再加）。
- **归一化是默认开启不是可选**：01 / 02 都强制 `normalize_embeddings=True`，理由见下方"思考题答案"的三条（内积 = 余弦、距离度量统一、和训练目标对齐）。想要"未归一化"向量的场景是少数，按需改 `_embed_local` 一行即可。

如果你的场景需要"每次返回带元数据"（比如 `[{vec, model, dim, took_ms}, ...]`），就在外层加一个 wrapper——但**保持 `embed()` 的签名是 `list[str] → list[list[float]]`**，不要把它升成 Pydantic 那种重型接口。toy 阶段越简单越好。

### 3.4 如何扩展更多 provider

加一个后端（比如 Cohere / Voyage / 智谱）只要三步：

1. 写一个 `embed_xxx(texts) -> list[list[float]]`，签名和 `embed_openai` / `embed_ollama` 一致；
2. 在 `_REGISTRY` 字典里加一行 `"xxx": embed_xxx`；
3. 在 `_xxx_available()` 里写探活逻辑（多数是检查某个 env 或 HTTP 探活），在 `main()` 里加一段"skipped / ok"打印。

不要在 `route()` 里写 `if/elif` 分发——它会污染字典分发的简洁性，新加 provider 时 diff 噪声大、容易漏 case。生产代码用 `inspect.getmembers` 模式更进一步——连那行注册都省了，只在 import 时扫一遍类变量 `_FACTORY_NAME`。s04 用最朴素的字面量字典是同等思路的最小版。

### 3.5 实际跑出来的向量形状

把 `01` 跑在仓库自带的 `samples/` 上，得到的真实片段长这样（用于对照"归一化后的向量是单位球面上的点"）：

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

### 3.6 跑出来是什么样

`01` 的预期输出（具体数字由 `samples/` 决定）：

```
维度: 512, chunks: 4
```

512 是 BGE-small-zh-v1.5 的输出维度；4 是从 samples 抽出的非空 chunk 数（PDF 前 2 段 + DOCX 前 2 段）。**首次跑会从 HF Hub 下载 ~100MB**——网络不通时直接 `ConnectionError`，见 §一.3 第 1 条。

`02` 的预期输出（本机无 OpenAI key 且无 Ollama）：

```
provider: local, dim: 512, count: 3
[openai] skipped, set LLM_API_KEY (and LLM_BASE_URL) to enable
[ollama] skipped, set EMBED_BASE_URL and run `ollama serve` to enable
```

第二、三行的 "skipped" 是 **graceful fallback**——02 故意不抛异常，让你能在 demo 机器上跑通整条 demo 链，缺哪个后端就只缺哪个。设上 `LLM_API_KEY` 之后 `embed_openai` 才会真发请求，输出从 `skipped` 变成 `[openai] ok: provider=openai, dim=1536`。

**Troubleshooting**：

- `ConnectionError` / `HF Hub unreachable`：内网 / 离线环境跑 01 会失败；构建镜像时预下载 `HF_HOME=/path/to/cache`，或 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 强制走本地缓存。
- `openai.AuthenticationError`：02 走 openai 分支时 `LLM_API_KEY` 没设或已过期；检查 `.env`。
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

我们的 toy 方案（BGE-small-zh-v1.5）在"维度 / 是否本地 / 是否需 key"上只占第一行——能跑但不抗多语言。生产代码在生产里会选 BGE-m3 / text-embedding-3-large 之一作为 1024 / 3072 维的 default，**rerank 友好的 1024 维变体**在 MTEB 检索任务上常常不输 3072 维而省一半存储。

### 4.2 选型速记

- **要快、只要中文、零成本起步** → `bge-small-zh-v1.5`（本教程 demo）；
- **要多语言、混合检索** → `bge-m3`，1024 维有 rerank-friendly 变体；
- **要省运维、英文为主** → `text-embedding-3-small`，1536 维，OpenAI 兼容；
- **要 rerank 配套** → 选 1024 维 BGE-m3 或 1024 维 Cohere multilingual，和 s07 的 BGE-reranker-large 维度对齐；
- **要先看清每个后端边界条件再选** → 用本章 `02` 把 openai / ollama 都跑一遍，看清楚"什么 env 缺了就 graceful skip"。

### 4.3 思考题

**为什么 BGE 输出的向量需要 `normalize_embeddings=True`？如果忘了归一化会怎样？**

答案见下方"思考题答案"——三条理由：内积 = 余弦、距离度量统一、和训练目标对齐。


## 思考题答案

### Q: 为什么 BGE 输出的向量需要 `normalize_embeddings=True`？

**核心原因：归一化后内积 ≡ 余弦相似度，可以直接当相似度用；同时让距离度量统一，无论向量长度。**

### 1. 内积 = 余弦相似度（数学）

余弦相似度的定义是：

```
cos(A, B) = (A · B) / (||A|| · ||B||)
```

如果 A、B 都先 L2 归一化到单位向量（`||A|| = ||B|| = 1`）：

```
cos(A, B) = A · B
```

也就是说，归一化之后，**两个向量的内积就是它们的余弦相似度**。下游向量检索（Chroma / FAISS / Milvus）选 `inner_product` 度量就等同于选 `cosine`，不用再算范数，检索速度也更快（少一次 sqrt）。

### 2. 距离度量统一（工程）

向量库通常提供三种距离：

- `inner_product`（内积，越大越相似）
- `cosine`（余弦，越大越相似）
- `L2`（欧氏距离，越小越相似）

没归一化的向量：**内积受向量长度影响** —— 一条长文档的向量 norm 天然大，内积天然高，会被错误地判为"更相似"。归一化后所有向量长度=1，这三种度量在单位球面上**等价**，你选哪个都不会错。

### 3. 与训练目标对齐（语义）

BGE 这类 Embedding 模型在训练时用的就是**余弦相似度**做对比学习（InfoNCE 之类的损失函数）。归一化之后再算相似度，推理和训练的目标一致；不归一化算内积会偏向"长向量"，召回率和训练时报告的 benchmark 对不上。

### 4. 一句话总结

`normalize_embeddings=True` 把"任意长度向量"变成"单位球面上的点"，让内积、余弦、L2 距离在排序上等价，既快又准，**和 BGE 训练时的相似度目标一致** —— 所以 s04 默认开启，而不是可选。
