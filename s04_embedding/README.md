# s04 文本嵌入 (Embedding) — 把 chunk 翻译成 512 维真语义向量

[上一章 s03 → · 下一章 s05 → ... → s12]

> *"`SentenceTransformer(...).encode(...)` 三行跑通的，是 s01 词袋模型留下'无语义'问题的最小答案；同一接口后挂 openai / ollama，是让 32 行代码能切后端不冲索引的最小扩容器"*
>
> **链路位置**: 离线索引链路第三步 (s02 → s03 → **s04** → s05)
> **代码文件**: local_bge.py · provider_routing.py

> 环境准备: 见 root README §快速开始 — `pip install sentence-transformers` (首次跑 `local_bge.py`/`provider_routing.py` 会从 HuggingFace Hub 下载 `BAAI/bge-small-zh-v1.5` 约 100MB,后续命中本地 `~/.cache/huggingface/hub/`)；`provider_routing.py` 切 openai / ollama 需要 `LLM_API_KEY` / `EMBED_BASE_URL`

---

## 问题

s01 的"字面匹配 + 词袋 + LLM"骨架跑通了整条 RAG 链路,但**召回质量停在"无语义"层面**——词袋给"披露"和"公开披露"零分还是给零分,s01 暴露的"找不到同义词"的局限谁也没填。s04 的任务就是把"无语义"问题钉死在 s03 输出的 chunk 上,让 s05 的向量库能接住一份"语义可对齐"的输入。

**第一,词袋卡不住同义词与跨语序变体**。s01 的 2-gram 词频向量配上余弦相似度,在"披露"和"公开披露"之间能给非零分(共享一个 2-gram);在"披露"和"公告"之间给零分(字符完全不相交);在"狗咬人"和"人咬狗"之间给完全相同的分(词频向量相同)。这类问题不是 chunking 能修的——chunk 切得再准,向量化那一步用错模型或漏归一化,出来的向量就是噪声,s05 写 Chroma / s06 做 dense 召回全部踩空。

**第二,模型选了谁,索引就被锁死**。Embedding 模型选型的**决定一旦做下,整条索引链都跟着锁**:换 BGE-small-zh (512 维) 换成 OpenAI text-embedding-3-small (1536 维),s05 的 Chroma collection 会立刻报 `Dimension mismatch`;换 bge-small-zh 换成 bge-small-en,中文文档的向量空间直接失真,中文检索质量归零。换模型意味着"全部 chunk 重新 embed + 索引重建",几小时到几天的索引成本,不是单元脚本能承受的——这正是玩具 RAG 上生产时第一道悬崖。

**第三,真语义模型再强也只是召回一半**。即便 BGE 把"披露 / 公告 / 公开披露"压到向量空间的近邻位置,top-k 召回出来的 3 段还需要 s07 rerank 重排,prompt 还要 s08 工业模板拼装,embedding 不解决"挑最相关",更不解决"组织成答案"。把 embedding 当"装好了 RAG"是最常见的误解,s04 必须把这个边界显式画出来:embedding 是把"对的段落"找出来的**第一步,不是整条流水线**。

把三条合起来看,**embedding 层的脆弱性不在 demo 上体现,在切后端 + 切语种 + 切维度时集中爆发**。s04 的任务就是先用 `local_bge.py` 跑通最小骨架(免 key、512 维归一化),再用 `provider_routing.py` 把同一接口扩成"env 切换后端"的分发器——你看到本地 BGE / OpenAI / Ollama 三家在 prompt 级别是怎么被 if/elif 串起来的,生产里怎么避坑,以及**为什么 ChatGPT 那种"接好"体验背后是 `EmbeddingModel` 抽象类 + `EMBEDDING_FACTORY` 注册表**。如果不在 s04 看清 toy 后端的边界条件,后面 s06 hybrid 检索加任何一路 dense 都可能撞上一个本地 embedder 跑不动的沉默 500。

---

## 解决方案

s04 用 **两个递进的脚本** 把"chunk → 512 维向量"这条翻译层跑起来。每一步解决前一步的局限,但也留下新的脆弱性:

```
代码 1 (本地 BGE)              代码 2 (env-driven 三后端)
┌────────────────┐      ┌───────────────────────────┐
│ 加载 BGE-small- │      │ _REGISTRY 字典            │
│ zh-v1.5       │      │ + local/openai/ollama     │
│       │        │ ───▶ │ 按 EMBED_PROVIDER 选后端  │
│ encode(        │      │                           │
│  normalize=    │      │ 返回 (provider,           │
│  True)         │      │          list[vec])        │
│       │        │      │                           │
│       ▼        │      │   缺 key → graceful skip  │
│ list[list[     │      │                           │
│  float]]       │      │                           │
└────────────────┘      └───────────────────────────┘
  512 维归一化、单后端      三后端切换、graceful fallback
```

| 脚本 | 解决什么 | 留下什么局限 | 何时用 |
|---|---|---|---|
| `local_bge.py` | chunk → 512 维 L2 归一化向量,免 API key,`@lru_cache` 不重载 | 单后端(BGE 中文);英文 / 多语言丢精度;首次依赖联网下载模型 | toy / 教学 / 中文 demo / 离线实验 |
| `provider_routing.py` | env-driven 在 local / openai / ollama 三家之间分发,统一 `list[list[float]]` schema | 无 retry / rate-limit;Ollama 逐文本 POST 没批处理;本地 BGE 仍要联网下载 | 生产 demo / 多后端 AB / provider 迁移过渡 |

两脚本的关系是一条**教学主干**:代码 1 把"加载一个免费本地模型 + 归一化 + 返回纯向量"跑通,暴露"只用一家"的局限——多语言场景或生产 prompt 里没 key 怎么办;代码 2 把代码 1 的接口形状(`list[str] → list[list[float]]`)原封不动搬到"env-driven dispatcher",暴露"切后端"的工程成本——加 provider 写新函数 + 注册一行,失败时 graceful skip 不让 demo 机器全崩。**s04 看清新旧后端的边界条件,后续章节填空——s05 修 dim 锁死(Chroma collection 显式记录 `model_name`),s06 修单路 dense(sparse BM25 补一路),s07 修 rerank 缺位,BGE-reranker-large 是 dense 召回后的第二步精排**。

---

## 代码 1: 本地 BGE Embedding ([local_bge.py](local_bge.py))

### 工作原理

**做一件事**: 用 `sentence-transformers` 加载 `BAAI/bge-small-zh-v1.5` 中文专用模型,把 s03 切好的 chunk 跑成 512 维 L2 归一化的稠密向量,下游 s05/s06 按统一 schema `list[list[float]]` 消费,不关心来源是什么模型。

**5 步**:
1. `@lru_cache(maxsize=1)` 装饰 `_local_model()`,首次调用从 HF Hub 加载 `BGE-small-zh-v1.5`(~100MB)到 `~/.cache/huggingface/hub/`,同进程后续直接命中内存
2. `_embed_local(texts)` 调 `model.encode(texts, normalize_embeddings=True)`——把每段文本编成 512 维向量并把 L2 范数归一到 1(归一化后才让"内积 ≡ 余弦相似度",FAISS/Chroma 选 `inner_product` 度量就等同选 `cosine`)
3. `embed_local(texts)` 是公开 API,内部再包一层 `list(texts)` 兼容 tuple / generator 输入;返回 `list[list[float]]`——Python 原生 list 而非 numpy,好处是 JSON 可序列化、和 OpenAI/Ollama SDK 返回结构一致
4. `main()` 直接读 `samples/server_whitepaper.pdf` 前 2 段 + `samples/disclosure.docx` 前 2 段(`PdfReader.extract_text()` / `Document.paragraphs`),拼成 4 个 chunk,跑 `embed_local` 出 `len(vecs) == 4, len(vecs[0]) == 512`
5. `os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")` 让 `EMBED_MODEL` env 覆盖默认模型名——但**改了 `EMBED_MODEL` 必须重建 s05 的索引**,512 ↔ 1024 ↔ 1536 之间不互通

```python
# 中间片段: lru_cache + L2 归一化 encode
@lru_cache(maxsize=1)
def _local_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5"))


def _embed_local(texts: list[str]) -> list[list[float]]:
    model = _local_model()
    return [v.tolist() for v in model.encode(texts, normalize_embeddings=True)]


def embed_local(texts: list[str]) -> list[list[float]]:
    """对输入文本跑本地 BGE 模型,返回 list[list[float]],每行已归一化.

    用法供外部 import;本单元 main() 也直接调用。
    """
    return _embed_local(list(texts))
```

**完整函数**:

```python
@lru_cache(maxsize=1)
def _local_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5"))


def _embed_local(texts: list[str]) -> list[list[float]]:
    model = _local_model()
    return [v.tolist() for v in model.encode(texts, normalize_embeddings=True)]


def embed_local(texts: list[str]) -> list[list[float]]:
    """对输入文本跑本地 BGE 模型,返回 list[list[float]],每行已归一化.

    用法供外部 import;本单元 main() 也直接调用。
    """
    return _embed_local(list(texts))


def main() -> None:
    sys.path.insert(0, str(WORKDIR))
    # 直接读 PDF/DOCX,不再依赖 s02/s03;让本单元真正 self-contained。
    from pypdf import PdfReader
    from docx import Document

    def _pdf(path: Path) -> list[str]:
        return [(p.extract_text() or "").strip() for p in PdfReader(path).pages if (p.extract_text() or "").strip()]

    def _docx(path: Path) -> list[str]:
        return [p.text.strip() for p in Document(path).paragraphs if p.text.strip()]

    paras = _pdf(SAMPLES / "server_whitepaper.pdf")[:2] + _docx(SAMPLES / "disclosure.docx")[:2]
    chunks = [t for t in paras if t]
    vecs = embed_local(chunks[:4])
    print(f"维度: {len(vecs[0])}, chunks: {len(vecs)}")
```

### 试一下

```bash
python s04_embedding/local_bge.py
```

实测输出(本机首次跑会先下载 ~100MB 模型,后续靠 `lru_cache` 秒回):

```
维度: 512, chunks: 4
```

- 4 chunks(PDF 前 2 段 + DOCX 前 2 段) → 4 行 512 维 L2 归一化向量
- `len(vecs) == 4` 验证 schema 行数对齐;`len(vecs[0]) == 512` 验证列宽对齐——下游 s05 写 Chroma 时 `collection.add(embeddings=vecs, ...)` 不会爆 `Dimension mismatch`

**观察**:本地 BGE 把"青蓝科技股份有限公司 / 2024 年度财务信息披露报告"和"紫光恒越 R3630 G5 / 产品白皮书"分到向量空间的不同区域——"财报" cluster 和"硬件白皮书" cluster 在 512 维里肉眼可分。**但 chunk 里"信息披露"和"对外披露"两个词同义不同形,BGE 把它们压到向量近邻位置**——这就是词袋"披露 vs 公告 = 0 分"问题在 BGE 上的修复路径。但 BGE 仍只解决了"召回"一步:`vecs[0]` 和 query vec 的 cosine 排序只是 top-k 候选,s07 rerank / s08 prompt 仍然是空缺。

### 为什么不只写这一种

``local_bge.py`` 把 toy 后端跑通了,但只把"本地 BGE"这一家做了:

- **只对中文友好**:`bge-small-zh-v1.5` 是中文专用模型,英文 / 代码 / 多语言混合输入时向量空间失真——需要 `bge-m3` (1024 维多语言) 或 `bge-small-en` 才行;s04 不切语种路由,纯靠 `EMBED_MODEL` env 改,业务代码分不清"chunk 该走哪个模型"
- **没 retry / rate-limit 包装**:HF Hub 在线下载时偶发 5xx 直接抛,生产环境应包 `tenacity` 指数退避;`@lru_cache` 同进程缓存有效,跨进程失效,微服务里每个 worker 都重新加载
- **首跑强依赖网络**:离线 / 内网 / 镜像构建失败的机器上 `SentenceTransformer(...)` 直接抛 `OSError: Can't find model`;生产通常在镜像阶段预下载并把 `HF_HOME` 指向挂载卷
- **API 不一致**:OpenAI 的 `embeddings.create(input=[...], model="text-embedding-3-small")` 接受 batch list,sentence-transformers 也支持 batch——两个 SDK 风格不同,代码 1 只演示了一家

解决方案指向 **代码 2 (provider 字典分发)** + **后续章节填坑**——s04 自己用 `_REGISTRY` dict 把 openai / ollama 串进同一 `embed()` 接口,s05 用 Chroma collection name 显式记录 `model_name` 防止 dim 错配。

---

## 代码 2: EMBED_PROVIDER 路由分发 ([provider_routing.py](provider_routing.py))

### 工作原理

**做一件事**: 把代码 1 的"本地 BGE 直跑"扩成"`_REGISTRY` 字典 + env-driven dispatcher",让 `EMBED_PROVIDER=local|openai|ollama` 一键切换,统一签名 `embed(texts: list[str]) -> list[list[float]]` 不变,s05+ 拿到不论来自哪家后端的向量都按同一形状消费。

**4 步**:
1. `_REGISTRY = {"local": _embed_local, "openai": embed_openai, "ollama": embed_ollama}` 是和 RAGFlow `EmbeddingModel` dict 同思路的最小版本——加 provider 写一个函数 + 注册一行,不动 `route()` 调度逻辑
2. `embed_openai(texts)` 走 `openai.OpenAI` SDK,默认 `text-embedding-3-small` (1536 维),base_url 兼容(`LLM_BASE_URL` 可指向 Azure / proxy);没 `LLM_API_KEY` 时由调用方决定是否抛错
3. `embed_ollama(texts)` 走 `requests.post(EMBED_BASE_URL + "/api/embeddings", json={"model": ..., "prompt": t})`——逐文本 POST(没用 `inputs=[...]` 批处理,Ollama 原生支持批但本实现为简化每次单 prompt)
4. `route(texts)` 按 `EMBED_PROVIDER` env 选后端,返回 `(provider, list[list[float]])`——返回 provider 名让上层日志清晰;`_openai_available()` / `_ollama_available()` 分别检测 `LLM_API_KEY` 是否设 + `EMBED_BASE_URL/api/tags` 是否能 reach,不可用时 `main()` 走 `print("[openai] skipped, ...")` 的 graceful fallback

```python
# 中间片段: _REGISTRY dict + route() dispatcher
_REGISTRY = {
    "local": _embed_local,
    "openai": embed_openai,
    "ollama": embed_ollama,
}


def route(texts: list[str]) -> tuple[str, list[list[float]]]:
    """按 EMBED_PROVIDER 选 backend,返回 (provider_name, vectors)。"""
    provider = os.environ.get("EMBED_PROVIDER", "local")
    fn = _REGISTRY[provider]
    return provider, fn(texts)


def embed_ollama(texts: list[str]) -> list[list[float]]:
    """EMBED_BASE_URL/api/embeddings,默认 bge-m3 (1024 维多语言)."""
    import requests
    url = os.environ.get("EMBED_BASE_URL", "http://localhost:11434") + "/api/embeddings"
    model = os.environ.get("EMBED_MODEL", "bge-m3")
    return [requests.post(url, json={"model": model, "prompt": t}).json()["embedding"] for t in texts]
```

**完整函数**:

```python
def _embed_local(texts: list[str]) -> list[list[float]]:
    """EMBED_PROVIDER=local 时跑 sentence-transformers——和 local_bge.py 同款,但独立实现。"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5"))
    return [v.tolist() for v in model.encode(list(texts), normalize_embeddings=True)]


def embed_openai(texts: list[str]) -> list[list[float]]:
    """OpenAI 兼容 /v1/embeddings,默认 text-embedding-3-small (1536 维)."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    model = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
    resp = client.embeddings.create(input=list(texts), model=model)
    return [d.embedding for d in resp.data]


def embed_ollama(texts: list[str]) -> list[list[float]]:
    """EMBED_BASE_URL/api/embeddings,默认 bge-m3 (1024 维多语言)."""
    import requests
    url = os.environ.get("EMBED_BASE_URL", "http://localhost:11434") + "/api/embeddings"
    model = os.environ.get("EMBED_MODEL", "bge-m3")
    return [requests.post(url, json={"model": model, "prompt": t}).json()["embedding"] for t in texts]


_REGISTRY = {
    "local": _embed_local,
    "openai": embed_openai,
    "ollama": embed_ollama,
}


def route(texts: list[str]) -> tuple[str, list[list[float]]]:
    """按 EMBED_PROVIDER 选 backend,返回 (provider_name, vectors)。"""
    provider = os.environ.get("EMBED_PROVIDER", "local")
    fn = _REGISTRY[provider]
    return provider, fn(texts)


def _openai_available() -> bool:
    return bool(os.environ.get("LLM_API_KEY"))


def _ollama_available() -> bool:
    host = os.environ.get("EMBED_BASE_URL", "http://localhost:11434")
    try:
        import requests
        r = requests.get(host + "/api/tags", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def main() -> None:
    provider, vecs = route(DEMOS)
    print(f"provider: {provider}, dim: {len(vecs[0])}, count: {len(vecs)}")

    if not _openai_available():
        print("[openai] skipped, set LLM_API_KEY (and LLM_BASE_URL) to enable")
    else:
        provider2, vecs2 = route([DEMOS[0]])
        print(f"[openai] ok: provider={provider2}, dim={len(vecs2[0])}")

    if not _ollama_available():
        print(f"[ollama] skipped, set EMBED_BASE_URL and run `ollama serve` to enable")
    else:
        provider3, vecs3 = route([DEMOS[0]])
        print(f"[ollama] ok: provider={provider3}, dim={len(vecs3[0])}")
```

### 试一下

```bash
# 默认 local,免 key
python s04_embedding/provider_routing.py
```

切 OpenAI:

```bash
EMBED_PROVIDER=openai LLM_API_KEY=sk-... python s04_embedding/provider_routing.py
```

切 Ollama(`ollama pull bge-m3` + `ollama serve`):

```bash
EMBED_PROVIDER=ollama EMBED_BASE_URL=http://localhost:11434 \
  python s04_embedding/provider_routing.py
```

无 `LLM_API_KEY` / ollama 没起的输出(本机 demo 默认状态):

```
provider: local, dim: 512, count: 3
[openai] skipped, set LLM_API_KEY (and LLM_BASE_URL) to enable
[ollama] skipped, set EMBED_BASE_URL and run `ollama serve` to enable
```

设上 `LLM_API_KEY` 后:

```
provider: openai, dim: 1536, count: 3
[openai] ok: provider=openai, dim=1536
[ollama] skipped, set EMBED_BASE_URL and run `ollama serve` to enable
```

- `provider: local/openai/ollama` 一行打当前路由 + dim + count——`dim` 让"切模型是否重建索引"的判断有具体数字可比
- `[openai] skipped` / `[ollama] skipped` 是 **graceful fallback**——代码 2 故意不抛异常,让你在 demo 机器上跑通整条 demo 链,缺哪个后端就只缺哪个;设上 env 后从 `skipped` 变 `ok`

**观察**: `_REGISTRY` 字典 + env dispatcher 把"切后端"的工程成本降到最小——加 cohere / jina 一行注册,不动 `route()`。但有 key 也跑出 `provider=local, dim: 512` 而不是 `provider=openai, dim: 1536` 时,根因是 `EMBED_PROVIDER` 还停在 `local`——`route()` 是按 env **当前值**查表的,设了 key 没设 provider 它就还走 local。这是分布式配置(dotenv override + 服务端 env vars 共存)里的经典陷阱。**生产里这个错配会让"明明有 key 怎么还是本地 512 维"出现,s05 Chroma 写 1536 维时报 `Dimension mismatch`**。

### 为什么不只写这一种

``provider_routing.py`` 把"切后端"工程化了,但仍有几类固有限制:

- **没 retry / rate-limit 包装**:openai 偶发 5xx / ollama 长连接 timeout 直接抛,生产环境应把所有异常包成统一 `EmbeddingError`,调用方看一种类型就能重试或换 provider
- **没 batched ollama fallback**:本实现**逐文本 POST** Ollama,3 个句子 = 3 次请求;Ollama 原生支持 `inputs=[...]` 一把提交,大 batch 时延迟成倍放大
- **本地 BGE 仍依赖联网**:`SentenceTransformer(...)` 首次触发模型下载,即便路由层没选 `local`,按需加载 lazy import 后台也会偷偷联网;**真离线环境需要 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` + 镜像阶段预下载**
- **provider 异构维度混用**:local=512 / openai-default=1536 / ollama-bge-m3=1024——三家维度不同,s05 Chroma collection 一旦混 embed 一段就 `Dimension mismatch`,得在 collection name 显式登记 `model_name+dim` 才能防住

解决方案指向 **后续章节填坑**——s05 Chroma `collection.add(embeddings=vecs, metadatas=[{"model": "bge-small-zh-v1.5", "dim": 512}, ...])` 让元数据自描述;s06 hybrid 检索把 dense 和 BM25 sparse 两条路拼起来;ChatGPT 那种"接好"体验背后是 `rag/llm/embedding_model.py:EmbeddingModel` 抽象类 + `EMBEDDING_FACTORY` 注册表,我们 s04 是更朴素的 dict 版 MVP。

---

## 接下来

s04 是 RAG 链路的**最小骨架 + provider 路由**:把 s03 的 chunk 翻译到向量空间,让 s05 的 Chroma 能接住、s06 的 dense 召回能跑通。``local_bge.py`` 把"加载 BGE-small-zh + 归一化 + 返回纯向量"做出来,``provider_routing.py`` 把"切后端"的工程成本显式化——加 provider 一行注册、缺 key 不崩。这两件事合起来,给出了后续章节的填空入口:

- **dim 错配沉默爆雷** — ``provider_routing.py`` 暴露了"local=512 / openai=1536 / ollama=1024 三家维度不同"的现实,s05 Chroma collection 一旦用 `provider_a` embed 一批再用 `provider_b` embed 另一批,`collection.add(embeddings=vecs)` 直接 `Dimension mismatch`。s05 必须把 `model_name` + `dim` 写进 collection metadata,创建时校验,防止"分批 embed 跨模型"——这是 RAGFlow `BuiltinEmbed` 字典的工业形态
- **单路 dense 召回天花板** — ``local_bge.py`` 的 BGE 把"披露 / 公告"压到向量近邻是修了 s01 的"无语义",但 BGE 不擅长精确词命中("青蓝科技" 这种 firm name 召回要靠 BM25 sparse 补)。s06 hybrid 检索 = dense BGE top-k + BM25 top-k → RRF / 加权融合,把两种语义侧召回都跑一遍,single-pass 双路
- **缺 rerank 精排** — ``local_bge.py`` 的 512 维向量在 top-k 上给 cosine 排序,但"召回"≠"挑最相关"——BGE-reranker-large 是个 cross-encoder,对 query-doc 对重打 relevance 分,s07 用它做精排
- **首跑联网 + 镜像构建成本** — ``local_bge.py`` 首次从 HF Hub 下 ~100MB 模型,CI / 离线内网环境直接挂。生产应在构建镜像阶段预下载 + 把 `HF_HOME` 指向挂载卷;真离线环境用 `HF_HUB_OFFLINE=1` 强制走本地缓存,这块在 s05 上线时一并处理

s05 **向量索引**: 把任一 provider 输出的 `list[list[float]]` 持久化到 Chroma,带上 `chunk_id / text / page / source / model_name / dim` 五键元数据,让 s06 dense cosine 召回在数十万 chunk 上跑 ANN 检索而不是顺序扫。**provider 路由与存储层的边界在 s05 定型——换 provider 不冲索引、换索引库不动 provider**。

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
