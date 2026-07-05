# s04 文本嵌入 (Embedding)

## Units

| # | Unit | Goal | Entry |
|---|------|------|-------|
| 01 | [local_bge](units/01_local_bge/README.md) | 本地 `BAAI/bge-small-zh-v1.5`，512 维归一化向量，免 key | `python units/01_local_bge/code.py` |
| 02 | [provider_routing](units/02_provider_routing/README.md) | `EMBED_PROVIDER` env 在 local / openai / ollama 之间分发 | `python units/02_provider_routing/code.py` |

## 问题
关键词匹配"苹果"≠"iPhone"，需要语义。Embedding 把句子变成向量，让"苹
果 / iPhone / 手机 / fruit"在 512 维空间里距离近。

## 最小解法
`code.py` 实现 `embed(texts: list[str]) -> list[list[float]]`，按 `EMBED_PROVIDER`
环境变量分三段：
- `local` (默认) — `sentence-transformers` 加载 `BAAI/bge-small-zh-v1.5`，
  512 维，免 key，`@lru_cache(maxsize=1)` 缓存模型，重跑 <1s；
- `openai` — `openai.OpenAI` 走 OpenAI 兼容协议，调 `LLM_API_KEY /
  LLM_BASE_URL`，模型默认 `text-embedding-3-small`；
- `ollama` — `requests` POST 到 `EMBED_BASE_URL/api/embeddings`，模型默认 `bge-m3`。

```bash
cd D:/study/rag_study/learn-ragflow
python s04_embedding/code.py
```

## 跑起来
输入 4 个 chunk → 输出 `维度: 512, 前 4 个块的向量数: 4`。首次跑 local
会从 HF Hub 下载 ~100MB 模型到 `~/.cache/huggingface/hub/`；后续运行
靠 `lru_cache` 直接命中内存模型，秒回。切 OpenAI：把 `.env` 里
`EMBED_PROVIDER=openai`、填上 `LLM_API_KEY` 即可；切 Ollama：起
`ollama pull bge-m3`，设 `EMBED_PROVIDER=ollama`。

## 真实世界的问题
1. **模型维度不一致会爆索引**。同一向量库混用 512 维 (BGE-small-zh) 和
   1536 维 (text-embedding-3-small) 会让 Chroma / Milvus 直接报错；RAGFlow
   在 `BuiltinEmbed.MAX_TOKENS` (`embedding_model.py:222`) 用字典显式登记
   每个模型的维度上限，避免运行时发现。
2. **中英文混排选错模型**。BGE 系列按语言分 (`bge-small-zh` 中文、
   `bge-small-en` 英文)，混排文档不分语种直接 embed 会让中英向量在同一
   空间失真；RAGFlow 用 `BuiltinEmbed.MAX_TOKENS` 把 token 上限和模型绑
   定、避免错配。
3. **第一次下载 100MB+ 模型**。本地模式离线时直接报错；生产环境通常
   在构建镜像时预下载、运行时挂 `HF_HOME=/path/to/cache`。我们这里
   依赖 `~/.cache/huggingface` 默认路径。

## RAGFlow 怎么做的
详见 `../ragflow_notes/embedding_routing.md`。一句话总结：**声明式注册
+ 字典分发**——每个 provider 写成一个 `_FACTORY_NAME="OpenAI"` 的类，
`__init__.py` 用 `inspect` 自动把它们塞进 `EmbeddingModel` 字典，避免
`if/elif` 链；新增 provider 改一行 + 加一个类即可。

## 思考题
**为什么 BGE 输出的向量需要 `normalize_embeddings=True`？**

归一化后所有向量都落在单位球面上，**内积 ≡ 余弦相似度**，可以直接拿内
积当相似度排序用；同时让距离度量统一（点积、L2、cosine 在单位球上等
价），下游 Chroma / FAISS 选哪种度量都能正确比较，不会有"短向量内积
天然小"这种隐性偏差。BGE 模型本身在训练时也是按余弦相似度优化的，
归一化后再算相似度才和训练目标对齐。