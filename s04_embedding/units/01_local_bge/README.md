# s04 / Unit 01 — 本地 BGE Embedding (BAAI/bge-small-zh-v1.5)

> 由浅入深第 1 步：免 key、离线(或准离线)可跑的中文嵌入 backbone，输出 512 维归一化向量。  
> unit 02 会基于同样的 `EMBED_PROVIDER` 字典分发思路，把 openai / ollama 串进同一套接口。

## 这是什么

`code.py` 提供一个 `embed_local(texts)` 函数，把任意字符串列表送进 `sentence-transformers` 加载的
`BAAI/bge-small-zh-v1.5`(默认模型名，可用 `EMBED_MODEL` 覆盖)，模型跑完 `model.encode(..., normalize_embeddings=True)`
返回 `list[list[float]]`，每行是 512 维、长度 1 的单位向量。模型用 `@lru_cache(maxsize=1)` 缓存，
第二次跑同一进程不重载。

## 跑起来

```bash
python s04_embedding/units/01_local_bge/code.py
```

输出：

```
维度: 512, chunks: 4
```

首次跑 local 会从 HF Hub 下载 ~100MB 模型到 `~/.cache/huggingface/hub/`；后续运行靠 `lru_cache`
直接命中内存模型，秒回。

## 它做对了什么

- **离线 / 免 key**：不需要任何外部 API，第一行就能跑通；
- **归一化**：所有向量落在单位球面上 → 内积 ≡ 余弦相似度，下游选点积 / L2 / cosine 哪种度量都对；
- **模型小**：单文件 ~100MB，嵌入 4 个 chunk 在 CPU 上 < 1s，重跑靠 `lru_cache` 几乎瞬时。

## 它做错了什么

- **只对中文友好**：`bge-small-zh-v1.5` 是中文专用模型，英文 / 代码 / 长文档混合输入时向量空间会失真——
  对照 RAGFlow `BuiltinEmbed.MAX_TOKENS` 用字典显式登记每个模型的语种和维度上限，避免错配；
- **大 batch 缺 GPU**：CPU 上一次塞 1000+ 句会跑分钟级，需要 GPU 或 ONNX 量化才快；
- **首次依赖网络**：HF Hub 下载 100MB+，离线 / 内网环境直接报错；生产通常在构建镜像时预下载并把
  `HF_HOME` 指到挂载卷。

## 对照 ragflow 怎么做的

RAGFlow 把每个 Embedding provider(BGE / OpenAI / Tongyi / BaiduYiyan / Voyage / SiliconFlow /
HuggingFace / Ollama …)写成继承 `Base` 的类、用 `_FACTORY_NAME` 类变量挂"对外名"，由
`rag/llm/__init__.py` 在 import 时 `inspect.getmembers` 自动塞进 `EmbeddingModel` 字典，零条件分支。
我们本单元的"bge 作为本地实现"只是这个机制的最小切片——unit 02 会扩展到 openai / ollama。

参考：[`ragflow_notes/embedding_routing.md`](../../../../ragflow_notes/embedding_routing.md)

## 思考题

**为什么 BGE 输出的向量需要 `normalize_embeddings=True`？如果忘了归一化会怎样？**

提示：归一化让"内积"和"余弦相似度"在数值上等价，下游用点积或 L2 都能直接比较；不归一化时短文本向量
天然小、长文本向量天然大，会让检索结果被"长度"而非"语义"主导。BGE 训练时也按余弦相似度优化，忘了
归一化相当于和训练目标错位。
