# s04 / Unit 02 — Provider 路由：EMBED_PROVIDER 字典分发

> 由浅入深第 2 步：把 unit 01 的本地 BGE 思路扩展成"env-driven dispatcher"——同一接口后挂多个后端。  
> 与 unit 01 不同，本单元不 import 任何前序 unit，分发层独立。

## 这是什么

`code.py` 暴露一个 `route(texts)`，按 `EMBED_PROVIDER` 环境变量选：

- `local`（默认）— `sentence-transformers` 跑 `BAAI/bge-small-zh-v1.5`，512 维；
- `openai` — `openai.OpenAI` 走 OpenAI 兼容协议，模型默认 `text-embedding-3-small`，读 `LLM_API_KEY` / `LLM_BASE_URL`；
- `ollama` — `requests` POST 到 `EMBED_BASE_URL/api/embeddings`（默认 `http://localhost:11434`），模型默认 `bge-m3`。

注册表 `_REGISTRY = {"local": ..., "openai": ..., "ollama": ...}` 是和 RAGFlow `EmbeddingModel` 字典
同思路的最小版本——新增 provider 只要写一个函数 + 注册一行。

## 跑起来

```bash
# 默认 local,免 key
python s04_embedding/units/02_provider_routing/code.py

# 切 openai
EMBED_PROVIDER=openai LLM_API_KEY=sk-... python s04_embedding/units/02_provider_routing/code.py

# 切 ollama
EMBED_PROVIDER=ollama EMBED_BASE_URL=http://localhost:11434 python s04_embedding/units/02_provider_routing/code.py
```

输出示例（本机无 ollama / 无 key）：

```
provider: local, dim: 512, count: 3
[openai] skipped, set LLM_API_KEY (and LLM_BASE_URL) to enable
[ollama] skipped, set EMBED_BASE_URL and run `ollama serve` to enable
```

## 它做对了什么

- **同一接口三个后端**：调用方只 `route(texts)`，后端切换零代码改动；
- **graceful fallback**：缺 key / ollama 没起时打印 `skipped, set env to enable`，不会让本地 demo 崩；
- **env-only 配置**：切换 = 改一个 env 变量，不需要重新打包。

## 它做错了什么

- **没 retry / 没 rate-limit**：openai 偶发 5xx / ollama 长连接 timeout 直接抛，单元外的 retry 还得自己写——
  对照 RAGFlow 把所有异常统一包成 `EmbeddingError`（`embedding_model.py:46-54`），调用方只看一种类型就能重试或换 provider；
- **没 batched ollama fallback**：本实现逐文本 POST，N 个句子 = N 次请求；Ollama 原生支持 `inputs=[...]` 一把提交，
  缺批处理在大 batch 时延迟成倍放大；
- **本地 BGE 仍依赖联网**：第一次 `SentenceTransformer(...)` 还会触发模型下载，路由层假设离线就废了。

## 对照 ragflow 怎么做的

RAGFlow 的 `select_embedding`（`rag/llm/__init__.py:163-192`）用 `inspect.getmembers` 在 import 时把
所有继承 `Base` 的 provider 类按 `_FACTORY_NAME` 自动塞进 `EmbeddingModel` 字典，dispatch 端只查表、
零 `if/elif`。我们这里的 `_REGISTRY` 是同一思路、但用最朴素的字面量字典——想加第四条 provider，
RAGFlow 写一个类 + 给它 `_FACTORY_NAME`，我们写一个函数 + 注册一行。

参考：[`ragflow_notes/embedding_routing.md`](../../../../ragflow_notes/embedding_routing.md)

## 思考题

**为什么 `_REGISTRY` 用字面量字典而不是 `if/elif` 链？RAGFlow 用 `inspect.getmembers` 自动扫的目的是什么？**

提示：新增一个 provider 时，"字典版"只在文件底部加一行 `"Xxx": fn_xxx`；`if/elif` 链要改 dispatch 函数——
后者每次加 provider 都动调度代码，diff 噪声大、自动测试也容易漏；`inspect` 自动扫更进一步，连注册那行也省了。
