# s12 部署 — Docker Compose 把 MVP 包成服务

> **本章定位**：s12 是 RAG 的"上线面"。s08 的 `answer()` 只能在命令行跑 `python s08_prompt_generate/code_01_prompt_template.py`，然后交互式输入问题——s12 把这条链路包成一个 HTTP 服务，再装进容器里，**让同事 / 前端 / 其他服务用 `curl` 就能调**。整章有 4 个文件：1 个 `code_01_fastapi_docker.py` 启动器 + 3 个 supporting files（`app.py` FastAPI + `Dockerfile` + `docker-compose.yml`）。详细定位见 s00 §1.4；RAGFlow 实现见本章末"## RAGFlow 实现"。

---

## 一、章节介绍

把"本地命令行能跑"升级为"HTTP 服务能调"，核心动作有三件：① 用 **FastAPI**（或 Flask / Django）把 s04 → s06 → s07 → s08 这条链路包成 `POST /qa {question: ...} → {text, citations}`；② 用 **Dockerfile** 把 Python 运行时 + 业务代码 + 系统依赖（tesseract）一起烧进镜像；③ 用 **docker-compose.yml** 声明端口、卷挂载、环境变量，让 `docker compose up --build` 一条命令拉起整个服务。本章的 4 个文件就是这三件动作的最小落地。把 s12 放进 RAG 全景看：**s12 是把 s08 的 `answer()` 函数从"我在自己机器上跑得通"升级为"别人用 `curl` 就能调"**——同时把"装 Python / 下模型 / 重建索引"这些环境成本转嫁给 Docker 镜像。部署不是把代码"传上去"就完了——它是一个**契约**：服务端要保证镜像能跑、端口能访问、索引能读、LLM key 能用；客户端只要发 HTTP 就能拿到答案。

### 1.1 核心定义与端到端契约

**部署 MVP(Deployment MVP)** 是把"本地命令行能跑"升级为"HTTP 服务能调"的过程。"本机能跑"和"别人能用"是两类交付——这道鸿沟由 4 类典型失败堆起来：

1. **Python / 依赖 / 索引要对方全装一遍**——同事 clone 仓库后要 `pip install -r requirements.txt` + `cp .env.example .env` 填 key + 跑 s04/s05 重建索引 + 下 BGE 模型权重（~100MB)+ 下 bge-reranker-base(~1GB)——**5 个步骤任一卡住就报 traceback**。**生产解法**：用 Docker 把 Python + 系统依赖 + pip 包 + 业务代码 + 模型权重全部装进镜像，对方只要 `docker compose up --build` 一行命令。
2. **没法给前端 / 其他服务调**——命令行交互式输入只适合"开发者自己玩"。前端要做个搜索框、其他服务要做 API 集成，**没有 HTTP 端点就只能写文件 / 跑子进程**，耦合重。**生产解法**：用 FastAPI(pydantic 强类型 + OpenAPI 文档自动生成）把 `answer()` 包成 `POST /qa`，`curl` / fetch / requests 都能调。
3. **重启 = 数据丢失 / 索引丢失**——本地跑 `python s08_prompt_generate/code_01_prompt_template.py` 时索引在内存；kill 重启就重建。多人协作 / 服务器跑就尴尬。**生产解法**：把 Chroma 索引目录用 volume 挂载进容器（`:ro` 只读），容器重启不丢、重新 build 不丢、多副本读同一份索引。
4. **冷启动慢 + 没有任何观测**——`POST /qa` 第一次请求 Chroma hnswlib 要 mmap 索引文件，3-5 秒延迟；之后没人知道"现在 /qa 平均延迟多少 / 错误率多少 / LLM 调用成本多少"。**生产解法**：① 容器预热（`docker compose run --rm rag python -c "import chromadb; chromadb.PersistentClient('...').get_collection('docs')"` 预热后 SIGTERM，再起服务）；② 加 `/healthz` 端点给 K8s liveness probe；③ 暴露 Prometheus `/metrics` 抓 `请求量 / 延迟分位 / 错误率 / LLM token 消耗`。MVP 都不带。

每条失败模式都对应一种工业级解法——Docker 镜像打包、HTTP 端点暴露、volume 挂载持久化、预热 + 监控可观测。**s12 的目标不是解决它们，而是把它们显式暴露出来，让你看到"本机命令行 RAG"和"生产 HTTP RAG"之间的边界**。这跟 s10 把"向量召回答不全'实体之间关系'"、s11 把"表格被拍扁 / 扫描件返回空"显式对比是同一种思路——**叙述载体从"图函数 + 1 跳 query"换成"FastAPI 包装 + Docker 镜像 + compose 编排 + 启动器"，但"先跑通 toy、再讲清楚 toy 在哪里会塌"的教学哲学是一致的**。

```
   本地命令行 RAG                                容器化 HTTP 服务
   python s08/code.py                            docker compose up --build
        │                                              │
        │  FastAPI 包装                                 │
        │  ─────────────▶                              │
        │  POST /qa 端点                               │
        │  QARequest(BaseModel)                       │
        ▼                                              ▼
   app.py (FastAPI 实例)                         容器 rag:8000
   ──────────▶                                  ──────────▶
   Dockerfile (python:3.11-slim + tesseract)    镜像层缓存: 基础→依赖→代码
        │                                              │
        │  docker-compose.yml                          │
        │  ─────────────▶                              │
        │  build context / ports / env_file            │
        │  volumes: samples + _chroma:ro               ▼
        ▼                                       curl POST /qa → {text, citations}
   启动器 code.py: .env gating + 索引 gating + subprocess docker compose up
```

> 💡 **一句话总结**：部署 MVP = FastAPI 包装（HTTP 入口）+ Dockerfile（镜像）+ docker-compose.yml（编排）+ 启动器（前置校验）。
>
> 让 RAG 从"我在自己机器上跑得通"升级为"别人用 `curl` 就能调"——同时把"配环境"成本转嫁给 Docker。

### 1.2 FastAPI 包装的端点 schema

`app.py` 暴露的 `POST /qa` schema 长这样——`{question: str} → {text: str, citations: list[dict]}`：

```python
# 请求
class QARequest(BaseModel):
    question: str

# 响应(透传 s08 answer() 的返回值)
{"text": "项目里包含:规格表、内存表、电源表...", "citations": [{"source": "samples/server_whitepaper.pdf", "page": 2, "text": "..."}]}
```

请求 schema 只有一个字段 `question`；响应 schema 直接复用 s08 `answer()` 的 `{"text", "citations"}` 字段——**这是 s04 → s06 → s07 → s08 链路"端到端 JSON 可序列化"的红利**：只要每一段函数的输入输出都是 JSON 安全的，HTTP 边界就成了纯"序列号 + 拆包"问题。`app.py` 内部按 `embed → hybrid_search → rerank → answer` 的顺序串起来，每一段函数的输入输出都跟上游 s04-s08 一致。

> 💡 端点 schema 不是 FastAPI 独有——它来自 HTTP REST 规范。FastAPI 把这套规范用 pydantic 做强类型校验（`QARequest(BaseModel)`），请求体字段缺失 / 类型错会直接 422。

### 1.3 Dockerfile 的层结构

`Dockerfile` 把镜像拆成 4 层，每层只承担一件事——**这是 Docker 层缓存的关键**：

```dockerfile
FROM python:3.11-slim                          # 第 1 层: 基础镜像 (150MB)
WORKDIR /app
RUN apt-get install -y tesseract-ocr build-essential   # 第 2 层: 系统依赖 (tesseract 给 s11 OCR, build-essential 给 chroma-hnswlib 编译)
COPY ../requirements.txt .                      # 第 3 层: 依赖声明
RUN pip install -r requirements.txt             # 第 4 层: 业务依赖 (改一次 requirements 才会重 build 这一层)
COPY .. /app                                    # 第 5 层: 业务代码 (改代码重 build 这一层)
EXPOSE 8000
CMD ["uvicorn", "s12_deployment.app:app", ...]
```

关键设计：① **基础镜像层（1-2)** 几乎不变——`python:3.11-slim` 拉一次就缓存；② **依赖层（3-4)** 只在 `requirements.txt` 改了才重 build——这是为什么 `COPY requirements.txt .` 和 `RUN pip install` 必须分两步（合并成一步会导致代码改动也触发 pip 整条重装）；③ **代码层（5)** 是"高频改动层"——业务代码改了重 build 第 5 层，前 4 层全部命中缓存。MVP 不分模型层（`bge` 权重 ~100MB、`bge-reranker-base` ~1GB)——这些随 `pip install` 一起烧进第 4 层，**改一行代码 `docker compose build` 就触发完整重 build**，是 demo 的代价。生产里模型走 volume / S3 挂载，详见 §三。

> 💡 层结构不是 Dockerfile 独有——它来自 Docker 镜像的 union file system 设计。理解这一点能少踩很多"改一行代码为什么重 build 半小时"的坑。

### 1.4 docker-compose 的服务声明

`docker-compose.yml` 把"镜像怎么 build、端口怎么映射、卷怎么挂、环境怎么传"全部声明成 YAML，**让编排成为可复现的配置文件而不是命令脚本**：

```yaml
services:
  rag:
    build:
      context: ..                    # 项目根 (能 COPY 到 requirements.txt + 所有 s0X_* 源码)
      dockerfile: s12_deployment/Dockerfile
    ports: ["8000:8000"]
    env_file: [../.env]              # 把 LLM_API_KEY 透传给容器
    volumes:
      - ../samples:/app/samples:ro                          # 样本只读
      - ../s05_vector_index/_chroma:/app/s05_vector_index/_chroma   # 索引只读
```

关键设计：① **build.context 是项目根**——这样 Dockerfile 能 `COPY .. /app` 把整个仓库代码带进镜像，包括 s04-s11 的所有依赖；② **`env_file: ../.env`**——容器内 `/app/.env` 自动被 Docker 加载，s08 进程读 `LLM_API_KEY` 跟本地一致；③ **只读挂载**(`:ro`)——容器不会误写索引 / 样本，重 build 不丢数据；④ **路径用 `../` 前缀**——相对 `docker-compose.yml` 文件位置解析，本地跑和 CI 跑行为一致。

> 💡 YAML 声明式编排不是 docker-compose 独有——K8s manifests / Helm charts / Nomad HCL 都是同一种思路。MVP 走 compose 是因为它零依赖、单文件、上手 5 分钟。

---

## 二、fastapi_docker：[code_01_fastapi_docker.py](code_01_fastapi_docker.py)

> 把 s08 的 `answer()` 包成 FastAPI，再用 `docker compose` 一键起服务。这也是 s12 唯一的一步——没有步骤拆分，因为整章只跑 1 个容器。

[`code_01_fastapi_docker.py`](code_01_fastapi_docker.py)

### 概念

本节做两件事：① `s12_deployment/app.py` 用 FastAPI 把 s04 → s06 → s07 → s08 串成 `POST /qa {question: ...} → {text, citations}`；② `Dockerfile` + `docker-compose.yml` 把这个 FastAPI 应用包成镜像并跑起来，只挂 `.env` / `samples` / `_chroma` 三样东西，其他都靠镜像里 `pip install -r requirements.txt` 兜底。`code_01_fastapi_docker.py` 干的事就是先做两道前置校验（`.env` 存在、`s05_vector_index/_chroma` 存在），然后 `subprocess.run(["docker", "compose", "up", "--build"], cwd=s12_deployment)`。

s12 之所以只有这一步：整章就 4 个文件（`app.py` / `Dockerfile` / `docker-compose.yml` + 本 `code_01_fastapi_docker.py`），组合只有一种，没有"由浅入深"的多档梯度——`docker compose up --build` 是终点，没有中间状态可设。所以 1 步就是全部。

### 跑一遍

```bash
# 0. 前提:跑通 s04 / s05 / s07 (生成索引 + 装好 bge)
# 1. 配 .env (项目根)
cp .env.example .env   # 填 LLM_API_KEY

# 2. 启动 (任选一种)
python s12_deployment/code_01_fastapi_docker.py
# 内部执行 docker compose up --build,会自动 build 镜像 + 启服务

# 3. 测一发
curl -X POST http://localhost:8000/qa \
  -H 'Content-Type: application/json' \
  -d '{"question":"项目里有哪些表?"}'

# 4. 停掉
# 在跑 code_01_fastapi_docker.py 的终端 Ctrl-C
```

输出示例（成功时终端会看到 docker compose 的 build + 容器日志流）：

```
[+] Running 2/2
 ✔ Network s12_deployment_default  Created                             0.0s
 ✔ Container s12_deployment-rag-1  Created                             0.1s
Attaching to s12_deployment-rag-1
s12_deployment-rag-1  | INFO:     Started server process [1]
s12_deployment-rag-1  | INFO:     Waiting for application startup.
s12_deployment-rag-1  | INFO:     Application startup complete.
s12_deployment-rag-1  | INFO:     Uvicorn running on http://0.0.0.0:8000
```

如果前置条件不满足：

```
❌ .env 不存在,请先 cp .env.example .env 并填 LLM_API_KEY
```

### 看输出

把 `app.py` 串起来的整条链路跑通后，实测的 `POST /qa` 请求和响应长这样（输入 `"项目里有哪些表?"`）：

```bash
# 请求
curl -X POST http://localhost:8000/qa \
  -H 'Content-Type: application/json' \
  -d '{"question":"项目里有哪些表?"}'

# 响应
{
  "text": "项目里包含:规格表(整机规格表 + 各组件详细规格)、内存表、电源表...",
  "citations": [
    {"source": "samples/server_whitepaper.pdf", "page": 2, "text": "三、整机规格 组件 规格 说明 处理器 2 × 第三代 Intel Xeon 可扩展处理器 最高 40 核 / 80 线程..."}
  ]
}
```

或者无 `LLM_API_KEY` 时（graceful-skip）：

```json
{
  "text": "[skipped: LLM_API_KEY not set]",
  "citations": [
    {"source": "samples/server_whitepaper.pdf", "page": 2, "text": "三、整机规格 ..."},
    {"source": "samples/server_whitepaper.pdf", "page": 1, "text": "二、关键特性 ..."}
  ]
}
```

`citations` 数组的每条对应 s07 精排后喂给 s08 的 top-k hit——**HTTP 边界只做"序列号 + 拆包"**，真正的 LLM 答不答、拒答不拒答由 s08 决定。

容器日志流（实测成功）：

```
[+] Running 2/2
 ✔ Network s12_deployment_default  Created                             0.0s
 ✔ Container s12_deployment-rag-1  Created                             0.1s
Attaching to s12_deployment-rag-1
s12_deployment-rag-1  | INFO:     Started server process [1]
s12_deployment-rag-1  | INFO:     Waiting for application startup.
s12_deployment-rag-1  | INFO:     Application startup complete.
s12_deployment-rag-1  | INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 局限与下一步

本段做对了什么 — 把 s04 → s06 → s07 → s08 的整条 RAG 管线收敛到 `POST /qa` 一个 FastAPI 端点,再用 docker compose 一键起服务,本地端到端可跑通。


- **没有 health-check** — `docker compose up` 不等服务 `Application startup complete` 就认为"成功"，万一容器内 FastAPI import 阶段就崩（比如缺某个 pip 包），终端只会看到 `exited with code 1`，新用户分不清是镜像坏了还是代码坏了。生产 compose 应该加 `healthcheck: test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"]` + `depends_on: condition: service_healthy`。
- **没有 rolling restart / log streaming** — `Ctrl-C` 直接杀容器，日志只在当前终端，没有 `docker compose logs -f` 的统一入口，也没有 `restart: unless-stopped` 的开机自启。
- **索引没预热** — Chroma 的 hnswlib 首次 query 要 mmap 索引文件，冷启动第一次 `/qa` 会有 3-5 秒延迟；这步没做 warm-up 请求，生产环境首单用户会感知到。
- **没有鉴权 / 限流** — `POST /qa` 谁都能调，LLM token 钱随便烧；也没有 rate limit，一个脚本可以瞬间打满 token 配额。


- **`docker: command not found`** — 没装 Docker Desktop。本章需要它，装好之后 `docker --version` 应该能跑。Windows 用户注意 WSL2 后端。
- **`failed to solve: failed to read dockerfile`** — `docker-compose.yml` 里 `build.context` 和 `dockerfile` 路径对不上。本章的 compose 已经显式写成 `dockerfile: s12_deployment/Dockerfile`，别手贱改回 `build: ..`（那样 Docker 会去项目根找 `Dockerfile`，找不到）。
- **build 时 `pip install` 报 chroma-hnswlib 编译错** — Dockerfile 里装了 `build-essential` 就是给这个兜底的；如果还报错，通常是网络问题，换 pip 镜像源。
- **`/qa` 返回 500 + "LLM_API_KEY not set"** — 容器没拿到 `.env`。compose 里 `env_file: ../.env` 是相对 compose 文件位置的相对路径，确认 `.env` 在项目根。
- **`/qa` 返回 500 + "Collection docs does not exist"** — 挂载路径不对。`s05_vector_index/_chroma` 必须先在主机上存在（跑过 s05），compose 里是 `:ro` 只读挂载，容器里就能看到 Chroma 的持久化文件。
- **容器跑起来但 `/qa` 报 503 "Chroma 索引不存在"** — `_get_col()` 在 `WORKDIR / s05_vector_index / _chroma` 不存在时返 503(status 503 = Service Unavailable），提示用户先跑 s05；这是上面 lazy-load 设计的预期行为。
- **`code_01_fastapi_docker.py` 打印 `❌ .env 不存在` 直接退出** — `main()` 第 1 步 gating：`.env` 不在就不调 `docker compose up`，免得白烧镜像层。补 `.env` 后重跑。
- **`code_01_fastapi_docker.py` 打印 `❌ 索引不存在` 直接退出** — `main()` 第 2 步 gating：`s05_vector_index/_chroma` 不在就退出。先 `python s05_vector_index/code_01_vector_index.py` 重建索引。

---

## 三、核心函数一览

| 函数 / 文件 | 文件 | 输入 | 输出 | 一句话解释 |
|---|---|---|---|---|
| `app` | `app.py` | — | `FastAPI` 实例 | FastAPI 应用,`POST /qa` 端点入口(uvicorn 启动用) |
| `QARequest` | `app.py` | — | `pydantic.BaseModel` | 请求 schema:`{"question": str}`;类型错 / 缺字段 → 422 |
| `_get_col()` | `app.py` | — | `chromadb.Collection` 或抛 `HTTPException(503)` | 第一次请求时 lazy-load Chroma collection;索引不存在或 collection 缺失返 503,**省得在 import 阶段崩** |
| `qa(req)` | `app.py` | `QARequest` | `dict` (透传 s08 `answer()` 的 `{text, citations}`) | `POST /qa` handler:`_get_col → embed → hybrid_search → rerank → answer` 五步串行 |
| `main()` | `code_01_fastapi_docker.py` | — | 启动 docker compose 或友好退出 | 启动器:`.env` gating → 索引 gating → `subprocess.run(["docker", "compose", "up", "--build"])` |
| `Dockerfile` | `Dockerfile` | — | 镜像 (≈ 2GB) | `python:3.11-slim` + tesseract + build-essential + `pip install -r requirements.txt` + 业务代码 |
| `docker-compose.yml` | `docker-compose.yml` | — | 1 个 `rag` 服务跑起来 | build context=项目根、port 8000、env_file=.env、mount samples / _chroma(:ro) |

### 本章的设计取舍

s12 的代码拆得很细，每个函数 / 配置文件都对应一种"上线动作"的角色。schema 把"启动契约"封装掉，让上层只关心 HTTP 边界：

- **`QARequest` / `answer()` 输出**：`{question: str} → {text: str, citations: list[dict]}`——这是 s12 唯一的对外契约，**只透传 s08 `answer()` 的返回值**。任何上游替换（改成 Anthropic / Bedrock / Ollama）只要保持这个 schema，s12 端点形状不动。
- **`@lru_cache` 模型缓存模式从 s08 / s10 不适用于 s12**——本章不调 LLM，部署层不涉及模型加载（`app.py` 走 `embed()` 函数，模型由 s04 / s07 内部自己缓存，FastAPI 包装层不需要重复缓存）。`embedding-routing` 这种模型生命周期管理是完整生产化方案（见下条扩展指南），不在 MVP 范围。
- **`build context = 项目根` 显式声明**——`docker-compose.yml` 里 `build.context: ..` 是显式写在配置文件里的，不能隐式靠"Dockerfile 在哪就 build 哪"。这保证 Dockerfile 里的 `COPY .. /app` 能完整带进整个仓库代码，s04-s11 的所有依赖都在镜像里。
- **`env_file: ../.env` 相对路径**——compose 文件路径解析，不是相对启动 compose 的 cwd 解析。这让"在项目根跑 compose"和"在 s12 目录跑 compose"行为一致，**避免本地能跑 CI 跑挂**。
- **volume 只读挂载（`:ro`)**——索引和样本目录都 `:ro` 挂载，容器不会误写、重 build 不丢数据。多副本读同一份索引也能工作（Chroma 允许多读单写）。
- **`_get_col()` lazy-load 而非 import 阶段加载**——`_get_col()` 在第一次请求时才 `chromadb.PersistentClient(...)`，**而不是在 `app.py` 模块加载时就调**。这意味着：① import 阶段不会因 `_chroma/` 不存在而崩（只是延迟到第一次请求）；② 容器启起来不一定需要索引存在（`_get_col` 返回 `None` 时给 503 而不是 traceback）；③ 索引重建后，容器下次请求就能拿到新索引，不需要重启容器。
- **`POST /qa` 只暴露一个问题字段**：不在 schema 里加 `top_k` / `temperature` / `model` 等调参入口——MVP 阶段接口形状保持最小，生产再加。FastAPI 的 `pydantic` 校验保证缺字段 / 类型错返 422 而不是 500。

如果你的场景需要"鉴权 / rate limit / 多租户"，就在 `app.py` 加 `@app.middleware` 或 `Depends(get_api_key)`——但**保持 `POST /qa` 端点形状是 `{question} → {text, citations}` 不变**，鉴权层透明加挂，不要替换 schema。

---

## RAGFlow 实现

RAGFlow 的部署在 `docker/` 目录：docker-compose 把 API 服务 + Elasticsearch + Redis + MinIO（对象存储）+ 可选 Infinity 编排成多容器，docker-compose.healthcheck + restart： always 保证服务高可用。`.env` 通过 docker secrets 注入，不进镜像。

**设计取舍**：多容器编排对应 RAGFlow 的多组件依赖（API 调 LLM、检索查 ES、文件存 MinIO、会话存 Redis）——单容器只够 toy，进生产必须按组件拆。s12 toy 的单 `docker-compose.yml` + 2 服务（api + chroma）是这条主线的最简版。

详细摘录与 5-15 行 "为什么这样写" 的分析见 [`docs/reference/ragflow-notes/deployment.md`](../docs/reference/ragflow-notes/deployment.md)。

---

## 选型速记

### 主流部署范式速览

下面这张表把 RAG 系统的部署路径按"服务数量 / 存储 / 模型部署 / 监控 / 适用场景"列出来：

| 范式 | 服务数量 | 存储 | 模型部署 | 监控 | 适用场景 |
|---|---|---|---|---|---|
| **单容器 FastAPI(本章 MVP)** | 1 | 本地文件 + 内存 | 进程内 `import` | 无 | 教学 / 快速原型 / demo / 单人用 |
| **compose + 健康检查 + 网关** | 2-3(API + 网关 + 监控) | 本地 volume | 进程内 / 模型 sidecar | Prometheus + Grafana | 小团队 / 内网 demo / 早期产品 |
| **多容器编排(docker-compose 6-10 服务)** | 6-10 | S3 / MinIO + ES | `tei-*` / `vLLM` 独立 | Prometheus + Grafana + Loki | 中小规模生产 / 单租户 |
| **K8s + 微服务** | 10+ | S3 + ES 集群 + MySQL 集群 | `tei-gpu` + vLLM + 队列 | 完整 observability 栈 | 多租户生产 / 大规模 / 高可用 |
| **Serverless + 托管向量库** | N/A(函数) | Pinecone / Weaviate Cloud | 调用托管 Embedding API | 云厂商自带 | 流量波动大 / 不想运维 / 创业 MVP |

我们的 toy `app.py` + `Dockerfile` + `docker-compose.yml` 在范式复杂度上只占第一行——**单容器 FastAPI**；完整生产方案走多容器编排，**多一道抽象就多一道观测点 + 一个失败模式**。教学 demo 选 MVP 因为它跑通快、依赖少、依赖全在 `docker compose up --build` 这一行命令里可见；**生产请按"用户量 / 数据量 / 是否多租户 / 是否要可观测"做 tier 选型**(MVP → compose + 网关 → 多容器 → K8s）。

- **教学 / 快速原型 / demo / 单人用** → 本章 MVP(1 容器 FastAPI + compose），5 分钟跑起来，curl 调得通，改代码重 build 慢但可接受；
- **小团队 / 内网 demo / 早期产品** → 加 `/healthz` + nginx 网关（鉴权 + 限流）+ Prometheus 指标，3 个服务，代码 +50 行换 +200% 可用性；
- **中小规模生产 / 单租户** → 切多容器 compose(10 服务，模型独立部署 + ES 集群 + MinIO），加 9 个容器换"能扛中等流量、能备份、能水平扩展"，运维成本 3x；
- **多租户生产 / 大规模 / 高可用** → K8s + Helm + ArgoCD（完整 CI/CD + 滚动升级 + 灰度发布 + 多区域容灾），加 1 层抽象换"能扛大规模、能自动恢复、能灰度"，运维成本 10x 但可观测性 +10x；
- **Serverless / 不想运维** → 托管向量库（Pinecone / Weaviate Cloud)+ 云函数（AWS Lambda / Vercel)+ 托管 Embedding API，**0 运维**，但**单次调用成本 3-5x 自建**、数据合规要额外评估；
- **要先看清每个边界再选** → 用本章代码入口把"1 容器"和"compose + 网关"各跑一次，对比"5 分钟跑起来"和"3 个服务 + 50 行配置"——这是最简单的"部署方案 A/B"实验。

### 扩展指南

加一个新部署目标（docker-compose 多服务 / K8s Helm chart / Serverless）只要三步：

1. **多服务 compose**：复制 `docker-compose.yml`，加 `nginx` / `prometheus` / `tei-embedding` 三个 service，`volumes` 共享 `_chroma/` + `samples/`，`depends_on` 加 healthcheck；**K8s**：写 `k8s/deployment.yaml` + `service.yaml` + `configmap.yaml`（镜像 = 当前 `Dockerfile` build 出来的），`kubectl apply -f k8s/` 一键起；**Serverless**：写 `vercel.json` 或 `serverless.yml`，把 `app.py` 的 `POST /qa` 暴露成函数，存储走托管（Pinecone / Upstash Redis）；
2. `app.py` 的 `POST /qa` 入口不用动——它只读 Chroma + 调 s07/s08 函数，`Dockerfile` 之外的部署形态都把它当"上游 + 环境变量"对待；不要在 `app.py` 里写 `if DEPLOY_MODE == "k8s": ...`——污染入口职责；
3. 给 README 加一段"它跟单容器比，赢在哪 / 输在哪"的对照（compose：本地起 3 服务 / 5 分钟 / 无 K8s 复杂度；K8s：滚动升级 + 灰度 / Helm chart 200 行；Serverless：0 运维 / 冷启动 200ms）。

不要把部署形态判断塞进 `app.py` 或 `code_01_fastapi_docker.py`——它俩只懂"FastAPI + Chroma + 单容器"。本章 MVP 只跑单容器 compose，但 `Dockerfile` 是干净的 base image，多服务 / K8s / Serverless 都从它派生。

---

## 思考题

1. **生产环境你还会加哪些服务？**
2. **为什么 `code_01_fastapi_docker.py` 在 `.env` 不存在时直接退出、不进入 docker build？**
3. **怎么让 FastAPI 服务"高可用"？**

（答案见文末「思考题答案」）

---

## 思考题答案

### Q1. 生产环境你还会加哪些服务？

把"单容器 FastAPI"推到生产，通常至少再加这四类：

#### 1. 监控 + 可观测性： Prometheus + Grafana (+ Loki 日志）

现状：容器 stdout 打印到 `docker compose up` 的终端，关掉就没了；没有任何指标暴露。

加什么：

- **Prometheus** (`prom/prometheus` 镜像） — 时序数据库，定期 scrape 各服务的 `/metrics` 端点，存 `请求量 / 延迟分位 / 错误率 / 内存占用 / GPU 利用率`。给 uvicorn 加 `prometheus-fastapi-instrumentator` 中间件一行代码就暴露 `/metrics`。
- **Grafana** (`grafana/grafana` 镜像） — 可视化。Prometheus 是数据源，画"过去 1 小时 P99 延迟""LLM 调用成功率"这种看板。
- **Loki** + Promtail — 集中日志。Promtail 跑在每个节点收集容器 stdout，Loki 索引，Grafana 搜。比"登服务器 `docker logs`"好用一万倍。

成本： 3 个镜像 + 1 个配置文件。CPU 几乎不占，内存加起来 1-2GB。

#### 2. 错误追踪： Sentry （或自建 GLITCH-Trace)

现状：容器 500 错误只在日志里，用户那边"我问了没回应"工程师完全不知。

加什么：

- **Sentry** (`sentry-self-hosted` 或 SaaS) — 前端 + 后端都装 SDK，自动捕获异常 + 上下文 （用户/请求/堆栈/环境变量白名单）。
- 关键配置：**数据脱敏** (`.env` 里的 `LLM_API_KEY` 千万别上报），**采样率** （生产建议 10%，全量打爆配额），**告警 webhook** （新错误 5 分钟内没人 ack 就 @oncall）。

成本： SaaS 免费额度够小项目；自建要 3-5 个容器。

#### 3. 独立模型服务： vLLM / TGI （替代直接 `import sentence-transformers`)

现状：Embedding 和 Rerank 都在 API 进程里同步跑，BGE-reranker-base 一次 query ~500ms，期间 API 完全被占住。

加什么：

- **vLLM** (LLM) 或 **TEI** (Embedding) / **Text Generation Inference** (Rerank) — 把模型独立成 HTTP 服务，gpu 显存占满后通过 batch + PagedAttention 撑高并发。API 进程只发 HTTP，几毫秒返回，可以同时跑几十个请求。
- 好处： ① API 进程无状态，水平扩展容易；② GPU 单独调度，不用每个 API 副本都吃显存；③ 模型版本升级只重启模型服务。

成本： 至少 1 张 GPU （本地） 或 1 个 GPU 实例 （云上 ~¥2/小时）。

#### 4. 对象存储： S3 / MinIO （替代 samples 目录挂载）

现状：`samples/` 是本地目录，容器靠 volume 挂载读。生产场景 "用户上传 1000 个 PDF"——文件存哪？

加什么：

- **S3** (AWS / 阿里云 OSS / 腾讯云 COS) — 云上托管，按量付费，11 个 9 持久性。
- **MinIO** — 自建，API 兼容 S3，放内网。最小两节点做副本，几个 TB 容量随便堆。
- 配套：上传走预签名 URL （前端直传，不经过 API），API 只存对象 key；处理流水线监听 S3 事件 (S3 Event Notification) 触发解析。

成本： S3 几分钱 GB/月；MinIO 自建主要是磁盘钱。

#### MVP 现状 vs 生产的最小可用集

| 组件 | MVP | 生产最小集 |
|---|---|---|
| API 服务 | 1 个 FastAPI 容器 | 多个 + 网关 |
| 监控 | 无 | Prometheus + Grafana |
| 日志 | 容器 stdout | Loki / ELK |
| 错误追踪 | 无 | Sentry |
| 模型推理 | 进程内 | vLLM / TGI 独立服务 |
| 文件存储 | samples 目录 | S3 / MinIO |
| 鉴权 | 无 | API key / JWT / OAuth2 |
| 队列 | 无 | Celery / Redis Streams |

MVP 是"能跑"，上面这些是"能上线"。中间还隔着一个"能卖钱"——那个阶段要做的事情更多 (A/B 测试 / 灰度发布 / 多区域容灾），但和 RAG 本身没关系了，属于通用 SaaS 工程范畴。

### Q2. 为什么 `code_01_fastapi_docker.py` 在 `.env` 不存在时直接退出、不进入 docker build?

省镜像层缓存——如果 `env_file: ../.env` 缺失，构建会在 `pip install` 之后才发现 `.env` 读不到，**整个镜像烧完才发现跑不起来**，得 `docker compose down --rmi all` 清理才能重来。提前在 host 上 gating 失败可以**只浪费 1 行 print 的时间、不浪费 2GB 镜像层**。同理索引 gating——`s05_vector_index/_chroma` 不存在时容器启动后第一次 `/qa` 才会暴露问题，**提前 gating 把"运行时报错"变成"启动时可见"**，跟 s11 中 OCR 部分缺包 / 缺二进制 / 缺图三类异常 catch 是同一种"早暴露"思路。

### Q3. 怎么让 FastAPI 服务"高可用"?

3 个方向叠加：① 进程级——加 `restart: unless-stopped`(compose 字段，容器崩了自动拉起）+ 加 `/healthz` 端点给 K8s liveness probe（崩了不优雅退出会被 probe 杀掉）；② 流量级——compose scale 起来多副本（`docker compose up --scale rag=3`），前面套 nginx / APISIX 做负载均衡 + 健康检查剔除坏副本；③ 数据级——Chroma 索引是本地文件，**多副本同时读没问题，同时写会锁**。生产里换 ES / Milvus 这种原生支持分布式 + 主从复制的引擎，`depends_on: condition: service_healthy` 串启动顺序 + `replicas: 3` 设副本数。MVP 不做高可用——单进程单容器，崩了手动 `docker compose up` 重启即可。
下一章 — 这一节把"召回 → 排序 → 生成 → 服务化"中的某一环跑通,留下 +1 章填下一档的实现;每加一档,缺失上层就越明显,直到 s12 把所有环节收敛到 FastAPI 服务。

### troubleshooting

- **`docker: command not found`** — 没装 Docker Desktop。本章需要它，装好之后 `docker --version` 应该能跑。Windows 用户注意 WSL2 后端。
- **`failed to solve: failed to read dockerfile`** — `docker-compose.yml` 里 `build.context` 和 `dockerfile` 路径对不上。本章的 compose 已经显式写成 `dockerfile: s12_deployment/Dockerfile`，别手贱改回 `build: ..`（那样 Docker 会去项目根找 `Dockerfile`，找不到）。
- **build 时 `pip install` 报 chroma-hnswlib 编译错** — Dockerfile 里装了 `build-essential` 就是给这个兜底的；如果还报错，通常是网络问题，换 pip 镜像源。
- **`/qa` 返回 500 + "LLM_API_KEY not set"** — 容器没拿到 `.env`。compose 里 `env_file: ../.env` 是相对 compose 文件位置的相对路径，确认 `.env` 在项目根。
- **`/qa` 返回 500 + "Collection docs does not exist"** — 挂载路径不对。`s05_vector_index/_chroma` 必须先在主机上存在（跑过 s05），compose 里是 `:ro` 只读挂载，容器里就能看到 Chroma 的持久化文件。
- **容器跑起来但 `/qa` 报 503 "Chroma 索引不存在"** — `_get_col()` 在 `WORKDIR / s05_vector_index / _chroma` 不存在时返 503(status 503 = Service Unavailable），提示用户先跑 s05；这是上面 lazy-load 设计的预期行为。
- **`code_01_fastapi_docker.py` 打印 `❌ .env 不存在` 直接退出** — `main()` 第 1 步 gating：`.env` 不在就不调 `docker compose up`，免得白烧镜像层。补 `.env` 后重跑。
- **`code_01_fastapi_docker.py` 打印 `❌ 索引不存在` 直接退出** — `main()` 第 2 步 gating：`s05_vector_index/_chroma` 不在就退出。先 `python s05_vector_index/code_01_vector_index.py` 重建索引。

---

## 三、核心函数一览

| 函数 / 文件 | 文件 | 输入 | 输出 | 一句话解释 |
|---|---|---|---|---|
| `app` | `app.py` | — | `FastAPI` 实例 | FastAPI 应用,`POST /qa` 端点入口(uvicorn 启动用) |
| `QARequest` | `app.py` | — | `pydantic.BaseModel` | 请求 schema:`{"question": str}`;类型错 / 缺字段 → 422 |
| `_get_col()` | `app.py` | — | `chromadb.Collection` 或抛 `HTTPException(503)` | 第一次请求时 lazy-load Chroma collection;索引不存在或 collection 缺失返 503,**省得在 import 阶段崩** |
| `qa(req)` | `app.py` | `QARequest` | `dict` (透传 s08 `answer()` 的 `{text, citations}`) | `POST /qa` handler:`_get_col → embed → hybrid_search → rerank → answer` 五步串行 |
| `main()` | `code_01_fastapi_docker.py` | — | 启动 docker compose 或友好退出 | 启动器:`.env` gating → 索引 gating → `subprocess.run(["docker", "compose", "up", "--build"])` |
| `Dockerfile` | `Dockerfile` | — | 镜像 (≈ 2GB) | `python:3.11-slim` + tesseract + build-essential + `pip install -r requirements.txt` + 业务代码 |
| `docker-compose.yml` | `docker-compose.yml` | — | 1 个 `rag` 服务跑起来 | build context=项目根、port 8000、env_file=.env、mount samples / _chroma(:ro) |

### 本章的设计取舍

s12 的代码拆得很细，每个函数 / 配置文件都对应一种"上线动作"的角色。schema 把"启动契约"封装掉，让上层只关心 HTTP 边界：

- **`QARequest` / `answer()` 输出**：`{question: str} → {text: str, citations: list[dict]}`——这是 s12 唯一的对外契约，**只透传 s08 `answer()` 的返回值**。任何上游替换（改成 Anthropic / Bedrock / Ollama）只要保持这个 schema，s12 端点形状不动。
- **`@lru_cache` 模型缓存模式从 s08 / s10 不适用于 s12**——本章不调 LLM，部署层不涉及模型加载（`app.py` 走 `embed()` 函数，模型由 s04 / s07 内部自己缓存，FastAPI 包装层不需要重复缓存）。`embedding-routing` 这种模型生命周期管理是完整生产化方案（见下条扩展指南），不在 MVP 范围。
- **`build context = 项目根` 显式声明**——`docker-compose.yml` 里 `build.context: ..` 是显式写在配置文件里的，不能隐式靠"Dockerfile 在哪就 build 哪"。这保证 Dockerfile 里的 `COPY .. /app` 能完整带进整个仓库代码，s04-s11 的所有依赖都在镜像里。
- **`env_file: ../.env` 相对路径**——compose 文件路径解析，不是相对启动 compose 的 cwd 解析。这让"在项目根跑 compose"和"在 s12 目录跑 compose"行为一致，**避免本地能跑 CI 跑挂**。
- **volume 只读挂载（`:ro`)**——索引和样本目录都 `:ro` 挂载，容器不会误写、重 build 不丢数据。多副本读同一份索引也能工作（Chroma 允许多读单写）。
- **`_get_col()` lazy-load 而非 import 阶段加载**——`_get_col()` 在第一次请求时才 `chromadb.PersistentClient(...)`，**而不是在 `app.py` 模块加载时就调**。这意味着：① import 阶段不会因 `_chroma/` 不存在而崩（只是延迟到第一次请求）；② 容器启起来不一定需要索引存在（`_get_col` 返回 `None` 时给 503 而不是 traceback）；③ 索引重建后，容器下次请求就能拿到新索引，不需要重启容器。
- **`POST /qa` 只暴露一个问题字段**：不在 schema 里加 `top_k` / `temperature` / `model` 等调参入口——MVP 阶段接口形状保持最小，生产再加。FastAPI 的 `pydantic` 校验保证缺字段 / 类型错返 422 而不是 500。

如果你的场景需要"鉴权 / rate limit / 多租户"，就在 `app.py` 加 `@app.middleware` 或 `Depends(get_api_key)`——但**保持 `POST /qa` 端点形状是 `{question} → {text, citations}` 不变**，鉴权层透明加挂，不要替换 schema。

---

## RAGFlow 实现

RAGFlow 的部署在 `docker/` 目录：docker-compose 把 API 服务 + Elasticsearch + Redis + MinIO（对象存储）+ 可选 Infinity 编排成多容器，docker-compose.healthcheck + restart： always 保证服务高可用。`.env` 通过 docker secrets 注入，不进镜像。

**设计取舍**：多容器编排对应 RAGFlow 的多组件依赖（API 调 LLM、检索查 ES、文件存 MinIO、会话存 Redis）——单容器只够 toy，进生产必须按组件拆。s12 toy 的单 `docker-compose.yml` + 2 服务（api + chroma）是这条主线的最简版。

详细摘录与 5-15 行 "为什么这样写" 的分析见 [`docs/reference/ragflow-notes/deployment.md`](../docs/reference/ragflow-notes/deployment.md)。

---

## 选型速记

### 主流部署范式速览

下面这张表把 RAG 系统的部署路径按"服务数量 / 存储 / 模型部署 / 监控 / 适用场景"列出来：

| 范式 | 服务数量 | 存储 | 模型部署 | 监控 | 适用场景 |
|---|---|---|---|---|---|
| **单容器 FastAPI(本章 MVP)** | 1 | 本地文件 + 内存 | 进程内 `import` | 无 | 教学 / 快速原型 / demo / 单人用 |
| **compose + 健康检查 + 网关** | 2-3(API + 网关 + 监控) | 本地 volume | 进程内 / 模型 sidecar | Prometheus + Grafana | 小团队 / 内网 demo / 早期产品 |
| **多容器编排(docker-compose 6-10 服务)** | 6-10 | S3 / MinIO + ES | `tei-*` / `vLLM` 独立 | Prometheus + Grafana + Loki | 中小规模生产 / 单租户 |
| **K8s + 微服务** | 10+ | S3 + ES 集群 + MySQL 集群 | `tei-gpu` + vLLM + 队列 | 完整 observability 栈 | 多租户生产 / 大规模 / 高可用 |
| **Serverless + 托管向量库** | N/A(函数) | Pinecone / Weaviate Cloud | 调用托管 Embedding API | 云厂商自带 | 流量波动大 / 不想运维 / 创业 MVP |

我们的 toy `app.py` + `Dockerfile` + `docker-compose.yml` 在范式复杂度上只占第一行——**单容器 FastAPI**；完整生产方案走多容器编排，**多一道抽象就多一道观测点 + 一个失败模式**。教学 demo 选 MVP 因为它跑通快、依赖少、依赖全在 `docker compose up --build` 这一行命令里可见；**生产请按"用户量 / 数据量 / 是否多租户 / 是否要可观测"做 tier 选型**(MVP → compose + 网关 → 多容器 → K8s）。

- **教学 / 快速原型 / demo / 单人用** → 本章 MVP(1 容器 FastAPI + compose），5 分钟跑起来，curl 调得通，改代码重 build 慢但可接受；
- **小团队 / 内网 demo / 早期产品** → 加 `/healthz` + nginx 网关（鉴权 + 限流）+ Prometheus 指标，3 个服务，代码 +50 行换 +200% 可用性；
- **中小规模生产 / 单租户** → 切多容器 compose(10 服务，模型独立部署 + ES 集群 + MinIO），加 9 个容器换"能扛中等流量、能备份、能水平扩展"，运维成本 3x；
- **多租户生产 / 大规模 / 高可用** → K8s + Helm + ArgoCD（完整 CI/CD + 滚动升级 + 灰度发布 + 多区域容灾），加 1 层抽象换"能扛大规模、能自动恢复、能灰度"，运维成本 10x 但可观测性 +10x；
- **Serverless / 不想运维** → 托管向量库（Pinecone / Weaviate Cloud)+ 云函数（AWS Lambda / Vercel)+ 托管 Embedding API，**0 运维**，但**单次调用成本 3-5x 自建**、数据合规要额外评估；
- **要先看清每个边界再选** → 用本章代码入口把"1 容器"和"compose + 网关"各跑一次，对比"5 分钟跑起来"和"3 个服务 + 50 行配置"——这是最简单的"部署方案 A/B"实验。

### 扩展指南

加一个新部署目标（docker-compose 多服务 / K8s Helm chart / Serverless）只要三步：

1. **多服务 compose**：复制 `docker-compose.yml`，加 `nginx` / `prometheus` / `tei-embedding` 三个 service，`volumes` 共享 `_chroma/` + `samples/`，`depends_on` 加 healthcheck；**K8s**：写 `k8s/deployment.yaml` + `service.yaml` + `configmap.yaml`（镜像 = 当前 `Dockerfile` build 出来的），`kubectl apply -f k8s/` 一键起；**Serverless**：写 `vercel.json` 或 `serverless.yml`，把 `app.py` 的 `POST /qa` 暴露成函数，存储走托管（Pinecone / Upstash Redis）；
2. `app.py` 的 `POST /qa` 入口不用动——它只读 Chroma + 调 s07/s08 函数，`Dockerfile` 之外的部署形态都把它当"上游 + 环境变量"对待；不要在 `app.py` 里写 `if DEPLOY_MODE == "k8s": ...`——污染入口职责；
3. 给 README 加一段"它跟单容器比，赢在哪 / 输在哪"的对照（compose：本地起 3 服务 / 5 分钟 / 无 K8s 复杂度；K8s：滚动升级 + 灰度 / Helm chart 200 行；Serverless：0 运维 / 冷启动 200ms）。

不要把部署形态判断塞进 `app.py` 或 `code_01_fastapi_docker.py`——它俩只懂"FastAPI + Chroma + 单容器"。本章 MVP 只跑单容器 compose，但 `Dockerfile` 是干净的 base image，多服务 / K8s / Serverless 都从它派生。

---

## 思考题

1. **生产环境你还会加哪些服务？**
2. **为什么 `code_01_fastapi_docker.py` 在 `.env` 不存在时直接退出、不进入 docker build？**
3. **怎么让 FastAPI 服务"高可用"？**

（答案见文末「思考题答案」）

---

## 思考题答案

### Q1. 生产环境你还会加哪些服务？

把"单容器 FastAPI"推到生产，通常至少再加这四类：

#### 1. 监控 + 可观测性： Prometheus + Grafana (+ Loki 日志）

现状：容器 stdout 打印到 `docker compose up` 的终端，关掉就没了；没有任何指标暴露。

加什么：

- **Prometheus** (`prom/prometheus` 镜像） — 时序数据库，定期 scrape 各服务的 `/metrics` 端点，存 `请求量 / 延迟分位 / 错误率 / 内存占用 / GPU 利用率`。给 uvicorn 加 `prometheus-fastapi-instrumentator` 中间件一行代码就暴露 `/metrics`。
- **Grafana** (`grafana/grafana` 镜像） — 可视化。Prometheus 是数据源，画"过去 1 小时 P99 延迟""LLM 调用成功率"这种看板。
- **Loki** + Promtail — 集中日志。Promtail 跑在每个节点收集容器 stdout，Loki 索引，Grafana 搜。比"登服务器 `docker logs`"好用一万倍。

成本： 3 个镜像 + 1 个配置文件。CPU 几乎不占，内存加起来 1-2GB。

#### 2. 错误追踪： Sentry （或自建 GLITCH-Trace)

现状：容器 500 错误只在日志里，用户那边"我问了没回应"工程师完全不知。

加什么：

- **Sentry** (`sentry-self-hosted` 或 SaaS) — 前端 + 后端都装 SDK，自动捕获异常 + 上下文 （用户/请求/堆栈/环境变量白名单）。
- 关键配置：**数据脱敏** (`.env` 里的 `LLM_API_KEY` 千万别上报），**采样率** （生产建议 10%，全量打爆配额），**告警 webhook** （新错误 5 分钟内没人 ack 就 @oncall）。

成本： SaaS 免费额度够小项目；自建要 3-5 个容器。

#### 3. 独立模型服务： vLLM / TGI （替代直接 `import sentence-transformers`)

现状：Embedding 和 Rerank 都在 API 进程里同步跑，BGE-reranker-base 一次 query ~500ms，期间 API 完全被占住。

加什么：

- **vLLM** (LLM) 或 **TEI** (Embedding) / **Text Generation Inference** (Rerank) — 把模型独立成 HTTP 服务，gpu 显存占满后通过 batch + PagedAttention 撑高并发。API 进程只发 HTTP，几毫秒返回，可以同时跑几十个请求。
- 好处： ① API 进程无状态，水平扩展容易；② GPU 单独调度，不用每个 API 副本都吃显存；③ 模型版本升级只重启模型服务。

成本： 至少 1 张 GPU （本地） 或 1 个 GPU 实例 （云上 ~¥2/小时）。

#### 4. 对象存储： S3 / MinIO （替代 samples 目录挂载）

现状：`samples/` 是本地目录，容器靠 volume 挂载读。生产场景 "用户上传 1000 个 PDF"——文件存哪？

加什么：

- **S3** (AWS / 阿里云 OSS / 腾讯云 COS) — 云上托管，按量付费，11 个 9 持久性。
- **MinIO** — 自建，API 兼容 S3，放内网。最小两节点做副本，几个 TB 容量随便堆。
- 配套：上传走预签名 URL （前端直传，不经过 API），API 只存对象 key；处理流水线监听 S3 事件 (S3 Event Notification) 触发解析。

成本： S3 几分钱 GB/月；MinIO 自建主要是磁盘钱。

#### MVP 现状 vs 生产的最小可用集

| 组件 | MVP | 生产最小集 |
|---|---|---|
| API 服务 | 1 个 FastAPI 容器 | 多个 + 网关 |
| 监控 | 无 | Prometheus + Grafana |
| 日志 | 容器 stdout | Loki / ELK |
| 错误追踪 | 无 | Sentry |
| 模型推理 | 进程内 | vLLM / TGI 独立服务 |
| 文件存储 | samples 目录 | S3 / MinIO |
| 鉴权 | 无 | API key / JWT / OAuth2 |
| 队列 | 无 | Celery / Redis Streams |

MVP 是"能跑"，上面这些是"能上线"。中间还隔着一个"能卖钱"——那个阶段要做的事情更多 (A/B 测试 / 灰度发布 / 多区域容灾），但和 RAG 本身没关系了，属于通用 SaaS 工程范畴。

### Q2. 为什么 `code_01_fastapi_docker.py` 在 `.env` 不存在时直接退出、不进入 docker build?

省镜像层缓存——如果 `env_file: ../.env` 缺失，构建会在 `pip install` 之后才发现 `.env` 读不到，**整个镜像烧完才发现跑不起来**，得 `docker compose down --rmi all` 清理才能重来。提前在 host 上 gating 失败可以**只浪费 1 行 print 的时间、不浪费 2GB 镜像层**。同理索引 gating——`s05_vector_index/_chroma` 不存在时容器启动后第一次 `/qa` 才会暴露问题，**提前 gating 把"运行时报错"变成"启动时可见"**，跟 s11 中 OCR 部分缺包 / 缺二进制 / 缺图三类异常 catch 是同一种"早暴露"思路。

### Q3. 怎么让 FastAPI 服务"高可用"?

3 个方向叠加：① 进程级——加 `restart: unless-stopped`(compose 字段，容器崩了自动拉起）+ 加 `/healthz` 端点给 K8s liveness probe（崩了不优雅退出会被 probe 杀掉）；② 流量级——compose scale 起来多副本（`docker compose up --scale rag=3`），前面套 nginx / APISIX 做负载均衡 + 健康检查剔除坏副本；③ 数据级——Chroma 索引是本地文件，**多副本同时读没问题，同时写会锁**。生产里换 ES / Milvus 这种原生支持分布式 + 主从复制的引擎，`depends_on: condition: service_healthy` 串启动顺序 + `replicas: 3` 设副本数。MVP 不做高可用——单进程单容器，崩了手动 `docker compose up` 重启即可。