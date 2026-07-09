# s12 部署 — Docker Compose 把 MVP 包成服务

> **章节定位**：RAG 的"上线面"。s08 的 `answer()` 只能在命令行跑 `python s08_prompt_generate/code.py`，然后交互式输入问题——s12 把这条链路包成一个 HTTP 服务，再装进容器里，**让同事 / 前端 / 其他服务用 `curl` 就能调**，不用装 Python、不用下模型、不用配索引。**整章就 1 个 unit、4 个文件**（`app.py` + `Dockerfile` + `docker-compose.yml` + 启动器 `code.py`），因为 `docker compose up --build` 本身已经是"上线"动作的最小形式——没有"由浅入深"的多档梯度。
>
> **章节结构**：本章是项目里唯一一个 1-unit 章节。**unit 01**（`fastapi_docker`）一次性讲完 FastAPI 包装 + Dockerfile + docker-compose + 启动器代码 4 件事；对照 s12 的聚合入口 `code.py`（importlib 委托到 unit 01）就是同一份代码。**scope 注意**：本章只演示"单容器 FastAPI"——没有 health check / 没有鉴权 / 没有指标暴露 / 没有模型独立部署 / 没有对象存储 / 没有任务队列；RAGFlow 把这套系统拆成 10+ 容器（ES / MySQL / Redis / MinIO / sandbox / 主 API / OCR / 监控）协同工作，详见 §四。

---

## 章节导航

| Unit | 标题 | 入口 |
| --- | --- | --- |
| 01 | fastapi_docker（FastAPI 包装 + Docker Compose 一键起；含 4 个 supporting files） | [`units/01_fastapi_docker/code.py`](units/01_fastapi_docker/code.py) |

跑法（任选一种）：

```bash
# 路径 A: 走聚合入口（importlib 委托到 unit 01）
python s12_deployment/code.py

# 路径 B: 走单元入口
python s12_deployment/units/01_fastapi_docker/code.py
```

两个入口跑同一份逻辑：先做 `.env` 和 `_chroma` 前置校验，通过后调 `docker compose up --build`。

依赖：`fastapi` / `uvicorn` / `chromadb`（已在 `requirements.txt`）；`Docker Desktop`（系统级，本章最大的硬依赖）；项目根 `.env` 配 `LLM_API_KEY`；`s05_vector_index/_chroma` 索引已生成。

样本输入：调起来后任意 `curl -X POST http://localhost:8000/qa -d '{"question":"项目里有哪些表?"}'` 即可；本章不写自己的样本（上游 s04-s08 已在 `samples/` 上跑通）。

---

## 一、什么是"部署 MVP"？

### 1.1 核心定义

**部署 MVP（Deployment MVP）** 是把"本地命令行能跑"升级为"HTTP 服务能调"的过程——核心动作有三件：① 用 **FastAPI**（或 Flask / Django）把 s04 → s06 → s07 → s08 这条链路包成 `POST /qa {question: ...} → {text, citations}`；② 用 **Dockerfile** 把 Python 运行时 + 业务代码 + 系统依赖（tesseract）一起烧进镜像；③ 用 **docker-compose.yml** 声明端口、卷挂载、环境变量，让 `docker compose up --build` 一条命令拉起整个服务。本章的 4 个文件就是这三件动作的最小落地：

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

把它放进 RAG 全景看：**s12 是把 s08 的 `answer()` 函数从"我在自己机器上跑得通"升级为"别人用 `curl` 就能调"**——同时把"装 Python / 下模型 / 重建索引"这些环境成本转嫁给 Docker 镜像。

> 💡 **一句话总结**：部署 MVP = FastAPI 包装（HTTP 入口）+ Dockerfile（镜像）+ docker-compose.yml（编排）+ 启动器（前置校验）。
>
> 让 RAG 从"我在自己机器上跑得通"升级为"别人用 `curl` 就能调"——同时把"配环境"成本转嫁给 Docker。

部署不是把代码"传上去"就完了——它是一个**契约**：服务端要保证镜像能跑、端口能访问、索引能读、LLM key 能用；客户端只要发 HTTP 就能拿到答案。契约的两端都不可见对方的实现细节。

### 1.2 FastAPI 包装的端点 schema

`app.py` 暴露的 `POST /qa` schema 长这样——`{question: str} → {text: str, citations: list[dict]}`：

```python
# 请求
class QARequest(BaseModel):
    question: str

# 响应（透传 s08 answer() 的返回值）
{"text": "项目里包含：规格表、内存表、电源表...", "citations": [{"source": "samples/server_whitepaper.pdf", "page": 2, "text": "..."}]}
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

关键设计：① **基础镜像层（1-2）** 几乎不变——`python:3.11-slim` 拉一次就缓存；② **依赖层（3-4）** 只在 `requirements.txt` 改了才重 build——这是为什么 `COPY requirements.txt .` 和 `RUN pip install` 必须分两步（合并成一步会导致代码改动也触发 pip 整条重装）；③ **代码层（5）** 是"高频改动层"——业务代码改了重 build 第 5 层，前 4 层全部命中缓存。MVP 不分模型层（`bge` 权重 ~100MB、`bge-reranker-base` ~1GB）——这些随 `pip install` 一起烧进第 4 层，**改一行代码 `docker compose build` 就触发完整重 build**，是 demo 的代价。生产里模型走 volume / S3 挂载，详见 §四。

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

关键设计：① **build.context 是项目根**——这样 Dockerfile 能 `COPY .. /app` 把整个仓库代码带进镜像，包括 s04-s11 的所有依赖；② **`env_file: ../.env`**——容器内 `/app/.env` 自动被 Docker 加载，s08 进程读 `LLM_API_KEY` 跟本地一致；③ **只读挂载**（`:ro`）——容器不会误写索引 / 样本，重 build 不丢数据；④ **路径用 `../` 前缀**——相对 `docker-compose.yml` 文件位置解析，本地跑和 CI 跑行为一致。

> 💡 YAML 声明式编排不是 docker-compose 独有——K8s manifests / Helm charts / Nomad HCL 都是同一种思路。MVP 走 compose 是因为它零依赖、单文件、上手 5 分钟。

---

## 二、为什么要单独写一章"部署"？

`app.py` 51 行 + `Dockerfile` 6 行 + `docker-compose.yml` 11 行 + `code.py` 15 行 = 83 行就能让 s08 的 `answer()` 变成"curl 调得通"的服务。看起来不值得单独一章。但把它放进"前 11 章都在命令行跑"对照看会发现：**"本机能跑"和"别人能用"是两类交付**——这道鸿沟由 4 类典型失败堆起来。

### 2.1 真实世界的问题（4 条典型）

1. **Python / 依赖 / 索引要对方全装一遍**——同事 clone 仓库后要 `pip install -r requirements.txt` + `cp .env.example .env` 填 key + 跑 s04/s05 重建索引 + 下 BGE 模型权重（~100MB）+ 下 bge-reranker-base（~1GB）——**5 个步骤任一卡住就报 traceback**。**生产解法**：用 Docker 把 Python + 系统依赖 + pip 包 + 业务代码 + 模型权重全部装进镜像，对方只要 `docker compose up --build` 一行命令。RAGFlow `docker-compose-base.yml` 把 ES / MySQL / MinIO / Redis / 模型服务全部跑进容器，对端"装一个 Docker 就能用"。
2. **没法给前端 / 其他服务调**——命令行交互式输入只适合"开发者自己玩"。前端要做个搜索框、其他服务要做 API 集成，**没有 HTTP 端点就只能写文件 / 跑子进程**，耦合重。**生产解法**：用 FastAPI（pydantic 强类型 + OpenAPI 文档自动生成）把 `answer()` 包成 `POST /qa`，`curl` / fetch / requests 都能调。RAGFlow 的 `ragflow-cpu` 容器也是同款思路（FastAPI + uvicorn + nginx 静态资源）。
3. **重启 = 数据丢失 / 索引丢失**——本地跑 `python s08_prompt_generate/code.py` 时索引在内存；kill 重启就重建。多人协作 / 服务器跑就尴尬。**生产解法**：把 Chroma 索引目录用 volume 挂载进容器（`:ro` 只读），容器重启不丢、重新 build 不丢、多副本读同一份索引。RAGFlow 把 ES 索引挂成 named volume + multi-node 副本，配置在 `docker-compose-base.yml` 的 `volumes` 段。
4. **冷启动慢 + 没有任何观测**——`POST /qa` 第一次请求 Chroma hnswlib 要 mmap 索引文件，3-5 秒延迟；之后没人知道"现在 /qa 平均延迟多少 / 错误率多少 / LLM 调用成本多少"。**生产解法**：① 容器预热（`docker compose run --rm rag python -c "import chromadb; chromadb.PersistentClient('...').get_collection('docs')"` 预热后 SIGTERM，再起服务）；② 加 `/healthz` 端点给 K8s liveness probe；③ 暴露 Prometheus `/metrics` 抓 `请求量 / 延迟分位 / 错误率 / LLM token 消耗`。RAGFlow 走完整 observability 栈（Loki / Prometheus / Grafana），MVP 都不带。

### 2.2 为什么必须在部署上显式投入

每条失败模式都对应一种工业级解法——Docker 镜像打包、HTTP 端点暴露、volume 挂载持久化、预热 + 监控可观测。**s12 的目标不是解决它们，而是把它们显式暴露出来，让你看到"本机命令行 RAG"和"生产 HTTP RAG"之间的边界**。这跟 s10 把"向量召回答不全'实体之间关系'"、s11 把"表格被拍扁 / 扫描件返回空"显式对比是同一种思路——**叙述载体从"图函数 + 1 跳 query"换成"FastAPI 包装 + Docker 镜像 + compose 编排 + 启动器"，但"先跑通 toy、再讲清楚 toy 在哪里会塌"的教学哲学是一致的**。

这也是为什么本章只拆 **1 个 unit** 而不是 2-3 个——

- 整章就 4 个文件 + 1 个启动器，组合只有一种（"FastAPI + Dockerfile + compose + 启动器"），没有"由浅入深"的多档梯度。`docker compose up --build` 是终点，没有中间状态可设——你不会"先只跑 FastAPI 不跑 Docker"，也不会"先跑 compose 不跑 FastAPI"——它们是绑定的。
- 对照 s09 (Agent) 的 2 unit（"先 LLM 工具调用骨架、后 REPL 交互"）、s10 的 2 unit（"先 LLM 抽三元组、后 1 跳 query"）、s11 的 2 unit（"先表格、后 OCR"）——它们都有"无 LLM 走通 / 有 LLM 走通"或"纯文本 / 多模态"的天然分层。s12 没有这种分层，因为**部署的"深"是工程问题（多容器 / 监控 / 鉴权）而不是代码问题**——后者用 1 个 unit 教完前者 80%，剩下 20%（生产化：多容器拆分 + 健康检查 + 鉴权网关）按需切。

> 💡 "1 unit" 不是偷懒——它是诚实的工程表达。教学 demo 选 1 unit 是因为没有可拆的多档梯度；生产化（拆 10+ 容器）才需要分阶段讲，对应的是 §四 而不是 unit 拆分。

---

## 三、怎么做？

### 3.1 章节导航

| Unit | 主题 | 它解决什么 |
|---|---|---|
| [01_fastapi_docker](./units/01_fastapi_docker/README.md) | FastAPI 包装 + Dockerfile + docker-compose + 启动器 | "本机命令行 RAG" → "curl 调得通的 HTTP 服务" |

### 3.2 跑起来

```bash
# 0. 前提: 跑通 s04 / s05 / s07 (生成索引 + 装好 bge)
# 1. 配 .env (项目根)
cp .env.example .env   # 填 LLM_API_KEY

# 2. 启动 (任选一种)
python s12_deployment/code.py
#   或直接走单元入口:
python s12_deployment/units/01_fastapi_docker/code.py
# 内部执行 docker compose up --build,会自动 build 镜像 + 启服务

# 3. 测一发
curl -X POST http://localhost:8000/qa \
  -H 'Content-Type: application/json' \
  -d '{"question":"项目里有哪些表?"}'

# 4. 停掉
# 在跑 code.py 的终端 Ctrl-C
```

环境变量：`LLM_API_KEY` 必填（容器通过 `env_file: ../.env` 加载到进程环境）；`LLM_BASE_URL` / `LLM_MODEL` 可选（默认值见 s08）。

无 Docker 环境：本章的启动器会卡在 `docker: command not found`（subprocess 抛错前 `which docker` 检查）——这是预期，§3.4 troubleshooting 给解法。

无 `.env` / 无索引环境：启动器**先报错、不进入 docker build**（§3.3 启动器代码的第 1 / 2 步 gating），省得 Docker 层缓存一堆无效镜像、又得 `docker compose down --rmi all` 清理。**`/dev/null` 沙箱里通常打到这一步就退出**。

### 3.3 核心函数 / 组件一览

s12 的代码拆得很细，每个函数 / 配置文件都对应一种"上线动作"的角色：

| 函数 / 文件 | 文件 | 输入 | 输出 | 一句话解释 |
|---|---|---|---|---|
| `app` | `app.py` | — | `FastAPI` 实例 | FastAPI 应用，`POST /qa` 端点入口（uvicorn 启动用） |
| `QARequest` | `app.py` | — | `pydantic.BaseModel` | 请求 schema：`{"question": str}`；类型错 / 缺字段 → 422 |
| `_get_col()` | `app.py` | — | `chromadb.Collection` 或抛 `HTTPException(503)` | 第一次请求时 lazy-load Chroma collection；索引不存在或 collection 缺失返 503，**省得在 import 阶段崩** |
| `qa(req)` | `app.py` | `QARequest` | `dict` (透传 s08 `answer()` 的 `{text, citations}`) | `POST /qa` handler：`_get_col → embed → hybrid_search → rerank → answer` 五步串行 |
| `main()` | `units/01_fastapi_docker/code.py` | — | 启动 docker compose 或友好退出 | 启动器：`.env` gating → 索引 gating → `subprocess.run(["docker", "compose", "up", "--build"])` |
| `code.py` (顶层) | `code.py` | — | 等价于 unit 01 `main()` | 聚合入口：`importlib` 加载 unit 01 后调其 `main()`（与 s08 / s09 / s10 / s11 同款 importlib 委托） |
| `Dockerfile` | `Dockerfile` | — | 镜像 (≈ 2GB) | `python:3.11-slim` + tesseract + build-essential + `pip install -r requirements.txt` + 业务代码 |
| `docker-compose.yml` | `docker-compose.yml` | — | 1 个 `rag` 服务跑起来 | build context=项目根、port 8000、env_file=.env、mount samples / _chroma（:ro） |

注：**`@lru_cache` 模型缓存模式从 s08 / s10 不适用于 s12**——本章不调 LLM，部署层不涉及模型加载（`app.py` 走 `embed()` 函数，模型由 s04 / s07 内部自己缓存，FastAPI 包装层不需要重复缓存）。`embedding-routing` 这种模型生命周期管理是 RAGFlow 工业级方案（见 §四），不在 MVP 范围。

### 3.4 troubleshooting

- **`docker: command not found`** — 没装 Docker Desktop。本章需要它，装好之后 `docker --version` 应该能跑。Windows 用户注意 WSL2 后端。
- **`failed to solve: failed to read dockerfile`** — `docker-compose.yml` 里 `build.context` 和 `dockerfile` 路径对不上。本章的 compose 已经显式写成 `dockerfile: s12_deployment/Dockerfile`，别手贱改回 `build: ..`（那样 Docker 会去项目根找 `Dockerfile`，找不到）。
- **build 时 `pip install` 报 chroma-hnswlib 编译错** — Dockerfile 里装了 `build-essential` 就是给这个兜底的；如果还报错，通常是网络问题，换 pip 镜像源。
- **`/qa` 返回 500 + "LLM_API_KEY not set"** — 容器没拿到 `.env`。compose 里 `env_file: ../.env` 是相对 compose 文件位置的相对路径，确认 `.env` 在项目根。
- **`/qa` 返回 500 + "Collection docs does not exist"** — 挂载路径不对。`s05_vector_index/_chroma` 必须先在主机上存在（跑过 s05），compose 里是 `:ro` 只读挂载，容器里就能看到 Chroma 的持久化文件。
- **容器跑起来但 `/qa` 报 503 "Chroma 索引不存在"** — `_get_col()` 在 `WORKDIR / s05_vector_index / _chroma` 不存在时返 503（status 503 = Service Unavailable），提示用户先跑 s05；这是 §3.3 里 lazy-load 设计的预期行为。
- **`code.py` 打印 `❌ .env 不存在` 直接退出** — `main()` 第 1 步 gating：`.env` 不在就不调 `docker compose up`，免得白烧镜像层。补 `.env` 后重跑。
- **`code.py` 打印 `❌ 索引不存在` 直接退出** — `main()` 第 2 步 gating：`s05_vector_index/_chroma` 不在就退出。先 `python s05_vector_index/code.py` 重建索引。

### 3.5 如何切换到 RAGFlow 风格部署

加一种部署能力（health check / 模型独立 / 鉴权 / 监控）只要三步：

1. 在 `app.py` 加 `@app.get("/healthz") def healthz(): return {"ok": True}` + 在 `docker-compose.yml` 加 `healthcheck: test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"]` + `depends_on: condition: service_healthy`——让 compose 知道服务"真起来了"而不是"进程在跑"（参考 `docs/reference/ragflow-notes/deployment.md` §"为什么这样写"）；
2. 把 `app.py` 里 `from sentence_transformers import ...` / bge-reranker 这种重依赖拆出来独立成模型服务（`tei-cpu` / `tei-gpu` / `vllm`），FastAPI 进程发 HTTP 调模型——**API 进程无状态、模型单独调度、改模型不用重 build API 镜像**（参考 `docs/reference/ragflow-notes/deployment.md` 服务清单的 `tei-cpu` / `tei-gpu` 行）；
3. 在 compose 前套 nginx / APISIX 网关做鉴权（API key / JWT）和限流（rate limit），env_file 里的 `LLM_API_KEY` 改成从 Vault / KMS 拉（runtime secret），不在镜像层固化——参考 RAGFlow `docker-compose-base.yml` 的 `nginx` 服务（顶层 compose 集成）。

不要在 `app.py` 里写 `if auth_mode == "api_key": ... elif auth_mode == "jwt": ...` 之类分发——它会污染单一职责。`app.py` 只懂"接 `POST /qa`、调 s08、返结果"，鉴权放网关层。本章 MVP 只跑单容器无鉴权，但**接口形状留好了**——加 `/healthz` 端点不需要改业务代码、套网关不需要改 FastAPI 内部。

---

## 四、选型与思考题

### 4.1 主流部署范式速览

下面这张表把 RAG 系统的部署路径按"服务数量 / 存储 / 模型部署 / 监控 / 适用场景"列出来：

| 范式 | 服务数量 | 存储 | 模型部署 | 监控 | 适用场景 |
|---|---|---|---|---|---|
| **单容器 FastAPI（本章 MVP）** | 1 | 本地文件 + 内存 | 进程内 `import` | 无 | 教学 / 快速原型 / demo / 单人用 |
| **compose + 健康检查 + 网关** | 2-3（API + 网关 + 监控） | 本地 volume | 进程内 / 模型 sidecar | Prometheus + Grafana | 小团队 / 内网 demo / 早期产品 |
| **多容器编排（docker-compose 6-10 服务）** | 6-10 | S3 / MinIO + ES | `tei-*` / `vLLM` 独立 | Prometheus + Grafana + Loki | 中小规模生产 / 单租户 |
| **K8s + 微服务（RAGFlow 工业）** | 10+ | S3 + ES 集群 + MySQL 集群 | `tei-gpu` + vLLM + 队列 | 完整 observability 栈 | 多租户生产 / 大规模 / 高可用 |
| **Serverless + 托管向量库** | N/A（函数） | Pinecone / Weaviate Cloud | 调用托管 Embedding API | 云厂商自带 | 流量波动大 / 不想运维 / 创业 MVP |

我们的 toy `app.py` + `Dockerfile` + `docker-compose.yml` 在范式复杂度上只占第一行——**单容器 FastAPI**；RAGFlow 走完整工业路径，**多一道抽象就多一道观测点 + 一个失败模式**。教学 demo 选 MVP 因为它跑通快、依赖少、依赖全在 `docker compose up --build` 这一行命令里可见；**生产请按"用户量 / 数据量 / 是否多租户 / 是否要可观测"做 tier 选型**（MVP → compose + 网关 → RAGFlow compose → K8s）。

### 4.2 选型速记

- **教学 / 快速原型 / demo / 单人用** → 本章 MVP（1 容器 FastAPI + compose），5 分钟跑起来，curl 调得通，改代码重 build 慢但可接受；
- **小团队 / 内网 demo / 早期产品** → 加 `/healthz` + nginx 网关（鉴权 + 限流）+ Prometheus 指标，3 个服务，代码 +50 行换 +200% 可用性；
- **中小规模生产 / 单租户** → 切 RAGFlow compose（10 服务，模型独立部署 + ES 集群 + MinIO），加 9 个容器换"能扛中等流量、能备份、能水平扩展"，运维成本 3x；
- **多租户生产 / 大规模 / 高可用** → K8s + Helm + ArgoCD（完整 CI/CD + 滚动升级 + 灰度发布 + 多区域容灾），加 1 层抽象换"能扛大规模、能自动恢复、能灰度"，运维成本 10x 但可观测性 +10x；
- **Serverless / 不想运维** → 托管向量库（Pinecone / Weaviate Cloud）+ 云函数（AWS Lambda / Vercel）+ 托管 Embedding API，**0 运维**，但**单次调用成本 3-5x 自建**、数据合规要额外评估；
- **要先看清每个边界再选** → 用本章 unit 01 把"1 容器"和"compose + 网关"各跑一次，对比"5 分钟跑起来"和"3 个服务 + 50 行配置"——这是最简单的"部署方案 A/B"实验。

### 4.3 思考题

1. **生产环境你还会加哪些服务？**  
   答：见 [`thinking_answers.md`](./thinking_answers.md)。简版：① 监控（Prometheus + Grafana + Loki）；② 错误追踪（Sentry / 自建 Trace）；③ 独立模型服务（vLLM / TEI / TGI）；④ 对象存储（S3 / MinIO）。这四类是"能上线"和"能跑"的最小 gap；再往上加 API 网关 / 任务队列 / 配置中心 / 密钥管理才是"能卖钱"。

2. **为什么 `code.py` 在 `.env` 不存在时直接退出、不进入 docker build？**  
   答：省镜像层缓存——如果 `env_file: ../.env` 缺失，构建会在 `pip install` 之后才发现 `.env` 读不到，**整个镜像烧完才发现跑不起来**，得 `docker compose down --rmi all` 清理才能重来。提前在 host 上 gating 失败可以**只浪费 1 行 print 的时间、不浪费 2GB 镜像层**。同理索引 gating——`s05_vector_index/_chroma` 不存在时容器启动后第一次 `/qa` 才会暴露问题，**提前 gating 把"运行时报错"变成"启动时可见"**，跟 s11 unit 02 缺包 / 缺二进制 / 缺图三类异常 catch 是同一种"早暴露"思路。详见 [`thinking_answers.md`](./thinking_answers.md)。

3. **怎么让 FastAPI 服务"高可用"？**  
   答：3 个方向叠加：① 进程级——加 `restart: unless-stopped`（compose 字段，容器崩了自动拉起）+ 加 `/healthz` 端点给 K8s liveness probe（崩了不优雅退出会被 probe 杀掉）；② 流量级——compose scale 起来多副本（`docker compose up --scale rag=3`），前面套 nginx / APISIX 做负载均衡 + 健康检查剔除坏副本；③ 数据级——Chroma 索引是本地文件，**多副本同时读没问题，同时写会锁**。生产里换 ES / Milvus 这种原生支持分布式 + 主从复制的引擎，`depends_on: condition: service_healthy` 串启动顺序 + `replicas: 3` 设副本数。MVP 不做高可用——单进程单容器，崩了手动 `docker compose up` 重启即可。详见 [`thinking_answers.md`](./thinking_answers.md)。
