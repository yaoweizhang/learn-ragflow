# s12 部署 — Docker Compose 把 MVP 包成服务

## Units

| Unit | 主题 | 文件 |
|------|------|------|
| 01 | fastapi_docker (FastAPI 包装 + docker compose 一键起) | [`units/01_fastapi_docker/code.py`](units/01_fastapi_docker/code.py) |

## 问题

s08 的 `answer()` 只能在命令行跑 `python s08_prompt_generate/code.py`,
然后交互式输入问题。要想给别人用 (同事 / 前端 / 其他服务),需要把它
**包成一个 HTTP 服务**,而且最好**容器化**——这样对方不需要装
Python、不需要下模型权重、不需要配 chroma 索引,只需要一个 Docker。

## 最小解法

`s12_deployment/` 里四个文件 + 一个 FastAPI 包装:

- **`app.py`** — FastAPI 应用,`POST /qa` 接 `{"question": "..."}`,
  内部按 s04 → s06 → s07 → s08 的顺序跑一遍,把 `answer()` 的返回值
  (`{"text", "citations"}`) JSON 出去。
- **`Dockerfile`** — 基于 `python:3.11-slim`,装 tesseract (s11 OCR 用),
  `pip install -r requirements.txt`,CMD 跑 `uvicorn s12_deployment.app:app`。
- **`docker-compose.yml`** — 一个 `rag` 服务: build 上下文是项目根
  (这样能 COPY 到 `requirements.txt` 和所有 `s0X_*` 源码),端口 8000,
  挂载 `.env` / `samples` / `_chroma` 索引。
- **`code.py`** — 启动脚本。先检查 `.env` 和 `_chroma` 在不在,不在就
  友好报错;在就跑 `docker compose up --build`。

## 跑起来

```bash
# 0. 前提:跑通 s04 / s05 / s07 (生成索引 + 装好 bge)
# 1. 配 .env (项目根)
cp .env.example .env   # 填 LLM_API_KEY

# 2. 启动
python s12_deployment/code.py
# 内部执行 docker compose up --build,会自动 build 镜像 + 启服务

# 3. 测一发
curl -X POST http://localhost:8000/qa \
  -H 'Content-Type: application/json' \
  -d '{"question":"项目里有哪些表?"}'

# 4. 停掉
# 在跑 code.py 的终端 Ctrl-C
```

### troubleshooting

- **`docker: command not found`** — 没装 Docker Desktop。本章需要它,
  装好之后 `docker --version` 应该能跑。Windows 用户注意 WSL2 后端。
- **`failed to solve: failed to read dockerfile`** — `docker-compose.yml`
  里 `build.context` 和 `dockerfile` 路径对不上。本章的 compose 已经
  显式写成 `dockerfile: s12_deployment/Dockerfile`,别手贱改回
  `build: ..` (那样 Docker 会去项目根找 `Dockerfile`,找不到)。
- **build 时 `pip install` 报 chroma-hnswlib 编译错** — Dockerfile 里
  装了 `build-essential` 就是给这个兜底的;如果还报错,通常是网络问题,
  换 pip 镜像源。
- **`/qa` 返回 500 + "LLM_API_KEY not set"** — 容器没拿到 `.env`。
  compose 里 `env_file: ../.env` 是相对 compose 文件位置的相对路径,
  确认 `.env` 在项目根。
- **`/qa` 返回 500 + "Collection docs does not exist"** — 挂载路径不对。
  `s05_vector_index/_chroma` 必须先在主机上存在 (跑过 s05),compose
  里是 `:ro` 只读挂载,容器里就能看到 Chroma 的持久化文件。

## 真实世界的问题

1. **模型镜像太大** — `python:3.11-slim` 本身 150MB,但 `sentence-transformers`
   拉的 BGE 模型要 ~100MB,`bge-reranker-base` 要 ~1GB,`chromadb`
   又会下一堆 ONNX 依赖。`docker images` 看一眼可能 2GB+。生产里
   通常分两层:基础镜像 (Python + 系统依赖) 一层,模型权重走
   `docker volume` 挂载或对象存储 (S3 / MinIO),避免每次改代码都重
   下模型。
2. **索引需要预热** — Chroma 的 hnswlib 首次 query 要把索引文件
   mmap 进内存,冷启动第一次 `/qa` 请求可能 3-5 秒。RAGFlow 的 ES
   索引有 `cache.prewarming` 配置,我们这种小索引无所谓,但生产
   "TB 级 ES 索引"必须预热否则第一个用户等到天荒地老。
3. **监控 / 日志缺失** — `docker compose up` 默认把 stdout 打终端,
   一关终端日志就没了。生产至少要做: ① stdout 接 Fluentd / Loki
   集中存;② 加 `/healthz` 端点给 Kubernetes liveness probe;
   ③ Prometheus exporter 抓 `uvicorn` 的请求量 / 延迟 / 错误率。
   这些 MVP 都不带。
4. **没有鉴权** — `POST /qa` 谁都能调,LLM token 钱随便烧。生产必须
   加 API key / JWT 中间件,或者前面套一层 nginx + OAuth2 proxy。

## ragflow 怎么做的

见 [ragflow_notes/deployment.md](../ragflow_notes/deployment.md)。
要点:RAGFlow 把整套系统拆成 10+ 容器 (ES / Infinity / MySQL /
MinIO / Redis / 任务队列 / 沙箱执行 / 主 API / 视觉 OCR),靠
`depends_on: condition: service_healthy` 串起启动顺序。我们
MVP 反过来:**只跑 1 个容器**,把 ES 换成 Chroma、MySQL 换成
进程内存、MinIO 换成 samples 目录挂载。代价见上方"真实世界
的问题",收益是 5 分钟跑起来。

## 思考题

- **生产环境你还会加哪些服务?**
  答:见 `thinking_answers.md`。
