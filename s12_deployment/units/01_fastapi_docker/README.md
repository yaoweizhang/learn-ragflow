# s12 / Unit 01 — fastapi_docker

> 由浅入深第 1 步:把 s08 的 `answer()` 包成 FastAPI,再用 `docker compose` 一键起服务。  
> 这也是 s12 唯一的一步——没有单元拆分,因为 12 章只跑 1 个容器。

## 这是什么

本单元做两件事:① `s12_deployment/app.py` 用 FastAPI 把 s04 → s06 → s07 → s08 串成
`POST /qa {question: ...} → {text, citations}`;② `Dockerfile` + `docker-compose.yml` 把
这个 FastAPI 应用包成镜像并跑起来,只挂 `.env` / `samples` / `_chroma` 三样东西,其他
都靠镜像里 `pip install -r requirements.txt` 兜底。`code.py` 干的事就是先做两道前置
校验 (`.env` 存在、`s05_vector_index/_chroma` 存在),然后 `subprocess.run(["docker",
"compose", "up", "--build"], cwd=s12_deployment)`。

s12 之所以只拆 1 个 unit:整章就 4 个文件 (`app.py` / `Dockerfile` / `docker-compose.yml`
+ 本 `code.py`),组合只有一种,没有"由浅入深"的多档梯度——`docker compose up --build`
是终点,没有中间状态可设。所以 1 个 unit 就是全部。

## 跑起来

```bash
# 0. 前提:跑通 s04 / s05 / s07 (生成索引 + 装好 bge)
# 1. 配 .env (项目根)
cp .env.example .env   # 填 LLM_API_KEY

# 2. 启动 (任选一种)
python s12_deployment/code.py
#   或直接走单元:
python s12_deployment/units/01_fastapi_docker/code.py
# 内部执行 docker compose up --build,会自动 build 镜像 + 启服务

# 3. 测一发
curl -X POST http://localhost:8000/qa \
  -H 'Content-Type: application/json' \
  -d '{"question":"项目里有哪些表?"}'

# 4. 停掉
# 在跑 code.py 的终端 Ctrl-C
```

输出示例(成功时终端会看到 docker compose 的 build + 容器日志流):

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

如果前置条件不满足:

```
❌ .env 不存在,请先 cp .env.example .env 并填 LLM_API_KEY
```

## 它做对了什么

- **One-shot 部署** — `docker compose up --build` 一条命令同时处理 build + pull + network + volume + start,新人 clone 下来跑这一行就能用,不需要知道 Dockerfile 怎么写、port 怎么映射。
- **路径校验前置** — `.env` 和 `_chroma` 不存在会**先报错、不进入 docker build**,省得 Docker 层缓存一堆无效镜像、又得 `docker compose down --rmi all` 清理。
- **复用现有索引** — `s05_vector_index/_chroma` 用 `:ro` 只读挂载,容器重启 / 重新 build 都不会丢索引,只是被同一个 Chroma 进程读。
- **生产 artifact 分离** — `app.py` / `Dockerfile` / `docker-compose.yml` 是"被本单元包装的产物",单元 README 教学 why,但三个文件本身不被改——这样后续生产化(改 base 镜像、加 healthz)只动 artifact 不动教学代码。

## 它做错了什么

- **没有 health-check** — `docker compose up` 不等服务 `Application startup complete` 就认为"成功",万一容器内 FastAPI import 阶段就崩(比如缺某个 pip 包),终端只会看到 `exited with code 1`,新用户分不清是镜像坏了还是代码坏了。生产 compose 应该加 `healthcheck: test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"]` + `depends_on: condition: service_healthy`。
- **没有 rolling restart / log streaming** — `Ctrl-C` 直接杀容器,日志只在当前终端,没有 `docker compose logs -f` 的统一入口,也没有 `restart: unless-stopped` 的开机自启。
- **索引没预热** — Chroma 的 hnswlib 首次 query 要 mmap 索引文件,冷启动第一次 `/qa` 会有 3-5 秒延迟;这单元没做 warm-up 请求,生产环境首单用户会感知到。
- **没有鉴权 / 限流** — `POST /qa` 谁都能调,LLM token 钱随便烧;也没有 rate limit,一个脚本可以瞬间打满 token 配额。

## 对照 ragflow 怎么做的

参考:[`ragflow_notes/deployment.md`](../../../../ragflow_notes/deployment.md)

RAGFlow 把整套系统拆成 **10+ 容器**(ES / Infinity / MySQL / MinIO /
Redis / 任务队列 / 沙箱执行 / 主 API / 视觉 OCR),靠 `depends_on:
condition: service_healthy` 串起启动顺序。我们 MVP 反过来:**只跑 1 个
容器**,把 ES 换成 Chroma、MySQL 换成进程内存、MinIO 换成 samples
目录挂载。代价见上方"它做错了什么"(无健康检查 / 无横向扩展 / 无
多租户隔离),收益是 5 分钟跑起来。

镜像尺寸 trade-off:RAGFlow 的 `ragflow-cpu` 镜像把 Python 基础层、
模型权重层、业务代码层分开,改代码不用重下模型(模型用 volume 或
S3 挂载);我们 MVP 的 `python:3.11-slim` + `pip install -r
requirements.txt` 会把 BGE (~100MB)、reranker (~1GB)、chromadb ONNX
依赖全烧进镜像层,`docker images` 看一下可能 2GB+,**改一行代码
`docker compose build` 就会触发完整重 build**——这是 demo 的代价。

## 思考题

- **如果要把这单元升级到"开箱即用的 demo",你还会在 compose 里加什么?**  
  提示:`healthcheck` + `restart: unless-stopped` + 一个 `seed-data`
  sidecar(跑完 s05 后把索引预热一次再 SIGTERM)。完整答案见
  `thinking_answers.md`。
