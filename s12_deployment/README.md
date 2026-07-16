# s12 部署 — Docker Compose 把 MVP 包成服务

[上一章 s11 · 下一章(无 — 末章)]

> *"本机命令行能跑"和"别人用 curl 就能调"之间隔着一道悬崖 — 这道悬崖由镜像打包 / HTTP 边界 / 持久化 / 可观测四类问题堆起来"*
>
> **链路位置**: 端到端服务化 (s08 的 `answer()` → **s12 FastAPI + Docker**, 独立可跑)
> **代码文件**: fastapi_docker.py

> 环境准备: 见 root README §快速开始 — Docker Desktop 已装 + `pip install -r requirements.txt` + 项目根 `.env` 配 LLM_API_KEY + `s05_vector_index/_chroma` 索引已生成 (跑过 s05)

---

## 问题

s08 的 `answer()` 只能在命令行跑 `python s08_prompt_generate/prompt_template.py`,然后交互式输入问题 — **s11 结束时的链路是"开发者本人在自己机器上跑得通"**。这条"本机能跑"和"别人能用"之间隔着四道典型鸿沟,每一道都对应一类工业级失败:

**第一, Python / 依赖 / 索引要对方全装一遍**。同事 clone 仓库后要 `pip install -r requirements.txt` + `cp .env.example .env` 填 key + 跑 s04/s05 重建索引 + 下 BGE 模型权重 (~100MB) + 下 bge-reranker-base (~1GB) — **5 个步骤任一卡住就报 traceback**。**生产解法**: 用 Docker 把 Python + 系统依赖 + pip 包 + 业务代码 + 模型权重全部装进镜像,对方只要 `docker compose up --build` 一行命令。

**第二, 没法给前端 / 其他服务调**。命令行交互式输入只适合"开发者自己玩"。前端要做个搜索框、其他服务要做 API 集成,**没有 HTTP 端点就只能写文件 / 跑子进程**,耦合重。**生产解法**: 用 FastAPI (pydantic 强类型 + OpenAPI 文档自动生成) 把 `answer()` 包成 `POST /qa`,`curl` / fetch / requests 都能调。

**第三, 重启 = 数据丢失 / 索引丢失**。本地跑 `python s08_prompt_generate/prompt_template.py` 时索引在内存;kill 重启就重建。多人协作 / 服务器跑就尴尬。**生产解法**: 把 Chroma 索引目录用 volume 挂载进容器 (`:ro` 只读),容器重启不丢、重新 build 不丢、多副本读同一份索引。

**第四, 冷启动慢 + 没有任何观测**。`POST /qa` 第一次请求 Chroma hnswlib 要 mmap 索引文件,3-5 秒延迟;之后没人知道"现在 /qa 平均延迟多少 / 错误率多少 / LLM 调用成本多少"。**生产解法**: ① 容器预热 (docker compose run 预热后 SIGTERM 再起服务); ② 加 `/healthz` 端点给 K8s liveness probe; ③ 暴露 Prometheus `/metrics` 抓 `请求量 / 延迟分位 / 错误率 / LLM token 消耗`。MVP 都不带。

把这四类失败合起来看,**s12 的目标不是解决它们,而是把它们显式暴露出来,让你看到"本机命令行 RAG"和"生产 HTTP RAG"之间的边界**。这跟 s10 把"向量召回答不全实体之间关系"、s11 把"表格被拍扁 / 扫描件返回空"显式对比是同一种思路 — **叙述载体从"图函数 + 1 跳 query"换成"FastAPI 包装 + Docker 镜像 + compose 编排 + 启动器",但"先跑通 toy、再讲清楚 toy 在哪里会塌"的教学哲学是一致的**。s12 是末章,它的脆弱点也是真实生产里要继续填的洞 — 部署不是把代码"传上去"就完了,它是一个**契约**:服务端要保证镜像能跑、端口能访问、索引能读、LLM key 能用;客户端只要发 HTTP 就能拿到答案。

```
   本地命令行 RAG                                容器化 HTTP 服务
   python s08_prompt_generate/prompt_template.py                            docker compose up --build
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
   启动器 `fastapi_docker.py`.py: .env gating + 索引 gating + subprocess docker compose up
```

> 部署 MVP = FastAPI 包装 (HTTP 入口) + Dockerfile (镜像) + docker-compose.yml (编排) + 启动器 (前置校验)。

---

## 解决方案

s12 用 **4 个文件 + 1 个启动器** 把"s08 本机命令行 RAG"升级为"容器化 HTTP 服务"。`fastapi_docker.py` 启动器先做两道前置校验 (`.env` 存在、`s05_vector_index/_chroma` 存在),然后 `subprocess.run(["docker", "compose", "up", "--build"], cwd=s12_deployment)` — 一次启动管全部,没有多档梯度。s12 只有这一步,因为整章就 4 个文件 (`app.py` / `Dockerfile` / `docker-compose.yml` + 本 `fastapi_docker.py`),组合只有一种,`docker compose up --build` 是终点,没有中间状态可设。

**FastAPI 边界**: `app.py` 暴露的 `POST /qa` schema 长这样 — `{question: str} → {text: str, citations: list[dict]}`。请求 schema 只有一个字段 `question`;响应 schema 直接复用 s08 `answer()` 的 `{"text", "citations"}` 字段 — **这是 s04 → s06 → s07 → s08 链路"端到端 JSON 可序列化"的红利**:只要每一段函数的输入输出都是 JSON 安全的,HTTP 边界就成了纯"序列号 + 拆包"问题。`app.py` 内部按 `embed → hybrid_search → rerank → answer` 的顺序串起来,每一段函数的输入输出都跟上游 s04-s08 一致。

**Dockerfile 层结构** (`python:3.11-slim` 为底 + 4 层缓存):① 基础镜像层 (`python:3.11-slim` + `apt-get install tesseract-ocr build-essential`) 几乎不变;② 依赖层 (`COPY requirements.txt` + `pip install`) 只在 `requirements.txt` 改了才重 build;③ 代码层 (`COPY .. /app`) 是"高频改动层" — 业务代码改了重 build 这一层,前 4 层全部命中缓存。MVP 不分模型层 — `bge` 权重 ~100MB、`bge-reranker-base` ~1GB 随 `pip install` 一起烧进第 4 层,**改一行代码 `docker compose build` 就触发完整重 build**,是 demo 的代价。生产里模型走 volume / S3 挂载。

**docker-compose 声明式编排**:把"镜像怎么 build、端口怎么映射、卷怎么挂、环境怎么传"全部声明成 YAML,让编排成为可复现的配置文件而不是命令脚本。关键设计: ① `build.context` 是项目根 — Dockerfile 能 `COPY .. /app` 把整个仓库代码带进镜像,包括 s04-s11 的所有依赖;② `env_file: ../.env` — 容器内 `/app/.env` 自动被 Docker 加载,s08 进程读 `LLM_API_KEY` 跟本地一致;③ 只读挂载 (`:ro`) — 容器不会误写索引 / 样本,重 build 不丢数据;④ 路径用 `../` 前缀 — 相对 `docker-compose.yml` 文件位置解析,本地跑和 CI 跑行为一致。

| 文件 | 解决什么 | 留下什么局限 | 何时用 |
|---|---|---|---|
| `fastapi_docker.py` | 一行启动 + 两道前置校验 (.env / 索引) | 没 healthcheck / 没 rolling restart / 没预热 / 没鉴权限流 | 教学 / 本地起 demo / 单人开发 / 末章收尾 |

**部署 MVP 不是把代码"传上去"就完了,而是把"配环境 + HTTP 边界 + 持久化 + 可观测"四件事用 4 个文件交代清楚**。后续生产化 (Prometheus / Sentry / vLLM 独立服务 / S3 / 多副本 / K8s) 在 `### 为什么不只写这一种` 和 `## 思考题` 里展开。

---

## 代码 1: fastapi_docker 一行启动 ([fastapi_docker.py](fastapi_docker.py))

入口:[`fastapi_docker.py`](fastapi_docker.py)

把 s08 的 `answer()` 包成 FastAPI,再用 `docker compose` 一键起服务 — 这是 s12 唯一的一步。
整章只跑 1 个容器,没有步骤拆分。

### 工作原理

**做一件事**: 跑 `python s12_deployment/fastapi_docker.py`,内部先做 `.env` 存在 + 索引存在两道 gating,通过后 `subprocess.run(["docker", "compose", "up", "--build"], cwd=s12_deployment)` — 让 `app.py` (FastAPI 包装) + `Dockerfile` (镜像) + `docker-compose.yml` (编排) 三件套自动 build + 起来。

**N 步**:
1. 计算 `WORKDIR = Path(__file__).resolve().parents[1]` (项目根, 从 `s12_deployment/fastapi_docker.py` 上溯 2 级) 和 `S12_DIR = WORKDIR / "s12_deployment"`
2. **Gating 1**: `if not (WORKDIR / ".env").exists()` — 打印 `❌ .env 不存在,请先 cp .env.example .env 并填 LLM_API_KEY` 并 `return`,**免得白烧 2GB 镜像层**
3. **Gating 2**: `if not (WORKDIR / "s05_vector_index" / "_chroma").exists()` — 打印 `❌ 索引不存在,请先跑 s05` 并 `return`,把"运行时报错"变成"启动时可见"
4. **优雅降级**: `import shutil; if shutil.which("docker") is None` — 打印该敲的命令 (`cd s12_deployment && docker compose up --build`),CI / 学生笔记本无 Docker 时不崩
5. **一键启动**: `subprocess.run(["docker", "compose", "up", "--build"], cwd=S12_DIR, check=True)` — `cwd=s12_deployment` 让 compose 自动找到同名 yml,`--build` 强制重 build (代码改了也生效)

```python
# 中间片段: 两道 gating + 一行 subprocess
WORKDIR = Path(__file__).resolve().parents[1]
S12_DIR = WORKDIR / "s12_deployment"

def main() -> None:
    # 1. .env gating — 没有 LLM_API_KEY 就别 build,免得白烧镜像层
    if not (WORKDIR / ".env").exists():
        print("❌ .env 不存在,请先 cp .env.example .env 并填 LLM_API_KEY")
        return
    # 2. 索引 gating — Chroma 持久化目录必须在主机上,否则容器 :ro 挂载会空
    if not (WORKDIR / "s05_vector_index" / "_chroma").exists():
        print("❌ 索引不存在,请先跑 s05 (python s05_vector_index/chroma_build.py)")
        return
    # 3. 一键 build + up
    if shutil.which("docker") is None:
        print(f"⚠️  docker 未安装,本机手动: cd {S12_DIR} && docker compose up --build")
        return
    subprocess.run(["docker", "compose", "up", "--build"], cwd=S12_DIR, check=True)
```

**完整函数**:

```python
WORKDIR = Path(__file__).resolve().parents[1]
S12_DIR = WORKDIR / "s12_deployment"


def main() -> None:
    """s12 启动器: .env gating → 索引 gating → docker compose up --build (优雅降级支持无 Docker 环境)。"""
    # 1. .env gating — 没有 LLM_API_KEY 就别 build,免得白烧镜像层
    if not (WORKDIR / ".env").exists():
        print("❌ .env 不存在,请先 cp .env.example .env 并填 LLM_API_KEY")
        return
    # 2. 索引 gating — Chroma 持久化目录必须在主机上,否则容器 :ro 挂载会空
    if not (WORKDIR / "s05_vector_index" / "_chroma").exists():
        print("❌ 索引不存在,请先跑 s05 (python s05_vector_index/chroma_build.py)")
        return
    # 3. 一键 build + up — cwd 用 s12_deployment 让 compose 找到同名 yml
    #    若本机没装 docker (例如 CI / 学生笔记本) 则优雅降级,把"该敲的命令"打印出来
    import shutil
    if shutil.which("docker") is None:
        print("⚠️  docker 未安装,跳过实际 build/up 步骤。")
        print("   本机手动执行即可:")
        print(f"     cd {S12_DIR} && docker compose up --build")
        return
    subprocess.run(["docker", "compose", "up", "--build"], cwd=S12_DIR, check=True)


if __name__ == "__main__":
    main()
```

`app.py` / `Dockerfile` / `docker-compose.yml` 三个 supporting 文件的职责:**app.py** 是 FastAPI 应用,内部 `_get_col()` lazy-load Chroma collection,`qa(req)` handler 把 `_get_col → embed → hybrid_search → rerank → answer` 五步串行,索引缺失返 `HTTPException(503)`;**Dockerfile** 是 `python:3.11-slim` + `tesseract-ocr` + `build-essential` + `pip install -r requirements.txt` + 业务代码,`CMD ["uvicorn", "s12_deployment.app:app", ...]` 启服务;**docker-compose.yml** 是 `services.rag: { build.context=.., dockerfile=s12_deployment/Dockerfile, ports=[8000:8000], env_file=[../.env], volumes=[../samples:/app/samples:ro, ../s05_vector_index/_chroma:/app/s05_vector_index/_chroma] }`。

### 试一下

```bash
# 0. 前提: 跑通 s04 / s05 / s07 (生成索引 + 装好 bge)
# 1. 配 .env (项目根)
cp .env.example .env   # 填 LLM_API_KEY

# 2. 启动 (任选一种)
python s12_deployment/fastapi_docker.py
# 内部执行 docker compose up --build,会自动 build 镜像 + 启服务

# 3. 测一发
curl -X POST http://localhost:8000/qa \
  -H 'Content-Type: application/json' \
  -d '{"question":"项目里有哪些表?"}'

# 4. 停掉
# 在跑 fastapi_docker.py 的终端 Ctrl-C
```

输出示例 (成功时终端会看到 docker compose 的 build + 容器日志流):

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

`/qa` 实测响应 (输入 `"项目里有哪些表?"`):

```json
{
  "text": "项目里包含:规格表(整机规格表 + 各组件详细规格)、内存表、电源表...",
  "citations": [
    {"source": "samples/server_whitepaper.pdf", "page": 2, "text": "三、整机规格 组件 规格 说明 处理器 2 × 第三代 Intel Xeon 可扩展处理器 最高 40 核 / 80 线程..."}
  ]
}
```

**观察**: `citations` 数组的每条对应 s07 精排后喂给 s08 的 top-k hit — **HTTP 边界只做"序列号 + 拆包"**,真正的 LLM 答不答、拒答不拒答由 s08 决定。整条链路 (`embed → hybrid_search → rerank → answer`) 在容器里跑,本地端到端可调通。

### 为什么不只写这一种

``fastapi_docker.py`` + `app.py` + `Dockerfile` + `docker-compose.yml` 用 **单容器 FastAPI + compose 一键起服务** 解决了"让别人用 curl 就能调"的问题,但**留了 4 类典型生产缺口**:① **没 health-check** — `docker compose up` 不等服务 `Application startup complete` 就认为"成功",新用户分不清是镜像坏了还是代码坏了,生产应该加 `healthcheck: test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"]` + `depends_on: condition: service_healthy`;② **没 rolling restart / log streaming** — `Ctrl-C` 直接杀容器,没有 `restart: unless-stopped` 也没有 `docker compose logs -f` 统一入口;③ **索引没预热** — Chroma 的 hnswlib 首次 query 要 mmap 索引文件,冷启动第一次 `/qa` 会有 3-5 秒延迟;④ **没鉴权 / 限流** — `POST /qa` 谁都能调,LLM token 钱随便烧,也没有 rate limit。

s12 是末章 — **它把 s04 → s06 → s07 → s08 的整条 RAG 管线收敛到 `POST /qa` 一个 FastAPI 端点**,但"5 分钟跑起来"和"能扛生产流量"之间隔着监控 (Prometheus + Grafana + Loki)、错误追踪 (Sentry)、独立模型服务 (vLLM / TGI)、对象存储 (S3 / MinIO)、鉴权 (API key / JWT)、队列 (Celery / Redis Streams) — 这些是 `## 思考题` 里展开的填空目标,也是真实生产里要继续填的洞。部署不是把代码"传上去"就完了 — 它是一个**契约**;s12 把契约的最小集写出来,后续每一层 (监控 / 鉴权 / 副本 / K8s) 都是在这个契约上加签名。

---

## 接下来

s12 把"s08 本机命令行 RAG"升级为"容器化 HTTP 服务" — `fastapi_docker.py` 启动器 + `app.py` FastAPI 包装 + `Dockerfile` 镜像 + `docker-compose.yml` 编排 4 件套收敛成 `POST /qa` 一个端点。但每一步都留下生产缺口,这些是真实部署要继续填的洞:

- **没 healthcheck** — 容器崩了 K8s / compose 不知道;生产应该加 `/healthz` + `depends_on: condition: service_healthy`。
- **没 rolling restart / log streaming** — `Ctrl-C` 直接杀容器,日志只在当前终端;生产应该加 `restart: unless-stopped` + `docker compose logs -f` 统一入口。
- **冷启动慢** — Chroma hnswlib 首次 query 要 mmap 索引文件,3-5 秒延迟;生产应该 docker compose run 预热请求后 SIGTERM 再起服务。
- **没鉴权 / 限流** — `POST /qa` 谁都能调,LLM token 钱随便烧;生产应该加 API key + rate limit + nginx/APISIX 网关。
- **单容器单进程** — 崩了手动 `docker compose up` 重启;生产应该 `depends_on: condition: service_healthy` + `docker compose up --scale rag=3` + nginx 负载均衡,数据层换 ES / Milvus 这种支持分布式 + 主从复制的引擎。

**这是末章**。s12 把 s04 → s06 → s07 → s08 的"召回 → 排序 → 生成 → 服务化"全链路收敛成 FastAPI 一个端点;**每一章的局限,都是生产化要继续填的洞,也是 RAG 工程师从 demo 到 production 的下一步**。

---

## 思考题

1. **生产环境你还会加哪些服务？**
2. **为什么 `fastapi_docker.py` 在 `.env` 不存在时直接退出、不进入 docker build？**
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

### Q2. 为什么 `fastapi_docker.py` 在 `.env` 不存在时直接退出、不进入 docker build?

省镜像层缓存——如果 `env_file: ../.env` 缺失，构建会在 `pip install` 之后才发现 `.env` 读不到，**整个镜像烧完才发现跑不起来**，得 `docker compose down --rmi all` 清理才能重来。提前在 host 上 gating 失败可以**只浪费 1 行 print 的时间、不浪费 2GB 镜像层**。同理索引 gating——`s05_vector_index/_chroma` 不存在时容器启动后第一次 `/qa` 才会暴露问题，**提前 gating 把"运行时报错"变成"启动时可见"**，跟 s11 中 OCR 部分缺包 / 缺二进制 / 缺图三类异常 catch 是同一种"早暴露"思路。

### Q3. 怎么让 FastAPI 服务"高可用"?

3 个方向叠加：① 进程级——加 `restart: unless-stopped`(compose 字段，容器崩了自动拉起）+ 加 `/healthz` 端点给 K8s liveness probe（崩了不优雅退出会被 probe 杀掉）；② 流量级——compose scale 起来多副本（`docker compose up --scale rag=3`），前面套 nginx / APISIX 做负载均衡 + 健康检查剔除坏副本；③ 数据级——Chroma 索引是本地文件，**多副本同时读没问题，同时写会锁**。生产里换 ES / Milvus 这种原生支持分布式 + 主从复制的引擎，`depends_on: condition: service_healthy` 串启动顺序 + `replicas: 3` 设副本数。MVP 不做高可用——单进程单容器，崩了手动 `docker compose up` 重启即可。

> Docker 缺命令 / compose 路径错 / `pip install` 编译错 / `/qa` 500 系列 / gating 报错 等排错细节见 ``fastapi_docker.py`` 的 `### 局限与下一步`。