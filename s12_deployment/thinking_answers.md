# s12 思考题答案

## 生产环境你还会加哪些服务?

把"单容器 FastAPI"推到生产,通常至少再加这四类:

### 1. 监控 + 可观测性: Prometheus + Grafana (+ Loki 日志)

现状:容器 stdout 打印到 `docker compose up` 的终端,关掉就没了;没有任何
指标暴露。

加什么:

- **Prometheus** (`prom/prometheus` 镜像) — 时序数据库,定期 scrape
  各服务的 `/metrics` 端点,存 `请求量 / 延迟分位 / 错误率 / 内存占用
  / GPU 利用率`。给 uvicorn 加 `prometheus-fastapi-instrumentator` 中间件
  一行代码就暴露 `/metrics`。
- **Grafana** (`grafana/grafana` 镜像) — 可视化。Prometheus 是数据源,
  画"过去 1 小时 P99 延迟""LLM 调用成功率"这种看板。
- **Loki** + Promtail — 集中日志。Promtail 跑在每个节点收集容器 stdout,
  Loki 索引,Grafana 搜。比"登服务器 `docker logs`"好用一万倍。

成本: 3 个镜像 + 1 个配置文件。CPU 几乎不占,内存加起来 1-2GB。

### 2. 错误追踪: Sentry (或自建 GLITCH-Trace)

现状:容器 500 错误只在日志里,用户那边"我问了没回应"工程师完全不知。

加什么:

- **Sentry** (`sentry-self-hosted` 或 SaaS) — 前端 + 后端都装 SDK,
  自动捕获异常 + 上下文 (用户/请求/堆栈/环境变量白名单)。
- 关键配置:**数据脱敏** (`.env` 里的 `LLM_API_KEY` 千万别上报),
  **采样率** (生产建议 10%,全量打爆配额),**告警 webhook**
  (新错误 5 分钟内没人 ack 就 @oncall)。

成本: SaaS 免费额度够小项目;自建要 3-5 个容器。

### 3. 独立模型服务: vLLM / TGI (替代直接 `import sentence-transformers`)

现状:Embedding 和 Rerank 都在 API 进程里同步跑,BGE-reranker-base
一次 query ~500ms,期间 API 完全被占住。

加什么:

- **vLLM** (LLM) 或 **TEI** (Embedding) / **Text Generation
  Inference** (Rerank) — 把模型独立成 HTTP 服务,gpu 显存占满后
  通过 batch + PagedAttention 撑高并发。API 进程只发 HTTP,几毫秒
  返回,可以同时跑几十个请求。
- 好处: ① API 进程无状态,水平扩展容易;② GPU 单独调度,不用每个
  API 副本都吃显存;③ 模型版本升级只重启模型服务。

成本: 至少 1 张 GPU (本地) 或 1 个 GPU 实例 (云上 ~¥2/小时)。

### 4. 对象存储: S3 / MinIO (替代 samples 目录挂载)

现状:`samples/` 是本地目录,容器靠 volume 挂载读。生产场景
"用户上传 1000 个 PDF"——文件存哪?

加什么:

- **S3** (AWS / 阿里云 OSS / 腾讯云 COS) — 云上托管,按量付费,
  11 个 9 持久性。
- **MinIO** — 自建,API 兼容 S3,放内网,RAGFlow docker-compose 里
  就跑的这个。最小两节点做副本,几个 TB 容量随便堆。
- 配套:上传走预签名 URL (前端直传,不经过 API),API 只存对象 key;
  处理流水线监听 S3 事件 (S3 Event Notification) 触发解析。

成本: S3 几分钱 GB/月;MinIO 自建主要是磁盘钱。

### 其它常见补充 (按优先级排)

- **API 网关** (nginx / Kong / APISIX) — 统一鉴权 / 限流 / 路由。
- **任务队列** (Celery + Redis / BullMQ) — 把"解析 PDF""批量 Embedding"
  从同步请求里挪出来,API 只接任务 ID。
- **配置中心** (Consul / Nacos / etcd) — 不再让 `.env` 文件满天飞。
- **密钥管理** (Vault / AWS KMS) — `LLM_API_KEY` 不放环境变量,
  运行时动态拉。
- **服务网格** (Istio / Linkerd) — 多服务时 mTLS + 流量管理;MVP
  单服务阶段不需要。

### MVP 现状 vs 生产的最小可用集

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

MVP 是"能跑",上面这些是"能上线"。中间还隔着一个"能卖钱"——
那个阶段要做的事情更多 (A/B 测试 / 灰度发布 / 多区域容灾),但
和 RAG 本身没关系了,属于通用 SaaS 工程范畴。
