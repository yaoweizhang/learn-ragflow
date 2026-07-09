# RAGFlow 怎么做: 部署架构 — 10+ 服务拆分的取舍

## 来源
- 仓库: https://github.com/infiniflow/ragflow
- 文件: `docker/docker-compose-base.yml` + `docker/docker-compose.yml`
- 行号: base L1-L356,顶层 compose L1-L152
- commit: `828c5789f651d4c4ebe4645190b8b8d244144fe0`
- 引用日期: 2026-07-04

## 一句话
RAGFlow 是个完整产品,把"文档解析 / 向量检索 / 关系数据 / 消息队列
/ 对象存储 / 沙箱执行 / 监控"拆成 10+ 容器协同工作;我们 MVP 12 章只跑
**1 个 FastAPI 容器**,所有数据放本地文件 + 内存,把"工程复杂度"换成了
"上手速度"。

## 服务清单 (base + 顶层 compose 合并去重)

顶层 `docker-compose.yml` `include: ./docker-compose-base.yml`,实际拉起的服务 (按 base 文件顺序 + 顶层 profile):

| 服务 | 镜像 / 来源 | 角色 | 我们的取舍 |
|---|---|---|---|
| `es01` | `elasticsearch:${STACK_VERSION}` | 主选全文 + 向量检索引擎 (BM25 + HNSW + 倒排) | **不跑** — s05 用 Chroma 本地文件代替 |
| `opensearch01` | `opensearch:2.19.1` | ES 替代方案 | **不跑** — 同上 |
| `infinity` | `infiniflow/infinity:v0.7.0` | 另一种向量 + 全文引擎 (RAGFlow 自家深度优化) | **不跑** |
| `oceanbase` | `oceanbase/oceanbase-ce:4.4.1.0` | 分布式关系库 (替代 MySQL) | **不跑** |
| `seekdb` | `oceanbase/seekdb:latest` | 同上轻量替代 | **不跑** |
| `sandbox-executor-manager` | `infiniflow/sandbox-executor-manager` | Agent 沙箱 (代码执行隔离) | **不跑** — MVP 没工具调用 |
| `mysql` | `mysql:8.0.39` | 主关系库 (用户/租户/知识库元数据) | **不跑** — 用进程内存代替 |
| `minio` | `pgsty/minio:...` | 对象存储 (原始 PDF / 解析结果) | **不跑** — 用 samples 目录挂载 |
| `redis` | `valkey/valkey:8` | 队列 + 任务调度 + 缓存 | **不跑** — 单进程同步处理 |
| `nats` | `nats:2.14.2` | 异步消息 (Go 重写后用) | **不跑** |
| `tei-cpu` / `tei-gpu` | `${TEI_IMAGE_*}` | HuggingFace Text Embeddings Inference (本地 Embedding 服务) | **不跑** — 直接 import sentence-transformers |
| `kibana` | `kibana:${STACK_VERSION}` | ES 可视化 | **不跑** |
| `deepdoc` | `deepdoc_oss:latest` (顶层 compose) | 视觉 OCR / 表格识别 HTTP 服务 | **不跑** — s11 用了 tesseract 兜底 |
| `ragflow-cpu` / `ragflow-gpu` | `${RAGFLOW_IMAGE}` (顶层 compose) | 主 RAG API + Web 静态资源 (nginx 在容器内) | **等价物:我们的 FastAPI `rag` 服务** |

合计: base 12 个 + 顶层 3 个 (deepdoc / ragflow-cpu / ragflow-gpu),但
用 `profiles` 互相排斥,实际跑起来通常 6-7 个容器一起启。

## 为什么这样写

- **ES / Infinity 单独拆**: 它们是 RAGFlow 的核心检索引擎,既吃内存 (默认
  `MEM_LIMIT`) 又吃磁盘 (索引落盘 + 副本),更重要的是能**水平扩展**
  (按 tenant_id 分索引、按 `number_of_shards` 切分)。拆成独立服务
  方便单独调优内存 / 单独重启 / 单独备份,不影响 API 主进程。我们 MVP
  文档量小,Chroma 单文件 SQLite + hnswlib 完全够,放进程里 0 运维。
- **MySQL / Redis / MinIO 拆三件套**: 它们是不同性质的"状态"——关系数据
  (用户/租户)、队列/缓存 (任务调度)、二进制对象 (原始文件)。三种
  存储的备份策略、扩缩容策略、故障域完全不同,硬塞进一个进程既难
  维护又难扩展。MVP 用户量 = 1,这三类状态都在内存 / 本地目录里,
  重启即丢,可接受。
- **MVP 12 章只跑 1 个容器**: 我们没有多租户 (省 MySQL)、没有任务队列
  (省 Redis)、没有大文件 (省 MinIO)、没有沙箱 (省 executor-manager)、
  没有水平扩展需求 (省 ES 集群)。把 FastAPI 起一个进程,把 Chroma
  索引目录 + samples 目录挂进去,`POST /qa` 一问一答,这就是**最小可
  部署的 RAG 服务**。代价是: ① 没有用户隔离;② 索引文件锁住时无法
  多副本跑;③ LLM key 走环境变量而不是 KMS。生产化要做的事,见
  `thinking_answers.md` 思考题。
