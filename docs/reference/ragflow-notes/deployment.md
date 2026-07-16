# RAGFlow 部署架构

## 一句话
RAGFlow 把"文档解析 / 向量检索 / 关系数据 / 消息队列 / 对象存储 / 沙箱 / 监控"拆成 10+ 容器；MVP 12 章只跑 1 个 FastAPI 容器，把"工程复杂度"换成"上手速度"。

## 来源
- 仓库：https://github.com/infiniflow/ragflow
- 模块：`docker/docker-compose-base.yml` + `docker/docker-compose.yml`
- 关联：本仓库 s12 `fastapi_docker.py`（1 容器版）

## 服务清单

顶层 compose 用 `include: ./docker-compose-base.yml`，实际拉起的服务（base + 顶层 profile 合并去重）：

| 服务 | 角色 | 我们的取舍 |
|---|---|---|
| `es01` / `opensearch01` / `infinity` | 全文 + 向量检索引擎（BM25 + HNSW + 倒排） | **不跑** — s05 用 Chroma 本地文件代替 |
| `oceanbase` / `seekdb` | 分布式关系库 | **不跑** |
| `sandbox-executor-manager` | Agent 沙箱（代码执行隔离） | **不跑** — MVP 没工具调用 |
| `mysql` | 主关系库（用户/租户/知识库元数据） | **不跑** — 用进程内存代替 |
| `minio` | 对象存储（原始 PDF / 解析结果） | **不跑** — 用 samples 目录挂载 |
| `redis` / `nats` | 队列 + 任务调度 + 缓存 / 异步消息 | **不跑** — 单进程同步处理 |
| `tei-cpu` / `tei-gpu` | HuggingFace TEI Embedding 服务 | **不跑** — 直接 import sentence-transformers |
| `kibana` | ES 可视化 | **不跑** |
| `deepdoc` | 视觉 OCR / 表格识别 HTTP 服务 | **不跑** — s11 用了 tesseract 兜底 |
| `ragflow-cpu` / `ragflow-gpu` | 主 RAG API + Web 静态资源 | **等价物：我们的 FastAPI `rag` 服务** |

合计 base + 顶层共 15 个，但用 `profiles` 互相排斥，实际跑起来通常 6-7 个容器一起启。

## 为什么这样写

- **ES / Infinity 单独拆**：核心检索引擎既吃内存（`MEM_LIMIT`）又吃磁盘（索引落盘 + 副本），更重要的是能**水平扩展**（按 tenant_id 分索引、按 `number_of_shards` 切分）。拆成独立服务方便单独调优内存 / 单独重启 / 单独备份，不影响 API 主进程。MVP 文档量小，Chroma 单文件 SQLite + hnswlib 完全够，放进程里 0 运维。
- **MySQL / Redis / MinIO 拆三件套**：三类不同性质的"状态"——关系数据 / 队列缓存 / 二进制对象，备份策略、扩缩容策略、故障域完全不同，硬塞进一个进程既难维护又难扩展。MVP 用户量 = 1，三类状态都在内存 / 本地目录里，重启即丢，可接受。
- **MVP 只跑 1 个容器**：没有多租户（省 MySQL）、没有任务队列（省 Redis）、没有大文件（省 MinIO）、没有沙箱（省 executor-manager）、没有水平扩展需求（省 ES 集群）。把 FastAPI 起一个进程，把 Chroma 索引目录 + samples 目录挂进去，`POST /qa` 一问一答，就是**最小可部署的 RAG 服务**。代价：① 没有用户隔离；② 索引文件锁住时无法多副本跑；③ LLM key 走环境变量而不是 KMS。
