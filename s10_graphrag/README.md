# s10 GraphRAG — 把"实体之间的关系"建成图来查

> **章节定位**:RAG 的"关系面"。s06-s09 都把"找相关段落"当终点——但有一类问题,**段落相关性不够**:"X 和 Y 有什么关系?"、"提到 Z 的产品都有哪些?"、"A 公司投资了谁?被谁投资?"。向量检索给你"包含 X 的段"和"包含 Y 的段",但**不会告诉你 X 和 Y 之间那条边**。本章把段落里**实体之间的指向关系**抽出来,建一张图;查的时候"先定位起点实体 → 沿着边走 1 跳 / N 跳"——这就是 GraphRAG 的最小形态。
>
> **章节结构**:本章用 2 个 unit 走完"从 LLM 抽三元组到 1 跳图查询"——**unit 01** 演示手写 prompt 抽 (head, rel, tail) 三元组 + JSON 容错解析(`<think>` strip / ```json``` fence strip / dict·list fallback),**unit 02** 跑纯内存 1 跳图查询(不调 LLM,O(1) `dict.get`)。
>
> **scope 注意**:本章实现是**手写 LLM 抽取 + 进程内 dict 图**——不是 RAGFlow `general/extractor.py` 那种并发 `asyncio.Semaphore` + entity resolution + Leiden 社区检测的工业实现。

---

## 章节导航

| Unit | 标题 | 入口 |
| --- | --- | --- |
| 01 | LLM 抽实体关系三元组 + 持久化 JSONL(自带容错解析:`<think>` strip / ```json``` fence strip / dict·list fallback) | [`code_01_extract.py`](code_01_extract.py) |
| 02 | 1 跳图查询(纯内存,O(1) `dict.get`,无 LLM;章节核心) | [`code_02_query.py`](code_02_query.py) |

跑法:

```bash
python s10_graphrag/code_01_extract.py    # 抽三元组,落 _graph.jsonl
python s10_graphrag/code_02_query.py      # 离线查图(交互式)
```

依赖:复用 s05-s08 全部产出 + `openai` SDK + `pypdf` + `python-docx`(已在 requirements.txt);`LLM_API_KEY` 必填——unit 01 调 LLM 抽实体;unit 02 离线查询,不依赖 key。

---

## 一、什么是 GraphRAG?

### 1.1 核心定义

**GraphRAG(Graph-based Retrieval-Augmented Generation)** 是一种把"段落检索"升级为"图谱查询"的范式:把每段文字里的实体和实体间关系抽出来,建一张知识图谱;查的时候先定位起点实体、再沿着边(关系)走 N 跳邻居,最后把邻居信息拼成上下文喂给 LLM。它的核心思想是——**段落的"相似度"只回答"哪段相关",而图的"邻接关系"才能回答"实体之间有什么关系"**。

```
   chunk 段落文字                (head, rel, tail) 三元组                1 跳邻居查询
   "R3630 G5 支持 32 条 DDR4"    ┌───────────────────────────┐         query_graph(graph, "R3630 G5")
        │                        │ (R3630 G5, 支持内存, DDR4 3200)      │
        │  EXTRACT_PROMPT + LLM   │ (R3630 G5, 面向场景, AI 推理)         ▼
        ▼                        │ (R3630 G5, 适配行业, 金融)         [("支持内存", "DDR4 3200"),
   list[dict]                     └───────────────────────────┘          ("面向场景", "AI 推理"),
   每段一次 LLM 抽取                                              ("适配行业", "金融")]
        │
        ▼
   build_graph → dict[head] → set[(rel, tail)]   ← O(1) dict.get
        │
        ▼
   save_graph → _graph.jsonl (持久化, 离线可重查)
```

> 💡 **一句话总结**:GraphRAG = 实体抽取(Extraction)+ 图构建(Graph)+ 图检索(Graph Query)。
>
> 让 RAG 从"找相似的段"升级为"沿着边走的图查询"——既能查"X 和 Y 之间什么关系",也能查"Z 的所有合作伙伴"。

GraphRAG 不是替代向量检索,而是在向量检索**答不全**的地方补一刀。两者的职责清晰分工:

| 检索范式 | 答得好的问题 | 答不好的问题 |
|---|---|---|
| **向量检索(s06-s08)** | "X 是什么?"、"X 的关键参数?"、"X 和 Y 类似吗?" | "X 和 Y 之间什么关系?"、"提到 Z 的所有产品?"、"A 公司的供应链有哪些环节?" |
| **图检索(s10)** | "X 的 1 跳 / N 跳邻居"、"X 出发走哪条关系链能到 Y"、"X 所属的社区里有谁" | "X 是什么"(没有节点信息时图也不知道)、长段落里的细粒度语义 |

> 💡 "图"和"向量"在 RAG 系统里是**互补关系**——一个看相似度,一个看邻接关系。

### 1.2 知识图谱的三元组 schema

GraphRAG 的图谱基础数据长这样——`(head, rel, tail)` 三元组列表:

```
(head="紫光恒越技术有限公司", rel="版权所有", tail="紫光恒越 R3630 G5 双路机架式服务器 产品白皮书 v1.0")
(head="紫光恒越 R3630 G5",     rel="支持内存", tail="DDR4 3200 内存(最大 32 条)")
(head="紫光恒越 R3630 G5",     rel="面向场景", tail="AI 推理")
```

把同一段文字里所有三元组合并,就得到一张**有向图**——`dict[head] → set[(rel, tail)]`。边的方向**永远是 head → tail**,"X 拥有 Y" 和 "Y 被 X 拥有" 是两条不同边(不能因为 tail 相同就合并)。MVP 把这条约束写进 `build_graph` 的 `graph[t["head"]].add((t["rel"], t["tail"]))`——`set` 里 `(rel, tail)` 是联合键,**关系不同就算同一对实体也是两条边**。

> 💡 三元组 schema 不是 GraphRAG 独有——它来自知识图谱领域的 Resource Description Framework (RDF) 标准。GraphRAG 是把这个 schema 套到 RAG 检索管线上的产物。

### 1.3 从三元组到图查询:1 跳邻居最小例子

把上面 3 条三元组喂给 `build_graph` + `query_graph`:

```python
graph = build_graph([[
    {"head": "紫光恒越技术有限公司", "rel": "版权所有",   "tail": "R3630 G5 白皮书"},
    {"head": "R3630 G5",            "rel": "支持内存",   "tail": "DDR4 3200"},
    {"head": "R3630 G5",            "rel": "面向场景",   "tail": "AI 推理"},
]])
query_graph(graph, "R3630 G5")
# → [("支持内存", "DDR4 3200"),
#    ("面向场景", "AI 推理")]
query_graph(graph, "紫光恒越技术有限公司")
# → [("版权所有", "R3630 G5 白皮书")]
query_graph(graph, "不存在的实体")
# → []   # 1 跳查不到 = (空集)
```

**关键观察**:从 `R3630 G5` 出发能拿到 2 条边(1 跳邻居),从 `紫光恒越技术有限公司` 出发能拿到 1 条边;不存在的实体直接返空——**O(1) `dict.get` 的硬约束,不调 LLM、不查向量库**,纯内存结构遍历。MVP 就这一条最简单的"图查询"。

### 1.4 GraphRAG 与向量检索的本质区别

把它放进 RAG 全景看:**s06-s08 是"段落相似度 → 生成"的检索范式**,**s10 是"实体邻接 → 生成"的图检索范式**。同一问题两种处理对比:

| 维度 | 向量检索 (s06-s08) | 图检索 (s10) |
|---|---|---|
| **索引单元** | 文本块(chunk) | 实体 + 关系边(triple) |
| **匹配方式** | 余弦相似度(embed → top-k) | 邻接关系(dict.get → 1 跳 / BFS → N 跳) |
| **适合问题** | "X 是什么"、"X 的关键参数" | "X 和 Y 什么关系"、"X 的合作伙伴" |
| **索引成本** | 一次 embed(秒级/段) | 每段一次 LLM 抽取(秒级/段,token 贵) |
| **查询成本** | 一次 embed + 向量召回(毫秒) | 内存 dict.get(微秒) |
| **失败模式** | 召回不到、相似段不含答案 | 实体名歧义、图谱不连通、三元组抽取不全 |
| **实现成本** | 30 行 embed + Chroma 查询 | 30 行 LLM prompt + 30 行 dict 图构建 |

本章只演示**最小图检索**——手写 prompt 抽三元组 + 1 跳 `dict.get`;RAGFlow `general/extractor.py` 把"抽取"做成并发 `asyncio.Semaphore(MAX_CONCURRENT=10)`,把"图存储"做成 Elasticsearch 倒排索引(`knowledge_graph_kwd` 区分 entity / relation / community_report),见 §四。

---

## 二、为什么单独写一章 GraphRAG

`extract_triples(text)` + `build_graph(triples_list)` + `query_graph(graph, entity)` 加一起 80 行就能跑出"图谱 + 1 跳查询"。看起来不值得单独一章。但把它放进 s08 的"向量召回"对照看会发现:**"段落相似度"和"实体邻接关系"答的是两类不同问题**——这道鸿沟由 3 类典型失败堆起来。

### 2.1 真实世界的问题 (3 条典型)

1. **"X 和 Y 之间什么关系"答不上**——向量检索给你"包含 X 的段"和"包含 Y 的段",但**不会告诉你 X 和 Y 之间那条边**。如果 X 和 Y 之间的"版权所有 / 投资 / 合作"关系只在第 3 段被一笔带过,top-k 不一定命中那一段;但**图谱里这条边是显式存在的**,`query_graph(graph, X)` 直接拿到。**生产解法**:双路召回——向量 + 图同时跑,命中任一即返回。RAGFlow `KGSearch` 走的就是这条路(见 §四)。
2. **"提到 Z 的所有产品"召回不全**——向量检索靠相似度,可能漏掉"和 Z 距离远但关系明确"的段落;图谱里 Z 是节点,所有"rel: 提及 / 包含 Z"的边都是显式的,`query_graph(graph, Z)` 一次拿到全集。**生产解法**:实体名 → 邻接边列表 一步搞定,O(1) `dict.get` 比向量召回更准更便宜。
3. **宏观问题"文档集主要在讲什么"答不好**——这是 GraphRAG 的进阶用法:把图做 hierarchical Leiden 社区检测,每个社区生成一段 `community_report` summary,存为 `knowledge_graph_kwd="community_report"` 的块,用来回答"文档集宏观主题是什么"这种需要跨实体聚合的问题。**MVP 不做社区检测**,留作 RAGFlow 对照项(见 `docs/reference/ragflow-notes/graph_extraction.md`)。**生产解法**:hierarchical Leiden + LLM summary 喂宏观问题。

### 2.2 为什么必须在 GraphRAG 上显式投入

每条失败模式都对应一种工业级解法——双路召回(图 + 向量)、实体邻接扩展、社区检测 + summary。**s10 的目标不是解决它们,而是把它们显式暴露出来,让你看到纯向量检索的边界**。这跟 s08 把"toy prompt 在哪里会塌"显式对比是同一种思路——**叙述载体从"4 条 prompt 硬约束"换成"3 个图函数 + 1 跳 query"**,但"先跑通 toy、再讲清楚 toy 在哪里会塌"的教学哲学是一致的。

这也是为什么本章有 2 个 unit 而不是 1 个:

- **unit 01**——跑通最小骨架(`EXTRACT_PROMPT` + `_llm_json` + `extract_triples` + `build_graph` + `save_graph`),演示"LLM 能不能稳定吐 JSON 数组"。把"抽取"和"查询"拆成 2 个 unit 是为了让"LLM 抽取的容错设计"和"纯内存查询"分两段讲——单步看到抽取的脆弱性(`<think>` 推理块 / ```json``` fence / list vs dict),多步看到图查询的纯净性(不调 LLM、O(1) 查)。
- **unit 02**——在 unit 01 之上加 `load_graph` + `query_graph` + 交互式 REPL,演示完整图查询流程。复用 unit 01 落盘的 `_graph.jsonl`(JSONL 每行一个 `{head, rel, tail}`),新增的 30 行只关心"图怎么读回 + 怎么查询 + 怎么交互"。

---

## 三、怎么做？

### 3.1 跑起来

```bash
pip install openai pypdf python-docx   # 已在 requirements.txt
python s10_graphrag/code_01_extract.py   # 抽三元组 → _graph.jsonl
python s10_graphrag/code_02_query.py     # 交互式查图
# 查哪个实体: 紫光恒越技术有限公司
```

环境变量:

- `LLM_API_KEY` — OpenAI 兼容 API key;**必填**,unit 01 调 LLM 抽实体。无 key 时 unit 01 会 `KeyError`。
- `LLM_BASE_URL` — 默认 `https://api.openai.com/v1`,可换任意 OpenAI 兼容 endpoint(MiniMax / DeepSeek / 智谱 / 月之暗面等)。
- `LLM_MODEL` — 默认 `gpt-4o-mini`,可换 `MiniMax-M3`、`deepseek-chat` 等。

无 key / 离线环境跑 unit 02:

```bash
python s10_graphrag/code_02_query.py   # 不调 LLM,只读 _graph.jsonl
# 图为空或缺失: .../s10_graphrag/_graph.jsonl
# 请先跑: python s10_graphrag/code_01_extract.py
```

### 3.2 核心函数一览

s10 的代码拆得很细,每个函数都对应一种"实体抽取 / 图构建 / 图查询"的角色:

| 函数 | 文件 | 输入 | 输出 | 一句话解释 |
|---|---|---|---|---|
| `EXTRACT_PROMPT` | `code_01_extract.py` | — | `str` | system prompt 里的"抽三元组 + JSON 数组"常量 |
| `_llm_json(prompt)` | `code_01_extract.py` | `str` | `list[dict]` | OpenAI 兼容 LLM 调用 + `<think>` 剥除 + ```json``` fence 剥除 + dict·list fallback(`@lru_cache` 模型缓存模式同 s08 unit 01) |
| `extract_triples(text)` | `code_01_extract.py` | `str` | `list[dict]` | 对一段文字跑抽取,返回 `[{head, rel, tail}, ...]` |
| `build_graph(triples_list)` | `code_01_extract.py` | `list[list[dict]]` | `dict[head, set[(rel, tail)]]` | 把所有 chunk 的三元组合并成图 |
| `save_graph(graph, path)` | `code_01_extract.py` | `(dict, Path)` | `None` | 持久化 JSONL,每行 `{head, rel, tail}` |
| `load_graph(path)` | `code_02_query.py` | `Path` | `dict[head, set[(rel, tail)]]` | 从 JSONL 读回图(缺失返空) |
| `query_graph(graph, entity)` | `code_02_query.py` | `(dict, str)` | `list[(rel, tail)]` | 1 跳邻居查询,按 `(rel, tail)` 字母序排便于对照 |
| `main()` (unit 01) | `code_01_extract.py` | — | 打印图节点数 / 边数 | unit 01 演示入口,跑完落盘 `_graph.jsonl` |
| `main()` (unit 02) | `code_02_query.py` | — | 交互式 REPL | unit 02 演示入口,输入实体名查 1 跳邻居 |

### 3.3 图设计取舍

为什么 schema 用 `(head, rel, tail)` 三元组 + `dict[head] → set[(rel, tail)]`、而不是别的?几个常见取舍的折中:

- **三元组 vs N 元组**——MVP 走 3 元组,够演示"实体 + 关系"的核心点;N 元组(带时间 / 强度 / 来源)能携带更多信息但 prompt 复杂度翻倍。**生产里 RAGFlow 用 4 元组**(head / tail / rel / description + strength),见 `docs/reference/ragflow-notes/graph_extraction.md`。
- **`dict[head]` vs `dict[(head, tail)]`**——MVP 走 `dict[head] → set[(rel, tail)]`,**从 head 出发查 1 跳邻居是 O(1) `dict.get`**;从 tail 反查会变成 O(N) 遍历("被 X 投资的所有公司"要扫所有 edges)。**生产里要建双向索引**(同时存 `dict[tail] → set[(rel, head)]`),RAGFlow 的 `n_hop_with_weight` 字段就是预先展开的双向邻接表。
- **JSON vs 结构化分隔符**——MVP 走裸 JSON(`response_format={"type": "json_object"}` + `<think>` 剥除);RAGFlow 走**三段式分隔符**(`{tuple_delimiter}` + `{record_delimiter}` + `{completion_delimiter}`,默认 `<SEP>` / `##` / `<|COMPLETE|>`),见 `docs/reference/ragflow-notes/graph_extraction.md` §1。**分隔符格式正确率 >90%**(实测 Claude / GPT-4),裸 JSON 在中文 / MiniMax-M3 / qwen 这一档模型上格式正确率 <30%——MVP 走裸 JSON 是为了让 prompt 足够简单、能跑通核心点;生产请切分隔符。
- **`set` vs `list` 存邻接边**——MVP 用 `set`,**同一 `(rel, tail)` 多次出现自动去重**;`list` 简单但会在"同段重复抽到同一三元组"时污染边集合。代价是 `set` 不保证顺序,unit 02 用 `sorted(graph.get(entity, set()))` 显式排。
- **`build_graph` 单层 vs 多层**——MVP 走单层 dict(节点名即 key);RAGFlow 把同一实体名跨段出现时**合并 description**(`<SEP>` 拼接,超过 12 段送 LLM 摘要压缩),并按 `entity_type` 分桶做 entity resolution。**MVP 完全不做合并**——`紫光恒越` 和 `紫光恒越技术有限公司` 是两个节点,召回时只能命中其中一个。

### 3.4 如何切换到工业级 GraphRAG

加一种 GraphRAG 策略(Leiden 社区检测 / entity resolution / 结构化分隔符 prompt)只要三步:

1. 把 `_llm_json(prompt)` 换成走分隔符的版本(prompt 改成 `<|><SEP><|COMPLETE|>` 三段式,`utils.handle_single_entity_extraction` 解析),从结构化 token 序列里取 `(entity<tuple_delimiter>name<tuple_delimiter>type<tuple_delimiter>description)` 而不是 JSON `dict`;
2. 在 `build_graph` 之后加 `entity_resolution(graph)` 两阶段管线(字符串相似度粗筛 → LLM batch 精审,参考 `rag/graphrag/entity_resolution.py:81-150`),把同义实体合并成 canonical name;
3. 加 `hierarchical_leiden(graph, max_cluster_size=12)` 跑社区检测,每层每个社区送 LLM 生成 `community_report`,存为 `knowledge_graph_kwd="community_report"` 的块,喂宏观问题。

不要在 `extract_triples` 里写 `if mode == "json": ... elif mode == "delimiter": ...` 之类分发——它会污染单一职责。`extract_triples` 只懂 JSON,`main()` 懂全抽取模式。本章 MVP 只跑 JSON,但接口形状留好了。

### 3.5 实际跑出来的图形状

把 unit 01 跑在仓库自带的 `samples/` 上,落盘的 `_graph.jsonl` 长这样(实测,`MiniMax-M3 over minimaxi.com`,samples = `server_whitepaper.pdf` + `disclosure.docx`,只取前 8 个 chunk):

```jsonl
{"head": "紫光恒越 R3630 G5", "rel": "适配行业", "tail": "金融"}
{"head": "紫光恒越 R3630 G5", "rel": "提供接口", "tail": "Web GUI"}
{"head": "紫光恒越 R3630 G5", "rel": "内置模块", "tail": "BMC"}
{"head": "紫光恒越 R3630 G5", "rel": "面向场景", "tail": "AI 推理"}
{"head": "紫光恒越 R3630 G5", "rel": "支持内存", "tail": "DDR4 3200 内存(最大 32 条)"}
```

unit 02 跑起来:

```
图节点数: 8, 边数: 6

查哪个实体 (回车退出): 紫光恒越技术有限公司
  紫光恒越技术有限公司 --版权所有--> 紫光恒越 R3630 G5 双路机架式服务器 产品白皮书 v1.0
  紫光恒越技术有限公司 --拥有版权--> 紫光恒越 R3630 G5 双路机架式服务器 产品白皮书 v1.0

查哪个实体 (回车退出): 不存在的实体xyz
  (无结果——'不存在的实体xyz' 不在图中或没有出边)
```

不同次跑节点数 / 边数会小幅抖动(LLM 在 `temperature=0` 下对长 prompt 仍有少量随机性;chunk 0/1/2 是封面 + 目录,信息密度低,模型决定抽不抽也有差异)——**这是 LLM 抽取的固有现象,不是 bug**。生产里要做的是把 `temperature=0` + 多次 retry + LLM cache 叠起来(`set_llm_cache` / `get_llm_cache`,见 `docs/reference/ragflow-notes/graph_extraction.md` §2),让重复跑可复现。

**Troubleshooting**:

- `KeyError: 'LLM_API_KEY'`:unit 01 必填 key,`.env` 加 `LLM_API_KEY=sk-...`;unit 02 不依赖 key 但需要先跑过 unit 01。
- `_llm_json` 返回 `[]` / 图节点数抖动大:LLM 没 honor `response_format=json_object`,把 `<think>...</think>` 推理块或 ```json``` fence 当 JSON 解析失败。MVP 已做 `re.sub` 剥除,但仍可能因模型输出版本不同失败——切到结构化分隔符(`<SEP>` 三段式)能从源头解决,参考 `docs/reference/ragflow-notes/graph_extraction.md`。
- `pypdf / docx` 解析空:PDF 是扫描件(无文本层),需要 OCR;DOCX 是图片型。s11 专门讲多模态。
- `UnicodeEncodeError: 'gbk' codec can't encode character`:Windows 控制台编码问题,跑前 `set PYTHONIOENCODING=utf-8`(s05-s09 同问题)。
- "紫光恒越" 和 "紫光恒越技术有限公司" 召回不全:MVP 不做 entity resolution,两个名字是两个节点。生产里走 `entity_resolution.py` 两阶段管线(粗筛 + LLM 精审),详见 `docs/reference/ragflow-notes/graph_extraction.md` §4。

---

## 四、选型与思考题

### 4.1 主流 GraphRAG 范式速览

下面这张表把 GraphRAG 系统的实现路径按"prompt 形式 / 抽取并发度 / 图存储 / 查询路径 / 社区检测"列出来:

| 范式 | prompt 形式 | 抽取并发度 | 图存储 | 查询路径 | 社区检测 | 适用场景 |
|---|---|---|---|---|---|---|
| **手写 JSON prompt + 内存 dict(本章 MVP)** | JSON `response_format` | 串行 | 进程内 dict + JSONL | `dict.get` 1 跳 | 无 | 教学 / 快速原型 / 离线可复现 |
| **结构化分隔符 prompt + 倒排索引(LightRAG / RAGFlow light)** | `<SEP>` 三段式 | 可并发 | Elasticsearch / Infinity | 向量召回实体 + 多跳扩展 | 无 | 生产单租户 / 几万 chunks |
| **微软 GraphRAG 原版 + community summary(general)** | few-shot + 分隔符 | `asyncio.Semaphore(10)` | 倒排索引 + `community_report` chunk | `KGSearch` + 多跳 `n_hop_with_weight` | hierarchical Leiden | 答宏观问题 / 跨实体聚合 |
| **Neo4j / TigerGraph 工业图数据库** | 任意 | 任意 | 独立图数据库 | Cypher / Gremlin | 内置 Louvain / Leiden | 图遍历密集 / 大规模实体 |

我们的 toy `extract_triples` + `query_graph` 在范式复杂度上只占第一行——**手写 JSON + 内存 dict**;RAGFlow 走完整 general 路径,**多一道抽象就多一道观测点 + 一个失败模式**。教学 demo 选 MVP 因为它跑通快、依赖少、依赖全在 prompt 里可见;**生产请按"格式正确率 / 抽取成本 / 是否答宏观问题"做 tier 选型**(MVP → 切分隔符 → RAGFlow general → Neo4j)。

### 4.2 选型速记

- **教学 / 快速原型 / 离线可复现** → 本章 MVP(手写 JSON prompt + 内存 dict + 1 跳 query),无并发、无 merge、无社区检测,代码 ≤ 150 行;
- **生产单租户 / 几万 chunks** → 切 LightRAG / RAGFlow light(`<SEP>` 三段式 + 倒排索引),格式正确率 + 抽取并发度上来了,代码 +200 行换 +300% 鲁棒性;
- **答宏观问题 / 跨实体聚合** → RAGFlow general(hierarchical Leiden + community summary),加一层抽象换"能答'文档集在讲什么'"的能力,token 成本翻倍;
- **图遍历密集 / 大规模实体** → Neo4j / TigerGraph 工业图数据库,Cypher / Gremlin 比 `dict.get` 表达力强,运维成本 10x 但可观测性 +10x;
- **要先看清每个边界再选** → 用本章 unit 01 把"手写 JSON"和"切分隔符"各跑一次,对比格式正确率——这是最简单的"抽取 prompt A/B"实验。

### 4.3 思考题

1. **如果两段文字里同一实体名字不同("产品 A" vs "A 型")怎么办?**
2. **三元组 schema 为什么是 `(head, rel, tail)` 而不是 `dict[entity, attrs]`?**
3. **1 跳不够用怎么办?**

(答案见文末「思考题答案」)

---

## unit 01 — LLM 抽实体关系三元组 + 持久化 JSONL (`code_01_extract.py`)

> 由浅入深第 1 步:把每个 chunk 喂给 LLM 让它吐 `(head, rel, tail)` 三元组,合并成图,写到 `s10_graphrag/_graph.jsonl` 供 unit 02 离线查询。
> unit 02 只读 JSONL,不再调 LLM。

### 这是什么

`code_01_extract.py` 是一个"LLM 抽取 + 容错解析 + 内存合并 + 落盘"的小教学版 GraphRAG 抽取器:

- `EXTRACT_PROMPT` —— 喂文字给 LLM,要求吐 JSON 数组,每项 `{head, rel, tail}`,没有就 `[]`;
- `_llm_json(prompt)` —— 调 OpenAI 兼容接口,**对 MiniMax-M3 的三种坏输出做兜底**:剥 `<think>...</think>` 推理块、剥 ```json``` 围栏、再容忍 `dict / list` 两种 JSON 顶层结构。任一失败返 `[]`,**不抛异常**,让上游 `build_graph` 当作"这个 chunk 没抽到"继续;
- `extract_triples(text)` —— 一段文字 → 一组三元组;
- `build_graph(triples_list)` —— 把所有 chunk 的结果合并成 `dict[head] → set[(rel, tail)]`;
- `save_graph(graph, path)` —— 落盘到 JSONL,每行一个 triple(便于追加 / 复查 / 不加载整图就能 grep);
- `main()` —— **self-contained**:内联 pypdf + python-docx + 简化版 chunking(双换行分段 + 500 字 cap + 取前 8 段),跑固定输入,打印节点 / 边数,落盘。

### 跑起来

```bash
python s10_graphrag/code_01_extract.py
```

输出示例(MiniMax-M3 over minimaxi.com,samples = server_whitepaper.pdf + disclosure.docx,只取前 8 个 chunk):

```
chunks: 8
图节点数: 8, 边数: 6
持久化: s10_graphrag/_graph.jsonl
```

不同次跑节点 / 边数会小幅抖动(LLM 在 temperature=0 下对长 prompt 仍有少量随机性;chunk 0/1/2 是封面 + 目录,信息密度低,模型决定抽不抽也有差异)——这是 LLM 抽取的固有现象,不是 bug。

### 它做对了什么

- **没有 schema 也能跑**:`EXTRACT_PROMPT` 只要求 `(head, rel, tail)` 三元组列表,不限定实体类型(人 / 公司 / 产品)、不要求 entity description、不要求 relationship strength——结构最简,新手也能秒看明白"图是怎么来的";
- **MiniMax-M3 兼容性**:三个客户端兜底串成一行兜底链——剥 `<think>`,剥 ```json``` fence,容忍 dict 或 list 顶层。这套写法 s08 / s09 也在用,跨章节一致;
- **失败不 crash**:JSON parse 失败直接返 `[]`,所以 8 个 chunk 里偶尔有 1-2 个解析坏掉也不会中断整条管线;
- **JSONL 持久化**:一行一个 triple,文本可读、可 grep、可 `wc -l`,不依赖任何图数据库——unit 02 一个 `for line in f` 就能跑 1 跳查询。

### 它做错了什么

- **没有 entity resolution**:`"紫光恒越"` / `"紫光恒越技术有限公司"` / `"紫光恒越技术"` 是 3 个独立节点,3 条独立边——召回只有命中其中一个名字时才能拿到边。生产里要 entity resolution(embedding 聚类 + LLM 判断合并);
- **没有 entity_types 白名单**:模型可能把章节标题(`"3.1 技术规格"`)当成实体,抽到一堆噪声节点。RAGFlow 的 `DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event", "category"]` 就是这一刀;
- **每 chunk 一次 LLM**:8 段 8 次同步调用,顺序阻塞;生产里走 `asyncio.Semaphore(10)` + LLM cache,几千段也不超时;
- **没有并发、没有重试、没有缓存**:LLM 偶尔抽到空、超时、限流,我们的实现一律当成"没抽到";生产要 retry + cache,同一 chunk 重跑不重抽;
- **没有节点 / 边合并策略**:同实体跨段出现时,`build_graph` 直接当新节点处理;生产要走 merge_nodes(type 取 Counter 最大值、description 拼 `<SEP>`)和 merge_edges(weight 累加、keywords 并集)。

---

## unit 02 — 1 跳图查询(纯内存,无 LLM) (`code_02_query.py`)

> 由浅入深第 2 步:加载 unit 01 落盘的 `_graph.jsonl`,在 `dict[head] → set[(rel, tail)]` 上跑 O(1) 的 1 跳邻居查询。
> 这是 GraphRAG 的"读"半边——和单元 01 的"写"半边分开,便于离线调试。

### 这是什么

`code_02_query.py` 把 unit 01 写出的 JSONL 重新加载回内存的图结构,提供最朴素的 1 跳查询:

- `load_graph(path)` —— 按行读 JSONL,反向重建 `dict[head] → set[(rel, tail)]`;同时把 `tail` 也注册成节点,即便它没有出边也能被 query 命中(便于"这个实体存在但孤立"的诊断);
- `query_graph(graph, entity)` —— `graph.get(entity, set())`,O(1);返回前按 `(rel, tail)` 字母序排,便于对比不同次抽取的结果;
- `main()` —— 加载 → 打印节点 / 边数 → 循环输入实体 → 打印 1 跳邻居,直到空行退出。**完全不调 LLM**,只要 `_graph.jsonl` 存在就跑得起。

### 跑起来

```bash
# 1. 先跑一次抽取(生成 _graph.jsonl)
python s10_graphrag/code_01_extract.py

# 2. 再跑查询(可重复跑,不调 LLM)
python s10_graphrag/code_02_query.py
```

实测(MiniMax-M3 抽完 8 个 chunk 后的图):

```
图节点数: 8, 边数: 6
查哪个实体 (回车退出): 紫光恒越技术有限公司
  紫光恒越技术有限公司 --版权所有--> 紫光恒越 R3630 G5 双路机架式服务器 产品白皮书 v1.0

查哪个实体 (回车退出): 不存在的实体xyz
  (无结果——'不存在的实体xyz' 不在图中或没有出边)

查哪个实体 (回车退出):
```

### 它做对了什么

- **O(1) 查询**:`dict.get(entity)` 是哈希查找,8 节点和 80 万节点同一个 latency——只要 entity 名字完全匹配,立刻返回;
- **确定性**:同一个 `_graph.jsonl` 跑 N 次结果完全一致,适合做"prompt 改了之后,看 query 输出有没有变"的回归对照;
- **离线 / 零成本**:不调 LLM、不调 embedding、不调任何外部服务——CI 里跑测试、或调试实体名 / prompt 时可以放心重跑;
- **可中断 / 可恢复**:循环读输入,空行退出——调试时反复改 entity 名快速验证,不用每次重启进程。

### 它做错了什么

- **没有多跳**:`"X 的竞争对手的合作伙伴"` 是 3 跳,1 跳答不全。要 BFS 自己写,而且 3 跳以上必须上 community summary(否则 LLM 拿到的 context 拼接成本太高);
- **没有 entity resolution**:`"紫光恒越"` 和 `"紫光恒越技术有限公司"` 是两个键,query 命中哪个名字才能拿哪个的边——用户一般不会知道图里到底注册了哪个变体名;
- **没有语义匹配**:用户输入 `"紫光"` 也命中不了 `"紫光恒越技术有限公司"`——要么靠 entity resolution 提前归一化,要么 LLM 在 prompt 里把 user query 改写一遍;
- **没有路径权重 / 方向语义**:`set[(rel, tail)]` 不记录反向边、不记录 confidence,复杂关系(`"A 投资 B"` vs `"B 被 A 投资"`)分不清谁主动谁被动;
- **没有图遍历可视化 / 解释**:`query_graph` 只返边列表,不出"为什么是这条边"、"这条边来自哪个 chunk"——bad case 排查要回到 `_graph.jsonl` 里手 grep;
- **节点孤立时看不出来**:`graph.get("孤立实体")` 直接返空集,虽然 unit 01 已经把 tail 注册成节点(让"存在但孤立"能被命中),但**没有入边的节点**和**不存在的节点**在我们的实现里都返回空集,提示完全一样。

---

## 思考题答案

### Q1. 如果两段文字里同一实体名字不同("产品 A" vs "A 型")怎么办?

这是 **entity resolution / entity linking** 问题——图谱质量的天花板就卡在这里。如果不做,"紫光恒越"和"紫光恒越技术有限公司"是图里两个不同节点,所有下游查询(PageRank、社区检测、邻居扩展)都会被撕碎。

按代价从低到高排三种解法:

**1) 规则归一化**(最便宜,能解决 60% 中文场景)——抽取后做字符串归一化:
- 去前后缀("公司"、"有限公司"、"股份有限公司"),去空格、全角半角;
- 中文数字 → 阿拉伯数字;
- 拼音模糊匹配("海光" ≈ "Hygon");
- 用一个简单的别名词典(业务给的 `known_aliases = {"海光": "Hygon", "紫光恒越": "紫光恒越技术有限公司"}`)。

缺点:维护词典贵、对未见过的别名零能力。

**2) Embedding 余弦相似度 + LLM 判断**(生产主流)——先对所有节点名 embed,再层次聚类(cos > 0.85 归一组),每组丢给 LLM 问"这些名字是不是指同一实体"。RAGFlow `rag/graphrag/entity_resolution.py` 就是这个套路——对实体描述做 embed → 聚类 → 让 LLM 给定 canonical name。

**3) 让 LLM 在抽取阶段直接出 canonical name**(最准但最贵)——prompt 改成"请用 canonical name 写实体;如果同一实体出现多种写法,用出现频率最高的那个"。微软 GraphRAG 原版 prompt 不这么做,是图建好后单独跑 entity resolution——把"抽取"和"归一"解耦,归一阶段可以重跑、可以换策略。

**生产推荐**:1 + 2 组合——规则先做一遍快速合并兜底常见中文后缀,剩余疑似冲突送 embedding + LLM 判断。RAGFlow 三种都做(`entity_resolution.py` + `normalize_node_names` 做 `upper().strip()` + `html.unescape`,见 `general/leiden.py:58`)。

### Q2. 三元组 schema 为什么是 `(head, rel, tail)` 而不是 `dict[entity, attrs]`?

`dict[entity, attrs]` 是属性图(property graph),节点带属性、关系不带独立语义;`(head, rel, tail)` 是 RDF 风格关系图,**关系本身是一等公民**——`紫光恒越 R3630 G5 --支持内存--> DDR4 3200` 里的"支持内存"是一条独立边,可查、可遍历、可做社区检测。属性图适合"一个实体有几条属性"(Neo4j Property Graph),关系图适合"实体之间有什么明确关系"(RDF / OWL)。GraphRAG 选 RDF 风格是因为**关系本身携带查询信号**——`query_graph(graph, X)` 拿到的就是"X 出发到 Y 的关系集合",而非"X 的属性集合"。

### Q3. 1 跳不够用怎么办?

BFS 自己写 2-3 跳。`graph.get(x)` 拿到 1 跳邻居后,把邻居当起点再 `graph.get(邻居)` 拿到 2 跳……用 `visited` set 防环、`queue` 维护待访问节点即可。但 N 跳查询有 2 个边界:① token 爆炸——一跳 50 条边、2 跳 2500 条、3 跳 125000 条,喂 LLM 前要 rerank + cap top-k;② 召回失真——跳得越远、信号越弱,生产上 2-3 跳是经验上限。3 跳以上走社区 summary(hierarchical Leiden)才合算。