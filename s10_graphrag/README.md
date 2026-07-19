# s10 GraphRAG — 把"实体之间的关系"建成图来查

[上一章 s09 → · 下一章 s11 → s12]

> *"段落的'相似度'只回答'哪段相关', 而图的'邻接关系'才能回答'实体之间有什么关系' — 抽三元组 + 建图 + 1 跳查询, 80 行跑出'关系面'"*
>
> **链路位置**: RAG 的"关系面"补充 (s06-s09 是"段落相似度", s10 是"实体邻接")
> **代码文件**: extract.py · query.py

> 环境准备: 见 root README §快速开始 — `pip install openai pypdf python-docx` + `.env` 配 `LLM_API_KEY` (仅代码 1 需要, 代码 2 纯内存离线跑)

---

## 问题

s06-s09 都把"找相关段落"当终点 — 向量召回 + rerank + 拼 prompt, 答"X 是什么"、"X 的关键参数"这类问题很稳。但有一类问题, **段落相关性根本不够**: "X 和 Y 有什么关系?"、"提到 Z 的产品都有哪些?"、"A 公司投资了谁、被谁投资?"。这三种问题拆开看, 都是同一类的不同切面 — 答案不在"某一段的文字里", 而在"实体与实体之间的那条边上"。

**第一, "X 和 Y 之间什么关系"答不上**。向量检索给你"包含 X 的段"和"包含 Y 的段", 但**不会告诉你 X 和 Y 之间那条边**。如果 X 和 Y 之间的"版权所有 / 投资 / 合作"关系只在第 3 段被一笔带过, top-k 不一定命中那一段; 即便命中, LLM 也得从散落的文字里自己拼出关系。这是余弦相似度的固有盲区: 它衡量"文字像不像", 不衡量"实体连没连"。

**第二, "提到 Z 的所有产品"召回不全**。向量检索靠相似度打分, 可能漏掉"和 Z 距离远但关系明确"的段落 — 一份文档里 Z 出现在开头, 相关产品散在后面十几段, top-k 只捞回最相似的几段, 剩下的静默丢失。用户以为"我问了全集", 实际拿到子集。图谱里 Z 是一个节点, 所有"提及 / 包含 Z"的边都是显式的, 一次 `dict.get` 拿到全集。

**第三, 宏观问题"文档集主要在讲什么"答不好**。这需要跨实体聚合 — 把整份文档里所有实体和关系汇总成主题。向量检索只能召回 top-k 段落, 没有"全局视野"; 段落相似度对"宏观概括"这类问题天然无力。这是 GraphRAG 的进阶用法 (hierarchical Leiden 社区检测 + community summary), MVP 不做, 但问题本身要先看清。

这三种失败有一个共同解法 — **把"段落检索"补充为"图谱查询"**: 把每段文字里的实体和实体间关系抽成 `(head, rel, tail)` 三元组, 建一张知识图谱; 查的时候先定位起点实体, 再沿着边 (关系) 走 N 跳邻居, 把邻居信息拼成上下文喂给 LLM。这就是 **GraphRAG (Graph-based Retrieval-Augmented Generation)**。它**不是替代向量检索, 而是在向量检索答不全的地方补一刀** — 一个看相似度, 一个看邻接关系, 在 RAG 系统里是互补关系。

s10 的任务就是把"抽三元组 → 建图 → 1 跳查询"用 **80 行 Python 跑通一遍**: 手写 prompt 抽三元组 + 内存 `dict` 建图 + `dict.get` 查 1 跳邻居 — 不做 entity resolution、不做并发、不做社区检测。目标不是解决上面三个问题, 而是把它们显式暴露出来, 让你看清纯向量检索的边界, 以及"图这一刀"补在哪里。

---

## 解决方案

s10 用 **两个递进的脚本** 把 GraphRAG 的"写图"和"读图"两半边跑起来。每一步解决前一步的局限, 但也留下新的脆弱性。

```
   chunk 段落文字                (head, rel, tail) 三元组           1 跳邻居查询
   代码 1 (写图)                                                    代码 2 (读图)
   ┌──────────────────┐        ┌───────────────────────┐         ┌──────────────────┐
   │ 每段喂 LLM       │        │ (R3630 G5, 支持内存, DDR4) │        │ load_graph        │
   │ EXTRACT_PROMPT   │ ─────▶ │ (R3630 G5, 面向场景, AI)   │ ─────▶ │ query_graph       │
   │       │          │        │ (公司, 版权所有, 白皮书)    │        │  dict.get(entity) │
   │       ▼          │        └───────────────────────┘         │  O(1) 1 跳邻居    │
   │ build_graph      │                    │                     └──────────────────┘
   │ dict[head]→set   │                    ▼
   │       │          │        save_graph → _graph.jsonl (持久化)
   └──────────────────┘         ← `extract.py` 写, `query.py` 读, 通过 JSONL 解耦
```

| 脚本 | 解决什么 | 留下什么局限 | 何时用 |
|---|---|---|---|
| `extract.py` | LLM 抽三元组 → 内存合并 `dict[head] → set[(rel, tail)]` → 落盘 `_graph.jsonl` | 无 entity resolution (同名异写各是各); 无并发 / 重试 / 缓存; 无节点合并 | 教学 / 快速原型 / 离线 ETL |
| `query.py` | 读 `_graph.jsonl` 回内存, `dict.get` 跑 O(1) 1 跳查询 | 无多跳; 严格 `name == name` 匹配; 无方向语义 / 路径权重 | 离线可重跑 / 调抽取 prompt 的回归对照 |

两脚本的关系是一条**写-读主干**: 代码 1 把"文字 → 三元组 → 图 → JSONL"做出来, 暴露"LLM 抽取抖动 + 同名异写分裂"的局限 — `紫光恒越` 和 `紫光恒越技术有限公司` 是两个节点; 代码 2 把落盘的图读回内存做 1 跳查询, 暴露"只能查 1 跳 + 严格字符匹配"的局限 — "X 的竞争对手的合作伙伴"这种 3 跳问题答不了。**写-读拆开的价值**: 代码 2 离线零成本可重跑, 调代码 1 的抽取 prompt 时反复重抽, 代码 2 不用每次重走 LLM。

---

## 代码 1: LLM 抽实体关系三元组 ([extract.py](extract.py))

### 工作原理

**做一件事**: 把每个 chunk 喂给 LLM, 让它吐 `(head, rel, tail)` 三元组, 合并成内存图, 落盘到 `_graph.jsonl` 供代码 2 离线查询。

**5 步**:
1. `EXTRACT_PROMPT` — 喂文字给 LLM, 要求吐 JSON, 每项 `{head, rel, tail}`, 没有就返 `[]`
2. `_llm_json(prompt)` — 调 OpenAI 兼容接口, 对 MiniMax-M3 的坏输出做兜底: 剥 `<think>...</think>` 推理块、剥 ` ```json ``` ` 围栏、容忍 `dict / list` 两种 JSON 顶层结构。任一失败返 `[]` **不抛异常**, 让上游当"这个 chunk 没抽到"继续
3. `extract_triples(text)` — 一段文字 → 一组三元组 (`_llm_json` 套 prompt)
4. `build_graph(triples_list)` — 把所有 chunk 的结果合并成 `dict[head] → set[(rel, tail)]`; `set` 里 `(rel, tail)` 是联合键, 关系不同就算同一对实体也是两条边
5. `save_graph(graph, path)` — 落盘 JSONL, 每行一个 triple (便于追加 / grep / 不加载整图就能复查)

```python
# 中间片段: build_graph 把三元组合并成有向图, set 自动去重
for triples in triples_list:
    for t in triples:
        for node in (t["head"], t["tail"]):
            graph.setdefault(node, set())
        graph[t["head"]].add((t["rel"], t["tail"]))
```

**完整函数**:

```python
def _llm_json(prompt: str) -> list[dict]:
    """调 OpenAI 兼容 LLM, 期望返回 JSON 数组或带 'triples' 键的 dict。
    失败一律返回 []——让上层 build_graph 当成"没抽到"继续, 而不是 crash。"""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["LLM_API_KEY"],
                    base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"))
    resp = client.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()   # 剥推理块
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()  # 剥围栏
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return obj.get("triples", [])
        return []
    except json.JSONDecodeError:
        return []


def extract_triples(text: str) -> list[dict]:
    """对一段文字跑抽取, 返回 [{head, rel, tail}, ...]。"""
    return _llm_json(EXTRACT_PROMPT.format(text=text))


def build_graph(triples_list: list[list[dict]]) -> dict:
    """把所有 chunk 的三元组合并成 dict[head] -> set[(rel, tail)]。"""
    graph: dict[str, set[tuple[str, str]]] = {}
    for triples in triples_list:
        for t in triples:
            for node in (t["head"], t["tail"]):
                graph.setdefault(node, set())
            graph[t["head"]].add((t["rel"], t["tail"]))
    return graph


def save_graph(graph: dict, path: Path) -> None:
    """把图持久化成 JSONL(每行 {head, rel, tail}), set 转 list 便于 JSON 序列化。"""
    with path.open("w", encoding="utf-8") as f:
        for head, edges in graph.items():
            for rel, tail in edges:
                f.write(json.dumps({"head": head, "rel": rel, "tail": tail}, ensure_ascii=False) + "\n")
```

### 试一下

```bash
python s10_graphrag/extract.py
```

实测输出 (MiniMax-M3 over minimaxi.com, samples = `server_whitepaper.pdf` + `disclosure.docx`, 只取前 8 个 chunk):

```
chunks: 8
图节点数: 8, 边数: 6
持久化: s10_graphrag/_graph.jsonl
```

落盘的 `_graph.jsonl` 长这样 (实测):

```jsonl
{"head": "紫光恒越 R3630 G5", "rel": "适配行业", "tail": "金融"}
{"head": "紫光恒越 R3630 G5", "rel": "提供接口", "tail": "Web GUI"}
{"head": "紫光恒越 R3630 G5", "rel": "内置模块", "tail": "BMC"}
{"head": "紫光恒越 R3630 G5", "rel": "面向场景", "tail": "AI 推理"}
{"head": "紫光恒越 R3630 G5", "rel": "支持内存", "tail": "DDR4 3200 内存(最大 32 条)"}
```

**观察**: 不同次跑节点 / 边数会小幅抖动 — `temperature=0` 下 LLM 对长 prompt 仍有少量随机性, 且 chunk 0/1/2 是封面 + 目录、信息密度低, 模型决定抽不抽也有差异。这是 LLM 抽取的固有现象, **不是 bug**。生产做法 (retry + LLM cache) 见 `docs/reference/ragflow-notes/graph_extraction.md` §2。

### 为什么不只写这一种

代码 1 只做"抽 + 建 + 存", 留下三个大坑: 没有 **entity resolution** (`紫光恒越` / `紫光恒越技术有限公司` 是 3 个独立节点, 召回只命中其中一个名字才拿得到边); 没有 **entity_types 白名单** (模型可能把章节标题 `"3.1 技术规格"` 当实体抽成噪声节点); 没有**并发 / 重试 / 缓存 / 节点合并** (8 段 8 次同步调用顺序阻塞, 同实体跨段出现直接当新节点)。这张图能写出来, 但还查不了 — 见代码 2 把它读回内存做 1 跳查询。

---

## 代码 2: 1 跳图查询 (纯内存, 无 LLM) ([query.py](query.py))

### 工作原理

**做一件事**: 加载代码 1 落盘的 `_graph.jsonl` 回内存, 在 `dict[head] → set[(rel, tail)]` 上跑 O(1) 的 1 跳邻居查询, 全程不调 LLM。

**3 步**:
1. `load_graph(path)` — 按行读 JSONL, 反向重建 `dict[head] → set[(rel, tail)]`; 同时把 `tail` 也注册成节点 (即便它没有出边也能被 query 命中, 便于"这个实体存在但孤立"的诊断); 缺失 / 空文件返空图
2. `query_graph(graph, entity)` — `graph.get(entity, set())`, O(1); 返回前按 `(rel, tail)` 字母序排, 便于对比不同次抽取的结果
3. `main()` — 加载 → 打印节点 / 边数 → 循环输入实体打印 1 跳邻居, 直到空行退出。**完全不调 LLM**, 只要 `_graph.jsonl` 存在就跑得起

```python
# 中间片段: 1 跳查询就是一次 dict.get + sorted
def query_graph(graph: dict, entity: str) -> list[tuple[str, str]]:
    """从 entity 出发取 1 跳邻居, 按 (rel, tail) 字母序排, 便于对照。"""
    return sorted(graph.get(entity, set()))
```

**完整函数**:

```python
def load_graph(path: Path) -> dict[str, set[tuple[str, str]]]:
    """从 JSONL 读回 dict[head] -> set[(rel, tail)]。缺失 / 空文件返空图。"""
    graph: dict[str, set[tuple[str, str]]] = {}
    if not path.exists():
        return graph
    import json
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            graph.setdefault(t["head"], set()).add((t["rel"], t["tail"]))
            # 顺手把 tail 也注册成节点(就算它没出边, 也让它"存在", 便于 query 命中)。
            graph.setdefault(t["tail"], set())
    return graph


def query_graph(graph: dict, entity: str) -> list[tuple[str, str]]:
    """从 entity 出发取 1 跳邻居, 按 (rel, tail) 字母序排, 便于对照。"""
    return sorted(graph.get(entity, set()))
```

### 试一下

```bash
# 1. 先跑一次抽取 (生成 _graph.jsonl)
python s10_graphrag/extract.py

# 2. 再跑查询 (可重复跑, 不调 LLM)
python s10_graphrag/query.py
```

交互输入实体, 看 1 跳邻居 (实测, MiniMax-M3 抽完 8 个 chunk 后的图):

```
图节点数: 8, 边数: 6
查哪个实体 (回车退出): 紫光恒越技术有限公司
  紫光恒越技术有限公司 --版权所有--> 紫光恒越 R3630 G5 双路机架式服务器 产品白皮书 v1.0

查哪个实体 (回车退出): 不存在的实体xyz
  (无结果——'不存在的实体xyz' 不在图中或没有出边)

查哪个实体 (回车退出):
```

**观察**: 从 `紫光恒越技术有限公司` 出发拿到 1 条边, 不存在的实体直接返空 — **O(1) `dict.get` 的硬约束, 不调 LLM、不查向量库**, 纯内存结构遍历。写-读拆开的好处在这里最明显: 同一份 `_graph.jsonl` 跑 N 次结果完全一致, 适合做"prompt 改了之后, 看 query 输出有没有变"的回归对照。

### 为什么不只写这一种

代码 2 只做 1 跳查询, 留下几个硬边界: **没有多跳** (`"X 的竞争对手的合作伙伴"` 是 3 跳, 要 BFS 自己写, 且 3 跳以上必须上 community summary 否则 context 拼接成本爆炸); **没有 entity resolution / 语义匹配** (严格 `name == name`, `紫光恒越` 查不到 `紫光恒越技术有限公司` 的边); **没有方向语义 / 路径权重** (`set[(rel, tail)]` 不记反向边、不记 confidence, `"A 投资 B"` vs `"B 被 A 投资"` 分不清主被动)。这些都指向 s11 (多模态补全表格 / OCR) 和生产级 GraphRAG (双路召回 + 社区检测) — 见"接下来"。

---

## 接下来

s10 是 GraphRAG 的**最小骨架**: 手写 prompt 抽三元组 + 内存 `dict` 建图 + `dict.get` 查 1 跳。它把"关系面"这一刀补在了向量检索答不全的地方, 但每一步都极简, 这些脆弱性是生产级 GraphRAG 的填空目标:

- **代码 1 无 entity resolution** — `紫光恒越` / `紫光恒越技术有限公司` 是独立节点, 召回被撕碎。生产走两阶段管线 (embedding 聚类粗筛 + LLM 精审), RAGFlow 的 `general/entity_resolution.py` 就是这条路 — 错误率比"一次性抽 + 合并"低 30-50%。
- **代码 1 无并发 / 无 entity_types 白名单** — 8 段顺序阻塞、章节标题被当噪声实体。生产走 `asyncio.Semaphore(10)` + `DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event", "category"]` 白名单过滤。
- **代码 2 只有 1 跳 + 严格字符匹配** — 多跳问题答不了、同名异写查不全。生产走双路召回 (向量 + 图同时跑, 命中任一即返回, RAGFlow `KGSearch`) + hierarchical Leiden 社区检测 (答"文档集在讲什么"这种跨实体聚合的宏观问题)。

s11 **多模态**: 代码 1 的 self-contained loader 内联了 pypdf + python-docx, 遇到扫描件 PDF / 图片型 DOCX 会解析空 — s11 专门讲多模态, 用 `VisionParser` + OCR 兜底, 让 GraphRAG 的抽取输入不再依赖"有文本层"的文档。这是把"玩具图谱"推向"工业图谱"的输入侧加固。

详细摘录与"为什么这样写"的分析见 [`docs/reference/ragflow-notes/graph_extraction.md`](../docs/reference/ragflow-notes/graph_extraction.md)。

---

## 思考题

1. **如果两段文字里同一实体名字不同（"产品 A" vs "A 型"）怎么办？**
2. **三元组 schema 为什么是 `(head, rel, tail)` 而不是 `dict[entity, attrs]`？**
3. **1 跳不够用怎么办？**

（答案见文末「思考题答案」）

---

## 思考题答案

### Q1. 如果两段文字里同一实体名字不同（"产品 A" vs "A 型"）怎么办？

这是 **entity resolution / entity linking** 问题——图谱质量的天花板就卡在这里。如果不做，"紫光恒越"和"紫光恒越技术有限公司"是图里两个不同节点，所有下游查询（PageRank、社区检测、邻居扩展）都会被撕碎。

按代价从低到高排三种解法：

**1) 规则归一化**（最便宜，能解决 60% 中文场景）——抽取后做字符串归一化：
- 去前后缀（"公司"、"有限公司"、"股份有限公司"），去空格、全角半角；
- 中文数字 → 阿拉伯数字；
- 拼音模糊匹配（"海光" ≈ "Hygon"）；
- 用一个简单的别名词典（业务给的 `known_aliases = {"海光": "Hygon", "紫光恒越": "紫光恒越技术有限公司"}`）。

缺点：维护词典贵、对未见过的别名零能力。

**2) Embedding 余弦相似度 + LLM 判断**（生产主流）——先对所有节点名 embed，再层次聚类（cos > 0.85 归一组），每组丢给 LLM 问"这些名字是不是指同一实体"。RAGFlow `rag/graphrag/entity_resolution.py` 就是这个套路——对实体描述做 embed → 聚类 → 让 LLM 给定 canonical name。

**3) 让 LLM 在抽取阶段直接出 canonical name**（最准但最贵）——prompt 改成"请用 canonical name 写实体；如果同一实体出现多种写法，用出现频率最高的那个"。微软 GraphRAG 原版 prompt 不这么做，是图建好后单独跑 entity resolution——把"抽取"和"归一"解耦，归一阶段可以重跑、可以换策略。

**生产推荐**：1 + 2 组合——规则先做一遍快速合并兜底常见中文后缀，剩余疑似冲突送 embedding + LLM 判断。RAGFlow 三种都做（`entity_resolution.py` + `normalize_node_names` 做 `upper().strip()` + `html.unescape`，见 `general/leiden.py`）。

### Q2. 三元组 schema 为什么是 `(head, rel, tail)` 而不是 `dict[entity, attrs]`?

`dict[entity, attrs]` 是属性图（property graph），节点带属性、关系不带独立语义；`(head, rel, tail)` 是 RDF 风格关系图，**关系本身是一等公民**——`紫光恒越 R3630 G5 --支持内存--> DDR4 3200` 里的"支持内存"是一条独立边，可查、可遍历、可做社区检测。属性图适合"一个实体有几条属性"(Neo4j Property Graph），关系图适合"实体之间有什么明确关系"(RDF / OWL）。GraphRAG 选 RDF 风格是因为**关系本身携带查询信号**——`query_graph(graph, X)` 拿到的就是"X 出发到 Y 的关系集合"，而非"X 的属性集合"。

### Q3. 1 跳不够用怎么办？

BFS 自己写 2-3 跳。`graph.get(x)` 拿到 1 跳邻居后，把邻居当起点再 `graph.get(邻居)` 拿到 2 跳……用 `visited` set 防环、`queue` 维护待访问节点即可。但 N 跳查询有 2 个边界：① token 爆炸——一跳 50 条边、2 跳 2500 条、3 跳 125000 条，喂 LLM 前要 rerank + cap top-k；② 召回失真——跳得越远、信号越弱，生产上 2-3 跳是经验上限。3 跳以上走社区 summary(hierarchical Leiden）才合算。


> 图为空 / 节点名歧义 / entity resolution 等现象详见 ``extract.py`` / ``query.py`` 的 `### 局限与下一步`。

