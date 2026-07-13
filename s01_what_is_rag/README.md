# s01 RAG 入门 — 把"开卷考试"用 30 行代码跑一遍

> **本章定位**：s01 是 12 章的入门章，3 个脚本递进：朴素子串 → 词袋向量 → 完整 RAG 链路。详细定位见 s00 §1.4；RAGFlow 实现见本章末"## RAGFlow 实现"。

---

## 一、章节介绍

把 RAG 全链路的"最小闭环"放在第 1 章，原因是后续 11 章都会回到这条主干做替换。一上来如果讲"文档解析 → embedding → 向量库 → rerank → prompt"，读者会"只见森林不见树"——看得到每个环节，看不到为什么这套环节能组合出 RAG。本章用最朴素的 3 个脚本把"retrieve + augment + generate" 这三个动词绑死在一条线上，后面的章节再讲每个动词怎么替换成工业实现。

### 1.1 核心定义

**RAG = 检索 + 增强 + 生成**。详细定义 + 开卷考试比喻见 [s00 §1.1](../s00_concepts/README.md#一1-一句话定义)。本章要解决的是：把这条主干在 30-80 行代码里跑一遍。

3 个脚本的递进关系（子串 → 词袋向量 → 完整链路）：

```
01 (子串)                02 (词袋向量)                03 (完整 RAG)
┌────────────────┐        ┌─────────────────────┐       ┌──────────────────────┐
│ 段落列表       │        │ 段落 → 词频向量     │       │ 同 02 检索            │
│       │        │        │       │             │       │       │              │
│       ▼        │        │       ▼             │       │       ▼              │
│ 子串匹配       │ ─演进─▶ │ cosine 排序 top-k   │ ─演进─▶ │ top-k 拼 prompt      │
│       │        │        │       │             │       │       │              │
│       ▼        │        │       ▼             │       │       ▼              │
│ 直接返回       │        │ 直接返回            │       │ LLM 生成答案         │
└────────────────┘        └─────────────────────┘       └──────────────────────┘
   没排序、没语义              sparse 语义、有分数              真正"开卷"
```

两条主轴线：检索质量（01 → 02 → s04 → s06）和生成质量（01 → 03 → s07 → s08）。

### 1.2 真实世界的问题

1. **检索质量不足**——朴素子串找不到同义词（问"营收"找不到"营业收入"）、找到关键词不一定是答案（一段提到"应收账款"的列表，关键词命中很多，但不是用户想问的"如何计提坏账"）。**对应 RAG 的召回问题**，用 s04 embedding + s06 混合检索解决。
2. **生成质量不足**——LLM 在没有资料约束时会编造数字（"按惯例审计费用通常为 50 万元"），甚至在 prompt 里被诱导偏离资料。**对应 RAG 的 prompt 工程 + 拒答问题**，用 s07 rerank 把更准的资料喂给 LLM，s08 prompt 模板做硬约束。

### 1.3 为什么必须在第 1 章就讲清楚

- **建立直觉**：让 LLM"开卷考试"，先得有个"卷"。本章 3 个脚本就是"卷"的最朴素形态——子串匹配、词袋向量、top-k 拼 prompt。
- **锁定主干**：12 章的任何一章，无论讲切块 / embedding / 向量库 / rerank，最终都对应这条主干的一个环节替换。第 1 章锁定了这条主干，后面 11 章填空就好。
- **给后续章节留接口**：02 的 `retrieve(q, paragraphs, k=3)` 是 s04-s06 替换的目标；03 的 `build_prompt` 是 s08 替换的目标；03 的 `call_llm` 是 s08 / s09 替换的目标。本章接口形状留好了，后面章节照着替换。

### 1.4 基础数据 schema

01 和 02 的输入都是同一份段落列表：

```python
paragraphs: list[str]    # 整份文档切成段落(按 `\n\n` 切)
```

02 把每段转成词袋向量：

```python
vocab: dict[str, int]                    # {token: index}
vec: list[int]                           # 长度 == len(vocab),值为该 token 在段落里的出现次数
cosine(a, b) = dot(a, b) / (norm(a) * norm(b))   # 范围 [0, 1],越大越相似
```

03 在 02 的检索之上拼 prompt：

```
用户问题 ─▶ retrieve(q, paragraphs, k=3) ─▶ top-3 hits
                                                  │
                                                  ▼
                                          build_prompt
                                                  │
                                                  ▼
                                            LLM.generate
                                                  │
                                                  ▼
                                              答案
```

prompt 用 `<context>...</context>` 标签包裹资料，避免 prompt injection。

---

## 二、朴素关键词检索（子串匹配）：[code_01_naive_keyword.py](code_01_naive_keyword.py)

> 对应 s00 章 "什么是 RAG" 中的核心直觉：让 LLM "开卷考试"，先得有个"卷"。

### 概念

最朴素的检索策略：

1. 把文档读成段落列表；
2. 把用户问题拆词；
3. 找第一个段落里有任意一个词的；
4. 把那段返回。

30 行代码，零外部依赖。

入口：[`code_01_naive_keyword.py`](code_01_naive_keyword.py)

### 跑一遍

```bash
python s01_what_is_rag/code_01_naive_keyword.py
```

输入对照（用 `samples/disclosure.docx` 实测）：

| 输入 | 输出 |
|---|---|
| `披露` | `相关信息披露详见财务报表附注三(二十五)、五 (二)1 及十五(二)。` |
| `外星人` | `I don't know.` |

### 看输出

**01 跑出来（实测，`samples/disclosure.docx`）：**

```
[query] 披露
[hit]  相关信息披露详见财务报表附注三(二十五)、五 (二)1 及十五(二)。

[query] 外星人
[hit]  I don't know.
```

"披露"命中说明子串匹配能召回字面相关的段落；"外星人"返回 `I don't know.` 说明零命中时不让 LLM 编——为后续章节埋下"拒答"的种子。

### 局限与下一步（这就是后面章节要解决的）

本段做对了什么 — 用 30 行 python-docx + 子串匹配跑通 RAG 闭环最小形状，零外部依赖。

- **找不到同义词**——问"营收"找不到"营业收入"。
- **找到关键词不一定是答案**——段落里出现"应收账款"，但讲的是会计科目列表，不是用户想问的"如何计提坏账"。
- **没有评分**——第一个命中就返回，多个相关时不能排序。

这两个问题分别对应 RAG 系统的两大难题：

- **召回（recall)** → s04 embedding + s06 混合检索
- **精排（precision)** → s07 rerank + s08 prompt

- **`ModuleNotFoundError: No module named 'docx'`**：`pip install python-docx`。
- 永远返回 `I don't know.`：`DISCLOSURE.docx` 路径错了 / 段落切得太碎；print 一段 paragraphs[0] 验加载。

下一章 s02 如何解决 — 把段落级文档加载抽象成统一 schema `{text, page, source}`,把 python-docx 这一类文件读取切干净,s03+ 才能在不感知文件类型的前提下做切块。

---

## 三、词袋向量 + 余弦相似度：[code_02_vector_basics.py](code_02_vector_basics.py)

> 词袋（bag-of-2-grams)+ 手写余弦，省去 embedding 模型下载，让 s01 自包含。
> 后面 s04 用 BGE 真向量替代这套玩具；s05 用 Chroma 持久化索引。

### 概念

1. 把每段切成 2-gram（中文每 2 字 1 个 token）；
2. 全部 token 组成词表 `vocab: {token: index}`；
3. 每段转成词频向量 `vec = [词频 in vocab]`；
4. 问题转同样形状的向量；
5. 余弦相似度 = "问题向量" 与 "段落向量" 的夹角；
6. 按分排序返回 Top-3。

入口：[`code_02_vector_basics.py`](code_02_vector_basics.py)

### 跑一遍

```bash
python s01_what_is_rag/code_02_vector_basics.py
# 交互:输入查询(如"披露")
```

输出示例（按相似度分排序的前 3 段）：

```
Top-3 与你的问题最相关的段落(按向量余弦排序):
[1] score=0.342
    相关信息披露详见财务报表附注三(二十五)、五 (二)1 及十五(二)...
[2] score=0.215
    ...
```

### 看输出

**02 跑出来（实测，`samples/disclosure.docx`，交互输入"披露"）：**

```
[vocab] 共 N 个 2-gram token
[query] 披露
Top-3 与你的问题最相关的段落(按向量余弦排序):
[1] score=0.342
    相关信息披露详见财务报表附注三(二十五)、五 (二)1 及十五(二)...
[2] score=0.215
    ...
```

**跟 01 的差别**：01 只返第一段（无评分），02 返 Top-3 + 余弦分（有排序）。`[1]` 永远是分数最高那一段，即便它是"披露"字面量最密集的那段也不一定是真答案——这正是 03 + s07 要解决的问题。

### 局限与下一步（这就是后面章节要解决的）

本段做对了什么 — 用 2-gram 词袋 + 手写余弦把"任意段落相对查询的可量化相关度"算出来了,并按 Top-3 排序,无 NumPy 依赖,让 RAG 检索这一步有"分"。

- **词袋维度爆炸**——每段可能 100+ unique token，sparse；不像 BGE 是 dense 512 维真语义向量。
- **丢位置信息**——"披露在第 1 句"和"披露在第 3 句"对词袋向量没差别。
- **丢上下文**——"摘要式披露"和"详细披露"在词袋层分不开。
- **没真语义**——"营收"和"营业收入"在字面上无关，词袋给 0。

生产里要解决就是 s04（真语义 embedding)+ s07(cross-encoder 精排）。

- **`EOFError when piped`**：02 的 `input("问点啥: ")` 在 `< /dev/null` 下抛 EOFError——交互模式是主用方式；想脚本化跑就直接改 `main()` 里的 `q = ...`。
- 词表过大 / 内存炸：把 2-gram 换成 3-gram 词汇量指数级增长；demo 阶段保持 2-gram。
- top-1 不是真答案：词袋的本质问题；想"营收"和"营业收入"互通，等 s04 BGE embedding。

下一章 s02 如何解决 — 把词汇量从"每段 unique token 100+"压下来,先做 chunking 把段内稀疏维拆成局部稠密的小块,再交给 s04 embedding。这一步不解决词袋 vs dense 的本质对立,但把"分块"做了,s04 才有合适的输入颗粒度。

---

## 四、完整 RAG 链路：检索 + Prompt + LLM：[code_03_augmented_llm.py](code_03_augmented_llm.py)

> 这一章是"s01 → RAG 全链路"的最小闭环；s02-s08 把每一环换成真工业实现。

### 概念

三段代码：

- `retrieve(q, paragraphs, k=3)`——把 2 的向量检索原样搬过来。
- `build_prompt(question, hits)`——把 hits 渲染成 `[1] ... [2] ... [3] ...`，包进 `<context>` 标签。
- `call_llm(prompt)`——调 OpenAI 兼容的 `/chat/completions`；缺 API key 时直接跳过。

完整流程：

```
用户问题 ─▶ retrieve (02 词袋向量) ─▶ top-3 hits
                                               │
                                               ▼
                                       build_prompt
                                               │
                                               ▼
                                       LLM.generate
                                               │
                                               ▼
                                           答案
```

入口：[`code_03_augmented_llm.py`](code_03_augmented_llm.py)

### 跑一遍

```bash
# .env 里有 LLM_API_KEY 就走端到端;无 key 时 graceful-skip 只打印 prompt
python s01_what_is_rag/code_03_augmented_llm.py
```

可选：自定义 base / model（在 `.env` 里设 `LLM_BASE` / `LLM_MODEL` 即可，code 顶部 `load_dotenv(override=True)` 会读到）。

无 key 输出示例：

```
[retrieve] 召回 3 段
  [1] 相关信息披露详见财务报表附注三(二十五)...
  [2] ...

[prompt]
你只能依据 <context> 标签内的资料回答问题;
若资料不足以回答,请回复「我不知道」。

<context>
[1] ...
[2] ...
[3] ...
</context>

问题: 关联方披露
回答:

[llm] LLM_API_KEY 未设置,跳过真实生成...
```

### 看输出

**03 跑出来（实测，无 key）：**

```
[retrieve] 召回 3 段
  [1] 相关信息披露详见财务报表附注三(二十五)...
  [2] ...

[prompt]
你只能依据 <context> 标签内的资料回答问题;
若资料不足以回答,请回复「我不知道」。

<context>
[1] ...
[2] ...
[3] ...
</context>

问题: 关联方披露
回答:

[llm] LLM_API_KEY 未设置,跳过真实生成...
```

有 key 时 `call_llm` 会真发请求，LLM 输出接在 `回答:` 后；无 key 时只打印 prompt 形状，让你确认"检索 + 拼 prompt"链路正确但 LLM 步骤被优雅跳过。

### 局限与下一步（这就是后面章节要解决的）

本段做对了什么 — 用 60 行内代码把 `retrieve → build_prompt → call_llm` 这条 RAG 三动词闭环跑通,prompt 里硬约束"资料外回答「我不知道」"+`<context>` 边界,把第三章教学 demo 的 hallucination 风险压在 prompt 工程可达的范围内。

- **极简 prompt 模板**——RAGFlow 的 `rag/prompts/generator.py` 维护多语言多场景 prompt，带 `<|COMPLETE|>` 哨兵和明确的"回答字数限制"等。本章对应其中"纯检索 + 纯生成"分支。
- **没有 rerank**——top-3 不一定最相关；s07 会补 cross-encoder。
- **没有 hybrid 召回**——本章只有词袋向量；RAGFlow 走 `weighted_sum(BM25, vector)`（详见 `docs/reference/ragflow-notes/hybrid_retrieval.md`）。
- **无引用检测**——如果 LLM 答了一段不在 `<context>` 里的话（"按惯例审计费用通常为 50 万元"），本章只靠 prompt 约束。生产里通常还要在输出侧用字符串匹配 / LLM-as-judge 检测"未引用"段。

- **`LLM_API_KEY 未设置`**：03 是预期行为——只打印 prompt，验证检索 + 拼 prompt 链路正确。设置 `LLM_API_KEY` 后才会真调 LLM。
- **`LLM_BASE_URL` 报错 401 / 404**：检查 `.env` 或环境变量里的 base / model 是否跟所用服务匹配（OpenAI / DeepSeek / 智谱 / Anthropic / 自部署 vLLM）。
- LLM 编数字：prompt 约束可能没生效；把"回答「我不知道」"提到 prompt 最开头，让 system 强化这条优先级。
- 想离线跑：无 key 时只打印 prompt——这就是"教学 demo 的兜底形状"，不动代码也能验链路。

下一章 s02 如何解决 — 进入工业实现第一站:把"文档加载"从 python-docx 单类型扩到 PDF + DOCX,产出 `{text, page, source}` 统一 schema,让 s03 chunking 不再关心文件类型分支。

---

## 五、核心函数一览

| 函数 | 文件 | 输入 | 输出 | 一句话解释 |
|---|---|---|---|---|
| `retrieve(q, paragraphs)` | `code_01_naive_keyword.py` | 问题、段落列表 | 第一个命中段落 / `"I don't know."` | 子串匹配第一段 |
| `vocab_for(paragraphs)` | `code_02_vector_basics.py` | 段落列表 | `{token: index}` | 2-gram 词表 |
| `cosine(a, b)` | `code_02_vector_basics.py` | 两个等长 list[float] | float ∈ [0, 1] | 手写余弦(避免 NumPy 依赖) |
| `retrieve(q, paragraphs, k)` | `code_02_vector_basics.py` | 问题、段落列表、k | top-k 段落 | 词袋向量 top-k |
| `retrieve(q, paragraphs, k)` | `code_03_augmented_llm.py` | 问题、段落列表、k | top-k 段落 | 同 02 |
| `build_prompt(question, hits)` | `code_03_augmented_llm.py` | 问题、top-k 段落 | 拼好的 prompt 字符串 | `<context>...</context>` 包裹 |
| `call_llm(prompt)` | `code_03_augmented_llm.py` | prompt 字符串 | LLM 返回字符串 | OpenAI 兼容 `/chat/completions`;缺 key 时跳过 |
| `main()` (01) | `code_01_naive_keyword.py` | — | 段落 + 查询输出 | 01 入口 |
| `main()` (02) | `code_02_vector_basics.py` | 交互输入查询 | top-3 + 分 | 02 入口 |
| `main()` (03) | `code_03_augmented_llm.py` | — | prompt + LLM 输出 | 03 入口 |

## 六、跨代码协同

环境变量：03 需要 `LLM_API_KEY`（可选 `LLM_BASE` / `LLM_MODEL`，指向任意 OpenAI 兼容服务）。无 key 时跳过真实生成，只打印 prompt 验证链路。

三个 code 文件约定同一份 schema：`paragraphs = list[str]` 输入 → `retrieve` 返回 `list[{text, score, ...}]` 命中；02 / 03 的 `retrieve(q, paragraphs, k)` 签名完全一致——这是把"召回"封装掉的代价：**调用方不需要知道 01 是子串匹配、02 是词袋向量、03 又调一遍 02**，统一接口降低后续章节替换成本。

## RAGFlow 实现

RAGFlow 把 RAG 主干实现成 12 个独立模块——解析、切块、embedding、索引、召回、重排、prompt、生成 各自可替换。本教程的 12 章地图见 [s00 §1.4](../s00_concepts/README.md#一4-12-章对照)；RAGFlow 在每一层的工程化做法见 s02-s12 各章末的 RAGFlow 实现小节。s01 跑的是这条主干的最朴素直连版——`query → retrieve → augment → generate`，RAGFlow 的模块化设计正好对应这条主干的"每一环可独立替换"。

| s01 里的环节 | 工业实现 | 教程章节 |
|---|---|---|
| `python-docx` 读段落 | `pypdf` / `python-docx` + `pdfplumber` + 多 Parser 调度 | s02 |
| 按段落切(不切) | 固定字符 cap + 句界切 / 父子块 / 表格感知 | s03 |
| 词袋 sparse 向量 | BGE dense 512 维真语义向量 | s04 |
| 内存 list | Chroma / Elasticsearch / Infinity 持久化索引 | s05 |
| cosine only | BM25 + dense `weighted_sum` 融合 | s06 |
| 无 | cross-encoder 重排序 + PageRank | s07 |
| 极简 `<context>` | 多语言模板 + 哨兵 + 角标 | s08 |
| OpenAI 兼容 | OpenAI / DeepSeek / 智谱 / Anthropic / Bedrock / Ollama | s08 |

工业版 vs s01 的对照：

| 步骤 | s01 | RAGFlow 真实实现 | 教程章节 |
|---|---|---|---|
| 文档解析 | `python-docx` | `deepdoc/parser/{pdf,docx}.py` | s02 |
| 切块 | 按段落 | `naive_merge` token-aware + `hierarchical_merge` | s03 |
| Embedding | 词袋 sparse 2-gram | BGE small-zh dense 512 | s04 |
| 索引 | 内存 list | Chroma / Infinity / Elasticsearch | s05 |
| 召回 | cosine only | BM25 + 向量 `weighted_sum` | s06 |
| 精排 | 无 | cross-encoder 重排序 + PageRank | s07 |
| Prompt | 极简 `<context>` | 多语言模板 + 哨兵 + 角标 | s08 |
| LLM | OpenAI 兼容 | Anthropic / OpenAI / Bedrock / Ollama | s08 |

---

## 选型速记

### 主流 RAG 范式速览

下面这张表把 RAG 系统按"流程长度 / 检索深度 / 是否可解释 / 适用场景"列出来：

| 范式 | 检索流程 | 检索深度 | 可解释 | 适用场景 |
|---|---|---|---|---|
| **Naive RAG(本章 MVP)** | query → top-k → LLM | 1 段 | 弱(无打分详情) | 教学 / 演示 |
| **Advanced RAG**(s04-s08) | query → embed → BM25+dense → rerank → LLM | 2 段 | 中 | 中小规模生产 |
| **Modular RAG**(s09/s10) | query → Agent 路由 → 多模态 → LLM | N 跳 | 强 | 多源 / 多跳 |

我们的 toy `retrieve + build_prompt + call_llm` 在复杂度上只占第一行——**Naive RAG**；RAGFlow 走完整 Modular，**多一道抽象就多一道观测点 + 一个失败模式**。教学 demo 选 MVP 因为它跑通快、依赖少；**生产请按"语料规模 / 检索质量 / 可观测性"做 tier 选型**(Naive → Advanced → Modular）。

- **教学 / 玩具 / 文档 < 100 段** → 本章 MVP（子串 / 词袋 + cosine），零依赖、能跑通
- **中小规模生产 / 文档 100-100k 段** → s04 BGE + s06 BM25+dense 融合 + s07 rerank + s08 prompt 模板
- **检索质量敏感 / 命中率优先** → 高级 RAG 叠加：HyDE 查询重写 + 多路召回 + cross-encoder
- **复杂多源 / 跨系统 / 多跳推理** → 模块化 RAG：s09 Agent + s10 GraphRAG + 工具调用
- **要想清楚 toy 跟生产的边界** → 用本章 02 把"词袋 vs BGE"、03 把"无 rerank vs 有 rerank"各跑一次，对比输出

### 扩展指南

加一层 RAG 能力（换 LLM / 加 rerank / 加 hybrid）只要三步：

1. 在 `code_03_augmented_llm.py` 的 `build_prompt(hits, question)` 之后插入一个 `rerank(hits, question)` 钩子，s07 的 `rerank(query, hits, top_k=3)` 直接接进来，`hits` 还是 `[(text, source, page)]` 的统一形状，LLM 看到的就是前 3 条精排后的 context；
2. 把 03 的 `call_llm(question, prompt)` 里的 `model=` 参数抽出来，从环境变量 `LLM_MODEL` 读（默认 `gpt-4o-mini`），切 Claude / 本地 vLLM / Qwen 都只改 env 不改代码；
3. 把 02 的 `vocab + tfidf` 子串匹配换成 `embed_local(query)` 返回的 512 维向量，余弦检索回 `chunks_emb` 矩阵，s04 的 BGE 把它接进来，**接口形状留好了，只换实现**。

不要把"加 rerank / 换 LLM / 加 hybrid" 写在 `retrieve()` 里——它只懂子串，加 hybrid 会污染单一职责。`retrieve` 只懂 toy，**main() 懂全 RAG 模式**(MVP → +rerank → +hybrid → s12 完整 Modular）。本章只跑 MVP，但接口形状留好了。

---

## 思考题

1. **怎么把 01 的子串匹配改成 Top-3 候选段？最简单的打分怎么算？**
2. **如果两段都包含"披露"两次，词袋向量会怎么算？它分得开"摘要式披露"和"详细披露"吗？**
3. **如果 LLM 答了一段不在 `<context>` 里的话（比如"按惯例审计费用通常为 50 万元"），怎么从工程上防住？**

（答案见文末「思考题答案」）

---

## 思考题答案

### Q1. 怎么把子串匹配改成 Top-3 候选段？最简单的打分怎么算？

**最简单的版本：数命中的关键词数量。**

遍历段落，对每个段落计 `score = sum(1 for w in question.split() if w.lower() in p.lower())`，按分数排序，取前 3 个非零段落返回。如果全 0，仍然返回 `"I don't know."`。

**为什么这是"向量检索"的原始形态？**

朴素子串打分有两个根本问题：

1. **词不匹配**——"营收"和"主营业务收入"在字面上无关，但语义上强相关。子串打分给 0，向量相似度会给高分。
2. **字面命中 ≠ 语义相关**——一段提到"应收账款"的列表，关键词命中很多，但不是答案。向量相似度会被"语义方向"压低匹配分。

接下来的章节里，**关键词命中次数 → BM25 → 向量相似度 → Cross-Encoder 重排序**，可以看作"打分函数"的一次次升级。从本章的 toy 起步，逐章替换打分方式，直到能稳定选出 top-k 段落再喂给 LLM。

### Q2. 如果两段都包含"披露"两次，词袋向量会怎么算？它分得开"摘要式披露"和"详细披露"吗？

词袋向量给两段的"披露"维度的值都是 2——分不开。

**词袋丢了两类信息**：

1. **位置信息**——"披露"在段首还是段尾、在哪个句子，对词袋向量没差别。
2. **上下文信息**——"摘要式披露"和"详细披露"在词袋层都是"披露" + 一些别的词的组合，但组合方式（句法结构、上下文关联词）完全丢了。

生产里要解决：

- **真语义 embedding**(s04)——BGE 把整段文本压成 512 维 dense 向量，语义相近的段在向量空间里距离近，"摘要式披露"和"详细披露"的向量会自然分开。
- **cross-encoder 精排**(s07)——把 query 和每段一起喂进 transformer，让模型看到"披露"在上下文里扮演什么角色。

### Q3. 如果 LLM 答了一段不在 `<context>` 里的话（比如"按惯例审计费用通常为 50 万元")，怎么从工程上防住？

三道防线：

1. **Prompt 硬约束**——`build_prompt` 里加"若不在 <context> 内，回答「我不知道」"（本章已加）。这是最弱的一道防线，LLM 在压力下仍可能编故事。
2. **输出侧引用检测**——生成完用字符串匹配 / LLM-as-judge 扫一遍答案里每个事实句，要求每句话末尾贴引用 `[i]`。没有引用的句子标红或丢弃。RAGFlow 的 `_draw_highlight` + `chunk_id` 关联就是这套。
3. **答案渲染层**——UI 渲染时强制每句话末尾贴引用 `[i]`，没有引用的句子标红、不给用户看。这层不在引擎范围，是前端的事。

本章只做了第 1 层；s08 会做第 2 层（工业 prompt 模板 + 拒答检测）；第 3 层是 UI 层的事，不在引擎范围。
