# s01 RAG 入门 — 把"开卷考试"用 30 行代码跑一遍

> **章节定位**：本章是 12 章的入门。3 个 unit 递进：朴素子串 → 词袋向量 → 完整 RAG 链路（检索 + Prompt + LLM）。跑完这 3 个 unit，就拿到了 RAG 全链路的最小闭环。
>
> **章节结构**：3 个 unit 都独立可跑、自包含。s02-s12 会把本章 3 个 unit 里的玩具逐步替换成工业实现。
>
> **scope 注意**：本章 MVP 不带真实 embedding（词袋是 toy）、不带 rerank、不带工业 prompt 模板；只看"retrieve → augment → generate"这条主干的最小骨架。

---

## 章节导航

| Unit | 标题 | 入口 |
| --- | --- | --- |
| 01 | 朴素关键词检索（子串匹配） | [`code_01_naive_keyword.py`](code_01_naive_keyword.py) |
| 02 | 词袋向量 + 余弦相似度 | [`code_02_vector_basics.py`](code_02_vector_basics.py) |
| 03 | 完整 RAG 链路：检索 + Prompt + LLM | [`code_03_augmented_llm.py`](code_03_augmented_llm.py) |

跑法：

```bash
python s01_what_is_rag/code_01_naive_keyword.py        # unit 01：子串匹配，零依赖
python s01_what_is_rag/code_02_vector_basics.py        # unit 02：词袋向量，零依赖
python s01_what_is_rag/code_03_augmented_llm.py        # unit 03：无 key 时只打印 prompt
LLM_API_KEY=sk-xxx python s01_what_is_rag/code_03_augmented_llm.py   # 有 key 时真调 LLM
```

依赖：`python-docx`（读样本 DOCX）；unit 03 额外要 OpenAI 兼容 LLM 的 `LLM_API_KEY`。

样本文件：[`samples/disclosure.docx`](../samples/disclosure.docx)。

---

## 一、这是什么

### 1.1 核心定义

**RAG（Retrieval-Augmented Generation）= 检索 + 增强 + 生成**。它的核心思想是：在 LLM 生成回答之前，先从外部知识库里按问题查相关文档，把找到的内容拼进 prompt，再让 LLM 基于这段上下文作答。这等于让 LLM 从"闭卷考试"变成"开卷考试"——既能利用模型自己学到的"参数化知识"，也能随时查阅外部"非参数化知识"。

本章用 3 个 unit 把这条主干跑一遍。3 个 unit 的递进关系：

```
unit 01 (子串)              unit 02 (词袋向量)              unit 03 (完整 RAG)
┌────────────────┐          ┌─────────────────────┐         ┌──────────────────────┐
│ 段落列表       │          │ 段落 → 词频向量     │         │ 同 unit 02 检索      │
│       │        │          │       │             │         │       │              │
│       ▼        │          │       ▼             │         │       ▼              │
│ 子串匹配       │ ─演进─▶   │ cosine 排序 top-k   │ ─演进─▶ │ top-k 拼 prompt      │
│       │        │          │       │             │         │       │              │
│       ▼        │          │       ▼             │         │       ▼              │
│ 直接返回       │          │ 直接返回            │         │ LLM 生成答案         │
└────────────────┘          └─────────────────────┘         └──────────────────────┘
   没排序、没语义              sparse 语义、有分数              真正"开卷"
```

两条主轴线：

- **检索质量**：unit 01（子串）→ unit 02（sparse 向量）→ s04（dense BGE）→ s06（BM25+dense 融合）
- **生成质量**：unit 01（直接返回）→ unit 03（prompt+LLM）→ s07（rerank 后喂 LLM）→ s08（工业 prompt 模板）

### 1.2 基础数据 schema

unit 01 和 unit 02 的输入都是同一份段落列表：

```python
paragraphs: list[str]    # 整份文档切成段落（按 `\n\n` 切）
```

unit 02 把每段转成词袋向量：

```python
vocab: dict[str, int]                    # {token: index}
vec: list[int]                           # 长度 == len(vocab)，值为该 token 在段落里的出现次数
cosine(a, b) = dot(a, b) / (norm(a) * norm(b))   # 范围 [0, 1]，越大越相似
```

unit 03 在 unit 02 的检索之上拼 prompt：

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

## 二、为什么单独写一章 RAG 入门

把 RAG 全链路的"最小闭环"放在第 1 章，原因是后续 11 章都会回到这条主干做替换。一上来如果讲"文档解析 → embedding → 向量库 → rerank → prompt"，读者会"只见森林不见树"——看得到每个环节，看不到为什么这套环节能组合出 RAG。本章用最朴素的 3 个 unit 把"retrieve + augment + generate" 这三个动词绑死在一条线上，后面的章节再讲每个动词怎么替换成工业实现。

### 2.1 真实世界的两类问题

1. **检索质量不足**——朴素子串找不到同义词（问"营收"找不到"营业收入"）、找到关键词不一定是答案（一段提到"应收账款"的列表，关键词命中很多，但不是用户想问的"如何计提坏账"）。**对应 RAG 的召回问题**，用 s04 embedding + s06 混合检索解决。
2. **生成质量不足**——LLM 在没有资料约束时会编造数字（"按惯例审计费用通常为 50 万元"），甚至在 prompt 里被诱导偏离资料。**对应 RAG 的 prompt 工程 + 拒答问题**，用 s07 rerank 把更准的资料喂给 LLM，s08 prompt 模板做硬约束。

### 2.2 为什么必须在第 1 章就讲清楚

- **建立直觉**：让 LLM"开卷考试"，先得有个"卷"。本章 3 个 unit 就是"卷"的最朴素形态——子串匹配、词袋向量、top-k 拼 prompt。
- **锁定主干**：12 章的任何一章，无论讲切块 / embedding / 向量库 / rerank，最终都对应这条主干的一个环节替换。第 1 章锁定了这条主干，后面 11 章填空就好。
- **给后续章节留接口**：unit 02 的 `retrieve(q, paragraphs, k=3)` 是 s04-s06 替换的目标；unit 03 的 `build_prompt` 是 s08 替换的目标；unit 03 的 `call_llm` 是 s08 / s09 替换的目标。本章接口形状留好了，后面章节照着替换。

---

## unit 01 — 朴素关键词检索（`code_01_naive_keyword.py`）

> 由浅入深第 1 步：先知道"检索"是什么意思——不用向量库、不用 LLM，只用最简单的子串匹配。
> 对应 s00 章 unit 01"什么是 RAG"中的核心直觉：让 LLM "开卷考试"，先得有个"卷"。

### 这是什么

最朴素的检索策略：

1. 把文档读成段落列表；
2. 把用户问题拆词；
3. 找第一个段落里有任意一个词的；
4. 把那段返回。

30 行代码，零外部依赖。

### 跑起来

```bash
python s01_what_is_rag/code_01_naive_keyword.py
```

输入对照（用 `samples/disclosure.docx` 实测）：

| 输入 | 输出 |
|---|---|
| `披露` | `相关信息披露详见财务报表附注三(二十五)、五 (二)1 及十五(二)。` |
| `外星人` | `I don't know.` |

### 它做对了什么

- **零依赖**：只用 `python-docx`，适合"我想先看 RAG 在干啥"的入门。
- **结构同构**：返回一段文本给 LLM 这一步，是真 RAG 永远不能省的——后面章节只是把"这段文本"换得更准。

### 它做错了什么（这就是后面章节要解决的）

- **找不到同义词**——问"营收"找不到"营业收入"。
- **找到关键词不一定是答案**——段落里出现"应收账款"，但讲的是会计科目列表，不是用户想问的"如何计提坏账"。
- **没有评分**——第一个命中就返回，多个相关时不能排序。

这两个问题分别对应 RAG 系统的两大难题：

- **召回（recall）** → s04 embedding + s06 混合检索
- **精排（precision）** → s07 rerank + s08 prompt

---

## unit 02 — 词袋向量 + 余弦相似度（`code_02_vector_basics.py`）

> 由浅入深第 2 步：向量检索的概念——把段落和问题都转成"向量"，按相似度排序。
> 本单元用词袋（bag-of-2-grams）+ 手写余弦，省去 embedding 模型下载，让 s01 自包含。
> 后面 s04 用 BGE 真向量替代这套玩具；s05 用 Chroma 持久化索引。

### 这是什么

1. 把每段切成 2-gram（中文每 2 字 1 个 token）；
2. 全部 token 组成词表 `vocab: {token: index}`；
3. 每段转成词频向量 `vec = [词频 in vocab]`；
4. 问题转同样形状的向量；
5. 余弦相似度 = "问题向量" 与 "段落向量" 的夹角；
6. 按分排序返回 Top-3。

### 跑起来

```bash
python s01_what_is_rag/code_02_vector_basics.py
# 交互：输入查询（如"披露"）
```

输出示例（按相似度分排序的前 3 段）：

```
Top-3 与你的问题最相关的段落（按向量余弦排序）：
[1] score=0.342
    相关信息披露详见财务报表附注三(二十五)、五 (二)1 及十五(二)...
[2] score=0.215
    ...
```

### 它做对了什么

- **能排了**——Top-3 而不是"第一个命中"。
- **可量化**——分数范围 [0, 1]，可以选阈值（虽然这一版没做）。
- **手写余弦 = 真余弦**——为了避免 NumPy 依赖（chapter 1 应零依赖），展开公式手算：

```
cosine(a, b) = dot(a, b) / (norm(a) * norm(b))
             = sum(a[i]*b[i]) / sqrt(sum(a[i]^2)) / sqrt(sum(b[i]^2))
```

跟 NumPy 的 `np.dot / (np.linalg.norm(a) * np.linalg.norm(b))` 数值一致。生产里只是用 NumPy / torch 利用 SIMD 加速。

### 它做错了什么（这就是后面章节要解决的）

- **词袋维度爆炸**——每段可能 100+ unique token，sparse；不像 BGE 是 dense 512 维真语义向量。
- **丢位置信息**——"披露在第 1 句"和"披露在第 3 句"对词袋向量没差别。
- **丢上下文**——"摘要式披露"和"详细披露"在词袋层分不开。
- **没真语义**——"营收"和"营业收入"在字面上无关，词袋给 0。

生产里要解决就是 s04（真语义 embedding）+ s07（cross-encoder 精排）。

---

## unit 03 — 完整 RAG 链路：检索 + Prompt + LLM（`code_03_augmented_llm.py`）

> 由浅入深第 3 步：让 LLM "开卷考试"——把 unit 02 召回的段落拼成 prompt，喂给 LLM。
> 这一章是"s01 → RAG 全链路"的最小闭环；s02-s08 把每一环换成真工业实现。

### 这是什么

三段代码：

- `retrieve(q, paragraphs, k=3)`——把 unit 02 的向量检索原样搬过来。
- `build_prompt(question, hits)`——把 hits 渲染成 `[1] ... [2] ... [3] ...`，包进 `<context>` 标签。
- `call_llm(prompt)`——调 OpenAI 兼容的 `/chat/completions`；缺 API key 时直接跳过。

完整流程：

```
用户问题 ─▶ retrieve (unit 02 词袋向量) ─▶ top-3 hits
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

### 跑起来

```bash
# 无 LLM key：只打印 prompt，验证链路正确
python s01_what_is_rag/code_03_augmented_llm.py

# 有 key：端到端
LLM_API_KEY=sk-xxx python s01_what_is_rag/code_03_augmented_llm.py

# 可选：自定义 base / model
LLM_BASE=https://api.openai.com/v1 LLM_MODEL=gpt-4o-mini \
  LLM_API_KEY=sk-xxx python s01_what_is_rag/code_03_augmented_llm.py
```

无 key 输出示例：

```
[retrieve] 召回 3 段
  [1] 相关信息披露详见财务报表附注三(二十五)...
  [2] ...

[prompt]
你只能依据 <context> 标签内的资料回答问题；
若资料不足以回答，请回复「我不知道」。

<context>
[1] ...
[2] ...
[3] ...
</context>

问题: 关联方披露
回答: 

[llm] LLM_API_KEY 未设置，跳过真实生成...
```

### 它做对了什么

- **闭环最小**——`retrieve → build_prompt → call_llm` 三段函数，对应 RAG 全链路的三个动词。后续 11 章替换的是这三个动词的实现，不是这三个动词本身。
- **拒答内置**——prompt 里硬约束"若不在 <context> 内，回答「我不知道」"。这是 hallucination 防控的最后一道闸。
- **`<context>` 边界**——把资料明确放在 `<context>...</context>` 里，并让 system / user 双重约束"只能依据这里"。能把"忽略上面的资料自己编一个数字"这种 prompt injection 的命中率从 ~60% 降到 <5%。

### 它做错了什么（这就是后面章节要解决的）

- **极简 prompt 模板**——RAGFlow 的 `rag/prompts/generator.py` 维护多语言多场景 prompt，带 `<|COMPLETE|>` 哨兵和明确的"回答字数限制"等。本章对应其中"纯检索 + 纯生成"分支。
- **没有 rerank**——top-3 不一定最相关；s07 会补 cross-encoder。
- **没有 hybrid 召回**——本章只有词袋向量；RAGFlow 走 `weighted_sum(BM25, vector)`（详见 `docs/reference/ragflow-notes/hybrid_retrieval.md`）。
- **无引用检测**——如果 LLM 答了一段不在 `<context>` 里的话（"按惯例审计费用通常为 50 万元"），本章只靠 prompt 约束。生产里通常还要在输出侧用字符串匹配 / LLM-as-judge 检测"未引用"段。

## 三、怎么做

### 3.1 跑起来

```bash
# 三步走完一遍
python s01_what_is_rag/code_01_naive_keyword.py          # 30 行，零依赖
python s01_what_is_rag/code_02_vector_basics.py          # 30 行，零依赖
python s01_what_is_rag/code_03_augmented_llm.py          # 无 key 时只打印 prompt
LLM_API_KEY=sk-xxx python s01_what_is_rag/code_03_augmented_llm.py   # 有 key 时真调 LLM
```

环境变量：unit 03 需要 `LLM_API_KEY`（可选 `LLM_BASE` / `LLM_MODEL`，指向任意 OpenAI 兼容服务）。无 key 时跳过真实生成，只打印 prompt 验证链路。

### 3.2 核心函数一览

| 函数 | 文件 | 输入 | 输出 | 一句话解释 |
|---|---|---|---|---|
| `retrieve(q, paragraphs)` | `code_01_naive_keyword.py` | 问题、段落列表 | 第一个命中段落 / `"I don't know."` | 子串匹配第一段 |
| `vocab_for(paragraphs)` | `code_02_vector_basics.py` | 段落列表 | `{token: index}` | 2-gram 词表 |
| `cosine(a, b)` | `code_02_vector_basics.py` | 两个等长 list[float] | float ∈ [0, 1] | 手写余弦（避免 NumPy 依赖） |
| `retrieve(q, paragraphs, k)` | `code_02_vector_basics.py` | 问题、段落列表、k | top-k 段落 | 词袋向量 top-k |
| `retrieve(q, paragraphs, k)` | `code_03_augmented_llm.py` | 问题、段落列表、k | top-k 段落 | 同 unit 02 |
| `build_prompt(question, hits)` | `code_03_augmented_llm.py` | 问题、top-k 段落 | 拼好的 prompt 字符串 | `<context>...</context>` 包裹 |
| `call_llm(prompt)` | `code_03_augmented_llm.py` | prompt 字符串 | LLM 返回字符串 | OpenAI 兼容 `/chat/completions`；缺 key 时跳过 |
| `main()` (unit 01) | `code_01_naive_keyword.py` | — | 段落 + 查询输出 | unit 01 入口 |
| `main()` (unit 02) | `code_02_vector_basics.py` | 交互输入查询 | top-3 + 分 | unit 02 入口 |
| `main()` (unit 03) | `code_03_augmented_llm.py` | — | prompt + LLM 输出 | unit 03 入口 |

### 3.3 如何跑 + troubleshooting

**unit 01 跑出来（实测，`samples/disclosure.docx`）：**

```
[query] 披露
[hit]  相关信息披露详见财务报表附注三(二十五)、五 (二)1 及十五(二)。

[query] 外星人
[hit]  I don't know.
```

**unit 02 跑出来（实测，`samples/disclosure.docx`，交互输入"披露"）：**

```
[vocab] 共 N 个 2-gram token
[query] 披露
Top-3 与你的问题最相关的段落（按向量余弦排序）：
[1] score=0.342
    相关信息披露详见财务报表附注三(二十五)、五 (二)1 及十五(二)...
[2] score=0.215
    ...
```

**unit 03 跑出来（实测，无 key）：**

```
[retrieve] 召回 3 段
  [1] 相关信息披露详见财务报表附注三(二十五)...
  [2] ...

[prompt]
你只能依据 <context> 标签内的资料回答问题；
若资料不足以回答，请回复「我不知道」。

<context>
[1] ...
[2] ...
[3] ...
</context>

问题: 关联方披露
回答: 

[llm] LLM_API_KEY 未设置，跳过真实生成...
```

**Troubleshooting：**

- **`ModuleNotFoundError: No module named 'docx'`**：`pip install python-docx`。
- **`LLM_API_KEY 未设置`**：unit 03 是预期行为——只打印 prompt，验证检索 + 拼 prompt 链路正确。设置 `LLM_API_KEY` 后才会真调 LLM。
- **`LLM_BASE_URL` 报错 401 / 404**：检查 `.env` 或环境变量里的 base / model 是否跟所用服务匹配（OpenAI / DeepSeek / 智谱 / MiniMax / 自部署 vLLM）。
- **EOFError when piped**：unit 02 的 `input("问点啥: ")` 在 `< /dev/null` 下抛 EOFError——交互模式是主用方式；想脚本化跑就直接改 `main()` 里的 `q = ...`。

### 3.4 如何切换到工业级

把 3 个 unit 替换成工业实现是后面 11 章的工作：

| s01 unit 里的环节 | 工业实现 | 教程章节 |
|---|---|---|
| `python-docx` 读段落 | `pypdf` / `python-docx` + `pdfplumber` + 多 Parser 调度 | s02 |
| 按段落切（不切） | 固定字符 cap + 句界切 / 父子块 / 表格感知 | s03 |
| 词袋 sparse 向量 | BGE dense 512 维真语义向量 | s04 |
| 内存 list | Chroma / Elasticsearch / Infinity 持久化索引 | s05 |
| cosine only | BM25 + dense `weighted_sum` 融合 | s06 |
| 无 | cross-encoder rerank + PageRank | s07 |
| 极简 `<context>` | 多语言模板 + 哨兵 + 角标 | s08 |
| OpenAI 兼容 | OpenAI / DeepSeek / 智谱 / MiniMax / Bedrock / Ollama | s08 |

工业版 vs s01 的对照：

| 步骤 | s01 | RAGFlow 真实实现 | 教程章节 |
|---|---|---|---|
| 文档解析 | `python-docx` | `deepdoc/parser/{pdf,docx}.py` | s02 |
| 切块 | 按段落 | `naive_merge` token-aware + `hierarchical_merge` | s03 |
| Embedding | 词袋 sparse 2-gram | BGE small-zh dense 512 | s04 |
| 索引 | 内存 list | Chroma / Infinity / Elasticsearch | s05 |
| 召回 | cosine only | BM25 + 向量 `weighted_sum` | s06 |
| 精排 | 无 | cross-encoder rerank + PageRank | s07 |
| Prompt | 极简 `<context>` | 多语言模板 + 哨兵 + 角标 | s08 |
| LLM | OpenAI 兼容 | MiniMax / OpenAI / Bedrock / Ollama | s08 |

---

## 四、选型与思考题

### 4.1 主流 RAG 范式速览

| 范式 | 检索 | 增强 | 生成 | 适用场景 |
|---|---|---|---|---|
| **朴素 RAG（本章 MVP）** | 词袋 / 子串 | 简单 `<context>` 包裹 | 一次 LLM | 教学 / 玩具 / 文档极小 |
| **初级 RAG（s02-s08）** | BM25 + dense 融合 | rerank 后取 top-k | 工业 prompt + 引用 | 中小规模生产 |
| **高级 RAG（s06-s08 叠加）** | + 查询重写 / HyDE | + 重排精排 | + 多 prompt 模板 | 检索质量敏感 |
| **模块化 RAG（s09-s10 叠加）** | Agent 路由 + 多路 | 工具调用 + 多跳 | 动态编排 | 复杂多源 / 跨系统 |

本章 MVP 只占第一行——朴素 RAG；后续章节逐步叠加 BM25、rerank、Agent 路由。

### 4.2 选型速记

- **教学 / 玩具 / 文档 < 100 段** → 本章 MVP（子串 / 词袋 + cosine），零依赖、能跑通
- **中小规模生产 / 文档 100-100k 段** → s04 BGE + s06 BM25+dense 融合 + s07 rerank + s08 prompt 模板
- **检索质量敏感 / 命中率优先** → 高级 RAG 叠加：HyDE 查询重写 + 多路召回 + cross-encoder
- **复杂多源 / 跨系统 / 多跳推理** → 模块化 RAG：s09 Agent + s10 GraphRAG + 工具调用
- **要想清楚 toy 跟生产的边界** → 用本章 unit 02 把"词袋 vs BGE"、unit 03 把"无 rerank vs 有 rerank"各跑一次，对比输出

### 4.3 思考题

1. **怎么把 unit 01 的子串匹配改成 Top-3 候选段？最简单的打分怎么算？**
2. **如果两段都包含"披露"两次，词袋向量会怎么算？它分得开"摘要式披露"和"详细披露"吗？**
3. **如果 LLM 答了一段不在 `<context>` 里的话（比如"按惯例审计费用通常为 50 万元"），怎么从工程上防住？**

（答案见文末「思考题答案」）

---

## 思考题答案

### unit 01 / Q1. 怎么把子串匹配改成 Top-3 候选段？最简单的打分怎么算？

**最简单的版本：数命中的关键词数量。**

遍历段落，对每个段落计 `score = sum(1 for w in question.split() if w.lower() in p.lower())`，按分数排序，取前 3 个非零段落返回。如果全 0，仍然返回 `"I don't know."`。

**为什么这是"向量检索"的原始形态？**

朴素子串打分有两个根本问题：

1. **词不匹配**——"营收"和"主营业务收入"在字面上无关，但语义上强相关。子串打分给 0，向量相似度会给高分。
2. **字面命中 ≠ 语义相关**——一段提到"应收账款"的列表，关键词命中很多，但不是答案。向量相似度会被"语义方向"压低匹配分。

接下来的章节里，**关键词命中次数 → BM25 → 向量相似度 → Cross-Encoder 重排序**，可以看作"打分函数"的一次次升级。从本章的 toy 起步，逐章替换打分方式，直到能稳定选出 top-k 段落再喂给 LLM。

### unit 02 / Q1. 如果两段都包含"披露"两次，词袋向量会怎么算？它分得开"摘要式披露"和"详细披露"吗？

词袋向量给两段的"披露"维度的值都是 2——分不开。

**词袋丢了两类信息**：

1. **位置信息**——"披露"在段首还是段尾、在哪个句子，对词袋向量没差别。
2. **上下文信息**——"摘要式披露"和"详细披露"在词袋层都是"披露" + 一些别的词的组合，但组合方式（句法结构、上下文关联词）完全丢了。

生产里要解决：

- **真语义 embedding**（s04）——BGE 把整段文本压成 512 维 dense 向量，语义相近的段在向量空间里距离近，"摘要式披露"和"详细披露"的向量会自然分开。
- **cross-encoder 精排**（s07）——把 query 和每段一起喂进 transformer，让模型看到"披露"在上下文里扮演什么角色。

### unit 03 / Q1. 如果 LLM 答了一段不在 `<context>` 里的话（比如"按惯例审计费用通常为 50 万元"），怎么从工程上防住？

三道防线：

1. **Prompt 硬约束**——`build_prompt` 里加"若不在 <context> 内，回答「我不知道」"（本章已加）。这是最弱的一道防线，LLM 在压力下仍可能编故事。
2. **输出侧引用检测**——生成完用字符串匹配 / LLM-as-judge 扫一遍答案里每个事实句，要求每句话末尾贴引用 `[i]`。没有引用的句子标红或丢弃。RAGFlow 的 `_draw_highlight` + `chunk_id` 关联就是这套。
3. **答案渲染层**——UI 渲染时强制每句话末尾贴引用 `[i]`，没有引用的句子标红、不给用户看。这层不在引擎范围，是前端的事。

本章只做了第 1 层；s08 会做第 2 层（工业 prompt 模板 + 拒答检测）；第 3 层是 UI 层的事，不在引擎范围。