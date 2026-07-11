# s03 文本分块 — 把段落切成"长度可控、语义相对完整"的小块

> **章节定位**：RAG 离线流水线的第二步——把 s02 吐出的"页 / 段"切成语义完整、长度可控、可被 embedding 索引的 `list[{text, chunk_id, page, source}]`。
>
> **章节结构**：2 个脚本。01 跑通 ≤ 500 字符 cap + 句界切的最小骨架；02 把同一套规则喂给真实样本，把表格 / 父子块 / 跨段引用三类失败摆出来。
>
> **scope 注意**：本章节围绕 *toy splitter* 这一层给出概念 / 问题 / MVP / 工业对照的完整弧线——不展开 LangChain `RecursiveCharacterTextSplitter` 的所有配置项，也不引入 SemanticChunker / Unstructured by_title（那些是工具选择，不是本章 MVP）。

---

## 章节导航

| 序号 | 标题 | 入口 |
| --- | --- | --- |
| 01 | 最小可跑分块：500 字符 cap + 句界切 | [`code_01_basic_chunk.py`](code_01_basic_chunk.py) |
| 02 | chunker 在真实样本上的三类失败 | [`code_02_chunk_failures.py`](code_02_chunk_failures.py) |

跑法：

```bash
python s03_chunking/code_01_basic_chunk.py       # 500 字符 cap + 句界切的 toy
python s03_chunking/code_02_chunk_failures.py    # 把 toy 喂给真实样本，看三类失败
```

依赖：纯 Python 标准库（无额外依赖；不引入 tiktoken / LangChain）。

样本输入：s02 的 loader 输出（`samples/server_whitepaper.pdf` + `samples/disclosure.docx`）。

---

## 一、章节介绍

`re.split(...)` 一行就能切，似乎不值得单写一章。但把这种 toy 切法扔到真实样本上，会冒出四类典型问题，每一类都对应一类工业解法。

### 1.1 核心定义

**文本分块（Text Chunking）** 是把 s02 的 `list[{text, page, source}]`（每段通常几百到上千字符）切成**更小、可索引的语义单元**——每个 chunk 通常带 `text`（块正文）、`chunk_id`（`{source}#{page}#p{n}`）、`source`、`page` 四个字段。它的上游是 s02 的 loader，下游是 embedding（s04）和向量索引（s05）。

```
    段 (页或 paragraph)                       chunks
    "二、关键特性\n计算密度..."                  ┌─ "二、关键特性\n计算密度..."(短段整段)
    [500+ 字符]                                 ├─ "...单台 2U..."(长段按句界切)
        │                                       └─ {text, chunk_id=source#page#pN, source, page}
        │  max_chars cap=500 + [。.!?] 句界正则
        ▼
    chunk_by_paragraph → split_long_paragraph
```

把它放进 RAG 全景看：**分块决定了检索的单位**。如果 chunk 切在句子中间，embedding 把残句算成完整语义；如果 chunk 跨多个主题，召回时会同时拉回一堆无关上下文；分块的质量直接决定了检索的信噪比。

### 1.2 三种典型边界

业界对"在哪里切"有三类策略，按简单到复杂排：

1. **固定大小切分**——按字符 / token 长度硬切（如每 500 字符一刀）。实现最简单，但最容易把句子拦腰截断；
2. **句子边界切分**——在 `[.。!?！？]` 之后切，保留每句完整；这是 MVP 的策略，平衡了"够小"和"语义完整"；
3. **版面 / 结构感知切分**——按 Markdown 标题、PDF 版面块、表格 parent-child 关系切；这是工业方案（如 RAGFlow）的做法。

### 1.3 真实世界的问题

1. **句子被切断**——固定 500 字符 cap 直接砍到中间："今天我们学习 RAG。它包括检索、生成两部分。" → "今天我们学习 RAG。它包括检索、生" + "成两部分"。Embedding 把残句算成完整语义，检索时匹配到一半句子就以为命中了；
2. **表格被切碎**——规格表 24LFF / 12LFF / 4SFF 横向排在一行，`pypdf.extract_text()` 抽出来是没换行的长串，500 字符 cap 在表格中间切断，每行 chunk 只看到一半字段；
3. **chunk 太小浪费 embedding 预算**——"第四节 分季度财务数据"这种 10 字符的节标题单独成 chunk。embedding 一次调用几毫秒，单 chunk 没语义也要花钱；
4. **chunk 太大稀释语义**——单 chunk 800 字符嵌入进 768 维向量，所有句子被平均成一个语义点，"内存"和"CPU"在向量空间里距离拉不开；
5. **跨段落引用断裂**——"见上表""收入结构如下" 这种指代词单独成 chunk 后，召回的是指代词本身，而不是它指向的实体。

### 1.4 为什么 500 字符是常见 cap

embedding 模型的 `max_seq_len` 通常是 512 token（BGE 系列）或 8192 token（Mistral / BGE-large）。500 中文字符 ≈ 1000-1500 token——**MVP 已经超 BGE 的硬上限**。我们故意保留这个"超限"作为教学锚点：s04 的 embedding 任务会演示"长 chunk 被截断 → 语义稀释"。工业代码应该用 tiktoken 算 token 数（典型 cap=128 token）而不是字符数。

### 1.5 与传统信息检索的对应

分块本质上是 **"给非结构化文本建立索引单位"**，跟传统搜索里的"段落 / 句子"切分同构——只是单位要适配 embedding 模型，而不是适配 BM25 的词袋。

每条都对应不同的工业级解法——版面识别、token-aware cap、parent-child 回填——这些是 s04+ 的扩展空间。**s03 的目标不是解决它们，而是把它们显式暴露出来，让你看到 toy 方案的边界**。

这也是为什么本章用两个脚本递进：

- **01**——跑通最小骨架（`chunk_by_paragraph` + `split_long_paragraph`，500 字符 cap + 中英句界切）；
- **02**——把同一套函数喂给真实样本，把"看不见的损失"摆出来：表格切成 2 块时第 0 块停在 mid-row "8GB"、节标题变成 10 字符的孤岛、"收入结构如下"和后续数字段永远不见面。

这也是为什么我们不直接用 LangChain `RecursiveCharacterTextSplitter`——它在底层解决了"递归找分隔符"，但你看不清每一步切在哪、为什么切。先见失败，再看修复，比直接用封装库学到的多。

---

## 二、详细解说

### 2.1 最小可跑分块：500 字符 cap + 句界切

> 02 会跑同一套函数到真实样本上，展示哪些情况它会崩。

#### 这是什么

1. `split_long_paragraph(text, max_chars)` — lookbehind 正则在 `[.。!?！？]` 之后切，把超长段落切成 ≤ max_chars 的若干块；极端情况（无标点的规格表）按字符硬切兜底；
2. `chunk_by_paragraph(docs, max_chars=500)` — 短段整段保留为 1 块，长段调 `split_long_paragraph` 后展开多块；每块带 `chunk_id = {source}#{page}#p{n}`；
3. **复用 s02 中 01 的 `load_pdf` / `load_docx`**——loader 是上游契约，本脚本不重复实现。

入口：[`code_01_basic_chunk.py`](code_01_basic_chunk.py)

#### 跑起来

```bash
python s03_chunking/code_01_basic_chunk.py
```

输出：

```
输入段落 31 → 输出块 34
最大块长度 452 字符 (cap=500)
server_whitepaper.pdf#1#p0 | 紫光恒越 R3630 G5 双路机架式服
务器
产品白皮书 ...
server_whitepaper.pdf#1#p1 | 二、关键特性
计算密度 ...
server_whitepaper.pdf#2#p2 | 三、整机规格
组件 规格 说明 ...
```

#### 它做对了什么

- **token-aware 边界**：按 `[.。!?！？]` 切而不是裸字符切，中英文段落都不会拦腰砍断句子；
- **硬切兜底**：单句本身超过 `max_chars`（无标点规格表）按字符切，最坏情况下输出不超过 `2 * max_chars`；
- **chunk_id 稳定可引用**：`{source}#{page}#p{n}` 形式让 s04+ 可以直接引用具体 chunk；
- **零新依赖**：只用 `re` + `pathlib`。

#### 它做错了什么

- **表格被切碎**：`pypdf.extract_text()` 输出的表格是挤在一行的长串，句界切不到、只能硬切 → 每个 chunk 只看到半张表；
- **父子块概念缺失**：500 字封顶的扁平列表，召回后 LLM 拿不到"完整语义单位"，回答"Q3 营收多少"只能看到切碎的片段；
- **跨段落引用断裂**："见上表""如表 3 所示" 这种指代词单独成 chunk 后，检索召回的是指代词本身、不是它指向的实体。

02 会在 `samples/` 上把这三类失败各跑一遍。

#### 思考题

**如果一段就是 800 字但语义完整（比如一段财务披露），是该切还是不该切？**

提示：固定字符切分解决不了这个问题——切了句子被拦腰截断；不切单 chunk 太长 embedding 模型失真（BGE `max_seq_len=512`，但超长块会稀释语义）。RAGFlow 的 parent-child 是答案：整段作为 parent 保留语义完整性，内部再切小 child 用于召回匹配，命中后把 parent 整体塞给 LLM。详见 02 的 `demo_parent_child`。

### 2.2 chunker 在真实样本上的三类失败

> 01 在 toy 上能跑；放到真实 `samples/` 上会崩在哪？
> 本脚本定位 3 类问题 + 引出工业解法。

#### 这是什么

把 01 的 `chunk_by_paragraph` 喂给真实样本（`server_whitepaper.pdf` 4 页 + `disclosure.docx` 27 段），把"看不见的损失"暴露出来：

1. **表格被切碎**——`pypdf.extract_text()` 输出的规格表挤在一行里，句界切不到、只能字符硬切，每个 chunk 只看到半张表；
2. **父子块缺失**——节标题（"第四节 分季度财务数据"）单独成 10-char chunk，用户问"Q3 营收"时检索命中的是标题，数字本身在 DOCX tables 里（被 s02 loader 丢了）；
3. **跨段引用断裂**——"收入结构如下""见表 3"这种指代词单独成 chunk 后无主语，召回的是指代词本身而不是它指向的实体。

入口：[`code_02_chunk_failures.py`](code_02_chunk_failures.py)

#### 跑起来

```bash
python s03_chunking/code_02_chunk_failures.py
```

输出片段：

```
================================================================
[a] 表格被切碎 — 整机规格表 hard-cut
================================================================
BEFORE (整段 562 字符):
三、整机规格
组件 规格 说明
处理器 2 × 第三代 Intel Xeon 可
扩展处理器
...
AFTER  (cap=500 → 切成 2 块):
  chunk 0 len= 396 | 末行: NAND
  chunk 1 len= 164 | 末行: ...
→ 失败点: 第 0 块末尾停在 '8GB' (BMC 那行的中间),
          第 1 块从 'NAND' 开头继续,失去 '组件 / 规格 / 说明' 列对齐语义。
```

```
================================================================
[b] 父子块缺失 — 节标题 chunk 与数据本体分离
================================================================
BEFORE (用户问 'Q3 营收多少'):
  召回命中 chunk_id=disclosure.docx#None#p18 len=11 text='第四节 分季度财务数据'
AFTER  (季度数据本体在 DOCX tables,共 3 张表):
  (本样本 DOCX 表未含 Q3 字面量; 但所有季度数字都只在 tables 里,chunker 看不到)
```

```
================================================================
[c] 跨段引用断裂 — 指代词单独成 chunk
================================================================
BEFORE (过短的 header-only chunks,语义为零):
  id=disclosure.docx#None#p8  len=15 | '2024 年度财务信息披露报告'
  id=disclosure.docx#None#p13 len=10 | '第二节 主要财务数据'
  id=disclosure.docx#None#p18 len=11 | '第四节 分季度财务数据'
```

#### 它做对了什么

- **暴露问题**：每个 demo 都打印 before/after 片段，让"为什么这种切法不够"肉眼可见；
- **解法对照**：每个失败都点名工业方案的对应模块（`_concat_downward` / `naive_merge` / `hierarchical_merge` / `attach_media_context`）；
- **量化损失**：表格切成 2 块时第 0 块停在 mid-row "8GB"——直接给 LLM 看它能不能拼回去；
- **零新依赖**：只 import 01 + 标准库 + `python-docx`（已装）。

#### 它做错了什么

- 它是个 demo，不是 fix——没有真的把表拼回去、没有真的建父子树、没有真的回填 context；
- 依赖 s02 中 01 的 loader（后者已丢 DOCX tables）；如果想看到表格里的季度数字，要么改 s02 loader 要么用 `python-docx` 直读；
- 只测了 3 类失败，真实场景还有 (d) 页眉页脚污染 / (e) 多栏错位 / (f) 扫描件 OCR 缺失，那些是 s02 + s11 的事。

#### 思考题

**如果只允许修 1 类失败，优先修哪类？为什么？**

提示：决策不是技术问题——是产品问题。如果你的语料以**白皮书 / 财报** 为主，表格切碎会让检索准度断崖下降（用户问"内存最大多大"命中半张表 → LLM 瞎编）；如果以**长文档 / 法规** 为主，父子块缺失让 LLM 永远只看片段；如果是**导航型** 文档（FAQ），跨段引用最多让用户体验不好但不影响答案正确性。你要结合自己的 samples 分布选。

---

## 三、怎么做？

### 3.1 跑起来

```bash
python s03_chunking/code_01_basic_chunk.py
python s03_chunking/code_02_chunk_failures.py
```

依赖：`s03` 复用 `s02` 的 `load_pdf` / `load_docx`（`importlib` 按文件路径加载，避免把 `s02` 装成顶层包）。先把 s02 跑通，s03 才能跑。

### 3.2 核心函数一览

| 函数 | 文件 | 输入 | 输出 | 一句话解释 |
|---|---|---|---|---|
| `split_long_paragraph(text, max_chars)` | `code_01_basic_chunk.py` | `str`, 整数 cap | `list[str]` | 超长段落按 `[.。!?！？]` 切，再贪心装桶；单句超 cap 按字符硬切兜底 |
| `chunk_by_paragraph(docs, max_chars=500)` | `code_01_basic_chunk.py` | `list[dict]`, 整数 cap | `list[dict]` | 短段整段保留；长段调 `split_long_paragraph`；每块带 `chunk_id` |
| `main()` (01) | `code_01_basic_chunk.py` | — | 打印块数 + 最大长度 + 前 3 个 chunk | 01 演示入口 |
| `demo_table_split()` | `code_02_chunk_failures.py` | — | 打印规格表 hard-cut 前后 | 失败 a：表格被切到 mid-row |
| `demo_parent_child()` | `code_02_chunk_failures.py` | — | 打印 header-only chunk 与数据本体 | 失败 b：父子块缺失 |
| `demo_cross_ref()` | `code_02_chunk_failures.py` | — | 打印短 chunk 列表 + DOCX 原文 | 失败 c：跨段引用断裂 |
| `main()` (02) | `code_02_chunk_failures.py` | — | 调用上面 3 个 demo + 引出工业解法 | 02 演示入口 |

### 3.3 chunker 设计取舍

为什么是"500 字符 cap + 句界切"而不是别的？这是几个常见取舍的折中：

- **字符数 vs token 数**：本教程用字符。中文字符 ≈ 2-3 token（取决于分词器），500 字符 ≈ 1000-1500 token——**已经超过 BGE 的 512 token 上限**。我们故意保留这个"超限"作为教学锚点，让 s04 的 embedding 任务能演示"长 chunk 被截断"的真实后果；
- **500 vs 200 vs 1024**：500 是一个"够大能讲完一个事实、又够小能放进一两屏 LLM context"的折中。生产里如果 embedding 是 1024 token cap（如 `bge-large`），chunk 应改 256 字符；如果是 BM25-only，1024 字符也 OK；
- **句界正则 `[.。!?！？]` vs 段落 `\n\n`**：段落之间靠 s02 已经分好了；段内的进一步切只能靠句界。空格在中文里不可靠（CJK 文本没词边界），所以只认标点；
- **不做重叠**：MVP 默认 `overlapped_percent=0`。重叠会增加 embedding 索引大小和后续检索去重复杂度，toy 阶段先不引入；RAGFlow 用 `overlapped_percent=10-20` 提高跨边界召回率；
- **不建父子树**：MVP 是扁平 `list[dict]`，没有 parent / child 关系。RAGFlow 的 `_concat_downward` 输出的"父块"和 `naive_merge` 输出的"子块"是两个不同列表——我们这里把两层压成一层，召回精度自然会下来；
- **`chunk_id = {source}#{page}#p{n}`**：`p{n}` 是**输出顺序**而非段内位置——确保 s04+ 引用稳定，不会因为切分顺序变化而错位。

如果你的语料以**表格 / 财报**为主，500 字符 cap 必须显著缩小（200 左右），否则每张表都会被拦腰切；如果你处理的是**小说 / 长文**，可以放宽到 1024 字符以保留叙事完整性。

### 3.4 实际跑出来的 schema 形状

把 `01` 跑在仓库自带的 `samples/` 上，得到的真实片段长这样：

```python
# 输入（s02 loader 输出）
[
  {"text": "紫光恒越 R3630 G5 双路机架式服务器\n产品白皮书  ·  v1.0...", "page": 1, "source": "server_whitepaper.pdf"},
  {"text": "二、关键特性\n计算密度：单台 2U 机箱...", "page": 1, "source": "server_whitepaper.pdf"},
  {"text": "三、整机规格\n组件 规格 说明\n处理器 ...", "page": 2, "source": "server_whitepaper.pdf"},
  ...
]

# 输出（s03 chunker）
[
  {"text": "紫光恒越 R3630 G5 双路机架式服\n务器\n产品白皮书  ·  v1.0...", "page": 1, "source": "server_whitepaper.pdf", "chunk_id": "server_whitepaper.pdf#1#p0"},
  {"text": "二、关键特性\n计算密度：单台 2U 机箱内集成两颗处理器...", "page": 1, "source": "server_whitepaper.pdf", "chunk_id": "server_whitepaper.pdf#1#p1"},
  {"text": "三、整机规格\n组件 规格 说明\n处理器 2 × 第三代 Intel Xeon 可 扩展处理器...", "page": 2, "source": "server_whitepaper.pdf", "chunk_id": "server_whitepaper.pdf#2#p2"},
  ...
]
```

下游 embedding（s04）拿到这个列表时，**不需要知道 chunk 是怎么来的**——它只关心 `text` 和 `chunk_id` 两个字段。这就是分块层把"边界检测"封装掉的价值：后续章节按统一接口消费即可。

### 3.5 跑出来是什么样

`01` 的预期输出（具体数字由 `samples/` 决定）：

```
输入段落 31 → 输出块 34
最大块长度 452 字符 (cap=500)
server_whitepaper.pdf#1#p0 | 紫光恒越 R3630 G5 双路机架式服
务器
产品白皮书  ·  v1.0  ·  仅用于 RAG 教程测试
server_whitepaper.pdf#1#p1 | 二、关键特性
计算密度：单台 2U 机箱内集成两颗处理器、32 条内存 DIMM 与 10 个 PCIe 4.0 槽位，
server_whitepaper.pdf#2#p2 | 三、整机规格
组件 规格 说明
处理器 2 × 第三代 Intel Xeon 可
扩展处理器
最高 40 核/80 线程
```

31 段（4 页 PDF + 27 段 DOCX）→ 34 块，**最大块 452 字符，未触发硬切兜底**。`chunk_id` 按 `{source}#{page}#p{n}` 稳定生成，可被 s04+ 直接引用。

`02` 的预期输出（节选）：

```
================================================================
[a] 表格被切碎 — 整机规格表 hard-cut
================================================================
BEFORE (整段 562 字符):
三、整机规格
组件 规格 说明
处理器 2 × 第三代 Intel Xeon 可 扩展处理器 ...
AFTER  (cap=500 → 切成 2 块):
  chunk 0 len= 396 | 末行: NAND
  chunk 1 len= 164 | 末行: ...
→ 失败点: 第 0 块末行停在 '8GB' (BMC 那行的中间),失去列对齐
→  工业解法: pdf_parser.py 用 XGBoost 30 特征识别表格 layout_type=table

================================================================
[b] 父子块缺失 — 节标题 chunk 与数据本体分离
================================================================
BEFORE (用户问 'Q3 营收多少'):
  召回命中 chunk_id=disclosure.docx#None#p18 len=11 text='第四节 分季度财务数据'
AFTER  (季度数据本体在 DOCX tables,共 3 张表):
  (本样本 DOCX 表未含 Q3 字面量;但所有季度数字都只在 tables 里,chunker 看不到)

================================================================
[c] 跨段引用断裂 — 指代词单独成 chunk
================================================================
BEFORE (过短的 header-only chunks):
  id=disclosure.docx#None#p8  len=15 | '2024 年度财务信息披露报告'
  id=disclosure.docx#None#p13 len=10 | '第二节 主要财务数据'
  id=disclosure.docx#None#p18 len=11 | '第四节 分季度财务数据'
```

**Troubleshooting**：

- `ModuleNotFoundError: No module named 'pypdf'`：s03 复用 s02 的 loader，先把 s02 跑通。
- 输出块数 ≤ 输入段数：每个 PDF / DOCX 段几乎都被整段保留（因为大多数段 < 500 字符）；长段才会展开成多块。如果你的样本全是 1000 字符的段，输出块数会显著大于输入段数。
- 输出顺序不对：检查 `len(out)` 的位置——`chunk_id` 里的 `p{n}` 是输出顺序而非输入顺序，确保 `len(out)` 在 append 后递增。

---

## 四、选型与思考题

### 4.1 主流分块工具速览

下面这张表把社区常用的几类 chunker 按"边界检测 / 单位 / 是否版面感知 / 部署"列出来：

| 工具 | 边界检测 | 单位 | 版面 / 结构感知 | 部署 | 适用场景 |
|---|---|---|---|---|---|
| **`chunk_by_paragraph`**（本教程） | 句界正则 `[.。!?！？]` | 500 字符 | 无 | 本地，纯 Python | 文本型 PDF / DOCX 教学 |
| **LangChain `RecursiveCharacterTextSplitter`** | 递归分隔符 `\n\n → \n → 。 → 字符` | 字符（可配 token） | 无 | 本地 | 通用快速原型 |
| **LangChain `SemanticChunker`** | 相邻句 embedding 距离 + 百分位阈值 | 由内容决定 | 语义级 | 本地 + embedding 模型 | 长文 / 主题切换明显的文档 |
| **LangChain `MarkdownHeaderTextSplitter`** | Markdown 标题级别 | 由结构决定 | 结构级 | 本地 | Markdown / 技术文档 |
| **Unstructured `by_title`** | Title 元素边界 | `max_characters` | 版面级（`Title` / `NarrativeText` / `ListItem`） | 本地 | 多格式混排报告 |
| **RAGFlow DeepDoc** | XGBoost 30 特征 + 句界 | tiktoken 128 token | 版面 + 层级 + 媒体回填 | 本地 | 复杂 PDF / 表格 / 多栏 |

我们的 toy `chunk_by_paragraph` 在边界检测上跟 LangChain 第二行同量级（句界 / 递归），在版面感知上是空白——这是为什么 02 必须显式暴露"表格切碎""父子块缺失"两类失败。

### 4.2 选型速记

- **教学 / 快速原型** → `chunk_by_paragraph` (本教程) 或 `RecursiveCharacterTextSplitter`；
- **Markdown / 技术文档** → `MarkdownHeaderTextSplitter`（保留章节元数据）；
- **需要语义边界**（小说、访谈）→ `SemanticChunker`（要 embedding 模型，慢）；
- **多格式混排报告** → `Unstructured by_title`（版面感知，但本地依赖重）；
- **复杂 PDF / 表格 / 多栏** → RAGFlow DeepDoc（4 层流水线，本地 + XGBoost 模型）；
- **要先看清错误再选工具** → 用本章 `02` 的真实样本把损失量化，再决定要不要换。

### 4.3 思考题

**如果一段就是 800 字但语义完整（比如一段财务披露），是该切还是不该切？**

答案见下方"思考题答案"。


## 思考题答案

### Q: 如果一段就是 800 字但语义完整，是该切还是不该切？

**该切，但要换一种切法——按"父-子"层级切，而不是单层扁平切。**

### 单层固定字符切为什么不行

固定字符切分（我们 s03 的做法）解决不了"800 字但语义完整"的两难：

- **不切**：单 chunk 800 字，超出 BERT/BGE 类 Embedding 模型的 max_seq_len（典型 512-8192），即便没超，长文本平均化后语义向量失真，召回率下降。
- **切了**：句子被拦腰截断，Embedding 把残句当成完整语义单位，反而召回到错的片段；LLM 拿到半句话又生成幻觉。

无论哪种，固定字符切分都在"粒度"和"完整性"之间二选一。

### RAGFlow 的解法：parent-child

RAGFlow 用 **parent-child 双层结构**：

1. **父块 (parent)**：用版面识别（`_concat_downward` + XGBoost `updown_cnt_mdl`）把视觉相邻的文本框递归合并成段落/表格块，**保留 800 字的语义完整性**。
2. **子块 (child)**：在父块内部，用 `naive_merge` 按句界 `\n。；！？` + `chunk_token_num=128` token 上限切成小块，**保证 Embedding 友好**。
3. **召回与生成分离**：检索时用 child 的 Embedding 算相似度（细粒度召回），命中后把整个 parent 的文本返回给 LLM（完整语义单位）。

这样"800 字但语义完整"的段落就成了 1 个 parent，内部切 4 个 child 各 200 字，召回任何 child 都把 800 字 parent 整体返回——粒度和完整性同时满足。

### 为什么这是"最小解法"的升级

我们 s03 的 500 字 cap 是教学原型，够跑通管道、够讲清"为什么需要分块"。但任何生产 RAG 系统要做到"长段落也能正确回答"，都绕不开 parent-child —— 这是 RAGFlow / LangChain `ParentDocumentRetriever` / LlamaIndex `HierarchicalNodeParser` 的共同选择。

### 参考

- [`docs/reference/ragflow-notes/deepdoc_chunking.md`](../../docs/reference/ragflow-notes/deepdoc_chunking.md)：parent-child 在 RAGFlow `pdf_parser.py` / `rag/nlp/__init__.py:naive_merge` 的具体实现。
