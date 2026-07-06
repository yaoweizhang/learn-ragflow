# s02 文档加载 — 章节总览

> **章节定位**：RAG 离线流水线的第一步——把 PDF / DOCX 这些**非结构化文档**读成结构化段落。  
> **章节定位**：本章节围绕 *单文件加载* 这一层给出概念 / 问题 / MVP / 工业对照的完整弧线,**不引入 Unstructured / 多分块策略**(那些留到 s03 / s11)。

---

## 一、什么是文档加载？

### 1.1 核心定义

**文档加载（Document Loading）** 是把磁盘上的 PDF / DOCX / HTML / Markdown 等文件，转成程序可以继续处理的 `list[dict]`——每段通常带三个字段：`text`（正文）、`page`（页码；DOCX 时常为 `None`）、`source`（文件名）。它的上游是文件系统，下游是切块（s03）、Embedding（s04）、写向量库（s05）。

把它放进 RAG 全景看：**s02 是离线索引链路的入口**。如果入口吐出来的是错位、漏表、被页眉污染的段落，后面 embedding 再贵、检索策略再花哨也救不回来。这就是社区那句老话——**Garbage In, Garbage Out**——在 RAG 里被反复引用的原因。

### 1.2 三个核心任务

文档加载器在 RAG 流水线里做三件事：

1. **解析（Parse）**——按文件格式调不同的库（PDF 走 `pypdf` / `PyMuPDF`，DOCX 走 `python-docx`），从字节流里把文字、表格、图片位置抠出来；
2. **抽取元数据（Extract）**——页码、章节标题、文档作者等。这一步对**溯源**（用户问"出处"时能反查）和**过滤**（比如去掉页眉页脚）至关重要；
3. **对齐 schema（Normalize）**——把不同格式的输出整理成同一种数据结构，下游切块器不需要为每种格式重写。

### 1.3 与传统 ETL 的对应

文档加载的"解析 → 抽取 → 对齐"三步，和传统数据工程里的 **ETL（Extract-Transform-Load）** 几乎是一一对应的：

| 文档加载 | ETL | 共同目标 |
|---|---|---|
| 解析 | Extract | 把异构字节流还原成结构 |
| 抽取元数据 | Schema 推断 | 给数据加上类型 / 来源标签 |
| 对齐 schema | Transform | 输出统一形态给下游 |

ETL 处理的是数据库表 / 日志，文档加载处理的是 PDF / DOCX。**目标都一样——把杂乱的原始数据清洗并对齐为适合后续检索和建模的标准化语料**。

---

## 二、为什么要单独写一章加载？

`pypdf` 几行就能跑，`python-docx` 也是。看起来不值得单独成章。但把它扔进真实样本就会发现，30 行版本和生产级方案之间隔着一道悬崖——这道悬崖由几类典型问题堆起来：

### 2.1 真实世界的问题（2-4 条典型）

1. **编码与乱码**——DOCX 在 Windows GBK 控制台打印时容易报 `UnicodeEncodeError`；老 PDF 用非 UTF-8 字符映射（如 GBK / Big5）直接出乱码；CJK 字体的 PDF 还可能因字体子集化丢失字符；
2. **扫描件读不出文字**——`pypdf` 对纯图像页（扫描件、手机拍的合同）`extract_text()` 返回 `""`，整页被静默丢弃；
3. **复杂版面错位**——双栏 PDF 按字符位置扫，左右栏的尾部会接到一起；多栏学术论文尤其严重；
4. **表格 / 图片丢失或拍扁**——DOCX 的表格存在 `Document.tables` 而非 `Document.paragraphs`，按段读会把整张表吞掉；PDF 的表格在 `extract_text()` 里被拍成没对齐的碎片；图片 OCR 文本完全消失。

### 2.2 这些问题为什么必须显式面对

每条都对应着不同的工业级解法——OCR、版面分析、表格识别——这些都是 s11 的主题。**s02 的目标不是解决它们，而是把它们显式暴露出来，让你看到 toy 方案的边界**。

这也是为什么本章用两个 unit 递进：

- **unit 01**——跑通最小骨架（`pypdf` + `python-docx` → `list[{text, page, source}]`）；
- **unit 02**——把同一套函数喂给真实样本，把"看不见的损失"摆出来：丢多少字、错位多严重、表里写了什么但读者看不见。

这也是为什么我们不直接用 `Unstructured` / `PyMuPDF4LLM` 这类更"省心"的库——它们在底层解决了这些问题，但你看不到**哪些原本会出错、错在哪**。先见错误，再看修复，比直接用封装库学到的多。

---

## 三、怎么做？

### 3.1 章节导航

| Unit | 主题 | 它解决什么 | 对照 RAGFlow |
|---|---|---|---|
| [01_basic_load](./units/01_basic_load/README.md) | 最小可跑加载 (PDF + DOCX) | "统一 schema 是什么样" | `deepdoc/parser/{pdf,docx}_parser.py` |
| [02_failure_modes](./units/02_failure_modes/README.md) | 真实样本上的失败模式 | "为什么 unit 01 在 prod 不够" | `deepdoc/parser/utils.py` VisionParser |

### 3.2 跑起来

```bash
pip install pypdf python-docx
python s02_doc_loading/units/01_basic_load/code.py
python s02_doc_loading/units/02_failure_modes/code.py
# 旧路径仍可用（聚合入口）:
python s02_doc_loading/code.py
```

### 3.3 核心函数一览

s02 的代码非常薄，但每个函数都对应一种文件格式到统一 schema 的桥接：

| 函数 | 文件 | 输入 | 输出 | 一句话解释 |
|---|---|---|---|---|
| `load_pdf(path)` | `units/01_basic_load/code.py` | `Path` (PDF) | `list[{text, page, source}]` | `pypdf.PdfReader` 逐页 `extract_text()`，跳过空页，page 从 1 起 |
| `load_docx(path)` | `units/01_basic_load/code.py` | `Path` (DOCX) | `list[{text, page=None, source}]` | `python-docx.Document.paragraphs` 顺序读，仅保留非空段（**注意：表格不在内**） |
| `main()` (unit 01) | `units/01_basic_load/code.py` | — | 打印段落数 + 片段 | 演示入口；unit 02 复用 `load_pdf` / `load_docx` |
| `show_pdf_failure()` | `units/02_failure_modes/code.py` | — | 打印每页长度与首字 | 把 unit 01 的输出按页铺开，肉眼看出多栏错位 |
| `show_docx_table_loss()` | `units/02_failure_modes/code.py` | — | 打印表内字符数 | 直接量化"被吞掉的表格字数" |
| `main()` (unit 02) | `units/02_failure_modes/code.py` | — | 调用上面两个 + 引出 ragflow | unit 02 演示入口 |

### 3.4 schema 设计取舍

为什么是 `{text, page, source}` 三个字段而不是别的形状？这是几个常见取舍的折中：

- **每页一段 vs 整篇一段**：本教程按"页"切。好处是 `page` 字段天然可溯源，坏处是 PDF 单页内的多个段落会被绑死成一个 chunk——s03 的切块器要做这件事；
- **`source = filename` vs 全路径**：我们只存文件名。整路径在跨机器迁移时会泄漏本地结构、也会重复前缀浪费存储；
- **`page` 在 DOCX 时为 `None` 而不是 `0` 或 `-1`**：这样下游"判断页码是否存在"用 `if page is None` 比 `if page` 更稳，避免误把 `0` 当缺省；
- **不做去重 / 不做清洗**：s02 只负责"读"，不去重、不去页眉——那是 s03 / s11 的事。职责单一，方便替换。

如果你的语料源需要额外的元数据（比如作者、章节标题、创建时间），就在 schema 里加字段——但**保持向后兼容**：新字段给默认值，老代码不崩。

### 3.5 如何扩展更多格式

加一种新格式（比如 Markdown / HTML / Excel）只要三步：

1. 写一个 `load_xxx(path) -> list[dict]`，签名和 `load_pdf` / `load_docx` 一致；
2. 在 `main()` 里 `from xxx import ...` 并按格式分发；
3. 给单元 README 加一段失败模式描述（unit 02 的精神——先暴露问题再换工具）。

不要在 `load_pdf` 里写 if-else 分发——它会污染单一职责。`load_pdf` 只懂 PDF，`main()` 懂全格式。

### 3.6 实际跑出来的 schema 形状

把 `unit 01` 跑在仓库自带的 `samples/` 上，得到的真实片段长这样（用于对照"统一 schema 在两格式上是同一形状"）：

```python
# PDF 的段落
[
  {"text": "紫光恒越 R3630 G5 是面向企业核心业务...", "page": 1, "source": "server_whitepaper.pdf"},
  {"text": "三、整机规格\n组件  规格  说明\n处理器  2 × 第三代 Intel Xeon...", "page": 2, "source": "server_whitepaper.pdf"},
  ...
]

# DOCX 的段落（page 为 None）
[
  {"text": "青蓝科技股份有限公司\n2024 年度财务信息披露报告", "page": None, "source": "disclosure.docx"},
  {"text": "一、公司基本情况\n...", "page": None, "source": "disclosure.docx"},
  ...
]
```

下游切块器（s03）拿到这两种列表时，**不需要知道来源是 PDF 还是 DOCX**——它只关心 `text` / `page` / `source` 三个字段。这就是 schema 对齐的价值：**格式差异被吸收在加载层，后续章节不用再分情况处理**。

### 3.7 跑出来是什么样

`unit 01` 的预期输出（具体数字由 `samples/` 决定）：

```
PDF 段落数: 4, DOCX 段落数: 27
PDF 第 1 段前 100 字: 紫光恒越 R3630 G5 双路机架式服
务器
产品白皮书  ·  v1.0  ·  仅用于 RAG 教程测试
一、产品概述
紫光恒越 R3630 G5 是面向企业核心业务、AI 推理与虚拟化负载设计
DOCX 第 1 段前 100 字: 青蓝科技股份有限公司 ...
```

4 是白皮书 PDF 解析出的非空页数（4 页都有内容）；27 是披露报告里的非空段落数。**3 张表格的内容不在这里**——这正是 `unit 02` 要揭示的问题。

`unit 02` 的预期输出（节选）：

```
[PDF] 4 页抽出的段落 (page, len, first 60 字):
  page= 1 len= 861 | 紫光恒越 R3630 G5 双路机架式服 务器 产品白皮书  ·  v1.0  ·  仅用于 RAG 教程测试 一、产品
  page= 2 len= 562 | 三、整机规格 组件 规格 说明 处理器 2 × 第三代 Intel Xeon 可 扩展处理器 最高 40 核/80 线程
  ...

[DOCX] paragraphs(非空)=27, tables=3, 表格内总字符=572
  → unit 01 的 load_docx 只读 paragraphs，丢失 572 字符（3 张表）
```

**Troubleshooting**：

- Windows GBK 控制台打印中文报 `UnicodeEncodeError: 'gbk' codec can't encode character '\xa0'`：控制台编码问题不是代码 bug，跑前 `set PYTHONIOENCODING=utf-8`。
- `len(pdf) == 0`：PDF 是纯扫描件，`pypdf` 抽不出文字——见 §二.1 第 2 条。
- `ModuleNotFoundError: No module named 'pypdf'`：`pip install pypdf python-docx` 后再跑。

---

## 四、对照 RAGFlow + 思考题

### 4.1 ragflow 怎么做的

RAGFlow 在 `deepdoc/parser/pdf_parser.py` 里维护了一组 PDF 解析器：`PlainParser`（纯文本 fallback）、`VisionParser`（视觉 LLM + OCR）、`TxtParser` 等。完整摘录与 3 条 "为什么这样写" 的分析见 [`ragflow_notes/deepdoc_pdf_parsing.md`](../ragflow_notes/deepdoc_pdf_parsing.md)；表格识别 / OCR fallback 的对应实现细节见 [`ragflow_notes/multimodal_parsing.md`](../ragflow_notes/multimodal_parsing.md)。

一句话对比：RAGFlow 把"读 PDF"拆成"先把页面栅格化成高分辨率图像 → 视觉模型识别版面/表格/文字 → 按坐标回填段落"三步走，跟 `code.py` 的一行 `extract_text()` 是两个量级——它专为扫描件、复杂表格、多栏版面设计。本章只要懂"它存在、它为什么贵"，s11 会真上手调它。

### 4.2 主流加载工具速览

下面这张表把社区常用的几类加载器按"格式覆盖 / 是否带版面识别 / 是否本地 / 是否需要 GPU"列出来，方便选型时快速对照：

| 工具 | 格式 | 版面/表格识别 | 部署 | 适用场景 |
|---|---|---|---|---|
| **pypdf**（本教程 demo） | PDF | 无 | 本地 | 文本型 PDF 快速解析 |
| **python-docx**（本教程 demo） | DOCX | 部分（仅段落级） | 本地 | 结构简单的 Word 报告 |
| **PyMuPDF4LLM** | PDF | 无（侧重转 Markdown） | 本地 | 科研文献、技术手册 |
| **Unstructured** | PDF/DOCX/HTML/MD | 有（hi_res 策略） | 本地 | 多格式混排、需要结构化标签 |
| **LlamaParse** | PDF | 强 | 商业 API | 法律合同、学术论文 |
| **MinerU** | PDF | 强（LayoutLMv3 + YOLO） | 本地 + GPU | 学术文献、财务报表 |

我们的 toy 方案（`pypdf` + `python-docx`）在格式覆盖上只占第一行 / 第二行——能跑，但不抗复杂版面。要往生产走，至少要在 `Unstructured` 这一行附近才有底。

### 4.3 选型速记

- **要快、只处理文本型 PDF** → `pypdf` / `PyMuPDF`；
- **要多格式、结构化标签** → `Unstructured`（local）或 `LlamaParse`（API）；
- **要 GPU 加速的版面识别** → `MinerU` / `Marker`；
- **要先看清错误再选工具** → 用本章 `unit 02` 的真实样本把损失量化，再决定要不要换。

### 4.4 思考题

**怎么检测某页是扫描件？给一个简单的启发式。**

参考答案见 [`thinking_answers.md`](./thinking_answers.md)。
