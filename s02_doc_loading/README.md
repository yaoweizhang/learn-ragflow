# s02 文档加载 — 把 PDF / DOCX 读成 RAG 能用的段落

> **本章定位**：s02 是 RAG 离线流水线的第一步——把 PDF / DOCX 这些**非结构化文档**读成结构化段落。详细定位见 s00 §1.4；RAGFlow 实现见本章末"## RAGFlow 实现"。

---

## 一、章节介绍

`pypdf` 几行就能跑，`python-docx` 也是。看起来不值得单独成章。但把它扔进真实样本就会发现，30 行版本和生产级方案之间隔着一道悬崖——这道悬崖由几类典型问题堆起来。

### 1.1 核心定义

**文档加载（Document Loading）** 是把磁盘上的 PDF / DOCX / HTML / Markdown 等文件，转成程序可以继续处理的 `list[dict]`——每段通常带三个字段：`text`（正文）、`page`（页码；DOCX 时常为 `None`）、`source`（文件名）。它的上游是文件系统，下游是切块（s03）、Embedding（s04）、写向量库（s05）。

```
 PDF / DOCX 字节流               list[{text, page, source}]
 (samples/*.pdf / *.docx)        段落级结构化数据
        │                                  │
        │  pypdf.PdfReader                 │
        │  ────────────────────────────▶   │
        │  .extract_text() 每页            │
        │                                  │
        │  python-docx                     │
        │  ────────────────────────────▶   │
        │  Document.paragraphs             │
        ▼                                  ▼
   "解析 → 抽取 → 对齐"  ──▶  s03 chunking  / s04 embedding  / s05 index
```

把它放进 RAG 全景看：**s02 是离线索引链路的入口**。如果入口吐出来的是错位、漏表、被页眉污染的段落，后面 embedding 再贵、检索策略再花哨也救不回来。这就是社区那句老话——**Garbage In， Garbage Out**——在 RAG 里被反复引用的原因。

### 1.2 三个核心任务

文档加载器在 RAG 链路里做三件事：

1. **解析（Parse）**——按文件格式调不同的库（PDF 走 `pypdf` / `PyMuPDF`，DOCX 走 `python-docx`），从字节流里把文字、表格、图片位置抠出来；
2. **抽取元数据（Extract）**——页码、章节标题、文档作者等。这一步对**溯源**（用户问"出处"时能反查）和**过滤**（比如去掉页眉页脚）至关重要；
3. **对齐 schema（Normalize）**——把不同格式的输出整理成同一种数据结构，下游切块器不需要为每种格式重写。

### 1.3 真实世界的问题

1. **编码与乱码**——DOCX 在 Windows GBK 控制台打印时容易报 `UnicodeEncodeError`；老 PDF 用非 UTF-8 字符映射（如 GBK / Big5）直接出乱码；CJK 字体的 PDF 还可能因字体子集化丢失字符；
2. **扫描件读不出文字**——`pypdf` 对纯图像页（扫描件、手机拍的合同）`extract_text()` 返回 `""`，整页被静默丢弃；扫描件 detection + OCR 兜底是 s11 主题，本章只识别问题不提供方案；
3. **复杂版面错位**——双栏 PDF 按字符位置扫，左右栏的尾部会接到一起；多栏学术论文尤其严重；
4. **表格 / 图片丢失或拍扁**——DOCX 的表格存在 `Document.tables` 而非 `Document.paragraphs`，按段读会把整张表吞掉；PDF 的表格在 `extract_text()` 里被拍成没对齐的碎片；图片 OCR 文本完全消失。

---

## 二、最小可跑加载：PDF + DOCX → 统一 schema：[c01_basic_load.py](c01_basic_load.py)

> 02 会跑同一套函数到真实样本上，演示哪些情况它会崩。

### 概念

1. `load_pdf(path)` — `pypdf.PdfReader` 逐页 `extract_text()`，page 从 1 开始；
2. `load_docx(path)` — `python-docx` 按 `paragraphs` 顺序读，仅保留非空段；
3. 输出统一 `{text, page, source}` schema，page 在 DOCX 时为 `None`。

入口：[`c01_basic_load.py`](c01_basic_load.py)

### 跑一遍

```bash
python s02_doc_loading/c01_basic_load.py
```

输出：

```
PDF 段落数: 4, DOCX 段落数: 27
PDF 第 1 段前 100 字: 紫光恒越 R3630 G5 双路机架式服务器 产品白皮书 ...
DOCX 第 1 段前 100 字: 青蓝科技股份有限公司 ...
```

### 看输出

把 `01` 跑在仓库自带的 `samples/` 上，得到的 schema 真实片段长这样（用于对照"统一 schema 在两格式上是同一形状"）：

```python
# PDF 的段落
[
  {"text": "紫光恒越 R3630 G5 是面向企业核心业务...", "page": 1, "source": "server_whitepaper.pdf"},
  {"text": "三、整机规格\n组件  规格  说明\n处理器  2 × 第三代 Intel Xeon...", "page": 2, "source": "server_whitepaper.pdf"},
  ...
]

# DOCX 的段落(page 为 None)
[
  {"text": "青蓝科技股份有限公司\n2024 年度财务信息披露报告", "page": None, "source": "disclosure.docx"},
  {"text": "一、公司基本情况\n...", "page": None, "source": "disclosure.docx"},
  ...
]
```

下游切块器（s03）拿到这两种列表时，**不需要知道来源是 PDF 还是 DOCX**——它只关心 `text` / `page` / `source` 三个字段。这就是 schema 对齐的价值：**格式差异被吸收在加载层，后续章节不用再分情况处理**。

### 局限与下一步

本段做对了什么 — 把 PDF 和 DOCX 都归一化成同一份 `{text, page, source}` schema,让下游 s03+ 不需要分文件类型分支,接口形状留好了,后面章节只换实现。

- **DOCX 表格被吃掉**：`Document.paragraphs` 不含 `tables`，所有表内文字都丢；
- **PDF 多栏排版错位**：双栏 PDF 抽出来的文本会把左栏底部接到右栏顶部；
- **扫描件完全没救**：`extract_text()` 对图片型 PDF 返回空字符串。

- Windows GBK 控制台打印中文报 `UnicodeEncodeError: 'gbk' codec can't encode character '\xa0'`：控制台编码问题不是代码 bug，跑前 `set PYTHONIOENCODING=utf-8`。
- `len(pdf) == 0`：PDF 是纯扫描件，`pypdf` 抽不出文字——见 §一。3 第 2 条。
- `ModuleNotFoundError: No module named 'pypdf'`： `pip install pypdf python-docx` 后再跑。

下一章 s03 如何解决 — 把这份 schema 上的 `text` 单元按"句界 + token cap"切块,这样 s04 的 embedding 拿到的是局部稠密的小块,而不是一整页的稀疏长文本;而 PDF 多栏错位的问题由切块器在 token 粒度上重排缓解,表格补全留到 s11。

---

## 三、真实样本上的失败模式：[c02_failure_modes.py](c02_failure_modes.py)

> 01 在 toy 上能跑；放到真实 `samples/` 上会崩在哪？
> 本脚本定位问题 + 引出 ragflow 的工业解法。

### 概念

把 01 的 `load_pdf` / `load_docx` 喂给真实样本 (`samples/server_whitepaper.pdf` 4 页 + `samples/disclosure.docx` 27 段），把"看不见的损失"暴露出来：

1. **PDF 多栏错位**——`pypdf.extract_text()` 在双栏 PDF 上按字符位置扫，左右栏会交错；
2. **DOCX 表格丢失**——`python-docx.Document.paragraphs` 不含 `Document.tables`，表内所有文字静默丢弃。

入口：[`c02_failure_modes.py`](c02_failure_modes.py)

### 跑一遍

```bash
python s02_doc_loading/c02_failure_modes.py
```

输出片段：

```
[PDF] 4 页抽出的段落 ...
  page= 1 len= 612 | 紫光恒越 R3630 G5 双路机架式服务器 ...
  page= 2 len=1024 | 产品型号 产品白皮书 文档版本 ...
  ...

[DOCX] paragraphs(非空)=27, tables=3, 表格内总字符=572
  → 01 的 load_docx 只读 paragraphs,丢失 572 字符(3 张表)
```

### 看输出

`02` 的预期输出（节选）：

```
[PDF] 4 页抽出的段落 (page, len, first 60 字):
  page= 1 len= 861 | 紫光恒越 R3630 G5 双路机架式服 务器 产品白皮书  ·  v1.0  ·  仅用于 RAG 教程测试 一、产品
  page= 2 len= 562 | 三、整机规格 组件 规格 说明 处理器 2 × 第三代 Intel Xeon 可 扩展处理器 最高 40 核/80 线程
  ...

[DOCX] paragraphs(非空)=27, tables=3, 表格内总字符=572
  → 01 的 load_docx 只读 paragraphs,丢失 572 字符(3 张表)
```

01 的预期输出（具体数字由 `samples/` 决定）：

```
PDF 段落数: 4, DOCX 段落数: 27
PDF 第 1 段前 100 字: 紫光恒越 R3630 G5 双路机架式服
务器
产品白皮书  ·  v1.0  ·  仅用于 RAG 教程测试
一、产品概述
紫光恒越 R3630 G5 是面向企业核心业务、AI 推理与虚拟化负载设计
DOCX 第 1 段前 100 字: 青蓝科技股份有限公司 ...
```

4 是白皮书 PDF 解析出的非空页数（4 页都有内容）；27 是披露报告里的非空段落数。**3 张表格的内容不在这里**——这正是 `02` 要揭示的问题。

### 局限与下一步

本段做对了什么 — 把 01 在真实样本上的失败模式量化出来(白皮书多栏错位 + 披露表丢失 572 字符),给后续章节提供"该改什么、量化损失多大"的对照表,而非修补方案。

- 暂时什么也没"做对"——它的目的就是展示 01 在 prod 上的失败。下一步要么换 loader (s11 表格抽取） 要么换格式（structured extraction）。

- `len(pdf) == 0` 且 `tables=0`：遇到的不是扫描件是空文档——确认 `samples/` 目录里有 `server_whitepaper.pdf` 和 `disclosure.docx`，不是空文件占位。
- 表格内字符数巨大（数千）：说明该 PDF 还含合并单元格 / 跨页表，`pypdf` 提取可能拍扁得更厉害——这是 s11 表格抽取的事。

下一章 s03 如何解决 — 在不重写 01 loader 的前提下,把 page-level 长文本切成 token-cap 级小块,让多栏错位的影响在切片粒度上被吸收;表格丢失留给 s11 多模态抽取补全。

---

## 四、核心函数一览

| 函数 | 文件 | 输入 | 输出 | 一句话解释 |
|---|---|---|---|---|
| `load_pdf(path)` | `c01_basic_load.py` | `Path` | `list[{text, page, source}]` | 逐页 `PdfReader(path).pages` + `extract_text()`;非空页才进 list;`page` 从 1 起编 |
| `load_docx(path)` | `c01_basic_load.py` | `Path` | `list[{text, page: None, source}]` | `Document(path).paragraphs` 逐段抽,空段过滤;`page` 在 DOCX 时强制 `None`(非 0/非 -1) |
| `main()` (01) | `c01_basic_load.py` | — | 打印 schema + 段数 | 01 演示入口,把 `samples/` 下两份文件分别喂 `load_pdf` / `load_docx` |
| `show_pdf_failure()` | `c02_failure_modes.py` | — | 打印 PDF 段落数 + 长度 + 前 60 字 | 02 失败演示 a:`extract_text()` 在双栏 PDF 上把左栏底接右栏顶 |
| `show_docx_table_loss()` | `c02_failure_modes.py` | — | 打印 paragraphs / tables / 表格字符数 | 02 失败演示 b:`Document.paragraphs` 不含 `tables` —— 整张表被吞掉 |
| `main()` (02) | `c02_failure_modes.py` | — | 调用上面两个 demo + 引出工业解法 | 02 演示入口,只暴露 01 在 prod 上的损失量化,不生产新数据 |

## 五、跨代码 schema 设计取舍

为什么是 `{text, page, source}` 三个字段而不是别的形状？这是几个常见取舍的折中：

- **每页一段 vs 整篇一段**：本教程按"页"切。好处是 `page` 字段天然可溯源，坏处是 PDF 单页内的多个段落会被绑死成一个 chunk——s03 的切块器要做这件事；
- **`source = filename` vs 全路径**：我们只存文件名。整路径在跨机器迁移时会泄漏本地结构、也会重复前缀浪费存储；
- **`page` 在 DOCX 时为 `None` 而不是 `0` 或 `-1`**：这样下游"判断页码是否存在"用 `if page is None` 比 `if page` 更稳，避免误把 `0` 当缺省；
- **不做去重 / 不做清洗**：s02 只负责"读"，不去重、不去页眉——那是 s03 / s11 的事。职责单一，方便替换。

如果你的语料源需要额外的元数据（比如作者、章节标题、创建时间），就在 schema 里加字段——但**保持向后兼容**：新字段给默认值，老代码不崩。

## 六、本章在 pipeline 中的位置

`01` 的输出（`{text, page, source}` schema）直接喂给 `02` 做失败检测；`01` 的输出同样直接喂给 s03 / s04 / s05 的整条离线管线。`02` 不生产新数据，只暴露 `01` 在 prod 上的损失量化——它的存在意义是"教学动机"，不参与主链路。

---

## RAGFlow 实现

RAGFlow 的文档解析在 `deepdoc/parser/` 目录下：PDF 走 `pdf_parser.py`（含 OCR 兜底），DOCX 走 `docx_parser.py`（含表格识别），Excel 走 `excel_parser.py`，图片走 `vision_parser.py`（用视觉模型）。

**设计取舍**：RAGFlow 不把"PDF 解析"塞进一个 2000 行的 `pdf.py`，而是按文件类型拆分，每种文件一个独立 parser + 调度器 `parser_factory.py`。接入新格式只需要写一个 parser 注册到 factory，不动其他代码。

详细摘录与 5-15 行 "为什么这样写" 的分析见 [`docs/reference/ragflow-notes/deepdoc_pdf_parsing.md`](../docs/reference/ragflow-notes/deepdoc_pdf_parsing.md)。

---

## 选型速记

### 主流加载工具速览

下面这张表把社区常用的几类加载器按"格式覆盖 / 是否带版面识别 / 是否本地 / 是否需要 GPU"列出来，方便选型时快速对照：

| 工具 | 格式 | 版面/表格识别 | 部署 | 适用场景 |
|---|---|---|---|---|
| **pypdf**(本教程 demo) | PDF | 无 | 本地 | 文本型 PDF 快速解析 |
| **python-docx**(本教程 demo) | DOCX | 部分(仅段落级) | 本地 | 结构简单的 Word 报告 |
| **PyMuPDF4LLM** | PDF | 无(侧重转 Markdown) | 本地 | 科研文献、技术手册 |
| **Unstructured** | PDF/DOCX/HTML/MD | 有(hi_res 策略) | 本地 | 多格式混排、需要结构化标签 |
| **LlamaParse** | PDF | 强 | 商业 API | 法律合同、学术论文 |
| **MinerU** | PDF | 强(LayoutLMv3 + YOLO) | 本地 + GPU | 学术文献、财务报表 |

我们的 toy 方案（`pypdf` + `python-docx`）在格式覆盖上只占第一行 / 第二行——能跑，但不抗复杂版面。要往生产走，至少要在 `Unstructured` 这一行附近才有底。

- **要快、只处理文本型 PDF** → `pypdf` / `PyMuPDF`；
- **要多格式、结构化标签** → `Unstructured`(local）或 `LlamaParse`(API）；
- **要 GPU 加速的版面识别** → `MinerU` / `Marker`；
- **要先看清错误再选工具** → 用本章 `02` 的真实样本把损失量化，再决定要不要换。

### 扩展指南

加一种新格式（。pptx / 。html / 。md）或加一层 OCR 兜底只要三步：

1. 写一个 `load_pptx(path) -> list[dict]` / `load_html(path) -> list[dict]`，**返回的 dict 必须沿用 `{text, page, source}` 三键 schema**，下游 chunker / embedder 不改一行；
2. 在 `c01_basic_load.py` 的 `main()` 里按 `path.suffix` 分发到对应 loader，**不要**在 `load_pdf` / `load_docx` 里写 `if suffix == '.pptx': ...`——污染单一职责；
3. 扫描件 PDF 兜底：在 `load_pdf` 里加 `if not text.strip(): return ocr_fallback(path)`，`ocr_fallback` 调到 s11 的 `ocr_image` 实现。

不要在 `load_pdf` 里塞 OCR 逻辑——它只懂 pypdf 文本提取，OCR 是另一个职责层。本章 MVP 只跑 PDF + DOCX，但 loader 形状已经预留到 6 个 suffix 都能挂上去。

---

## 思考题

1. **为什么 PDF 输出会有 4 段（4 页）但 DOCX 输出 27 段？这是"段落"的语义不同吗？**
2. **如果你的真实语料里 80% 是扫描件 PDF，01 的链路对它们返回空字符串。要不要给 01 加一层 OCR 兜底？还是放到 s11 单独做？**

（答案见文末「思考题答案」）

---

## 思考题答案

### Q1. 为什么 PDF 输出会有 4 段（4 页）但 DOCX 输出 27 段？这是"段落"的语义不同吗？

是的，统一 schema 时要分清两种"段落"：

- **PDF 的"段" = 页**(`page` 是结构边界）。`pypdf.extract_text()` 按页返一段字符串——一页内多段被绑在一起。
- **DOCX 的"段" = `\n` 切分的人工段落**(`python-docx` 的 `paragraphs` 是 Word 的 paragraph 元素）。

把 `page` 硬塞给 DOCX 用 `0` 或 `-1` 当 sentinel 都不优雅；**用 `None` 表达"这个 schema 字段对当前格式不适用"更稳**。下游拿到 `page=None` 时知道"不要按页号切片"——比如前端高亮时跳过 `page` 字段。统一 schema 时不要做假数据，直接 `None` 是最诚实的表达。

### Q2. 80% 扫描件，要不要给 01 加 OCR 兜底？

**推荐：放 s11，按需启用，不要进 01。**

判断依据：

1. **资源**：OCR 模型（PaddleOCR / Tesseract / GOT-OCR2）单个 50MB~1GB+；01 是入门骨架，强迫用户下模型 = 把"我想先看看 RAG"的人挡在门外。
2. **依赖复杂度**：OCR 涉及 GPU/CPU 推理、字体映射、CJK 识别率调参——这些都是独立子任务，不是 01 应承担的复杂度。
3. **可观测性**：02 的真实样本上有 80% 扫描件 → 这就是量化结果。它本身已经在告诉你"**01 在这种语料上不够用**，该跳到 s11"。**该跳就跳**，不要在 01 上贴膏药。

RAGFlow 的设计印证了这点：`VisionParser` 在 `deepdoc/parser/utils.py` 里是**单独的 dispatcher**——只有当 `RAGFlowPdfParser` 判定某页是扫描件时才会被调用，主路径不被 OCR 阻塞。

**实操建议**：你跑 02 看到扫描件率高 → 直接到 s11，看 RAGFlow 的 `VisionParser` 怎么集成；不需要也不应该改 01。
