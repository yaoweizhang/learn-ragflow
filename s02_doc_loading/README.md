# s02 文档加载 — 把 PDF / DOCX 读成 RAG 能用的段落

[上一章 s01 → · 下一章 s03 → ... → s12]

> *"`pypdf` 几行就能跑，`python-docx` 也是。扔进真实样本才会发现：30 行版本和生产级方案之间隔着一道悬崖 — 这道悬崖由'乱码 / 多栏错位 / 表被吞 / 扫描件静默丢' 四类典型故障堆起来"*
>
> **链路位置**: 离线索引链路第一步 (s02 → s03 → s04 → s05)
> **代码文件**: basic_load.py · failure_modes.py

> 环境准备: 见 root README §快速开始 — `pip install pypdf python-docx`

---

## 问题

`pypdf` 几行就能跑，`python-docx` 也是 — 看起来不值得单独成章。但把 demo loader 扔进真实样本就会发现，30 行版本和生产级方案之间隔着一道悬崖。这道悬崖由四类典型问题堆起来，每一类都在生产环境里反复出现：

**第一，编码与乱码**。DOCX 在 Windows GBK 控制台打印时容易报 `UnicodeEncodeError: 'gbk' codec can't encode character '\xa0'`；老 PDF 用非 UTF-8 字符映射（如 GBK / Big5）直接出乱码；CJK 字体的 PDF 还可能因字体子集化丢失字符。loader 本身没问题，输出在控制台层就崩了 — 控制台编码是部署层的隐式约束。

**第二，扫描件读不出文字**。`pypdf` 对纯图像页（扫描件、手机拍的合同）`extract_text()` 返回 `""`，整页被静默丢弃。用户以为"我传了 10 页 PDF，loader 应该给我 10 段文本"，实际拿到 0 段；后面 embedding / retrieval 在空文本上直接抛异常或返回空向量。扫描件 detection + OCR 兜底是 s11 主题，本章只识别问题不提供方案。

**第三，复杂版面错位**。双栏 PDF 按字符位置扫，左右栏的尾部会接到一起 — 左栏第 5 行接到右栏第 1 行，再接到左栏第 6 行；多栏学术论文尤其严重。loader 不知道"哪段属于哪栏"，把页面当成单栏流式文本，出来的段落对 RAG 来说是"被切碎又乱拼"的输入 — embedding 拿到的是语义混乱的字符串，retrieval top-k 会召回这种"碎段"。

**第四，表格 / 图片丢失或拍扁**。DOCX 的表格存在 `Document.tables` 而非 `Document.paragraphs`，按段读会把整张表吞掉；PDF 的表格在 `extract_text()` 里被拍成没对齐的碎片 (`列1 列2` 一行流，没有 row boundary)；图片 OCR 文本完全消失。对财报 / 合同 / 规格表这类强结构化文档，表格丢失 = 关键信息全丢 — 不是"丢 30% 召回率"，是"整份文档的核心信息蒸发"。

把这四种问题合起来看，**loader 的脆弱性不在 demo 上体现，在真实样本上才暴露**。这就是为什么 s02 必须先用 toy 跑通，再用真实样本量化损失 — 不是为了"写代码",而是为了"看清坑在哪里"。如果不在 s02 暴露这些坑，后面的 chunking / embedding / retrieval 全建立在沙滩上 — Garbage In, Garbage Out。

s02 的任务就是**先把最简单的 loader 跑起来，再用真实样本量化它的失败边界**，给后面 s03-s12 的工业解法留填空入口。本章是"看清坑"的章节，不是"填坑"的章节。

---

## 解决方案

文档加载器在 RAG 链路里做三件事，对应三个核心任务：

**第一，解析 (Parse)**——按文件格式调不同的库（PDF 走 `pypdf` / `PyMuPDF`，DOCX 走 `python-docx`），从字节流里把文字、表格、图片位置抠出来。**第二，抽取元数据 (Extract)**——页码、章节标题、文档作者等，对**溯源**（用户问"出处"时能反查）和**过滤**（比如去掉页眉页脚）至关重要。**第三，对齐 schema (Normalize)**——把不同格式的输出整理成同一种数据结构，下游切块器不需要为每种格式重写。

s02 用 **两个递进的脚本** 把这三件事跑起来。每一步解决前一步的局限，但也留下新的脆弱性：

```
代码 1 (最小可跑)               代码 2 (真实样本失败模式)
┌──────────────────┐         ┌───────────────────────┐
│ load_pdf /        │         │ 复用 load_pdf/load_docx│
│ load_docx         │ ────▶  │ + 量化 PDF 错位 +     │
│                  │         │   DOCX 丢表字符数      │
│ 输出 {text,page,  │         │                       │
│      source}     │         │ 输出损失量化 (不修)    │
└──────────────────┘         └───────────────────────┘
  toy 上跑通                    真实样本上暴露坑
```

| 脚本 | 解决什么 | 留下什么局限 | 何时用 |
|---|---|---|---|
| `basic_load.py` | PDF + DOCX → 统一 `{text, page, source}` schema | DOCX 表格被吞; PDF 多栏错位; 扫描件返回空串 | toy / 教学 / 文本型 PDF + 简单 DOCX |
| `failure_modes.py` | 量化 代码 1 在真实样本上的损失 (PDF 错位 + DOCX 丢 572 字符) | **不修** — 只量化; 表格/扫描件修复留给 s11 | 教学动机; 决策"要不要跳 s11" |

两脚本的关系是一条**教学主干**: 代码 1 把"PDF / DOCX → schema"做出来,暴露"toy 上 OK 真实样本失败"的局限 — 多栏错位让 embedding 拿到乱序文本, DOCX 丢表让关键信息蒸发; 代码 2 把代码 1 喂给真实样本量化损失 (PDF 4 段长度 861/562/413/744 + DOCX 27 段但丢了 572 字符 / 3 张表),暴露"代码 1 在这种语料上不够用"的结论。**s02 看清坑,后续章节填坑 — s03 chunking 缓解 PDF 多栏错位 (切片粒度重排), s11 多模态补全表格 / OCR**。

---

## 代码 1: 最小可跑加载 ([basic_load.py](basic_load.py))

### 工作原理

**做一件事**: 把 PDF / DOCX 两类最常见格式归一到同一种 `list[{text, page, source}]` schema,让下游 chunker / embedder 不用为每种格式重写。

**3 步**:
1. `load_pdf(path)` — `pypdf.PdfReader` 逐页 `extract_text()`,`page` 从 1 开始编,空页(纯扫描件)被 `if text.strip()` 过滤掉
2. `load_docx(path)` — `python-docx` 按 `Document(path).paragraphs` 顺序读,**只**保留非空段(去掉 Word 里"占位用的空段"),`page` 在 DOCX 时强制 `None`(不是 `0` / `-1` — 用 `None` 表达"此字段对当前格式不适用"最稳)
3. 两函数返回同一形状的 `list[dict]`,每段三键 schema,下游拿到任意一边都不需要 import 条件分支

```python
# 中间片段: DOCX 的 page=None 是有意的设计选择
def load_docx(path: Path) -> list[dict]:
    out = []
    for p in Document(path).paragraphs:
        if p.text.strip():
            out.append({"text": p.text, "page": None, "source": path.name})
    return out
```

**完整函数**:

```python
def load_pdf(path: Path) -> list[dict]:
    """逐页抽 text, 空页过滤, page 从 1 起编."""
    out = []
    for i, page in enumerate(PdfReader(path).pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            out.append({"text": text, "page": i, "source": path.name})
    return out


def load_docx(path: Path) -> list[dict]:
    """按 Word paragraph 抽, 空段过滤, page 强制 None(DOCX 无页概念)."""
    out = []
    for p in Document(path).paragraphs:
        if p.text.strip():
            out.append({"text": p.text, "page": None, "source": path.name})
    return out


def main() -> None:
    pdf = load_pdf(SAMPLES / "server_whitepaper.pdf")
    docx = load_docx(SAMPLES / "disclosure.docx")
    print(f"PDF 段落数: {len(pdf)}, DOCX 段落数: {len(docx)}")
    print("PDF 第 1 段前 100 字:", pdf[0]["text"][:100])
    print("DOCX 第 1 段前 100 字:", docx[0]["text"][:100])
```

### 试一下

```bash
python s02_doc_loading/basic_load.py
```

实测输出:

```
PDF 段落数: 4, DOCX 段落数: 27
PDF 第 1 段前 100 字: 紫光恒越 R3630 G5 双路机架式服
务器
产品白皮书  ·  v1.0  ·  仅用于 RAG 教程测试
一、产品概述
紫光恒越 R3630 G5 是面向企业核心业务、AI 推理与虚拟化负载设计
DOCX 第 1 段前 100 字: 青蓝科技股份有限公司
```

- PDF 拿到 4 段(4 页都有内容); DOCX 拿到 27 段(披露报告里的非空 Word 段落数)
- 两格式返回**同一形状**的 `list[{text, page, source}]`,只 `page` 字段值不同 (PDF: 1/2/3/4, DOCX: None)

**观察**: schema 对齐的价值在两格式上是同一形状 — 下游 s03 chunker 拿到 `pdf` / `docx` 任一边,处理逻辑完全相同,不用 `if source.endswith('.pdf')` 之类的分支判断。`page=None` 在 DOCX 上是**有意的诚实表达** — Word 没有"页"这个结构概念(段落才是结构边界),硬塞 `0` 或 `-1` 当 sentinel 不优雅,下游用 `if page is None` 判断比 `if page` 稳。但 toy 上跑通不代表真实样本上不出问题 — 27 段里没算 DOCX 的 3 张表(见代码 2 量化),4 段 PDF 也没检测多栏错位。

### 为什么不只写这一种

``basic_load.py`` 在 toy / 文本型 PDF 上够用,但在真实样本上暴露出三类固有限制:

- **DOCX 表格被吃掉**:`Document.paragraphs` 不含 `Document.tables`,所有表内文字静默丢弃 — 财报的"主营业务收入构成"表、合同的"违约责任"表,整张蒸发
- **PDF 多栏排版错位**:双栏 PDF 抽出来的文本把左栏底部接到右栏顶部 — embedding 拿到"被切碎又乱拼"的字符串,retrieval top-k 召回碎段
- **扫描件完全没救**:`extract_text()` 对图片型 PDF 返回空字符串,被 `if text.strip()` 静默过滤掉 — 用户以为传了 10 页,实际拿到 0 段

解决方案指向 **代码 2 (量化损失)** + **后续章节填坑** — s03 chunking 缓解 PDF 多栏错位(切片粒度重排让碎段影响被吸收),s11 多模态补全表格抽取 + OCR 兜底。

---

## 代码 2: 真实样本上的失败模式 ([failure_modes.py](failure_modes.py))

### 工作原理

**做一件事**: 把代码 1 的 `load_pdf` / `load_docx` 喂给 `samples/` 下的真实样本 (`server_whitepaper.pdf` 4 页 + `disclosure.docx` 27 段),量化"看不见的损失" — DOCX 表格丢多少字符、PDF 段落长度是否合理,而不修复。

**4 步**:
1. 用 `importlib.util.spec_from_file_location` 加载 `basic_load.py`(目录以数字开头,普通 `import` 报 SyntaxError) — 复用 `load_pdf` / `load_docx` 不重写
2. `show_pdf_failure()`: 跑 `load_pdf` 拿到 4 段,逐段打印 `(page, len, first 60 字)` — 长度 / 内容让"多栏错位"肉眼可见
3. `show_docx_table_loss()`: 直接打开 `Document(path)`,**额外**遍历 `doc.tables` 累加 `cell.text` 长度 — 这部分是 ``basic_load.py`` 的 `load_docx` 没读的
4. 打印 "`basic_load.py` 的 load_docx 只读 paragraphs, 丢失 N 字符 (M 张表)" 量化结论 + 指向 RAGFlow 的 `deepdoc/parser/` 工业解法

```python
# 中间片段: DOCX 表格字符数累加 — `basic_load.py` 没做这部分
table_text_len = sum(
    len(cell.text)
    for tbl in doc.tables
    for row in tbl.rows
    for cell in row.cells
)
print(f"[DOCX] paragraphs(非空)={para_count}, tables={table_count}, 表格内总字符={table_text_len}")
print(f"  → `basic_load.py` 的 load_docx 只读 paragraphs，丢失 {table_text_len} 字符（{table_count} 张表）")
```

**完整函数**:

```python
def show_pdf_failure() -> None:
    """跑 `basic_load.py` 的 load_pdf, 逐段打印 (page, len, first 60 字) — 让多栏错位肉眼可见."""
    pdf = load_pdf(SAMPLES / "server_whitepaper.pdf")
    print(f"[PDF] {len(pdf)} 页抽出的段落 (page, len, first 60 字):")
    for seg in pdf:
        print(f"  page={seg['page']:>2} len={len(seg['text']):>4} | {seg['text'][:60].replace(chr(10), ' ')}")


def show_docx_table_loss() -> None:
    """额外遍历 doc.tables, 量化 `basic_load.py` 的 load_docx 丢多少字符 / 几张表."""
    from docx import Document
    path = SAMPLES / "disclosure.docx"
    doc = Document(path)
    para_count = sum(1 for p in doc.paragraphs if p.text.strip())
    table_count = len(doc.tables)
    table_text_len = sum(
        len(cell.text)
        for tbl in doc.tables
        for row in tbl.rows
        for cell in row.cells
    )
    print(f"\n[DOCX] paragraphs(非空)={para_count}, tables={table_count}, 表格内总字符={table_text_len}")
    print(f"  → `basic_load.py` 的 load_docx 只读 paragraphs，丢失 {table_text_len} 字符（{table_count} 张表）")


def main() -> None:
    show_pdf_failure()
    show_docx_table_loss()
    print("\n→ RAGFlow 的解法: deepdoc/parser/pdf_parser.py 用 XGBoost 版面分析;")
    print("  deepdoc/parser/docx_parser.py 同时遍历 paragraphs + tables")
```

### 试一下

```bash
python s02_doc_loading/failure_modes.py
```

实测输出:

```
[PDF] 4 页抽出的段落 (page, len, first 60 字):
  page= 1 len= 861 | 紫光恒越 R3630 G5 双路机架式服 务器 产品白皮书  ·  v1.0  ·  仅用于 RAG 教程测试 一、产品
  page= 2 len= 562 | 三、整机规格 组件 规格 说明 处理器 2 × 第三代 Intel Xeon 可 扩展处理器 最高 40 核/80 线程
  page= 3 len= 413 | 四、应用场景 云数据中心：作为通用计算节点支撑私有云与混合云平台，配合虚拟化与容器平台提供高 密度的虚拟机/容器实例；典
  page= 4 len= 744 | 五、可靠性与可维护性 冗余设计：电源、风扇、Boot 盘、PCIe 控制器均支持 N+1 冗余；内存支持镜像、备用与 纠

[DOCX] paragraphs(非空)=27, tables=3, 表格内总字符=572
  → `basic_load.py` 的 load_docx 只读 paragraphs，丢失 572 字符（3 张表）
```

- PDF 4 页都有内容 (861/562/413/744 字符) — 长度差异本身就在暗示"页内容密度不均",但**多栏错位藏在字符串内部,长度看不出来**(语义混乱在 embedding 层才暴露)
- DOCX 27 段看着不少,但 3 张表的 572 字符蒸发 — 对财报类文档,这 572 字符往往就是"主营业务收入构成"这种核心结构化数据

**观察**: ``failure_modes.py`` 不生产新数据,只量化损失 — 这是它"demo 性质"的核心:不修,只量。它的存在意义是**教学动机** — 让读者在 s02 就看清"代码 1 在 prod 上不够用,决策要不要跳 s11"。如果不量化,用户会以为 27 段 DOCX = "loader 跑通了",实际关键信息丢了 572 字符。如果你的真实语料 80% 是扫描件 PDF,代码 2 的量化结果就是"代码 1 完全不能用"的直接证据。

### 为什么不只写这一种

``failure_modes.py`` 是"不修只量化"的 demo — 它**故意**不修任何问题,目的就是让你看清 代码 1 的损失边界后再决定要不要跳到工业方案。它自身不解决任何问题:

- **多栏错位未修** — PDF 抽出来的字符串对 RAG 仍然是"被切碎又乱拼"的输入;切片粒度的重排留给 s03 chunking 在 token-cap 级小块上缓解,版面分析 + YOLO 检测留给 s11
- **表格丢失未修** — 572 字符仍在蒸发;s11 多模态抽取(`deepdoc/parser/docx_parser.py` 同时遍历 `paragraphs + tables`)是工业解法,本章不动
- **扫描件未修** — `extract_text()` 对图片型返回空,被静默过滤;s11 的 `VisionParser` + OCR(PaddleOCR / Tesseract / GOT-OCR2)是兜底

**实操建议**: 跑 代码 2 看到扫描件率高 / 表格字符数大 → 直接跳 s11 看 RAGFlow 怎么集成;不需要也不应该在 代码 1 上贴膏药(OCR 50MB~1GB+ 的模型下载成本,不该压在入门骨架里)。

---

## 接下来

s02 是文档加载的**最小骨架 + 失败边界量化**:把"toy RAG"推向"工业 RAG"的第一步 — 没有稳定的文档加载,后面的 chunking / embedding / retrieval 全建立在沙滩上。``basic_load.py`` 把"PDF / DOCX → 统一 `{text, page, source}` schema"做出来,``failure_modes.py`` 把"代码 1 在真实样本上的损失"量化出来 — 这两件事合起来,给出了后续章节的填空入口:

- **PDF 多栏错位** — ``failure_modes.py`` 看到的 861/562/413/744 长度看着正常,但字符串内部语义混乱,embedding 会把碎段当正常段处理。s03 chunking 在 token-cap 级小块上重排,让多栏错位的影响在切片粒度被吸收;s11 版面分析(YOLO / LayoutLMv3)做根因修复
- **DOCX 丢表 572 字符** — 对财报 / 合同这类强结构化文档,丢表 = 核心信息蒸发。s11 `deepdoc/parser/docx_parser.py` 同时遍历 `paragraphs + tables` 是工业解法;本章只量化,不修
- **扫描件静默丢** — ``basic_load.py`` 的 `if text.strip()` 过滤把整页丢掉,用户拿不到任何错误信号。s11 `VisionParser` + OCR(PaddleOCR / Tesseract / GOT-OCR2)兜底;本章识别问题不提供方案

s03 **chunking**: 把 `basic_load.py` 输出的 `{text, page, source}` schema 上的 `text` 单元按"句界 + token cap"切块 — 让 s04 的 embedding 拿到的是局部稠密的小块,而不是一整页的稀疏长文本;同时切片粒度上的重排能缓解 PDF 多栏错位的影响。

---

## 思考题

1. **为什么 PDF 输出会有 4 段（4 页）但 DOCX 输出 27 段？这是"段落"的语义不同吗？**
2. **如果你的真实语料里 80% 是扫描件 PDF，代码 1 的链路对它们返回空字符串。要不要给 代码 1 加一层 OCR 兜底？还是放到 s11 单独做？**

---

## 思考题答案

### Q1. 为什么 PDF 输出会有 4 段（4 页）但 DOCX 输出 27 段？这是"段落"的语义不同吗？

是的，统一 schema 时要分清两种"段落"：

- **PDF 的"段" = 页**(`page` 是结构边界）。`pypdf.extract_text()` 按页返一段字符串——一页内多段被绑在一起。
- **DOCX 的"段" = `\n` 切分的人工段落**(`python-docx` 的 `paragraphs` 是 Word 的 paragraph 元素）。

把 `page` 硬塞给 DOCX 用 `0` 或 `-1` 当 sentinel 都不优雅；**用 `None` 表达"这个 schema 字段对当前格式不适用"更稳**。下游拿到 `page=None` 时知道"不要按页号切片"——比如前端高亮时跳过 `page` 字段。统一 schema 时不要做假数据，直接 `None` 是最诚实的表达。

### Q2. 80% 扫描件，要不要给 代码 1 加 OCR 兜底？

**推荐：放 s11，按需启用，不要进 代码 1。**

判断依据：

1. **资源**：OCR 模型（PaddleOCR / Tesseract / GOT-OCR2）单个 50MB~1GB+；代码 1 是入门骨架，强迫用户下模型 = 把"我想先看看 RAG"的人挡在门外。
2. **依赖复杂度**：OCR 涉及 GPU/CPU 推理、字体映射、CJK 识别率调参——这些都是独立子任务，不是 代码 1 应承担的复杂度。
3. **可观测性**：代码 2 的真实样本上有 80% 扫描件 → 这就是量化结果。它本身已经在告诉你"**代码 1 在这种语料上不够用**，该跳到 s11"。**该跳就跳**，不要在 代码 1 上贴膏药。

RAGFlow 的设计印证了这点：`VisionParser` 在 `deepdoc/parser/utils.py` 里是**单独的 dispatcher**——只有当 `RAGFlowPdfParser` 判定某页是扫描件时才会被调用，主路径不被 OCR 阻塞。

**实操建议**：你跑 代码 2 看到扫描件率高 → 直接到 s11，看 RAGFlow 的 `VisionParser` 怎么集成；不需要也不应该改 代码 1。