# s11 多模态 — 表格抽取 (pdfplumber) + OCR (tesseract)

> **本章定位**：s11 是 RAG 的"多模态面"。s02 用 `pypdf` 抽文本——遇到两类输入会翻车：① **结构化表格** 被拍扁成一段连续文字，行列结构丢失；② **扫描件 / 图片型 PDF** 根本没有文字层，`pypdf.extract_text()` 返回空字符串。详细定位见 s00 §1.4；RAGFlow 实现见本章末"## RAGFlow 实现"。

---

## 一、章节介绍

**多模态 RAG(Multimodal RAG)** 是把"非纯文本"输入——表格、扫描件、图片、公式、图表——纳入 RAG 检索链路的范式。RAG 主线（s02-s08）默认输入是**有文本层的 PDF / DOCX**；一旦落到扫描件、合并单元格表格、截图、嵌入公式的论文上，传统 `pypdf.extract_text()` 路径就会吐出空字符串或拍扁结构。本章把"多模态"这条侧翼跑通最小骨架：**表格 → list of dicts，图像 → 字符串**，下游 chunking / embedding 还能继续走 s03-s08 那套。把 s11 放进 RAG 全景看：**s11 是 s02 主线的"侧翼补丁"**——给主流程无法处理的"非纯文本"输入一个分支出口，让整条 RAG 链路从"只读有文字层的文档"扩到"读扫描件、读表格、读图片里的字"。

```
                 输入 (PDF / 图片)
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   pypdf 文本层    pdfplumber       pytesseract
   (s02 主线)      extract_tables   image_to_string
        │              │              │
        │ 行列二维      │ OCR 平铺字符串
        │              │
        ▼              ▼              ▼
   list[段]       list[{page,    纯字符串(无坐标)
   s03 chunk      rows=[[...]]}  s03 chunk 按行/段切
        │              │              │
        └──────────────┴──────────────┘
                       ▼
            s04 embedding + s05/s06/s07/s08 复用文本主线
```

> 💡 **一句话总结**：多模态 RAG = 文本解析（s02)+ 表格抽取（s11 中表格那一步）+ OCR(s11 中 OCR 那一步）。
>
> 让 RAG 从"只读有文字层的文档"扩展到"读扫描件、读表格、读图片里的字"——同时让"段落相似度"无法回答的"哪一格是哪一格"暴露出来。

`extract_tables(pdf_path)` + `ocr_image(image_path)` 加一起 40 行就能跑出"表格 → list of dicts"和"图像 → 字符串"。看起来不值得单独一章。但把它放进 s02 的 `pypdf.extract_text()` 对照看会发现：**"文本层"和"非文本层"是两类输入，由 3 类典型失败堆起来**。

### 1.1 多模态相对纯文本 RAG 的边界

多模态不是替代文本 RAG，而是补一道工序。两者的职责清晰分工：

| 输入类型 | s02-s08 主线 | s11 多模态补刀 |
|---|---|---|
| **有文本层的 PDF(可复制粘贴)** | 直接 `pypdf.extract_text()` → chunk → embed | 不需要 |
| **有文本层 + 表格** | 表格被拍扁成一段连续文字,行列结构丢失 | `pdfplumber.extract_tables()` 抽行列(表格那一步) |
| **扫描件 / 图片型 PDF**(无文本层) | `pypdf.extract_text()` 返回空 | `pytesseract.image_to_string()` OCR(OCR 那一步) |
| **图片里的公式 / 图表** | 完全无救 | OCR 也只能读字;公式 / 图表要视觉 LLM(超出本章 MVP) |

> 💡 "多模态"和"文本 RAG"是**互补工序**——前者是后者的预处理插件，不是替代关系。

### 1.2 真实世界的问题（3 条典型）

每条失败模式都对应一种工业级解法，把它们显式暴露出来是 s11 的目标——s11 **不是解决它们，而是把它们显式展示**，让你看到纯 `pypdf` 文本 RAG 的边界。这跟 s10 把"向量召回答不全'实体之间关系'"显式对比是同一种思路——**叙述载体从"图函数 + 1 跳 query"换成"表格函数 + OCR 函数"**，但"先跑通 toy、再讲清楚 toy 在哪里会塌"的教学哲学是一致的。

1. **表格被拍扁——"第三行第二列那个数字是多少"答不出来**——`pypdf` 按行扫文字流，碰到 `|` 分隔的伪表格就直接 `text + "\n"` 拼起来，**行列结构消失**。chunking 时如果把整张表当成一段喂给 embedding，召回段里"数字是 8TB"但模型读不出"这是内存那一格的容量"。**生产解法**：用 `pdfplumber.extract_tables()` 把表格按行列抽出来，按行 chunking 或拼成 markdown 表。RAGFlow 的 `_table_transformer_job` 走得更远——视觉模型识别 cell 边框再 per-cell OCR（见 §四）。
2. **扫描件返回空——整份 PDF 一个字都查不到**——见 [s02 §1.3](../s02_doc_loading/README.md#一3-真实世界的问题) 已识别此问题；本章提供 OCR 解法。很多企业内部 PDF 是扫描件（合同、财务报告、盖章文件），`pypdf.extract_text()` 返回空字符串。整份文档进 embedding 阶段就变成"空索引"——用户问什么都召回不到。**生产解法**：检测到文本层为空时走 OCR fallback。RAGFlow 的 `_is_garbled_text` 检测阈值（`pdf_parser.py` L1559）触发后清空文本层、走 `recognize_batch`（见 §四）。
3. **乱码判定误伤——纯文本 PDF 被强制走 OCR，跑 5–10 倍时间**——这是生产里更隐蔽的坑：subset 字体把 CJK 映射成 ASCII(PUA / font-encoding garbling），看起来是 ASCII 字符、其实是有毒文本层。`_is_garbled_text` 阈值过低会误判，**让一份纯文本 PDF 也跑 5–10 倍 OCR 时间**。**生产解法**：先用一个小样本试探乱码率，再决定要不要全量走 OCR。MVP 不做乱码检测——假设 PDF 是干净的，复杂场景切 RAGFlow（见 §四）。

### 1.3 表格的二维 schema

表格抽取那一步的基础数据长这样——`list[{"page": i, "rows": [[cell, ...], ...]}`：

```python
[
    {"page": 2, "rows": [
        ['组件', '规格', '说明'],
        ['处理器', '2 × 第三代 Intel Xeon 可扩展处理器', '最高 40 核 / 80 线程 ...'],
        ['内存',   '32 × DDR4 3200MHz DIMM',           '最高 8TB ...'],
        ...
    ]},
]
```

第一行通常是表头、后面是数据行；`rows` 是**原样二维 list**，列对齐 + 行顺序不丢——下游可以直接 `pandas.DataFrame(rows)`、按行 chunking、或拼成 markdown 表喂给 LLM。MVP 走"启发式画线 + 行级 chunk"：每条记录一页表，每张表内部按行切。生产里要的是"行 / 列 / 跨格 + 坐标"——RAGFlow `_table_transformer_job` 走 `TableStructureRecognizer` 视觉模型 + per-cell OCR（见 §四）。

> 💡 二维 schema 不是表格抽取独有——它来自 RDBMS 的 `result_set` 概念。多模态 RAG 把这个 schema 套到 PDF 解析上，再交给下游 chunking / embedding。

### 1.4 OCR 的字符串输出 schema

OCR 那一步的基础数据长这样——`str`（平铺）：

```python
"服务器规格\n\n处理器: 2 x 第三代 Intel Xeon 可扩展处理器\n\n内存: 32 x DDR4 3200MHz DIMM\n..."
```

`pytesseract.image_to_string(..., lang="chi_sim+eng")` 返回**纯字符串**——丢掉了每个字的空间位置。后果是：把这段字符串丢进 chunking / embedding pipeline 之后，**你再也回不到 PDF 原文**——用户问"那个数字是多少"，模型答对了，但你无法在 PDF 上高亮 / 无法让用户校验答案对不对。生产里 OCR 输出必须带 `(x, y)` 坐标（用 `image_to_data` 拿 word-level bbox），落到 chunk metadata（见「思考题答案」）。MVP 不带坐标——**只能贴文本，不能点回原页**，是 MVP 到工业最大的鸿沟。

> 💡 平铺字符串 schema 是 OCR 的最低公约数。视觉 LLM(GPT-4V / Qwen-VL）能直接读图 + 出结构化输出，但代价是 GPU + 模型几十 MB + 推理秒级——MVP 不走到那一步。

---

## 二、表格抽取 (pdfplumber)：[c01_table_extract.py](c01_table_extract.py)

> 用 pdfplumber 把 PDF 里的表格按行列原样抽出来。
> 这是"结构化表格"这一类多模态输入的最小可解，下游 chunking 通常按行切。

[`c01_table_extract.py`](c01_table_extract.py)

### 概念

`c01_table_extract.py` 打开 PDF，逐页调用 `page.extract_tables()`，把"至少有一行含非空白单元格"的表收进 `out`，每条记录形如 `{"page": i, "rows": [[cell, ...], ...]}`——一个二维数组，第一行通常是表头、后面是数据行。

`pdfplumber.extract_tables()` 是启发式画线算法：对**带边框 / 带网格线**的规整表格（白皮书规格表、CSV-like 表）很顶；碰到无线表格、跨页表、合并单元格就掉链子。本节重点是"先把基本盘跑通"——MVP 的核心产物就是 list of dicts，下游 chunking / embedding 直接吃。

### 跑一遍

```bash
python s11_multimodal/c01_table_extract.py
```

输出示例（`samples/server_whitepaper.pdf`，`pdfplumber` 0.10+）：

```
PDF 表格数: 1
--- page 2 ---
['', '组件', '规格', '说明']
['处理器', '2 × 第三代 Intel Xeon 可扩展处理器', '最高 40 核 / 80 线程 ...']
['内存', '32 × DDR4 3200MHz DIMM', '最高 8TB ...']
```

### 看输出

样本 PDF 只在 page 2 有 1 张 13×3 的规格表（组件 / 规格 / 说明）；短文本块或扫描 PDF `pdfplumber.extract_tables()` 会返回空列表，跟"识别失败"是不同语义——调用前判 `None` / `len() == 0` 区分这两种。具体 `extract_tables()` 输出的列 / 行内容见上节 `### 跑一遍`。

### 局限与下一步

本段做对了什么 — 把 PDF 里的二维表格按行列结构保留下来,产出的 list of dicts 直接喂给下游 chunking / embedding,不会退化成 "列与列粘连的纯文本"。


- **不处理合并单元格**：pdfplumber 的启发式会把合并 cell 拆成多个相同值，或把空 cell 当成 None；真实报告里"季度合计 / 全行合计"这类合并格经常读错。
- **不识别无线表格**：很多现代 PDF 用空白对齐而不是画线（政府报告、研报），pdfplumber 会当文本读、根本不进 `extract_tables()`。
- **跨页表会断成两半**：白皮书里"长表翻页"很常见，本实现不会合并；真实场景要靠 page 坐标 + 行列结构相似度判定要不要拼。
- **没有表头检测**：第一行被当 header，但白皮书经常有"标题段 + 表格"，pdfplumber 会把标题行吞进表里；真实场景要单独 detect header。
- **只信 pdfplumber 的启发式**：无线表、合并格、艺术化排版全崩；这是接下来 OCR 那步不会修、但生产要修的事。


- **`pdfplumber` 没装**：`pip install pdfplumber`（已在 `requirements.txt`，但单独跑的时候仍需 `pip install`）。
- 样例 PDF 抽不出表：你的样本可能纯文字。试试换成有表格的，或者用 `pdfplumber` 打开后 `page.find_tables()` 看 `len()`。
- 中文字符乱码 / 拆字：pdfplumber 的字符提取基于位置，遇到 CJK 字体子集化（订阅字体、嵌入子集）会有空格插错位或字符截断；升级到 `pdfplumber>=0.10` + 加 `pdfplumber.open(path, password=...)` 兜底。
- 单元格里有 `\n` 换行：本节原样保留；下游 chunking 阶段再 strip。

---


下一章 — 这一节把"召回 → 排序 → 生成 → 服务化"中的某一环跑通,留下 +1 章填下一档的实现;每加一档,缺失上层就越明显,直到 s12 把所有环节收敛到 FastAPI 服务。

## 三、OCR (pytesseract)：[c02_ocr.py](c02_ocr.py)

> 用 pytesseract + Pillow 把图片里的字（中英混排）抽成字符串。
> 这是"图像里的字"这一类多模态输入的最小可解——扫描件 / 图片型 PDF 的兜底。

[`c02_ocr.py`](c02_ocr.py)

### 概念

`c02_ocr.py` 用 Pillow 打开图片，调 pytesseract 转交**系统 tesseract 二进制**做识别，返回字符串。`lang="chi_sim+eng"` 同时支持简体中文 + 英文——够覆盖绝大多数中文 RAG 场景。

pytesseract 只是 Python 壳——**真正的 OCR 引擎是系统二进制 `tesseract`**。装包忘了装二进制、或装了二进制没装 `chi_sim` 语言包，是 99% 的踩坑来源。

### 跑一遍

```bash
# 1. Python 依赖
pip install pytesseract Pillow

# 2. 系统 tesseract 二进制(按平台三选一)
#    Windows: https://github.com/UB-Mannheim/tesseract/wiki 下载安装包,勾上 Chinese (Simplified)
#    macOS:   brew install tesseract tesseract-lang
#    Linux:   sudo apt install tesseract-ocr tesseract-ocr-chi-sim

# 3. 跑脚本(默认无图,按回车跳过;想跑就输入图片绝对路径)
python s11_multimodal/c02_ocr.py
```

输出示例（中文扫描件 + `chi_sim+eng`）：

```
可选: 输入图片路径跑 OCR (回车跳过): /tmp/page.png
服务器规格
处理器: 2 x 第三代 Intel Xeon 可扩展处理器
内存: 32 x DDR4 3200MHz DIMM
...
```

无图片输入时：

```
可选: 输入图片路径跑 OCR (回车跳过):
OCR skipped: 未提供图片路径
```

### 看输出

输入图片路径时会调 `pytesseract.image_to_string(...)`；缺包 / 缺二进制 / 缺图三类异常分别 catch，打印针对性提示而不是 traceback。无图输入默认跳过 + 真扫中文图片两种 trace 见上节 `### 跑一遍`。

### 局限与下一步

本段做对了什么 — 把图片/扫描件里的文字还原成下游可消费的字符串,复用 s02 主线的 `{text, page, source}` schema,让 OCR 之后的 chunking / embedding 走同一条管道。


- **没有版面分析**：tesseract 按行扫，**不知道哪段是标题、哪段是正文、哪段是表格**；输出是平铺字符串，要下游自己用空白 / 标点切段落。
- **不识别表格结构**：表格单元格的字能读出来，但**行列结构丢了**——OCR 输出跟表格抽取那一步的 `extract_tables` 输出完全不在一个坐标系，没法拼回"哪一格是哪一格"。生产里要么走工业 `TableStructureRecognizer`，要么上 PaddleOCR / mineru。
- **依赖系统二进制**：脚本本身不绑 tesseract 版本；服务器部署 / Docker 镜像要单独装 `tesseract-ocr` + `tesseract-ocr-chi-sim`，CI 容易漏。
- **大图慢**：单页 3000×4000 扫描件跑 chi_sim+eng 大概要 5–15 秒；批量处理要起 multiprocessing 或换 GPU 后端（PaddleOCR / mineru）。
- **中文准确率 80–90%**：低分辨率扫描、字体倾斜、加粗混排都会掉；生产通常要 ① 放大 2–3 倍再 OCR；② 加 jieba 分词 + 编辑距离纠错；③ 上视觉 LLM。


- **`pytesseract` 没装**：`pip install pytesseract Pillow`。
- **`TesseractNotFoundError: tesseract is not installed or it's not in your PATH`**：`pytesseract` 只是 Python 壳，**真正的 OCR 引擎是系统二进制 `tesseract`**。
  - Windows：从 https://github.com/UB-Mannheim/tesseract/wiki 下载安装包，安装时勾上"Chinese (Simplified)"语言包；记下安装路径（如 `C:\Program Files\Tesseract-OCR\`）；要么把它加到 PATH，要么在脚本顶部加 `pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"`。
  - macOS：`brew install tesseract tesseract-lang`。
  - Linux：`sudo apt install tesseract-ocr tesseract-ocr-chi-sim`。
- **`OSError: [Errno 2] No such file or directory: 'tesseract'`**：同上，二进制没装。
- **OCR 中文乱码 / 错字**：99% 是没装 `chi_sim` 语言包；`lang="chi_sim+eng"` 跟系统装的语言包必须对得上。
- **`EOFError` when piped**：OCR 那步的 `input()` 在 `< /dev/null` 管道下抛 EOFError——交互模式是主用模式；想脚本化跑就直接传图片路径或改为 `argparse`(MVP 不做）。

---

## 四、核心函数一览

| 函数 | 文件 | 输入 | 输出 | 一句话解释 |
|---|---|---|---|---|
| `extract_tables(pdf_path)` | `c01_table_extract.py` | `Path` | `list[{"page", "rows"}]` | 逐页 `pdfplumber.extract_tables()` + 双重空表过滤(表非空 + 至少一行有非空白 cell) |
| `main()` (表格抽取) | `c01_table_extract.py` | — | 打印表数 + 前 2 张表前 3 行 | 表格抽取演示入口(基于 `samples/server_whitepaper.pdf`) |
| `main()` (OCR) | `c02_ocr.py` | 交互输入图片路径 | 字符串 OR 三类跳过提示 | OCR 演示入口:缺包 / 缺二进制 / 缺图三类异常分别 catch |

环境变量：无 key / 无 LLM / 无网络——本章纯本地计算（表格抽取 CPU-only；OCR 那步调本地系统二进制）。

样本文件：[`samples/server_whitepaper.pdf`](../samples/server_whitepaper.pdf)（紫光恒越 R3630 G5 双路机架式服务器白皮书，page 2 含 1 张 13×3 规格表）。OCR 那步无自带样本图——脚本默认回车跳过；想跑就传任意 PNG/JPG/TIFF 路径。

## 五、跨代码协同

s11 的代码拆得很细，每个函数都对应一种"非纯文本输入"的角色。两个 code 文件的 schema 各管一头，汇到下游走同一条文本管道：

- **表格抽取 schema (`list[{"page", "rows"}]`)**：`page` 是 PDF 页码(从 1 起编)、`rows` 是二维 `list[list[str]]`，**行列结构保留**(列对齐、单元格文本)。这跟 s02 的扁平 `list[{text, page, source}]` 不同：表格不能用一段 `text` 表示,否则 s04 的 BGE 拿到的就是"列与列粘连的纯文本",检索时按行召回会乱。**何时改**：表格行 ≥100、且你需要按列做 schema-aware 召回时,把 `rows` 升级成 `list[{"cells": [...], "header": [...]}]` + 单独的"行索引"字段,下游按列过滤才有可能。
- **OCR schema (`str`)**：纯文本输出,跟 s02 的 `text` 字段**完全一致**——这是 s11 的关键设计：OCR 不引入新 schema,直接复用 s02 的下游。**何时改**：OCR 输出包含版面信息(段落 vs 标题 vs 图注)时,把字符串升级成 `list[{text, role, bbox}]`,下游分块器能按 role 选策略。
- **双重空表过滤策略**：`pdfplumber.extract_tables()` 在"页面有表格区域但全是空白 / 只有边框"的 PDF 上返回 `[[]]`(空表),s11 的 `extract_tables` 用 `if not rows: continue` + `if not any(cell and cell.strip() for cell in row): continue` 双重过滤,只保留"真有数据"的表。**何时改**：当你的语料源大量是空白表格(纯版式文档)时,改单层过滤能减少 30%+ 的伪 chunk。
- **不存图块坐标**：OCR / 表格抽取**不输出 bbox**——s11 的 schema 只到"内容"层,不到"位置"层。代价是 s08 引用时只能给 page + 文本,不能给 bbox;收益是 schema 简单、好序列化。**何时改**：UI 需要"点击答案跳到 PDF 坐标"时,把 `list[{"text", "page", "bbox": [x0, y0, x1, y1]}]` 加上,s05 的 Chroma metadata 多存 `bbox` 字段。
- **上下游契约**：`c01` 输出 `list[{"page", "rows"}]` → 调用方按行切(`for row in rows: row_text = " | ".join(row)`)→ 进 s03 chunking(s02 主线复用);`c02` 输出 `str` → 直接进 s03 chunking(走"行/段"启发式)。两者通过**"纯文本数据流"汇入 s04-s08 主线**,下游不需要分情况处理多模态 vs 纯文本。


## RAGFlow 实现

RAGFlow 的多模态在 `deepdoc/parser/` 里复用视觉模型：表格用 `pdfplumber` / `camelot` 抽行 × 列结构，扫描件用 `vision_parser.py`（视觉模型 + pytesseract 兜底），图片用 `vision_model` 描述生成。多 OCR backend 可切换（pytesseract / paddleocr / 商业 OCR API）。

**设计取舍**：表格当一维文本处理就丢列结构；视觉模型能处理扫描件但贵——RAGFlow 的策略是"能 pdfplumber 抽就 pdfplumber，扫不出来才上 vision"。s11 toy 的"pdfplumber → 失败 → pytesseract"是这条主线的最朴素版。

详细摘录与 5-15 行 "为什么这样写" 的分析见 [`docs/reference/ragflow-notes/multimodal_parsing.md`](../docs/reference/ragflow-notes/multimodal_parsing.md)。

---

## 选型速记

### 主流多模态范式速览

下面这张表把多模态 RAG 系统的实现路径按"表格策略 / OCR 引擎 / 坐标回写 / 视觉后端 / 适用场景"列出来：

| 范式 | 表格策略 | OCR 引擎 | 坐标回写 | 视觉后端 | 适用场景 |
|---|---|---|---|---|---|
| **启发式画线 + 平铺 OCR(本章 MVP)** | `pdfplumber.extract_tables()` | `pytesseract` + 系统 `tesseract` | 无 | 单一(pytesseract) | 教学 / 快速原型 / 干净 PDF |
| **乱码检测触发 OCR + per-box OCR** | 同 MVP(启发式画线) | `pytesseract` 或 `PaddleOCR` | word-level bbox | 可选 PaddleOCR | 生产单租户 / 中等复杂度 |
| **视觉模型表格识别 + bbox 回写(工业 light)** | `TableStructureRecognizer` + per-cell OCR | PaddleOCR / mineru / 自家 ONNX | ES `chunk_count_with_pos` | 多后端可选 | 生产多租户 / 复杂版式 |
| **视觉 LLM 直接读图** | 不分(视觉 LLM 看图) | GPT-4V / Qwen-VL | 自然语言描述 | 视觉 LLM API | 图片 / 公式 / 图表 |

我们的 toy `extract_tables` + `ocr_image` 在范式复杂度上只占第一行——**启发式画线 + 平铺 OCR**；工业方案走完整 light 路径，**多一道抽象就多一道观测点 + 一个失败模式**。教学 demo 选 MVP 因为它跑通快、依赖少、依赖全在 `pdfplumber` / `pytesseract` 里可见；**生产请按"PDF 复杂度 + 是否答带坐标问题 + 速度需求"做 tier 选型**(MVP → 加乱码检测 → light → 视觉 LLM）。

- **教学 / 快速原型 / 干净 PDF** → 本章 MVP(`pdfplumber.extract_tables()` + `pytesseract.image_to_string`），无坐标、无乱码检测、无视觉后端切换，代码 ≤ 50 行；
- **生产单租户 / 中等复杂度 PDF** → 加乱码检测（`_is_garbled_text`)+ word-level bbox(`image_to_data`），切 PaddleOCR 后端，代码 +50 行换 +200% 鲁棒性；
- **生产多租户 / 复杂版式（无线表 / 跨页表 / 合并格）** → 工业 light(`TableStructureRecognizer` + 视觉后端可插拔 + 坐标回写到 ES），加 2 层抽象换"能答'第三行第二列那个数字是多少'"的能力，token 成本翻倍；
- **图片 / 公式 / 图表** → 视觉 LLM(GPT-4V / Qwen-VL），OCR 只读字、读不出趋势线和柱状图，视觉 LLM 能直接"看图说话"，但代价是 GPU + 模型几十 MB + 推理秒级；
- **要先看清每个边界再选** → 用本章表格抽取那一步把"启发式画线"和"扫到无线表 / 跨页表 / 合并格"各跑一次，对比输出——这是最简单的"表格抽取 A/B"实验。

### 扩展指南

加一种新 modality（音频 → 文本 / 视频关键帧 → 文本）只要三步：

1. 写一个 `transcribe_audio(path: Path) -> list[dict]`（用 `whisper` / `faster-whisper`，返回 `[{text, start, end, source}]`）或 `extract_video_frames(path, fps=1) -> list[dict]`（用 `cv2` 抽帧后喂给 `ocr_image`），**返回 dict 形状必须沿用 `{text, source, page/timestamp}`**，下游 s03 chunker / s04 embedder 不改一行；
2. 在 `c01_table_extract.py` 的 `main()` 里加 `if AUDIO_PATH: transcribe_audio(AUDIO_PATH)` / `if VIDEO_PATH: extract_video_frames(VIDEO_PATH)` 分支，**不要**在 `extract_tables` 里写 `if path.suffix == '.mp3': ...`——污染职责；
3. 给代码文件 README 加一段"它跟 PDF 表格抽取比，赢在哪 / 输在哪"的对照（音频：会议纪要 / 时间戳对齐难；视频关键帧：监控 / 视频太大 fps 难选）。

不要把音频 / 视频逻辑塞进 `extract_tables` 或 `ocr_image`——它俩只懂"PDF 页面 + 图像"。本章 MVP 只跑 PDF 表格 + OCR，但 modality 维度留好了挂音频 / 视频不动核心逻辑。

---

## 思考题

1. **怎么把 OCR 结果跟原文段落对齐？**
2. **怎么判断一份 PDF 该走 pdfplumber 还是走 OCR？**
3. **怎么判断表格是"真表"还是"页眉 / 页脚 + 短文本碰巧排成表格形"？**

（答案见文末「思考题答案」）

---

## 思考题答案

### Q1. 怎么把 OCR 结果跟原文段落对齐？

OCR 引擎（无论 tesseract 还是 PaddleOCR）默认返回**纯字符串**——丢掉了每个字的空间位置。后果是：拿这段字符串丢进 chunking / embedding pipeline 之后，**你再也回不到 PDF 原文**：用户问"那个数字是多少"，模型答对了，但你无法在 PDF 上高亮 / 无法让用户校验答案对不对。

解决思路：**OCR 输出框的 `(x, y)` 坐标映射回 PDF 坐标，每个 word / line 都带 page bbox**。

#### 具体三步

**1) OCR 输出改成带坐标的格式**

`tesseract` 有两种坐标输出方式：

- `pytesseract.image_to_data(image, output_type=Output.DICT)` 返回每个 word 的 `level / page_num / block_num / par_num / line_num / word_num / left / top / width / height / text / conf`——这是 word-level 坐标。
- 或者 `pytesseract.image_to_boxes(image)` 返回每个字符的 `(char, left, bottom, right, top)`。

生产里 word-level 颗粒度最常用。每个 word 一条记录：`{"text": "紫光", "x0": 120, "top": 340, "x1": 180, "bottom": 360, "conf": 95.2}`。

**2) 把图像坐标映射回 PDF 页面坐标**

OCR 是在**栅格化的图像**上跑的（这张图可能是 `pdfplumber` 把整页渲染成 300 DPI 的 PNG）。要把图像坐标 `(x_img, y_img)` 变回 PDF 坐标 `(x_pdf, y_pdf)`：

```
scale = dpi / 72       # PDF 默认 72 DPI
x_pdf = x_img / scale
y_pdf = (img_height - y_img) / scale   # 注意 y 轴方向,图像是 top-down,PDF 是 bottom-up
```

更简单的做法：直接用 OCR 的**相对坐标**作为"段落在页面上的哪个区域"，下游存进 chunk metadata 时打 `page_bbox` 标签，**不需要转回绝对 PDF 坐标**——前端拿到 `page=3, bbox=(120,340,180,360)` 直接在 PDF.js 里按比例还原渲染即可。

**3) 段内聚合 + 段落回填**

word-level 坐标拿到后，按 `(block_num, par_num, line_num)` 分组聚成行、聚成段。每段附 `{"page": 3, "bbox": (x0_min, top_min, x1_max, bottom_max)}`。然后：

- 跟 PDF 的文本层做**重叠检测**(IoU > 0.7 视为同一段）——如果 OCR 段跟文本层某段重叠，说明文本层有，OCR 是冗余备份；
- 不重叠的 OCR 段才是"文本层没抽到的内容"——扫描页 / 图片里的字——把它的 bbox 跟 chunk 一起存进 ES。

**4) 存到 chunk metadata，检索时一并返回**

ES 索引时给每个 chunk 多塞两个字段：`page`（页码）、`page_bbox`(`(x0, top, x1, bottom)` 元组）。前端拿到召回结果时，点击 chunk 就能跳转到 PDF 对应位置 + 用 CSS 把那个 bbox 框起来高亮。

### Q2. 怎么判断一份 PDF 该走 pdfplumber 还是走 OCR?

抽前 200 字符，乱码率（连续不可打印字符 / 频次异常的拉丁字符）超阈值（`threshold=0.3`）就走 OCR；或先用 `len(page.chars)` 判文本层是否为空；还可以检测 `_is_garbled_by_font_encoding` 处理 subset 字体把 CJK 映射成 ASCII 的特殊情况。RAGFlow 的 `_is_garbled_text` 就是这个套路（`pdf_parser.py` L1559）。MVP 不做检测——假设 PDF 干净，扫到乱码样本会"静默失败"。

### Q3. 怎么判断表格是"真表"还是"页眉 / 页脚 + 短文本碰巧排成表格形"?

行数（< 2 行大概率是页眉）+ 列数（< 2 列大概率是文本块）+ 单元格内文本长度分布（短文本成片出现可能不是表）+ 边框检测（`page.lines` / `page.rects` 是否构成闭合矩形）。或者直接用 `LayoutRecognizer` 标 `type=="table"` 再信。MVP 走"双重循环 + 至少一行有非空白 cell"启发式——对干净 PDF 够用，对"页眉碰巧排成表格形"会误判。

下一章 — 这一节把"召回 → 排序 → 生成 → 服务化"中的某一环跑通,留下 +1 章填下一档的实现;每加一档,缺失上层就越明显,直到 s12 把所有环节收敛到 FastAPI 服务。

> OCR 引擎安装 (`pytesseract` / `tesseract` 二进制 / `chi_sim` 语言包 / 多 OS 路径) 等细节见 `c02` 的 `### 局限与下一步`。