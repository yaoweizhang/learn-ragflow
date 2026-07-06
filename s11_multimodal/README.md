# s11 多模态 — 表格抽取 (pdfplumber) + OCR (tesseract)

> **章节定位**：RAG 的"多模态面"。s02 用 `pypdf` 抽文本——遇到两类输入会翻车：① **结构化表格** 被拍扁成一段连续文字，行列结构丢失；② **扫描件 / 图片型 PDF** 根本没有文字层，`pypdf.extract_text()` 返回空字符串。这两类"非纯文本"输入是经典 RAG 链路里最容易塌的环节。本章用 2 个 unit 把两类输入的最简可解跑通——**unit 01** 用 `pdfplumber.extract_tables()` 把表格按行列原样抽成 `list[{"page", "rows"}]`，**unit 02** 用 `pytesseract` + 系统 `tesseract` 二进制把图片里的字转成字符串。
>
> **章节结构**：本章用 2 个 unit 走完"结构化表格 → 图像里的字"——**unit 01** 是 CPU-only 的纯 Python PDF 解析（5 秒内跑完），**unit 02** 是 Python 壳 + 系统二进制 OCR（中文 `chi_sim+eng` 兜底；缺 tesseract 二进制时优雅跳过）。
>
> **scope 注意**：本章实现是**启发式画线 + 字符串平铺**——不是 RAGFlow `_table_transformer_job` 的视觉模型识别，也不是 RAGFlow `per-box OCR` 的 bbox 回写。RAGFlow 走的是后者，见 §四。

---

## 章节导航

| Unit | 标题 | 入口 |
| --- | --- | --- |
| 01 | 表格抽取 (pdfplumber)：逐页 `extract_tables()` + 空表过滤 + 行 chunk 友好输出 | [`units/01_table_extract/code.py`](units/01_table_extract/code.py) |
| 02 | OCR (pytesseract)：`chi_sim+eng` 中英混排 + 三类异常优雅跳过（缺包 / 缺二进制 / 缺图） | [`units/02_ocr/code.py`](units/02_ocr/code.py) |

跑法：

```bash
python s11_multimodal/units/01_table_extract/code.py   # 抽表格,基于 samples/server_whitepaper.pdf
python s11_multimodal/units/02_ocr/code.py             # 输入图片路径跑 OCR(回车跳过)
# 旧路径仍可用 (聚合入口,等价于 unit 01):
python s11_multimodal/code.py
```

依赖：`pdfplumber`（已在 requirements.txt）；`pytesseract` + `Pillow`（需 `pip install pytesseract Pillow`）；unit 02 额外要**系统装 `tesseract` 二进制 + `chi_sim` 语言包**——这是踩坑大头，详见 §3.4 troubleshooting。

样本文件：本章用 `samples/server_whitepaper.pdf`（紫光恒越 R3630 G5 双路机架式服务器白皮书，3–5 页，page 2 含 1 张 13×3 规格表）演示 unit 01；unit 02 无自带样本图（脚本默认回车跳过；想跑就传任意 PNG/JPG/TIFF 路径）。

---

## 一、什么是"多模态 RAG"？

### 1.1 核心定义

**多模态 RAG（Multimodal RAG）** 是把"非纯文本"输入——表格、扫描件、图片、公式、图表——纳入 RAG 检索链路的范式。RAG 主线（s02-s08）默认输入是**有文本层的 PDF / DOCX**；一旦落到扫描件、合并单元格表格、截图、嵌入公式的论文上，传统 `pypdf.extract_text()` 路径就会吐出空字符串或拍扁结构。本章把"多模态"这条侧翼跑通最小骨架：**表格 → list of dicts，图像 → 字符串**，下游 chunking / embedding 还能继续走 s03-s08 那套。

> 💡 **一句话总结**：多模态 RAG = 文本解析（s02）+ 表格抽取（s11 unit 01）+ OCR（s11 unit 02）。
>
> 让 RAG 从"只读有文字层的文档"扩展到"读扫描件、读表格、读图片里的字"——同时让"段落相似度"无法回答的"哪一格是哪一格"暴露出来。

多模态不是替代文本 RAG，而是补一道工序。两者的职责清晰分工：

| 输入类型 | s02-s08 主线 | s11 多模态补刀 |
|---|---|---|
| **有文本层的 PDF（可复制粘贴）** | 直接 `pypdf.extract_text()` → chunk → embed | 不需要 |
| **有文本层 + 表格** | 表格被拍扁成一段连续文字，行列结构丢失 | `pdfplumber.extract_tables()` 抽行列（unit 01） |
| **扫描件 / 图片型 PDF**（无文本层） | `pypdf.extract_text()` 返回空 | `pytesseract.image_to_string()` OCR（unit 02） |
| **图片里的公式 / 图表** | 完全无救 | OCR 也只能读字；公式 / 图表要视觉 LLM（超出本章 MVP） |

> 💡 "多模态"和"文本 RAG"是**互补工序**——前者是后者的预处理插件，不是替代关系。

### 1.2 表格的二维 schema

unit 01 的表格抽取基础数据长这样——`list[{"page": i, "rows": [[cell, ...], ...]}`：

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

### 1.3 OCR 的字符串输出 schema

unit 02 的 OCR 基础数据长这样——`str`（平铺）：

```python
"服务器规格\n\n处理器: 2 x 第三代 Intel Xeon 可扩展处理器\n\n内存: 32 x DDR4 3200MHz DIMM\n..."
```

`pytesseract.image_to_string(..., lang="chi_sim+eng")` 返回**纯字符串**——丢掉了每个字的空间位置。后果是：把这段字符串丢进 chunking / embedding pipeline 之后，**你再也回不到 PDF 原文**——用户问"那个数字是多少"，模型答对了，但你无法在 PDF 上高亮 / 无法让用户校验答案对不对。生产里 OCR 输出必须带 `(x, y)` 坐标（用 `image_to_data` 拿 word-level bbox），落到 chunk metadata（见 [thinking_answers.md](./thinking_answers.md)）。MVP 不带坐标——**只能贴文本，不能点回原页**，是 MVP 到工业最大的鸿沟。

> 💡 平铺字符串 schema 是 OCR 的最低公约数。视觉 LLM（GPT-4V / Qwen-VL）能直接读图 + 出结构化输出，但代价是 GPU + 模型几十 MB + 推理秒级——MVP 不走到那一步。

---

## 二、为什么要在 RAG 上显式补"多模态"这一刀？

`extract_tables(pdf_path)` + `ocr_image(image_path)` 加一起 40 行就能跑出"表格 → list of dicts"和"图像 → 字符串"。看起来不值得单独一章。但把它放进 s02 的 `pypdf.extract_text()` 对照看会发现：**"文本层"和"非文本层"是两类输入，由 3 类典型失败堆起来**。

### 2.1 真实世界的问题（3 条典型）

1. **表格被拍扁——"第三行第二列那个数字是多少"答不出来**——`pypdf` 按行扫文字流，碰到 `|` 分隔的伪表格就直接 `text + "\n"` 拼起来，**行列结构消失**。chunking 时如果把整张表当成一段喂给 embedding，召回段里"数字是 8TB"但模型读不出"这是内存那一格的容量"。**生产解法**：用 `pdfplumber.extract_tables()` 把表格按行列抽出来，按行 chunking 或拼成 markdown 表。RAGFlow 的 `_table_transformer_job` 走得更远——视觉模型识别 cell 边框再 per-cell OCR（见 §四）。
2. **扫描件返回空——整份 PDF 一个字都查不到**——很多企业内部 PDF 是扫描件（合同、财务报告、盖章文件），`pypdf.extract_text()` 返回空字符串。整份文档进 embedding 阶段就变成"空索引"——用户问什么都召回不到。**生产解法**：检测到文本层为空时走 OCR fallback。RAGFlow 的 `_is_garbled_text` 检测阈值（`pdf_parser.py` L1559）触发后清空文本层、走 `recognize_batch`（见 §四）。
3. **乱码判定误伤——纯文本 PDF 被强制走 OCR，跑 5–10 倍时间**——这是生产里更隐蔽的坑：subset 字体把 CJK 映射成 ASCII（PUA / font-encoding garbling），看起来是 ASCII 字符、其实是有毒文本层。`_is_garbled_text` 阈值过低会误判，**让一份纯文本 PDF 也跑 5–10 倍 OCR 时间**。**生产解法**：先用一个小样本试探乱码率，再决定要不要全量走 OCR。MVP 不做乱码检测——假设 PDF 是干净的，复杂场景切 RAGFlow（见 §四）。

### 2.2 为什么必须在多模态上显式投入

每条失败模式都对应一种工业级解法——`pdfplumber` 行级 chunking、OCR fallback、视觉模型 + 坐标回写。**s11 的目标不是解决它们，而是把它们显式暴露出来，让你看到纯 `pypdf` 文本 RAG 的边界**。这跟 s10 把"向量召回答不全'实体之间关系'"显式对比是同一种思路——**叙述载体从"图函数 + 1 跳 query"换成"表格函数 + OCR 函数"**，但"先跑通 toy、再讲清楚 toy 在哪里会塌"的教学哲学是一致的。

这也是为什么本章有 2 个 unit 而不是 1 个：

- **unit 01**——跑通"结构化表格"的最小骨架（`extract_tables` + `pdfplumber`），演示"启发式画线 + 行级 chunk"。把"表格抽取"和"OCR"拆成 2 个 unit 是为了让"无外部依赖的纯 Python 解析"和"依赖系统二进制的 OCR 兜底"分两段讲——单步看到表格抽取的稳定性（依赖 `pdfplumber` + 一个 PDF），多步看到 OCR 的脆弱性（依赖 Python 包 + 系统二进制 + 语言包 + 图片本身）。
- **unit 02**——在 unit 01 之上加 `pytesseract` + `Pillow` + 系统 `tesseract` 二进制，演示完整 OCR 流程。**核心是优雅降级**：缺包 / 缺二进制 / 缺图三类异常分别 catch，给针对性提示而不是抛 traceback——这跟 s05 / s09 的 EOFError 处理是同一种"生产可用 vs 教学演示"的分水岭。

---

## 三、怎么做？

### 3.1 章节导航

| Unit | 主题 | 它解决什么 | 对照 RAGFlow |
|---|---|---|---|
| [01_table_extract](./units/01_table_extract/README.md) | `pdfplumber.extract_tables()` + 空表过滤 + 行级 chunk 友好输出 | "表格被 `pypdf` 拍扁" → "按行列原样抽" | `deepdoc/parser/pdf_parser.py` 的 `_table_transformer_job`（L409–L527，视觉 TSR + per-cell OCR + 坐标回写） |
| [02_ocr](./units/02_ocr/README.md) | `pytesseract.image_to_string(..., lang="chi_sim+eng")` + 三类异常优雅跳过 | "扫描件返回空字符串" → "图片里的字转成字符串" | `deepdoc/parser/pdf_parser.py` 的 `_is_garbled_text` 触发 OCR fallback（L1549–L1575）+ per-box OCR（L778–L790） |

### 3.2 跑起来

```bash
pip install pdfplumber pytesseract Pillow   # pdfplumber 已在 requirements.txt；pytesseract + Pillow 见 §3.4 troubleshooting
python s11_multimodal/units/01_table_extract/code.py   # 抽表格
python s11_multimodal/units/02_ocr/code.py             # OCR(默认回车跳过,输入图片路径跑)
# 旧路径仍可用 (聚合入口,等价于 unit 01):
python s11_multimodal/code.py
```

环境变量：无 key / 无 LLM / 无网络——本章纯本地计算（unit 01 CPU-only；unit 02 调本地系统二进制）。

样本文件：[`samples/server_whitepaper.pdf`](../samples/server_whitepaper.pdf)（紫光恒越 R3630 G5 双路机架式服务器白皮书，page 2 含 1 张 13×3 规格表）。unit 02 无自带样本图——脚本默认回车跳过；想跑就传任意 PNG/JPG/TIFF 路径。

### 3.3 核心函数一览

s11 的代码拆得很细，每个函数都对应一种"非纯文本输入"的角色：

| 函数 | 文件 | 输入 | 输出 | 一句话解释 |
|---|---|---|---|---|
| `extract_tables(pdf_path)` | `units/01_table_extract/code.py` | `Path` | `list[{"page", "rows"}]` | 逐页 `pdfplumber.extract_tables()` + 双重空表过滤（表非空 + 至少一行有非空白 cell） |
| `main()` (unit 01) | `units/01_table_extract/code.py` | — | 打印表数 + 前 2 张表前 3 行 | unit 01 演示入口（基于 `samples/server_whitepaper.pdf`） |
| `main()` (unit 02) | `units/02_ocr/code.py` | 交互输入图片路径 | 字符串 OR 三类跳过提示 | unit 02 演示入口：缺包 / 缺二进制 / 缺图三类异常分别 catch |
| 聚合入口 | `code.py`（顶层） | — | 等价于 unit 01 的输出 | 旧路径兼容；内部 `importlib` 加载 unit 01 后跑其 `main()` |

注：`@lru_cache` 模型缓存模式同 s08 unit 01——本章不涉及 LLM，缓存模式**不适用**（无 LLM 调用）。

### 3.4 如何跑 + troubleshooting

**unit 01 跑出来（实测，`samples/server_whitepaper.pdf`，`pdfplumber` 0.10+）：**

```
PDF 表格数: 1
--- page 2 ---
['组件', '规格', '说明']
['处理器', '2 × 第三代 Intel Xeon 可\n扩展处理器', '最高 40 核/80 线程，单核\n最高 3.7GHz，TDP 上限\n270W']
['芯片组', 'Intel C621A', '支持 PCIe 4.0 × 80 lanes']
```

样本 PDF 只在 page 2 有 1 张 13×3 的规格表（组件 / 规格 / 说明）；短文本块或扫描 PDF `pdfplumber.extract_tables()` 会返回空列表，跟"识别失败"是不同语义——调用前判 `None` / `len() == 0` 区分这两种。

**unit 02 跑出来（实测，交互模式默认回车跳过）：**

```
可选: 输入图片路径跑 OCR (回车跳过):
OCR skipped: 未提供图片路径
```

输入图片路径时会调 `pytesseract.image_to_string(...)`；缺包 / 缺二进制 / 缺图三类异常分别 catch，打印针对性提示而不是 traceback。

**Troubleshooting：**

- **`pdfplumber` 没装**：`pip install pdfplumber`（已在 `requirements.txt`，但单独跑 unit 时仍需 `pip install`）。
- **`pytesseract` 没装**：`pip install pytesseract Pillow`。
- **`TesseractNotFoundError: tesseract is not installed or it's not in your PATH`**：`pytesseract` 只是 Python 壳，**真正的 OCR 引擎是系统二进制 `tesseract`**。
  - Windows：从 https://github.com/UB-Mannheim/tesseract/wiki 下载安装包，安装时勾上"Chinese (Simplified)"语言包；记下安装路径（如 `C:\Program Files\Tesseract-OCR\`）；要么把它加到 PATH，要么在脚本顶部加 `pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"`。
  - macOS：`brew install tesseract tesseract-lang`。
  - Linux：`sudo apt install tesseract-ocr tesseract-ocr-chi-sim`。
- **`OSError: [Errno 2] No such file or directory: 'tesseract'`**：同上，二进制没装。
- **OCR 中文乱码 / 错字**：99% 是没装 `chi_sim` 语言包；`lang="chi_sim+eng"` 跟系统装的语言包必须对得上。
- **样例 PDF 抽不出表**：你的样本可能纯文字。试试换成有表格的，或者用 `pdfplumber` 打开后 `page.find_tables()` 看 `len()`。
- **`EOFError` when piped**：`unit 02` 的 `input()` 在 `< /dev/null` 管道下抛 EOFError——交互模式是主用模式；想脚本化跑就直接传图片路径或改为 `argparse`（MVP 不做）。

### 3.5 如何切换到 RAGFlow 风格多模态

加一种多模态策略（视觉 OCR / TableStructureRecognizer / bbox 回写）只要三步：

1. 在 `extract_tables` 之后加 `_table_transformer_job`：先用 `LayoutRecognizer` 圈出 `type=="table"` 的 bbox，再把表格 crop 当图喂给 `TableStructureRecognizer` 识别 cell 边框，最后对旋转后的表格图重 OCR、把 OCR bbox 跟 cell 坐标对齐——把 `{"page", "rows"}` 升级为带 `x0/y0/x1/y1` + `x0_rotated/...` + `label` 的结构化输出（参考 `ragflow_notes/multimodal_parsing.md` §3）；
2. 在 `main()`（unit 02）之前加 `_is_garbled_text` 检测：抽每页前 200 字符算乱码率，超阈值（`threshold=0.3`）就清空文本层走 OCR——避免"纯文本 PDF 被强制跑 5–10 倍时间"的反模式；
3. 把 OCR 输出从 `image_to_string` 改成 `image_to_data` 拿 word-level bbox，落到 chunk metadata 的 `page` + `page_bbox` 字段——前端"点击答案 → 跳回 PDF + 高亮"才有坐标可用（详见 [thinking_answers.md](./thinking_answers.md)）。

不要在 `extract_tables` 里写 `if mode == "pdfplumber": ... elif mode == "tsr": ...` 之类分发——它会污染单一职责。`extract_tables` 只懂启发式画线，`main()` 懂全抽取模式。本章 MVP 只跑启发式画线 + 平铺 OCR，但接口形状留好了。

---

## 四、对照 RAGFlow + 思考题

### 4.1 ragflow 怎么做的

RAGFlow 的多模态模块在 `deepdoc/parser/` 下走"**视觉后端可插拔 + 乱码检测触发 OCR + 表格结构化识别**"三条核心策略。MVP 走的是"启发式画线 + 平铺 OCR"——**3 个最关键的设计决策**：

- **视觉后端可插拔（`deepdoc/vision/__init__.py`）**——RAGFlow 在 `pdf_parser.py` L42 一次性 import 5 个视觉模块（`OCR, AscendLayoutRecognizer, LayoutRecognizer, Recognizer, TableStructureRecognizer`），用户按"文档复杂度 + 模型可用性 + 速度需求"挑后端：自家 DeepDoc ONNX OCR（CPU 可跑、体积小）→ PaddleOCR（中文精度高）→ mineru（数学公式 / 多栏学术 PDF）→ docling_parser / opendataloader_parser（云端重型）。`deepdoc/parser/` 目录至少挂了 8 个 parser 后端（`pdf_parser.py`, `paddleocr_parser.py`, `mineru_parser.py`, `docling_parser.py`, `opendataloader_parser.py`, `tcadp_parser.py`, `figure_parser.py`）。**MVP 只绑 pytesseract**——简单够用，但服务器部署 / Docker 镜像要单独装 `tesseract-ocr` + `tesseract-ocr-chi-sim`，CI 容易漏。
- **乱码检测触发 OCR（`pdf_parser.py` L1549–L1575）**——RAGFlow 在每页文本层抽完后调 `_is_garbled_text(sample_text, threshold=0.3)` 判乱码率，超阈值就 `self.page_chars[pi] = []` 清空走 OCR；同时还有 `_is_garbled_by_font_encoding` 处理 subset 字体把 CJK 映射成 ASCII 的"看起来是 ASCII 但其实有毒"的特殊场景。**关键工程考量**：纯文本 PDF 不会无脑触发 OCR——无脑触发会让一份纯文本 PDF 跑 5–10 倍时间。MVP 不做乱码检测——假设 PDF 干净，扫到乱码样本会"静默失败"。
- **表格结构化识别 + per-cell OCR（`pdf_parser.py` L409–L527, L778–L790）**——RAGFlow 把表格的"行 / 列 / 跨格"和"单元格里的小字"分开识别：① `LayoutRecognizer` 圈 `type=="table"` bbox（L437–L479）；② `_table_transformer_job` 把每张表 crop 成图喂 `TableStructureRecognizer` 识别 cell 边框；③ **对旋转后的表格图重 OCR**，OCR bbox 跟 cell 坐标对齐（`_ocr_rotated_tables`，L488–L490 + L558–L701）；④ 落到 `self.tb_cpns` 时带 `pn / layoutno / table_index` + 原始 / 旋转后坐标 + `label`（`"table row"` / `"table column"` / `"table spanning cell"` / `"table header"`），**坐标回写到 ES chunk metadata，前端能"点答案跳回原表单元格"**。MVP 的 `extract_tables` 只返 `{"page", "rows"}`——没有坐标、没有合并格标注、没有 header 标注——够 chunking，不够"点击答案 → 高亮回原表"。

完整摘录与 3 条"为什么这样"的分析见 [`ragflow_notes/multimodal_parsing.md`](../ragflow_notes/multimodal_parsing.md)。**一句话对比**：RAGFlow 把"多模态"做成**视觉后端可插拔 + 乱码检测触发 OCR + 视觉模型表格结构识别 + 坐标回写**——能处理扫描件 / 无线表 / 跨页表 / 旋转表 / 合并格；**本章 MVP 走"启发式画线 + 平铺 OCR + 优雅跳过"**，**接口形状留好了**，生产按需切。

### 4.2 主流多模态范式速览

下面这张表把多模态 RAG 系统的实现路径按"表格策略 / OCR 引擎 / 坐标回写 / 视觉后端 / 适用场景"列出来：

| 范式 | 表格策略 | OCR 引擎 | 坐标回写 | 视觉后端 | 适用场景 |
|---|---|---|---|---|---|
| **启发式画线 + 平铺 OCR（本章 MVP）** | `pdfplumber.extract_tables()` | `pytesseract` + 系统 `tesseract` | 无 | 单一（pytesseract） | 教学 / 快速原型 / 干净 PDF |
| **乱码检测触发 OCR + per-box OCR** | 同 MVP（启发式画线） | `pytesseract` 或 `PaddleOCR` | word-level bbox | 可选 PaddleOCR | 生产单租户 / 中等复杂度 |
| **视觉模型表格识别 + bbox 回写（RAGFlow light）** | `TableStructureRecognizer` + per-cell OCR | PaddleOCR / mineru / 自家 ONNX | ES `chunk_count_with_pos` | 多后端可选 | 生产多租户 / 复杂版式 |
| **视觉 LLM 直接读图（RAGFlow figure_parser）** | 不分（视觉 LLM 看图） | GPT-4V / Qwen-VL | 自然语言描述 | 视觉 LLM API | 图片 / 公式 / 图表 |

我们的 toy `extract_tables` + `ocr_image` 在范式复杂度上只占第一行——**启发式画线 + 平铺 OCR**；RAGFlow 走完整 light 路径，**多一道抽象就多一道观测点 + 一个失败模式**。教学 demo 选 MVP 因为它跑通快、依赖少、依赖全在 `pdfplumber` / `pytesseract` 里可见；**生产请按"PDF 复杂度 + 是否答带坐标问题 + 速度需求"做 tier 选型**（MVP → 加乱码检测 → RAGFlow light → 视觉 LLM）。

### 4.3 选型速记

- **教学 / 快速原型 / 干净 PDF** → 本章 MVP（`pdfplumber.extract_tables()` + `pytesseract.image_to_string`），无坐标、无乱码检测、无视觉后端切换，代码 ≤ 50 行；
- **生产单租户 / 中等复杂度 PDF** → 加乱码检测（`_is_garbled_text`）+ word-level bbox（`image_to_data`），切 PaddleOCR 后端，代码 +50 行换 +200% 鲁棒性；
- **生产多租户 / 复杂版式（无线表 / 跨页表 / 合并格）** → RAGFlow light（`TableStructureRecognizer` + 视觉后端可插拔 + 坐标回写到 ES），加 2 层抽象换"能答'第三行第二列那个数字是多少'"的能力，token 成本翻倍；
- **图片 / 公式 / 图表** → 视觉 LLM（GPT-4V / Qwen-VL / `figure_parser.py`），OCR 只读字、读不出趋势线和柱状图，视觉 LLM 能直接"看图说话"，但代价是 GPU + 模型几十 MB + 推理秒级；
- **要先看清每个边界再选** → 用本章 unit 01 把"启发式画线"和"扫到无线表 / 跨页表 / 合并格"各跑一次，对比输出——这是最简单的"表格抽取 A/B"实验。

### 4.4 思考题

1. **怎么把 OCR 结果跟原文段落对齐？**  
   答：OCR 引擎（无论 tesseract 还是 PaddleOCR）默认返回**纯字符串**——丢掉了每个字的空间位置。解决思路是 OCR 输出改成带坐标的格式（`pytesseract.image_to_data` 拿 word-level bbox），把图像坐标映射回 PDF 页面坐标（`scale = dpi / 72`，`y` 轴翻转），段内聚合 + 段落回填（按 `(block_num, par_num, line_num)` 分组），存到 chunk metadata 的 `page` + `page_bbox` 字段。详见 [`thinking_answers.md`](./thinking_answers.md)。

2. **怎么判断一份 PDF 该走 pdfplumber 还是走 OCR？**  
   答：抽前 200 字符，乱码率（连续不可打印字符 / 频次异常的拉丁字符）超阈值（`threshold=0.3`）就走 OCR；或先用 `len(page.chars)` 判文本层是否为空；还可以检测 `_is_garbled_by_font_encoding` 处理 subset 字体把 CJK 映射成 ASCII 的特殊情况。RAGFlow 的 `_is_garbled_text` 就是这个套路（`pdf_parser.py` L1559）。MVP 不做检测——假设 PDF 干净，扫到乱码样本会"静默失败"。

3. **怎么判断表格是"真表"还是"页眉 / 页脚 + 短文本碰巧排成表格形"？**  
   答：行数（< 2 行大概率是页眉）+ 列数（< 2 列大概率是文本块）+ 单元格内文本长度分布（短文本成片出现可能不是表）+ 边框检测（`page.lines` / `page.rects` 是否构成闭合矩形）。或者直接用 `LayoutRecognizer` 标 `type=="table"` 再信。MVP 走"双重循环 + 至少一行有非空白 cell"启发式——对干净 PDF 够用，对"页眉碰巧排成表格形"会误判，详见 [unit 01 README](./units/01_table_extract/README.md) 思考题。