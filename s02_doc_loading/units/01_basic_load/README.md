# s02 / Unit 01 — 最小可跑加载：PDF + DOCX → 统一 schema

> 由浅入深第 1 步：把 PDF 和 DOCX 都读成 `list[{text, page, source}]`，作为后续章节的输入契约。  
> unit 02 会跑同一套函数到真实样本上，演示哪些情况它会崩。

## 这是什么

1. `load_pdf(path)` — `pypdf.PdfReader` 逐页 `extract_text()`，page 从 1 开始；
2. `load_docx(path)` — `python-docx` 按 `paragraphs` 顺序读，仅保留非空段；
3. 输出统一 `{text, page, source}` schema，page 在 DOCX 时为 `None`。

## 跑起来

```bash
python s02_doc_loading/units/01_basic_load/code.py
```

输出：

```
PDF 段落数: 4, DOCX 段落数: 27
PDF 第 1 段前 100 字: 紫光恒越 R3630 G5 双路机架式服务器 产品白皮书 ...
DOCX 第 1 段前 100 字: 青蓝科技股份有限公司 ...
```

## 它做对了什么

- **零依赖外的最小化**：`pypdf` + `python-docx` 覆盖最常见两种格式；
- **schema 一致**：PDF 和 DOCX 喂给下游切块器时是同一种形状。

## 它做错了什么

- **DOCX 表格被吃掉**：`Document.paragraphs` 不含 `tables`，所有表内文字都丢；
- **PDF 多栏排版错位**：双栏 PDF 抽出来的文本会把左栏底部接到右栏顶部；
- **扫描件完全没救**：`extract_text()` 对图片型 PDF 返回空字符串。

## 对照 ragflow 怎么做的

RAGFlow 的 `deepdoc/parser/` 是按文件类型 dispatcher：

- `deepdoc/parser/pdf_parser.py` 用 `pdfplumber` + 自训练 XGBoost 模型（30 特征，详见 `ragflow_notes/deepdoc_chunking.md`）做版面分析，能识别多栏 / 表格 / 图；
- `deepdoc/parser/docx_parser.py` 同时处理 `paragraphs` 和 `tables`；
- `deepdoc/parser/utils.py` 里有 VisionParser 兜底扫描件。

参考：[`ragflow_notes/deepdoc_chunking.md`](../../../../ragflow_notes/deepdoc_chunking.md)

## 思考题

**为什么 PDF 输出会有 4 段（4 页）但 DOCX 输出 27 段？这是"段落"的语义不同吗？**

提示：PDF 的"段" = 页（page 是结构边界）；DOCX 的"段" = `\n` 切分的人工段落。统一 schema 时要不要把"页"硬塞给 DOCX？
