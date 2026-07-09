# s02 / Unit 02 — 真实样本上的失败模式

> 由浅入深第 2 步：unit 01 在 toy 上能跑；放到真实 `samples/` 上会崩在哪？  
> 本单元定位问题 + 引出 ragflow 的工业解法。

## 这是什么

把 unit 01 的 `load_pdf` / `load_docx` 喂给真实样本 (`samples/server_whitepaper.pdf` 4 页 + `samples/disclosure.docx` 27 段)，把"看不见的损失"暴露出来：

1. **PDF 多栏错位**——`pypdf.extract_text()` 在双栏 PDF 上按字符位置扫，左右栏会交错；
2. **DOCX 表格丢失**——`python-docx.Document.paragraphs` 不含 `Document.tables`，表内所有文字静默丢弃。

## 跑起来

```bash
python s02_doc_loading/units/02_failure_modes/code.py
```

输出片段：

```
[PDF] 4 页抽出的段落 ...
  page= 1 len= 612 | 紫光恒越 R3630 G5 双路机架式服务器 产品白皮书 ...
  page= 2 len=1024 | 产品型号 产品白皮书 文档版本 ...
  ...

[DOCX] paragraphs(非空)=27, tables=3, 表格内总字符=572
  → unit 01 的 load_docx 只读 paragraphs，丢失 572 字符（3 张表）
```

## 它做对了什么

- **暴露问题**：列出真实样本上的"丢失字符数"和"页段错位"，给后续章节优化提供量化目标；
- **解法对照**：每个失败都点名 ragflow 的对应模块。

## 它做错了什么

- 暂时什么也没"做对"——它的目的就是展示 unit 01 在 prod 上的失败。下一步要么换 loader (s11 表格抽取) 要么换格式（structured extraction）。

## 对照 ragflow 怎么做的

`deepdoc/parser/` 下的两个核心模块直接对应本单元的两个失败：

- **`deepdoc/parser/pdf_parser.py`** —— `RAGFlowPdfParser` 内部用 `pdfplumber` + XGBoost 30 特征模型识别多栏 / 表格 / 图，详见 [`docs/reference/ragflow-notes/deepdoc_chunking.md`](../../../../docs/reference/ragflow-notes/deepdoc_chunking.md)；
- **`deepdoc/parser/docx_parser.py`** —— `DocxParser` 同步遍历 `paragraphs` + `tables`，每段打 `section` 标签（`text` / `table` / `image`）；
- **`deepdoc/parser/utils.py`** —— VisionParser 用 OCR 模型兜底扫描件（s11 会更细讲）。

参考：[`docs/reference/ragflow-notes/multimodal_parsing.md`](../../../../docs/reference/ragflow-notes/multimodal_parsing.md)

## 思考题

**如果你的真实语料里 80% 是扫描件 PDF，unit 01 的链路对它们返回空字符串。要不要给 unit 01 加一层 OCR 兜底？还是放到 s11 单独做？**

提示：放 unit 01 会让所有用户都跑 OCR 模型，下载 1GB+；放 s11 是"按需启用"，更工程化。答案见 [`../../thinking_answers.md`](../../thinking_answers.md)。
