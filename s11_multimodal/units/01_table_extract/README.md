# s11 / Unit 01 — 表格抽取 (pdfplumber)

> 由浅入深第 1 步：用 pdfplumber 把 PDF 里的表格按行列原样抽出来。  
> 这是"结构化表格"这一类多模态输入的最小可解，下游 chunking 通常按行切。

## 这是什么

`extract_tables(pdf_path)` 打开 PDF，逐页调用 `page.extract_tables()`，把"至少有
一行含非空白单元格"的表收进 `out`，每条记录形如 `{"page": i, "rows": [[cell, ...], ...]}`
——一个二维数组，第一行通常是表头、后面是数据行。

`pdfplumber.extract_tables()` 是启发式画线算法：对**带边框 / 带网格线**的规整表格
（白皮书规格表、CSV-like 表）很顶；碰到无线表格、跨页表、合并单元格就掉链子。
本单元重点是"先把基本盘跑通"——MVP 的核心产物就是 list of dicts，下游 chunking /
embedding 直接吃。

## 跑起来

```bash
python s11_multimodal/units/01_table_extract/code.py
```

输出示例（`samples/server_whitepaper.pdf`，`pdfplumber` 0.10+）：

```
PDF 表格数: 1
--- page 2 ---
['', '组件', '规格', '说明']
['处理器', '2 × 第三代 Intel Xeon 可扩展处理器', '最高 40 核 / 80 线程 ...']
['内存', '32 × DDR4 3200MHz DIMM', '最高 8TB ...']
```

## 它做对了什么

- **跨页**：逐页循环，把 page 序号写进每条记录，下游能定位回原 PDF。
- **空表过滤**：`pdfplumber.extract_tables()` 偶尔返回 `[["", "", ...]]` 这种"网格
  存在但全是空白"的空表；`any(any(c.strip() ...))` 双重循环把它们丢掉。
- **保留原始结构**：`rows` 是原样二维 list，列对齐、行顺序不丢；下游可以直接
  转 `pandas.DataFrame`、按行 chunking、或拼成 markdown 表喂给 LLM。

## 它做错了什么

- **不处理合并单元格**：pdfplumber 的启发式会把合并 cell 拆成多个相同值，
  或把空 cell 当成 None；真实报告里"季度合计 / 全行合计"这类合并格经常读错。
- **不识别无线表格**：很多现代 PDF 用空白对齐而不是画线（政府报告、研报），
  pdfplumber 会当文本读、根本不进 `extract_tables()`。
- **跨页表会断成两半**：白皮书里"长表翻页"很常见，本实现不会合并；
  真实场景要靠 page 坐标 + 行列结构相似度判定要不要拼。
- **没有表头检测**：第一行被当 header，但白皮书经常有"标题段 + 表格"，
  pdfplumber 会把标题行吞进表里；真实场景要单独 detect header。
- **只信 pdfplumber 的启发式**：无线表、合并格、艺术化排版全崩；
  这是接下来 unit 02 不会修、但生产要修的事。

## 对照 ragflow 怎么做的

RAGFlow 的表格抽取在 `deepdoc/parser/pdf_parser.py` 走的是另一条路——
**视觉模型 `TableStructureRecognizer` + per-cell OCR**：

- 先用 `LayoutRecognizer` 把所有 `type=="table"` 的 bbox 圈出来（L437–L479）。
- `_table_transformer_job` 把表格 crop 成图，喂给 `TableStructureRecognizer`
  识别 cell 边框 + 行列结构（L409–L527）。
- 再对旋转后的表格图重 OCR，把 OCR 的 bbox 跟 cell 坐标对齐（L488–L490,
  `_ocr_rotated_tables`）。
- 最终 `self.tb_cpns` 里每条记录带 `pn / layoutno / table_index` + 原始 +
  旋转后坐标 + `label`（`"table row"` / `"table column"` / `"table spanning cell"` /
  `"table header"`）——**坐标回写到 ES chunk metadata，前端能"点答案跳回原表单元格"**。

vs MVP：`extract_tables` 只返回 `{"page", "rows"}`，**没有坐标、没有合并格
标注、没有 header 标注**——够 chunking，不够"点击答案 → 高亮回原表"。

参考：[`docs/reference/ragflow-notes/multimodal_parsing.md`](../../../../docs/reference/ragflow-notes/multimodal_parsing.md)

## 思考题

- **怎么判断表格是"真表"还是"页眉 / 页脚 + 短文本碰巧排成表格形"？**
  提示：行数、列数、单元格内文本长度分布；或者直接用 `LayoutRecognizer` 标
  `type=="table"` 再信。