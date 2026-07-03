# RAGFlow 怎么做: 多模态解析（表格 + OCR fallback）

## 来源
- 仓库: https://github.com/infiniflow/ragflow
- 文件: `deepdoc/parser/pdf_parser.py`
- OCR fallback 关键行: L1549–L1575（garbled 检测触发 OCR）、L778–L790（per-box OCR 渲染与识别）、L786（`self.ocr.recognize_batch`）
- 表格结构识别关键行: L409–L527（`_table_transformer_job`，含 TSR + 旋转 + 坐标回写）
- 视觉后端选型关键行: L42（`from deepdoc.vision import OCR, AscendLayoutRecognizer, LayoutRecognizer, Recognizer, TableStructureRecognizer`）
- commit: `828c5789f651d4c4ebe4645190b8b8d244144fe0`
- 引用日期: 2026-07-04
- GitHub 链接: https://github.com/infiniflow/ragflow/blob/828c5789f651d4c4ebe4645190b8b8d244144fe0/deepdoc/parser/pdf_parser.py

## OCR fallback 关键代码

```python
# L1549–L1575：检测 pdfplumber 抽出的字符是不是"乱码"——是的话清空、走 OCR
for pi, page_ch in enumerate(self.page_chars):
    if not page_ch:
        continue
    sample = page_ch if len(page_ch) <= 200 else page_ch[:200]
    sample_text = "".join(c.get("text", "") for c in sample)
    if self._is_garbled_text(sample_text, threshold=0.3):
        logging.warning(
            "Page %d: pdfplumber extracted mostly garbled characters (%d chars), "
            "clearing to use OCR fallback.",
            page_from + pi + 1, len(page_ch),
        )
        self.page_chars[pi] = []
        continue
    if self._is_garbled_by_font_encoding(page_ch):
        logging.warning(
            "Page %d: detected font-encoding garbled text "
            "(subset fonts with no CJK output, %d chars), "
            "clearing to use OCR fallback.",
            page_from + pi + 1, len(page_ch),
        )
        self.page_chars[pi] = []
```

```python
# L778–L790：把每个文本框对应的页面区域裁出来，扔给 OCR 引擎识别
for b in bxs:
    if not b["text"]:
        if img_np is None:
            img_np = np.asarray(img)
        left, right, top, bott = b["x0"] * ZM, b["x1"] * ZM, b["top"] * ZM, b["bottom"] * ZM
        b["box_image"] = self.ocr.get_rotate_crop_image(img_np, np.array([[left, top], [right, top], [right, bott], [left, bott]], dtype=np.float32))
        boxes_to_reg.append(b)
texts = self.ocr.recognize_batch([b["box_image"] for b in boxes_to_reg], device_id)
```

```python
# L42：把 OCR / 布局 / 表格结构识别作为可替换的"视觉后端"统一导入
from deepdoc.vision import OCR, AscendLayoutRecognizer, LayoutRecognizer, Recognizer, TableStructureRecognizer
```

## 它为什么这样写（3 条）

1. **为什么 RAGFlow 接入 PaddleOCR + mineru + DeepXDE + 自家 ONNX 等多个后端？**
   OCR 不是单一引擎能搞定的事。**生产 PDF 里至少有 4 类烂文本**：① 扫描件（pdfplumber 返回空）→ 必须 OCR；② subset 字体把 CJK 映射成 ASCII（PUA / font-encoding garbling，L1559, L1568）→ 文本层有毒；③ 复杂版式（多栏、表格、图文混排）→ 需要 layout recognizer 给出 bbox；④ 表格里的内容 → 需要 TableStructureRecognizer 识别 cell 边框再 OCR 单元。**不同引擎各有所长**：自家 DeepDoc 的 ONNX OCR 体积小、CPU 可跑；PaddleOCR 中文识别精度高；mineru 在数学公式 / 多栏学术 PDF 上效果好；docling_parser / opendataloader_parser 是云端重型方案。RAGFlow 在 `deepdoc/parser/` 目录里给 PDF 至少挂了 8 个 parser 后端（`pdf_parser.py`, `paddleocr_parser.py`, `mineru_parser.py`, `docling_parser.py`, `opendataloader_parser.py`, `tcadp_parser.py`, `figure_parser.py`），用户按"文档复杂度 + 模型可用性 + 速度需求"挑一个。

2. **为什么表格用专门的 `_table_transformer_job` / `table_rec` 步骤，跟文本抽离？**
   表格的难点是**结构 + 内容**两件事必须分开再合：① 表格的"行 / 列 / 跨格"是 2D 几何关系，普通 OCR 按行扫会把它拍扁成一段连续文字；② 表格单元里的字往往字号小、字密，单独 OCR 精度更高。RAGFlow 的做法：① `LayoutRecognizer` 先把所有 `type=="table"` 的 bbox 圈出来（L437–L479）；② `_table_transformer_job` 把每个表格 crop 出来当一张图；③ 喂给 `TableStructureRecognizer`（`self.tbl_det(imgs)`，L486）拿到 cell 坐标；④ **再对旋转后的表格图重新 OCR**，把 OCR 的 bbox 跟 cell 坐标对齐（L488–L490, L558–L701 的 `_ocr_rotated_tables`）。这样"结构识别"和"内容识别"是两个独立模型的事——结构由 TSR 视觉模型管，内容由 OCR 模型管，互不污染。MVP 的 `pdfplumber.extract_tables()` 用启发式画线算法一行行扫，对规整表格够用，对无线表格 / 跨页表格就崩。

3. **跟 MVP 版本的差异（结构化输出 + 坐标回写）**
   MVP 的 `extract_tables` 只返回 `list[dict]`，每个 dict 形如 `{"page": i, "rows": [[...], [...]]}`——**行列是平铺的二维数组**，没有坐标、没有合并单元格信息、没有"哪一格属于哪一行 / 哪一列"的标注。RAGFlow 的输出（`_table_transformer_job` 之后落到 `self.tb_cpns`）是一个 `list[dict]`，每条记录带：① `pn`（页码）、`layoutno`（本页第几个表）、`table_index`（全局表序号）；② `x0/y0/x1/y1` 的原始页面坐标 + `x0_rotated/y0_rotated/x1_rotated/y1_rotated`（旋转后的图内坐标，因为表格会被旋转 0°/90°/180°/270° 试哪个角度 OCR 得分最高）；③ `label` 取值如 `"table row"` / `"table column"` / `"table spanning cell"` / `"table header"`。**坐标回写的关键意义**：检索时用户问"第三行第二列那个数字是多少"，embedding 召回的是一段文字，**没有坐标就无法高亮 / 无法点回原表 / 无法做 cell-level 对比**。RAGFlow 把坐标直接存到 ES 的 chunk metadata 里（`kb.chunk_count_with_pos` 这类字段），前端就能"点击答案 → 跳回 PDF 对应表格 + 对应单元格"。这是工程上从"能用"到"好用"的关键一步。

## 实操提醒

- **CPU vs GPU**：`self.ocr.recognize_batch(..., device_id)` 的 `device_id` 决定跑哪块 GPU；CPU 跑也行但慢。MVP 30 行不涉及这个。
- **OCR 语言包**：RAGFlow 的 `OCR` 类初始化时会加载中英语言模型（`chi_sim` + `eng`）；生产部署要把 `*.traineddata` 放到 `tessdata/` 目录。MVP 用 `pytesseract` 直接传 `lang="chi_sim+eng"`，本地要有对应 traineddata。
- **乱码判定的代价**：`_is_garbled_text` 是 heuristic，对真正的中文 PDF（不乱码）几乎不会触发；只有 subset 字体 + CID mapping 那类"看起来是 ASCII 但其实是 CJK"的 PDF 才会误判 OCR fallback，**会让一份纯文本 PDF 跑 5–10 倍时间**。生产里通常先用一个小样本试探，再决定要不要全量走 OCR。