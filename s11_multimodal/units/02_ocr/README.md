# s11 / Unit 02 — OCR (pytesseract)

> 由浅入深第 2 步：用 pytesseract + Pillow 把图片里的字（中英混排）抽成字符串。  
> 这是"图像里的字"这一类多模态输入的最小可解——扫描件 / 图片型 PDF 的兜底。

## 这是什么

`ocr_image(image_path)` 用 Pillow 打开图片，调 pytesseract 转交**系统 tesseract
二进制**做识别，返回字符串。`lang="chi_sim+eng"` 同时支持简体中文 + 英文——
够覆盖绝大多数中文 RAG 场景。

pytesseract 只是 Python 壳——**真正的 OCR 引擎是系统二进制 `tesseract`**。
装包忘了装二进制、或装了二进制没装 `chi_sim` 语言包，是 99% 的踩坑来源。

## 跑起来

```bash
# 1. Python 依赖
pip install pytesseract Pillow

# 2. 系统 tesseract 二进制（按平台三选一）
#    Windows: https://github.com/UB-Mannheim/tesseract/wiki 下载安装包，勾上 Chinese (Simplified)
#    macOS:   brew install tesseract tesseract-lang
#    Linux:   sudo apt install tesseract-ocr tesseract-ocr-chi-sim

# 3. 跑脚本（默认无图，按回车跳过；想跑就输入图片绝对路径）
python s11_multimodal/units/02_ocr/code.py
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

## 它做对了什么

- **多语言兜底**：`chi_sim+eng` 同时识别中文 + 英文 + 数字 + 标点；对付
  "中英混排技术文档"够用，不需要写语言探测。
- **标准 Pillow 输入**：`pytesseract.image_to_string(Image.open(path))` 接受
  任何 PIL 支持的格式（PNG / JPG / TIFF / PDF 单帧），pipeline 接入零成本。
- **优雅降级**：缺 `pytesseract` 包 / 缺 tesseract 二进制 / 图不存在——三
  类异常分别 catch，给出针对性提示而不是抛 traceback。

## 它做错了什么

- **没有版面分析**：tesseract 按行扫，**不知道哪段是标题、哪段是正文、哪段
  是表格**；输出是平铺字符串，要下游自己用空白 / 标点切段落。
- **不识别表格结构**：表格单元格的字能读出来，但**行列结构丢了**——
  OCR 输出跟 unit 01 的 `extract_tables` 输出完全不在一个坐标系，没法拼回
  "哪一格是哪一格"。生产里要么走 RAGFlow 的 `TableStructureRecognizer`，要么
  上 PaddleOCR / mineru。
- **依赖系统二进制**：脚本本身不绑 tesseract 版本；服务器部署 / Docker 镜像
  要单独装 `tesseract-ocr` + `tesseract-ocr-chi-sim`，CI 容易漏。
- **大图慢**：单页 3000×4000 扫描件跑 chi_sim+eng 大概要 5–15 秒；
  批量处理要起 multiprocessing 或换 GPU 后端（PaddleOCR / mineru）。
- **中文准确率 80–90%**：低分辨率扫描、字体倾斜、加粗混排都会掉；
  生产通常要 ① 放大 2–3 倍再 OCR；② 加 jieba 分词 + 编辑距离纠错；③ 上视觉 LLM。

## 对照 ragflow 怎么做的

RAGFlow 的 OCR 在 `deepdoc/parser/utils.py` 的 `VisionParser` 类里：
**用户可在 PaddleOCR / mineru / DeepXDE / 自家 ONNX OCR 之间选后端**
（`deepdoc/vision/__init__.py` 统一入口，`deepdoc/parser/pdf_parser.py` L42
`from deepdoc.vision import OCR, Recognizer, TableStructureRecognizer, ...`）。

关键差异：

- **OCR 不是 fallback 而是并行选项**——RAGFlow 在 `pdf_parser.py` L1549–L1575
  检测 pdfplumber 抽出的文本是不是乱码（`_is_garbled_text` / `_is_garbled_by_font_encoding`），
  是才清空、走 OCR；**纯文本 PDF 不会无脑触发 OCR**（无脑触发会让一份
  纯文本 PDF 跑 5–10 倍时间）。
- **per-box OCR**——L778–L790 把每个文本框对应的页面区域裁出来，扔给
  `self.ocr.recognize_batch([b["box_image"] ...])`；**保留 bbox**，可以回写到
  ES chunk metadata。
- **表格走专门 `_table_transformer_job`**——L409–L527 用 `TableStructureRecognizer`
  视觉模型识别 cell 边框，**再对旋转后的表格图重 OCR**，把 OCR 的 bbox 跟 cell
  坐标对齐；表格 OCR 是结构化的，不是单元 02 这种平铺字符串。

vs MVP：`ocr_image` 输出平铺字符串，**没有 bbox、没有版面、没有表格结构**——
MVP 是"能 OCR"，RAGFlow 是"OCR 后能定位回原图 / 原 cell"。

参考：[`ragflow_notes/multimodal_parsing.md`](../../../../ragflow_notes/multimodal_parsing.md)

## 思考题

- **怎么判断一份 PDF 该走 pdfplumber 还是走 OCR？**
  提示：抽前 200 字符，乱码率（连续不可打印字符 / 频次异常的拉丁字符）
  超阈值就走 OCR；或先用 `len(page.chars)` 判文本层是否为空。RAGFlow 的
  `_is_garbled_text` 就是这个套路（`pdf_parser.py` L1559）。