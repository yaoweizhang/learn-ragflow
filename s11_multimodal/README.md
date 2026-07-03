# s11 多模态 — 表格抽取 (pdfplumber) + OCR (tesseract)

## 问题

s02 的 `pypdf` 把表格拍扁成一段连续文字、丢掉了行列；扫描件 / 图片型 PDF
根本没有文字层，`pypdf.extract_text()` 返回空字符串。这两类"多模态"输入——
**结构化表格**和**图像里的字**——是经典 RAG 链路最容易翻车的地方。

## 最小解法

`s11_multimodal/code.py`：

```python
def extract_tables(pdf_path):
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            for t in page.extract_tables():
                if t and any(any(c and c.strip() for c in row) for row in t):
                    out.append({"page": i, "rows": t})
    return out


def ocr_image(image_path):
    import pytesseract
    from PIL import Image
    return pytesseract.image_to_string(Image.open(image_path), lang="chi_sim+eng")
```

两个函数互相独立：`extract_tables` 只吃 PDF，输出 `list[{"page", "rows"}]`；
`ocr_image` 只吃图片路径，输出字符串。可以分别接进 chunking pipeline——
表格用行做 chunk，OCR 文本按段落再切。

## 跑起来

```bash
# 1. 装依赖（pdfplumber 已装；pytesseract + Pillow 见 README troubleshooting）
pip install pdfplumber pytesseract Pillow

# 2. 跑脚本
python s11_multimodal/code.py
```

实测（`samples/server_whitepaper.pdf`，`pdfplumber` 0.11.9）：

```
PDF 表格数: 38
--- page 7 ---
['', '机型', '', '', 'SFF', '', '', 'LFF', '']
--- page 10 ---
['', '序号', '', '', '含义', '']
```

（项目笔记说 25 张，`pdfplumber.extract_tables()` 比预期更激进，把很多
小区域也当表格；不影响下游使用——下游会按"行数 × 列数"过滤太碎的。）

OCR 部分（可选）：脚本最后会问"输入图片路径跑 OCR"，直接回车跳过；
想跑需要先装系统 tesseract 二进制（见下方"troubleshooting"）。

### troubleshooting

- **`pdfplumber` 没装**: `pip install pdfplumber`。
- **`pytesseract` 没装**: `pip install pytesseract Pillow`。
- **`TesseractNotFoundError: tesseract is not installed or it's not in your PATH`**:
  `pytesseract` 只是 Python 壳，**真正的 OCR 引擎是系统二进制 `tesseract`**。
  - Windows: 从 https://github.com/UB-Mannheim/tesseract/wiki 下载安装包，
    安装时勾上"Chinese (Simplified)"语言包；记下安装路径（如
    `C:\Program Files\Tesseract-OCR\`）；要么把它加到 PATH，要么在脚本顶部加
    `pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"`。
  - macOS: `brew install tesseract tesseract-lang`。
  - Linux: `sudo apt install tesseract-ocr tesseract-ocr-chi-sim`。
- **`OSError: [Errno 2] No such file or directory: 'tesseract'`**: 同上，二进制没装。
- **OCR 中文乱码 / 错字**: 99% 是没装 `chi_sim` 语言包；`lang="chi_sim+eng"`
  跟系统装的语言包必须对得上。
- **样例 PDF 抽不出表**: 你的样本可能纯文字。试试换成有表格的，或者
  用 `pdfplumber` 打开后 `page.find_tables()` 看 `len()`。

## 真实世界的问题

1. **OCR 错字**——Tesseract 在中文场景准确率大概 80–90%；扫描分辨率低、
   字体倾斜、加粗混排都会掉。生产里通常要：① 把图放大 2–3 倍再 OCR；
   ② 加语言模型后处理（用 jieba 分词 + 编辑距离纠错）；③ 上更强的引擎
   （PaddleOCR / mineru / 视觉 LLM），但代价是 GPU + 模型几十 MB。
2. **表格结构识别失败**——`pdfplumber.extract_tables()` 用启发式画线检测，
   碰到无线表格、跨页表格、合并单元格就会丢列或合并行。生产里要上视觉
   模型（TableStructureRecognizer / TableNet）单独识别 cell 边框，再把
   OCR 出来的文字贴回对应 cell。RAGFlow 的 `_table_transformer_job` 就是
   这个套路（见 [ragflow_notes/multimodal_parsing.md](../ragflow_notes/multimodal_parsing.md)）。
3. **多语言混排**——中英文混排 + 数字 + 标点，`lang="chi_sim+eng"` 是
   兜底，但碰上繁体 / 日文 / 韩文就歇菜。生产要给每个文档探测主语种
   （用 `langdetect` 或首 200 字符扔给 LLM 判），动态切语言包。
4. **图片 / 公式 / 图表**——OCR 只读字，**图里的趋势线、柱状图、公式
   完全没救**。要靠视觉 LLM（GPT-4V / Qwen-VL）"看图说话"。这是 RAGFlow
   `figure_parser.py` 干的事，超出本章 MVP 范围。

## ragflow 怎么做的

见 [ragflow_notes/multimodal_parsing.md](../ragflow_notes/multimodal_parsing.md)。
要点：RAGFlow 把"读 PDF"拆成三个阶段——
① 文本层先抽（pdfplumber）；② 检测到乱码 / 扫描页就 fallback 到 OCR
（自研 ONNX OCR / PaddleOCR / mineru 任选一个后端）；③ 表格单独走
`TableStructureRecognizer` 视觉模型 + 重 OCR，输出带 cell 坐标。

## 思考题

- **怎么把 OCR 结果跟原文段落对齐？**
  答：见 `thinking_answers.md`。