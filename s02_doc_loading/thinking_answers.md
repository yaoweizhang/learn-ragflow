# s02 思考题答案

## Q: 怎么检测某页是扫描件？给一个简单的启发式

**启发式**：

```python
def is_scanned_page(page, min_chars: int = 20) -> bool:
    text = (page.extract_text() or "").strip()
    return len(text) < min_chars
```

判定逻辑 —— 对任意一页：

1. **空字符串 / 纯空白**：`extract_text()` 返回 `""` 或全是空白，说明 PDF 这一页根本没有文本层（典型扫描件）。
2. **字符数过少**：返回了文字但远低于一页该有的信息量（比如 < 20 字符），也是可疑扫描件 —— 可能 PDF 是图片里嵌了几个 OCR 残留字符。
3. 进阶一点可加"图像占比"：如果 `page.images` 数量很多、且 `image.width × image.height` 占页面面积 > 80%，更倾向于扫描件。

判定出来后，**对扫描件不要直接 `skip`，而是标记成 `needs_ocr=True`**，把这一页送进 OCR 流水线（RAGFlow 的 `VisionParser` 就是这么做的 —— 见 [`../docs/reference/ragflow-notes/deepdoc_pdf_parsing.md`](../docs/reference/ragflow-notes/deepdoc_pdf_parsing.md)）。

**为什么这个启发式管用但不够**：靠"字符数"会误伤"文字确实很少的页面"（比如封面、目录页、纯图说明页）。生产里 RAGFlow 用的是更稳的"图像 → OCR → 版面识别"流水线 —— 这正是 **s11 多模态** 会展开的内容：扫描件检测 + OCR + 版面分析是一个完整子任务，不能用一行启发式糊弄过去。