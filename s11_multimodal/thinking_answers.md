# s11 思考题答案

## 怎么把 OCR 结果跟原文段落对齐？

OCR 引擎（无论 tesseract 还是 PaddleOCR）默认返回**纯字符串**——丢掉了
每个字的空间位置。后果是：拿这段字符串丢进 chunking / embedding pipeline
之后，**你再也回不到 PDF 原文**：用户问"那个数字是多少"，模型答对了，
但你无法在 PDF 上高亮 / 无法让用户校验答案对不对。

解决思路：**OCR 输出框的 `(x, y)` 坐标映射回 PDF 坐标，每个 word / line
都带 page bbox**。

### 具体三步

**1) OCR 输出改成带坐标的格式**

`tesseract` 有两种坐标输出方式：

- `pytesseract.image_to_data(image, output_type=Output.DICT)` 返回每个 word 的
  `level / page_num / block_num / par_num / line_num / word_num / left / top /
  width / height / text / conf`——这是 word-level 坐标。
- 或者 `pytesseract.image_to_boxes(image)` 返回每个字符的 `(char, left,
  bottom, right, top)`。

生产里 word-level 颗粒度最常用。每个 word 一条记录：`{"text": "紫光",
"x0": 120, "top": 340, "x1": 180, "bottom": 360, "conf": 95.2}`。

**2) 把图像坐标映射回 PDF 页面坐标**

OCR 是在**栅格化的图像**上跑的（这张图可能是 `pdfplumber` 把整页渲染成
300 DPI 的 PNG）。要把图像坐标 `(x_img, y_img)` 变回 PDF 坐标 `(x_pdf,
y_pdf)`：

```
scale = dpi / 72       # PDF 默认 72 DPI
x_pdf = x_img / scale
y_pdf = (img_height - y_img) / scale   # 注意 y 轴方向，图像是 top-down，PDF 是 bottom-up
```

更简单的做法：直接用 OCR 的**相对坐标**作为"段落在页面上的哪个区域"，
下游存进 chunk metadata 时打 `page_bbox` 标签，**不需要转回绝对 PDF
坐标**——前端拿到 `page=3, bbox=(120,340,180,360)` 直接在 PDF.js 里
按比例还原渲染即可。

**3) 段内聚合 + 段落回填**

word-level 坐标拿到后，按 `(block_num, par_num, line_num)` 分组聚成行、
聚成段。每段附 `{"page": 3, "bbox": (x0_min, top_min, x1_max, bottom_max)}`。
然后：

- 跟 PDF 的文本层做**重叠检测**（IoU > 0.7 视为同一段）——如果 OCR 段
  跟文本层某段重叠，说明文本层有，OCR 是冗余备份；
- 不重叠的 OCR 段才是"文本层没抽到的内容"——扫描页 / 图片里的字——
  把它的 bbox 跟 chunk 一起存进 ES。

**4) 存到 chunk metadata，检索时一并返回**

ES 索引时给每个 chunk 多塞两个字段：`page`（页码）、`page_bbox`
（`(x0, top, x1, bottom)` 元组）。前端拿到召回结果时，点击 chunk 就
能跳转到 PDF 对应位置 + 用 CSS 把那个 bbox 框起来高亮。

### RAGFlow 怎么做的

RAGFlow 的 OCR fallback 路径（`pdf_parser.py` L778–L790）给每个 box
的 OCR 结果都带 bbox（`box_image` + `left/top/right/bott` 是从
`LayoutRecognizer` 拿的页面 bbox），最终落到 ES 的 chunk metadata 里。
前端 RAGFlow UI 的"点击答案跳转原文 + 高亮表格单元格"功能，就靠这套
坐标——`_table_transformer_job` 还额外存了 `x0_rotated / x1_rotated`
（表格被旋转后 OCR 的坐标系），因为表格 OCR 要试 4 个旋转角度挑最佳。
这是工程上"能用"到"好用"的关键一步——MVP 不带坐标，所以**只能贴文本，
不能点回原页**。