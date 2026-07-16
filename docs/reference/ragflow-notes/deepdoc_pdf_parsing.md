# RAGFlow PDF 解析

## 一句话
RAGFlow 把 PDF 解析拆成"pdfplumber 拿页面对象 → 栅格化成图 → 视觉模型读文字与坐标"三步，对扫描件 / 复杂版式 / 表格都能兜底；MVP 用 `PdfReader.extract_text` 一条路，对这几种情况全废。

## 来源
- 仓库：https://github.com/infiniflow/ragflow
- 模块：`deepdoc/parser/pdf_parser.py`（`RAGFlowPdfParser` 父类 + `PlainParser` / `VisionParser` / `TxtParser` 子类）
- 关联：本仓库 s02 `basic_load.py`（纯文本 30 行版）、s11 多模态（视觉层兜底）

## 代码

```python
class VisionParser(RAGFlowPdfParser):
    def __init__(self, vision_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vision_model = vision_model
        self.outlines = []

    def __images__(self, fnm, zoomin=3, page_from=0, page_to=MAXIMUM_PAGE_NUMBER, callback=None):
        try:
            with sys.modules[LOCK_KEY_pdfplumber]:
                self.pdf = pdfplumber.open(fnm) if isinstance(fnm, str) else pdfplumber.open(BytesIO(fnm))
                self.page_images = [p.to_image(resolution=72 * zoomin).annotated for i, p in enumerate(self.pdf.pages[page_from:page_to])]
                self.total_page = len(self.pdf.pages)
        except Exception:
            self.page_images = None
            self.total_page = 0
            logging.exception("VisionParser __images__")

    def __call__(self, filename, from_page=0, to_page=MAXIMUM_PAGE_NUMBER, **kwargs):
        callback = kwargs.get("callback", lambda prog, msg: None)
        zoomin = kwargs.get("zoomin", 3)
        self.__images__(fnm=filename, zoomin=zoomin, page_from=from_page, page_to=to_page, callback=callback)
```

## 为什么这样写

- **库选型：`pdfplumber` + 每页栅格化成图**。`pypdf.extract_text()` 对纯文本 PDF 够用，碰到扫描件就只能返回空串。`VisionParser` 用 `pdfplumber.open(...)` 拿到页面对象，再调 `p.to_image(resolution=72 * zoomin).annotated` 把整页渲染成高分辨率 PNG —— 文本层不靠谱，就走视觉层。
- **多后端 + 自动 fallback**。`RAGFlowPdfParser` 是父类，子类有 `PlainParser` / `VisionParser` / `TxtParser` 等多个后端；任务执行器按文档类型 / OCR 探测结果挑一个。`VisionParser.__images__` 把 `LOCK_KEY_pdfplumber` 当全局信号量加锁（`with sys.modules[LOCK_KEY_pdfplumber]`），允许多进程并发而不爆 ONNX runtime 显存；`try/except` 把渲染失败降级成 `page_images=None`，单页炸了不会让整篇文档挂掉。
- **跟 MVP 的差异**。`basic_load.py` 只有 `PdfReader(path).pages[i].extract_text()` 一条路，对扫描件 / 表格 / 页眉页脚毫无办法。RAGFlow 的三阶段：(1) `__images__` 把页面变成 72 × `zoomin` DPI 的图；(2) 视觉模型（OCR + 布局识别 + 表格结构识别）从图里读出文字与坐标；(3) 按 bbox 排序回填成段落。代价：要装 ONNX 模型 + GPU/CPU 推理 + 几十 MB 依赖，换来"扫描件 / 复杂版式 / 表格"全场景能跑。