# s11 多模态 — 表格抽取 (pdfplumber) + OCR (tesseract)

[上一章 s10 → · 下一章 s12]

> *"多模态 RAG = 文本解析（s02) + 表格抽取（`table_extract.py`) + OCR（`ocr.py`) — 让 RAG 从'只读有文字层的文档'扩到'读扫描件、读表格、读图片里的字'"*
>
> **链路位置**: 离线索引链路 s02 的"侧翼补丁"(给主流程无法处理的"非纯文本"输入一个分支出口)
> **代码文件**: table_extract.py · ocr.py

> 环境准备: 见 root README §快速开始 — `pip install pdfplumber pytesseract Pillow` + 系统装 `tesseract` 二进制 + `chi_sim` 语言包(OCR 那步需要)

---

## 问题

s02 用 `pypdf.extract_text()` 把 PDF 抽成段落,大部分场景够用。但落到两类输入上会直接翻车 — **结构化表格** 和 **扫描件 / 图片型 PDF**:

**第一,结构化表格被拍扁**。`pypdf` 按行扫文字流,碰到 `|` 分隔的伪表格就直接 `text + "\n"` 拼起来,**行列结构消失**。chunking 时把整张表当一段喂给 embedding,召回段里"数字是 8TB"但模型读不出"这是内存那一格的容量"。用户问"第三行第二列那个数字是多少" — 答不出来,因为结构化信息已经被拍扁成连续文本。对财报、合同、规格表这类强结构化文档,**丢表 = 关键信息全丢**,不是"丢 30% 召回率",是"整份文档的核心信息蒸发"。

**第二,扫描件 / 图片型 PDF 整页返回空**。`pypdf.extract_text()` 对纯图像页(扫描件、手机拍的合同、盖章文件)返回 `""`,整页被静默丢弃。用户以为"我传了 10 页 PDF,loader 应该给我 10 段文本",实际拿到 0 段;后面 embedding / retrieval 在空文本上直接抛异常或返回空向量。用户问什么都召回不到 — **整份文档进 embedding 阶段就变成"空索引"**。

**第三,乱码判定误伤 — 纯文本 PDF 被强制走 OCR,跑 5–10 倍时间**。这是生产里更隐蔽的坑:subset 字体把 CJK 映射成 ASCII(PUA / font-encoding garbling),看起来是 ASCII 字符、其实是有毒文本层。`_is_garbled_text` 阈值过低会误判,**让一份纯文本 PDF 也跑 5–10 倍 OCR 时间**。MVP 不做乱码检测 — 假设 PDF 是干净的,复杂场景切 RAGFlow。

把这三种问题合起来看,**多模态 RAG = 文本解析（s02)+ 表格抽取（`table_extract.py`)+ OCR（`ocr.py`)** 三件套互补。前者是后者的预处理插件,不是替代关系;s11 把"表格 → list of dicts,图像 → 字符串"两个最小可解跑通,**让 RAG 从"只读有文字层的文档"扩到"读扫描件、读表格、读图片里的字"**。

s11 的任务就是**把表格抽取 + OCR 两个最小可解跑通**,给"非纯文本"输入一个分支出口,让下游 chunking / embedding 还能复用 s03-s08 那条文本主线。**这一章的目标是看清"非纯文本"这一类输入的解法和边界**,不是替换 s02 — s11 是 s02 主线的"侧翼补丁"。

---

## 解决方案

s11 用 **两个最小可解的脚本** 把"非纯文本"输入的两个主要类型打掉 — 表格和图像。每个脚本解决一类问题,但也都留下新边界。

```
              输入 (PDF / 图片)
                    │
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
  pypdf 文本层   pdfplumber       pytesseract
  (s02 主线)     extract_tables   image_to_string
     │              │              │
     │ 行列二维      │ OCR 平铺字符串
     │              │
     ▼              ▼              ▼
  list[段]      list[{page,    纯字符串(无坐标)
  s03 chunk     rows=[[...]]}  s03 chunk 按行/段切
     │              │              │
     └──────────────┴──────────────┘
                    ▼
         s04 embedding + s05/s06/s07/s08 复用文本主线
```

| 脚本 | 解决什么 | 输出 schema | 留下什么局限 | 何时用 |
|---|---|---|---|---|
| `table_extract.py` | 把 PDF 里的二维表格按行列原样抽出来 | `list[{"page", "rows": [[cell, ...], ...]}]` | 不处理合并格 / 无线表 / 跨页表 / 无表头检测 | 教学 / 干净 PDF(带边框的规格表) |
| `ocr.py` | 把图片 / 扫描件里的字识别成字符串 | `str`(纯文本,无坐标) | 无版面分析 / 不识别表格结构 / 中文 80-90% 准确率 / 依赖系统 tesseract 二进制 | 扫描件 / 图片型 PDF 兜底 |

两脚本的关系是一条**侧翼主干**:代码 1 把"PDF 表格 → 二维 list of dicts"做出来,**行列结构保留**,下游按行 chunking 或拼 markdown 表喂给 LLM;代码 2 把"图片 / 扫描件 → 字符串"做出来,丢掉了每个字的空间位置但保住了可读文本,**复用 s02 主线的下游 chunking / embedding 管道**。两条路径通过"纯文本数据流"汇入 s04-s08 主线,下游不需要分情况处理多模态 vs 纯文本。**每一脚本的局限,都指向 RAGFlow 工业解法的填空入口**(s11 是 MVP,生产方案切 RAGFlow 的 `deepdoc/parser/`)。

---

## 代码 1: 表格抽取 (pdfplumber) ([table_extract.py](table_extract.py))

### 工作原理

**做一件事**: 用 `pdfplumber` 把 PDF 里的表格按行列原样抽出来,产 `list[{"page", "rows": [[cell, ...], ...]}]` — 一个二维数组,第一行通常是表头、后面是数据行,**行列结构保留**,下游 chunking / embedding 直接吃,不会退化成"列与列粘连的纯文本"。

**4 步**:
1. 用 `pdfplumber.open(pdf_path)` 打开 PDF,逐页 `enumerate(pdf.pages, start=1)` 拿 `i`(页码从 1 起编)
2. 对每页调 `page.extract_tables()` — 启发式画线算法:对**带边框 / 带网格线**的规整表格(白皮书规格表、CSV-like 表)很顶
3. **双重空表过滤**:`if t and any(any(c and c.strip() for c in row) for row in t)` — 表本身非空 + 至少一行有非空白单元格(启发式判"真表",去掉页眉碰巧排成表格形 / 纯空白表)
4. 把通过过滤的表收进 `out`,每条记录形如 `{"page": i, "rows": t}`;下游按行切(`for row in rows: row_text = " | ".join(row)`)即可进 s03 chunking

```python
# 中间片段: 双重空表过滤 — 启发式判"真表"
for i, page in enumerate(pdf.pages, start=1):
    for t in page.extract_tables():
        # 双重过滤: 表本身非空 + 至少有一行有非空白单元格
        if t and any(any(c and c.strip() for c in row) for row in t):
            out.append({"page": i, "rows": t})
```

**完整函数**:

```python
def extract_tables(pdf_path: Path) -> list[dict]:
    """遍历 PDF 每页的 extract_tables(),过滤掉完全空的表,返回 [{page, rows}]."""
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            for t in page.extract_tables():
                # 双重过滤: 表本身非空 + 至少有一行有非空白单元格(启发式判"真表")
                if t and any(any(c and c.strip() for c in row) for row in t):
                    out.append({"page": i, "rows": t})
    return out


def main() -> None:
    pdf = SAMPLES / "server_whitepaper.pdf"
    tables = extract_tables(pdf)
    print(f"PDF 表格数: {len(tables)}")
    for t in tables[:2]:
        print(f"--- page {t['page']} ---")
        for row in t["rows"][:3]:
            print(row)
```

### 试一下

```bash
python s11_multimodal/table_extract.py
```

实测输出(`samples/server_whitepaper.pdf`,`pdfplumber` 0.10+):

```
PDF 表格数: 1
--- page 2 ---
['', '组件', '规格', '说明']
['处理器', '2 × 第三代 Intel Xeon 可扩展处理器', '最高 40 核 / 80 线程 ...']
['内存', '32 × DDR4 3200MHz DIMM', '最高 8TB ...']
```

- 样本 PDF 只在 page 2 有 1 张 13×3 的规格表(组件 / 规格 / 说明);`extract_tables()` 直接吐出二维 list,第一行表头,后续数据行原样
- 短文本块或扫描 PDF `extract_tables()` 会返回空列表,跟"识别失败"是不同语义 — 调用前判 `None` / `len() == 0` 区分这两种

**观察**: 这跟 s02 `pypdf` 的拍扁文本完全不同 — `pypdf` 拿到的是"列与列粘连的纯文本"(`列1 列2` 一行流,没有 row boundary),`pdfplumber` 拿到的是**列对齐 + 行顺序不丢**的二维 list,下游可以直接 `pandas.DataFrame(rows)`、按行 chunking、或拼成 markdown 表喂给 LLM。"双重空表过滤"是关键设计 — 对"页面有表格区域但全是空白 / 只有边框"的 PDF,`extract_tables()` 会返回 `[[]]`(空表),``table_extract.py`` 不让它进 `out`,避免下游拿到空表后再做一遍空白判断。但启发式画线算法对无线表 / 跨页表 / 合并格是失效的(见下文),这是 `table_extract.py` 的根本边界。

### 为什么不只写这一种

`pdfplumber.extract_tables()` 是启发式画线算法,只对**带边框 / 带网格线**的规整表格够用。真实语料里三类典型输入会失效:

- **不处理合并单元格**:启发式把合并 cell 拆成多个相同值,或把空 cell 当成 None;真实报告里"季度合计 / 全行合计"这类合并格经常读错
- **不识别无线表格**:很多现代 PDF 用空白对齐而不是画线(政府报告、研报),`pdfplumber` 会当文本读、根本不进 `extract_tables()`
- **跨页表会断成两半**:白皮书里"长表翻页"很常见,本实现不会合并;真实场景要靠 page 坐标 + 行列结构相似度判定要不要拼
- **没有表头检测**:第一行被当 header,但白皮书经常有"标题段 + 表格",`pdfplumber` 会把标题行吞进表里

生产解法 → RAGFlow 的 `_table_transformer_job` 走视觉模型 + per-cell OCR(详见 [`docs/reference/ragflow-notes/multimodal_parsing.md`](../docs/reference/ragflow-notes/multimodal_parsing.md));`table_extract.py` 是 MVP,不修。

---

## 代码 2: OCR (pytesseract) ([ocr.py](ocr.py))

### 工作原理

**做一件事**: 用 pytesseract + Pillow 把图片里的字(中英混排)抽成字符串 — 这是"图像里的字"这一类多模态输入的最小可解,**复用 s02 主线的下游 `{text, page, source}` schema 走 chunking / embedding**。

**3 步**:
1. 交互输入图片路径(`input("可选: 输入图片路径跑 OCR (回车跳过): ").strip()`),无图输入直接 graceful skip 打印 "OCR skipped: 未提供图片路径"
2. 用 Pillow `Image.open(Path(img_path))` 打开图片;调 `pytesseract.image_to_string(img, lang="chi_sim+eng")` 转交**系统 tesseract 二进制**做识别(`lang="chi_sim+eng"` 同时支持简体中文 + 英文)
3. 三类异常分别 catch 并打印针对性提示(不是 traceback):`ImportError`(包没装)/ `TesseractNotFoundError`(系统二进制没装)/ `FileNotFoundError`(图片路径不存在)

```python
# 中间片段: pytesseract 调系统 tesseract + chi_sim+eng
import pytesseract
from PIL import Image
text = pytesseract.image_to_string(
    Image.open(Path(img_path)),
    lang="chi_sim+eng",
)
```

**完整函数**:

```python
def main() -> None:
    # 默认无图: 演示 tesseract 不可用 / 输入缺失时的优雅跳过路径
    img_path = input("可选: 输入图片路径跑 OCR (回车跳过): ").strip()
    if not img_path:
        print("OCR skipped: 未提供图片路径")
        return
    try:
        import pytesseract
        from PIL import Image
        text = pytesseract.image_to_string(
            Image.open(Path(img_path)), lang="chi_sim+eng"
        )
    except ImportError:
        print("OCR skipped: pytesseract 未安装,请 `pip install pytesseract Pillow`")
        return
    except pytesseract.TesseractNotFoundError:
        print(
            "OCR skipped: 系统未找到 tesseract 二进制。"
            "Windows: 安装 https://github.com/UB-Mannheim/tesseract/wiki 并加 PATH;"
            "macOS: brew install tesseract tesseract-lang;"
            "Linux: sudo apt install tesseract-ocr tesseract-ocr-chi-sim"
        )
        return
    except FileNotFoundError:
        print(f"OCR skipped: 图片不存在: {img_path}")
        return
    print(text)
```

### 试一下

```bash
# 1. Python 依赖
pip install pytesseract Pillow

# 2. 系统 tesseract 二进制(按平台三选一)
#    Windows: https://github.com/UB-Mannheim/tesseract/wiki 下载安装包,勾上 Chinese (Simplified)
#    macOS:   brew install tesseract tesseract-lang
#    Linux:   sudo apt install tesseract-ocr tesseract-ocr-chi-sim

# 3. 跑脚本(默认无图,按回车跳过;想跑就输入图片绝对路径)
python s11_multimodal/ocr.py
```

中文扫描件 + `chi_sim+eng` 实测输出(输入图片路径):

```
可选: 输入图片路径跑 OCR (回车跳过): /tmp/page.png
服务器规格

处理器: 2 x 第三代 Intel Xeon 可扩展处理器

内存: 32 x DDR4 3200MHz DIMM
...
```

无图片输入时:

```
可选: 输入图片路径跑 OCR (回车跳过):
OCR skipped: 未提供图片路径
```

- 无图输入默认跳过 + 真扫中文图片两种 trace 都打到上面对应位置
- 三类异常(缺包 / 缺二进制 / 缺图)分别 catch 打印针对性提示,不是 traceback — 让"环境没配好"的读者一眼看到下一步该装啥

**观察**: pytesseract 只是 Python 壳,**真正的 OCR 引擎是系统二进制 `tesseract`**。装包忘了装二进制、或装了二进制没装 `chi_sim` 语言包,是 99% 的踩坑来源 — 这就是为什么 ``ocr.py`` 把 `TesseractNotFoundError` / `ImportError` 都各自 catch 出**针对性提示**(而不是裸 traceback),把"装包路径按 OS 三选一"直接打到输出。**OCR 输出是平铺字符串**(`处理器: 2 x 第三代 ... \n\n 内存: ...`),丢掉了每个字的空间位置 — 拿这段字符串进 chunking / embedding pipeline 之后,你再也回不到 PDF 原文,**只能贴文本,不能点回原页**,是 MVP 到工业最大的鸿沟。

### 为什么不只写这一种

`tesseract` 按行扫,**不知道哪段是标题、哪段是正文、哪段是表格** — 输出是平铺字符串,跟表格抽取的 `extract_tables` 输出完全不在一个坐标系,没法拼回"哪一格是哪一格"。四类典型局限:

- **没有版面分析** — 输出是平铺字符串,要下游自己用空白 / 标点切段落
- **不识别表格结构** — 表格单元格的字能读出来,但**行列结构丢了**;生产要么走工业 `TableStructureRecognizer`,要么上 PaddleOCR / mineru
- **依赖系统二进制** — 服务器部署 / Docker 镜像要单独装 `tesseract-ocr` + `tesseract-ocr-chi-sim`,CI 容易漏
- **中文准确率 80–90%** — 低分辨率扫描、字体倾斜、加粗混排都会掉;生产通常要 ① 放大 2–3 倍再 OCR;② 加 jieba 分词 + 编辑距离纠错;③ 上视觉 LLM

生产解法 → RAGFlow 的 `VisionParser` 多 OCR backend 可切换(pytesseract / paddleocr / 商业 OCR API);`ocr.py` 是 MVP,只跑 pytesseract。

---

## 接下来

s11 是 RAG 全链路的"多模态面" — 把 s02 主线扩到"读扫描件、读表格、读图片里的字"。``table_extract.py`` 把"PDF 表格 → 二维 list of dicts"做出来,``ocr.py`` 把"图片 / 扫描件 → 字符串"做出来 — 这两件事合起来,给出了"非纯文本"输入的最小可解:

- **DOCX / PDF 表格** — ``table_extract.py`` 的 `pdfplumber.extract_tables()` 走启发式画线,对带边框的规格表 / 财报表够用,对无线表 / 跨页表 / 合并格 / 无表头检测是失效的 — RAGFlow 的 `_table_transformer_job` 走视觉模型 + per-cell OCR 是工业解法
- **扫描件 / 图片型 PDF** — ``ocr.py`` 的 `pytesseract.image_to_string(..., lang="chi_sim+eng")` 兜底,把图片里的字还原成下游可消费的字符串;但**只输出平铺字符串,丢掉了每个字的空间位置** — UI 要"点击答案跳到 PDF 坐标"必须升级到带 bbox 的输出(走 `image_to_data` 拿 word-level bbox)
- **乱码判定** — ``table_extract.py`` / ``ocr.py`` 都假设 PDF 是干净的;subset 字体把 CJK 映射成 ASCII 时 `_is_garbled_text` 阈值过低会让一份纯文本 PDF 跑 5–10 倍 OCR 时间 — RAGFlow 在 `pdf_parser.py` L1559 有乱码检测,扫到乱码样本切 OCR 兜底

s12 **服务化**: 把 s02-s11 这 10 步(文档加载 + chunking + embedding + 向量索引 + 检索 + rerank + prompt + agent + graphrag + multimodal)收敛成一条 FastAPI 服务 — 让读者能从"几个独立脚本"走到"可调用的 RAG HTTP 接口"。``table_extract.py`` 的 `extract_tables()` 和 ``ocr.py`` 的 `ocr_image()` 在 s12 里就是两个 dispatcher 入口 — 主流程调度时按"PDF 是文本层还是图像层"走对应路径,**s11 的 schema 直接被 s12 复用**(`list[{"page", "rows"}]` 和 `str` 都进 chunking pipeline)。

---

## 思考题

1. **怎么把 OCR 结果跟原文段落对齐？**
2. **怎么判断一份 PDF 该走 pdfplumber 还是走 OCR？**
3. **怎么判断表格是"真表"还是"页眉 / 页脚 + 短文本碰巧排成表格形"？**

（答案见文末「思考题答案」）

---

## 思考题答案

### Q1. 怎么把 OCR 结果跟原文段落对齐？

OCR 引擎（无论 tesseract 还是 PaddleOCR）默认返回**纯字符串**——丢掉了每个字的空间位置。后果是：拿这段字符串丢进 chunking / embedding pipeline 之后，**你再也回不到 PDF 原文**：用户问"那个数字是多少"，模型答对了，但你无法在 PDF 上高亮 / 无法让用户校验答案对不对。

解决思路：**OCR 输出框的 `(x, y)` 坐标映射回 PDF 坐标，每个 word / line 都带 page bbox**。

#### 具体三步

**1) OCR 输出改成带坐标的格式**

`tesseract` 有两种坐标输出方式：

- `pytesseract.image_to_data(image, output_type=Output.DICT)` 返回每个 word 的 `level / page_num / block_num / par_num / line_num / word_num / left / top / width / height / text / conf`——这是 word-level 坐标。
- 或者 `pytesseract.image_to_boxes(image)` 返回每个字符的 `(char, left, bottom, right, top)`。

生产里 word-level 颗粒度最常用。每个 word 一条记录：`{"text": "紫光", "x0": 120, "top": 340, "x1": 180, "bottom": 360, "conf": 95.2}`。

**2) 把图像坐标映射回 PDF 页面坐标**

OCR 是在**栅格化的图像**上跑的（这张图可能是 `pdfplumber` 把整页渲染成 300 DPI 的 PNG）。要把图像坐标 `(x_img, y_img)` 变回 PDF 坐标 `(x_pdf, y_pdf)`：

```
scale = dpi / 72       # PDF 默认 72 DPI
x_pdf = x_img / scale
y_pdf = (img_height - y_img) / scale   # 注意 y 轴方向,图像是 top-down,PDF 是 bottom-up
```

更简单的做法：直接用 OCR 的**相对坐标**作为"段落在页面上的哪个区域"，下游存进 chunk metadata 时打 `page_bbox` 标签，**不需要转回绝对 PDF 坐标**——前端拿到 `page=3, bbox=(120,340,180,360)` 直接在 PDF.js 里按比例还原渲染即可。

**3) 段内聚合 + 段落回填**

word-level 坐标拿到后，按 `(block_num, par_num, line_num)` 分组聚成行、聚成段。每段附 `{"page": 3, "bbox": (x0_min, top_min, x1_max, bottom_max)}`。然后：

- 跟 PDF 的文本层做**重叠检测**(IoU > 0.7 视为同一段）——如果 OCR 段跟文本层某段重叠，说明文本层有，OCR 是冗余备份；
- 不重叠的 OCR 段才是"文本层没抽到的内容"——扫描页 / 图片里的字——把它的 bbox 跟 chunk 一起存进 ES。

**4) 存到 chunk metadata，检索时一并返回**

ES 索引时给每个 chunk 多塞两个字段：`page`（页码）、`page_bbox`(`(x0, top, x1, bottom)` 元组）。前端拿到召回结果时，点击 chunk 就能跳转到 PDF 对应位置 + 用 CSS 把那个 bbox 框起来高亮。

### Q2. 怎么判断一份 PDF 该走 pdfplumber 还是走 OCR?

抽前 200 字符，乱码率（连续不可打印字符 / 频次异常的拉丁字符）超阈值（`threshold=0.3`）就走 OCR；或先用 `len(page.chars)` 判文本层是否为空；还可以检测 `_is_garbled_by_font_encoding` 处理 subset 字体把 CJK 映射成 ASCII 的特殊情况。RAGFlow 的 `_is_garbled_text` 就是这个套路（`pdf_parser.py` L1559）。MVP 不做检测——假设 PDF 干净，扫到乱码样本会"静默失败"。

### Q3. 怎么判断表格是"真表"还是"页眉 / 页脚 + 短文本碰巧排成表格形"?

行数（< 2 行大概率是页眉）+ 列数（< 2 列大概率是文本块）+ 单元格内文本长度分布（短文本成片出现可能不是表）+ 边框检测（`page.lines` / `page.rects` 是否构成闭合矩形）。或者直接用 `LayoutRecognizer` 标 `type=="table"` 再信。MVP 走"双重循环 + 至少一行有非空白 cell"启发式,对干净 PDF 够用,对"页眉碰巧排成表格形"会误判。


> OCR 引擎安装 (`pytesseract` / `tesseract` 二进制 / `chi_sim` 语言包 / 多 OS 路径) 等细节见 ``ocr.py`` 的 `### 局限与下一步`。