# s02 — 文档加载 (PDF + DOCX)

## 问题

`s01` 里的玩具 RAG 用的是手敲字符串。真实场景下，语料长这样：

- 一份 4 页的 PDF 白皮书《紫光恒越 R3630 G5 双路机架式服务器 产品白皮书》，按页加载之后，每页的 `extract_text()` 顺序大概率是**乱**的 —— 多栏排版会把左栏底部的几行接到右栏顶部，行内空格与连字符在 `pypdf` 输出里也是随机的。
- 一份 .docx 披露报告（《青蓝科技股份有限公司 2024 年度财务信息披露报告》），里头有标题、列表、表格、内嵌图。如果用 `python-docx` 按 `paragraphs` 顺序读，表格里的文字会**整体消失**，因为它们存在 `tables` 集合里，不在 `paragraphs` 里。

反例：把这两份文件扔进 `pypdf` / `python-docx`，想"原文照搬"地灌进检索，得到的 chunk 几乎都是错位的、缺表格的。RAG 系统最怕的就是**输入即垃圾**，后面的 embedding 再贵也救不回来。

## 最小解法

把 PDF / DOCX 读成统一形状的 `list[dict]`，每段都带 `text` / `page` / `source`。后续 s03+ 都吃这个接口。

```python
#!/usr/bin/env python3
"""
s02 文档加载 — 把 PDF/DOCX 读成结构化段落，每段带来源和页码。

运行: python s02_doc_loading/code.py
需要: pip install pypdf python-docx；samples/server_whitepaper.pdf + samples/disclosure.docx
"""
from pathlib import Path
from pypdf import PdfReader
from docx import Document

WORKDIR = Path(__file__).parent.parent
SAMPLES = WORKDIR / "samples"


def load_pdf(path: Path) -> list[dict]:
    out = []
    for i, page in enumerate(PdfReader(path).pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            out.append({"text": text, "page": i, "source": path.name})
    return out


def load_docx(path: Path) -> list[dict]:
    out = []
    for p in Document(path).paragraphs:
        if p.text.strip():
            out.append({"text": p.text, "page": None, "source": path.name})
    return out


def main() -> None:
    pdf = load_pdf(SAMPLES / "server_whitepaper.pdf")
    docx = load_docx(SAMPLES / "disclosure.docx")
    print(f"PDF 段落数: {len(pdf)}, DOCX 段落数: {len(docx)}")
    print("PDF 第一段前 100 字:", pdf[0]["text"][:100])


if __name__ == "__main__":
    main()
```

## 跑起来

```bash
pip install pypdf python-docx
python s02_doc_loading/code.py
```

在仓库根目录跑会得到（具体数字由样本文件决定）：

```
PDF 段落数: 4, DOCX 段落数: 27
PDF 第一段前 100 字: 紫光恒越 R3630 G5 双路机架式服\n务器\n产品白皮书  ·  v1.0  ·  仅用于 RAG 教程测试\n一、产品概述
```

4 = 白皮书 PDF 解析出的非空页数（4 页，每页都有内容），27 = 披露报告里非空段落数（3 张表不在 `paragraphs` 里，所以单独数不到）。

**Troubleshooting**：

- Windows GBK 控制台打印第二行可能报 `UnicodeEncodeError: 'gbk' codec can't encode character '\xa9'`。这是控制台编码问题不是代码 bug，跑命令前加 `set PYTHONIOENCODING=utf-8` 即可。
- 如果 `len(pdf) == 0`，说明 PDF 是纯扫描件 —— `pypdf` 抽不出文字。这正是下面"真实世界的问题"第 1 条。

## 真实世界的问题

`pypdf` + `python-docx` 撑得起 demo，扛不住生产。这套 30 行版本有三个明显短板：

1. **扫描件读不出文字**。手机拍的合同、老 PDF 扫描件，`page.extract_text()` 直接返回 `""`。我们用空字符串 `if text.strip()` 直接 skip，结果就是"这份文档没内容"——丢失整篇。
2. **表格 / 图片被拍扁或丢失**。PDF 里的表格，`extract_text()` 输出的顺序依赖绘制顺序，多半把整张表搅成一行行没对齐的碎片；DOCX 里的表格干脆不在 `Document.paragraphs` 里，得另走 `Document.tables` 路径。我们现在的代码完全没处理 `tables`。
3. **页眉 / 页脚污染**。每个 PDF 页都自动带着"第 N 页 / 共 M 页 / 公司名 / 文档编号"，这些重复串会被分块后灌进 embedding，污染相似度匹配 —— 用户问技术问题时，命中的是页眉。

这三条直接连到 s11（多模态）：扫表件要走 OCR，复杂版面要走视觉模型，页眉页脚要走版面识别 + 规则过滤。

## ragflow 怎么做的

RAGFlow 在 `deepdoc/parser/pdf_parser.py` 里维护了一组 PDF 解析器：`PlainParser`（纯文本 fallback）、`VisionParser`（视觉 LLM + OCR）、`TxtParser` 等。完整摘录与 3 条 "为什么这样写" 的分析见 [`ragflow_notes/deepdoc_pdf_parsing.md`](../ragflow_notes/deepdoc_pdf_parsing.md)。

一句话总结：RAGFlow 把"读 PDF"拆成"先把页面栅格化成高分辨率图像 → 视觉模型识别版面/表格/文字 → 按坐标回填段落"三步走，跟我们 `code.py` 的"一行 `extract_text()`"完全是两个量级 —— 它是为扫描件、复杂表格、多栏版面设计的工业级方案。本章只要懂"它存在、它为什么贵"，后面 s11 会真上手调它。

## 思考题

**怎么检测某页是扫描件？给一个简单的启发式。**

参考答案见 [`thinking_answers.md`](./thinking_answers.md)。