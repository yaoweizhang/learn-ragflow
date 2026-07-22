#!/usr/bin/env python3
"""
s11 / unit 01 — 表格抽取：用 pdfplumber 从 PDF 里抓所有非空表格，输出 list[{page, rows}]。

本单元聚焦"结构化表格"这一类多模态输入，
把每个表格的行列按二维 list 原样吐出来——下游 chunking 时通常按行切。

运行: python s11_multimodal/table_extract.py
需要: pip install pdfplumber；samples/server_whitepaper.pdf 至少含 1 张表格

术语速览 (本文件首次出现):
- pdfplumber: 基于 pdfminer.six 的 PDF 解析库,可读到字符/矩形/线条等底层对象
- 表格抽取 (table extraction): 把 PDF 里的二维表格按行列原样抽出来
- extract_tables(): pdfplumber 的启发式画线抽表 API,返回 list[list[cell]]
- 双重空表过滤: 表本身非空 + 至少一行有非空白单元格,启发式排除"页眉伪表格"
- 上下文管理器 (with): `with pdfplumber.open(...) as pdf` 自动关闭文件句柄
"""
from pathlib import Path
import pdfplumber

WORKDIR = Path(__file__).resolve().parents[1]
SAMPLES = WORKDIR / "samples"


def extract_tables(pdf_path: Path) -> list[dict]:
    """遍历 PDF 每页的 extract_tables()，过滤掉完全空的表，返回 [{page, rows}]。"""
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            for t in page.extract_tables():
                # 双重过滤：表本身非空 + 至少有一行有非空白单元格（启发式判"真表"）
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


if __name__ == "__main__":
    main()