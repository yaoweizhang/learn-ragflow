#!/usr/bin/env python3
"""
s11 多模态 — 表格抽取 (pdfplumber) + OCR (pytesseract)。

运行: python s11_multimodal/code.py
需要: pip install pdfplumber pytesseract Pillow；系统装 tesseract；samples/server_whitepaper.pdf 含表格
"""
from pathlib import Path
import pdfplumber

WORKDIR = Path(__file__).parent.parent
SAMPLES = WORKDIR / "samples"


def extract_tables(pdf_path: Path) -> list[dict]:
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            for t in page.extract_tables():
                if t and any(any(c and c.strip() for c in row) for row in t):
                    out.append({"page": i, "rows": t})
    return out


def ocr_image(image_path: Path) -> str:
    import pytesseract
    from PIL import Image
    return pytesseract.image_to_string(Image.open(image_path), lang="chi_sim+eng")


def main() -> None:
    pdf = SAMPLES / "server_whitepaper.pdf"
    tables = extract_tables(pdf)
    print(f"PDF 表格数: {len(tables)}")
    for t in tables[:2]:
        print(f"--- page {t['page']} ---")
        for row in t["rows"][:3]:
            print(row)
    img_path = input("可选: 输入图片路径跑 OCR (回车跳过): ").strip()
    if img_path:
        print(ocr_image(Path(img_path)))


if __name__ == "__main__":
    main()