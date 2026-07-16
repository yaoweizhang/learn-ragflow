#!/usr/bin/env python3
"""
s11 / unit 02 — OCR：用 pytesseract + Pillow 跑中英文 OCR。

本单元聚焦"图像里的字"这一类多模态输入，
把扫描件 / 图片型 PDF 的像素转成可检索文本。

运行: python s11_multimodal/ocr.py
需要: pip install pytesseract Pillow + 系统装 tesseract 二进制 + chi_sim 语言包。
      缺任何一项脚本会优雅跳过，打印提示而不是炸栈。
"""
from pathlib import Path

WORKDIR = Path(__file__).resolve().parents[1]


def main() -> None:
    # 默认无图：演示 tesseract 不可用 / 输入缺失时的优雅跳过路径
    img_path = input("可选: 输入图片路径跑 OCR (回车跳过): ").strip()
    if not img_path:
        print("OCR skipped: 未提供图片路径")
        return
    try:
        import pytesseract
        from PIL import Image
        text = pytesseract.image_to_string(Image.open(Path(img_path)), lang="chi_sim+eng")
    except ImportError:
        print("OCR skipped: pytesseract 未安装，请 `pip install pytesseract Pillow`")
        return
    except pytesseract.TesseractNotFoundError:
        print(
            "OCR skipped: 系统未找到 tesseract 二进制。"
            "Windows: 安装 https://github.com/UB-Mannheim/tesseract/wiki 并加 PATH；"
            "macOS: brew install tesseract tesseract-lang；"
            "Linux: sudo apt install tesseract-ocr tesseract-ocr-chi-sim"
        )
        return
    except FileNotFoundError:
        print(f"OCR skipped: 图片不存在: {img_path}")
        return
    print(text)


if __name__ == "__main__":
    main()
