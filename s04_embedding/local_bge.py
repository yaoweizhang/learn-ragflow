#!/usr/bin/env python3
"""
s04 / unit 01 — 本地 BGE Embedding：sentence-transformers 加载 BAAI/bge-small-zh-v1.5，
512 维,输出归一化向量,免 API key。本单元是章节的最小可跑 backbone,
unit 02 的路由分发会把同款本地实现当成 EMBED_PROVIDER=local 的 default。

运行: python s04_embedding/local_bge.py
需要: pip install sentence-transformers；首次运行会从 HF Hub 下载 ~100MB 模型到
~/.cache/huggingface/hub/，离线环境会失败——生产通常构建镜像时预下载并挂
HF_HOME。

术语速览 (本文件首次出现):
- Embedding: 把文本映射成稠密实数向量的过程,语义相近的文本向量距离近
- SentenceTransformer: Hugging Face 的句向量库,一行代码加载预训练模型
- BGE (BAAI General Embedding): 智源研究院开源的中英文通用 embedding 模型族
- HF Hub: Hugging Face 模型仓库,可用 model_id 一键下载模型
- 归一化 (L2 norm): 把向量缩到单位长度,使 cosine 等价于点积
- lru_cache: Python 标准装饰器,缓存函数结果避免重复加载模型
"""
import os
import sys
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path(__file__).resolve().parents[1]
SAMPLES = WORKDIR / "samples"

# Windows + urllib3>=2.5 + botocore 旧版兼容补丁;新环境不需要。
# 必须在任何 sentence_transformers/datasets 间接 import 前强制把
# urllib3.util.ssl_ 加载好,否则补丁路径晚于 botocore 报错就来不及了。
try:
    import urllib3.util.ssl_ as _ssl
    if not hasattr(_ssl, "DEFAULT_CIPHERS"):
        _ssl.DEFAULT_CIPHERS = "DEFAULT@SECLEVEL=2"
except Exception:
    pass


@lru_cache(maxsize=1)
def _local_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5"))


def _embed_local(texts: list[str]) -> list[list[float]]:
    model = _local_model()
    return [v.tolist() for v in model.encode(texts, normalize_embeddings=True)]


def embed_local(texts: list[str]) -> list[list[float]]:
    """对输入文本跑本地 BGE 模型,返回 list[list[float]],每行已归一化。

    用法供外部 import;本单元 main() 也直接调用。
    """
    return _embed_local(list(texts))


def main() -> None:
    sys.path.insert(0, str(WORKDIR))
    # 直接读 PDF/DOCX,不再依赖 s02/s03;让本单元真正 self-contained。
    from pypdf import PdfReader
    from docx import Document

    def _pdf(path: Path) -> list[str]:
        return [(p.extract_text() or "").strip() for p in PdfReader(path).pages if (p.extract_text() or "").strip()]

    def _docx(path: Path) -> list[str]:
        return [p.text.strip() for p in Document(path).paragraphs if p.text.strip()]

    paras = _pdf(SAMPLES / "server_whitepaper.pdf")[:2] + _docx(SAMPLES / "disclosure.docx")[:2]
    chunks = [t for t in paras if t]
    vecs = embed_local(chunks[:4])
    print(f"维度: {len(vecs[0])}, chunks: {len(vecs)}")


if __name__ == "__main__":
    main()
