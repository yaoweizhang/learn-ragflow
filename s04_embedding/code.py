#!/usr/bin/env python3
"""
s04 Embedding — 三种后端：local (BGE) / openai / ollama，默认 local 免 key。

运行: python s04_embedding/code.py
需要: 跑通 s03；首次跑 local 会下载 BAAI/bge-small-zh-v1.5 (~100MB)
"""
import os
import sys
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path(__file__).parent.parent
SAMPLES = WORKDIR / "samples"

# Windows + urllib3>=2.5 + botocore 旧版兼容补丁；新环境不需要
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


def _embed_openai(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    model = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
    resp = client.embeddings.create(input=texts, model=model)
    return [d.embedding for d in resp.data]


def _embed_ollama(texts: list[str]) -> list[list[float]]:
    import requests
    url = os.environ.get("EMBED_BASE_URL", "http://localhost:11434") + "/api/embeddings"
    model = os.environ.get("EMBED_MODEL", "bge-m3")
    return [requests.post(url, json={"model": model, "prompt": t}).json()["embedding"] for t in texts]


def embed(texts: list[str]) -> list[list[float]]:
    provider = os.environ.get("EMBED_PROVIDER", "local")
    return {"local": _embed_local, "openai": _embed_openai, "ollama": _embed_ollama}[provider](texts)


def main() -> None:
    sys.path.insert(0, str(WORKDIR))
    from s03_chunking.code import load_pdf, load_docx, chunk_by_paragraph
    docs = load_pdf(SAMPLES / "server_whitepaper.pdf")[:2] + load_docx(SAMPLES / "disclosure.docx")[:2]
    chunks = chunk_by_paragraph(docs)
    vecs = embed([c["text"] for c in chunks[:4]])
    print(f"维度: {len(vecs[0])}, 前 4 个块的向量数: {len(vecs)}")


if __name__ == "__main__":
    main()