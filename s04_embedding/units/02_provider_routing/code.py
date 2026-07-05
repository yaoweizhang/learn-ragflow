#!/usr/bin/env python3
"""
s04 / unit 02 — Provider 路由:按 EMBED_PROVIDER 环境变量在三个后端之间分发。

本单元故意不复用 unit 01 的 BGE 实现——分发层独立,方便单测;如果选
EMBED_PROVIDER=local,本单元会直接 import sentence-transformers 跑同一模型,
行为和 unit 01 一致,但代码不依赖。

ENV:
  EMBED_PROVIDER    local(default) | openai | ollama
  LLM_API_KEY       openai 必填;base_url 兼容 OpenAI 协议,默认 api.openai.com/v1
  LLM_BASE_URL      openai 兼容端点(可指向 azure / proxy)
  EMBED_BASE_URL    ollama 端点,默认 http://localhost:11434
  EMBED_MODEL       openai 默认 text-embedding-3-small,ollama 默认 bge-m3

运行: python s04_embedding/units/02_provider_routing/code.py
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

DEMOS = [
    "什么是 RAG?",
    "Retrieval-Augmented Generation",
    "Embedding 把句子变向量。",
]


def _embed_local(texts: list[str]) -> list[list[float]]:
    """EMBED_PROVIDER=local 时跑 sentence-transformers——和 unit 01 同款,但独立实现。"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5"))
    return [v.tolist() for v in model.encode(list(texts), normalize_embeddings=True)]


def embed_openai(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    model = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
    resp = client.embeddings.create(input=list(texts), model=model)
    return [d.embedding for d in resp.data]


def embed_ollama(texts: list[str]) -> list[list[float]]:
    import requests
    url = os.environ.get("EMBED_BASE_URL", "http://localhost:11434") + "/api/embeddings"
    model = os.environ.get("EMBED_MODEL", "bge-m3")
    return [requests.post(url, json={"model": model, "prompt": t}).json()["embedding"] for t in texts]


_REGISTRY = {
    "local": _embed_local,
    "openai": embed_openai,
    "ollama": embed_ollama,
}


def route(texts: list[str]) -> tuple[str, list[list[float]]]:
    """按 EMBED_PROVIDER 选 backend,返回 (provider_name, vectors)。"""
    provider = os.environ.get("EMBED_PROVIDER", "local")
    fn = _REGISTRY[provider]
    return provider, fn(texts)


def _openai_available() -> bool:
    return bool(os.environ.get("LLM_API_KEY"))


def _ollama_available() -> bool:
    host = os.environ.get("EMBED_BASE_URL", "http://localhost:11434")
    try:
        import requests
        r = requests.get(host + "/api/tags", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def main() -> None:
    provider, vecs = route(DEMOS)
    print(f"provider: {provider}, dim: {len(vecs[0])}, count: {len(vecs)}")

    if not _openai_available():
        print("[openai] skipped, set LLM_API_KEY (and LLM_BASE_URL) to enable")
    else:
        provider2, vecs2 = route([DEMOS[0]])
        print(f"[openai] ok: provider={provider2}, dim={len(vecs2[0])}")

    if not _ollama_available():
        print(f"[ollama] skipped, set EMBED_BASE_URL and run `ollama serve` to enable")
    else:
        provider3, vecs3 = route([DEMOS[0]])
        print(f"[ollama] ok: provider={provider3}, dim={len(vecs3[0])}")


if __name__ == "__main__":
    main()
