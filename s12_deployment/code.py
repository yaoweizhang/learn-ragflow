#!/usr/bin/env python3
"""
s12 部署 — 用 docker compose 把 MVP 跑起来。

运行: cd s12_deployment && docker compose up --build
访问: curl -X POST http://localhost:8000/qa -H 'Content-Type: application/json' -d '{"question":"..."}'
需要: Docker Desktop 已装；.env 配好 LLM_API_KEY
"""
import os
import subprocess
from pathlib import Path

WORKDIR = Path(__file__).parent


def main() -> None:
    if not (WORKDIR.parent / ".env").exists():
        print("❌ .env 不存在，请先 cp .env.example .env 并填 LLM_API_KEY")
        return
    if not (WORKDIR.parent / "s05_vector_index" / "_chroma").exists():
        print("❌ 索引不存在，请先跑 s05")
        return
    subprocess.run(["docker", "compose", "up", "--build"], cwd=WORKDIR, check=True)


if __name__ == "__main__":
    main()
