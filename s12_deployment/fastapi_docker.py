#!/usr/bin/env python3
"""
s12 / unit 01 — fastapi_docker：把 s08 的 answer() 包成 FastAPI 服务,
然后用 docker compose 一键启动整条链路。

本单元作为聚合入口的最底层，
是 s12 唯一一个 unit:管 FastAPI 应用 (app.py) + Dockerfile +
docker-compose.yml 的"一次启动"工作流。

运行: python s12_deployment/fastapi_docker.py
需要: Docker Desktop 已装；项目根 .env 配好 LLM_API_KEY；
      s05_vector_index/_chroma 索引已生成。
"""
import subprocess
from pathlib import Path

# 工作区 = 项目根(从 s12_deployment/fastapi_docker.py 上溯 2 级)
WORKDIR = Path(__file__).resolve().parents[1]
S12_DIR = WORKDIR / "s12_deployment"


def main() -> None:
    # 1. .env gating — 没有 LLM_API_KEY 就别 build,免得白烧镜像层
    if not (WORKDIR / ".env").exists():
        print("❌ .env 不存在,请先 cp .env.example .env 并填 LLM_API_KEY")
        return
    # 2. 索引 gating — Chroma 持久化目录必须在主机上,否则容器 :ro 挂载会空
    if not (WORKDIR / "s05_vector_index" / "_chroma").exists():
        print("❌ 索引不存在,请先跑 s05 (python s05_vector_index/chroma_build.py)")
        return
    # 3. 一键 build + up — cwd 用 s12_deployment 让 compose 找到同名 yml
    #    若本机没装 docker (例如 CI / 学生笔记本) 则优雅降级,把"该敲的命令"打印出来
    import shutil
    if shutil.which("docker") is None:
        print("⚠️  docker 未安装,跳过实际 build/up 步骤。")
        print("   本机手动执行即可:")
        print(f"     cd {S12_DIR} && docker compose up --build")
        return
    subprocess.run(["docker", "compose", "up", "--build"], cwd=S12_DIR, check=True)


if __name__ == "__main__":
    main()
