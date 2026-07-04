"""FastAPI 包装 s08 的 answer 函数。"""
import sys
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

WORKDIR = Path(__file__).parent.parent
sys.path.insert(0, str(WORKDIR))
from s08_prompt_generate.code import answer  # noqa: E402
from s04_embedding.code import embed  # noqa: E402
from s06_retrieval.code import hybrid_search  # noqa: E402
from s07_rerank.code import rerank  # noqa: E402
import chromadb  # noqa: E402

app = FastAPI()
COL = chromadb.PersistentClient(path=str(WORKDIR / "s05_vector_index" / "_chroma")).get_collection("docs")


class QARequest(BaseModel):
    question: str


@app.post("/qa")
def qa(req: QARequest) -> dict:
    qv = embed([req.question])[0]
    cands = hybrid_search(COL, req.question, qv, k=10)
    top = rerank(req.question, cands, top_k=3)
    return answer(req.question, top)
