"""FastAPI 包装 s08 的 answer 函数。"""
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

WORKDIR = Path(__file__).parent.parent
sys.path.insert(0, str(WORKDIR))
from s08_prompt_generate.code import answer  # noqa: E402
from s04_embedding.code import embed  # noqa: E402
from s06_retrieval.code import hybrid_search  # noqa: E402
from s07_rerank.code import rerank  # noqa: E402
import chromadb  # noqa: E402

app = FastAPI()
COL = None  # 在第一次请求时 lazy-load；索引不存在时给清晰错误


def _get_col():
    global COL
    if COL is not None:
        return COL
    db_path = WORKDIR / "s05_vector_index" / "_chroma"
    if not db_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Chroma 索引不存在。请先在仓库根目录运行: python s05_vector_index/code.py",
        )
    try:
        COL = chromadb.PersistentClient(path=str(db_path)).get_collection("docs")
    except (ValueError, Exception) as e:
        # chromadb 在 collection 不存在时抛 ValueError("Collection docs does not exist")
        raise HTTPException(
            status_code=503,
            detail=f"Chroma collection 'docs' 不存在: {e}。请先运行 s05_vector_index/code.py。",
        )
    return COL


class QARequest(BaseModel):
    question: str


@app.post("/qa")
def qa(req: QARequest) -> dict:
    col = _get_col()
    qv = embed([req.question])[0]
    cands = hybrid_search(col, req.question, qv, k=10)
    top = rerank(req.question, cands, top_k=3)
    return answer(req.question, top)
