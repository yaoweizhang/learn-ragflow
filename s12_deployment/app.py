"""FastAPI 包装 s08 的 answer 函数。

把 s04 本地 BGE embed + s06 混合检索 fusion + s07 BGE rerank + s08 prompt answer
串成一条 POST /qa 链路:s04 embed_local(question) → s06 hybrid_topk(docs, q, qv,
dense_score_fn, k=10, α=0.95) → s07 rerank(q, cands, top_k=3) → s08 answer(q, top)。

术语速览 (本文件首次出现):
- FastAPI: Python 的现代 Web 框架,基于 pydantic 强类型 + 自动生成 OpenAPI 文档
- pydantic BaseModel: 用类型注解定义请求 schema,字段自动校验
- HTTPException: FastAPI 抛 HTTP 错误的工具,如 503 表示服务暂不可用
- lazy-load (懒加载): 第一次请求时才加载 collection,避免启动期就重
- uvicorn: ASGI 服务器,FastAPI 的标准运行器
- POST /qa: REST 端点,前端 fetch / curl 都能调用
"""
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

WORKDIR = Path(__file__).parent.parent
sys.path.insert(0, str(WORKDIR))
from s04_embedding.local_bge import embed_local  # noqa: E402
from s06_retrieval.hybrid_fusion import hybrid_topk  # noqa: E402
from s07_rerank.cross_encoder_rerank import rerank  # noqa: E402
from s08_prompt_generate.prompt_template import answer  # noqa: E402
import chromadb  # noqa: E402

app = FastAPI()
COL = None  # 在第一次请求时 lazy-load；索引不存在时给清晰错误
DOCS: list[dict] = []  # 缓存:{text, source, page, chunk_id} from chroma collection
VEC_BY_ID: dict[str, list[float]] = {}  # 缓存:chunk_id → 预存向量(避免 /qa 每次重 embed)


def _cosine(a: list[float], b: list[float]) -> float:
    """cosine 相似度,假设已 L2 归一化(BGE normalize_embeddings=True)。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _get_col():
    global COL, DOCS, VEC_BY_ID
    if COL is not None:
        return COL
    db_path = WORKDIR / "s05_vector_index" / "_chroma"
    if not db_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Chroma 索引不存在。请先在仓库根目录运行: python s05_vector_index/chroma_build.py",
        )
    try:
        COL = chromadb.PersistentClient(path=str(db_path)).get_collection("docs")
    except (ValueError, Exception) as e:
        # chromadb 在 collection 不存在时抛 ValueError("Collection docs does not exist")
        raise HTTPException(
            status_code=503,
            detail=f"Chroma collection 'docs' 不存在: {e}。请先运行 s05_vector_index/chroma_build.py。",
        )
    # 一次性加载所有 docs + 预存向量,缓存到模块级变量。
    # /qa 每次请求只跑 BM25 + dense cosine + α 加权融合(廉价);无需再 embed 文档。
    raw = COL.get(include=["embeddings", "documents", "metadatas"])
    for cid, doc, meta, vec in zip(
        raw["ids"], raw["documents"], raw["metadatas"], raw["embeddings"]
    ):
        page = meta.get("page")
        try:
            page = int(page) if page else None
        except (ValueError, TypeError):
            page = None
        DOCS.append({
            "text": doc,
            "source": meta.get("source", ""),
            "page": page,
            "chunk_id": cid,
        })
        # chromadb 可能返回 numpy.ndarray,转 list 保 hybrid_topk + 余弦兼容
        VEC_BY_ID[cid] = list(vec) if not isinstance(vec, list) else vec
    return COL


class QARequest(BaseModel):
    question: str


@app.post("/qa")
def qa(req: QARequest) -> dict:
    _get_col()  # 触发 lazy-load + 缓存 docs / vectors
    qv = embed_local([req.question])[0]

    def _dense_score(chunk: dict) -> float:
        return _cosine(qv, VEC_BY_ID[chunk["chunk_id"]])

    # BM25 + dense α 加权融合;alpha=0.95 偏向量,跟 s06 unit 02 默认一致
    cands = hybrid_topk(DOCS, req.question, qv, _dense_score, k=10, alpha=0.95)
    top = rerank(req.question, cands, top_k=3)
    return answer(req.question, top)


if __name__ == "__main__":
    import uvicorn

    # 也可 `uvicorn s12_deployment.app:app --host 0.0.0.0 --port 8000` 直接起;
    # 这里保留 python app.py 起服务的方式,方便本地快速跑。
    uvicorn.run(app, host="0.0.0.0", port=8000)
