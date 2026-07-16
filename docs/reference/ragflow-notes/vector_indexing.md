# RAGFlow 向量索引选型

## 一句话
RAGFlow 把 Elasticsearch 和 Infinity 作为两类"向量 + 全文"二合一的
后端，统一封装在 `DocStoreConnection` 抽象类后面；不用 Chroma 是因为
它缺多租户隔离、原生 BM25 + 向量融合、分片副本。

## 来源
- 仓库：https://github.com/infiniflow/ragflow
- 模块：`common/doc_store/`、`rag/utils/es_conn.py`、`rag/utils/infinity_conn.py`
- 关联：本仓库 s05 `chroma_build.py` / `chroma_query.py` 用 Chroma 做 toy 实现

## 代码：ES 索引创建

```python
def create_idx(self, index_name: str, dataset_id: str, vector_size: int, parser_id: str = None):
    # parser_id is used by Infinity but not needed for ES (kept for interface compatibility)
    if self.index_exist(index_name, dataset_id):
        return True
    try:
        return IndicesClient(self.es).create(index=index_name,
                                             settings=self.mapping["settings"],
                                             mappings=self.mapping["mappings"])
    except Exception:
        self.logger.exception("ESConnection.createIndex error %s" % index_name)
```

## 为什么不用 Chroma

- **多租户 + 分片副本是硬需求**。RAGFlow 索引名模板是 `ragflow_<tenant_id>_<kb_id>`（见 `conf/mapping.json`），每个租户的每个知识库是一套独立 ES 索引，天然支持物理隔离 + 副本 + 分片。Chroma 的 persistent client 是单文件 SQLite + 本地 hnswlib，做不到这点，上百个团队 / 几亿文档就崩。
- **BM25 + 向量要在同一个 `search` 里融合**。`ESConnection.search` 用 `MatchTextExpr / MatchDenseExpr / FusionExpr` 拼一个 `bool` 查询，文本打分和向量打分在同一轮 ES 查询里出分，再按 `vector_similarity_weight` 加权融合。Chroma 的 `query` 只支持 `where` 元数据过滤 + 向量，**没有原生 BM25**，要混合检索只能外挂 ES，反而变成"ES+Chroma"两个系统。
- **元数据过滤 + 聚合 + 排序**。RAGFlow 在 ES 上跑 `terms / range / rank_feature` 三件套，还按 `order_by` 字段、`agg_fields` 桶灵活排 / 聚合；Chroma 的 `where` 只支持 `$eq / $in / $and / $or`，做不了大规模按权限 / 标签过滤的检索 + 统计报表。Infinity 这条线在亿级 + 多列过滤上更强（`rag/utils/infinity_conn.py` 直接编 `filter_fulltext` + 向量相似度），取舍一致。
