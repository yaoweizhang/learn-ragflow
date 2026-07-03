# RAGFlow 怎么做: 向量索引选型 (ES / Infinity, 不用 Chroma)

## 来源
- 仓库: https://github.com/infiniflow/ragflow
- 文件: `common/doc_store/es_conn_base.py`
- 行号: L128-L141 (create_idx) + L141-L160 续 (附 ES 文档)
- commit: `828c5789f651d4c4ebe4645190b8b8d244144fe0`
- 引用日期: 2026-07-03
- GitHub 链接: https://github.com/infiniflow/ragflow/blob/828c5789f651d4c4ebe4645190b8b8d244144fe0/common/doc_store/es_conn_base.py#L128-L141

## 一句话
RAGFlow 把 `Elasticsearch` 和 `Infinity` 作为两类"向量 + 全文"二合一的
后端,统一封装在 `DocStoreConnection` 抽象类 (`doc_store_base.py:113`)
后面,通过 `create_idx / search / insert` 这套一致接口让上层无感切换;
Chroma 没进选型,是因为它缺少 RAGFlow 真正需要的能力 —— 多租户隔离、
原生 BM25 + 向量融合、分片副本、可水平扩到亿级。

## 代码 (节选, ES 索引创建)

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

## 它为什么这样写 (为什么不用 Chroma)

- **多租户 + 分片副本是 RAGFlow 的硬需求,Chroma 没有**。RAGFlow 索引名
  模板是 `ragflow_<tenant_id>_<kb_id>` (见 `conf/mapping.json` 的 `index.number_of_shards`
  等设置),每个租户的每个知识库是一套独立 ES 索引,天然支持物理隔离
  + 副本 + 分片 (单 ES 集群水平扩到 PB 级)。Chroma 的 persistent
  client 是单文件 SQLite + 本地 hnswlib,做不到这点,生产场景一上
  百个团队 / 几亿文档就崩。
- **BM25 + 向量要在同一个 `search` 里融合,不是分两次取**。RAGFlow 的
  `ESConnection.search` (`rag/utils/es_conn.py:141-230`) 用
  `MatchTextExpr / MatchDenseExpr / FusionExpr` 拼一个 `bool` 查询,
  文本打分 (`query_string`) 和向量打分 (`s.knn`) 在同一轮 ES 查询里
  出分,再按 `vector_similarity_weight=0.5` 加权融合 (`es_conn.py:194-202`)。
  Chroma 的 `query` 只支持 `where` 元数据过滤 + 向量,**没有原生 BM25**,
  要混合检索只能外挂 Elasticsearch,反而变成"ES+Chroma"两个系统,
  运维成本翻倍。
- **元数据过滤 + 聚合 + 排序是 Chroma 的弱项**。RAGFlow 在 ES 上
  跑 `terms / range / rank_feature` 三件套 (`es_conn.py:166-187`),
  还按 `order_by` 字段 (`asc/desc`)、`agg_fields` (terms 桶) 灵活排
  /聚合 (`es_conn.py:243-254`),这些在 ES 里是基础操作。Chroma 的
  `where` 只支持 `$eq / $in / $and / $or` 简单条件,`get`/`peek` 只能
  按 `limit/offset` 拉,**做不了大规模按权限 / 标签过滤的检索 + 统计
  报表**。Infinity 这条线在亿级 + 多列过滤上更猛
  (`rag/utils/infinity_conn.py:94-200` 直接编 `filter_fulltext` +
  向量相似度),也是同样的取舍。
