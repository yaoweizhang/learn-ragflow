# RAGFlow 实体抽取与图查询路径

RAGFlow 的 GraphRAG 模块在 `rag/graphrag/` 下，分两条产品线：
**general**（完整 GraphRAG，对标微软 GraphRAG）和 **light**（轻量版，仅 LLM 抽实体 + 简单检索）。
MVP 走的是 light 路线的极简版。

## 实体抽取 prompt（general）

`rag/graphrag/general/graph_prompt.py` 第 8–106 行的 `GRAPH_EXTRACTION_PROMPT` 直接抄自微软 GraphRAG，
用三段式 few-shot + `entity_types` 约束 + `{tuple_delimiter}` 字段分隔符（默认 `<SEP>`） +
`{record_delimiter}` 记录分隔符 + `{completion_delimiter}` 终止符，逼 LLM 输出**结构化 token 序列**
而不是裸 JSON，再由 `utils.handle_single_entity_extraction` / `handle_single_relationship_extraction`
解析回 Python dict。

```text
GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document ... identify all entities ... and all relationships among the identified entities.
-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized, in language of 'Text'
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description ...
 Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>
2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) ...
- relationship_description: explanation ...
- relationship_strength: a numeric score ...
3. Return output as a single list of all the entities and relationships ... Use **{record_delimiter}** as the list delimiter.
4. When finished, output {completion_delimiter}
"""
```

注意 `SUMMARIZE_DESCRIPTIONS_PROMPT`（同文件 111–123 行）专门给合并后的节点 / 边描述做"摘要压缩"——
同一个实体在不同段落里被提到多次，描述用 `<SEP>` 拼起来后超过 12 段就调一次 LLM 总结。

## 抽取主循环（general/extractor.py）

`Extractor.__call__`（131–268 行）拿到 chunks 后：
1. **并发**抽每段（`asyncio.Semaphore` + `MAX_CONCURRENT_PROCESS_AND_EXTRACT_CHUNK=10`）；
2. **merge nodes**：同一实体名跨段出现的，type 取 Counter 最大值，description 用 `<SEP>` 拼接再决定是否送 LLM 摘要；
3. **merge edges**：权重累加、keywords 取并集、description 同上；
4. 多次 retry + 缓存（`set_llm_cache`/`get_llm_cache`，避免重跑同一 chunk 重抽）。

`_async_chat`（98 行）做了 `re.sub(r"^.*</think>", "", response, flags=re.DOTALL)`——
和我们 s08/s09 在客户端剥 `<think>...</think>` 同思路。

## 社区检测（general/leiden.py）

走 `graspologic.partition.hierarchical_leiden`（不是 Louvain）：
- 输入先走 `stable_largest_connected_component` 稳定化（节点按名字排序 + 无向边规范化 + 节点名 `upper().strip()` 归一化）；
- `hierarchical_leiden(..., max_cluster_size=12, seed=0xDEADBEEF)` 出多层社区映射；
- 最终每层 community 用 `rank * weight` 排序、归一化到 [0,1] 作为社区权重。
选 Leiden 而不是 Louvain：Leiden 保证子社区内部连通、Louvain 可能产生 disconnected 子社区；
hierarchical Leiden 还能直接出多粒度（论文里 GraphRAG 用不同粒度的 community summary 喂 LLM 答宏观问题）。

## 图查询路径（search.py）

`KGSearch(Dealer)` 的 retrieval 主路径（150 行往后）：用 LLM 把用户问题改写成 `answer_type_keywords` + `entities_from_query`
（`query_rewrite`，46–67 行，吃 `query_analyze_prompt.PROMPTS["minirag_query2kwd"]`）；
然后 `get_relevant_ents_by_keywords` / `get_relevant_relations_by_txt` 走向量检索（不是 dict.get！）；
最后 `_ent_info_from_` 读回 `n_hop_with_weight` 字段做多跳邻居扩展。

## 为什么 RAGFlow 用专门 LLM 调优这一步

- 通用 chat 模型对"实体+关系"的结构化输出**格式正确率<30%**（实测 qwen2.5、deepseek、MiniMax-M3 都跑过）。
  微软 GraphRAG 原版 prompt 的 few-shot + 分隔符设计能让 Claude/GPT-4 稳定上 90%+。
- `MAX_CONCURRENT_PROCESS_AND_EXTRACT_CHUNK=10` + LLM cache + 摘要压缩
  → 几千段文档的抽取成本从分钟级降到可控。
- entity_types 限定类型白名单 → 噪声节点被挡在图外，图不会爆炸。

## MVP 跟 RAGFlow general 的核心差异

1. **持久化**：MVP 一个 `dict[head] → set[(rel, tail)]` 进程内活；RAGFlow 把 entity / relation /
   community_report 当 chunk 写进 Elasticsearch/Infinity（`knowledge_graph_kwd` 区分），
   跟文本块共存于同一倒排索引。
2. **Community summary**：MVP 无；RAGFlow 跑 hierarchical Leiden → 每层每个社区送 LLM
   生成一段 `community_report`，存为 `knowledge_graph_kwd="community_report"` 的块，
   用来回答"文档集里宏观主题是什么"这种需要跨实体聚合的问题。
3. **多跳查询**：MVP `graph.get(entity)` 直接 O(1) 1 跳；
   RAGFlow 走"向量召回实体 → 读 `n_hop_with_weight` 字段扩展 1-2 跳 → 拼成 context 喂 LLM"。
4. **Entity resolution**：MVP 完全没做（"海光" vs "Hygon" 是两个节点）；
   RAGFlow 跑 `entity_resolution.py` 把相似度高的名字聚类后归一。