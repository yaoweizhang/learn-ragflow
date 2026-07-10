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
4. **Entity resolution**：MVP 完全没做（"紫光恒越" vs "紫光恒越技术有限公司"是
   两个节点；alias 列表如 `{"海光": "Hygon"}` 也不在 LLM 抽取范围里）；
   RAGFlow 跑 `entity_resolution.py` 把相似度高的名字聚类后归一。

## Light 路径：LightRAG 风格的更简 prompt

`rag/graphrag/light/graph_prompt.py` 顶部注明 "Reference: LightRAG"——
light 路径走的是 [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) 的极简 schema。
默认分隔符、实体类型都和 general 路径**严格一致**（可共享解析器），
但 prompt 模板短得多，没有 `<SEP>` 多分隔符那一套、也没有 community / hierarchy：

```python
PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event", "category"]

PROMPTS["entity_extraction"] = """---Goal---
Given a text document ... identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

---Steps---
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Provide a comprehensive description ... *based solely on the information present in the input text*. **Do not infer or hallucinate information not explicitly stated.** ...
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities ... identify all pairs of (source_entity, target_entity) that are *clearly related* ...
- relationship_description: explanation ...
- relationship_strength: a numeric score indicating strength ...
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. ...
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships ... Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}
..."""
```

四个固定分隔符（`<|>`, `##`, `<|COMPLETE|>`）+ 三段式 tuple/record/completion = 跟 general 同款解析器
`utils.handle_single_entity_extraction` 可以直接吃 light 输出，零修改。
差异在 prompt 内 prompt 长度、few-shot 示例数量、是否生成 content_keywords（general 没有这一项）。

MVP 跟 light 的差异就只剩 prompt 长度了——我们的 `_llm_chunks` 跑的是**裸 JSON**，
不带分隔符，格式正确率低。light 路径 prompt 长度 1/2，few-shot 也只放 1-2 个，
对中文 / MiniMax-M3 / qwen 这一档模型友好得多。

## Entity resolution 的两阶段管线

`rag/graphrag/entity_resolution.py` 的 `EntityResolution.__call__` 实现了一个
"**字符串相似度粗筛 → LLM batch 精审**"两阶段管线：

```python
nodes = sorted(graph.nodes())
entity_types = sorted(set(graph.nodes[node].get("entity_type", "-") for node in nodes))
node_clusters = {entity_type: [] for entity_type in entity_types}

for node in nodes:
    node_clusters[graph.nodes[node].get("entity_type", "-")].append(node)

candidate_resolution = {entity_type: [] for entity_type in entity_types}
for k, v in node_clusters.items():
    candidate_resolution[k] = [(a, b) for a, b in itertools.combinations(v, 2)
                               if (a in subgraph_nodes or b in subgraph_nodes)
                                  and self.is_similarity(a, b)]
num_candidates = sum([len(candidates) for _, candidates in candidate_resolution.items()])
callback(msg=f"Identified {num_candidates} candidate pairs")
...
resolution_batch_size = 100
max_concurrent_tasks = 5
semaphore = asyncio.Semaphore(max_concurrent_tasks)

async def limited_resolve_candidate(candidate_batch, result_set, result_lock):
    ...
    selected_pairs = await asyncio.wait_for(
        self._resolve_candidate(candidate_batch, result_set, result_lock, task_id),
        timeout=timeout_sec,
    )
```

**两阶段**：
1. **粗筛**：`is_similarity(a, b)` 先用字符串相似度（中文走 2-gram 字符重叠、英文走
   `editdistance.eval` / max(len)，阈值内部定）把所有节点两两配对，砍掉 90%+ 候选。
   按 `entity_type` 分桶——"海光 CPU 公司"和"海光 鲲鹏处理器"如果类型不同，根本不进同一桶。
2. **精审**：剩下的候选每 100 个一批送 LLM（`_resolve_candidate` 用
   `ENTITY_RESOLUTION_PROMPT` 几个 few-shot 让模型回答"这两个实体能不能合并成同一个"），
   并发上限 5（`asyncio.Semaphore`），加 timeout 280s。每批结果**写 checkpoint**，中断可恢复。

MVP 完全没做 entity resolution。后果：白皮书里 "紫光恒越 / 紫光恒越技术有限公司 / 紫光恒越技术" 是 3 个独立节点，
3 条独立 edges，召回只有命中其中一个时才能拿到——幻觉和召回率都会被吞。生产里 entity resolution
是必选项，不是可选项。