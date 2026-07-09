# s10 / Unit 01 — LLM 抽实体关系三元组 + 持久化 JSONL

> 由浅入深第 1 步:把每个 chunk 喂给 LLM 让它吐 `(head, rel, tail)` 三元组,合并成图,写到 `s10_graphrag/_graph.jsonl` 供 unit 02 离线查询。  
> unit 02 只读 JSONL,不再调 LLM。

## 这是什么

`code.py` 是一个"LLM 抽取 + 容错解析 + 内存合并 + 落盘"的小教学版 GraphRAG 抽取器:

- `EXTRACT_PROMPT` —— 喂文字给 LLM,要求吐 JSON 数组,每项 `{head, rel, tail}`,没有就 `[]`;
- `_llm_json(prompt)` —— 调 OpenAI 兼容接口,**对 MiniMax-M3 的三种坏输出做兜底**:剥 `<think>...</think>` 推理块、剥 ```json ``` 围栏、再容忍 `dict / list` 两种 JSON 顶层结构。任一失败返 `[]`,**不抛异常**,让上游 `build_graph` 当作"这个 chunk 没抽到"继续;
- `extract_triples(text)` —— 一段文字 → 一组三元组;
- `build_graph(triples_list)` —— 把所有 chunk 的结果合并成 `dict[head] → set[(rel, tail)]`;
- `save_graph(graph, path)` —— 落盘到 JSONL,每行一个 triple(便于追加 / 复查 / 不加载整图就能 grep);
- `main()` —— **self-contained**:内联 pypdf + python-docx + 简化版 chunking(双换行分段 + 500 字 cap + 取前 8 段),跑固定输入,打印节点 / 边数,落盘。

## 跑起来

```bash
python s10_graphrag/units/01_extract/code.py
```

输出示例(MiniMax-M3 over minimaxi.com,samples = server_whitepaper.pdf + disclosure.docx,只取前 8 个 chunk):

```
chunks: 8
图节点数: 8, 边数: 6
持久化: s10_graphrag/_graph.jsonl
```

不同次跑节点 / 边数会小幅抖动(LLM 在 temperature=0 下对长 prompt 仍有少量随机性;chunk 0/1/2 是封面 + 目录,信息密度低,模型决定抽不抽也有差异)——这是 LLM 抽取的固有现象,不是 bug。

## 它做对了什么

- **没有 schema 也能跑**:`EXTRACT_PROMPT` 只要求 `(head, rel, tail)` 三元组列表,不限定实体类型(人 / 公司 / 产品)、不要求 entity description、不要求 relationship strength——结构最简,新手也能秒看明白"图是怎么来的";
- **MiniMax-M3 兼容性**:三个客户端兜底串成一行兜底链——剥 `<think>`,剥 ```json ``` fence,容忍 dict 或 list 顶层。这套写法 s08 / s09 也在用,跨章节一致;
- **失败不 crash**:JSON parse 失败直接返 `[]`,所以 8 个 chunk 里偶尔有 1-2 个解析坏掉也不会中断整条管线;
- **JSONL 持久化**:一行一个 triple,文本可读、可 grep、可 `wc -l`,不依赖任何图数据库——unit 02 一个 `for line in f` 就能跑 1 跳查询。

## 它做错了什么

- **没有 entity resolution**:`"紫光恒越"` / `"紫光恒越技术有限公司"` / `"紫光恒越技术"` 是 3 个独立节点,3 条独立边——召回只有命中其中一个名字时才能拿到边。生产里要 entity resolution(embedding 聚类 + LLM 判断合并);
- **没有 entity_types 白名单**:模型可能把章节标题(`"3.1 技术规格"`)当成实体,抽到一堆噪声节点。RAGFlow 的 `DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event", "category"]` 就是这一刀;
- **每 chunk 一次 LLM**:8 段 8 次同步调用,顺序阻塞;生产里走 `asyncio.Semaphore(10)` + LLM cache,几千段也不超时;
- **没有并发、没有重试、没有缓存**:LLM 偶尔抽到空、超时、限流,我们的实现一律当成"没抽到";生产要 retry + cache,同一 chunk 重跑不重抽;
- **没有节点 / 边合并策略**:同实体跨段出现时,`build_graph` 直接当新节点处理;生产要走 merge_nodes(type 取 Counter 最大值、description 拼 `<SEP>`)和 merge_edges(weight 累加、keywords 并集)。

## 对照 ragflow 怎么做的

`docs/reference/ragflow-notes/graph_extraction.md` 给出了 RAGFlow 两条路径的对照:

- **general 路径**:抄微软 GraphRAG 三段式 prompt(`entity_types` 白名单 + `<SEP>` / `<record_delimiter>` / `<completion_delimiter>` 三类分隔符 + few-shot),输出**结构化 token 流**而不是裸 JSON,正确率 30% → 90%+。我们这里走的是裸 JSON,模型偶尔不 honor `response_format=json_object`;
- **light 路径**:HKUDS LightRAG 风格,同样用 `<|>` / `##` / `<|COMPLETE|>` 三类分隔符,但 prompt 短得多(1/2 长度),对中文 / qwen / MiniMax-M3 这档模型友好。生产里**建议切到 light prompt** + `entity_types` 白名单,正确率能上 80%+;
- **entity_resolution.py 两阶段管线**(`rag/graphrag/entity_resolution.py:81-150`):**字符串相似度粗筛**(中文 2-gram、英文 editdistance)→ **LLM batch 精审**(每 100 个一批、并发上限 5、timeout 280s、checkpoint 持久化)。粗筛按 `entity_type` 分桶——不同类型的实体根本不进同一候选集。这是 MVP 完全缺失的环节。

参考:[`docs/reference/ragflow-notes/graph_extraction.md`](../../../../docs/reference/ragflow-notes/graph_extraction.md)

## 思考题

**为什么 `EXTRACT_PROMPT` 不直接限定 `entity_types` 白名单?限定了会出什么问题?**

提示:不限白名单 → 章节标题 / 编号 / 单位(`"3.1"`、`"500GB"`)都被当实体,图噪声大、召回稀释;限白名单 → 模型可能**过度保守**,白皮书里"鲲鹏 920 处理器"如果不在 `["organization", "person", "geo", "event", "category"]` 里就被吞掉,关键实体反而漏。RAGFlow 的折中是用 `DEFAULT_ENTITY_TYPES` 五个类型兜底,加上"宁滥勿缺"的 prompt 措辞("如果不确定,标记为 category")。