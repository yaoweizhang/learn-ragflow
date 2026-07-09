# s10 / Unit 02 — 1 跳图查询(纯内存,无 LLM)

> 由浅入深第 2 步:加载 unit 01 落盘的 `_graph.jsonl`,在 `dict[head] → set[(rel, tail)]` 上跑 O(1) 的 1 跳邻居查询。  
> 这是 GraphRAG 的"读"半边——和单元 01 的"写"半边分开,便于离线调试。

## 这是什么

`code.py` 把 unit 01 写出的 JSONL 重新加载回内存的图结构,提供最朴素的 1 跳查询:

- `load_graph(path)` —— 按行读 JSONL,反向重建 `dict[head] → set[(rel, tail)]`;同时把 `tail` 也注册成节点,即便它没有出边也能被 query 命中(便于"这个实体存在但孤立"的诊断);
- `query_graph(graph, entity)` —— `graph.get(entity, set())`,O(1);返回前按 `(rel, tail)` 字母序排,便于对比不同次抽取的结果;
- `main()` —— 加载 → 打印节点 / 边数 → 循环输入实体 → 打印 1 跳邻居,直到空行退出。**完全不调 LLM**,只要 `_graph.jsonl` 存在就跑得起。

## 跑起来

```bash
# 1. 先跑一次抽取(生成 _graph.jsonl)
python s10_graphrag/units/01_extract/code.py

# 2. 再跑查询(可重复跑,不调 LLM)
python s10_graphrag/units/02_query/code.py
```

实测(MiniMax-M3 抽完 8 个 chunk 后的图):

```
图节点数: 8, 边数: 6
查哪个实体 (回车退出): 紫光恒越技术有限公司
  紫光恒越技术有限公司 --版权所有--> 紫光恒越 R3630 G5 双路机架式服务器 产品白皮书 v1.0

查哪个实体 (回车退出): 不存在的实体xyz
  (无结果——'不存在的实体xyz' 不在图中或没有出边)

查哪个实体 (回车退出):
```

## 它做对了什么

- **O(1) 查询**:`dict.get(entity)` 是哈希查找,8 节点和 80 万节点同一个 latency——只要 entity 名字完全匹配,立刻返回;
- **确定性**:同一个 `_graph.jsonl` 跑 N 次结果完全一致,适合做"prompt 改了之后,看 query 输出有没有变"的回归对照;
- **离线 / 零成本**:不调 LLM、不调 embedding、不调任何外部服务——CI 里跑测试、或调试实体名 / prompt 时可以放心重跑;
- **可中断 / 可恢复**:循环读输入,空行退出——调试时反复改 entity 名快速验证,不用每次重启进程。

## 它做错了什么

- **没有多跳**:`"X 的竞争对手的合作伙伴"` 是 3 跳,1 跳答不全。要 BFS 自己写,而且 3 跳以上必须上 community summary(否则 LLM 拿到的 context 拼接成本太高);
- **没有 entity resolution**:`"紫光恒越"` 和 `"紫光恒越技术有限公司"` 是两个键,query 命中哪个名字才能拿哪个的边——用户一般不会知道图里到底注册了哪个变体名;
- **没有语义匹配**:用户输入 `"紫光"` 也命中不了 `"紫光恒越技术有限公司"`——要么靠 entity resolution 提前归一化,要么 LLM 在 prompt 里把 user query 改写一遍;
- **没有路径权重 / 方向语义**:`set[(rel, tail)]` 不记录反向边、不记录 confidence,复杂关系(`"A 投资 B"` vs `"B 被 A 投资"`)分不清谁主动谁被动;
- **没有图遍历可视化 / 解释**:`query_graph` 只返边列表,不出"为什么是这条边"、"这条边来自哪个 chunk"——bad case 排查要回到 `_graph.jsonl` 里手 grep;
- **节点孤立时看不出来**:`graph.get("孤立实体")` 直接返空集,虽然 unit 01 已经把 tail 注册成节点(让"存在但孤立"能被命中),但**没有入边的节点**和**不存在的节点**在我们的实现里都返回空集,提示完全一样。

## 对照 ragflow 怎么做的

`docs/reference/ragflow-notes/graph_extraction.md` 描述的 RAGFlow KGSearch 主路径:

- **召回不靠 dict.get**:走"LLM 改写 query → 向量召回实体 → 读 `n_hop_with_weight` 字段扩展多跳"(`rag/graphrag/search.py` 的 `KGSearch` 类)。即使用户 query 是 `海光` 这种别名,向量检索能命中"海光 CPU"和"海光信息技术";
- **多跳由存储层支持**:`n_hop_with_weight` 字段在写入时算好 1-2 跳邻接表,查询时只读不计算,所以"2 跳邻居"和"1 跳邻居"同 latency;
- **community_report 答宏观问题**:hierarchical Leiden 社区检测后,每社区一段 summary,用来答"文档集主要在讲什么"——unit 02 的 1 跳查询根本答不了这种问题,因为它必须**跨实体聚合**才能看出主题;
- **entity_resolution.py 2 阶段管线**:见 unit 01 README 的对照部分。RAGFlow 把别名归一做到写入前,所以查询阶段不用再处理"这个名字到底是不是同一个东西"。

参考:[`docs/reference/ragflow-notes/graph_extraction.md`](../../../../docs/reference/ragflow-notes/graph_extraction.md)

## 思考题

**如果用户输入 `"紫光恒越"` 但图里只有 `"紫光恒越技术有限公司"` 和 `"紫光恒越技术"` 两个节点,怎么在不调 embedding 的前提下让 query 至少"提示一下有哪些近似节点"?**

提示:加一道"模糊查"——输入 `"紫光恒越"` 时,如果 `query_graph` 返空,就 fallback 一次"子串匹配"或 `difflib.get_close_matches`,把近似的几个节点名打出来让用户挑。RAGFlow 不这么干(它走 embedding 召回),但**纯字符串**方案在 8-节点 toy 图上够用,而且能让你看清"没有 entity resolution 时,查询阶段的兜底能兜到什么程度"。