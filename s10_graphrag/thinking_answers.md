# s10 思考题答案

## 1. 如果两段文字里同一实体名字不同（"产品 A" vs "A 型"）怎么办？

这是 **entity resolution / entity linking** 问题——图谱质量的天花板就卡在这里。
如果不做，"紫光恒越"和"紫光恒越技术有限公司"是图里两个不同节点，所有下游
查询（PageRank、社区检测、邻居扩展）都会被撕碎。

按代价从低到高排三种解法：

**1) 规则归一化**（最便宜，能解决 60% 中文场景）——抽取后做字符串归一化：
- 去前后缀（"公司"、"有限公司"、"股份有限公司"），去空格、全角半角；
- 中文数字 → 阿拉伯数字；
- 拼音模糊匹配（"海光" ≈ "Hygon"）；
- 用一个简单的别名词典（业务给的 `known_aliases = {"海光": "Hygon", "紫光恒越": "紫光恒越技术有限公司"}`）。

缺点：维护词典贵、对未见过的别名零能力。

**2) Embedding 余弦相似度 + LLM 判断**（生产主流）——
先对所有节点名 embed，再层次聚类（cos > 0.85 归一组），每组丢给 LLM 问
"这些名字是不是指同一实体"。RAGFlow `rag/graphrag/entity_resolution.py`
就是这个套路——对实体描述做 embed → 聚类 → 让 LLM 给定 canonical name。

**3) 让 LLM 在抽取阶段直接出 canonical name**（最准但最贵）——
prompt 改成"请用 canonical name 写实体；如果同一实体出现多种写法，用
出现频率最高的那个"。微软 GraphRAG 原版 prompt 不这么做，是图建好后
单独跑 entity resolution——把"抽取"和"归一"解耦，归一阶段可以重跑、
可以换策略。

**生产推荐**：1 + 2 组合——规则先做一遍快速合并兜底常见中文后缀，剩余
疑似冲突送 embedding + LLM 判断。RAGFlow 三种都做（`entity_resolution.py` +
`normalize_node_names` 做 `upper().strip()` + `html.unescape`，见
`general/leiden.py:58`）。