# s09 思考题答案

## 1. 如果模型一直选 retrieve 不 finish 怎么办？

这是个真实的失败模式——LLM 训练数据里"helpful assistant 应该是先查再答"
的偏置很强，又因为 retrieve 的 observation 总是非空（哪怕不相关），
模型找不到"应该停"的信号，会一路 retrieve 到 `max_steps` 撞顶。

按代价从低到高排三种解法：

**1) `max_steps` 上限**（MVP 已经做了）——硬天花板。最坏情况是用户
看到"Max steps reached."。优点：简单、零误判；缺点：用户体验差，
它治不住"模型本来可以答对但就是停不下来"。

**2) 在 system prompt 强调"必须 finish"**——把 `TOOLS_DESC` 里的
工具描述改成更显式的"每个问题**最多调用一次 retrieve**，再调用一次
retrieve 还没有答案就用已有的 Observation 给出 finish，否则就回答
'我不知道'并 finish"。这是**软约束**，依赖模型遵从指令，但对主流
模型（GPT-4o / Claude / Qwen-Max）效果不错；缺点是 prompt 越长越
稀释，工具一多就顾不过来。

**3) 检测重复 Action**——运行时加一道"刚才两步的 `Action / ActionInput`
对是不是一模一样"的检查，命中就**强制 finish**（observation 是
`"我重复了同一次检索，无法获取更多新信息"`）。这是**硬约束**，
不依赖模型听话。MVP 的 `max_steps=5` 实际是它的弱化版（撞 5 步就
终止，不管内容）。

**生产推荐**：1 + 2 + 3 都做——`max_steps` 是兜底，prompt 软约束
是主力，重复检测是保险。RAGFlow 三种都做（`max_rounds=5` +
system prompt 强调 + `is_canceled()` 主动取消 + `Categorize` 组件
做"这条路走不通跳走"，本质就是"用结构化组件替代字符串解析里的
软约束"）。
