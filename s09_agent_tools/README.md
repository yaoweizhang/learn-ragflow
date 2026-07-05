# s09 Agent 与工具 — 让模型自己决定要不要查文档

## Units

| Unit | 标题 | 入口 |
| --- | --- | --- |
| 01 | 工具调用（单步）：把 retrieve / finish 写进 prompt，让 LLM 自己挑工具 | [`units/01_tool_call/code.py`](units/01_tool_call/code.py) |
| 02 | ReAct 循环：Thought → Action → Observation 多步循环 + JSON 失败反馈 + max_steps 兜底（章节核心） | [`units/02_react_loop/code.py`](units/02_react_loop/code.py) |

## 问题

s08 把"先检索 → 塞 context → 调 LLM → 拿答案"做成了一条硬管线。
但其实**不是每个问题都该走这条路**：

- "你好"、"1+1 等于几"——根本没文档能查，检索是浪费。
- 闲聊 / 简单数学 / 已知的知识——直接答就行。
- 多跳问题 / 反问语境——可能要先答后查，或者查了再查。

经典 RAG 强行"无差别先检索"是反模式：检索贵、噪声大、还
可能把模型带偏。本章把"要不要查"这个决策**还给 LLM**。

## 最小解法

在 `s09_agent_tools/code.py` 里做了一个**ReAct 风格的循环**：

1. 把"两个工具"和"回答格式"塞进 system prompt。
2. 让 LLM 每轮吐 `Thought / Action / ActionInput` 三行。
3. 解析这三行，路由到 `_retrieve`（s05-s07 拼起来）或 `finish`。
4. 把工具结果写回 messages 当 `Observation`，最多跑 5 轮。

```text
Thought: 用户问的是 R3630 G5 的内存插槽数量，文档里应该有。
Action: retrieve
ActionInput: {"query": "R3630 G5 内存插槽数量"}
→ (Observation: - (server_whitepaper.pdf#1) ... - (server_whitepaper.pdf#2) ... )
Thought: 找到了。
Action: finish
ActionInput: {"answer": "R3630 G5 配备 32 个 DIMM 内存插槽。"}
```

`run_agent(question)` 负责解析 + 路由 + 终止；`_retrieve(query)` 把
s05-s08 的整条管线（embed → hybrid_search → rerank）打包成一个 tool。
回答格式写在 `TOOLS_DESC` 常量里，改 prompt 不用动主循环。

## 跑起来

```bash
python s09_agent_tools/code.py
# 问: R3630 G5 的内存插槽数量
```

实测（MiniMax-M3 over minimaxi.com）：

```
Q: R3630 G5 的内存插槽数量
A: R3630 G5 服务器配备 32 个 DIMM 插槽（内存插槽），支持双路第三代 Intel Xeon 可扩展处理器。
```

跳过检索的对照（"1+1等于几"）：

```
Q: 1+1等于几
A: 1+1等于2。
```

模型**一轮直接 finish**，没碰 `_retrieve`——这是"工具选择权还给 LLM"
的关键观察：同一个 system prompt 里，模型会自己判断"这种问题查不查"。

## 真实世界的问题

1. **解析 LLM 输出不稳定**——模型可能吐 `Action: retrieve` 但
   漏 `ActionInput`，或者多写一段解释把 JSON 冲断。MVP 的 regex
   `r"Action:\s*(\w+)\s*\nActionInput:\s*(.+)"` 是脆弱的（"再
   坚持一下"是常见解法）；生产里用 OpenAI/Anthropic 的 **tool_calls
   字段** 让 API 帮你解析——模型不再"打字"给你，是结构化返回参数。
   RAGFlow 的 `LLMToolPluginCallSession` 走的就是这条路
   （见 `ragflow_notes/agent_tools.md`）。

2. **工具爆炸**——工具一多 system prompt 装不下，模型选择准确率
   暴跌。治理：① 按"用户意图"分组路由（先分类器决定走哪组工具）；
   ② 工具描述写得"互斥"（不同工具描述之间不重叠），别让模型选错；
   ③ 拆成多层 agent（supervisor + sub-agent），sub-agent 只看到自己
   的工具集。RAGFlow 用 `Agent` 组件嵌套（`self._load_tool_obj` 接
   子组件）实现"agent-as-tool"。

3. **多步规划的失败恢复**——模型选了 retrieve 但查不到东西
   （observation 是空），或者连续选同一个工具（卡死）。MVP 的
   `max_steps=5` 只是兜底天花板，**没法恢复**。生产里要：
   ① 在 system prompt 强调"资料里没有就 finish 说'我不知道'，
   别再 retrieve"；② 检测重复 action（同一个 `Action / ActionInput`
   出现两次就强制 finish）；③ 加一道"反思"——observation 不相关时
   主动改写 query 再试一次。RAGFlow 把"反思"做成独立组件
   （`Categorize` / `Switch` 分支），失败时跳到另一条路径。

## ragflow 怎么做的

见 [ragflow_notes/agent_tools.md](../ragflow_notes/agent_tools.md)。
要点：RAGFlow 把 agent 做成**可插拔的 DAG**——`agent/canvas.py` 的
`Canvas.run` 是一个**异步生成器**，按 `path` 数组顺序跑组件；不是
循环解析 LLM 的字符串输出，而是用 OpenAI 兼容的 `tool_calls` 字段
让模型直接返回结构化调用（`agent/component/agent_with_tools.py`）。
死循环的防护是 `max_rounds=5`（默认）+ `is_canceled()` 检查 +
`Categorize` 组件做"这条路走不通就跳走"。

## 思考题

- **如果模型一直选 retrieve 不 finish 怎么办？**
  答：见 `thinking_answers.md`。
