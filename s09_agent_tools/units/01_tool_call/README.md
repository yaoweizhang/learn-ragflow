# s09 / Unit 01 — 工具调用（单步）：让 LLM 自己选 retrieve 还是 finish

> 由浅入深第 1 步：把 2 个工具塞进 system prompt，调一次 LLM，正则抠 `Action / ActionInput`，跑工具。  
> unit 02 会把这一轮包成 `Thought → Action → Observation` 循环，跑多步。

## 这是什么

本单元只走**一轮** agent 决策：

1. `TOOLS_DESC` 把 2 个工具（`retrieve(query)` / `finish(answer)`）和输出格式（`Thought` / `Action` / `ActionInput`）写进 system prompt；
2. `_llm(messages)` 调 OpenAI 兼容接口（无 key 时降级，演示假设 retrieve），并剥掉 MiniMax / DeepSeek R1 的 `<think>...</think>` 推理块；
3. 用同款 regex `r"Action:\s*(\w+)\b\s*ActionInput:\s*(.+)"`（DOTALL）从 LLM 原话里抠出工具名 + 参数；
4. `single_shot(question)` 把原话、解析的 action、解析的 payload、工具返回的 observation 一并打印。

跑过 `python s09_agent_tools/units/01_tool_call/code.py` 会看到：问"内存插槽数量"时 LLM 选 `retrieve`；问"1+1 等于几"时 LLM 直接选 `finish`——**同一个 system prompt，LLM 自己决定要不要查文档**。

## 跑起来

```bash
python s09_agent_tools/units/01_tool_call/code.py
# 问: R3630 G5 的内存插槽数量
```

无 `LLM_API_KEY` 时打印演示（假设 LLM 选 retrieve 并跑检索管线）：

```
[Q] R3630 G5 的内存插槽数量

[skipped: LLM_API_KEY not set] — 演示假设 LLM 选了 retrieve:

[LLM raw]
Thought: 用户问的是 R3630 G5 的内存插槽数量,文档里应该有。
Action: retrieve
ActionInput: {"query": "R3630 G5 内存插槽数量"}

[Parsed action] retrieve
[Parsed payload] {'query': 'R3630 G5 内存插槽数量'}

[Observation]
- (server_whitepaper.pdf#1) 紫光恒越 R3630 G5 双路机架式服务器 ...
- (server_whitepaper.pdf#2) 配备 32 个 DIMM 内存插槽 ...
...
```

## 它做对了什么

- **同一个 system prompt 里 LLM 会自己挑工具**：问"内存"它选 `retrieve`，问"1+1"它直接 `finish`——把"要不要查"的决策**还给 LLM** 是本章核心观察；
- **Action / ActionInput 解析**：regex 同时兼容"ActionInput 在新行 / 与 Action 同行 / 被 ```json 围栏包住"三种常见写法，Markdown 围栏剥掉再 `json.loads`；
- **JSON 解析失败兜底**：解析不出来不崩，返回 `(JSON 解析失败: ...)` 让调用方知道——为 unit 02 的"反馈回 messages 让 LLM 自己修正"埋钩子；
- **self-contained**：内联 chroma + s04 embed + s06 hybrid + s07 rerank，不依赖 chapter-root。

## 它做错了什么

- **只能走一轮**：observation 出来就停了，没有"看了结果再决定下一步"的循环——这就是 unit 02 要解决的；
- **没有 plan / execute 分层**：LLM 想回答复杂多跳问题（先查 A 再查 B 再总结）做不了；
- **LLM 输出格式仍然脆弱**：regex 抓不到时单元就退化成"打印原话"——生产里应该用 OpenAI / Anthropic 的 `tool_calls` 字段让 API 帮你解析；
- **没有死循环防护**：单步问题，但已经能看出"LLM 一直选 retrieve 不 finish"在多步场景下是定时炸弹。

## 对照 ragflow 怎么做的

RAGFlow 把"工具调用"做成了**结构化 API 调用**，不是从 LLM 自由文本里 regex 抠：

- `agent/component/agent_with_tools.py` 的 `Agent.__init__` 把工具喂给 LLM 客户端（`self.chat_mdl.bind_tools(self.toolcall_session, self.tool_meta)`），LLM 返回的 `tool_calls[].function.name + arguments` 直接被 `LLMToolPluginCallSession.tool_call` 解析成 `name + arguments` 字典、查 `self.tools[name]` 调实例的 `invoke(**arguments)`；
- 工具列表本身来自 `agent/component/` 下面的 30 多个**声明式组件**（`Begin` / `LLM` / `Agent` / `Categorize` / `Switch` / ...），每个组件有自己的 `Param` schema，`Canvas.load()` 统一校验——本单元的 `TOOLS_DESC` 是把所有"组件声明 + 协议"塞进一段 prose prompt，加新工具就要改 prompt，RAGFlow 加新工具是加一个文件 + 注册一行；
- 组件既能当 canvas 节点跑、也能被 Agent 当工具调——"agent-as-tool"是它的复用哲学。

参考：[`docs/reference/ragflow-notes/agent_tools.md`](../../../../docs/reference/ragflow-notes/agent_tools.md)

## 思考题

**如果 LLM 没吐出 `Action:` 行（比如直接给了完整答案），本单元的处理是"打印原话 + observation 写 (no Action line parsed)"。这对 unit 02 的循环意味着什么？**

提示：unit 02 的 `run_agent` 会把这种"LLM 直接答"当成 `finish` 处理吗？还是会把它当成"JSON 解析失败"喂回 messages 让模型再试一轮？哪种行为更符合用户预期？