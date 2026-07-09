# s09 / Unit 02 — ReAct 循环：Thought → Action → Observation

> 由浅入深第 2 步：把 unit 01 的"单步工具调用"包成循环，跑多步，JSON 失败时让 LLM 自己修正。  
> chapter-root 的 `s09_agent_tools/code.py` 聚合入口**委托**给本单元（章节的核心概念就是循环）。

## 这是什么

本单元是 s09 章节的**主要概念**——agent 循环：

1. 用 unit 01 的 `TOOLS_DESC` / `_llm` / `_retrieve`（importlib 加载，不依赖 chapter-root）；
2. `run_agent(question, max_steps=5)` 维护 messages 列表，每轮：调 LLM → regex 抠 `Action / ActionInput` → `json.loads` → 路由 `retrieve` 或 `finish` → 把结果写回 messages 当 `Observation`；
3. **JSON 解析失败**时把原文当 Observation 反馈回去（"上一次 ActionInput 不是合法 JSON，原文已回显: ..."），让 LLM 下一轮自己修正；
4. **`max_steps=5` 兜底**——超过就返回 `"Max steps reached."`，防 LLM 死循环；
5. 返回 `{answer, trace}`，`trace` 是每轮的 `{step, thought, action, obs}`，便于打印 / 调试 / 单元测试。

跑 `python s09_agent_tools/units/02_react_loop/code.py` 会看到：问"内存插槽" → retrieve → finish 两步；问"1+1" → finish 一步直接结束。

## 跑起来

```bash
python s09_agent_tools/units/02_react_loop/code.py
# 问: R3630 G5 的内存插槽数量
```

无 `LLM_API_KEY` 时打印演示 trace：

```
[Q] R3630 G5 的内存插槽数量

[skipped: LLM_API_KEY not set] — 演示 trace 形状:

--- step 1 ---
Thought: 用户问的是 R3630 G5 的内存插槽数量。
Action:  retrieve
Obs:     - (server_whitepaper.pdf#2) 配备 32 个 DIMM 内存插槽 ...

--- step 2 ---
Thought: 找到了。
Action:  finish
Obs:     R3630 G5 配备 32 个 DIMM 内存插槽。

[A] R3630 G5 配备 32 个 DIMM 内存插槽。
```

## 它做对了什么

- **多步推理 + 多步工具调用**：LLM 看到 observation 后能决定下一步——查了 A 再查 B，或者查完直接 finish；
- **JSON 失败可恢复**：LLM 偶尔吐 `ActionInput: {query: 内存}`（少了引号）时，把原文回显当 Observation 让它下一轮自己改格式；
- **死循环兜底**：`max_steps=5` 是硬上限，超过就放弃——比"LLM 一直 retrieve 不 finish"无限耗下去强；
- **`trace` 结构化返回**：每轮的 thought / action / obs 都存下来，方便后续做日志、可视化、单元测试断言（"应该 retrieve 至少 1 次"）；
- **复用 unit 01**：本单元不重复写工具描述、LLM 客户端、检索函数——只关心**循环控制**本身。

## 它做错了什么

- **没有 DAG 分支**：循环是单链——"先 retrieve 后 finish"，没法表达"先 categorize，按类别走不同路径"；RAGFlow 的 `Categorize` + `Switch` 组件支持多路分发；
- **没有 plan-first**：LLM 不会先输出"我要先查 X 再查 Y"再执行——每轮只看上一步 observation 决定下一步，多跳问题容易跑偏；
- **没有并行工具**：一次只能调一个工具，不能"同时查 A 和 B"——RAGFlow 的 `Iteration` 组件能 fan-out；
- **没有"反思"组件**：observation 不相关时模型只能靠 prompt 里一句"找不到就 finish"自保，没有"主动改写 query 再试"的反射动作；
- **死循环防护弱**：只靠 `max_steps` 兜底，没检测"同一个 Action 重复出现 N 次"——`max_steps=5` 够短所以问题不显，但放大到 `max_steps=20` 就明显；
- **外部中断不支持**：没法像 RAGFlow 那样写 Redis `cancel` 标记让前端主动中断。

## 对照 ragflow 怎么做的

RAGFlow 的 agent 不是字符串解析循环，是**可插拔 DAG + 结构化 tool_calls**：

- `agent/canvas.py` 的 `Canvas` 继承自 `Graph`，`Canvas.load()` 把 DSL（JSON 格式的工作流定义）解析成 `self.components`（节点字典）+ `self.path`（执行顺序数组）；`Canvas.run` 是个**异步生成器**按 `path` 顺序依次 yield `node_started` / `message` / `node_finished` 事件给前端流式渲染；
- `agent/component/` 下面是 30 多个**声明式组件**（`Begin` / `LLM` / `Agent` / `Categorize` / `Switch` / `Iteration` / `Loop` / `Message` / ...），每个组件有自己的 `Param` schema，`Canvas.load()` 统一校验——加新工具 = 加一个文件 + 注册一行，不用改循环；
- `agent/component/agent_with_tools.py` 的 `Agent.__init__` 把工具喂给 LLM 客户端（`bind_tools`），LLM 返回结构化 `tool_calls[].function.name + arguments`，客户端查 `self.tools[name]` 调实例——不是从 LLM 自由文本里 regex 抠；
- 死循环的三道防线：① `LLMBundle(..., max_rounds=5)`（LLM 调用次数硬上限）；② `is_canceled()` 检查（写 Redis 的 `cancel` 标记，前端可主动中断）；③ `Categorize` 组件的 `_extend_path` 让一条路走不通时跳到另一条；
- planner 是 `Categorize` + `Switch`（多路分发）+ `Iteration` / `Loop`（带状态的任务队列）；组件之间通过 `{component_id@output_var}` 这样的 DSL 变量引用解耦，`Canvas.get_variable_value` 在执行时按需解析——本质是把 MVP 写死的 Python 控制流全部**数据化**了，可以 UI 编辑、可以保存、可以版本控制。

参考：[`docs/reference/ragflow-notes/agent_tools.md`](../../../../docs/reference/ragflow-notes/agent_tools.md)

## 思考题

**如果模型第二轮 retrieve 的 observation 还是空（检索没命中），单元会怎么样？是无害继续、还是触发死循环兜底？**

提示：看 `run_agent` 的循环体——它把 observation 当 user message 回写、**不**做"空就强制 finish"的检测。如果 retrieve 一直返回 `""`、LLM 又不主动 finish，会跑到 `max_steps=5` 才放弃。RAGFlow 是怎么用 `Categorize` 组件破这个局的？