# s09 Agent 与工具 — 把"要不要查"这个决策还给 LLM

> **本章定位**:s09 是 RAG 的"控制面"。前面 s01-s08 把"先检索 → 拼 context → 调 LLM"做成了一条硬管线;本章把"**要不要走这条管线**"这个决策**从代码搬到 LLM 自己手里**——用 ReAct 循环 + 2 个工具(`retrieve` / `finish`)让模型自己挑走哪条路。详细定位见 s00 §1.4；RAGFlow 实现见本章末"## RAGFlow 实现"。

---

## 章节导航

| 序号 | 标题 | 入口 |
| --- | --- | --- |
| 01 | 工具调用(单步):把 `retrieve` / `finish` 写进 system prompt,让 LLM 自己挑工具 | [`code_01_tool_call.py`](code_01_tool_call.py) |
| 02 | ReAct 循环:Thought → Action → Observation 多步循环 + JSON 失败反馈 + `max_steps` 兜底(章节核心) | [`code_02_react_loop.py`](code_02_react_loop.py) |

跑法:

```bash
python s09_agent_tools/code_01_tool_call.py    # 单步工具调用(1 轮 LLM)
python s09_agent_tools/code_02_react_loop.py   # ReAct 循环(最多 5 轮)
```

依赖:复用 s05-s08 全部产出 + `openai` SDK(已在 requirements.txt);`LLM_API_KEY` 可选——无 key 时走 graceful-skip 分支,只展示 trace 形状,不真调 LLM。把 s08 跑通,s09 才能跑。

---

## 一、章节介绍

### 1.1 核心定义：什么是 Agent + 工具调用?

**Agent(智能体)** 在 RAG 语境里指的是一种**让 LLM 自己控制程序流的范式**——模型不再是"被喂 context 然后回答"的应答者,而是"看问题 → 决定要不要查 → 调工具 → 看结果 → 再决定"的决策者。它要解决一个硬管线解决不了的问题:**不是每个问题都该先检索**。

经典 RAG(s01-s08)用了一条固定管线:用户问 → 检索 → 拼 context → LLM 答。这条管线对"资料里能找到答案"的问题很有效,但对三类典型问题**反咬一口**:

- **简单问题**——"你好"、"1+1等于几"——根本没文档能查,检索是浪费,还可能把不相关片段塞进 prompt 把 LLM 带偏;
- **闲聊 / 已知知识**——"Python 是动态类型吗"——直接答就行,查文档反而会引入噪声和错误引用;
- **多跳 / 反问语境**——"刚才那个 PDF 第几页讲的内存?"——可能要先答后查,或者查了再查,固定管线处理不了。

**ReAct(Reason + Act)** 是当前最主流的 agent 范式之一:模型在每轮生成 `Thought / Action / ActionInput` 三行,代码解析这三行、路由到对应工具、把工具结果当 `Observation` 写回 messages,让模型下一轮再决定。**Thought 让模型"先想再做"——这是把决策从代码搬到 LLM 的关键**;没有 Thought 的纯 function-calling 容易"乱调工具",有 Thought 后模型至少在文本里能"先解释为什么调"。

#### 本章的工具集:`retrieve` + `finish`

agent 的**工具集**就是它能调用的"动作"——本章最小化到 2 个,够演示"决策权在 LLM"这个核心点:

| 工具名 | 签名 | 作用 | 触发场景 |
|---|---|---|---|
| `retrieve` | `(query: str) -> str` | 把 s05-s07 整条管线(embed → hybrid_search → rerank)打包成一次检索,返回 top-3 命中渲染的字符串 | 用户问题里**包含需要查资料的关键词** —— 服务器型号、产品规格、API 参数等 |
| `finish` | `(answer: str) -> str` | 把 `answer` 字段当最终答案返回,终止循环 | LLM 已经能答了(闲聊 / 已知知识) **或** 资料里查到了该 finish |

工具集写到 `TOOLS_DESC` 常量里,改了 prompt 不用动主循环——这是 MVP 的关键解耦:RAGFlow 用 OpenAI 兼容的 `tool_calls` 字段把这一步结构化了(模型直接返回 `name + arguments`,不再"打字"给代码),但本章 MVP 走 prompt 内嵌 + 正则解析,把"为什么这样"暴露得更明显。

#### ReAct 循环的"三行式"格式

每轮 LLM 输出长这样(示例,见 code_02 的 `run_agent` 主循环):

```
Thought: 用户问的是 R3630 G5 的内存插槽数量,文档里应该有。
Action: retrieve
ActionInput: {"query": "R3630 G5 内存插槽数量"}
```

代码侧正则 `r"Action:\s*(\w+)\b\s*ActionInput:\s*(.+)"` 抠出 `Action` + `ActionInput`,路由:

- `Action: retrieve` → 调 `_retrieve(payload["query"])` → 把返回的字符串当 `Observation` 写回 messages
- `Action: finish` → 返回 `payload["answer"]`,循环结束
- JSON 解析失败 → 把"原文"当 `Observation` 反馈回去,让 LLM 下一轮自己修正(code_02 的硬约束)
- `max_steps` 撞顶 → 强制返回 "Max steps reached.",防止死循环

**关键观察**:同一个 system prompt 下,模型**对"1+1等于几"会一轮直接 finish**、**对"R3630 G5 的内存插槽数量"会先 retrieve 再 finish**——"工具选择权"真的回到了 LLM 手里。这是从"硬管线"到"agent"的最小跨越。

#### Agent 与硬管线的本质区别

把它放进 RAG 全景看:**s01-s08 是"检索 → 生成"的固定链路**,**s09 把"要不要走这条链路"做成模型可决策的 step**。下面是同一问题的两种处理对比:

| 维度 | 硬管线 (s01-s08) | Agent (s09) |
|---|---|---|
| **流程** | 离线:索引 / 在线:无差别先检索 → 拼 prompt → LLM 答 | 离线:索引 / 在线:LLM 决策 → 调工具 → 拿 observation → 再决策 → ... → finish |
| **简单问题 ("1+1")** | 强制检索 → 把噪声塞进 prompt → LLM 容易答非所问 | LLM 判 "不需要查" → 直接 `finish` |
| **复杂问题 (多跳)** | 一次检索 + 一次生成,资料不够就答错 | LLM 多轮 retrieve + 反思 + 改写 query |
| **拒答** | prompt 软约束("答'我不知道'"),依赖模型听话 | LLM 自己选 `finish("我不知道")`,决策显式 |
| **失败模式** | 检索坏了整条管线塌 | retrieve 失败 → LLM 看到 observation 还能改写 query 再试 |
| **实现成本** | 100 行 prompt 拼装 | 30 行 TOOLS_DESC + 50 行 run_agent 循环 |

本章只演示**最小 agent**——2 个工具 + 5 轮上限 + 手写 prompt 解析;LangChain AgentExecutor / RAGFlow `agent/component/agent_with_tools.py` 把这套结构化了(用 `tool_calls` 字段、async DAG、Categize 失败跳走),见 §四。

### 1.2 真实世界的问题

`_retrieve(query)` 调起来 30 行,`run_agent(question)` 写完 50 行——加一起不到 100 行就能跑出"LLM 自己决定查不查"。看起来不值得单独一章。但把它放进 s08 的"先检索再答"硬管线对照看会发现:**"模型默认会怎么决策"和"我们需要模型怎么决策"之间也隔着一道悬崖**——这道悬崖由 3 类典型失败堆起来。

#### 真实世界的问题 (3 条典型)

1. **LLM 不停 retrieve / 不 finish**——LLM 训练数据里"helpful assistant 应该是先查再答"的偏置很强,又因为 retrieve 的 observation 总是非空(哪怕不相关),模型找不到"应该停"的信号,会一路 retrieve 到 `max_steps` 撞顶。**生产解法 3 层**:
   - ① `max_steps` 上限——MVP 已经做了,硬天花板,治不住"模型本来可以答对但就是停不下来";
   - ② prompt 软约束——在 `TOOLS_DESC` 里写"每个问题**最多调用一次 retrieve**,再调用一次还没有答案就用已有 observation 给出 finish,否则答'我不知道'并 finish";对主流模型(GPT-4o / Claude / Qwen-Max)效果不错,缺点是 prompt 越长越稀释;
   - ③ 重复 action 检测——运行时加"前两步 `Action / ActionInput` 对是不是一模一样"的检查,命中强制 `finish`(`observation = "我重复了同一次检索,无法获取更多新信息"`),硬约束、不依赖模型听话。**MVP 只做 ①**,② ③ 留作生产加固项。详见「思考题答案」。
2. **LLM 输出的 `Action: retrieve` 漏 `ActionInput` / JSON 解析失败**——模型可能吐 `Action: retrieve` 但漏 ActionInput,或者多写一段解释把 JSON 冲断。MVP 的 regex `r"Action:\s*(\w+)\b\s*ActionInput:\s*(.+)"` 是脆弱的——**生产里用 OpenAI / Anthropic 的 `tool_calls` 字段让 API 帮你解析**,模型不再"打字"给你,而是结构化返回参数(`tool_calls[0].function.name + arguments`)。RAGFlow 的 `LLMToolPluginCallSession` 走的就是这条路(见 `docs/reference/ragflow-notes/agent_tools.md`)。
3. **工具爆炸 / 选错工具**——工具一多 system prompt 装不下,模型选择准确率暴跌(每多 1 个工具 LLM 选择难度指数级上升)。**生产治理 3 招**:
   - ① **按"用户意图"分组路由**——先分类器决定走哪组工具(检索组 / 计算组 / 查询组 ...),不把所有工具一次塞给 LLM;
   - ② **工具描述写得"互斥"**——不同工具的描述之间**不重叠**,别让模型在"A 也能做、B 也能做"之间纠结;
   - ③ **拆成多层 agent(supervisor + sub-agent)**——sub-agent 只看到自己的工具集,supervisor 决定调哪个 sub-agent。RAGFlow 用 `Agent` 组件嵌套(`self._load_tool_obj` 接子组件)实现 "agent-as-tool"。

#### 为什么必须在 agent 上显式投入

每条失败模式都对应一种工业级解法——`max_steps` 兜底 + prompt 软约束 + 重复检测、`tool_calls` 结构化字段、意图路由 + 工具互斥 + 多层 agent。**s09 的目标不是解决它们,而是把它们显式暴露出来,让你看到硬管线的边界**。这跟 s08 把"toy prompt 在哪里会塌"显式对比是同一种思路——**叙述载体从"prompt 4 条硬约束"换成"agent 2 工具 + 5 轮循环"**,但"先跑通 toy,再讲清楚 toy 在哪里会塌"的教学哲学是一致的。

这也是为什么本章有 2 个代码文件而不是 1 个:

- **code_01**——跑通最小骨架(`TOOLS_DESC` + `_llm` + `_retrieve` + `single_shot(question)`),演示"1 轮 LLM 就能自己选工具"。把工具调用和 ReAct 循环拆成 2 段是为了让"LLM 真的能挑工具"和"LLM 怎么多轮推理"分两段讲——单步看到决策能力,多步看到循环控制。
- **code_02**——在 code_01 之上加 `run_agent(question, max_steps=5)` 主循环,演示完整 ReAct。复用 code_01 的所有底座(importlib 加载,代码零重复),新增的 30 行只关心"循环控制 + 终止条件 + JSON 失败反馈"。

---

## 二、工具调用(单步)：[code_01_tool_call.py](code_01_tool_call.py)

入口：[`code_01_tool_call.py`](code_01_tool_call.py)

把 2 个工具塞进 system prompt,调一次 LLM,正则抠 `Action / ActionInput`,跑工具。
code_02 会把这一轮包成 `Thought → Action → Observation` 循环,跑多步。

### 这是什么

本节只走**一轮** agent 决策:

1. `TOOLS_DESC` 把 2 个工具(`retrieve(query)` / `finish(answer)`)和输出格式(`Thought` / `Action` / `ActionInput`)写进 system prompt;
2. `_llm(messages)` 调 OpenAI 兼容接口(无 key 时降级,演示假设 retrieve),并剥掉 MiniMax / DeepSeek R1 的 `<think>...</think>` 推理块;
3. 用同款 regex `r"Action:\s*(\w+)\b\s*ActionInput:\s*(.+)"`(DOTALL)从 LLM 原话里抠出工具名 + 参数;
4. `single_shot(question)` 把原话、解析的 action、解析的 payload、工具返回的 observation 一并打印。

跑过 `python s09_agent_tools/code_01_tool_call.py` 会看到:问"内存插槽数量"时 LLM 选 `retrieve`;问"1+1 等于几"时 LLM 直接选 `finish`——**同一个 system prompt,LLM 自己决定要不要查文档**。

### 跑起来

```bash
python s09_agent_tools/code_01_tool_call.py
# 问: R3630 G5 的内存插槽数量
```

无 `LLM_API_KEY` 时打印演示(假设 LLM 选 retrieve 并跑检索管线):

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

### 它做对了什么

- **同一个 system prompt 里 LLM 会自己挑工具**:问"内存"它选 `retrieve`,问"1+1"它直接 `finish`——把"要不要查"的决策**还给 LLM** 是本章核心观察;
- **Action / ActionInput 解析**:regex 同时兼容"ActionInput 在新行 / 与 Action 同行 / 被 ```json 围栏包住"三种常见写法,Markdown 围栏剥掉再 `json.loads`;
- **JSON 解析失败兜底**:解析不出来不崩,返回 `(JSON 解析失败: ...)` 让调用方知道——为 code_02 的"反馈回 messages 让 LLM 自己修正"埋钩子;
- **self-contained**:内联 chroma + s04 embed + s06 hybrid + s07 rerank,不依赖 chapter-root。

### 它做错了什么

- **只能走一轮**:observation 出来就停了,没有"看了结果再决定下一步"的循环——这就是 code_02 要解决的;
- **没有 plan / execute 分层**:LLM 想回答复杂多跳问题(先查 A 再查 B 再总结)做不了;
- **LLM 输出格式仍然脆弱**:regex 抓不到时本节就退化成"打印原话"——生产里应该用 OpenAI / Anthropic 的 `tool_calls` 字段让 API 帮你解析;
- **没有死循环防护**:单步问题,但已经能看出"LLM 一直选 retrieve 不 finish"在多步场景下是定时炸弹。

---

## 三、ReAct 循环：[code_02_react_loop.py](code_02_react_loop.py)

入口：[`code_02_react_loop.py`](code_02_react_loop.py)

把 code_01 的"单步工具调用"包成循环,跑多步,JSON 失败时让 LLM 自己修正。

### 这是什么

本节是 s09 章节的**主要概念**——agent 循环:

1. 用 code_01 的 `TOOLS_DESC` / `_llm` / `_retrieve`(importlib 加载,不依赖 chapter-root);
2. `run_agent(question, max_steps=5)` 维护 messages 列表,每轮:调 LLM → regex 抠 `Action / ActionInput` → `json.loads` → 路由 `retrieve` 或 `finish` → 把结果写回 messages 当 `Observation`;
3. **JSON 解析失败**时把原文当 Observation 反馈回去("上一次 ActionInput 不是合法 JSON,原文已回显: ..."),让 LLM 下一轮自己修正;
4. **`max_steps=5` 兜底**——超过就返回 `"Max steps reached."`,防 LLM 死循环;
5. 返回 `{answer, trace}`,`trace` 是每轮的 `{step, thought, action, obs}`,便于打印 / 调试 / 单元测试。

跑 `python s09_agent_tools/code_02_react_loop.py` 会看到:问"内存插槽" → retrieve → finish 两步;问"1+1" → finish 一步直接结束。

### 跑起来

```bash
python s09_agent_tools/code_02_react_loop.py
# 问: R3630 G5 的内存插槽数量
```

无 `LLM_API_KEY` 时打印演示 trace:

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

### 它做对了什么

- **多步推理 + 多步工具调用**:LLM 看到 observation 后能决定下一步——查了 A 再查 B,或者查完直接 finish;
- **JSON 失败可恢复**:LLM 偶尔吐 `ActionInput: {query: 内存}`(少了引号)时,把原文回显当 Observation 让它下一轮自己改格式;
- **死循环兜底**:`max_steps=5` 是硬上限,超过就放弃——比"LLM 一直 retrieve 不 finish"无限耗下去强;
- **`trace` 结构化返回**:每轮的 thought / action / obs 都存下来,方便后续做日志、可视化、单元测试断言("应该 retrieve 至少 1 次");
- **复用 code_01**:本节不重复写工具描述、LLM 客户端、检索函数——只关心**循环控制**本身。

### 它做错了什么

- **没有 DAG 分支**:循环是单链——"先 retrieve 后 finish",没法表达"先 categorize,按类别走不同路径";
- **没有 plan-first**:LLM 不会先输出"我要先查 X 再查 Y"再执行——每轮只看上一步 observation 决定下一步,多跳问题容易跑偏;
- **没有并行工具**:一次只能调一个工具,不能"同时查 A 和 B";
- **没有"反思"组件**:observation 不相关时模型只能靠 prompt 里一句"找不到就 finish"自保,没有"主动改写 query 再试"的反射动作;
- **死循环防护弱**:只靠 `max_steps` 兜底,没检测"同一个 Action 重复出现 N 次"——`max_steps=5` 够短所以问题不显,但放大到 `max_steps=20` 就明显;
- **外部中断不支持**:没法像 RAGFlow 那样写 Redis `cancel` 标记让前端主动中断。

---

## 四、其他 / 整体设计取舍

### 跑起来

```bash
pip install openai flag-embedding sentence-transformers   # 已在 requirements.txt
python s09_agent_tools/code_02_react_loop.py
# 问: R3630 G5 的内存插槽数量
```

环境变量:

- `LLM_API_KEY` — OpenAI 兼容 API key;**可选**,无 key 时走 graceful-skip 分支,只打印 trace 形状 + 演示用 observation。
- `LLM_BASE_URL` — 默认 `https://api.openai.com/v1`,可换任意 OpenAI 兼容 endpoint(MiniMax / DeepSeek / 智谱 / 月之暗面等)。
- `LLM_MODEL` — 默认 `gpt-4o-mini`,可换 `MiniMax-M3`、`deepseek-chat` 等。
- `EMBED_MODEL` — 默认 `BAAI/bge-small-zh-v1.5`,同 s04-s08。

无 key / 离线环境跑 code_01 / code_02:

```bash
unset LLM_API_KEY && python s09_agent_tools/code_01_tool_call.py
# 走 graceful-skip:打印 trace 形状 + 假设的 LLM 输出,不调检索管线
```

### 跨代码文件的 schema 设计取舍

为什么 `TOOLS_DESC` 写成"2 工具 + 三行式"、而不是别的?几个常见取舍的折中:

- **2 工具 vs N 工具**——MVP 只放 `retrieve` + `finish` 两个,够演示"LLM 自己决策"。**多 1 工具,选择难度指数级上升**(LLM 需要在更多选项里挑对);**少 1 工具,agent 退化成硬管线**。生产里 5-8 个工具是经验上限,超过要拆多层 agent(§2.1 第 3 条)。
- **Thought vs 无 Thought**——`Thought: ...` 这一行**让模型"先解释为什么调"**,降低乱调工具的概率。代价是 token +20-30%(每轮多一句思考);收益是"模型偶尔选错工具"的可观测性 +300%——你能在 trace 里看到"模型为什么选 retrieve",production 调试时**直接看 Thought 比看 Action 更有用**。
- **prompt 内嵌工具 vs `tool_calls` 字段**——MVP 走 prompt 内嵌 + 正则解析;OpenAI / Anthropic 的 `tool_calls` 字段让 API 直接返回结构化 `name + arguments`,**正则解析的脆弱性归零**。代价是依赖特定 API(provider lock-in),且 prompt 里看不到工具描述(变成"代码侧 schema")。**MVP 选 prompt 内嵌**是为了把"工具描述怎么写"暴露在 README 里;**生产选 tool_calls** 是为了鲁棒性。
- **`max_steps=5` 兜底 vs 无上限**——LLM 没"应该停"的信号时可能死循环,**`max_steps` 是必须的硬天花板**。经验值 3-7 步:少于 3 步,多跳问题答不完;多于 7 步,token 成本指数级上升且准确率反降(模型在长 context 里"忘掉"自己应该 finish)。MVP 选 5 是 RAGFlow `max_rounds=5` 的同款默认值。
- **JSON 失败反馈 vs 直接报错**——MVP 的 `run_agent` 在 `json.loads` 失败时**把"原文"当 Observation 写回 messages**,让 LLM 下一轮自己修正(给 `obs = "上一次 ActionInput 不是合法 JSON,原文已回显: ... 请严格按规范输出 JSON。"`)。**不直接报错**是为了让 agent 在"模型偶尔吐错 JSON"时自愈——生产里 5-10% 的轮次会触发这种自愈路径,直接报错会让用户体验断崖式下降。
- **graceful skip vs hard fail**——同 s08,`run_agent` 在无 `LLM_API_KEY` 时返回 "Max steps reached." + 演示 trace 形状,**不抛异常**。pipeline 在没配 key 的环境下也能跑、只展示 agent 决策形状。生产上**应该 fail-fast**,但教学 demo 走 graceful skip 让初学者少踩坑。

### 如何扩展更多 agent 范式

加一种 agent 策略(`tool_calls` 结构化字段 / 重复 action 检测 / 多层 agent supervisor)只要三步:

1. 把 `_llm(messages)` 换成支持 `tool_calls` 的版本(`client.chat.completions.create(..., tools=tool_schemas)`),从 `resp.choices[0].message.tool_calls` 取 `name + arguments` 替代正则解析;
2. 在 `run_agent` 主循环里加 "前两步 `Action / ActionInput` 对是不是一模一样" 的检查,命中强制 `finish`;
3. 把 `run_agent` 拆成 `supervisor(question)`(决定调哪个 sub-agent)+ `sub_agent(query, tools)`(只看到自己工具集),sub-agent 通过 `tool_calls` 嵌套调用。

不要在 `run_agent` 里写 `if mode == "single": ... elif mode == "structured": ...` 之类分发——它会污染单一职责。`run_agent` 只懂 prompt 内嵌 + 正则,`main()` 懂全 agent 模式。本章 MVP 只跑 prompt 内嵌,但接口形状留好了。

### 实际跑出来的 trace 形状

把 code_02 跑在仓库自带的 `samples/` 上,`run_agent` 返回的 trace 长这样(实测,`MiniMax-M3 over minimaxi.com`):

```
Q: R3630 G5 的内存插槽数量

--- step 1 ---
Thought: 用户问的是 R3630 G5 的内存插槽数量,文档里应该有。
Action:  retrieve
Obs:     - (server_whitepaper.pdf#1) 二、关键特性 计算密度:单台 2U 机箱内集成两颗处理器、32 条内存 DIMM 与 10 个 PCIe 4.0 扩展槽位...

--- step 2 ---
Thought: 找到了。
Action:  finish
Obs:     R3630 G5 配备 32 个 DIMM 内存插槽。

A: R3630 G5 配备 32 个 DIMM 内存插槽。
```

跳过检索的对照("1+1等于几"):

```
Q: 1+1等于几
--- step 1 ---
Thought: 这是个简单算术,不需要查文档。
Action:  finish
Obs:     1+1等于2。

A: 1+1等于2。
```

**关键观察**:同一个 system prompt 下,模型**对"1+1等于几"会一轮直接 finish**、**对"R3630 G5 的内存插槽数量"会先 retrieve 再 finish**——"工具选择权"真的回到了 LLM 手里。这就是从"硬管线"到"agent"的最小跨越。

无 `LLM_API_KEY` 时(`graceful-skip`):

```
Q: R3630 G5 的内存插槽数量
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

### troubleshooting

- `openai.AuthenticationError`: `LLM_API_KEY` 没设或失效;`.env` 加 `LLM_API_KEY=sk-...` 兜底,或 `unset LLM_API_KEY` 走 graceful-skip 分支。
- `openai.APIConnectionError`: 网络不可达;设 `LLM_BASE_URL=https://...` 走代理,或暂时 `unset LLM_API_KEY` 验证非 LLM 链路(retrieval/rerank/agent 循环)正常。
- `UnicodeEncodeError: 'gbk' codec can't encode character`: Windows 控制台编码问题,跑前 `set PYTHONIOENCODING=utf-8`(s05-s09 同问题)。
- `LLM 一路 retrieve 不 finish`: 撞 `max_steps=5` 兜底,见「思考题答案」第 1 题的 3 种解法(`max_steps` + prompt 软约束 + 重复 action 检测)。
- `LLM 输出 `Action: retrieve` 但漏 ActionInput`: code_02 会把"原文"当 Observation 反馈回去让模型下一轮自愈;若连续失败,把 `TOOLS_DESC` 里的格式说明加粗(`**严格按以下格式**`)提升遵从度。

---

## RAGFlow 实现

RAGFlow 的 Agent 在 `agent/` 目录下用 Canvas DAG 编排：每个节点是一个 tool 或 LLM 调用，节点之间用 `bind_tools()` 绑定依赖。Canvas 不强制按 linear 顺序执行，可以根据 query 类型走不同路径（FAQ / RAG / 工具调用）。

**设计取舍**：Canvas 把"按 query 走不同路径"做成可视化 DAG，而不是 hard-coded 流程。开发者可以在 UI 里拖拽节点、加边、调权重，不改代码就能调整 agent 行为。s09 toy 的 ReAct 循环是单线直连版，Canvas 是它的可视化升级。

详细摘录与 5-15 行 "为什么这样写" 的分析见 [`docs/reference/ragflow-notes/agent_tools.md`](../docs/reference/ragflow-notes/agent_tools.md)。

---

## 五、其他 / 选型与思考题

### 主流 agent 范式速览

下面这张表把 RAG 系统的 agent 范式按"工具描述形式 / 解析方式 / 死循环防护 / 工具数量"列出来:

| 范式 | 工具描述形式 | 解析方式 | 死循环防护 | 工具数量 | 适用场景 |
|---|---|---|---|---|---|
| **手写 ReAct (本章 MVP)** | system prompt 内嵌文字 | 正则抠 `Action / ActionInput` | `max_steps=5` 一道线 | 2-5 个 | 教学 / 快速原型 / 离线可复现 |
| **OpenAI `tool_calls` 字段** | 代码侧 schema(JSON) | API 直接返回 `tool_calls[0].function` | `tool_choice` + `max_rounds` | 5-15 个 | 生产单 agent(无嵌套) |
| **RAGFlow Canvas DAG** | DSL 存工作流 + `bind_tools` | 异步生成器按 path 跑 | `max_rounds=5` + `is_canceled()` + `Categorize` | 10-30 个(嵌套) | 生产多 agent / 多租户 / UI 编排 |
| **LangChain AgentExecutor** | `Tool` 类 + `tools=[]` | `agent_scratchpad` 解析 | `max_iterations` + `early_stopping_method` | 5-10 个 | 已有 LangChain 栈的工程 |
| **LlamaIndex ReActAgent** | `QueryEngineTool` 包装 | 同 LangChain | `max_iterations` + verbose trace | 5-10 个 | 已有 LlamaIndex 栈的工程 |

我们的 toy `run_agent` 在范式复杂度上只占第一行——**手写 ReAct**;RAGFlow 走完整 DAG,**多一道抽象就多一道观测点 + 一个失败模式**。教学 demo 选 MVP 因为它跑通快、依赖少、依赖全在 prompt 里可见;**生产请按"可观测性 vs 复杂度"做 tier 选型**(MVP → `tool_calls` → RAGFlow DAG → 多层 agent)。

### 选型速记

- **教学 / 快速原型 / 离线可复现** → 本章 MVP (手写 ReAct + prompt 内嵌 + graceful skip),无 API key 也能跑 trace 形状,代码 ≤ 200 行;
- **生产单 agent(无嵌套)** → 切 OpenAI / Anthropic `tool_calls` 字段,正则解析归零,工具数可以涨到 10-15 个,代码 +50 行换 +300% 鲁棒性;
- **生产多 agent / 多租户 / UI 编排** → RAGFlow Canvas DAG,异步生成器 + `Categorize` 失败跳走,工具数 10-30 个,实现成本 10x 但可观测性 +10x;
- **已有 LangChain / LlamaIndex 栈** → 复用框架的 `AgentExecutor` / `ReActAgent`,不重写 prompt 解析、不自己接 LLM 客户端;
- **要先看清每个边界再选** → 用本章 code_02 把"手写 ReAct"和"加重复 action 检测"各跑一次,对比"模型一路 retrieve 不 finish"的稳定性——这是最简单的"agent A/B"实验。

### 思考题

1. **如果模型一直选 retrieve 不 finish 怎么办?**
2. **ReAct 和 function-calling 的本质区别是什么?**
3. **工具一多 LLM 选错怎么办?**

(答案见文末「思考题答案」)

---

## 思考题答案

### Q1. 如果模型一直选 retrieve 不 finish 怎么办?

这是个真实的失败模式——LLM 训练数据里"helpful assistant 应该是先查再答"的偏置很强,又因为 retrieve 的 observation 总是非空(哪怕不相关),模型找不到"应该停"的信号,会一路 retrieve 到 `max_steps` 撞顶。

按代价从低到高排三种解法:

**1) `max_steps` 上限**(MVP 已经做了)——硬天花板。最坏情况是用户看到"Max steps reached."。优点:简单、零误判;缺点:用户体验差,它治不住"模型本来可以答对但就是停不下来"。

**2) 在 system prompt 强调"必须 finish"**——把 `TOOLS_DESC` 里的工具描述改成更显式的"每个问题**最多调用一次 retrieve**,再调用一次 retrieve 还没有答案就用已有的 Observation 给出 finish,否则就回答'我不知道'并 finish"。这是**软约束**,依赖模型遵从指令,但对主流模型(GPT-4o / Claude / Qwen-Max)效果不错;缺点是 prompt 越长越稀释,工具一多就顾不过来。

**3) 检测重复 Action**——运行时加一道"刚才两步的 `Action / ActionInput` 对是不是一模一样"的检查,命中就**强制 finish**(observation 是 `"我重复了同一次检索,无法获取更多新信息"`)。这是**硬约束**,不依赖模型听话。MVP 的 `max_steps=5` 实际是它的弱化版(撞 5 步就终止,不管内容)。

**生产推荐**:1 + 2 + 3 都做——`max_steps` 是兜底,prompt 软约束是主力,重复检测是保险。RAGFlow 三种都做(`max_rounds=5` + system prompt 强调 + `is_canceled()` 主动取消 + `Categorize` 组件做"这条路走不通跳走",本质就是"用结构化组件替代字符串解析里的软约束")。

### Q2. ReAct 和 function-calling 的本质区别是什么?

ReAct **每轮让模型先输出 Thought(思考过程)再选 Action**,function-calling **直接返回结构化 `tool_calls` 字段、不暴露思考过程**。ReAct 的优势是"可观测性 +300%"(trace 里能看到模型为什么选这个工具),代价是 token +20-30% + 解析脆弱(模型可能吐错格式);function-calling 的优势是"API 帮你解析、鲁棒性归零",代价是"看不到 Thought,production 调试更难"、依赖特定 provider。**MVP 选 ReAct** 是为了把"决策过程"暴露在 README 里;**生产可切 function-calling** 换鲁棒性。

### Q3. 工具一多 LLM 选错怎么办?

三招:① **按"用户意图"分组路由**——先分类器决定走哪组工具(检索组 / 计算组 / 查询组),不把所有工具一次塞给 LLM;② **工具描述写得"互斥"**——不同工具描述之间不重叠,别让模型在"A 也能做、B 也能做"之间纠结;③ **拆多层 agent**——supervisor 决定调哪个 sub-agent,sub-agent 只看到自己工具集,RAGFlow 用 `Agent` 组件嵌套实现 "agent-as-tool"。
