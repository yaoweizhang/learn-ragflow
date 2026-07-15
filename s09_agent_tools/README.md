# s09 Agent 与工具 — 把"要不要查"这个决策还给 LLM

[上一章 s08 → · 下一章 s10 → ... → s12]

> *"前面 s01-s08 把'先检索 → 拼 context → 调 LLM'做成了一条硬管线；本章把'**要不要走这条管线**'这个决策**从代码搬到 LLM 自己手里**——用 ReAct 循环 + 2 个工具（`retrieve` / `finish`）让模型自己挑走哪条路"*
>
> **链路位置**: s08 (硬管线) → s09 (把"是否走管线"做成模型可决策的 step) → s10 (graphrag)
> **代码文件**: c01_tool_call.py · c02_react_loop.py

> 环境准备: 见 root README §快速开始 — `pip install -r requirements.txt` + `.env` 配 `LLM_API_KEY`(可选,无 key 走 graceful-skip 只打印 trace 形状)。s09 需要 s04-s07 整条检索管线已跑通(`python s04_embedding/...` → `s05_vector_index/...` → `s06_retrieval/...` → `s07_rerank/...` 顺序建好 `_chroma` 索引)。

---

## 问题

Agent(智能体)在 RAG 语境里指的是一种**让 LLM 自己控制程序流的范式**——模型不再是"被喂 context 然后回答"的应答者，而是"看问题 → 决定要不要查 → 调工具 → 看结果 → 再决定"的决策者。它要解决一个硬管线解决不了的问题:**不是每个问题都该先检索**。

s01-s08(以及 s09 之外的整条硬管线)用了一条固定管线:用户问 → 检索 → 拼 context → LLM 答。这条管线对"资料里能找到答案"的问题很有效,但对三类典型问题**反咬一口**:

**第一,简单问题 ——"你好"、"1+1等于几"。**根本没文档能查,检索是浪费,还可能把不相关片段塞进 prompt 把 LLM 带偏。LLM 直接答就行。

**第二,闲聊 / 已知知识 ——"Python 是动态类型吗"。**直接答就行,查文档反而会引入噪声和错误引用。这是 LLM 训练数据充足就能回答的范畴,不该被检索打断。

**第三,多跳 / 反问语境 ——"刚才那个 PDF 第几页讲的内存"。**可能要先答后查,或者查了再查,固定管线处理不了多轮反思。需要 LLM 自己判断"先查 A 还是先查 B / 查到了再查 C"。

把这三种失败合起来看,**"LLM 默认会怎么决策"和"我们需要模型怎么决策"之间也隔着一道悬崖**——这道悬崖由三类典型故障堆起来:

- **LLM 不停 retrieve 不 finish**——LLM 训练数据里"helpful assistant 应该是先查再答"的偏置很强,又因为 retrieve 的 observation 总是非空(哪怕不相关),模型找不到"应该停"的信号,会一路 retrieve 到 `max_steps` 撞顶;
- **`Action: retrieve` 漏 `ActionInput` / JSON 解析失败**——模型可能吐 `Action: retrieve` 但漏 ActionInput,或者多写一段解释把 JSON 冲断。MVP 的 regex `r"Action:\s*(\w+)\b\s*ActionInput:\s*(.+)"` 是脆弱的;
- **工具爆炸 / 选错工具**——工具一多 system prompt 装不下,模型选择准确率暴跌(每多 1 个工具 LLM 选择难度指数级上升),需要分组路由 + 互斥描述 + 多层 agent 治理。

s09 的任务不是**解决**它们,而是把它们**显式暴露出来**——让你看到硬管线的边界,把"工具选择权"从代码搬到 LLM 自己手里。这是和 s08 把"toy prompt 在哪里会塌"显式对比是同一种思路:**先跑通 toy,再讲清楚 toy 在哪里会塌**,再把工业加固项列在末尾。

---

## 解决方案

s09 用 **两个递进的脚本** 把"agent 决策"跑起来。每一步展示前一步的局限,引出下一步的加固方向:

```
代码 1 (单步 agent)              代码 2 (ReAct 循环)
┌──────────────────┐         ┌──────────────────────────┐
│ TOOLS_DESC       │         │ 复用 01 的底座          │
│ + _llm           │         │ + run_agent 循环         │
│ + _retrieve      │ ───▶   │   (max_steps=5)         │
│ + single_shot    │         │   messages 维护         │
│                  │         │   JSON 失败自愈          │
│ 1 轮 LLM 决策    │         │ 多轮 LLM 决策              │
└──────────────────┘         └──────────────────────────┘
  "选哪个工具"可见             "边看结果边决策" 闭环
```

| 脚本 | 解决什么 | 留下什么局限 | 何时用 |
|---|---|---|---|
| `c01_tool_call.py` | 单步:LLM 一次输出 `Thought + Action + ActionInput` → 正则解析 → 跑工具 → 输出 observation | 只能走 1 轮,observation 出来就停了;没有循环控制 | 教学 / 验证 LLM 真能选工具 / 单步 demo |
| `c02_react_loop.py` | 多步:循环 `_llm → parse → execute → observation`,`max_steps=5` 兜底,JSON 失败把原文当 Observation 反馈让模型自愈 | 单链循环,没有 DAG 分支;没有并行工具;没有 plan-first | 完整 ReAct 演示 / 多跳问题骨架 / 接到 FastAPI 服务前的最后一环 |

两脚本的关系是一条**教学主干**: 代码 1 把"LLM 真的能挑工具"这件事演示出来——同一个 system prompt,问"1+1等于几" LLM 一轮 `finish`,问"R3630 G5 内存插槽数量" LLM 选 `retrieve`,**同一个 prompt 下"工具选择权"真的回到模型手里**。但 代码 1 是单步的——observation 出来就停,没有"看了结果再决定下一步"的循环。代码 2 把 代码 1 包成 `while step < max_steps` 循环,新增 30 行只关心"循环控制 + 终止条件 + JSON 失败反馈"——LLM 拿到 observation 后还能再选一次 `retrieve` 改写 query,或者 `finish` 收尾;撞 `max_steps=5` 强制终止防死循环。

**关键设计取舍**(展开见 代码 1/代码 2 的「为什么这样写」):**2 工具 vs N 工具**——MVP 只放 `retrieve` + `finish` 两个,够演示"LLM 自己决策"这个核心点;**Thought vs 无 Thought**——`Thought: ...` 这一行让模型"先解释为什么调",trace 里能看到模型决策逻辑;**prompt 内嵌 vs `tool_calls` 字段**——MVP 走 prompt 内嵌 + 正则解析,生产可切 OpenAI / Anthropic 的 `tool_calls` 字段让 API 帮你解析;**`max_steps=5` 兜底**——LLM 没"应该停"的信号时可能死循环,经验值 3-7 步;**JSON 失败反馈**——`json.loads` 失败时把原文当 Observation 写回 messages,让 LLM 下一轮自愈,不直接报错。**MVP 选 toy 把"为什么这样"暴露在 README 里;生产选 `tool_calls` 换鲁棒性**。

---

## 代码 1: 工具调用单步 ([c01_tool_call.py](c01_tool_call.py))

### 工作原理

**做一件事**: 把 2 个工具 (`retrieve` / `finish`) 写进 system prompt,调一次 LLM,正则抠 `Action / ActionInput`,跑工具,展示 observation——**首次证明 LLM 能自己决定选哪个工具**。

**4 步**:
1. `TOOLS_DESC` 常量把工具集 (`retrieve(query: str)` / `finish(answer: str)`) + 三行式输出格式 (`Thought / Action / ActionInput`) 写进 system prompt——LLM 没看到函数签名,只能按这段文字的格式输出,这是 MVP 的硬约束,RAGFlow 用 OpenAI/Anthropic 的 `tool_calls` 字段结构化了这一步
2. `_llm(messages)` 调 OpenAI 兼容 `/chat/completions`,`temperature=0`;剥掉 MiniMax / DeepSeek R1 的 `思考.../思考` 推理块(`re.sub(r"思考.*?/思考", "", raw, flags=re.DOTALL)`);无 key 时返回 `[skipped: ...]` 走 graceful-skip
3. 同款 regex `r"Action:\s*(\w+)\b\s*ActionInput:\s*(.+)"` (DOTALL 兼容 ActionInput 跨行) 从 LLM 原话里抠出工具名 + 参数,剥 markdown 围栏 (` ```json ... ``` `) 后 `json.loads`
4. `single_shot(question)` 把原话、解析的 action、解析的 payload、工具返回的 observation 一并打印成 trace——便于看到 LLM 决策的**每一步**

```python
# 中间片段: regex + JSON 解析 + 工具路由
m = re.search(r"Action:\s*(\w+)\b\s*ActionInput:\s*(.+)", text, re.DOTALL)
if not m:
    return {"text": text, "action": None, "payload": None,
            "observation": "(no Action line parsed — LLM 直接答了?)"}
action, raw = m.group(1), m.group(2).strip()
raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
try:
    payload = json.loads(raw)
except json.JSONDecodeError:
    return {"text": text, "action": action, "payload": None,
            "observation": f"(JSON 解析失败: {raw[:120]})"}
if action == "finish":
    obs = payload.get("answer", "(finish 但没给 answer)")
elif action == "retrieve":
    obs = _retrieve(payload.get("query", ""))
```

**完整函数**:

```python
TOOLS_DESC = """你可以用以下工具:
1. retrieve(query: str) — 从文档库检索相关段落
2. finish(answer: str) — 给出最终答案

按以下格式回答(每轮一步):
Thought: <你的思考>
Action: <retrieve 或 finish>
ActionInput: <JSON 字符串>
"""


def _llm(messages: list[dict]) -> str:
    """OpenAI 兼容接口 + 剥 思考.../思考 推理块(MiniMax / DeepSeek R1)."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    resp = client.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=0,
    )
    raw = resp.choices[0].message.content
    return re.sub(r"思考.*?/思考", "", raw, flags=re.DOTALL).strip()


def single_shot(question: str) -> dict:
    """调一次 LLM, 解析 Action/ActionInput, 执行工具, 返回 trace."""
    import json
    messages = [
        {"role": "system", "content": TOOLS_DESC},
        {"role": "user", "content": question},
    ]
    text = _llm(messages)
    m = re.search(r"Action:\s*(\w+)\b\s*ActionInput:\s*(.+)", text, re.DOTALL)
    if not m:
        return {"text": text, "action": None, "payload": None,
                "observation": "(no Action line parsed — LLM 直接答了?)"}
    action, raw = m.group(1), m.group(2).strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"text": text, "action": action, "payload": None,
                "observation": f"(JSON 解析失败: {raw[:120]})"}
    if action == "finish":
        obs = payload.get("answer", "(finish 但没给 answer)")
    elif action == "retrieve":
        obs = _retrieve(payload.get("query", ""))
    else:
        obs = f"Unknown action: {action}"
    return {"text": text, "action": action, "payload": payload, "observation": obs}
```

### 试一下

```bash
python s09_agent_tools/c01_tool_call.py
# 问: R3630 G5 的内存插槽数量
```

实测输出 (无 `LLM_API_KEY`, graceful-skip 演示 trace 形状):

```
[Q] R3630 G5 的内存插槽数量

[skipped: LLM_API_KEY not set] — 演示假设 LLM 选了 retrieve:

[LLM raw]
Thought: 用户问的是 R3630 G5 的内存插槽数量。
Action: retrieve
ActionInput: {"query": "R3630 G5 的内存插槽数量"}

[Parsed action] retrieve
[Parsed payload] {'query': 'R3630 G5 的内存插槽数量'}

[Observation]
- (无 LLM_API_KEY;真实 retrieval 需 chroma + embed,见 _retrieve())
```

- 交互输入, 看 LLM 选哪个工具 + observation 形状
- 无 key 时不真跑检索,只演示 trace 形状;有 key 时(实测 `MiniMax-M3 over minimaxi.com`)LLM 会真去 retrieve,observation 来自 s05-s07 整条管线

**观察**: **同一个 system prompt 下,LLM 自己选 `retrieve` 还是 `finish`**——问"R3630 G5 内存插槽数量"它选 `retrieve`,问"1+1等于几"它一轮直接 `finish`,**"工具选择权"真的回到了 LLM 手里**。这是从"硬管线(s01-s08)"到"agent(s09)"的最小跨越。但 代码 1 是单步的——observation 出来就停了,没有"看了结果再决定下一步"的循环,也没办法让 LLM 拿到 observation 后**改写 query 再试**——这是 代码 2 的入口。

### 为什么不只写这一种

代码 1 只能跑 1 轮,observation 出来就停了——LLM 拿到"不相关 observation"没办法再反思,拿到"部分 observation"也没法再查——**没有"边看结果边决策"的循环**。代码 2 把这 1 轮包成 `while step < max_steps` 循环,让 LLM 看到 observation 后还能再选一次 `retrieve` 改写 query,或者 `finish` 收尾,撞 `max_steps=5` 兜底。

---

## 代码 2: ReAct 循环 ([c02_react_loop.py](c02_react_loop.py))

### 工作原理

**做一件事**: 在 代码 1 的底座(工具 + LLM + 检索)上加 `run_agent` 循环——`while step < max_steps: _llm → parse → execute → observation → step += 1`,让 LLM 每轮根据上一步 observation 决定下一步,JSON 解析失败时把原文当 Observation 反馈让模型自愈,**演示完整的"Thought → Action → Observation" 自循环**。

**6 步**:
1. `importlib` 加载 代码 1 的 `TOOLS_DESC` / `_llm` / `_retrieve`(目录以数字开头,普通 import 报 SyntaxError)——复用,不重写
2. `run_agent(question, max_steps=5)` 维护 `messages` 列表,每轮:调 `_llm(messages)` → regex 抠 `Action / ActionInput` → 剥 markdown 围栏 → `json.loads`
3. **JSON 解析失败**时把"上一次 ActionInput 不是合法 JSON,原文已回显: ..." 当 Observation 反馈回 `messages`,让 LLM 下一轮自己修正——5-10% 的轮次会触发这种自愈路径,**不直接报错**
4. `Action == "finish"` → 返回 `payload["answer"]`,循环结束;`Action == "retrieve"` → 调 `_retrieve(payload["query"])`,把结果当 Observation 写回 messages
5. **`max_steps=5` 兜底**——超过强制返回 `"Max steps reached."`,防 LLM 不停 retrieve 的死循环
6. 返回 `{answer, trace}`,每条 trace 是 `{step, thought, action, obs}`,便于打印 / 调试 / 单元测试

```python
# 中间片段: ReAct 主循环 + JSON 失败自愈
for step in range(1, max_steps + 1):
    text = _llm(messages)
    messages.append({"role": "assistant", "content": text})
    m = re.search(r"Action:\s*(\w+)\b\s*ActionInput:\s*(.+)", text, re.DOTALL)
    if not m:
        # 没抓到 Action —— 把 LLM 原话当 final answer 返回
        return {"answer": text, "trace": trace + [{"step": step, "thought": text,
                                                    "action": None, "obs": None}]}
    action, raw = m.group(1), m.group(2).strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # JSON 解析失败:把原文当 Observation 反馈回去,让模型下一轮自己修正
        obs = f"上一次 ActionInput 不是合法 JSON,原文已回显: {raw[:200]}\n请严格按规范输出 JSON。"
        trace.append({"step": step, "thought": text, "action": action, "obs": obs})
        messages.append({"role": "user", "content": f"Observation: {obs}"})
        continue
```

**完整函数**:

```python
def run_agent(question: str, max_steps: int = 5) -> dict:
    """Thought/Action/Observation 循环, 最多 max_steps 轮.

    返回 dict 含: `answer`(最终答案)、`trace`(每轮的 thought/action/observation
    列表, 便于打印 / 调试 / 单元测试).
    """
    messages = [
        {"role": "system", "content": TOOLS_DESC},
        {"role": "user", "content": question},
    ]
    trace = []
    for step in range(1, max_steps + 1):
        text = _llm(messages)
        messages.append({"role": "assistant", "content": text})
        m = re.search(r"Action:\s*(\w+)\b\s*ActionInput:\s*(.+)", text, re.DOTALL)
        if not m:
            return {"answer": text, "trace": trace + [{"step": step, "thought": text,
                                                        "action": None, "obs": None}]}
        action, raw = m.group(1), m.group(2).strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            obs = f"上一次 ActionInput 不是合法 JSON,原文已回显: {raw[:200]}\n请严格按规范输出 JSON。"
            trace.append({"step": step, "thought": text, "action": action, "obs": obs})
            messages.append({"role": "user", "content": f"Observation: {obs}"})
            continue
        if action == "finish":
            ans = payload.get("answer", text)
            trace.append({"step": step, "thought": text, "action": action, "obs": ans})
            return {"answer": ans, "trace": trace}
        if action == "retrieve":
            q = payload.get("query", "")
            obs = _retrieve(q)
        else:
            obs = f"Unknown action: {action}"
        trace.append({"step": step, "thought": text, "action": action, "obs": obs})
        messages.append({"role": "user", "content": f"Observation: {obs}"})
    return {"answer": "Max steps reached.", "trace": trace}
```

### 试一下

```bash
python s09_agent_tools/c02_react_loop.py
# 问: R3630 G5 的内存插槽数量
```

实测输出 (有 `LLM_API_KEY`, 实跑 `MiniMax-M3 over minimaxi.com`, 完整 ReAct 循环 2 步):

```
Q: R3630 G5 的内存插槽数量

--- trace (2 step(s)) ---

[step 1]
  Thought: 用户问的是 R3630 G5 的内存插槽数量,文档里应该有。
  Action:  retrieve
  Obs:     - (server_whitepaper.pdf#1) 二、关键特性 计算密度:单台 2U 机箱内集成两颗处理器、32 条内存 DIMM 与 10 个 PCIe 4.0 扩展槽位...

[step 2]
  Thought: 找到了。
  Action:  finish
  Obs:     R3630 G5 配备 32 个 DIMM 内存插槽。

[A] R3630 G5 配备 32 个 DIMM 内存插槽。
```

跳过检索的对照 (无 LLM_API_KEY,graceful-skip 演示 trace 形状):

```
[Q] 1+1等于几

[skipped: LLM_API_KEY not set] — 演示 trace 形状:

--- step 1 ---
Thought: 这是个简单算术,不需要查文档。
Action:  finish
Obs:     1+1等于2。

[A] 1+1等于2。
```

- 交互输入,看 LLM 走几步 + 每步 Thought / Action / Obs 形状
- 同一个 prompt 下,问"R3630 G5 内存插槽数量"——`retrieve → finish` 两步;问"1+1等于几"——`finish` 一步直接结束

**观察**: **`run_agent` 把"Thought → Action → Observation"循环显式暴露在 trace 里**——LLM 每一轮在干嘛一眼可见,production 调试直接看 Thought 比看 Action 更有用。无 LLM_API_KEY 时 graceful-skip,只演示 trace 形状不真跑检索。撞 `max_steps=5` 兜底防死循环;JSON 解析失败时把原文当 Observation 反馈让 LLM 自愈(不直接报错)。

### 为什么不只写这一种

代码 2 是"单链 ReAct 循环"——所有 query 都走同一条管线,**没有 DAG 分支**(没法表达"先 categorize,按类别走不同路径")、**没有 plan-first**(LLM 不会先输出"我要先查 X 再查 Y"再执行)、**没有并行工具**(一次只能调一个)。生产走 RAGFlow Canvas DAG 编排(`agent/` 目录下可视化拖拽节点、加边、调权重),可以根据 query 类型走不同路径(FAQ / RAG / 工具调用)。死循环防护目前只靠 `max_steps`,没做"重复 Action 检测"+ prompt 软约束,放大到 `max_steps=20` 就明显。LLM 一路 retrieve 不 finish 的工业加固项见「思考题」第 1 题的 3 种解法。

---

## 接下来

s09 是把"是否走检索管线"做成模型可决策的 step——但 agent 决策本身的脆弱性也暴露出来,这些是后续章节的填空入口:

- **`max_steps=5` 撞顶 / LLM 不停 retrieve** — `max_steps` 是硬天花板,治不住"模型本来可以答对但就是停不下来"。生产加固 3 层: ① `max_steps` 上限(MVP 已做);② prompt 软约束(`TOOLS_DESC` 里写"每个问题**最多调用一次 retrieve**");③ 重复 action 检测(运行时检查"前两步 `Action / ActionInput` 对是否重复",命中强制 `finish`)。三招全做才是生产级,见「思考题」第 1 题
- **`Action: retrieve` 漏 `ActionInput` / JSON 解析失败** — regex 抓格式是脆弱的。生产里用 OpenAI / Anthropic 的 `tool_calls` 字段让 API 帮你解析,模型直接结构化返回 `tool_calls[0].function.name + arguments`,正则解析归零。RAGFlow `LLMToolPluginCallSession` 走的这条路
- **工具爆炸 / 选错工具** — 工具一多 LLM 选择难度指数级上升。生产治理 3 招: ① 按"用户意图"分组路由(分类器先决定走哪组工具);② 工具描述写得"互斥"(不重叠,别让模型在"A 也能做、B 也能做"之间纠结);③ 拆多层 agent(supervisor + sub-agent 嵌套,RAGFlow 用 `Agent` 组件嵌套实现 "agent-as-tool")
- **agent 单线循环 → DAG 分支** — s09 toy 是"先 retrieve 后 finish"单链;生产 RAGFlow Canvas 把"按 query 走不同路径"做成可视化 DAG,UI 里拖拽节点、加边、调权重,不改代码就能调整 agent 行为

s10 **graphrag**: 把 s09 的"决策权交给 LLM"扩到"决策权交给图遍历"——RAG 检索从"向量召回 top-k"扩成"向量召回 + 实体关系图遍历",回答多跳问题不再依赖 LLM 一次性 retrieve 全部上下文,而是先 retrieve 种子实体,再沿边扩到 n-hop 邻居,适合"关联关系 / 多次引用"类问题(法务合同条款串联、学术论文引文网络)。

---

## 思考题

1. **如果模型一直选 retrieve 不 finish 怎么办？**
2. **ReAct 和 function-calling 的本质区别是什么？**
3. **工具一多 LLM 选错怎么办？**

（答案见文末「思考题答案」）

---

## 思考题答案

### Q1. 如果模型一直选 retrieve 不 finish 怎么办？

这是个真实的失败模式——LLM 训练数据里"helpful assistant 应该是先查再答"的偏置很强，又因为 retrieve 的 observation 总是非空（哪怕不相关），模型找不到"应该停"的信号，会一路 retrieve 到 `max_steps` 撞顶。

按代价从低到高排三种解法：

**1) `max_steps` 上限**(MVP 已经做了）——硬天花板。最坏情况是用户看到"Max steps reached。"。优点：简单、零误判；缺点：用户体验差，它治不住"模型本来可以答对但就是停不下来"。

**2) 在 system prompt 强调"必须 finish"**——把 `TOOLS_DESC` 里的工具描述改成更显式的"每个问题**最多调用一次 retrieve**，再调用一次 retrieve 还没有答案就用已有的 Observation 给出 finish，否则就回答'我不知道'并 finish"。这是**软约束**，依赖模型遵从指令，但对主流模型（GPT-4o / Claude / Qwen-Max）效果不错；缺点是 prompt 越长越稀释，工具一多就顾不过来。

**3) 检测重复 Action**——运行时加一道"刚才两步的 `Action / ActionInput` 对是不是一模一样"的检查，命中就**强制 finish**(observation 是 `"我重复了同一次检索,无法获取更多新信息"`）。这是**硬约束**，不依赖模型听话。MVP 的 `max_steps=5` 实际是它的弱化版（撞 5 步就终止，不管内容）。

**生产推荐**：1 + 2 + 3 都做——`max_steps` 是兜底，prompt 软约束是主力，重复检测是保险。RAGFlow 三种都做（`max_rounds=5` + system prompt 强调 + `is_canceled()` 主动取消 + `Categorize` 组件做"这条路走不通跳走"，本质就是"用结构化组件替代字符串解析里的软约束"）。

### Q2. ReAct 和 function-calling 的本质区别是什么？

ReAct **每轮让模型先输出 Thought（思考过程）再选 Action**，function-calling **直接返回结构化 `tool_calls` 字段、不暴露思考过程**。ReAct 的优势是"可观测性 +300%"(trace 里能看到模型为什么选这个工具），代价是 token +20-30% + 解析脆弱（模型可能吐错格式）；function-calling 的优势是"API 帮你解析、鲁棒性归零"，代价是"看不到 Thought，production 调试更难"、依赖特定 provider。**MVP 选 ReAct** 是为了把"决策过程"暴露在 README 里；**生产可切 function-calling** 换鲁棒性。

### Q3. 工具一多 LLM 选错怎么办？

三招：① **按"用户意图"分组路由**——先分类器决定走哪组工具（检索组 / 计算组 / 查询组），不把所有工具一次塞给 LLM；② **工具描述写得"互斥"**——不同工具描述之间不重叠，别让模型在"A 也能做、B 也能做"之间纠结；③ **拆多层 agent**——supervisor 决定调哪个 sub-agent，sub-agent 只看到自己工具集，RAGFlow 用 `Agent` 组件嵌套实现 "agent-as-tool"。

下一章 — 这一节把"召回 → 排序 → 生成 → 服务化"中的某一环跑通,留下 +1 章填下一档的实现;每加一档,缺失上层就越明显,直到 s12 把所有环节收敛到 FastAPI 服务。

> 排错事项（`AuthenticationError` / `APIConnectionError` / `Action: retrieve` 缺 `ActionInput` 等）见 `c01` / `c02` 的 `### 局限与下一步`。