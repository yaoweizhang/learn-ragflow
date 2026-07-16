# RAGFlow Agent 与工具

## 一句话
RAGFlow 把 agent 做成可插拔 DAG：30+ 组件用 JSON DSL 拼 workflow，走 OpenAI 兼容 `tool_calls`（结构化 API，不是字符串解析）；MVP s09 的 `run_agent` 把"解析 + 路由 + 终止"硬编码，加第三个工具就要改 `_retrieve` 和正则。

## 来源
- 仓库：https://github.com/infiniflow/ragflow
- 模块：`agent/canvas.py`（DAG 执行引擎）、`agent/component/`（节点库）、`agent/tools/`（具体工具）、`agent/templates/`（JSON DSL 预制工作流）
- 关联：本仓库 s09 `tool_call.py` / `react_loop.py`

> 注：s09 brief 写的是 `agent/`，实际 RAGFlow 把 agent 实现拆成上面 4 块，下面挑 3 个最相关的文件展开。

## Canvas 的 DAG 执行引擎

`agent/canvas.py` 的 `Canvas.__init__` 和 `load()`：

```python
class Canvas(Graph):
    def __init__(self, dsl: str, tenant_id=None, task_id=None, ...):
        self.path = []
        self.components = {}
        self.dsl = normalize_chunker_dsl(json.loads(dsl))  # DSL 存的就是工作流定义
        ...

    def load(self):
        self.components = self.dsl["components"]
        for k, cpn in self.components.items():
            cpn_nms.add(cpn["obj"]["component_name"])
            param = component_class(cpn["obj"]["component_name"] + "Param")()
            param.update(cpn["obj"]["params"])
            try:
                param.check()
            except Exception as e:
                raise ValueError(self.get_component_name(k) + f": {e}")
            cpn["obj"] = component_class(cpn["obj"]["component_name"])(self, k, param)
        self.path = self.dsl["path"]
```

每个 `cpn` 是一个**组件实例**（`Begin` / `LLM` / `Agent` / `Categorize` / `Retrieval` / ...），`self.path` 是这次执行的节点顺序列表。`Canvas.run` 是个**异步生成器**，按 `path` 顺序依次 yield `workflow_started` / `node_started` / `message` / `node_finished` / `workflow_finished` 事件——前端用这个流做实时 UI 更新。

## Agent 组件怎么调工具

`agent/component/agent_with_tools.py` 的 `Agent.__init__`：

```python
class Agent(LLM, ToolBase):
    component_name = "Agent"

    def __init__(self, canvas, id, param: LLMParam):
        LLM.__init__(self, canvas, id, param)
        self.tools = {}
        for idx, cpn in enumerate(self._param.tools):
            cpn = self._load_tool_obj(cpn)  # 加载 Retrieval / Tavily / ...
            original_name = cpn.get_meta()["function"]["name"]
            indexed_name = f"{original_name}_{idx}"
            self.tools[indexed_name] = cpn
        ...
        self.chat_mdl = LLMBundle(...)
        self.tool_meta = []
        for indexed_name, tool_obj in self.tools.items():
            original_meta = tool_obj.get_meta()
            indexed_meta = deepcopy(original_meta)
            indexed_meta["function"]["name"] = indexed_name  # 多个同名工具用 _N 区分
            self.tool_meta.append(indexed_meta)
        ...
        self.toolcall_session = LLMToolPluginCallSession(self.tools, self.callback)
        if self.tool_meta:
            self.chat_mdl.bind_tools(self.toolcall_session, self.tool_meta)
```

关键点：**`self.chat_mdl.bind_tools(...)` 把工具列表喂给 LLM 客户端**，LLM 返回的 `tool_calls` 字段被 `LLMToolPluginCallSession.tool_call` 解析成 `name + arguments`、在 `self.tools` 字典里查实例、调 `tool_obj.invoke(**arguments)`，**整套过程是结构化的**——不是从 LLM 的自由文本里 regex 抠 `Action: ...` 行。

## 工具基类

`agent/tools/base.py` 的 `LLMToolPluginCallSession.tool_call`：

```python
class LLMToolPluginCallSession(ToolCallSession):
    def __init__(self, tools_map: dict[str, object], callback: partial):
        self.tools_map = tools_map
        self.callback = callback

    async def tool_call_async(self, name, arguments, request_timeout=10):
        assert name in self.tools_map, f"LLM tool {name} does not exist"
        ...
        tool_obj = self.tools_map[name]
        if hasattr(tool_obj, "invoke_async") and asyncio.iscoroutinefunction(tool_obj.invoke_async):
            resp = await tool_obj.invoke_async(**arguments)
        else:
            resp = await thread_pool_exec(tool_obj.invoke, **arguments)
        ...
        self.callback(name, arguments, resp, elapsed_time=elapsed)  # 写日志给前端
        return resp
```

`tools_map` 是个**字符串 → Tool 实例**的字典，`name` 是 LLM 在 `tool_calls[].function.name` 里返回的字符串。每个 Tool 实现 `invoke(**kwargs)` / `invoke_async(**kwargs)`，签名跟 LLM 看到的 `function.parameters` 完全对应——这是 RAGFlow 的"工具即组件"设计：同一个 `Begin / LLM / Categorize / Retrieval` 既能在 canvas 里当 DAG 节点跑，也能被 `Agent` 当工具调。

## 为什么这样写（3 个 bullet）

- **为什么把 agent 做成可插拔的 DAG？**
  工具 / 编排 / 控制流拆成 30 多个独立组件（`agent/component/` 下面 `Begin` / `LLM` / `Agent` / `Categorize` / `Switch` / `Iteration` / `Loop` / `Message` / `Invoke` / `UserFillUp` / ...），用户用 `agent/templates/*.json` 拼成自己的 workflow 存到 DB，`Canvas.run` 只是个通用执行器——**加一个新工具不用改框架**。MVP 的 `run_agent` 把"解析 + 路由 + 终止"三个职责硬编码在一起，加第三个工具（除了 `retrieve` / `finish`）就要动 `_retrieve` 和正则。RAGFlow 加新工具 = 在 `agent/tools/` 加一个 `xxx.py`、在 `__init__.py` 注册——`Agent` 自动发现。

- **怎么避免 LLM 进入死循环？**
  三道防线：① `LLMBundle(..., max_rounds=self._param.max_rounds)`（默认 5）——LLM 调用次数硬上限；② `is_canceled()` 检查（写 Redis 的 `cancel` 标记）——前端或后台可**主动**中断；③ 失败不重试而是**跳路径**——`Categorize` 组件的 `_extend_path` 让一条路走不通时跳到另一条。MVP 只能用 ①（`max_steps=5`），对 ② ③ 没设计。

- **跟 MVP 版本的差异（planner、task queue 等）？**
  MVP：`run_agent` 是**字符串解析的循环**——LLM 吐文本、regex 抠 `Action / ActionInput`、手动维护 `messages` 历史。RAGFlow 走**结构化 API**（OpenAI 兼容的 `tool_calls` 字段）——LLM 客户端直接返回 `{name, arguments}`，客户端调用 `bind_tools` 注册的 callback；并且**所有调用是流式**（`Canvas.run` 是 async generator），前端能边算边渲染。Planner 层是 `Categorize` + `Switch`（多路分发）+ `Iteration` / `Loop`（带状态的任务队列），而 MVP 的 planner 退化成"先 `retrieve` 后 `finish`"的两步固定模板。Task queue 的角色是 `self.path` 数组 + `self.variables` 全局变量：组件之间通过 `{component_id@output_var}` 这样的 DSL 变量引用解耦，`Canvas.get_variable_value` 在执行时按需解析——本质是把 MVP 写死的 Python 控制流全部**数据化**了，可以 UI 编辑、可以保存、可以版本控制。
