# s08 Prompt 与生成 — 让 LLM 带 `[i]` 角标引用答出来

[上一章 s07 → · 下一章 s09 → ... → s12]

> *"s07 把对的 chunk 顶到第一,但 LLM 默认不知道哪些来自资料、哪些是自己编的 — prompt 工程是把幻觉关进 `<context>` 笼子的最后一道栅栏"*
>
> **链路位置**: 在线检索链路的生成器 (s07 精排 → **s08 生成** → s09+)
> **代码文件**: c01_prompt_template.py

> 环境准备: 见 root README §快速开始 — `pip install -r requirements.txt` + `.env` 配 `LLM_API_KEY` (可选, 无 key 时 graceful-skip, 只跑 prompt 渲染不调 LLM)

---

## 问题

s07 用 cross-encoder 把"对的 chunk"重排到 top-3 — 但"召回 + 精排"解决的是**检索质量**,回答这一头仍有一道悬崖:**LLM 默认不知道哪些来自资料、哪些是自己编的**。即便把最相关的 3 段喂进 prompt,LLM 仍会**自信地编**"权威答案",而且不告诉你哪句是编的。这道悬崖由 3 类典型失败堆起来。

**第一, 幻觉 (hallucination)**。在没有外部资料约束时,LLM 会拿训练语料里"看起来合理"的内容硬拼 — "按惯例审计费用通常为 50 万元", 这个数字是它拼出来的, 不是从任何一份资料里查到的。**s01 的 prompt 极简版 ("只能依据 <context> 回答") 是第一道防线, 但太弱 — LLM 在长 prompt / 多轮对话下会"自由发挥"**, 工业上的失败模式是"答案看着很对, 但其实 80% 是 LLM 编的"。

**第二, 不可追溯 (citation misalignment)**。LLM 经常在原文里写 `[1]`, 但 prompt 里"第 1 条"实际不是它想引的那条 — rerank 顺序和 LLM 内部"哪条更相关"的判断不一致。**这就是"答案对、引用错" — 用户点 `[1]` 跳过去发现内容跟答案没关系, 信任崩塌**。法务 / 医疗场景下, 一次"答案对引用错"就足以判 RAG 系统死刑。

**第三, 资料外硬凑**。资料里没答案时, LLM 会**编一段看起来沾边但完全不是答案的输出**, 概率不高但代价极大 — 生产事故就出在这种"看起来合理但完全是编的"答案上。**单纯的 prompt 硬约束("若资料没有就答'我不知道'")在 temperature=0 + 约束靠前时 95%+ 拒答率, 但 prompt 长或被对话稀释时掉到 60-70%**。

把这三类问题合起来看, **LLM 输出"对不对"是 RAG 系统容错率最低的一环** — s05 落盘索引、s06 拉回候选、s07 在小池子上精排, **s08 是唯一一处"信 LLM"的环节, 所以 prompt 工程是 RAG 系统里投入产出比最高的组件**。很多团队花大力气调 embedding / rerank, 却在 prompt 这一步随手写一句"请基于以下资料回答", 然后看着 LLM 一本正经地胡说八道。s08 的目标不是解决这些失败, 而是把它们的边界显式暴露出来, 让你看到 toy prompt 的脆弱性 — 它只能在小池子上精排 (top-3), 且只能引用 prompt 里"已经被渲染"的编号, **如果 s07 精排阶段就漏了真正相关的 chunk, prompt 也救不回来**。

---

## 解决方案

s08 用 **一个脚本** 把"拼 prompt + 调 LLM + 解引用"三步跑通, 演示"prompt 工程把幻觉关进 `<context>` 笼子"。

```
       s07 hits (top-3)                      prompt                              LLM answer
       [{text, source, page}×3]        ┌──────────────────────────────┐    ┌──────────────────┐
              │                          │ [1 角色 / 硬约束]            │    │ "R3630 G5 配备  │
              │   _format_context(hits) │   只能依据 <context>         │    │  32 个 DIMM 插  │
              ▼                          │   没有就拒答                 │ ──▶│  槽[2]..."       │
       [i] (source#page) text 块        │   引用用 [i] 角标           │    │       │          │
       [1] (server_whitepaper.pdf#1)... │ [2 上下文块]                │    │ re.findall [i]  │
              │                          │   <context>                 │    │       ▼          │
              └───────────────────────▶  │     [1] ... [2] ... [3] ... │    │ citations list  │
                  PROMPT.format(...)     │   </context>                │    │ [{i, source, page}×N]
                                         │ [3 问题]                    │    └──────────────────┘
                                         │   问题: R3630 G5 内存插槽 │
                                         └──────────────────────────────┘
```

| 脚本 | 解决什么 | 留下什么局限 | 何时用 |
|---|---|---|---|
| `c01_prompt_template.py` | PROMPT 4 条硬约束 + `<context>` 定界符 + `[i]` 角标 + 拒答兜底; 无 LLM_API_KEY 时 graceful-skip | 拒答是 prompt 软约束; 不挡 `<context>` 内的恶意标记; 同 prompt 里塞引用规则不稳 (rerank 顺序 vs LLM 判断不一致) | 端到端 demo / 教学 prompt 工程底座 / s09+ 工业加固起点 |

整章只有 1 个代码文件, 因为 **prompt 模板和 LLM 调用本身是同一个原子动作的两端** — 拆成 2 段反而要单独跑一遍 top-3 重算 (内联 chroma + s04 BGE embed + s06 hybrid + s07 rerank), 多花 5-10 秒换不到可观测性收益; prompt 工程的关键变量是 PROMPT 字符串本身, 不是被它调用的下游函数。**每一步的局限, 都是后续章节 (s09 起) 要解决的入口**: s09 起接 chunking/loader 兜底,s11 修 LLM 调用脆性 (retry / streaming / 长上下文), s12 在 FastAPI 服务里把"召回 → 排序 → 生成"全链路固化成 production endpoint。

---

## 代码 1: Prompt 模板 + LLM 引用生成 ([c01_prompt_template.py](c01_prompt_template.py))

入口:[`c01_prompt_template.py`](c01_prompt_template.py)

把 s07 精排后的 top-3 hits 拼进 `<context>` 块, 调 LLM 生成带角标的答案。
这是"s07 给候选 → s08 给答案"的桥: 精排选出最相关的 3 条, prompt 强制 LLM 引用 + 拒答, 把幻觉关进 `<context>` 笼子里。

### 工作原理

**做一件事**: 用一个 4 条硬约束的 PROMPT 字符串把 s07 精排后的 top-3 hits 渲染进 `<context>` 块, 调 OpenAI 兼容 LLM 生成答案, 返回 `{text, citations}` — text 是 LLM 输出 (剥掉 `<think>...</think>` 推理块), citations 把命中的 `source` / `page` 一起带回给上游做追溯。

**N 步**:
1. 内联 chroma 加载 + s04 BGE embed (`BAAI/bge-small-zh-v1.5`, 512 维, `normalize_embeddings=True`) + s06 hybrid (BM25 + dense 等权 `α=0.5`) + s07 rerank (BGE-reranker-base), 拿到 top-3 hits
2. `PROMPT` 常量定义 4 条硬约束的中文 prompt 模板: 只能依据 `<context>` 回答 / 资料没有就答"我不知道" / 引用用 `[i]` 角标 / 中文 + 简洁直接
3. `_format_context(hits)` 把每条 hit 渲染成 `[i] (source#page) text` 形式, 编号跟 PROMPT 里的 `[1][2]` 一一对应
4. `answer(question, hits)` 先回填 `citations` (按 hits 顺序), 再检查 `LLM_API_KEY` — 无 key 时 graceful-skip 返回 `[skipped: LLM_API_KEY not set]` + 仍然填好的 citations
5. 有 key 时, `PROMPT.format(context=..., question=...)` 拼成最终 prompt, 调 OpenAI 兼容 `client.chat.completions.create()` (`temperature=0` 强化拒答稳定性)
6. `re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)` 剥掉推理模型的中间步骤块, 得到最终 `text`
7. 返回 `{text, citations}`, 打印 `A: <text>` + `引用: [...]`

```python
# 中间片段: _format_context + answer 的 graceful-skip 分支
def _format_context(hits: list[dict]) -> str:
    """把 hits 渲染成 `[i] (source#page) text` 块,跟 prompt 里 [1][2] 一一对应。"""
    blocks = []
    for i, h in enumerate(hits, start=1):
        loc = f"{h['source']}#{h.get('page', '?')}"
        blocks.append(f"[{i}] ({loc}) {h['text']}")
    return "\n\n".join(blocks)


def answer(question: str, hits: list[dict]) -> dict:
    citations = [
        {"index": i, "source": h["source"], "page": h.get("page")}
        for i, h in enumerate(hits, 1)
    ]
    if not os.environ.get("LLM_API_KEY"):
        return {
            "text": "[skipped: LLM_API_KEY not set]",
            "citations": citations,  # 即便没调 LLM,citations 仍回填
        }
    # ... prompt 拼接 + openai 调用 ...
```

**完整函数**:

```python
PROMPT = """你是严谨的问答助手,只能依据 <context> 里的资料回答问题。
- 如果资料中没有直接回答问题的内容,仅回答"我不知道",不要附加任何引用或相关但不直接回答问题的信息。
- 引用时用 [1]、[2] 这样的角标对应资料编号。
- 回答用中文,简洁直接。

<context>
{context}
</context>

问题: {question}
"""


def _format_context(hits: list[dict]) -> str:
    """把 hits 渲染成 `[i] (source#page) text` 块, 跟 prompt 里 [1][2] 一一对应."""
    blocks = []
    for i, h in enumerate(hits, start=1):
        loc = f"{h['source']}#{h.get('page', '?')}"
        blocks.append(f"[{i}] ({loc}) {h['text']}")
    return "\n\n".join(blocks)


def answer(question: str, hits: list[dict]) -> dict:
    """调用 OpenAI 兼容 LLM 生成答案, 返回 `{text, citations}`.

    无 `LLM_API_KEY` 时降级: 返回带 `[skipped: LLM_API_KEY not set]` 的 text,
    citations 仍然回填 (让调用方至少能拿到命中的 source/page)。
    """
    citations = [
        {"index": i, "source": h["source"], "page": h.get("page")}
        for i, h in enumerate(hits, 1)
    ]
    if not os.environ.get("LLM_API_KEY"):
        return {
            "text": "[skipped: LLM_API_KEY not set]",
            "citations": citations,
        }

    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    prompt = PROMPT.format(context=_format_context(hits), question=question)
    resp = client.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content
    # 去掉 <think>...</think> 块 (DeepSeek R1 / MiniMax 类推理模型的中间步骤)
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return {"text": text, "citations": citations}
```

### 试一下

```bash
python s08_prompt_generate/c01_prompt_template.py
# 问: R3630 G5 的内存插槽数量
```

`EOFError` 自动兜底为 `question='内存'`。首次跑会下载 `BAAI/bge-reranker-base` ~1GB (同 s07)。**无 `LLM_API_KEY` 时 graceful-skip** (`text="[skipped: LLM_API_KEY not set]"` + citations 仍回填):

```
loaded 28 chunks from samples/

--- top-3 after rerank ---
  #1 [server_whitepaper.pdf#1] rerank=0.954 | ... 内存、10 个 PCIe 4.0 扩展槽位 ...
  #2 [server_whitepaper.pdf#2] rerank=0.644 | 内存 32 × DDR4 3200 ECC RDIMM ...
  #3 [server_whitepaper.pdf#4] rerank=0.870 | 内存支持镜像、备用与纠错码(ECC)三种数据保护模式 ...

A: [skipped: LLM_API_KEY not set]
引用: [
  {'index': 1, 'source': 'server_whitepaper.pdf', 'page': 1},
  {'index': 2, 'source': 'server_whitepaper.pdf', 'page': 2},
  {'index': 3, 'source': 'server_whitepaper.pdf', 'page': 4}
]
```

有 `LLM_API_KEY` 时 (旧 README 实测 `query='内存'`, LLM 真实生成):

```
loaded 34 chunks from samples/

--- top-3 after rerank ---
  #1 [server_whitepaper.pdf#3] rerank=0.664 | 四、应用场景 云数据中心:作为通用计算节点支撑私有云与混合云平台,配合虚拟化与容器平台提供高 密度的
  #2 [server_whitepaper.pdf#1] rerank=0.550 | 二、关键特性 计算密度:单台 2U 机箱内集成两颗处理器、32 条内存 DIMM 与 10 个 PC
  #3 [server_whitepaper.pdf#4] rerank=0.527 | 五、可靠性与可维护性 冗余设计:电源、风扇、Boot 盘、PCIe 控制器均支持 N+1 冗余;内存

A: 根据 <context> 资料,关于内存的信息如下:
- **内存配置** [2]:单台 2U 机箱内集成 32 条内存 DIMM;在 880mm 标准机柜深度下支持纵向堆叠 24 台以上,单机柜可提供 60TB+ 内存 [1]。
- **数据保护模式** [3]:支持镜像、备用与纠错码(ECC)三种数据保护模式,通过 Intel Run Sure 技术可在单条内存故障时自动降级运行。
- **温度监控** [3]:BMC 内置传感器实时上报内存温度等关键指标,采样频率为 1Hz。
引用: [
  {'index': 1, 'source': 'server_whitepaper.pdf', 'page': 3},
  {'index': 2, 'source': 'server_whitepaper.pdf', 'page': 1},
  {'index': 3, 'source': 'server_whitepaper.pdf', 'page': 4}
]
```

拒答对照 (CEO 不在资料里, LLM 拒答兜底):

```
A: 我不知道。资料中未提及公司 CEO 的姓名,仅披露了最终控制方为朱蓉娟、彭韬夫妇[1]。
```

**观察**: PROMPT 里 4 条硬约束 + `<context>` 定界符 + `[i]` 角标 协同起作用 — `A:` 行的引用号 `[1][2][3]` 都落在 `hits` 列表的真实下标上, `citations` 里 `source/page` 与 prompt 里 `(server_whitepaper.pdf#X)` 完全对齐; CEO 这种"资料外"query 触发拒答兜底 (`"我不知道"`); 即便 LLM 用了"summarize"风格的列表格式, **每条 bullet 后都贴了角标**, 没出现"答案有内容但没引用"的失控。这是 prompt 工程最基本的可观测信号 — 拒答率、引号对齐率、风格稳定性。但你也能看到局限: `[3]` 被引用了两次 (数据保护 + 温度监控), LLM 自己"判断"哪段更相关 — rerank 顺序和 LLM 内部判断不完全一致, 这是 s11 的 citation 校验要拦下的事。

### 为什么不只写这一种

`c01` 用 **单段 PROMPT + `<context>` 定界符 + `[i]` 角标 + graceful-skip** 把"prompt 工程把幻觉关进笼子"跑通, 但**它只在 top-3 上做约束**, 留 5 类典型局限: **① 没有 streaming** — 长答案 (300+ token) 等几秒才出字, 生产上要 `stream=True` 让 TTFT < 500ms; **② 没有 retry / 超时** — OpenAI 5xx / 超时整个 pipeline 挂掉, 生产要 `tenacity` + `timeout=10s`; **③ 拒答是 prompt 软约束** — LLM 可以遵守也可以不遵守, prompt 长或被对话稀释时约束衰减, RAGFlow 的 `sufficiency_check` 把它升级成 JSON 显式决策; **④ 不挡 `<context>` 内的恶意标记** — 用户问题的 `[INST]` / `<system>` 已被定界符挡了, 但资料本身 (PDF 抽出的文本) 含 `<system>` 会直接进 prompt, RAGFlow 走"工具调用前的内容审查"兜底; **⑤ 同 prompt 里塞引用规则不稳** — prompt 里硬塞"引用时用 [i]"省 token, 但 LLM 经常**写 `[1]` 但引错** (rerank 顺序 vs LLM 内部判断不一致), RAGFlow 的解法是**双 pass (citation_plus)**: 先生成答案、再用 `citation_prompt` 让 LLM 二轮补引用。`c01` 只做了最基础的 `[i]` 解析 + 软约束拒答, ② ③ ④ ⑤ 留给 s11/s12 工业加固 — production tier 的"可观测性 vs 成本"分层选型 (单段 MVP → +sufficiency_check → +citation_plus → 完整流水线) 在 s11/s12 才完整覆盖。

---

## 接下来

s08 是在线检索链路的生成器: `c01` 把 s07 精排后的 top-3 hits 拼进 `<context>` 块、调 LLM 生成带 `[i]` 角标的答案。但每一步都留下脆弱点, 这些是后续章节的填空目标:

- **拒答是 prompt 软约束** — temperature=0 + 约束靠前时 95%+ 拒答率, prompt 长或被对话稀释时掉到 60-70%。生产上把"该不该拒答"做成 LLM 显式输出 `is_sufficient: true/false`, 可观测、可分支 (RAGFlow `sufficiency_check`)。
- **没有 retry / 超时 / streaming** — OpenAI 调用一旦 5xx / 超时整个 pipeline 挂掉; 长答案 (300+ token) 等几秒才出字。生产至少要 `tenacity` 指数退避 + `timeout=10s` + `stream=True` 让 TTFT < 500ms。`c01` 没接 — LLM 调用脆性留 s11 修。
- **不挡 `<context>` 内的恶意标记** — 资料本身 (PDF 抽出的文本) 含 `<system>` / `[INST]` 会直接进 prompt。生产渲染前 strip 掉, RAGFlow 走"工具调用前的内容审查"兜底。
- **同 prompt 里塞引用规则不稳** — LLM 经常写 `[1]` 但引错 (rerank 顺序和 LLM 内部判断不一致)。生产上做双 pass: 先生成答案、再用 `citation_prompt` 单独跑一遍补 / 修引用 (RAGFlow `citation_plus`), 引用准确率 +20-30% 但 token +100%。
- **检索质量是天花板** — s08 的所有加固都建立在"s07 精排后的 top-3 是真相关 chunk"的前提上。如果 s07 漏召了真正相关的 chunk, prompt 工程再精细也救不回来。**召回 (recall) 必须先高, 再谈生成 (faithfulness)**。

s09 起 — 把"召回 → 排序 → 生成"全链路每一段逐步工业加固; s11 修 LLM 调用脆性 (retry / streaming / 长上下文), s12 在 FastAPI 服务里把整条链路固化成 production endpoint。

---

## 思考题

1. **怎么让模型不引用第 5 条而第 5 条恰好是答案？**
2. **prompt 里的"如果资料里没有答案回答'我不知道'"真的管用吗？怎么验证？**
3. **LLM 写的 `[1]` 和 prompt 里"第 1 条"对不上怎么办？**

（答案见文末「思考题答案」）

---

## 思考题答案

### Q1. 怎么让模型不引用第 5 条而第 5 条恰好是答案？

**先想清楚为什么"恰好是答案"会被排到第 5。** 两种可能：

- **召回阶段漏了**：向量+BM25 都没把第 5 条的相关 chunk 排到 top-K。治本要回到 s04-s06：① 改 chunk 切分粒度（太粗的 chunk 包含答案但被无关内容稀释相似度）；② 加 query expansion（让 BM25 命中同义词）；③ 调 embedding 模型。
- **rerank 阶段压了**：cross-encoder 也认为第 5 条不算最相关。这种情况下"第 5 条"其实是 LLM 自己的判断，rerank 分低意味着检索系统也不觉得它"刚好"是答案。

**MVP 层（不换检索）的两个解法：**

1. **调 `top_k`**——这是最直接的。在 `s07_rerank.code.rerank(...)` 里把 `top_k` 调大（3 → 4 或 5），让第 5 条**进 prompt** 之后，模型反而**会**引用它。题目是"不让模型引用第 5 条"，所以反过来——把 `top_k` 调小到 3，第 5 条连 prompt 都进不去，引用就无从谈起。配套把 `s08_prompt_generate.code` 里 `rerank(..., top_k=3)` 一起改。
2. **在 PROMPT 里硬约束"只用最相关的 N 条"**——把 `PROMPT` 的 `"{context}"` 改写为"以下只包含最相关的 3 条资料，请仅基于这 3 条回答；如有歧义，引用最直接对应的那一条"。这条指令对模型"别往第 5 条想"有引导，但**不可靠**——LLM 对编号的遵守是软约束，遇到大模型或者长 context 时仍可能误引。

**根本办法是更精准的检索 + rerank**，把"答案是第 5 条"变成"答案是第 1 条"——这要求：

- 检索阶段用更大的 `k=20` 召回 + 更强的 embedding（换成 BGE-M3 或 OpenAI `text-embedding-3-large`）。
- rerank 阶段用中文优化版（BGE-reranker-v2-m3）替代 base 模型。
- query 改写：拿用户的原始 query 先让 LLM 改写/补全成更"信息检索友好"的表述，再去搜。

总结：**MVP 是"砍 top_k 把第 5 条挡在 prompt 外"；生产是"提升召回和 rerank 让第 5 条变第 1 条"。**

### Q2. prompt 里的"如果资料里没有答案回答'我不知道'"真的管用吗？怎么验证？

管用一部分，靠"显式约束 + temperature=0"。要严格验证，需要造一个**负样本集**（问题不在资料里），跑 N 次统计"答'我不知道'的比例"——经验上 temperature=0 + 约束靠前时 95%+ 拒答率，约束靠后或 prompt 过长时掉到 60-70%。更稳的做法是 ragflow 的 `sufficiency_check`：独立 LLM 调用判 `is_sufficient: false`，把"拒答"从 prompt 软约束升级成**显式 JSON 决策**。

### Q3. LLM 写的 `[1]` 和 prompt 里"第 1 条"对不上怎么办？

两种解法。① **prompt 里给 summary**——把每条 chunk 渲染成 `[1] (source#page) summary: ...`，让编号变成有语义的"代号"，LLM 引错就立刻能从 summary 看出来。② **解析后校验**——`citations = sorted(set(int(x) for x in re.findall(r'\[(\d+)\]', text)))`，任何 `c not in range(1, len(hits)+1)` 都视为无效引用。终极方案是 ragflow 双 pass：先生成答案、再用 `citation_prompt` 单独跑一遍补 / 修引用。
下一章 — 这一节把"召回 → 排序 → 生成 → 服务化"中的某一环跑通,留下 +1 章填下一档的实现;每加一档,缺失上层就越明显,直到 s12 把所有环节收敛到 FastAPI 服务。

> 排错事项（`AuthenticationError` / `APIConnectionError` / `KeyError: 'context'` 等）见 `c01` 的 `### 局限与下一步`。