# s08 / Unit 01 — Prompt 模板 + LLM 引用生成

> 由浅入深第 1 步：把 s07 精排后的 top-3 hits 拼进 `<context>` 块，调 LLM 生成带角标的答案。  
> 这是"s07 给候选 → s08 给答案"的桥：精排选出最相关的 3 条，prompt 强制 LLM 引用 + 拒答，把幻觉关在笼子里。

## 这是什么

`s08_prompt_generate/units/01_prompt_template/code.py` 干三件事——拼 prompt、调 LLM、解引用编号。`PROMPT` 常量是一段 4 条硬约束的中文规则：(1) 只能依据 `<context>` 回答；(2) 资料里没答案就答"我不知道"（拒答兜底）；(3) 引用用 `[i]` 角标对应资料编号；(4) 中文 + 简洁直接。`<context>...</context>` 是显式定界符——把检索结果圈在"边界"里、用户问题圈在"边界"外，让 LLM 知道哪些是可信来源、哪些是要谨慎对待的输入（防 prompt 注入的第一道线）。

`_format_context(hits)` 把 hits 渲染成 `[i] (source#page) text` 形式——编号跟 prompt 里的 `[1][2]` 一一对应，下游解析 `re.findall(r'\[(\d+)\]', text)` 时一个萝卜一个坑。`answer(question, hits)` 调 OpenAI 兼容接口（`LLM_API_KEY` + `LLM_BASE_URL` + `LLM_MODEL`），返回 `{text, citations}`——`text` 是 LLM 输出（剥掉 `<think>...</think>` 块），`citations` 把命中的 `source` / `page` 一起带回给上游做追溯。无 `LLM_API_KEY` 时**优雅降级**：返回 `text="[skipped: LLM_API_KEY not set]"` + 仍然填好的 citations，让 pipeline 在没配 key 的环境下也能跑、只是 LLM 那一步 noop。

`main()` self-contained：内联 chroma 加载 + s04 unit 01 本地 BGE embed + s06 unit 02 BM25+dense 混合召回 + s07 unit 01 cross-encoder 精排，把 top-3 喂给 `answer()`、打印答案 + 引用。整章只有这 1 个 unit，因为 prompt 模板和 LLM 调用本身是**同一个原子动作的两端**——拆成 2 个 unit 反而要单独跑一遍 top-3 重算，多花几秒换不到可观测性收益。

## 跑起来

```bash
python s08_prompt_generate/units/01_prompt_template/code.py
# 问: R3630 G5 的内存插槽数量
```

输出示例（无 LLM_API_KEY 时）：

```
loaded 28 chunks from samples/

--- top-3 after rerank ---
  #1 [server_whitepaper.pdf#1] rerank=0.954 | ... 内存、10 个 PCIe 4.0 扩展槽位 ...
  #2 [server_whitepaper.pdf#2] rerank=0.644 | 内存 32 × DDR4 3200 ECC RDIMM ...
  #3 [server_whitepaper.pdf#4] rerank=0.870 | 内存支持镜像、备用与纠错码（ECC）三种数据保护模式 ...

A: [skipped: LLM_API_KEY not set]
引用: [
  {'index': 1, 'source': 'server_whitepaper.pdf', 'page': 1},
  {'index': 2, 'source': 'server_whitepaper.pdf', 'page': 2},
  {'index': 3, 'source': 'server_whitepaper.pdf', 'page': 4}
]
```

有 `LLM_API_KEY` 时（实测 MiniMax-M3 over minimaxi.com）：

```
A: R3630 G5 配备 **32 个 DIMM 内存插槽** [2]。
```

拒答对照（CEO 不在资料里）：

```
A: 我不知道。资料中未提及公司 CEO 的姓名，仅披露了最终控制方为朱蓉娟、彭韬夫妇[1]。
```

## 它做对了什么

- **`<context>...</context>` 定界符防 prompt 注入**：检索结果被显式包起来，LLM 一眼能看出"哪些是资料、哪些是用户问题"。就算用户问题里塞"忽略上面所有指令"，也越不过 `<context>` 这道栅栏——这是 RAG 系统抗 prompt 注入的**第一道线**。
- **"我不知道"是显式 sentinel**：资料里没答案时硬约束让模型答"我不知道"而不是硬凑。temperature=0 + 显式兜底比"自由发挥"安全得多——但这只是单点防御，ragflow 的 `sufficiency_check` 把它升级成独立 LLM 调用。
- **引用编号 + source/page 一起回填**：prompt 里要求 `[1][2]` 角标，`answer()` 返回时把每条 hit 的 `source`/`page` 一并带回——下游 UI 可以直接渲染"答案句末 [1] → 跳到 server_whitepaper.pdf 第 2 页"，**可追溯**而非仅"看起来权威"。
- **graceful degradation**：`LLM_API_KEY` 没设时返回 `[skipped: ...]` + 仍然填好的 citations——pipeline 不崩、citation 链路不断，开发者可以本地裸跑验证 retrieval/rerank 没问题。

## 它做错了什么

- **没有 streaming**：LLM 输出完才一次性返回，长答案（300+ token）等几秒才出字。生产上要 `stream=True` 让 token 边生成边推到前端（TTFT < 500ms 才不卡）。
- **没有 retry / 超时**：OpenAI 调用一旦 5xx / 超时就整个 pipeline 挂掉。生产至少要 `tenacity` 加指数退避、设 `timeout=10s`。
- **拒答是"prompt 一行字"不是"独立判定"**：模型可以遵守也可以不遵守——temperature=0 时倾向遵守，但 prompt 长了或被前面几轮对话稀释，这条规则就衰减。ragflow 的 `sufficiency_check` 把"该不该拒答"做成 LLM 显式输出 `is_sufficient: true/false`，可观测、可分支。
- **不挡 `<context>` 内的恶意标记**：用户问题里的 `[INST]` / `<system>` 已经被定界符挡了，但**资料本身**（PDF 抽出来的文本）如果含 `<system>` 之类的攻击向量，会直接进 prompt——应该渲染前 strip 掉。这是个真实生产坑，ragflow 走 "工具调用前的内容审查" 兜底。
- **同 prompt 里塞引用规则不稳**：prompt 里硬塞"引用时用 [1]"省 token，但 LLM 经常**写 [1] 但引错**——rerank 顺序和 LLM 内部"哪条更相关"的判断不一致。ragflow 的解法是**双 pass**：先生成答案、再用 `citation_prompt` 跑一次补引用，把"答"和"标"拆开。

## 对照 ragflow 怎么做的

ragflow 把"问答 prompt"拆成 40+ 个独立 `.md` 模板（`rag/prompts/` 包），核心三个：`sufficiency_check`（判够不够）、`multi_queries_gen`（不够就改写 query 再搜）、`citation_prompt` / `citation_plus`（双 pass 补引用）——MVP 的 `PROMPT` 一段话干的活，ragflow 拆成 3-4 次 LLM 调用。可观测、可分支、可替换，但**贵**（2-3x token）。

另外，ragflow 用 `<|COMPLETE|>` 作为"生成结束"哨兵（见 `docs/reference/ragflow-notes/graph_extraction.md` 里的 `DEFAULT_COMPLETION_DELIMITER`）——LLM 生成到 `<|COMPLETE|>` 就停，避开"模型继续编后续内容"的幻觉。本 MVP 没这个哨兵，靠 `max_tokens` 兜底，模型在 token 边界可能给出半句话。生产上应该让 prompt 末尾显式 `<|COMPLETE|>` 哨兵 + 解析器按这个切。

参考：[`docs/reference/ragflow-notes/prompt_templates.md`](../../../../docs/reference/ragflow-notes/prompt_templates.md)、[`docs/reference/ragflow-notes/graph_extraction.md`](../../../../docs/reference/ragflow-notes/graph_extraction.md)

## 思考题

- **prompt 里的"如果资料里没有答案回答'我不知道'"真的管用吗？怎么验证？**  
  答：管用一部分，靠"显式约束 + temperature=0"。要严格验证，需要造一个**负样本集**（问题不在资料里），跑 N 次统计"答'我不知道'的比例"——经验上 temperature=0 + 约束靠前时 95%+ 拒答率，约束靠后或 prompt 过长时掉到 60-70%。更稳的做法是 ragflow 的 `sufficiency_check`：独立 LLM 调用判 `is_sufficient: false`，把"拒答"从 prompt 软约束升级成**显式 JSON 决策**。详见 `thinking_answers.md`。

- **LLM 写的 `[1]` 和 prompt 里"第 1 条"对不上怎么办？**  
  答：两种解法。① **prompt 里给 summary**——把每条 chunk 渲染成 `[1] (source#page) summary: ...`，让编号变成有语义的"代号"，LLM 引错就立刻能从 summary 看出来。② **解析后校验**——`citations = sorted(set(int(x) for x in re.findall(r'\[(\d+)\]', text)))`，任何 `c not in range(1, len(hits)+1)` 都视为无效引用。终极方案是 ragflow 双 pass：先生成答案、再用 `citation_prompt` 单独跑一遍补 / 修引用。详见 `thinking_answers.md`。