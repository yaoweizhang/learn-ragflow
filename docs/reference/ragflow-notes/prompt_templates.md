# RAGFlow 的 prompt 模板设计

RAGFlow 不在 `rag/prompts.py`（单文件），而是一个**包**：
`D:\study\rag_study\ragflow\rag\prompts\`，里面 40+ 个 `.md` 模板文件 +
`generator.py` 编排器（commit `828c5789f`）。它把"问答"拆成若干独立
的 prompt，每个 prompt 专门干一件事。

## 摘选 1：`citation_prompt.md`（给已有答案补角标的双 pass 设计）

`D:\study\rag_study\ragflow\rag\prompts\citation_prompt.md:1-13`

```text
Based on the provided document or chat history, add citations to the input text using
the format specified later.

# Citation Requirements:

## Technical Rules:
- Use format: [ID:i] or [ID:i] [ID:j] for multiple sources
- Place citations at the end of sentences, before punctuation
- Maximum 4 citations per sentence
- DO NOT cite content not from <context></context>
```

注意 RAGFlow **不是让 LLM 在生成答案时"边想边标 [1]"**——它把"答"和
"标"拆开：先生成答案文本、再用 `citation_prompt` 跑一次补引用。MVP
的 `PROMPT` 是一段话里硬塞"引用时用 [1]"，省 token 但牺牲稳定性。

## 摘选 2：`sufficiency_check.md`（拒答条件显式建模成独立 prompt）

`D:\study\rag_study\ragflow\rag\prompts\sufficiency_check.md:1-19`

```text
You are a information retrieval evaluation expert. Please assess whether the currently
retrieved content is sufficient to answer the user's question.

User question:
{{ question }}

Retrieved content:
{{ retrieved_docs }}

Please determine whether these content are sufficient to answer the user's question.

Output format (JSON):
{
    "is_sufficient": true/false,
    "reasoning": "...",
    "missing_information": [...]
}
```

RAGFlow 把"够不够"**显式建模**成一个独立 step。模型先回答 JSON 形
式的 `is_sufficient: true/false`，再决定走"直接生成答案"还是"调
`multi_queries_gen` 改写 query 再搜一遍"。这跟 MVP 的"在 prompt 里
写一句'如果资料里没有答案回答我不知道'"完全不同——MVP 是一条规则
当万能钥匙用，RAGFlow 是把规则变成一个**可观测、可分支**的决策点。

## 为什么 RAGFlow 用多 prompt 模板而不是一个

- **可观测**：每个 step 是独立的 LLM 调用，结果写进日志 / 评估器
  （`ragas` 友好）。MVP 把所有约束塞进一段 prompt，失败时只能看
  整段输出"哪里偏了"——分不清是检索漏了、还是 LLM 没遵守。
- **可分支**：`sufficiency_check` 返回 `false` 触发 `multi_queries_gen`
  改写 query、再走一次检索-生成。MVP 没这个回路，资料不够就直接
  拒答，浪费了"可能再查一次就够"的场景。
- **可替换**：citation prompt 单独维护，方便换格式 / 换语言（见
  `citation_prompt.md` 的 RTL Arabic 例外规则）。MVP 的引用规则和
  主 prompt 揉在一起，改规则要重写整段。

## 为什么它把"拒答"单独强调

`sufficiency_check` + `multi_queries_gen` 这对组合把"我不知道"从
"一句 prompt 兜底"升级成"流程级 fallback"：

1. **先判够不够**——避免模型在资料边缘命中时硬给个"勉强"的答案。
2. **不够就改写**——给 RAG 一次自救机会（同义词、补全、时间归一化）。
3. **还不行才拒答**——拒答是"穷尽手段"之后的下游动作，不是 LLM
   的个人发挥。

MVP 的 "我不知道" 是 prompt 里的一行字，模型可以遵守也可以不遵守
（temperature=0 时倾向遵守，但 prompt 长了就衰减）。RAGFlow 把"该不
该拒答"变成 LLM 显式输出的 JSON 字段，前端能渲染、评估能算召回率。

## 跟 MVP 版本的差异

| 维度 | MVP (s08) | RAGFlow |
|---|---|---|
| Prompt 数量 | 1 段（`PROMPT`） | 40+ `.md` 模板，每个任务一个 |
| 引用方式 | 边生成边标 `[1]` | 先生成、再用 `citation_prompt` 补 |
| 拒答 | prompt 一行兜底 | `sufficiency_check` JSON 判 + `multi_queries_gen` 改写 |
| 失败处理 | 看不到 | 每步 LLM 调用可观测、可分支 |
| 改写 query | 无 | 有（`multi_queries_gen.md`） |
| Token 成本 | 低（一次 LLM 调用） | 高（2-3 次），换来可控 |

MVP 的 `PROMPT` 在 RAGFlow 里对应的是 `citation_prompt` 那一段
（"DO NOT cite content not from `<context>`"），但**少了**前面
的 sufficiency 判定和后面的双 pass 补引用——这是教学仓库
的"够用就行"和工业 RAG 的"可治理"之间的边界。
