# s01: RAG 全链路最小骨架 — 从字面匹配到检索增强生成

[上一章(无) · 下一章 s02 → ... → s12]

> *"先字面匹配, 再向量召回, 最后接入 LLM — RAG 的三层递进, 30 行起跑通"*
>
> **链路位置**: 端到端 (代码 1/2/3 不依赖 s02-s06), 独立可跑
> **代码文件**: c01_substring_match.py · c02_bag_of_words.py · c03_rag_pipeline.py

> 环境准备: 见 root README §快速开始 — `pip install -r requirements.txt` + `.env` 配 LLM_API_KEY (仅代码 3 需要)

---

## 问题

大模型很会"说话", 但不会"答"。它不知道你公司昨天签的合同, 不知道上季度的财报数字, 不知道你团队昨天改的接口签名 — 不是它笨, 而是**它只见过训练截止日期之前的公开世界**。把这三种典型失败场景拆开看, 都是同一类问题的不同切面:

**第一, 训练截止 (knowledge cutoff)**。任何 LLM 都有一个训练数据截止时间, GPT-4 是 2023 年 10 月, Claude 是 2025 年初, 开源模型也类似。问它昨天的股价、最新法规、刚发布的论文摘要, 它要么沉默 ("截至我的训练数据..."), 要么编一段听起来合理但完全错误的内容。这个问题**没有银弹** — 模型的训练数据是固化的, 你没法在每次新数据出现时重训整个模型, 工业级的成本和时间都不允许。即便是号称"在线学习"的方案, 也只能做轻量微调, 没法把整份新文档塞进参数。

**第二, 私有数据 (private data)**。你的客户名单、内部 wiki、产品需求文档、合同附件、技术评审记录 — 这些数据从来没出现在任何公开训练集里。即便你用的是 GPT-5 还是 Claude-Opus, 它也读不到你没喂给它的东西。把整份公司文档塞进 prompt? 不现实, 一份 10 万字的 wiki 已经超出大多数模型的上下文窗口; 即便塞得下, 也稀释了模型对关键信息的注意力 (lost-in-the-middle 现象: 模型对中间段的注意力远低于首尾)。RAG 用"按需检索 + 拼 prompt" 的方式绕过这个限制: 不一次性塞整份文档, 只检索 top-k 相关段落拼进 prompt。

**第三, 幻觉 (hallucination)**。在没有外部资料约束的情况下, LLM 会**自信地编造细节** — "按惯例审计费用通常为 50 万元", 这个数字是它从训练语料里"看起来合理"地拼出来的, 不是从任何一份资料里查到的。幻觉是 LLM 最大的可信度杀手: 用户看到流畅、自信、带具体数字的回答, 不会怀疑它在编; 等到出问题时 (法务纠纷、财务误报、医疗误诊), 代价已经付出去。即便加了"如果你不知道请说不知道"的 prompt 约束, LLM 在压力下仍可能编造, 因为它的训练目标就是"对所有 query 给出流畅回答", 拒答对模型来说是低概率事件。

这三种失败有一个共同解法 — **让模型"开卷考试"**: 答题前先把相关参考资料塞进 prompt, 让模型依据资料回答, 而不是凭参数记忆。这套"先查资料, 再答问题"的范式就是 **RAG (Retrieval-Augmented Generation)**: **检索 (retrieve)** 找到相关段落, **增强 (augment)** 把段落拼进 prompt, **生成 (generate)** 让 LLM 依据这些段落作答。三步合一, 模型就从一个"闭卷答题者"变成一个"开卷答题者" — 编造的源头被资料覆盖, 私有数据被检索补齐, 训练截止被最新索引绕过。RAG 不是唯一解法 (还有 fine-tuning / prompt engineering / tool use), 但它是**性价比最高、落地最快、可解释最强**的一种: 不需要重训模型, 不需要标注数据, 每次回答都能附引用让用户验证。

s01 的任务就是把这三步用 **30-80 行 Python 跑通一遍**, 不依赖任何 embedding 模型或向量库 — 让"retrieve + augment + generate" 这三个动词绑死在一条线上, 后面 11 章再把每个动词替换成工业实现。本章是 12 章教程的入口, 也是后续所有章节的底座: 代码 2 的 `top_k(query, paragraphs, k)` 接口是 s04-s06 替换的目标, 代码 3 的 `build_prompt` 是 s08 替换的目标, 代码 3 的 `call_llm` 是 s09-s10 替换的目标。**接口形状留好了, 后续章节照着替换**。如果你只想跑通 RAG 的最小形态, 读完本章即可; 如果想上生产, 继续读 s02-s12。

---

## 解决方案

s01 用 **三个递进的脚本** 把 RAG 主干跑通。每一步解决前一步的局限, 但也留下新的脆弱性 — 这种"递进暴露脆弱性, 后续章节填空"的设计是 12 章教程的核心结构。

```
代码 1 (子串)              代码 2 (词袋)              代码 3 (RAG pipeline)
┌────────────────┐      ┌─────────────────┐      ┌──────────────────┐
│ 段落列表       │      │ 段落 → 词频向量 │      │ 同 代码 2 检索    │
│       │        │      │       │         │      │       │          │
│       ▼        │      │       ▼         │      │       ▼          │
│ 子串匹配       │ ───▶ │ cosine 排序     │ ───▶ │ top-k 拼 prompt  │
│       │        │      │ top-k           │      │       │          │
│       ▼        │      │       │         │      │       ▼          │
│ 返回第一段     │      │ 直接返回        │      │ LLM 生成答案     │
└────────────────┘      └─────────────────┘      └──────────────────┘
   没排序、没语义         sparse 语义、有分         真正"开卷"
```

| 脚本 | 解决什么 | 留下什么局限 | 何时用 |
|---|---|---|---|
| `c01_substring_match.py` | 字面命中 (query 出现在 chunk 中) | 找不到同义词 ("营收" 找不到 "营业收入"); 无打分 | 教学 / 强字面约束场景 (法条 / 代码标识符) |
| `c02_bag_of_words.py` | 词频向量 + 余弦排序 top-k | 维度爆炸; 丢语序 ("狗咬人" = "人咬狗"); 仍无真语义 | 玩具检索 / 教学 / 小语料 |
| `c03_rag_pipeline.py` | retrieve → build_prompt → call_llm 完整闭环 | 召回错则全错; prompt 极简; 无 rerank | 端到端 demo / 后续章节填空底座 |

三步的关系是一条**主干**: 代码 1 把"段落列表 → 第一个命中段落"做出来, 暴露"无打分"的局限 — 第一段不一定是答案段; 代码 2 把"段落列表 → top-k + 余弦分"做出来, 暴露"无语义"的局限 — 词袋给"披露"和"公开披露"高分, 但给"披露"和"公告"零分; 代码 3 把 代码 2 的 top-k 拼进 prompt 调 LLM, 暴露"召回错 + prompt 极简"的局限 — 召回错则全错, prompt 无引用跟踪则 LLM 编引用号。**每一章的局限, 都是下一章要解决的入口**。

---

## 代码 1: 子串字面匹配 (substring matching)

### 工作原理

**做一件事**: 判断 query 是否在 chunk 中作为字面子串出现, 命中即返回。

**5 步**:
1. 用 `python-docx` 把 `samples/disclosure.docx` 读成段落列表 `paragraphs: list[str]` — 跳过空段落, 只保留非空文本
2. 接收交互输入的 query (如 `披露`, `外星人`) — `input("问点啥: ").strip()` 去掉首尾空白
3. 把 query 用 `question.split()` 拆成词列表, 段落用 `.lower()` 转小写
4. 遍历段落, 对每段判断"是否有任意 query 词作为子串出现在段落中" (`any(w.lower() in p.lower() for w in question.split())`)
5. 第一个命中的段落直接返回; 全 miss 时返回 `I don't know.` — 拒答兜底, 不让 LLM 编造答案

```python
# 中间片段: 拆词 + 子串包含判断
for p in paragraphs:
    if any(w.lower() in p.lower() for w in question.split()):
        return p
return "I don't know."
```

**完整函数**:

```python
def load_paragraphs(path: Path) -> list[str]:
    """只取非空段落, 去掉 Word 文档里那些"占位用的空段"."""
    return [p.text for p in Document(path).paragraphs if p.text.strip()]


def fake_rag(question: str, paragraphs: list[str]) -> str:
    """子串字面匹配: 拿问题里的每个词去段落里找子串, 命中即返回第一段.
    全 miss 时返回拒答兜底 'I don't know.' — 不让 LLM 编造答案."""
    for p in paragraphs:
        if any(w.lower() in p.lower() for w in question.split()):
            return p
    return "I don't know."


def main() -> None:
    paragraphs = load_paragraphs(SAMPLE)
    q = input("问点啥: ").strip()
    print(fake_rag(q, paragraphs))
```

### 试一下

```bash
python s01_what_is_rag/c01_substring_match.py
```

实测输出 (交互输入 `披露`):

```
[query] 披露
[hit]  相关信息披露详见财务报表附注三(二十五)、五 (二)1 及十五(二)。
```

- 输入 query: `披露` — 命中 1 条 (`相关信息披露详见财务报表附注...`)
- 输入 query: `外星人` — miss (返回 `I don't know.`)

**观察**: 字面匹配对同义词 `公开` → `披露` 无效 — 试 query `公开` 时也会命中 (因为 chunk 里恰好有"公开"字面), 但 query `公告` 在 chunk 只写"披露"时仍然 miss。这暴露 代码 1 的根本局限: **字面包含 ≠ 语义相关**, 即便 chunk 在语义上等价, 字符不重合就不命中。这是 代码 2 / s04 要解决的语义召回问题。

### 为什么不只写这一种

字面匹配无法处理同义词 (`披露` 找不到 `公开`)、顺序变体、语义关联 — 见 代码 2 词袋模型。

---

## 代码 2: 词袋模型 (bag-of-words) + 余弦相似度

### 工作原理

**做一件事**: 把每条 chunk 表征为 2-gram 词频向量, 用余弦相似度排序 top-k 候选, 替代 代码 1 的"第一段即答案"。

**6 步**:
1. 用同款 `python-docx` 加载器读段落 (与 代码 1 共用 `load_paragraphs`)
2. 对每段做 2-gram 切词 (`text[i:i+2] for i in range(len(text)-1)`, 滑动窗口) — 中文每 2 字 1 个 token, 单字粒度丢信息
3. 全部段落 token 组成全局词表 `vocab: {token: index}`, 用 `dict.setdefault(tok, len(vocab))` 自动分配索引
4. 每段转成词频向量 `vec = [词频 in vocab]`, 长度 = `len(vocab)`, 用 `Counter(tokenize(text))` 计数
5. query 用同款 2-gram 切, 转同样形状的向量 (query 词表外的 token 维度为 0)
6. 算 query 向量与每段向量的余弦相似度 (`dot(a,b) / (norm(a) * norm(b))`), 排序返回 top-3

```python
# 中间片段: 2-gram tokenize + 词频向量
def tokenize(text: str) -> list[str]:
    text = re.sub(r"\s+", "", text)
    return [text[i : i + 2] for i in range(len(text) - 1)]

def vectorize(text: str, vocab: dict[str, int]) -> list[float]:
    counter = Counter(tokenize(text))
    return [float(counter.get(tok, 0)) for tok in vocab]
```

**完整函数**:

```python
def tokenize(text: str) -> list[str]:
    """2-gram 滑动窗口 tokenize. 不是 jieba, 但足够"向量检索"概念演示."""
    text = re.sub(r"\s+", "", text)
    return [text[i : i + 2] for i in range(len(text) - 1)]


def vectorize(text: str, vocab: dict[str, int]) -> list[float]:
    """把段落转成词频向量, 词表外 token 丢弃."""
    counter = Counter(tokenize(text))
    return [float(counter.get(tok, 0)) for tok in vocab]


def cosine(a: list[float], b: list[float]) -> float:
    """手写余弦相似度, 等价 numpy 的 dot / (norm(a) * norm(b))."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def top_k(query: str, paragraphs: list[str], k: int = 3) -> list[tuple[str, float]]:
    """2-gram 词袋 + 手写余弦, 返回按相似度排序的 top-k.
    等价于 s05 的 chroma.col.query() 在 dense 向量上的语义检索.
    概念等价于 s04 的 BGE embedding, 只是这里用词频向量代替, 省去模型下载."""
    # 全局词表: 所有段落 token 集合
    vocab: dict[str, int] = {}
    for p in paragraphs:
        for tok in set(tokenize(p)):
            vocab.setdefault(tok, len(vocab))

    para_vecs = [vectorize(p, vocab) for p in paragraphs]
    q_vec = vectorize(query, vocab)

    scored = [(p, cosine(q_vec, pv)) for p, pv in zip(paragraphs, para_vecs)]
    scored.sort(key=lambda x: -x[1])
    return scored[:k]


def main() -> None:
    paragraphs = load_paragraphs(SAMPLE)
    q = input("问点啥: ").strip()
    print(f"\nTop-3 与你的问题最相关的段落(按向量余弦排序):")
    for rank, (text, score) in enumerate(top_k(q, paragraphs, k=3), 1):
        snippet = text[:120].replace("\n", " ")
        print(f"\n[{rank}] score={score:.3f}")
        print(f"    {snippet}...")
```

### 试一下

```bash
python s01_what_is_rag/c02_bag_of_words.py
```

预期输出示例 (交互输入 `披露`):

```
[vocab] 共 N 个 2-gram token
[query] 披露
Top-3 与你的问题最相关的段落 (按向量余弦排序):
[1] score=0.342
    相关信息披露详见财务报表附注三(二十五)、五 (二)1 及十五(二)...
[2] score=0.215
    ...
[3] score=0.187
    ...
```

- 交互输入, 看 top-3 余弦分 (按相似度降序, 每段一个 `[rank] score=...`)

**观察**: 词袋维度爆炸 (2-gram 笛卡尔积, 每段 100+ unique token), 语序信息全丢 ("狗咬人" vs "人咬狗" 同分), 长 chunk 的 L2 范数主导相似度排序 (需要归一化才能消除长度偏置)。这是 代码 3 的召回质量上限, 也是 s04 BGE dense 向量要解决的。

### 为什么不只写这一种

词袋维度爆炸 (2-gram 笛卡尔积), 语序信息全丢, 仍无真语义 — 见 代码 3 RAG pipeline 把检索结果拼进 prompt。

---

## 代码 3: RAG pipeline (retrieve + augment + generate)

### 工作原理

**做一件事**: 完整 RAG 三步 — retrieve 候选 (复用 代码 2 词袋), augment 拼 prompt, generate 调 LLM。

**7 步**:
1. 加载段落 (复用 代码 1/代码 2 的 `python-docx` 加载, 保持 `paragraphs: list[str]` 输入形状一致)
2. 接收 query 输入 (`input("问点啥: ").strip()`)
3. `retrieve`: 复用 代码 2 的词袋 + 余弦, 取 top-3 段落 (`retrieve(q, paragraphs, k=3)`)
4. `build_prompt`: 把 top-3 渲染成 `[1] ... [2] ... [3] ...`, 每段一个 `[i]` 编号, 包进 `<context>...</context>` 标签做边界
5. 拼 system 约束 ("你只能依据 <context> 标签内的资料回答问题, 若资料不足以回答请回复「我不知道」") — prompt 硬约束最弱的一道防线, 防 LLM 自由发挥
6. `call_llm`: 调 OpenAI 兼容 `/chat/completions` (`urllib.request` 零 SDK 依赖); 无 `LLM_API_KEY` 时只打印 prompt 后停 (教学兜底, 便于无 key 机器验证链路)
7. 解析 LLM 输出, 把 `[i]` 引用号与 hits 对齐 (本题不实现, 见 s08 工业 prompt 模板)

```python
# 中间片段: build_prompt — 把 hits 渲染成 prompt
def build_prompt(question: str, hits: list[str]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] {h}" for i, h in enumerate(hits))
    return (
        "你只能依据 <context> 标签内的资料回答问题；\n"
        "若资料不足以回答，请回复「我不知道」。\n\n"
        f"<context>\n{ctx}\n</context>\n\n"
        f"问题: {question}\n"
        "回答: "
    )
```

**完整函数**:

```python
def retrieve(query: str, paragraphs: list[str], k: int = 3) -> list[str]:
    """复用 代码 2 的 2-gram 词袋 + 余弦, 取 top-k 段落."""
    vocab = build_vocab(paragraphs)
    para_vecs = [vectorize(p, vocab) for p in paragraphs]
    q_vec = vectorize(query, vocab)
    scored = sorted(
        zip(paragraphs, (cosine(q_vec, pv) for pv in para_vecs)),
        key=lambda x: -x[1],
    )
    return [p for p, _ in scored[:k]]


def build_prompt(question: str, hits: list[str]) -> str:
    """对照 docs/reference/ragflow-notes/prompt_templates.md 里的 'You are an AI assistant...' 模板.
    本章用极简版, 只保留 [i] (source) text 的渲染."""
    ctx = "\n\n".join(f"[{i + 1}] {h}" for i, h in enumerate(hits))
    return (
        "你只能依据 <context> 标签内的资料回答问题；\n"
        "若资料不足以回答，请回复「我不知道」。\n\n"
        f"<context>\n{ctx}\n</context>\n\n"
        f"问题: {question}\n"
        "回答: "
    )


def call_llm(prompt: str) -> str:
    """最小可用 OpenAI 兼容调用; 零 SDK 依赖."""
    if not LLM_API_KEY:
        return ""

    import json
    req = urllib.request.Request(
        f"{LLM_BASE}/chat/completions",
        data=json.dumps({
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }).encode(),
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def main() -> None:
    paragraphs = load_paragraphs(SAMPLE)
    q = input("问点啥: ").strip()

    hits = retrieve(q, paragraphs, k=3)
    print(f"\n[retrieve] 召回 {len(hits)} 段")
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] {h[:80].replace(chr(10), ' ')}...")

    prompt = build_prompt(q, hits)
    print(f"\n[prompt]\n{prompt}\n")

    if LLM_API_KEY:
        answer = call_llm(prompt)
        print(f"[llm] {answer}")
    else:
        print("[llm] LLM_API_KEY 未设置，跳过真实生成；如需 LLM 回答:")
        print("      LLM_API_KEY=sk-xxx python s01_what_is_rag/c03_rag_pipeline.py")
```

### 试一下

```bash
python s01_what_is_rag/c03_rag_pipeline.py
```

预期输出示例 (无 key, 交互输入 `关联方披露`):

```
[retrieve] 召回 3 段
  [1] 相关信息披露详见财务报表附注三(二十五)...
  [2] ...
  [3] ...

[prompt]
你只能依据 <context> 标签内的资料回答问题;
若资料不足以回答, 请回复「我不知道」。

<context>
[1] 相关信息披露详见财务报表附注三(二十五)...
[2] ...
[3] ...
</context>

问题: 关联方披露
回答:

[llm] LLM_API_KEY 未设置, 跳过真实生成...
```

- 无 `LLM_API_KEY`: 打印 retrieve 结果 + prompt 后停 (graceful-skip, 便于无 key 机器验证链路)
- 有 `LLM_API_KEY`: LLM 生成引用答案, 接在 `回答:` 后

**观察**: retrieve 候选 → build_prompt 拼接 → LLM 调用三段流水线的脆弱性。召回错 (代码 2 词袋丢语义) + prompt 极简 (无引用跟踪) + LLM 无 retry, 任一环节失败全错。这是后续章节每一段都要加固的入口: s04 修召回质量, s06 修单路召回, s08 修 prompt 极简, s11 修 LLM 调用脆性。

### 为什么不只写这一种

代码 3 是 RAG 的最小闭环, 但每一段都极简 — retrieve 只有词袋单路 (召回错则全错), prompt 无引用跟踪 (LLM 编 `[5]` 但 prompt 只有 `[1][2][3]` 是高频故障), call_llm 无 retry / streaming。后续 s06 加 hybrid 召回, s07 加 rerank, s08 加工业 prompt 模板 + 引用检测, 才能上生产。

---

## 接下来

s01 是 RAG 的最小骨架, 但每一步都很脆弱, 这些脆弱性是后续 11 章的填空目标:

- **代码 1 找不到同义词** — 字面匹配的固有限制, query `营收` 找不到 chunk 里的 `营业收入`, query `披露` 找不到 chunk 里的 `公开 / 公告`。这是 代码 2 词袋 → s04 BGE 真语义要解决的召回质量问题。
- **代码 2 词袋丢语义 + 丢语序** — 维度爆炸 (2-gram 笛卡尔积), 稀疏向量无真语义, "狗咬人" = "人咬狗" 同分, "摘要式披露" = "详细披露" 同分。这是 s04 BGE dense 512 维向量要解决的语义召回。
- **代码 3 召回错 + prompt 极简 + 无 rerank** — 召回质量是整条流水线的天花板 (top-3 不相关, LLM 拿到错的资料就编答案); prompt 无引用跟踪会让 LLM 编 `[5]` 但 prompt 里只有 `[1][2][3]`, 这是 s06 hybrid + s07 rerank + s08 工业 prompt 要解决的"召回 + 精排 + 生成" 全链路加固。

s02 **文档加载**: 给 RAG 一份"能读"的资料 — 把 `python-docx` 单文件解析扩成 PDF + DOCX + TXT 多 Parser 调度, 产出统一的 `{text, page, source}` schema, 让 代码 2 的输入不再依赖单一文件类型, s03 chunking 才有干净原料。这是把"玩具 RAG"推向"工业 RAG"的第一步 — 没有稳定的文档加载, 后面的 chunking / embedding / retrieval 都建立在沙滩上。

---

## 思考题

1. **代码 1 子串匹配为什么找不到同义词?** 举一个具体 query 例子说明。
2. **代码 2 词袋 + 余弦相似度, 为什么需要向量归一化?** 不归一化会怎样?
3. **代码 3 的 retrieve → build_prompt → call_llm 三步, 哪一步最脆弱?** 为什么?

---

## 思考题答案

### Q1. 代码 1 子串匹配为什么找不到同义词?

字面匹配只看 "字符包含", 不看语义。query `披露` 只在 chunk 中**字面包含** "披露" 时才命中; 当 chunk 写 "公开" / "公告" / "告示" 时, 即便语义等价也不命中。这在企业知识库场景 (同一概念的不同表述, 比如 "营收" / "营业收入" / "主营业务收入", 或者 "披露" / "公开" / "公告") 是高频故障 — 用户用自己习惯的词提问, 但文档里用的是公司内部术语, 子串匹配会全部 miss。

要解决: 代码 2 词袋模型把 query 和 chunk 都转成 2-gram 词频向量, 余弦相似度能在部分情况下给出非零分 (比如 `披露` 和 `公开披露` 共享 `披露` 这个 2-gram); 但更彻底的解决是 s04 的 BGE 真语义向量, 把语义相关的词压到向量空间的近邻位置 — 即便字符完全不重合, "披露" 和 "公告" 的 BGE 向量也会有高余弦相似度。

### Q2. 代码 2 词袋 + 余弦相似度, 为什么需要向量归一化?

余弦相似度的数学定义是 `cos(θ) = dot(a,b) / (norm(a) * norm(b))`, 本质上是**两个向量方向**的夹角, 与向量的**长度**无关。但在词袋场景下, 长 chunk 的向量 L2 范数远大于短 chunk — 同样是 "披露" 这个 token, 在 1000 字 chunk 里出现 1 次对应词频向量某个维度 = 1, 在 100 字 chunk 里出现 1 次对应同一维度也是 1, 但长 chunk 的总维度更多 (其他 token 也贡献非零维度), 总 L2 范数更大。

不归一化时, 长度主导相似度: 两个不相关的长 chunk 也会因大量 "非零维度" 拿到非零余弦分, top-k 排序退化成 "找最长 chunk", 失去 "找最相关 chunk" 的语义。归一化 (`normalize_embeddings=True` 或代码里的 `cosine()` 函数) 把所有向量投影到单位球面, 余弦分只反映方向相似度, 与长度无关, top-k 才能真正按语义排序。这是工业 embedding 检索 (BGE / OpenAI text-embedding-3) 默认开 `normalize_embeddings=True` 的原因。

### Q3. 代码 3 三步, 哪一步最脆弱?

**build_prompt 最脆弱**。retrieve 失败只是丢召回 (top-k 全错), call_llm 失败只是丢生成 (网络 / key 错 / 超时), 这两步失败模式**边界清晰**、可观测、可重试 — 监控告警可以精确定位是召回还是生成的问题; 但 build_prompt 模板的任何缺陷 (无引用跟踪、无 retry、无 streaming) 会**直接放大到 LLM 输出** — LLM 编一个 `[5]` 引用号, 而 prompt 里只有 `[1][2][3]`, 用户看到的就是一个看起来引用规范、实际全错的答案。

这种 "看起来很对" 的错误是 RAG 系统最难排查的故障, 因为它在 prompt 层无法被检测, 只能靠输出侧的引用验证 (字符串匹配 / LLM-as-judge 扫答案里每个事实句是否贴 `[i]`) 才能拦下。即便验证出 `[5]` 是幻觉引用, 重试一次也可能得到 `[7]`, 因为 LLM 没有强制只能引用 prompt 里有的编号。

生产里必须加固: s08 的工业 prompt 模板加 `<|COMPLETE|>` 哨兵 + 明确拒答边界 ("只能引用 [1]-[3] 内的编号, 资料外回答「我不知道」"); 输出侧加引用验证 + 重试机制 (扫到幻觉引用号就拒绝回答); UI 渲染层强制每句话末尾贴引用, 没引用的句子标红或丢弃。s01 只做了 prompt 硬约束这一层, 留给后续章节填空。