# s01 什么是 RAG

> 本章用 30 行 Python 代码做一个"假 RAG"，先把问题讲清楚，再讲真问题。

## 问题

LLM 看似什么都知道，但有三个它回答不好的场景：

1. **训练截止之后的事** —— 比如你问"我们公司去年的营收是多少"或"今天天气如何"。模型只见过截止日的语料，没法凭空知道你的私有数据。
2. **私有/内部数据** —— 公司年报、内部 wiki、合同 PDF……这些不在公开互联网上，模型从未见过。
3. **编造（幻觉）** —— 当模型不知道答案时，它常常一本正经地瞎编——尤其是数字、人名、引文这类"看起来很确定"的内容。

**RAG（Retrieval-Augmented Generation，检索增强生成）** 的直觉做法很简单：回答之前先"翻一翻"相关文档，把找到的片段塞进 prompt，再让 LLM 基于这段上下文回答。这样：

- 知识可以更新（换文档就行）
- 可以引用私有数据
- 生成的内容有据可查，幻觉更少

RAG 不是把 LLM 训练一遍，而是把"查资料"这个动作外置。

## 最小解法

下面这段 30 行的 `code.py` 实现了一个最朴素的 RAG：用户在终端输入一句话，程序到 `samples/disclosure.docx` 里**逐段扫描**，找第一个包含问题中任意一个词的段落，作为答案返回。完全不用向量数据库、完全不用 LLM 调用——只有一个文件、一个函数。

```python
#!/usr/bin/env python3
"""
s01 什么是 RAG — 用 30 行做一个"假 RAG"：在文档里找包含问题的段落，作为答案。

运行: python s01_what_is_rag/code.py
需要: 无外部依赖；samples/disclosure.docx 存在
"""
from pathlib import Path
from docx import Document

# 把 samples/ 路径写死：上层目录 + samples/disclosure.docx
WORKDIR = Path(__file__).parent.parent
SAMPLE = WORKDIR / "samples" / "disclosure.docx"


def load_paragraphs(path: Path) -> list[str]:
    # 只取非空段落，去掉 Word 文档里那些"占位用的空段"
    return [p.text for p in Document(path).paragraphs if p.text.strip()]


def fake_rag(question: str, paragraphs: list[str]) -> str:
    # 朴素检索：拿问题里的每个词去段落里找子串，命中就返回第一个命中的段落
    for p in paragraphs:
        if any(w.lower() in p.lower() for w in question.split()):
            return p
    return "I don't know."


def main() -> None:
    # 启动时一次性读完全部段落，之后的检索都在内存里
    paragraphs = load_paragraphs(SAMPLE)
    question = input("问点啥: ").strip()
    print(fake_rag(question, paragraphs))


if __name__ == "__main__":
    main()
```

读法和真实 RAG 完全同构：**先检索（retrieve），再拼上下文（augment），最后送生成（generate）**——只是这里第二步被简化成"直接返回找到的段落"，连 LLM 都不用。

## 跑起来

```bash
cd learn-ragflow
python s01_what_is_rag/code.py
```

输入对照（用 `samples/disclosure.docx` 实测）：

| 输入 | 输出 |
|---|---|
| `披露` | `相关信息披露详见财务报表附注三(二十五)、五 (二)1 及十五(二)。` |
| `外星人` | `I don't know.` |

把第一个命中段落直接当答案是这个 toy 的卖点：你能一眼看到"检索"这一段确实是真 RAG 不可缺少的第一步，哪怕后面对它做什么样的加工（重排序、切片、摘要、引用溯源），输入依然是这个段落清单。

## 真实世界的问题

这个 toy 演示了"检索"的概念，但任何让它跑在真实数据上的人都会立刻看到两个问题：

1. **找不到关键词就彻底失效** —— 用户问"营收多少钱"，而文档里写的是"主营业务收入"。朴素子串匹配连一行都返回不了，只能输出 `I don't know.`。真实场景下，提问和文档**几乎不会用同样的词**。
2. **找到了关键词，但段落是错的答案** —— 用户问"应收账款怎么计提坏账"，文档碰巧有一段提到"应收账款"但其实是讲会计科目列表，第一句话并不回答"计提坏账"的问题。关键词命中 ≠ 语义相关。

这两个问题分别对应：

- **检索召回（recall）** —— 怎么在词不匹配的情况下也找到相关段落？这正是后面章节讲的 chunking、embedding、向量检索要解决的事。
- **检索精排（precision）** —— 怎么在候选里挑出最相关的段落？这是 reranking 和 prompt 工程要解决的事。

所以从第二章开始，我们逐步把"朴素子串"换成"向量相似度 + 关键词召回"，再换成"重排序 + 多路融合"——本质上都是为了让上表的"输出"列更靠谱。

## ragflow 怎么做的

本章不展开 RAGFlow 的源码。先简单说一句它是什么、为什么本教程要读它：

[**RAGFlow**](https://github.com/infiniflow/ragflow) 是一个开源的、面向生产环境的 RAG 引擎。它的特点是**把文档解析 → 结构化分块 → 混合检索 → 引用溯源**串成了完整流水线，并把每一阶段都做了工程化（多种文档格式、OCR、表格理解、Agent 工具、人机协作）。

本教程的"最小 MVP + 工业对照"风格，就是想**一面让你动手写 30 行小玩具，一面对照 RAGFlow 看真实工程是怎么把这些小玩具拼起来、加约束、做权衡的**。每章的 README 第 5 节都会引用 `../ragflow_notes/` 里相应的源码摘录。

更多说明见 [`../ragflow_notes/README.md`](../ragflow_notes/README.md)。

## 思考题

**把 `fake_rag` 改成返回 Top-3 候选段落，怎么打分？**

（提示：最简单的版本就是数"命中的关键词数量"。答案见 `thinking_answers.md`。）

这一步其实就是后面"向量检索"的原始形态——只是用关键词命中次数代替了语义向量相似度。先把直觉搭起来，后面章节会反复回到这个对比。
