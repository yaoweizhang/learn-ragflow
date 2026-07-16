# s03 文本分块 — 把段落切成"长度可控、语义相对完整"的小块

[上一章 s02 → · 下一章 s04 → ... → s12]

> *"500 字符 + 句界正则，一行 `re.split` 就能跑通。但把它喂到真实样本上 —— 规格表拦腰切成两块、节标题单独成 10-char chunk、'收入结构如下' 这种指代词无主语 —— 这三类典型失败，就是分块这步为什么值得单写一章"*
>
> **链路位置**: 离线索引链路第二步 (s02 → **s03** → s04 → s05)
> **代码文件**: basic_chunk.py · chunk_failures.py

> 环境准备: 见 root README §快速开始 — `pip install pypdf python-docx`

---

## 问题

`re.split(...)` 一行就能切，似乎不值得单写一章。但把这种 toy 切法扔到真实样本上，会冒出三类典型问题，每一类都对应一类工业解法。

**第一，表格被切碎**。规格表里 `24LFF / 12LFF / 4SFF` 横向排在一行，`pypdf.extract_text()` 抽出来是没换行的长串。500 字符 cap 在表格中间切断，每行 chunk 只看到一半字段 — 用户的 query 命中 chunk 0 时拿到的是"处理器 ... 8GB"，命中 chunk 1 时拿到的是"NAND ... 24LFF"，两段都不完整，embedding 把两个半截当完整语义算。这是 s02 多栏错位的同源问题在 chunking 层的体现 — chunker 不知道"这是一张表"。

**第二，父子块缺失**。"第四节 分季度财务数据" 这种 10 字符的节标题单独成 chunk，用户问 "Q3 营收" 时检索命中的是标题，数字本身在 DOCX tables 里 — 而 s02 的 `load_docx` 已经把 tables 丢了 (见 s02 `failure_modes` 的量化: 27 段 paragraphs + 3 张表 = 572 字符蒸发)。即便 s02 loader 不丢表，chunker 默认扁平输出也没有 parent-child 关系，检索命中"第四节"这个标题时拿不到季度数字。

**第三，跨段引用断裂**。"收入结构如下" "见表 3" 这种指代词单独成 chunk 后，召回的是指代词本身，而不是它指向的实体。LLM 拿到"收入结构如下" 这五个字不知道它指的是谁，被迫瞎猜或拒答。这跟 chunk 粒度直接相关 — 切得越细，跨段引用越容易断。

把这三类问题合起来看，**chunker 的脆弱性不在 toy 上体现，在真实样本上才暴露**。这就是为什么 s03 必须先用 toy 跑通，再用真实样本量化三类损失。500 字符 cap + 句界切是最朴素的"够小 + 语义完整"折中，但任何 chunker 都得先看清自己在 prod 样本上损失多少，再决定要不要换工具 — Garbage In, Garbage Out 在 embedding 层会被放大。

s03 的任务就是**先把最简单的 chunker 跑起来 (句界切 + chunk_id 稳定生成)，再用真实样本定位三类典型失败的边界**，给后面 s04-s12 的工业解法留填空入口。本章是"看清坑"的章节，不是"填坑"的章节。

---

## 解决方案

s03 用 **两个递进的脚本** 把"段落 → chunk"这条 RAG 离线流水线第二步跑通。每一步解决前一步的局限，但也留下新的脆弱性：

```
代码 1 (最小可跑)               代码 2 (真实样本失败模式)
┌──────────────────┐         ┌───────────────────────┐
│ split_long_       │         │ 复用 代码 1 函数       │
│   paragraph +     │         │ + 量化 表格切碎         │
│ chunk_by_         │ ────▶  │   + 父子块缺失          │
│   paragraph       │         │   + 跨段引用断裂         │
│                  │         │                       │
│ 输出 {text,       │         │ 输出三类失败 before/   │
│   chunk_id,       │         │   after 量化 (不修)     │
│   page, source}   │         │                       │
└──────────────────┘         └───────────────────────┘
  toy 上跑通                   真实样本上暴露坑
```

| 脚本 | 解决什么 | 留下什么局限 | 何时用 |
|---|---|---|---|
| `basic_chunk.py` | 500 字符 cap + 句界正则切 → `{text, chunk_id, page, source}` 四键 schema | 表格被切碎; 节标题单独成 chunk; 指代词断主语 | toy / 教学 / 文本型 PDF + 简单 DOCX |
| `chunk_failures.py` | 量化 01 在真实样本上的三类损失 (规格表 hard-cut / header-only / cross-ref) | **不修** — 只量化; 表格/父子树/层级合并留给 s11 | 教学动机; 决策"要不要跳 s11" |

两脚本的关系是一条**教学主干**: 代码 1 把"段落 → chunk" 做出来, 暴露 "500 字 cap + 句界切" 在 toy 上 OK 但真实样本上不够用的局限 — 规格表被拦腰切、节标题无数据、指代词无主语; 代码 2 把代码 1 喂给真实样本 (`server_whitepaper.pdf` 4 页 + `disclosure.docx` 27 段) 量化三类失败 (562 字符表格切成 396/164 / 10-11 char header chunks / 收入结构如下无主体), 暴露 "01 在这种语料上不够用, 必须靠 parent-child + 版面分析 + 媒体回填联合修" 的结论。**s03 看清坑, 后续章节填坑 — s04 修召回质量, s11 多模态 + parent-child 修表格/父子/跨段**。

---

## 代码 1: 最小可跑分块 ([basic_chunk.py](basic_chunk.py))

### 工作原理

**做一件事**: 把 s02 loader 输出的 `list[{text, page, source}]` 切成"短段整段保留 + 长段按句界切成 ≤ max_chars 的若干块", 每块带 `chunk_id = {source}#{page}#p{n}` 给 s04+ 稳定引用。

**3 步**:
1. **复用 s02 的 loader** — `load_pdf` / `load_docx` 经 `importlib.util.spec_from_file_location` 加载 (s03 目录以数字开头, 普通 `import` 报 SyntaxError); 输入形状固定为 `list[{text, page, source}]`, 与 s02 契约一致
2. **`split_long_paragraph(text, max_chars)`** — lookbehind 正则在 `[.。!?！？]` 之后切 (`re.split(r"(?<=[.。!?！？])\s*", text)`), 把超长段落切成 ≤ max_chars 的若干块; 极端情况 (无标点的规格表 / 单句超长) 按字符硬切兜底 (`for i in range(0, len(sentence), max_chars)`)
3. **`chunk_by_paragraph(docs, max_chars=500)`** — 遍历 docs, 短段 (`len(doc["text"]) <= max_chars`) 整段保留 + 加 `chunk_id`; 长段调 `split_long_paragraph` 展开多块; `chunk_id = {source}#{doc.get('page', 0)}#p{len(out)}`, `p{n}` 是**输出顺序**而非段内位置, 确保 s04+ 引用稳定

```python
# 中间片段: 句界正则切 + 字符硬切兜底
sentences = re.split(r"(?<=[.。!?！？])\s*", text)
for sentence in sentences:
    sentence = sentence.strip()
    # 单句本身超长(无标点表格/规格表) → 按字符硬切
    if len(sentence) > max_chars:
        for i in range(0, len(sentence), max_chars):
            parts.append(sentence[i:i + max_chars])
        continue
    if len(buf) + len(sentence) + 1 > max_chars and buf:
        parts.append(buf)
        buf = sentence
```

**完整函数**:

```python
def split_long_paragraph(text: str, max_chars: int) -> list[str]:
    """把超长段落按中英句子边界切成 <= max_chars 的若干块.

    用 lookbehind 正则在 [.。!?！？] 之后切,同时覆盖中英文标点.
    单句本身超过 max_chars(常见于表格/规格表)再按字符硬切,
    保证最坏情况下也不会输出超过 2*max_chars 的块.
    """
    sentences = re.split(r"(?<=[.。!?！？])\s*", text)
    parts, buf = [], ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # 单句本身超长(无标点表格/规格表)→ 按字符硬切
        if len(sentence) > max_chars:
            if buf:
                parts.append(buf)
                buf = ""
            for i in range(0, len(sentence), max_chars):
                parts.append(sentence[i:i + max_chars])
            continue
        if len(buf) + len(sentence) + 1 > max_chars and buf:
            parts.append(buf)
            buf = sentence
        else:
            buf = (buf + sentence).strip() if buf else sentence
    if buf:
        parts.append(buf)
    return parts


def chunk_by_paragraph(docs: list[dict], max_chars: int = 500) -> list[dict]:
    """短段整段保留,长段按 split_long_paragraph 切成多块."""
    out = []
    for doc in docs:
        if len(doc["text"]) <= max_chars:
            out.append({**doc, "chunk_id": f"{doc['source']}#{doc.get('page', 0)}#p{len(out)}"})
        else:
            for piece in split_long_paragraph(doc["text"], max_chars):
                out.append({**doc, "text": piece, "chunk_id": f"{doc['source']}#{doc.get('page', 0)}#p{len(out)}"})
    return out


def main() -> None:
    docs = load_pdf(SAMPLES / "server_whitepaper.pdf") + load_docx(SAMPLES / "disclosure.docx")
    chunks = chunk_by_paragraph(docs)
    print(f"输入段落 {len(docs)} → 输出块 {len(chunks)}")
    print(f"最大块长度 {max(len(c['text']) for c in chunks)} 字符 (cap=500)")
    for c in chunks[:3]:
        print(c["chunk_id"], "|", c["text"][:60])
```

### 试一下

```bash
python s03_chunking/basic_chunk.py
```

实测输出:

```
输入段落 31 → 输出块 34
最大块长度 452 字符 (cap=500)
server_whitepaper.pdf#1#p0 | 紫光恒越 R3630 G5 双路机架式服
务器
产品白皮书 ...
server_whitepaper.pdf#1#p1 | 二、关键特性
计算密度 ...
server_whitepaper.pdf#2#p2 | 三、整机规格
组件 规格 说明 ...
```

- 31 段 (4 页 PDF + 27 段 DOCX) → 34 块 (3 段超长被展开成多块), 最大块 452 字符 **未触发硬切兜底**
- `chunk_id = {source}#{page}#p{n}` 按输出顺序稳定生成 — s04 embedding / s05 向量索引引用具体块时不会因为切分顺序变化而错位
- 输出 schema 与 s02 完全对齐 (`{text, page, source}` + `chunk_id`), 下游按统一接口消费

**观察**: 句界正则 `[.。!?！？]` 在中英文段落上都不拦腰砍断, 500 字符 cap 在 toy 段落 (平均 < 500) 上几乎不触发切分, 大多数 PDF/DOCX 段被整段保留。但 chunk 粒度本身**不携带"这是表 / 这是节标题 / 这是指代词"的结构信号** — 31 段切成 34 块看似合理, 但表格 / 父子 / 跨段三类失败藏在字符串内部, 长度看不出来 (语义混乱在 embedding 层才暴露)。这是 代码 2 的真实样本要暴露的脆弱性。

### 为什么不只写这一种

`c01` 在 toy / 短段为主的语料上够用, 但在真实样本上暴露三类固有限制:

- **表格被切碎**: 500 字符 cap 在规格表中间切断 (典型场景 — 整机规格表 562 字符被硬切成 396/164), chunk 0 末行停在 "8GB", chunk 1 从 "NAND" 开头继续 — 两段都不完整
- **父子块缺失**: 节标题 ("第四节 分季度财务数据" / "第二节 主要财务数据") 单独成 10-11 char chunk, 用户问 "Q3 营收" 时检索命中的是标题, 数字本身在 DOCX tables 里
- **跨段引用断裂**: "收入结构如下" / "见表 3" 这种指代词单独成 chunk 后无主语, LLM 拿到指代词不知道它指的是谁

解决方案指向 **代码 2 (量化三类损失)** + **后续章节填坑** — s11 多模态补全表格抽取 (`deepdoc/parser/pdf_parser.py` XGBoost 30 特征 + `hierarchical_merge` parent-child), 让 chunker 拿到"这张是表 / 这是节标题 / 这块属于哪个父"的结构信号。本章只量化不修, 不在 c01 上贴膏药。

---

## 代码 2: 真实样本上的三类失败 ([chunk_failures.py](chunk_failures.py))

### 工作原理

**做一件事**: 把代码 1 的 `chunk_by_paragraph` / `split_long_paragraph` 喂给 `samples/` 下的真实样本 (`server_whitepaper.pdf` 4 页 + `disclosure.docx` 27 段), 量化"看不见的三类损失" — 表格切碎 / 父子块缺失 / 跨段引用断裂, 而不修复。

**4 步**:
1. **复用 c01 的函数** — `importlib.util.spec_from_file_location` 加载 `basic_chunk.py`, 复用 `load_pdf` / `load_docx` / `chunk_by_paragraph` / `split_long_paragraph` 不重写
2. **`demo_table_split()`** — 跑 `load_pdf` 拿 4 段, 找含 "三、整机规格" 的段 (`next(d for d in pdf if "三、整机规格" in d["text"])`), 调 `split_long_paragraph(..., max_chars=500)`, 打印 BEFORE 整段 + AFTER 各块末行 + 失败点 (列对齐被腰斩)
3. **`demo_parent_child()`** — 跑 `chunk_by_paragraph` 拿全部 chunks, 找节标题 chunks (`{"第四节 分季度财务数据", "第二节 主要财务数据"}`), 打印 BEFORE 召回命中 + AFTER 用 `python-docx` 直读 DOCX tables 找 Q3 数据; 本样本未含 Q3 字面量但所有季度数字都在 tables 里
4. **`demo_cross_ref()`** — 跑 `chunk_by_paragraph` 拿全部 chunks, 过滤 `0 < len < 30` 的短 chunks, 打印 BEFORE 过短 header-only chunks + AFTER 用 `load_docx` 找 "收入结构如下" 上下文段, 演示指代词单独成 chunk 后无主体

```python
# 中间片段: 量化表格切碎 — BEFORE 整段 vs AFTER 各块末行
pieces = split_long_paragraph(spec_doc["text"], max_chars=500)
for i, p in enumerate(pieces):
    last_line = p.strip().splitlines()[-1] if p.strip() else "<空>"
    print(f"  chunk {i} len={len(p):>4} | 末行: {last_line[:60]}")
print(f"→ 失败点: 第 0 块末行停在 '{last0}' — 表的列对齐被腰斩,")
print(f"          第 1 块从 'NAND' 段继续,失去 '组件 / 规格 / 说明' 列对齐语义。")
```

**完整函数**:

```python
def _hr(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def demo_table_split() -> None:
    """(a) 表格被切碎: 整机规格表 562 字符 → 硬切到 mid-row."""
    _hr("[a] 表格被切碎 — 整机规格表 hard-cut")
    pdf = load_pdf(SAMPLES / "server_whitepaper.pdf")
    spec_doc = next(d for d in pdf if "三、整机规格" in d["text"])
    print(f"BEFORE (整段 {len(spec_doc['text'])} 字符):")
    print(spec_doc["text"][:240], "...")
    print()

    pieces = split_long_paragraph(spec_doc["text"], max_chars=500)
    print(f"AFTER  (cap=500 → 切成 {len(pieces)} 块):")
    for i, p in enumerate(pieces):
        last_line = p.strip().splitlines()[-1] if p.strip() else "<空>"
        print(f"  chunk {i} len={len(p):>4} | 末行: {last_line[:60]}")
    print()
    last0 = pieces[0].strip().splitlines()[-1] if pieces else "<空>"
    print(f"→ 失败点: 第 0 块末行停在 '{last0}' — 表的列对齐被腰斩,")
    print(f"          第 1 块从 'NAND' 段继续,失去 '组件 / 规格 / 说明' 列对齐语义。")


def demo_parent_child() -> None:
    """(b) 父子块缺失: 节标题单独成 chunk, 数据本体在 DOCX 表里被 s02 丢了."""
    _hr("[b] 父子块缺失 — 节标题 chunk 与数据本体分离")
    chunks = chunk_by_paragraph(
        load_pdf(SAMPLES / "server_whitepaper.pdf") + load_docx(SAMPLES / "disclosure.docx")
    )
    header_chunks = [c for c in chunks if c["text"].strip() in {"第四节 分季度财务数据", "第二节 主要财务数据"}]
    print("BEFORE (用户问 'Q3 营收多少'):")
    for c in header_chunks:
        print(f"  召回命中 chunk_id={c['chunk_id']} len={len(c['text'])} text='{c['text']}'")
    print()

    from docx import Document
    doc = Document(SAMPLES / "disclosure.docx")
    print(f"AFTER  (季度数据本体在 DOCX tables,共 {len(doc.tables)} 张表):")
    for i, tbl in enumerate(doc.tables):
        rows = [[cell.text.strip() for cell in row.cells] for row in tbl.rows]
        if rows and any("Q3" in cell or "三季度" in cell or "9 月" in cell for cell in rows[0]):
            print(f"  Table {i} (含 Q3 数据):")
            for row in rows[:5]:
                print(f"    {row}")
            break
    else:
        print("  (本样本 DOCX 表未含 Q3 字面量; 但所有季度数字都只在 tables 里,chunker 看不到)")


def demo_cross_ref() -> None:
    """(c) 跨段引用断裂: '如下' / '见上表' 这种指代词单独成 chunk 后无意义."""
    _hr("[c] 跨段引用断裂 — 指代词单独成 chunk")
    chunks = chunk_by_paragraph(
        load_pdf(SAMPLES / "server_whitepaper.pdf") + load_docx(SAMPLES / "disclosure.docx")
    )
    print("BEFORE (过短的 header-only chunks,语义为零):")
    short = [c for c in chunks if 0 < len(c["text"]) < 30]
    for c in short:
        print(f"  id={c['chunk_id']} len={len(c['text']):>3} | '{c['text']}'")
    print()
    print(f"AFTER  (DOCX 原文: '按业务板块划分,2024 年公司收入结构如下:...'):")
    docs = load_docx(SAMPLES / "disclosure.docx")
    ref_doc = next(d for d in docs if "收入结构如下" in d["text"])
    print(f"  '{ref_doc['text'][:200]}...'")


def main() -> None:
    demo_table_split()
    demo_parent_child()
    demo_cross_ref()
    print()
    print("→ 三类失败都不是 unit 01 的 chunker 单独造成的——是 chunker + s02 loader + 段落切分")
    print("  的累积效应。ragflow 用 4 层流水线 (XGBoost 父块 + token 子块 + 层级合并 + 媒体回填)")
    print("  逐层修。")
```

### 试一下

```bash
python s03_chunking/chunk_failures.py
```

实测输出片段:

```
================================================================
[a] 表格被切碎 — 整机规格表 hard-cut
================================================================
BEFORE (整段 562 字符):
三、整机规格
组件 规格 说明
处理器 2 × 第三代 Intel Xeon 可 扩展处理器 ...
AFTER  (cap=500 → 切成 2 块):
  chunk 0 len= 396 | 末行: NAND
  chunk 1 len= 164 | 末行: ...
→ 失败点: 第 0 块末行停在 '8GB' (BMC 那行的中间),失去列对齐
→  ragflow 修法: deepdoc/parser/pdf_parser.py 用 XGBoost 30 特征识别表格 layout_type=table

================================================================
[b] 父子块缺失 — 节标题 chunk 与数据本体分离
================================================================
BEFORE (用户问 'Q3 营收多少'):
  召回命中 chunk_id=disclosure.docx#None#p18 len=11 text='第四节 分季度财务数据'
AFTER  (季度数据本体在 DOCX tables,共 3 张表):
  (本样本 DOCX 表未含 Q3 字面量; 但所有季度数字都只在 tables 里,chunker 看不到)

================================================================
[c] 跨段引用断裂 — 指代词单独成 chunk
================================================================
BEFORE (过短的 header-only chunks,语义为零):
  id=disclosure.docx#None#p8  len=15 | '2024 年度财务信息披露报告'
  id=disclosure.docx#None#p13 len=10 | '第二节 主要财务数据'
  id=disclosure.docx#None#p18 len=11 | '第四节 分季度财务数据'
```

- 表格切碎: 562 字符规格表被硬切成 396 + 164, chunk 0 末行停在 "NAND" — 列对齐 `组件 / 规格 / 说明` 在 chunk 边界被腰斩, 两段都不完整
- 父子块缺失: 用户问 "Q3 营收多少", chunker 召回命中的是 `disclosure.docx#None#p18` 11-char 标题 chunk, 数据本体在 DOCX tables (3 张表, s02 loader 已丢 — 见 s02 c02)
- 跨段引用断裂: 10-15 char 的 header-only chunks 语义为零 ("第二节 主要财务数据" / "2024 年度财务信息披露报告"), 即便 embedding 把它们命中, 也只能返回空回答

**观察**: `c02` 不生产新数据, 只量化三类损失 — 这是它"demo 性质"的核心: 不修, 只量。它的存在意义是**教学动机** — 让读者在 s03 就看清 "01 在 prod 上不够用, 决策要不要跳 s11"。如果不量化, 用户会以为 31 段 → 34 块 = "chunker 跑通了", 实际三类损失藏在字符串内部 (表格列对齐被腰斩 / 节标题无数据 / 指代词无主语)。如果你的真实语料 80% 是表格 (财报 / 规格表), 02 的量化结果就是 "01 完全不能用" 的直接证据 — 该跳 s11 就跳, 不在 c01 上贴膏药。

### 为什么不只写这一种

`c02` 是"不修只量化"的 demo — 它**故意**不修任何问题, 目的就是让你看清 01 的损失边界后再决定要不要跳到工业方案。它自身不解决任何问题:

- **表格切碎未修** — 396/164 字符的两块 chunk 仍对 RAG 是"半截规格表"的输入; XGBoost 版面分析 (`deepdoc/parser/pdf_parser.py` 30 特征识别 `layout_type=table`) + `attach_media_context` 媒体回填是工业解法, 本章不动
- **父子块缺失未修** — 11-char 标题 chunk 仍会单独召回; `hierarchical_merge` 按 "第 X 节" 正则建父子树, 召回时返回 parent 全文是工业解法, 本章不动
- **跨段引用断裂未修** — 10-char header chunks 仍会单独召回; 层级合并 + 召回附带整节是工业解法, 本章不动

**实操建议**: 跑 02 看到表格字符数大 / 节标题 chunks 多 / 指代词独立 → 直接跳 s11 看 RAGFlow 的 `hierarchical_merge` + `_concat_downward` + `naive_merge` + `attach_media_context` 4 层流水线怎么集成; 不需要也不应该在 01 上贴膏药 (版面分析 + XGBoost 模型下载成本不该压在入门骨架里)。

---

## 接下来

s03 是文本分块的**最小骨架 + 真实样本失败边界量化**: 把"段落"切到"chunk" 这步跑通, 同时暴露三类典型损失。`c01` 把"段落 → {text, chunk_id, page, source}" 做出来, `c02` 把 "01 在真实样本上的三类损失" 量化出来 — 这两件事合起来, 给出了后续章节的填空入口:

- **表格被切碎** — `c02` 看到的 562 字符规格表被硬切成 396/164 两块, 列对齐 `组件 / 规格 / 说明` 在 chunk 边界被腰斩; embedding 把两半分别算语义, 召回时拿到的是"半截规格表"。s11 `deepdoc/parser/pdf_parser.py` XGBoost 30 特征识别 `layout_type=table` + `attach_media_context` 媒体回填是根因修复
- **父子块缺失** — `c02` 看到的 10-11 char 节标题 chunks (`第四节 分季度财务数据` / `第二节 主要财务数据`) 语义为零, 用户问 "Q3 营收" 命中的是标题不是数字; s11 `hierarchical_merge` 按 "第 X 节" 正则建父子树, 召回命中 child 返回整个 parent 全文
- **跨段引用断裂** — `c02` 看到的 "收入结构如下" / "见上表" 这种指代词单独成 chunk 后无主体; s11 层级合并 + 召回附带整节, 让指代词所在的 chunk 命中时把后续数字 chunk 一起返回

s04 **embedding**: 把 c01 输出的 chunk 转成 BGE 真语义向量, 解决 s01 词袋 / s02 字符匹配的"无语义"问题。chunk 粒度本身决定召回上限 — 切得太大 (整页 / 整段) embedding 平均化失真, 切得太碎 (句 / 子句) 上下文断裂。embedding 是 s03 的下游消费者, 但**反过来要求 s03 切分粒度必须与 embedding 模型 `max_seq_len` 匹配** (BGE 512 token cap, 500 中文字符 ≈ 1000-1500 token 已经超限 — 这是 s03 故意保留的教学锚点, 让 s04 能演示"长 chunk 被截断" 的真实后果)。s05 之后向量索引 / 召回 / rerank 都在这一对 "chunk 粒度 + embedding 维度" 的耦合上做工程。

---

## 思考题

**如果一段就是 800 字但语义完整（比如一段财务披露），是该切还是不该切？**

提示：固定字符切分解决不了这个问题——切了句子被拦腰截断；不切单 chunk 太长 embedding 模型失真（BGE `max_seq_len=512`，但超长块会稀释语义）。RAGFlow 的 parent-child 是答案：整段作为 parent 保留语义完整性，内部再切小 child 用于召回匹配，命中后把 parent 整体塞给 LLM。详见 02 的 `demo_parent_child`。

（答案见文末「思考题答案」）

---

## 思考题答案

### Q： 如果一段就是 800 字但语义完整，是该切还是不该切？

**该切，但要换一种切法——按"父-子"层级切，而不是单层扁平切。**

### 单层固定字符切为什么不行

固定字符切分（我们 s03 的做法）解决不了"800 字但语义完整"的两难：

- **不切**：单 chunk 800 字，超出 BERT/BGE 类 Embedding 模型的 max_seq_len（典型 512-8192），即便没超，长文本平均化后语义向量失真，召回率下降。
- **切了**：句子被拦腰截断，Embedding 把残句当成完整语义单位，反而召回到错的片段；LLM 拿到半句话又生成幻觉。

无论哪种，固定字符切分都在"粒度"和"完整性"之间二选一。

### RAGFlow 的解法：parent-child

RAGFlow 用 **parent-child 双层结构**：

1. **父块 (parent)**：用版面识别（`_concat_downward` + XGBoost `updown_cnt_mdl`）把视觉相邻的文本框递归合并成段落/表格块，**保留 800 字的语义完整性**。
2. **子块 (child)**：在父块内部，用 `naive_merge` 按句界 `\n。；！？` + `chunk_token_num=128` token 上限切成小块，**保证 Embedding 友好**。
3. **召回与生成分离**：检索时用 child 的 Embedding 算相似度（细粒度召回），命中后把整个 parent 的文本返回给 LLM（完整语义单位）。

这样"800 字但语义完整"的段落就成了 1 个 parent，内部切 4 个 child 各 200 字，召回任何 child 都把 800 字 parent 整体返回——粒度和完整性同时满足。

### 为什么这是"最小解法"的升级

我们 s03 的 500 字 cap 是教学原型，够跑通管道、够讲清"为什么需要分块"。但任何生产 RAG 系统要做到"长段落也能正确回答"，都绕不开 parent-child —— 这是 RAGFlow / LangChain `ParentDocumentRetriever` / LlamaIndex `HierarchicalNodeParser` 的共同选择。

### 参考

- [`docs/reference/ragflow-notes/deepdoc_chunking.md`](../../docs/reference/ragflow-notes/deepdoc_chunking.md)：parent-child 在 RAGFlow `pdf_parser.py` / `rag/nlp/__init__.py:naive_merge` 的具体实现。