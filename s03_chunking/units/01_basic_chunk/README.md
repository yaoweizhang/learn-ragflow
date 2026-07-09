# s03 / Unit 01 — 最小可跑分块：500 字符 cap + 句界切

> 由浅入深第 1 步：把 s02 输出的 `list[{text, page, source}]` 切成可灌进 embedding 的小块，契约是 `chunk_id + ≤ max_chars 文本`。  
> unit 02 会跑同一套函数到真实样本上，展示哪些情况它会崩。

## 这是什么

1. `split_long_paragraph(text, max_chars)` — lookbehind 正则在 `[.。!?！？]` 之后切，把超长段落切成 ≤ max_chars 的若干块；极端情况（无标点的规格表）按字符硬切兜底；
2. `chunk_by_paragraph(docs, max_chars=500)` — 短段整段保留为 1 块，长段调 `split_long_paragraph` 后展开多块；每块带 `chunk_id = {source}#{page}#p{n}`；
3. **复用 s02 unit 01 的 `load_pdf` / `load_docx`**——loader 是上游契约，本单元不重复实现。

## 跑起来

```bash
python s03_chunking/units/01_basic_chunk/code.py
```

输出：

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

## 它做对了什么

- **token-aware 边界**：按 `[.。!?！？]` 切而不是裸字符切，中英文段落都不会拦腰砍断句子；
- **硬切兜底**：单句本身超过 `max_chars`（无标点规格表）按字符切，最坏情况下输出不超过 `2 * max_chars`；
- **chunk_id 稳定可引用**：`{source}#{page}#p{n}` 形式让 s04+ 可以直接引用具体 chunk；
- **零新依赖**：只用 `re` + `pathlib`。

## 它做错了什么

- **表格被切碎**：`pypdf.extract_text()` 输出的表格是挤在一行的长串，句界切不到、只能硬切 → 每个 chunk 只看到半张表；
- **父子块概念缺失**：500 字封顶的扁平列表，召回后 LLM 拿不到"完整语义单位"，回答"Q3 营收多少"只能看到切碎的片段；
- **跨段落引用断裂**："见上表""如表 3 所示" 这种指代词单独成 chunk 后，检索召回的是指代词本身、不是它指向的实体。

unit 02 会在 `samples/` 上把这三类失败各跑一遍。

## 对照 ragflow 怎么做的

RAGFlow 的分块是 **4 层流水线**，MVP 把 4 层压成"500 字符 cap + 句界"一层：

- **父块**：XGBoost 30 特征驱动 `_concat_downward`（`pdf_parser.py:1052`）——视觉相邻的 text-box 由 `_updown_concat_features` 算 30 维特征（行距、跨页、版面类型、标点首尾…），`updown_cnt_mdl.predict(...) <= 0.5` 则跳过合并；
- **子块**：`naive_merge`（`rag/nlp/__init__.py:1155`）按 **tiktoken token 数**算 cap（默认 128），不是字符——BGE `max_seq_len=512`，500 中文字符 ≈ 1000-1500 token，MVP 直接爆掉；
- **层级**：`hierarchical_merge`（`nlp/__init__.py:1066`）按 `BULLET_PATTERN`（"一、" → "1." → "1.1" → "1.1.1"）分桶到编号标题级别，召回时能放大到节；
- **媒体回填**：`attach_media_context`（`nlp/__init__.py:497`）把表格/图片前后若干句当 context 拼回去——MVP 完全没做这一步。

参考：[`docs/reference/ragflow-notes/deepdoc_chunking.md`](../../../../docs/reference/ragflow-notes/deepdoc_chunking.md)

## 思考题

**如果一段就是 800 字但语义完整（比如一段财务披露），是该切还是不该切？**

提示：固定字符切分解决不了这个问题——切了句子被拦腰截断；不切单 chunk 太长 embedding 模型失真（BGE `max_seq_len=512`，但超长块会稀释语义）。RAGFlow 的 parent-child 是答案：整段作为 parent 保留语义完整性，内部再切小 child 用于召回匹配，命中后把 parent 整体塞给 LLM。详见 unit 02 的 `demo_parent_child`。