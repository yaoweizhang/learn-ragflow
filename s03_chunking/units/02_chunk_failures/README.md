# s03 / Unit 02 — chunker 在真实样本上的三类失败

> 由浅入深第 2 步：unit 01 在 toy 上能跑；放到真实 `samples/` 上会崩在哪？  
> 本单元定位 3 类问题 + 引出 ragflow 的工业解法。

## 这是什么

把 unit 01 的 `chunk_by_paragraph` 喂给真实样本（`server_whitepaper.pdf` 4 页 + `disclosure.docx` 27 段），把"看不见的损失"暴露出来：

1. **表格被切碎**——`pypdf.extract_text()` 输出的规格表挤在一行里，句界切不到、只能字符硬切，每个 chunk 只看到半张表；
2. **父子块缺失**——节标题（"第四节 分季度财务数据"）单独成 10-char chunk，用户问"Q3 营收"时检索命中的是标题，数字本身在 DOCX tables 里（被 s02 loader 丢了）；
3. **跨段引用断裂**——"收入结构如下""见表 3"这种指代词单独成 chunk 后无主语，召回的是指代词本身而不是它指向的实体。

## 跑起来

```bash
python s03_chunking/units/02_chunk_failures/code.py
```

输出片段：

```
================================================================
[a] 表格被切碎 — 整机规格表 hard-cut
================================================================
BEFORE (整段 562 字符):
三、整机规格
组件 规格 说明
处理器 2 × 第三代 Intel Xeon 可
扩展处理器
...
AFTER  (cap=500 → 切成 2 块):
  chunk 0 len= 396 | 末行: NAND
  chunk 1 len= 164 | 末行: ...
→ 失败点: 第 0 块末尾停在 '8GB' (BMC 那行的中间),
          第 1 块从 'NAND' 开头继续,失去 '组件 / 规格 / 说明' 列对齐语义。
```

```
================================================================
[b] 父子块缺失 — 节标题 chunk 与数据本体分离
================================================================
BEFORE (用户问 'Q3 营收多少'):
  召回命中 chunk_id=disclosure.docx#None#p18 len=11 text='第四节 分季度财务数据'
AFTER  (季度数据本体在 DOCX tables,共 3 张表):
  (本样本 DOCX 表未含 Q3 字面量; 但所有季度数字都只在 tables 里,chunker 看不到)
```

```
================================================================
[c] 跨段引用断裂 — 指代词单独成 chunk
================================================================
BEFORE (过短的 header-only chunks,语义为零):
  id=disclosure.docx#None#p8  len=15 | '2024 年度财务信息披露报告'
  id=disclosure.docx#None#p13 len=10 | '第二节 主要财务数据'
  id=disclosure.docx#None#p18 len=11 | '第四节 分季度财务数据'
```

## 它做对了什么

- **暴露问题**：每个 demo 都打印 before/after 片段，让"为什么这种切法不够"肉眼可见；
- **解法对照**：每个失败都点名 ragflow 的对应模块（`_concat_downward` / `naive_merge` / `hierarchical_merge` / `attach_media_context`）；
- **量化损失**：表格切成 2 块时第 0 块停在 mid-row "8GB"——直接给 LLM 看它能不能拼回去；
- **零新依赖**：只 import unit 01 + 标准库 + `python-docx`（已装）。

## 它做错了什么

- 它是个 demo，不是 fix——没有真的把表拼回去、没有真的建父子树、没有真的回填 context；
- 依赖 s02 unit 01 的 loader（后者已丢 DOCX tables）；如果想看到表格里的季度数字，要么改 s02 loader 要么用 `python-docx` 直读；
- 只测了 3 类失败，真实场景还有 (d) 页眉页脚污染 / (e) 多栏错位 / (f) 扫描件 OCR 缺失，那些是 s02 + s11 的事。

## 对照 ragflow 怎么做的

`docs/reference/ragflow-notes/deepdoc_chunking.md` 描述了 4 层流水线，每一层修一类失败：

| 失败 | ragflow 修法 | 文件:行 |
|---|---|---|
| 表格切碎 | `_concat_downward`（XGBoost 30 特征）识别 `layout_type=table` 当 parent | `pdf_parser.py:1052` |
| 表格孤立 | `attach_media_context` 把表格前后文本当 context 回填 | `nlp/__init__.py:497` |
| 父子块缺失 | `hierarchical_merge` 按 `BULLET_PATTERN`（"一、" → "1." → "1.1"）建层级树 | `nlp/__init__.py:1066` |
| 字符 cap 超 BGE 限额 | `naive_merge` 用 tiktoken 算 128 token cap，对齐 BGE | `nlp/__init__.py:1155` |

参考：[`docs/reference/ragflow-notes/deepdoc_chunking.md`](../../../../docs/reference/ragflow-notes/deepdoc_chunking.md)

## 思考题

**如果只允许修 1 类失败，优先修哪类？为什么？**

提示：决策不是技术问题——是产品问题。如果你的语料以**白皮书 / 财报** 为主，表格切碎会让检索准度断崖下降（用户问"内存最大多大"命中半张表 → LLM 瞎编）；如果以**长文档 / 法规** 为主，父子块缺失让 LLM 永远只看片段；如果是**导航型** 文档（FAQ），跨段引用最多让用户体验不好但不影响答案正确性。你要结合自己的 samples 分布选。