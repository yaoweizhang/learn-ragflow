# s03 文本分块

## 问题
固定 500 字符切有两个典型失败案例:
1. **句子被切断**。"今天我们学习 RAG。它包括检索、生成两部分。" →
   500 字符 cap 直接砍到中间,得到 "今天我们学习 RAG。它包括检索、生
   成两部",Embedding 把残句算成完整语义。
2. **表格被切断**。规格表 24LFF / 12LFF / 4SFF 横向排在一行,pypdf 抽
   出来是没换行的长串,500 字符 cap 在表格中间切断,每行 chunk 只看
   到一半字段。

## 最小解法
`code.py` 实现 `chunk_by_paragraph(docs, max_chars=500)`:
- 短段落(< 500 字符)直接整体保留为 1 个 chunk;
- 长段落先按中英句界 `(.。!?！？)` 切成句子,再贪心装桶,保证每 chunk
  长度 ≤ max_chars;极长无标点串(规格表)按字符硬切兜底;
- 每个 chunk 带 `chunk_id = {source}#{page}#p{n}`,供 s04+ 引用。

```bash
cd D:/study/rag_study/learn-ragflow
python s03_chunking/code.py
```

## 跑起来
输入 31 段（4 页 PDF + 27 段 DOCX）→ 输出 34 块,首 3 个 chunk:
```
server_whitepaper.pdf#1#p0 | 紫光恒越 R3630 G5 双路机架式服
务器
产品白皮书  ·  v1.0  ·  仅用于 RAG 教程测试
一、产品概述
紫光恒越 R3630 G5 是面向 ...
server_whitepaper.pdf#1#p1 | 二、关键特性
计算密度：单台 2U 机箱内集成两颗处理器、32 条内存 DIMM 与 10 个 PCIe 4.0 槽位 ...
server_whitepaper.pdf#2#p2 | 三、整机规格
组件 规格 说明
处理器 2 × 第三代 Intel Xeon 可
扩展处理器 ...
```
所有 chunk ≤ 500 字符,无非空块。

## 真实世界的问题
1. **表格整体性丢失**。我们用 `pypdf.extract_text()` 抽出的表格没有格线
   信息,所有列挤在一行里,切完后 LLM 不知道哪两列是同一行。
2. **父子块概念缺失**。我们的 chunk 是 500 字封顶的扁平列表;但 RAG
   召回后,LLM 需要上下文段才能正确回答——比如"Q3 营收多少"需要看到
   整段财务描述,而不是被切碎后的 200 字片段。
3. **跨段落引用断裂**。"见上表" "如表 3 所示" 这种指代词单独成 chunk
   后,检索召回的是指代词本身,而不是表 3 的实际内容。

## RAGFlow 怎么做的
详见 `../ragflow_notes/deepdoc_chunking.md`。一句话总结: **版面识别出父
块 (parent),按 token-aware + 句界切出子块 (child),召回时返回父块文
本给 LLM**。这样细粒度召回和完整语义单位两不误。

## 思考题
**如果一段就是 800 字但语义完整,是该切还是不该切?**

答: 固定字符切分解决不了这个问题。切,句子被拦腰截断;不切,单 chunk
太长 Embedding 模型失真 (BERT 类模型 max_seq_len=512,BGE 类 ~512-8192
但超长块仍会稀释语义)。RAGFlow 的 parent-child 是答案:整段作为 parent
保留语义完整性,内部再切小 child 用于召回匹配,命中后把 parent 整体
塞给 LLM。
