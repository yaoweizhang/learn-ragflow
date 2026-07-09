# 思考题答案

## 问题: 如果一段就是 800 字但语义完整,是该切还是不该切?

**该切,但要换一种切法——按"父-子"层级切,而不是单层扁平切。**

### 单层固定字符切为什么不行
固定字符切分(我们 s03 的做法)解决不了"800 字但语义完整"的两难:

- **不切**: 单 chunk 800 字,超出 BERT/BGE 类 Embedding 模型的 max_seq_len
  (典型 512-8192),即便没超,长文本平均化后语义向量失真,召回率下降。
- **切了**: 句子被拦腰截断,Embedding 把残句当成完整语义单位,反而召回
  到错的片段;LLM 拿到半句话又生成幻觉。

无论哪种,固定字符切分都在"粒度"和"完整性"之间二选一。

### RAGFlow 的解法: parent-child
RAGFlow 用 **parent-child 双层结构**:

1. **父块 (parent)**: 用版面识别(`_concat_downward` + XGBoost `updown_cnt_mdl`)
   把视觉相邻的文本框递归合并成段落/表格块,**保留 800 字的语义完整
   性**。
2. **子块 (child)**: 在父块内部,用 `naive_merge` 按句界 `\n。；！？` +
   `chunk_token_num=128` token 上限切成小块,**保证 Embedding 友好**。
3. **召回与生成分离**: 检索时用 child 的 Embedding 算相似度(细粒度召
   回),命中后把整个 parent 的文本返回给 LLM(完整语义单位)。

这样"800 字但语义完整"的段落就成了 1 个 parent,内部切 4 个 child
各 200 字,召回任何 child 都把 800 字 parent 整体返回——粒度和完整性
同时满足。

### 为什么这是"最小解法"的升级
我们 s03 的 500 字 cap 是教学原型,够跑通管道、够讲清"为什么需要分块"。
但任何生产 RAG 系统要做到"长段落也能正确回答",都绕不开 parent-child
——这是 RAGFlow / LangChain `ParentDocumentRetriever` / LlamaIndex
`HierarchicalNodeParser` 的共同选择。

### 参考
- `../docs/reference/ragflow-notes/deepdoc_chunking.md`: parent-child 在 RAGFlow
  `pdf_parser.py` / `rag/nlp/__init__.py:naive_merge` 的具体实现。
