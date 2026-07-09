# s06 / Unit 01 — BM25 词法召回 (hand-written BM25 + 中英分词)

> 由浅入深第 1 步：在内存里对 chunk 集合跑一遍 BM25，拿到"字面命中"分。  
> unit 02 会把这里的 BM25 分和 dense cosine 一起做加权融合，进入混合检索。

## 这是什么

`code.py` 实现一个 Robertson BM25 召回器：拿到 `docs` 后离线算 df/tf/IDF，query 来时按词项打分、按 BM25 分降序取 top-k。

- `tokenize(text)` —— 中文按 1-2 字滑动窗口 + 英文/数字单词拆分（lowercase），纯中文段落也有 df 命中；
- `BM25(docs, k1=1.5, b=0.75)` —— 构造期一次性算 `df` / `tf` / `avgdl`，查询期只跑 score 公式的 `tf` 部分；
- `score(query)` —— 对每个 doc 算 BM25 累加分（`IDF * tf * (k1+1) / (tf + k1*(1 - b + b*dl/avgdl))`）；
- `bm25_topk(docs, query, k)` —— 按 BM25 分排序，取分 > 0 的前 k 条；
- `main()` —— 内联 pypdf + python-docx + 500 字符 cap 句界切，跟 s02 unit 01 / s03 同一套"加载器复刻"取舍，跑固定 query `"应收账款 计提"` 打印 BM25 top-5。

## 跑起来

```bash
python s06_retrieval/units/01_bm25/code.py
```

输出示例：

```
loaded 34 chunks from samples/
query='应收账款 计提' → BM25 top-5:
  [disclosure.docx#-] bm25=... | ...
```

chunk 数与 s05 unit 01 一致（34），但走的不是向量空间，是字面 token 命中。

## 它做对了什么

- **中英混排分词**：英文走 `[a-z0-9]+`、中文走 1-2 字滑动窗口，纯中文段落也能命中 df；这对中文财报 / 招股书非常关键（没有空格分词的语料不能简单按空格切）；
- **BM25 的两个超参可配**：`k1`（TF 饱和点，默认 1.5）控制"同一词出现 100 次 vs 10 次"的增益差距；`b`（长度归一，默认 0.75）控制"长文档天然占优"的修正幅度——b=0 等于不归一，b=1 是完全归一；
- **便宜**：一次构造、多次查询，构造期复杂度 O(N * L)，查询期 O(|q| * N)；chunk 数在几千级别完全无感，几十万量级要换倒排索引；
- **可解释**：每个命中分都能反推到"哪几个 query token 命中了哪几段 tf"——出了 bad case 不用黑盒调试。

## 它做错了什么

- **没语义匹配**：`"内存"`和 `"RAM"`、`"营收"`和 `"营业收入"`在 BM25 下是 0 分；改写 / 同义 / 跨语种召回是 BM25 的死穴——这是 unit 02 必须叠 dense cosine 的根本原因；
- **中文分词太 naive**：1-2 字滑动窗口会产生大量噪声 token（`"应收"`、`"账款"`、`"应"`、`"收"`），df 散在海量 1-字 token 上 → 真正有判别力的 2-字词被稀释；生产应该走 `jieba` 之类带词典的 tokenizer；
- **没字段加权 / 没 BM25F**：所有 token 同权；title / heading / body 的 boost 要手工实现，RAGFlow 把这部分挪到了 `rank_feature` 第三层信号；
- **构造期没有倒排索引**：每次查询还是要扫所有 doc 的 tf 才能算分，chunk 数大时会成瓶颈；生产里 `bm25s` / `rank_bm25` / `pyserini` 都建倒排表，查询 O(|q| * avg_postings) 而不是 O(|q| * N)；
- **没有 query expansion / 同义词扩展**：`"AI"` 查不到 `"人工智能"`；生产里靠 `qryr`（ragflow 的 query analyzer）做改写、拼写纠正、term drop。

## 对照 ragflow 怎么做的

RAGFlow 在 `Dealer.search` 阶段把全文召回和向量召回一起送给 ES / Infinity，由底层 `FusionExpr("weighted_sum", ...)` 合并；BM25 全文打分是 ES / Infinity 自带能力（Lucene BM25），调用方不直接持有 BM25 实例。RAGFlow 的 BM25 不依赖手写——ES `matchText` / Infinity 的全文字段都是工业级实现（倒排索引、字段加权、query expansion 全包）。本单元 `BM25` 类只是一个**教学版**：让你看清 IDF + TF 饱和 + 长度归一三项是怎么落地的，production 不应该手写。

参考：[`docs/reference/ragflow-notes/hybrid_retrieval.md`](../../../../docs/reference/ragflow-notes/hybrid_retrieval.md)

## 思考题

**为什么 query `"应收"` 能命中含 `"应收账款"` 的 chunk，但 query `"应收账款"` 也能命中含 `"应收"` 的 chunk？两个 query 命中的 chunk 是同一批吗？**

提示：1-2 字滑动窗口让两个 query 都有 `应`、`收`、`应收`、`应收账`、`账款` 之类公共 token；具体命中的 doc 集合 + 分的高低取决于 doc 里这些 token 的 tf 和 dl。试着把 `tokenize` 改成只保留 2 字切分（不要 1 字），再跑一遍 `bm25_topk`，对比两组 top-5。