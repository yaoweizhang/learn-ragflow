# RAGFlow 怎么做: 文本分块 (parent-child + 版面 + token-aware + 上下文回填)

## 来源
- 仓库: https://github.com/infiniflow/ragflow
- 文件: `deepdoc/parser/pdf_parser.py`、`rag/nlp/__init__.py`
- commit: `828c5789f651d4c4ebe4645190b8b8d244144fe0`
- 引用日期: 2026-07-02

RAGFlow 的分块是 **4 层流水线**，不是"段落 → 句子"两步就完事：

```
PDF 视觉识别 → 父块 (parent blocks) ─┬→ token-aware 子块 (child chunks)
                                      ├→ hierarchical merge (Markdown / 编号标题)
                                      └→ attach_media_context (表格/图片回填上下文)
```

下面 4 个 snippet 一块对应一层。MVP 把 4 层压成"500 字符 cap + 句界"一层，本文逐层展开。

## 1. 父块：XGBoost 驱动的版面合并 `_concat_downward`

[pdf_parser.py](https://github.com/infiniflow/ragflow/blob/828c5789/deepdoc/parser/pdf_parser.py)

```python
# concat between rows
boxes = deepcopy(self.boxes)
blocks = []
while boxes:
    chunks = []

    def dfs(up, dp):
        chunks.append(up)
        i = dp
        while i < min(dp + 12, len(boxes)):
            ydis = self._y_dis(up, boxes[i])
            smpg = up["page_number"] == boxes[i]["page_number"]
            mh = self.mean_height[up["page_number"] - 1]
            mw = self.mean_width[up["page_number"] - 1]
            if smpg and ydis > mh * 4:
                break
            if not smpg and ydis > mh * 16:
                break
            ...
            if up["x1"] < down["x0"] - 10 * mw or up["x0"] > down["x1"] + 10 * mw:
                i += 1
                continue

            fea = self._updown_concat_features(up, down)
            if self.updown_cnt_mdl.predict(xgb.DMatrix([fea]))[0] <= 0.5:
                i += 1
                continue
            dfs(down, i + 1)
            boxes.pop(i)
            return

    dfs(boxes[0], 1)
    blocks.append(chunks)
```

**关键点**：视觉相邻只是"候选"，最终合并交给 XGBoost（`updown_cnt_mdl.predict(...) <= 0.5` 则跳过）。

## 2. 30 个特征：`_updown_concat_features`

[pdf_parser.py](https://github.com/infiniflow/ragflow/blob/828c5789/deepdoc/parser/pdf_parser.py)

```python
def _updown_concat_features(self, up, down):
    w = max(self.__char_width(up), self.__char_width(down))
    h = max(self.__height(up), self.__height(down))
    y_dis = self._y_dis(up, down)
    ...
    fea = [
        up.get("R", -1) == down.get("R", -1),          # 同一阅读序
        y_dis / h,                                       # 归一化行距
        down["page_number"] - up["page_number"],         # 跨页
        up["layout_type"] == down["layout_type"],        # 同区域类型
        up["layout_type"] == "text",
        down["layout_type"] == "text",
        up["layout_type"] == "table",
        down["layout_type"] == "table",
        # 句末标点（。！？；!?;+）、半句结束（，：'、0-9+-）
        True if re.search(r"([。？！；!?;+)）]|[a-z]\.)$", up["text"]) else False,
        True if re.search(r"[，：‘“、0-9（+-]$", up["text"]) else False,
        # 行首起头（标点 / 中文引号 / 数字 / 字母）
        True if re.search(r"(^.?[/,?;:\]，。；：’”？！》】）-])", down["text"]) else False,
        ...
        # 中英括号配对、跨行逗号未完句、xx/yy 法律引用模式、英文首字母、数字 / 比例 / 百分号结尾
        self._match_proj(down),
        True if re.match(r"[A-Z]", down["text"]) else False,
        True if re.match(r"[0-9.%,-]+$", down["text"]) else False,
    ]
    return fea
```

一共 **30 个布尔 / 数值特征**：阅读序、行距、跨页、版面类型（text/table/figure…）、标点首尾、中英括号是否配对、是否数字 / 比例结尾、大小写开头…… 训练一次后，整页 PDF 的"哪两个 text-box 该合并"不再靠硬编码规则，而是 30 维特征 + 简单 XGBoost 决策树。

`updown_concat_xgb.model` 在模型目录里随仓库发布（`deepdoc/parser/resnet/` 下）；`__init__.py` 里：

```python
self.updown_cnt_mdl = xgb.Booster()
self.updown_cnt_mdl.set_param({"device": "cpu"})
self.updown_cnt_mdl.load_model(os.path.join(model_dir, "updown_concat_xgb.model"))
```

## 3. 子块：token-aware + 可配重叠 `naive_merge`

[nlp/__init__.py](https://github.com/infiniflow/ragflow/blob/828c5789/rag/nlp/__init__.py)

```python
def naive_merge(sections: str | list, chunk_token_num=128,
                delimiter="\n。；！？", overlapped_percent=0):
    ...
    # Ensure that the length of the merged chunk does not exceed chunk_token_num
    if cks[-1] == "" or tk_nums[-1] > chunk_token_num * (100 - overlapped_percent) / 100.0:
        ...
        # Recount with the overlap prefix included, else chunks overshoot chunk_token_num.
        t = overlapped[int(len(overlapped) * (100 - overlapped_percent) / 100.0):] + t
        tk_nums[-1] = num_tokens_from_string(cks[-1] + t)
    ...
    # Custom delimiters ignore chunk_token_num: each segment is its own chunk.
    if not dels or num_tokens_from_string(sec) < chunk_token_num:
        ...
```

`chunk_token_num=128`（不是字符）。token 数用 `num_tokens_from_string(...)`（tiktoken `cl100k_base`）算——这是 BGE / GPT 这类模型的真实额度单位，不是字符。`overlapped_percent=0` 关掉重叠；>0 时从上一个 chunk 切 `(100-percent)/100` 处当作 prefix 拼到下一个 chunk，并**重算 token 数**，否则容易超 cap。

我们 MVP 是 `max_chars=500`，500 个中文字符 ≈ 1000-1500 个 token，BGE `max_seq_len=512` 直接爆掉。生产里分块必须用 token，不要用字符。

## 4. 层级合并 `hierarchical_merge`

[nlp/__init__.py](https://github.com/infiniflow/ragflow/blob/828c5789/rag/nlp/__init__.py)

```python
def hierarchical_merge(bull, sections, depth):
    if not sections or bull < 0:
        return []
    ...
    bullets_size = len(BULLET_PATTERN[bull])
    levels = [[] for _ in range(bullets_size + 2)]

    for i, (txt, layout) in enumerate(sections):
        for j, p in enumerate(BULLET_PATTERN[bull]):
            if re.match(p, txt.strip()):
                levels[j].append(i)
                break
        else:
            if re.search(r"(title|head)", layout) and not not_title(txt):
                levels[bullets_size].append(i)
            else:
                levels[bullets_size + 1].append(i)
    ...
```

`BULLET_PATTERN` 是一组按"标题级别"排序的正则：`"一、" → "1." → "1.1" → "1.1.1" → "- "` 这种有序匹配。函数把所有段落**按头部正则**分桶到对应级别，输出层级结构（`depth` 控制向下挖几层），让"召回到子段时把整节一起给 LLM"成为可能——这正是子句-父句回填的引擎。

我们 s03 的 `chunk_by_paragraph` 完全是平铺：把所有段落平级切开。生产里如果你做 RAG 排版整齐的财报或白皮书，层级结构对召回准度提升明显。

## 5. 媒体上下文回填 `attach_media_context`

[nlp/__init__.py](https://github.com/infiniflow/ragflow/blob/828c5789/rag/nlp/__init__.py)

```python
def attach_media_context(chunks, table_context_size=0, image_context_size=0):
    """
    Attach surrounding text chunk content to media chunks (table/image).
    Best-effort ordering: if positional info exists on any chunk, use it to
    order chunks before collecting context; otherwise keep original order.
    """
    from . import rag_tokenizer

    if not chunks or (table_context_size <= 0 and image_context_size <= 0):
        return chunks

    def is_image_chunk(ck):
        if ck.get("doc_type_kwd") == "image":
            return True
        ...
        return bool(ck.get("image")) and not has_text

    def is_table_chunk(ck):
        return ck.get("doc_type_kwd") == "table"

    def is_text_chunk(ck):
        return not is_image_chunk(ck) and not is_table_chunk(ck)
    ...
```

**职责**：表格 / 图片 chunk 自身没语义（光有数字或图），需要把"它前后若干句"的纯文本 chunk 拿来当上下文拼回去。函数先按 `position_int`（PDF 坐标）稳定排序，再按 `table_context_size` / `image_context_size`（默认 token 预算）回填。`return_context=True` 时只返回 context 段、不返回 media 本身，作为纯文本增强源。

MVP 完全没做这一步。后果：召回到一个孤立表格 cell，LLM 看到的 context 啥也不是。

## 它为什么这样写

- **视觉识别 → 父块 → token-aware 子块 → 上下文回填** 是一条收敛流水线，每层只解一类问题：
  - 版面识别的输出是"哪两段属于同一段"——特征驱动，不用写规则；
  - `naive_merge` 的输出是"≤128 token"的检索单位——token 精准对齐 BGE；
  - `hierarchical_merge` 的输出是"段落树"——召回时能放大到节；
  - `attach_media_context` 的输出是"表格 / 图片的可读段落"——LLM 看得懂表格。

- **为什么"字符 cap"不够**：BGE `max_seq_len=512` token，500 中文字符 ≈ 1000-1500 token → 截断后存进索引，embedding 质量塌。RAGFlow 用 tiktoken 而不是字符数算预算，是因为 token 才是模型真实额度。

- **视觉特征不是"奢侈"**。"1. 引言" 和 "1.1 背景" 两个标题在 PDF 上视觉距离 0，特征（"上一行末尾无句号"、"下一行首字母大写"、"版面类型不是 text"）让 XGBoost 自学出"这是两条独立标题"。手写规则写不完——这是 XGBoost 在 RAGFlow 唯一承担的角色（不是 embedding、不是 rerank，纯版面"该不该合"）。

## 跟 MVP 的核心差异

| 维度 | MVP `chunk_by_paragraph` | RAGFlow |
|---|---|---|
| 边界检测 | 字符 + 句界正则 | 版面 + XGBoost 30 特征 |
| 切分单位 | 500 字符（≈1000-1500 token） | 128 token（tiktoken cl100k_base）|
| 重叠 | 无 | `overlapped_percent` 配比，按 token 重算 |
| 层级 | 平铺 | `hierarchical_merge` 按编号标题分桶 |
| 表格 / 图片 | 跟普通段落同等对待 | `attach_media_context` 回填 token 预算的上下文 |
| 跨页 / 跨列 | 不感知 | 4 倍行距 / 16 倍跨页阈值 + XGBoost 兜底 |
