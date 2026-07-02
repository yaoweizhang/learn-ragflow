# RAGFlow 怎么做: 文本分块 (parent-child + 句界切分)

## 来源
- 仓库: https://github.com/infiniflow/ragflow
- 文件: `deepdoc/parser/pdf_parser.py`
- 行号: L1050-L1069
- commit: `828c5789f651d4c4ebe4645190b8b8d244144fe0`
- 引用日期: 2026-07-02
- GitHub 链接: https://github.com/infiniflow/ragflow/blob/828c5789f651d4c4ebe4645190b8b8d244144fe0/deepdoc/parser/pdf_parser.py#L1050-L1069

## 代码

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
            down = boxes[i]
            if not concat_between_pages and down["page_number"] > up["page_number"]:
                break
            # …XGBoost 模型判断是否纵向拼接
```

(注: 完整递归 `_concat_downward` 把识别出的文本框按视觉相邻性 + ML 模型
`updown_cnt_mdl` 决定是否合并,生成 `blocks` 即"父块"。同文件后续的
`rag/nlp/__init__.py:naive_merge` 再按 `chunk_token_num=128` + 中英句界
`\n。；！？` 把每个父块切小——这是"子块"。)

## 它为什么这样写

- **先用版面识别做"父块",再用句界做"子块"**。`_concat_downward` 的 `dfs`
  把视觉相邻的文本框(同列、垂直距离 < 4 倍行高、跨页距离 < 16 倍行高)
  递归合并成 `chunks`,再装进 `blocks`——这就是 parent。把每个 parent
  喂给 `naive_merge`,它按 `\n。；！？` 把段落拆到 ≤ 128 token 的子块——
  这就是 child。检索时用 child 的 embedding 算相似度,返回时把整个
  parent 的文本给 LLM。这样既保住了细粒度召回,又让生成端看到完整语义
  单位(避免表格/列表被拦腰切断)。
- **视觉特征驱动合并,不是固定规则**。第 1097 行的
  `self.updown_cnt_mdl.predict(...)` 用 XGBoost 模型判断上下两行是否
  属于同一段——特征包括行距、字号差、标点结尾、是否数字、是否同一列等
  (见 `_updown_concat_features` 的 30 个布尔/数值特征)。同一个 PDF 页
  上 "1. 引言" 和 "1.1 背景" 视觉上挨着,模型能区分这是两条标题不是
  一段长句。这种"视觉+ML"组合,纯文本 splitter 永远做不到。
- **跟我们 500 字符 cap 的差异**。我们的 `code.py` 走的是"如果段落超过
  500 字符就按句子切",问题是: (1) 段落边界靠 `pypdf.extract_text()` 的
  `\n` 切,扫描件或表格抽出来没换行就一坨; (2) 句子边界用正则,对
  "前/后置面板 24LFF (24*SATA/SAS)" 这种规格串无标点,会原样输出超长
  chunk(必须硬切兜底); (3) 完全没考虑"父块"——单个 500 字 chunk 召回
  到后,LLM 看不到上下文段。RAGFlow 把这三件事都做了:版面识别出父块 →
  token-aware splitter 出子块 → 召回时返回父块文本。这也是 README 真
  实问题里"表格整体性丢失、跨段落引用断裂"的工程答案。
