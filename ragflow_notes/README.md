# ragflow_notes/ — RAGFlow 源码摘录

## 用途

本目录**抄录 [RAGFlow](https://github.com/infiniflow/ragflow) 项目中的关键源码片段**，用于：

1. 每章 README 的第 5 节会引用这里的具体文件 / 行号；
2. 在自写 MVP 时提供"工业级对照"——为什么 RAGFlow 这么写、它解决了什么工程问题；
3. 给想深入 RAG 实现的读者提供一份带注释的阅读路径。

> **不是教程主线**。`code.py` 完全独立可跑；本目录是阅读材料，不是依赖。

## 引用规范

每个摘录文件遵循以下格式：

```markdown
# <RAGFlow 文件相对路径>

## 来源
- 仓库：https://github.com/infiniflow/ragflow
- 文件：`<path>`
- 行号：L<start>-L<end>
- commit: <RAGFlow commit hash>（pin 到的版本，见下方"同步状态"）
- 引用日期：YYYY-MM-DD

## 代码
\`\`\`python
<抄录的源码>
\`\`\`

## 它为什么这样写
- 关键设计点 1
- 关键设计点 2
- 与本章 MVP 的对比
```

**约定**：
- 摘录必须保留**原始缩进和注释**，不要"美化"。
- 每段摘录**聚焦一个主题**（一个函数、一个类、一段算法），不要大段抄。
- 必须在摘录顶部注明 **RAGFlow commit hash**，否则引用失去意义。

## 同步状态

## 当前 Pin

- commit: `828c5789f651d4c4ebe4645190b8b8d244144fe0`
- 日期: 2026-07-01
- 主题: `fix(agent/tools): GoogleScholar empty json output and ignored top_n (#16419)`
- 仓库: https://github.com/infiniflow/ragflow

> 上一版 pin：**TBD**（Task 2 完成后填入）。

抄录前请：

```bash
cd <ragflow-clone-dir>
git rev-parse HEAD        # 记录 commit
git log -1 --format='%ci' # 记录日期
```

写进对应摘录文件的"来源"区块。

## 免责声明

RAGFlow 仍在活跃演进。本目录的摘录**可能随时间过时**——代码路径、函数签名、内部 API 都会变。

引用本目录内容时，请：

1. **始终附上 pin 到的 RAGFlow commit + 引用日期**；
2. 与原仓库对比验证（特别是间隔 6 个月以上的引用）；
3. 以"RAGFlow 在该版本是这样实现的"为前提使用，不要把它写成永恒不变的规范。

## 目录布局建议

每章对应一个子目录或一组命名前缀，例如：

```
ragflow_notes/
├── README.md
├── s01_what_is_rag/
├── s02_document_loading/
│   ├── parser_pdf.md
│   └── parser_docx.md
├── ...
```

具体子目录由各章节 Task 创建。Task 0 仅创建本 README。
