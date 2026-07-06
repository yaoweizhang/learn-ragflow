# s05 向量索引 — 代码审计

**审计日期**: 2026-07-06
**审计范围**: `s05_vector_index/code.py`, `s05_vector_index/units/01_chroma_build/code.py`, `s05_vector_index/units/02_chroma_query/code.py`
**审计基线**: 大纲 / spec / `ragflow_notes/vector_indexing.md` / `units/*/README.md`
**审计目的**: 验证 README 4-段式叙述的"核心函数 / schema / 工具速览"和实际代码的对照情况

---

## 审计结果 (4 标准)

| # | 标准 | 等级 | 备注 |
|---|---|---|---|
| 1 | README 提过的函数都在代码里 | 对齐 | `get_chunks_and_vectors / build_index / search / _open_collection / main (x2) / code.py importlib 聚合入口` — README §3.3 表 6 行全覆盖,无虚构 |
| 2 | 代码里 main / 关键函数在 README 里讲清楚 | 对齐 | README §3.3 一句话描述每个函数的入参 / 出参 / 用途,跟代码 docstring 一致;`build_index / search` 的 `1 - distance → score` 翻分在 README §1.3 + §3.4 都说了 |
| 3 | README 的 sample output 跟实际跑出来一致 | 对齐 (有 known 文档偏差) | unit 01 跑出 `indexed 34 chunks into _chroma/ (collection=docs, dim=512)`(代码 line 142),README §3.7 写的是同一句 — 完全对齐;unit 02 跑出 `score=0.499 / 0.487 / 0.449`(实际),`unit 02/README.md` 写的是 `0.950 / 0.926 / 0.850`(预估值),**已知偏差**(详见 ledger,源自 s04 §2.1 提的 cosine 归一化实际数值 ≠ 字面"相似度") — 不在本次修复范围 |
| 4 | 死代码 / 多余 import | 小修 2 处 | 见下 |

---

## 4 标准 — 小修明细

### A1 (unit 01, line 19): `import sys` 未使用

```python
# BEFORE
import os
import re
import shutil
import sys                    # <-- 死 import
from pathlib import Path

# AFTER
import os
import re
import shutil
from pathlib import Path
```

- 影响: 0(从来没用到 `sys.xxx`)
- 验证: `grep -n sys s05_vector_index/units/01_chroma_build/code.py` 仅命中被删的那一行
- 行数: -1 行
- 类型: 死 import,可直接删

### A2 (unit 02, line 22): `SAMPLES = WORKDIR / "samples"` 未使用

```python
# BEFORE
WORKDIR = Path(__file__).resolve().parents[3]
SAMPLES = WORKDIR / "samples"          # <-- 死变量
DB_DIR = WORKDIR / "s05_vector_index" / "_chroma"

# AFTER
WORKDIR = Path(__file__).resolve().parents[3]
DB_DIR = WORKDIR / "s05_vector_index" / "_chroma"
```

- 影响: 0(unit 02 不读 samples 文件 — 它只读 unit 01 写好的 `_chroma/`)
- 验证: `grep -nE "SAMPLES" s05_vector_index/units/02_chroma_query/code.py` 在删除后 0 hit
- 行数: -1 行
- 类型: 死变量,可直接删

**两处合计 -2 行,远低于 5 行阈值。** 修复后 unit 01 + unit 02 仍以 exit 0 通过端到端跑通。

---

## 未触及的已知偏差 (不修,只记录)

| 项 | 位置 | 偏差 | 原因 |
|---|---|---|---|
| score 实际值 ≠ README 预期值 | `units/02_chroma_query/README.md` line 26-30 | README 写 0.950/0.926/0.850,实际跑 0.499/0.487/0.449 | 来自 s04 §2.1 + ledger:cosine 归一化后 top-1 score 大约 0.5 而不是 1.0;**这是数值现象,不是 bug** |
| `_chroma/` 目录命名 vs all-in-rag | (仓库约定) | 仓库用 `**/_chroma/`,全网示例多用 `chroma_db/` | `.gitignore` 已包含两条 pattern;**当前命名是历史决定**,不动 |
| unit 02 输入 EOF 路径 | `units/02_chroma_query/code.py:87` | `input()` 收到 EOF 会抛 `EOFError` | 跟 s07 unit 01 同问题;在 sweep 任务 2 中修了 s07,s05 unit 02 现在仍是 `input().strip()` 直接抛 — **不属本任务范围**,提到完整 review 时处理 |

---

## 审计未发现问题清单

- `importlib.util` 间接加载是 *故意为之*(单元目录以 `01_xxx` 数字开头,不能直接 `import`),跟 s02/s03/s04 同款做法,本任务范围内无需改
- `@lru_cache(maxsize=1)` 包 BGE model 加载是 *故意*(避免每次 embed 重新加载 100MB 模型,跨函数共享)
- `where={"source": "..."}` 这种预过滤语法在 README §3.4 和 `thinking_answers.md` 都讲清楚了,**`code.py` 当前没演示** —— 这是 *设计选择* 而非审计失败;thinking_answers.md 是练习册
- metadata `page` 字段转字符串存,见 README §3.4 第二段 —— 跟 `thinking_answers.md` 第 3 节对齐(0.5.x 拒 None / native segfault)
- BGE 模型 `normalize_embeddings=True` 设置保证 cosine = inner_product,README §1.3 已经解释
