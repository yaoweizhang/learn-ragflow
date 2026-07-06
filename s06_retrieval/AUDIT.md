# s06 检索 (Retrieval) — 代码审计

**审计日期**: 2026-07-06
**审计范围**: `s06_retrieval/code.py`, `s06_retrieval/units/01_bm25/code.py`, `s06_retrieval/units/02_hybrid_fusion/code.py`
**审计基线**: 新写的 README §3.3 函数表 + `ragflow_notes/hybrid_retrieval.md` + `units/*/README.md`
**审计目的**: 验证 README 4-段式叙述的"核心函数 / fusion 设计 / 工具速览"和实际代码的对照情况

---

## 审计结果 (4 标准)

| # | 标准 | 等级 | 备注 |
|---|---|---|---|
| 1 | README 提过的函数都在代码里 | 对齐 | README §3.3 表 9 行 (`tokenize` / `BM25` / `BM25.score` / `bm25_topk` / `_model` / `_embed` / `_cosine` / `hybrid_topk` / `main` x2) 全部在 code 中找到定义,无虚构 |
| 2 | 代码里 main / 关键函数在 README 里讲清楚 | 对齐 | README §3.3 表每行给出 file / input / output / 1-line purpose;`hybrid_topk` 的 `α·v + (1-α)·b_norm` 公式在 README §3.4 显式给出,与代码 line 102 完全一致 |
| 3 | README 的 sample output 跟实际跑出来一致 | 对齐 | unit 01 输出末 3 行 `[disclosure.docx#-] bm25=3.896...` / `bm25=3.871...` / `bm25=2.828...`,与 README §3.7 完全一致;unit 02 输出末 3 行 `final=0.503 / 0.487 / 0.471` + 子分 `0.95*vec + 0.05*bm25` 与 README §3.7 完全一致 |
| 4 | 死代码 / 多余 import | 小修 1 处 | 见下 |

---

## 4 标准 — 小修明细

### A1 (unit 01, line 19): `from typing import Iterable` 未使用

```python
# BEFORE
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable          # <-- 死 import

# AFTER
import math
import re
from collections import Counter
from pathlib import Path
```

- 影响: 0(从来没用到 `Iterable`)
- 验证: 删除后 `grep -n "Iterable" units/01_bm25/code.py` 0 hit
- 行数: -1 行
- 类型: 死 import,可直接删

**1 处合计 -1 行,远低于 5 行阈值。** 修复后 unit 01 + unit 02 + 聚合入口 `s06_retrieval/code.py` 均以 exit 0 通过端到端跑通。

---

## 未触及的已知项 (不修,只记录)

| 项 | 位置 | 状态 | 说明 |
|---|---|---|---|
| `score` 实际值范围 [0.4, 0.6] | unit 02 输出 | 已知 | 来自 s04 §2.1 + ledger:cosine 归一化后 top-1 score 通常在 0.5 而不是 1.0(BGE 编码差异);**数值现象,不是 bug** |
| `sys` 在 unit 02 | `units/02_hybrid_fusion/code.py:20, 42` | 用到 | `sys.modules[_spec01.name] = _mod01` (line 42) 实际用到,**不是死 import**,保留 |
| `importlib.util` 间接加载 | 三个 code 文件 | 故意 | 单元目录以数字开头,不能直接 `import`;跟 s02/s03/s04/s05 同款做法 |
| `EMBED_MODEL` env 默认 `BAAI/bge-small-zh-v1.5` | unit 02 line 54 | 跟 s04 对齐 | 跟 s04 unit 01 用同一款本地 BGE,无需重复解释 |
| Windows urllib3/botocore 兼容补丁 | unit 02 line 31-36 | 故意 | 防止 `urllib3.util.ssl_.DEFAULT_CIPHERS` 在 `botocore` 旧版本下缺失 |
| `alpha=0.95` 默认偏向量 | unit 02 line 80, 124 | 跟 ragflow 对齐 | README §3.4 解释为何选 0.95;`ragflow_notes/hybrid_retrieval.md` 中 `FusionExpr("weighted_sum", {"weights": "0.05,0.95"})` 也是向量主导 |

---

## 审计未发现问题清单

- `hybrid_topk` 的 fusion 公式 `alpha * v + (1 - alpha) * b_norm` 与 README §3.4 第一条公式**完全一致**(代码 line 102)
- `BM25.score` 的 Robertson 公式 `IDF * tf * (k1+1) / (tf + k1*(1-b+b*dl/avgdl))` 与 README §1.2 表中描述及 all-in-rag §一.1.1 公式结构一致(代码 line 136-141)
- `chunk_id` 命名 `{source}#{page}#p{n}` 在 fusion 命中中保留(line 99),与 s05 一致,s07 可直接复用
- `_cosine` 在已 L2 归一化向量上等价于内积(line 64-69),与 README §1.2 表 "cosine ≡ inner_product" 说法一致
- `bm25_topk` 按 BM25 分降序 + `scores[i] > 0` 过滤(line 148-153),避免返回全 0 分的"假命中";README §3.3 表第 4 行描述"取分 > 0 的前 k 条"对得上

---

## Forbidden content check

`git grep -nE '\[\^[0-9]|RAG 已死|参考文献' s06_retrieval/README.md s06_retrieval/AUDIT.md` → empty (exit 1).