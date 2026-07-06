# s07 重排序 (Rerank) — 代码审计

**审计日期**: 2026-07-06
**审计范围**: `s07_rerank/code.py`, `s07_rerank/units/01_cross_encoder_rerank/code.py`
**审计基线**: 新写的 README §3.3 函数表 + `ragflow_notes/rerank.md` + `units/01_cross_encoder_rerank/README.md`
**审计目的**: 验证 README 4-段式叙述的"核心函数 / rerank 设计 / 主流 rerank 速览"和实际代码的对照情况

---

## 审计结果 (4 标准)

| # | 标准 | 等级 | 备注 |
|---|---|---|---|
| 1 | README 提过的函数都在代码里 | 对齐 | README §3.3 表 7 行 (`_embed_model` / `_embed` / `_cosine` / `_reranker` / `rerank` / `_hybrid_topk` / `main`) 全部在 code 中找到定义,无虚构 |
| 2 | 代码里 main / 关键函数在 README 里讲清楚 | 对齐 | README §3.3 表每行给出 file / input / output / 1-line purpose;`rerank(query, hits, top_k=3)` 的签名 + 输入输出结构在 README §3.6 给出;`@lru_cache(maxsize=1)` 缓存策略在 §3.4 解释 |
| 3 | README 的 sample output 跟实际跑出来一致 | 小修 | README §3.6/§3.7 原值来自 Phase A 时旧 docs(用不同 chunk 编号 / 旧 score),实际跑出来 `query='内存'` 是 `loaded 34 chunks` + BEFORE/AFTER 不同分;已用 live run 实测替换 §3.6 例子和 §3.7 实测输出块 |
| 4 | 死代码 / 多余 import | 对齐 | 全文 grep 所有 import: `importlib.util` (间接加载)、`os` (`os.environ.get`)、`sys` (`sys.modules[_spec01.name]`)、`lru_cache` (用)、`Path` (用)、`load_dotenv` (用)、`urllib3.util.ssl_` (Windows 兼容补丁,跟 s05 / s06 同款);lazy import 的 `SentenceTransformer` / `FlagReranker` / `chromadb` 都在函数体内用到 |

---

## 4 标准 — 小修明细

### A1 (README §3.6 / §3.7): sample output 替换为 live run 结果

- 触发: 跑 `python s07_rerank/units/01_cross_encoder_rerank/code.py` 实际输出,末 3 行 BEFORE 是 `score=0.795 / 0.736 / 0.726`,AFTER 是 `rerank=0.664 / 0.550 / 0.527`,命中 `server_whitepaper.pdf#4 / #2 / #3 / #1` 等。原 README 沿用 Phase A 时 docs 的旧期望值 (`#1 rerank=0.954 vec=0.905`、`#2 rerank=0.913 vec=0.575`、`#3 rerank=0.644 vec=0.976`),跟 live run 不一致
- 影响: 教学影响 — 学习者按 README sample 期望值对不上 live run 会怀疑代码 bug;语义影响 — 0
- 修复: README §3.6 例子 + §3.7 实测输出块换成 live run 实测值,保留"rerank 把 vec 高但 bm25 也高的 `server_whitepaper.pdf#4` 压到第三、把 cross-encoder 觉得沾边的 `#3 应用场景` 顶到第一"这一不同步现象的解释
- 行数: README 改 ~14 行 (example 块 14 行 + 实测块 14 行),**纯 docstring,不计入 5 行代码阈值**

**代码层面 0 处小修;README 改 2 处样例块 (sample output 同步 live run)。** 修复后聚合入口 `s07_rerank/code.py` + `units/01_cross_encoder_rerank/code.py` 均以 exit 0 通过端到端跑通,末 3 行实测输出见下。

---

## 末 3 行 live run 输出 (criterion 3 证据)

```
loaded 34 chunks from samples/
query='内存', alpha=0.5 (BM25 + dense 等权融合)
...
  #1 [server_whitepaper.pdf#3] rerank=0.664 vec=0.545 | 四、应用场景 云数据中心...
  #2 [server_whitepaper.pdf#1] rerank=0.550 vec=0.559 | 二、关键特性 计算密度:单台 2U 机箱...
  #3 [server_whitepaper.pdf#4] rerank=0.527 vec=0.590 | 五、可靠性与可维护性 冗余设计...
```

(完整命令: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 30 python s07_rerank/units/01_cross_encoder_rerank/code.py < /dev/null`)

---

## 未触及的已知项 (不修,只记录)

| 项 | 位置 | 状态 | 说明 |
|---|---|---|---|
| `loaded 34 chunks` | unit 01 输出 | 已知 | 4 页白皮书 + 27 段披露报告 → 34 个 chunk (跟 s03 / s04 / s05 / s06 一致);数字跟 docs "32 chunks" 等描述不一致的,以 live run 实测为准 |
| `EMBED_MODEL` env 默认 `BAAI/bge-small-zh-v1.5` | unit 01 line 54 | 跟 s04 对齐 | 跟 s04 / s05 / s06 用同一款本地 BGE,无需重复解释 |
| Windows urllib3/botocore 兼容补丁 | unit 01 line 31-36 | 故意 | 防止 `urllib3.util.ssl_.DEFAULT_CIPHERS` 在 `botocore` 旧版本下缺失 (跟 s05 / s06 同款) |
| `use_fp16=False` (CPU 友好) | unit 01 line 74 | 跟 README §3.4 对齐 | CPU 跑 fp16 反而更慢,GPU 才开 fp16 |
| `alpha=0.5` 默认等权融合 | unit 01 line 80, 124 | 跟 README §3.6 对齐 | demo 上看 rerank 跟 vec 不同步最明显;s06 unit 02 是 `alpha=0.95` (术语密集场景),s07 unit 01 是 `alpha=0.5` (通用 demo) |
| `query` 兜底 `"内存"` | unit 01 line 124 | EOFError 兼容 | 防止 `input()` 在 non-TTY 环境 (CI / pipe) 抛 EOFError |

---

## 大修 (需用户 sign-off)

无。