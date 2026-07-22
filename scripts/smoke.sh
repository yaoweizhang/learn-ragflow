#!/usr/bin/env bash
#
# scripts/smoke.sh — 一键烟测整个 learn-ragflow 仓库
#
# 用法:
#   bash scripts/smoke.sh                 # 全跑(含 s08+ LLM 调用)
#   SMOKE_NO_LLM=1 bash scripts/smoke.sh  # 只跑 s01-s07(s08+ LLM 章跳过;CI 用)
#   SMOKE_OFFLINE=1 bash scripts/smoke.sh # 强制走本地 HF 缓存(沙箱/CI 环境)
#
# 退出码:
#   0 — 所有启用的步骤都通过
#   1 — 至少一步失败(打印失败章节名 + 最后 20 行输出)
#
# 设计取舍:
#   - 每个 chapter 用 `<<< ""` 喂 EOF,触发各 main() 里的 EOFError 守卫,
#     既不卡在 input() 上,又能验证非交互路径走通
#   - s04 的 BGE 模型 + s07 的 reranker 都从 ~/.cache/huggingface 读,
#     沙箱默认屏蔽 HF 时设 SMOKE_OFFLINE=1 即可
#   - s08+ 需 .env 里有 LLM_API_KEY;SMOKE_NO_LLM=1 时跳过
#   - s10 query.py 喂 "紫光恒越 R3630 G5" 实体名,触发 1 跳查询
#   - s12 启 uvicorn 后台 + curl POST /qa,拿回 200 + JSON 才算通过
set -u

cd "$(dirname "$0")/.."  # 切到仓库根
PYTHON=${PYTHON:-python}

# 环境变量
if [ -f .env ]; then
  set -a; . ./.env; set +a
fi
export HF_HUB_OFFLINE=${SMOKE_OFFLINE:-${HF_HUB_OFFLINE:-0}}
export TRANSFORMERS_OFFLINE=${SMOKE_OFFLINE:-${TRANSFORMERS_OFFLINE:-0}}
export TOKENIZERS_PARALLELISM=false  # 避免 HF tokenizer fork 警告

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
PASS=0; FAIL=0
FAILED_CHAPS=()

# 跑一个 python 脚本并捕获结果。$1=label, 剩下的是 cmd args
run_py() {
  local label="$1"; shift
  echo -e "${YELLOW}== $label ==${NC}"
  local out
  if out=$("$@" 2>&1); then
    echo "$out" | tail -3
    echo -e "${GREEN}PASS${NC} $label"
    PASS=$((PASS+1))
  else
    echo "$out" | tail -20
    echo -e "${RED}FAIL${NC} $label"
    FAIL=$((FAIL+1))
    FAILED_CHAPS+=("$label")
  fi
}

# ---- s01 朴素检索 (无依赖) ----
echo -e "${YELLOW}== s01 substring_match ==${NC}"
if out=$("$PYTHON" s01_what_is_rag/substring_match.py </dev/null 2>&1); then
  echo "$out" | tail -3
  echo -e "${GREEN}PASS${NC} s01 substring_match"; PASS=$((PASS+1))
else
  echo "$out" | tail -20
  echo -e "${RED}FAIL${NC} s01 substring_match"; FAIL=$((FAIL+1))
  FAILED_CHAPS+=("s01 substring_match")
fi

# ---- s02 文档加载 ----
run_py "s02 basic_load" "$PYTHON" s02_doc_loading/basic_load.py </dev/null

# ---- s03 切块 ----
run_py "s03 basic_chunk" "$PYTHON" s03_chunking/basic_chunk.py </dev/null

# ---- s04 本地 BGE embed ----
run_py "s04 local_bge" "$PYTHON" s04_embedding/local_bge.py </dev/null

# ---- s05 Chroma 索引构建 ----
run_py "s05 chroma_build" "$PYTHON" s05_vector_index/chroma_build.py </dev/null

# ---- s06 BM25 ----
run_py "s06 bm25" "$PYTHON" s06_retrieval/bm25.py </dev/null

# ---- s07 cross-encoder rerank ----
run_py "s07 cross_encoder_rerank" "$PYTHON" s07_rerank/cross_encoder_rerank.py </dev/null

# ---- s08+ 需要 LLM;SMOKE_NO_LLM=1 时整体跳过 ----
if [ "${SMOKE_NO_LLM:-0}" = "1" ]; then
  echo -e "${YELLOW}== SMOKE_NO_LLM=1, 跳过 s08/s09/s10/s12 ==${NC}"
else
  if [ -z "${LLM_API_KEY:-}" ] || [ "$LLM_API_KEY" = "sk-your-key-here" ]; then
    echo -e "${YELLOW}== LLM_API_KEY 未配置, 跳过 s08/s09/s10/s12 (设 SMOKE_NO_LLM=1 显式跳过) ==${NC}"
  else
    run_py "s08 prompt_template" "$PYTHON" s08_prompt_generate/prompt_template.py </dev/null
    run_py "s09 tool_call"      "$PYTHON" s09_agent_tools/tool_call.py </dev/null
    run_py "s09 react_loop"     "$PYTHON" s09_agent_tools/react_loop.py </dev/null
    run_py "s10 extract"        "$PYTHON" s10_graphrag/extract.py </dev/null
    # s10 query.py 读 stdin 输入实体名
    echo -e "${YELLOW}== s10 query (紫光恒越 R3630 G5) ==${NC}"
    if out=$(echo "紫光恒越 R3630 G5" | "$PYTHON" s10_graphrag/query.py 2>&1); then
      echo "$out" | tail -5
      echo -e "${GREEN}PASS${NC} s10 query"; PASS=$((PASS+1))
    else
      echo "$out" | tail -20
      echo -e "${RED}FAIL${NC} s10 query"; FAIL=$((FAIL+1))
      FAILED_CHAPS+=("s10 query")
    fi

    # ---- s12 FastAPI app + POST /qa ----
    echo -e "${YELLOW}== s12 app (FastAPI + uvicorn + curl /qa) ==${NC}"
    # setsid 让 python 起新 session,kill 时连同 uvicorn 子进程一起收
    setsid "$PYTHON" s12_deployment/app.py >/tmp/s12_app.log 2>&1 &
    APP_PID=$!
    sleep 18  # 等 uvicorn 起来 + 首次请求触发 lazy-load Chroma + BGE
    QA_RESP=$(curl -s -X POST http://127.0.0.1:8000/qa \
      -H 'Content-Type: application/json' \
      -d '{"question":"R3630 G5 有几个内存插槽?"}' || echo "CURL_FAIL")
    # 杀整个进程组(-PID),确保 uvicorn worker 不留僵尸占住 8000
    kill -- -$APP_PID 2>/dev/null || kill -TERM $APP_PID 2>/dev/null
    wait $APP_PID 2>/dev/null
    # 兜底:如果端口还被占,fuser 强杀
    fuser -k 8000/tcp 2>/dev/null || true
    if echo "$QA_RESP" | grep -q '"text"'; then
      echo "$QA_RESP" | head -1
      echo -e "${GREEN}PASS${NC} s12 app /qa"; PASS=$((PASS+1))
    else
      echo -e "${RED}FAIL${NC} s12 app /qa (response: $QA_RESP)"
      FAIL=$((FAIL+1))
      FAILED_CHAPS+=("s12 app /qa")
    fi
  fi
fi

# ---- 总结 ----
echo ""
echo "==================================="
echo -e "Passed: ${GREEN}$PASS${NC}    Failed: ${RED}$FAIL${NC}"
if [ $FAIL -gt 0 ]; then
  echo -e "${RED}Failed chapters:${NC}"
  for c in "${FAILED_CHAPS[@]}"; do echo "  - $c"; done
  exit 1
fi
echo -e "${GREEN}✅ smoke OK${NC}"