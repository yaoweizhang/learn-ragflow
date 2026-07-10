#!/usr/bin/env python3
"""
s09 / unit 02 — ReAct 循环:Thought → Action → Observation,跑多步,JSON 失败
时把原文当 Observation 反馈回去让 LLM 自己修正,`max_steps` 兜底防死循环。

复用 unit 01 的 `TOOLS_DESC` / `_llm` / `_retrieve`(importlib 加载)——
unit 01 已经把"工具 + LLM + 检索"封装好,本单元只关心**循环控制**:维护
messages、解析 action、路由工具、终止条件。

运行: python s09_agent_tools/code_02_react_loop.py
需要: 跑通 s09 unit 01; .env 里有 LLM_API_KEY(可选,无也能跑骨架)
"""
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# 复用 unit 01 的工具定义 + LLM 客户端 + 检索函数(importlib,不依赖 chapter-root)
_UNIT01 = Path(__file__).resolve().parent / "code_01_tool_call.py"
_spec = importlib.util.spec_from_file_location("s09_unit01_tool_call", _UNIT01)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
TOOLS_DESC = _mod.TOOLS_DESC
_llm = _mod._llm
_retrieve = _mod._retrieve


# ---------- ReAct 主循环 ----------
def run_agent(question: str, max_steps: int = 5) -> dict:
    """Thought/Action/Observation 循环,最多 max_steps 轮。

    返回 dict 含: `answer`(最终答案)、`trace`(每轮的 thought/action/observation
    列表,便于打印 / 调试 / 单元测试)。
    """
    messages = [
        {"role": "system", "content": TOOLS_DESC},
        {"role": "user", "content": question},
    ]
    trace = []
    for step in range(1, max_steps + 1):
        text = _llm(messages)
        messages.append({"role": "assistant", "content": text})
        # 兼容多种写法:ActionInput 在新行 / 与 Action 同行 / 带 markdown ```json 围栏
        m = re.search(r"Action:\s*(\w+)\b\s*ActionInput:\s*(.+)", text, re.DOTALL)
        if not m:
            # 没抓到 Action —— 把 LLM 原话当 final answer 返回
            return {"answer": text, "trace": trace + [{"step": step, "thought": text,
                                                        "action": None, "obs": None}]}
        action, raw = m.group(1), m.group(2).strip()
        # 剥掉 markdown 围栏
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            # JSON 解析失败:把原文当 Observation 反馈回去,让模型下一轮自己修正
            obs = f"上一次 ActionInput 不是合法 JSON,原文已回显: {raw[:200]}\n请严格按规范输出 JSON。"
            trace.append({"step": step, "thought": text, "action": action, "obs": obs})
            messages.append({"role": "user", "content": f"Observation: {obs}"})
            continue
        if action == "finish":
            ans = payload.get("answer", text)
            trace.append({"step": step, "thought": text, "action": action, "obs": ans})
            return {"answer": ans, "trace": trace}
        if action == "retrieve":
            q = payload.get("query", "")
            obs = _retrieve(q)
        else:
            obs = f"Unknown action: {action}"
        trace.append({"step": step, "thought": text, "action": action, "obs": obs})
        messages.append({"role": "user", "content": f"Observation: {obs}"})
    return {"answer": "Max steps reached.", "trace": trace}


# ---------- 入口 ----------
def main() -> None:
    try:
        question = input("问: ").strip() or "R3630 G5 的内存插槽数量"
    except EOFError:
        question = "R3630 G5 的内存插槽数量"
    print(f"[Q] {question}\n")

    if not os.environ.get("LLM_API_KEY"):
        # 骨架演示:展示 trace 结构和每一步的占位文本,不需要真调 LLM
        print("[skipped: LLM_API_KEY not set] — 演示 trace 形状:\n")
        fake_trace = [
            {"step": 1, "thought": "用户问的是 R3630 G5 的内存插槽数量。",
             "action": "retrieve",
             "obs": "- (server_whitepaper.pdf#2) 配备 32 个 DIMM 内存插槽 ..."},
            {"step": 2, "thought": "找到了。",
             "action": "finish",
             "obs": "R3630 G5 配备 32 个 DIMM 内存插槽。"},
        ]
        for t in fake_trace:
            print(f"--- step {t['step']} ---")
            print(f"Thought: {t['thought']}")
            print(f"Action:  {t['action']}")
            print(f"Obs:     {t['obs']}\n")
        print(f"[A] {fake_trace[-1]['obs']}")
        return

    result = run_agent(question)
    print(f"--- trace ({len(result['trace'])} step(s)) ---")
    for t in result["trace"]:
        print(f"\n[step {t['step']}]")
        print(f"  Thought: {t['thought']}")
        print(f"  Action:  {t['action']}")
        obs = t["obs"] or ""
        print(f"  Obs:     {obs[:200]}{'...' if len(obs) > 200 else ''}")
    print(f"\n[A] {result['answer']}")


if __name__ == "__main__":
    main()