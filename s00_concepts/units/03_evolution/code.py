"""s00 / Unit 03 — 三代 RAG 演进:初级 / 高级 / 模块化

Offline demo. Prints the stage pipeline of each RAG generation
and shows which chapters of this tutorial cover each stage.
"""

GENERATIONS = [
    {
        "name": "初级 RAG (Naive)",
        "stages": ["离线:解析→切块→embed→写库", "在线:问题embed→检索→LLM"],
        "stages_offline": 4,
        "stages_online": 3,
        "chapters": "s01-s03-s04-s05-s08(基础链路)",
    },
    {
        "name": "高级 RAG (Advanced)",
        "stages": [
            "离线:解析→切块→embed→写库",  # 同上
            "在线:问题embed→查询重写→双路召回(BM25+向量)→Rerank→LLM",
        ],
        "stages_offline": 4,
        "stages_online": 5,
        "chapters": "+ s06(混合检索) + s07(Rerank)",
    },
    {
        "name": "模块化 RAG (Modular)",
        "stages": [
            "离线:解析→切块→embed→写库 + 知识图谱/多模态旁路",
            "在线:Agent 路由→多路召回→融合→可选 Rerank→LLM",
        ],
        "stages_offline": 5,
        "stages_online": 6,
        "chapters": "+ s09(Agent) + s10(GraphRAG) + s11(多模态)",
    },
]

def main() -> None:
    print("RAG 演进 = 流程上'多出来的环节'越来越多\n" + "=" * 50)
    for g in GENERATIONS:
        print(f"\n{g['name']}")
        print(f"  离线环节数: {g['stages_offline']}    在线环节数: {g['stages_online']}")
        for s in g["stages"]:
            print(f"    • {s}")
        print(f"  本教程对应章节: {g['chapters']}")
    print()
    print("趋势:从'一条直线'到'积木式可编排'。每多一道工序就多一种优化点,")
    print("但也多一个失败模式 — 这就是为什么教程从 s01 的 5 行玩具开始,")
    print("逐步加环节加复杂度到 s12 的多容器部署。")

if __name__ == "__main__":
    main()
