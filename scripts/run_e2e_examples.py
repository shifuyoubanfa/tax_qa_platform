"""端到端验证脚本：用 10 个覆盖全链路分支的例子，对真实运行的系统跑一遍并打印 trace。

在链路里的位置：离线侧"验收工具"。它直接驱动在线编排器 Orchestrator.astream（与对外服务同一入口），
逐条产出 SSE 事件并汇总成可读 trace，用于"端到端真跑通"的验收存证（见 docs/e2e_验证报告.md）。

前置：基础设施已起 + 数据已灌（见 docs/e2e_验证报告.md "数据准备"）。
用法（从项目根目录，确保依赖环境已装好）：
    python -m scripts.run_e2e_examples

覆盖分支：经营数据Text2SQL ×2 / 社保结构化查表 / 商品编码结构化查表 / 精确法规·通用RAG /
政策汇集RAG / 通用问答RAG / 稽查案例RAG / 联网兜底(条件触发) / 多轮记忆。
"""
from __future__ import annotations

import asyncio

from config.constants import SSEEvent
from config.logging_config import get_logger, setup_logging
from app.schemas.chat import ChatRequest
from app.graph.pipeline import Orchestrator

logger = get_logger(__name__)
E = SSEEvent

# (编号, 期望分支, 查询, 会话id) —— 相同会话id用于演示多轮记忆承接。
EXAMPLES = [
    ("01", "经营数据·Text2SQL", "2024年各省的销售额排名是怎样的？", "s-t2s"),
    ("02", "经营数据·Text2SQL", "2024年各个品类的销售额分别是多少？", "s-t2s"),
    ("03", "社保·结构化查表", "杭州2024年的社保缴费基数上下限是多少？", "s-sb"),
    ("04", "商品编码·结构化查表", "天然气的税收分类编码是多少？", "s-pc"),
    ("05", "精确法规/通用·RAG", "小规模纳税人月销售额多少以内免征增值税？", "s-reg"),
    ("06", "政策汇集·RAG", "研发费用加计扣除有哪些优惠政策？", "s-pol"),
    ("07", "通用问答·RAG", "小型微利企业所得税有什么优惠？", "s-gen"),
    ("08", "稽查案例·RAG", "有没有虚开增值税专用发票的稽查案例？", "s-insp"),
    ("09", "联网兜底(条件触发)", "数字货币交易的个人所得税该怎么计算？", "s-web"),
    ("10", "多轮记忆(承接07)", "那它的认定标准是什么？", "s-gen"),
]


async def run_one(orch: Orchestrator, query: str, session: str) -> dict:
    """跑一条 query，把 SSE 事件流汇总成结构化结果。"""
    ev = {"intent": None, "retrieval": [], "refs": 0, "titles": [], "sql": None, "answer": "", "error": None}
    req = ChatRequest(query=query, user_id="e2e", session_id=session, top_k=5)
    async for msg in orch.astream(req):
        e, d = msg.event, msg.data
        if e == E.INTENT.value:
            ev["intent"] = d.get("intent")
        elif e == E.RETRIEVAL.value:
            ev["retrieval"].append("%s=%s" % (d.get("stage"), d.get("count")))
        elif e == E.REFERENCES.value:
            refs = d.get("references") or []
            ev["refs"] = len(refs)
            ev["titles"] = [(r.get("title") or "")[:18] for r in refs[:3]]
        elif e == E.SQL.value:
            ev["sql"] = {"sql": (d.get("sql") or "").replace("\n", " ")[:160],
                         "rows": d.get("row_count"), "err": d.get("error")}
        elif e == E.ANSWER_DELTA.value:
            ev["answer"] += d.get("text") or ""
        elif e == E.ERROR.value:
            ev["error"] = d.get("message")
    return ev


async def main() -> None:
    setup_logging()
    orch = Orchestrator()
    for cid, expect, q, sess in EXAMPLES:
        print("\n" + "=" * 70)
        print("[%s] 期望分支=%s\nQ: %s" % (cid, expect, q))
        try:
            ev = await run_one(orch, q, sess)
            print(" 意图     :", ev["intent"])
            print(" 检索阶段 :", " | ".join(ev["retrieval"]) or "(无, 走text2sql)")
            print(" 引用     :", ev["refs"], ev["titles"])
            if ev["sql"]:
                print(" SQL      :", ev["sql"]["sql"])
                print(" SQL结果  : rows=%s err=%s" % (ev["sql"]["rows"], ev["sql"]["err"]))
            ans = ev["answer"].replace("\n", " ").strip()
            print(" 答案     :", ans[:240] + ("…" if len(ans) > 240 else ""))
            if ev["error"]:
                print(" 错误     :", ev["error"])
        except Exception as exc:  # noqa: BLE001 - 单条失败不影响其它例子
            logger.error("[e2e] 例 %s 异常：%s", cid, exc, exc_info=True)
            print(" 异常     :", repr(exc)[:200])
    print("\n" + "=" * 70 + "\nE2E_DONE")


if __name__ == "__main__":
    asyncio.run(main())
