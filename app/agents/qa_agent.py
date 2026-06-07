"""QaAgent —— 给"主图本体问答(QA)"一个对外的 agent 名字的【零逻辑薄门面】（P1 门面 · 纯新增）。

本模块在整体链路里的位置 / 它到底是什么：
    QA 不是一个"子 Agent"，QA【就是主图本身】——
    load_memory → understand → route → rag_recall → coarse_rank → fine_rank → rerank
    → build_context → generate_answer → save_memory。
    现有的 ``app/graph/pipeline.py::Orchestrator.astream`` 已经把这张主图跑通，并用
    graph.astream(stream_mode=["updates","messages"]) 实现了【真·逐 token 流式】与【按节点名匹配的 SSE】。

    所以本门面【绝不重写 RAG、绝不抽取主图节点、绝不改主图拓扑】。它只做一件事：
    把"对 QA 的调用"原样【委托】给 Orchestrator，让 QA 在 agent 能力地图上有一个对外的名字
    （与 Text2SQLAgent / 结构化召回源 Agent 并列登记，便于阅读与对外描述），仅此而已。

为什么要有这个门面（既然它什么逻辑都不加）：
    纯粹是"命名 + 对外接口"的需要。能力地图里需要能指着说"QA 这一路对应 QaAgent"；但 QA 的真实
    实现就是主图本体，不该被复制/重写。门面让"名字"和"实现"解耦：名字在这里，实现仍在 Orchestrator。

硬约束遵守说明（经多 agent 评审裁定）：
    - 不引入任何 LLM supervisor / 工具循环：本门面只是直通调用，路由仍由确定性层(IntentClassifier
      + SearchRouter + route_decider)决定，一行不动。
    - 不改主图拓扑/节点名：astream 直接复用 Orchestrator.astream，逐 token 流与按节点名的 SSE
      （understand/rag_recall/.../generate_answer/...）保持原样，不被打断。

健壮性：Orchestrator 惰性持有图、astream 内部已全程 try/except 收敛为 error 事件；本门面不另加
异常处理（加了反而会吞掉/改变现有事件语义）。import 本模块无副作用、不连任何服务、不编译图。
"""
from __future__ import annotations

from typing import AsyncIterator, Optional

from config.logging_config import get_logger
from app.schemas.chat import ChatRequest, SSEMessage

logger = get_logger(__name__)


class QaAgent:
    """主图本体问答(QA)的对外门面：把调用直接委托给现有 Orchestrator（零逻辑）。

    用法（与直接用 Orchestrator 完全等价，只是换了个对外名字）::

        agent = QaAgent()
        async for msg in agent.astream(req):
            ...   # 与 Orchestrator.astream 产出的 SSEMessage 流逐条一致

    设计要点：本类不持有任何 RAG 逻辑，只持有一个【惰性】的 Orchestrator 实例并转发调用。
    """

    def __init__(self, orchestrator: Optional[object] = None) -> None:
        """构造门面。

        :param orchestrator: 可选注入一个现成的 Orchestrator（便于测试/复用）；
                             不传则在首次 astream 时惰性创建——保持 import 期零副作用、不编译主图。
        """
        # 惰性持有：不在 __init__ 里 import/构造 Orchestrator，避免 import 本模块就牵连 langgraph。
        self._orchestrator = orchestrator

    def _get_orchestrator(self):
        """惰性获取被委托的 Orchestrator（首次调用才创建）。

        :return: Orchestrator 实例（来自 app/graph/pipeline.py，QA 的真实实现）。
        """
        if self._orchestrator is None:
            # 惰性导入：仅在真正要跑链路时才碰 pipeline（其内部惰性编译 LangGraph 主图）。
            from app.graph.pipeline import Orchestrator
            self._orchestrator = Orchestrator()
            logger.info("[QaAgent] 惰性创建 Orchestrator（QA 的真实实现，门面仅转发）")
        return self._orchestrator

    def astream(self, req: ChatRequest) -> AsyncIterator[SSEMessage]:
        """对外流式入口：原样转发给 Orchestrator.astream，保留真·逐 token 流与按节点名的 SSE。

        说明：这里【直接返回】Orchestrator.astream(req) 的异步迭代器本身（不包一层 async for 再
        转发），从而 100% 保留其逐 token 行为与事件顺序——不重写、不缓冲、不改造任何一帧。

        :param req: 对外请求体。
        :return: 异步迭代器，逐条产出与 Orchestrator.astream 完全一致的 SSEMessage。
        """
        logger.info("[QaAgent] astream 委托给 Orchestrator user_id=%s session_id=%s",
                    req.user_id, req.session_id)
        return self._get_orchestrator().astream(req)

    async def run(self, req: ChatRequest) -> str:
        """非流式便捷入口：把委托得到的 answer_delta 事件拼接成完整答案字符串。

        在链路里的位置：这是给"不需要流式、只想要最终文本"的调用方的一个便捷封装；它【仍然走】
        Orchestrator.astream（同一张主图、同一套确定性路由），只是把流出的答案增量收拢成整段。
        非 answer_delta 的事件（intent/retrieval/references/sql/done/error）在此被透明跳过，
        因为本方法的契约就是"只回最终答案文本"。

        :param req: 对外请求体。
        :return: 拼接后的完整答案文本（无答案时返回空串）。
        """
        logger.info("[QaAgent] run（非流式封装）开始 user_id=%s session_id=%s",
                    req.user_id, req.session_id)
        from config.constants import SSEEvent

        pieces: list[str] = []
        async for msg in self.astream(req):
            if msg.event == SSEEvent.ANSWER_DELTA.value:
                pieces.append(str((msg.data or {}).get("text", "")))
        answer = "".join(pieces)
        logger.info("[QaAgent] run 结束，答案长度=%d", len(answer))
        return answer


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：仅验证门面可实例化、不连任何服务、不编译主图（惰性）。
    agent = QaAgent()
    print("[qa_agent 自测] QaAgent 实例化成功（未触发 Orchestrator 创建）=>",
          agent._orchestrator is None)
