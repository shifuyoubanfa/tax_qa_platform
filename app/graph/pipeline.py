"""全链路编排（Orchestrator）+ LangGraph 主图构建。
本模块在整体链路里的位置：最顶层"指挥中心"。对外只暴露一个能力——给定一次 ChatRequest，
按 加载记忆 -> QU -> 路由 -> (RAG 或 Text2SQL) -> 存记忆 的顺序驱动整条链路，并把每个阶段的
进展实时产出为 SSEMessage（意图/检索进度/引用/SQL/答案增量/完成/错误），交给 HTTP 路由层
转成 SSE 帧推给前端。

为什么"编排"同时提供两种形态：
1. build_pipeline_graph()：用 LangGraph StateGraph 把 nodes.py 的节点连成图（route 后条件边分流
   rag / text2sql），对标 掌柜问数 的 LangGraph 主图。它表达的是"整条链路的静态结构"，便于
   可视化、整体 invoke、以及 smoke 测试验证拓扑正确——这是"链路的图纸"。
2. Orchestrator.astream()：真正对外服务的入口，它【真正执行上面编译出的同一张 LangGraph 图】——
   用 graph.astream(stream_mode=["updates","messages"]) 同时订阅"节点产出"与"LLM token"两种流：
   节点产出映射成 intent/retrieval/references/sql 事件；generate_answer 节点的 LLM token 逐个产出
   answer_delta（真·逐 token 流式）。这是"链路的发动机"，与上面那张图同源同跑（不再手动逐节点驱动）。

健壮性：astream 全程 try/except，任何阶段异常都产出一条 error 事件并安全收尾，绝不让异常
冒泡导致连接挂死。

风格对标 掌柜问数 LangGraph 主图 + 掌柜智库 SSE 事件流。
"""
from __future__ import annotations

from typing import AsyncIterator, Optional

from config.constants import RouteType, SSEEvent
from config.logging_config import get_logger
from app.schemas.chat import ChatRequest, SSEMessage
from app.schemas.document import QUResult, Text2SQLResult
from app.graph.state import GraphState
from app.graph import nodes

logger = get_logger(__name__)


def build_pipeline_graph():
    """构建并编译 LangGraph 主图（整条问答链路的静态结构）。

    图结构::

        START -> load_memory -> understand -> route
        route --(rag)--> rag_recall -> coarse_rank -> fine_rank -> rerank -> build_context -> save_memory -> END
        route --(text2sql)--> text2sql -> save_memory -> END

    说明：该图用于"非流式整体执行 / 可视化 / smoke 测试拓扑"。流式对外服务走 Orchestrator.astream。

    :return: 编译后的 LangGraph 图对象（CompiledStateGraph）。
    :raise ImportError: 当 langgraph 未安装时由 import 抛出。
    """
    from langgraph.graph import StateGraph, START, END

    builder = StateGraph(GraphState)

    # 注册所有节点（节点实现来自 nodes.py）
    builder.add_node("load_memory", nodes.load_memory)
    builder.add_node("understand", nodes.understand)
    builder.add_node("route", nodes.route)
    builder.add_node("rag_recall", nodes.rag_recall)
    builder.add_node("coarse_rank", nodes.coarse_rank)
    builder.add_node("fine_rank", nodes.fine_rank)
    builder.add_node("rerank", nodes.rerank)
    builder.add_node("build_context", nodes.build_context)
    builder.add_node("generate_answer", nodes.generate_answer)   # RAG 答案生成（LLM 流式）
    builder.add_node("text2sql", nodes.text2sql)
    builder.add_node("save_memory", nodes.save_memory)

    # 串接边：入口 -> 记忆 -> QU -> 路由
    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "understand")
    builder.add_edge("understand", "route")

    # 路由后条件分流：RAG 链路 vs Text2SQL 链路
    builder.add_conditional_edges(
        "route",
        nodes.route_decider,
        {"rag": "rag_recall", "text2sql": "text2sql"},
    )

    # RAG 链路：召回 -> 粗排 -> 精排 -> 重排 -> 建上下文 -> 存记忆
    builder.add_edge("rag_recall", "coarse_rank")
    builder.add_edge("coarse_rank", "fine_rank")
    builder.add_edge("fine_rank", "rerank")
    builder.add_edge("rerank", "build_context")
    builder.add_edge("build_context", "generate_answer")
    builder.add_edge("generate_answer", "save_memory")

    # Text2SQL 链路：执行 Agent -> 存记忆
    builder.add_edge("text2sql", "save_memory")

    builder.add_edge("save_memory", END)

    compiled = builder.compile()
    logger.info("[编排] LangGraph 主图编译完成")
    return compiled


class Orchestrator:
    """全链路流式编排器：驱动 nodes.py 的各能力组件，按阶段产出 SSEMessage。

    用法（在 HTTP 路由层）::

        orchestrator = Orchestrator()
        async for msg in orchestrator.astream(req):
            yield format_sse(msg.event, msg.data)
    """

    def __init__(self) -> None:
        # 持有 Summarizer/Text2SQLAgent 的获取走 nodes 里的惰性单例，这里无需重复构造
        self._graph = None  # 惰性编译的主图（供需要整体 invoke 的场景）

    @property
    def graph(self):
        """惰性编译并返回 LangGraph 主图（首次访问才构建）。

        :return: 编译后的 LangGraph 图对象。
        """
        if self._graph is None:
            self._graph = build_pipeline_graph()
        return self._graph

    async def astream(self, req: ChatRequest) -> AsyncIterator[SSEMessage]:
        """驱动【编译后的 LangGraph 主图】跑完整条链路，并把图的进度流式映射成 SSE 事件。

        真·LangGraph 执行：用 graph.astream(stream_mode=["updates","messages"]) 同时订阅两种流——
        - "updates"：每个节点执行完产出的状态片段 -> 映射成 intent/retrieval/references/sql 事件；
        - "messages"：节点内 LLM 的 token 流 -> 只取 generate_answer 节点的 token，逐 token 产出
          answer_delta（真·逐 token 流式；其它节点的 LLM 调用如意图分类/改写按节点名过滤掉）。
        答案兜底：若 generate_answer 没产生 LLM token（如空上下文走兜底话术、非 LLM 产物），
        则按 updates 里的 answer 切块补发，保证答案一定到达前端。

        :param req: 对外请求体。
        :return: 异步迭代器，逐条产出 SSEMessage。
        :raise: 不向外抛；异常统一转为 error 事件。
        """
        logger.info("[编排] astream(LangGraph) 开始 user_id=%s session_id=%s query=%s",
                    req.user_id, req.session_id, req.query)
        init_state: GraphState = {
            "query": req.query,
            "user_id": req.user_id,
            "session_id": req.session_id,
            "top_k": req.top_k,
            "filters": req.filters or {},
        }
        answer_streamed = False  # generate_answer 是否已通过 messages 模式逐 token 流出过答案

        try:
            graph = self.graph  # 编译后的 LangGraph 图（惰性）
            # 关键：真正执行图，并同时订阅"节点产出(updates)"与"LLM token(messages)"两种流
            async for mode, chunk in graph.astream(init_state, stream_mode=["updates", "messages"]):
                if mode == "messages":
                    # chunk = (AIMessageChunk, metadata)；只取 generate_answer 节点的 token 当答案增量
                    msg, meta = chunk if isinstance(chunk, tuple) and len(chunk) == 2 else (None, {})
                    if msg is not None and (meta or {}).get("langgraph_node") == "generate_answer":
                        text = getattr(msg, "content", "") or ""
                        if text:
                            answer_streamed = True
                            yield SSEMessage(event=SSEEvent.ANSWER_DELTA.value, data={"text": text})
                elif mode == "updates":
                    # chunk = {节点名: 该节点写回的状态片段}
                    for node_name, update in (chunk or {}).items():
                        for sse in self._updates_to_sse(node_name, update or {}, answer_streamed):
                            yield sse

            yield SSEMessage(event=SSEEvent.DONE.value, data={"finished": True})
            logger.info("[编排] astream(LangGraph) 正常结束")
        except Exception as e:  # noqa: BLE001 顶层兜底：任何异常都转 error 事件，保证连接优雅收尾
            logger.error("[编排] astream 异常：%s", e, exc_info=True)
            yield SSEMessage(event=SSEEvent.ERROR.value, data={"message": f"服务处理异常：{e}"})

    def _updates_to_sse(self, node_name: str, update: dict, answer_streamed: bool) -> list[SSEMessage]:
        """把某个节点的状态产出(update)映射成对应的 SSE 事件列表（updates 流的消费）。

        :param node_name: 刚执行完的节点名。
        :param update: 该节点写回 state 的字段片段。
        :param answer_streamed: 截至目前 generate_answer 是否已通过 messages 流出过答案 token。
        :return: 要产出的 SSEMessage 列表（可能为空）。
        """
        E = SSEEvent
        if node_name == "understand":
            qu = update.get("qu")
            if qu is not None:
                return [SSEMessage(event=E.INTENT.value, data={
                    "intent": qu.intent,
                    "sub_queries": qu.sub_queries,
                    "is_short_query": qu.is_short_query,
                })]
        elif node_name == "rag_recall":
            return [SSEMessage(event=E.RETRIEVAL.value,
                               data={"stage": "recall", "count": len(update.get("recalled") or [])})]
        elif node_name == "coarse_rank":
            return [SSEMessage(event=E.RETRIEVAL.value,
                               data={"stage": "coarse_rank", "count": len(update.get("ranked") or [])})]
        elif node_name == "fine_rank":
            return [SSEMessage(event=E.RETRIEVAL.value,
                               data={"stage": "fine_rank", "count": len(update.get("ranked") or [])})]
        elif node_name == "rerank":
            return [SSEMessage(event=E.RETRIEVAL.value,
                               data={"stage": "rerank", "count": len(update.get("reranked") or [])})]
        elif node_name == "build_context":
            return [SSEMessage(event=E.REFERENCES.value,
                               data={"references": update.get("references") or []})]
        elif node_name == "text2sql":
            result: Optional[Text2SQLResult] = update.get("text2sql_result")
            msgs: list[SSEMessage] = []
            if result is not None:
                msgs.append(SSEMessage(event=E.SQL.value, data={
                    "sql": result.sql,
                    "columns": result.columns,
                    "row_count": result.row_count,
                    "error": result.error,
                }))
                # Text2SQL 结论不是 LLM token 流，这里切块"伪流式"产出
                for piece in _chunk_text(result.summary or "（无结论）", size=24):
                    msgs.append(SSEMessage(event=E.ANSWER_DELTA.value, data={"text": piece}))
            return msgs
        elif node_name == "generate_answer":
            # 正常情况答案已由 messages 模式逐 token 流出；若没流出(空上下文兜底话术非LLM产物)，按 answer 切块补发
            if not answer_streamed:
                return [SSEMessage(event=E.ANSWER_DELTA.value, data={"text": piece})
                        for piece in _chunk_text(update.get("answer") or "", size=24)]
        return []


def _chunk_text(text: str, size: int = 24) -> list[str]:
    """把整段文本切成固定长度的小块（用于 Text2SQL 结论的伪流式输出）。

    :param text: 待切分文本。
    :param size: 每块字符数。
    :return: 文本块列表。
    """
    if not text:
        return []
    return [text[i:i + size] for i in range(0, len(text), size)]


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：尝试编译主图、演示 _chunk_text，不真连任何 infra。
    print("[pipeline 自测] 分块 =>", _chunk_text("一二三四五六七八九十", size=4))
    try:
        g = build_pipeline_graph()
        print("[pipeline 自测] 主图编译成功 =>", type(g).__name__)
    except Exception as exc:  # noqa: BLE001
        print("[pipeline 自测] 主图编译跳过（缺 langgraph）：", exc)
