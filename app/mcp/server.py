"""税务能力 MCP Server（用 mcp SDK 的高层 FastMCP 暴露 4 个税务工具）。
本模块在整体链路里的位置：服务端"对外能力出口"。它不实现任何新业务，而是把平台已有的能力
（3 个细粒度工具：社保查表 ShebaoAgent / 商品编码查表 ProductCodeAgent / 经营数据查询 Text2SQLAgent；
外加 1 个高层一站式工具 tax_qa：跑完整问答 pipeline 的非流式版）
用 MCP 协议【包装成标准工具】对外暴露——这样任何支持 MCP 的大模型/客户端（也包括本平台自己的
app.clients.mcp_client）都能用统一协议调用这些税务能力。

为什么这么设计（把本地能力"用 MCP 协议暴露成 server"）：
1. 解耦：上层只认 MCP 工具名 + 入参，不关心底层是查 MySQL 还是跑 LangGraph，便于将来替换实现。
2. 复用：直接调现有 agent（search / arun），不重写业务，保证与直连链路行为一致。
3. 可组合：未来可把本 server 注册进百炼等工具市场，或被其它 agent 编排调用。

健壮性（很重要）：每个工具内部一律 try/except 降级返回"空结果/带 error 的 dict"，绝不向 MCP 框架抛异常，
避免单个工具失败拖垮整个 server 会话。

惰性：模块顶层只构建 FastMCP 实例并注册工具（不连任何外部服务）；agent / MySQL / LLM 都在工具
首次被调用时才惰性初始化（沿用各 agent 自身的惰性约定）。

启动方式：
    python -m app.mcp.server          # 以 stdio 传输运行（被 MCP 客户端作为子进程拉起）
    # 或： python scripts/run_mcp_server.py
"""
from __future__ import annotations

import asyncio
from typing import Any

from config.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 【高危·无限递归防护】server 进程内强制直连，杜绝工具回连自身造成无限子进程递归。
#
# 背景：开启 MCP(stdio) 时链路是——结构化 agent → McpClient → 以子进程方式拉起
# 'python -m app.mcp.server'。本 server 的工具内部又会调 agent.search；若此子进程里
# settings.mcp_enabled 仍为 True，agent 又会走 MCP 分支再次拉起一个新的 server 子进程……
# 如此层层拉子进程，形成无限递归直至把机器拖挂。
#
# 破环办法：在【模块顶层】(import settings 后、定义工具前) 把开关强制关掉。
# 这样凡是【加载了本 server 模块的进程】(子进程 / scripts.run_mcp_server) 其工具内部调
# agent 时一律走【直连 MySQL】，绝不会再回连自身的 MCP。
# 注意：主 FastAPI 进程并不 import 本模块，故它的 mcp_enabled 不受这里影响，
# 主进程该走 MCP 仍走 MCP——仅被拉起的 server 进程内部被强制直连。
# ---------------------------------------------------------------------------
from config.settings import settings

settings.mcp_enabled = False

# ---------------------------------------------------------------------------
# 构建 FastMCP 实例。
# 用 mcp SDK 的高层封装 FastMCP："起一个名字 + @mcp.tool() 装饰函数"即可把函数暴露成 MCP 工具，
# 框架自动从类型注解/docstring 生成工具的 input_schema 与描述。
# ---------------------------------------------------------------------------
from mcp.server.fastmcp import FastMCP  # 缺 mcp 时此处抛 ImportError（由调用方/测试感知并 skip）

mcp = FastMCP("tax-tools")


@mcp.tool()
async def query_social_security(region: str, year: str = "") -> list[dict]:
    """查询某地域(可选某年度)的社保缴费基数/比例。

    在链路里的位置：把 ShebaoAgent 的查表能力暴露成 MCP 工具。内部构造一个最小 QUResult
    （只填地域/年份实体）喂给 agent.search，再把命中的每一行结构化值（doc.metadata）作为一条结果返回。

    :param region: 地域名，如 "杭州"（必填；为空时 ShebaoAgent 会跳过查表返回空）。
    :param year: 年度，如 "2024"（可选，留空表示不限年度）。
    :return: 命中行的结构化字段列表 list[dict]（每个 dict = 一行"列名->值"）；无命中/失败返回 []。
    """
    logger.info("[MCP-Server] query_social_security region=%s year=%s", region, year)
    try:
        # 惰性导入：只有工具被调用时才拉起 agent 相关依赖，import server 本身不连库。
        from app.agents.structured_agents import get_structured_agent
        from app.schemas.document import QUResult, Entities

        agent = get_structured_agent("shebao_agent")
        if agent is None:
            logger.error("[MCP-Server] 取不到 shebao_agent，降级返回空")
            return []
        # 构造最小 QU：把工具入参塞进实体，复用 agent 既有的"按实体查表"逻辑。
        qu = QUResult(
            raw_query=f"{region}社保缴费基数比例",
            entities=Entities(region=[region] if region else [], year=year or None),
        )
        docs = await agent.search(qu)
        # 对外只回结构化行（metadata），避免把内部 Document 结构泄漏给 MCP 调用方。
        rows = [doc.metadata for doc in docs]
        logger.info("[MCP-Server] query_social_security 命中 %d 行", len(rows))
        return rows
    except Exception as e:  # noqa: BLE001 - 工具内绝不抛，降级返回空
        logger.error("[MCP-Server] query_social_security 失败：%s", e, exc_info=True)
        return []


@mcp.tool()
async def query_product_code(product_name: str) -> list[dict]:
    """按商品名查询税收分类编码。

    在链路里的位置：把 ProductCodeAgent 的查表能力暴露成 MCP 工具，逻辑与社保工具同构。

    :param product_name: 商品名，如 "笔记本电脑"（必填；为空时 agent 会跳过查表返回空）。
    :return: 命中行的结构化字段列表 list[dict]；无命中/失败返回 []。
    """
    logger.info("[MCP-Server] query_product_code product_name=%s", product_name)
    try:
        from app.agents.structured_agents import get_structured_agent
        from app.schemas.document import QUResult, Entities

        agent = get_structured_agent("product_code_agent")
        if agent is None:
            logger.error("[MCP-Server] 取不到 product_code_agent，降级返回空")
            return []
        qu = QUResult(
            raw_query=f"{product_name}税收分类编码",
            entities=Entities(product_name=[product_name] if product_name else []),
        )
        docs = await agent.search(qu)
        rows = [doc.metadata for doc in docs]
        logger.info("[MCP-Server] query_product_code 命中 %d 行", len(rows))
        return rows
    except Exception as e:  # noqa: BLE001 - 工具内绝不抛，降级返回空
        logger.error("[MCP-Server] query_product_code 失败：%s", e, exc_info=True)
        return []


@mcp.tool()
async def query_business_data(question: str) -> dict:
    """用自然语言查询经营/财税数据（Text2SQL）。

    在链路里的位置：把 Text2SQLAgent（NL->SQL->查数仓->自然语言结论）暴露成 MCP 工具。
    内部直接调 agent.arun，把结构化结果整理成对外友好的 dict。

    :param question: 自然语言问题，如 "2026年Q1总订单金额"。
    :return: dict(sql, columns, rows, summary, error)；任何失败都收敛进 error，不抛异常。
    """
    logger.info("[MCP-Server] query_business_data question=%s", question)
    try:
        from app.agents.text2sql_agent import Text2SQLAgent
        from app.schemas.document import QUResult

        agent = Text2SQLAgent()
        # Text2SQL 需要一个 QUResult（其实体可辅助 schema linking，这里给最小可用值）。
        result = await agent.arun(question, QUResult(raw_query=question))
        out = {
            "sql": result.sql,
            "columns": result.columns,
            "rows": result.rows,
            "summary": result.summary,
            "error": result.error,
        }
        logger.info("[MCP-Server] query_business_data 完成 row_count=%d 有错误=%s",
                    result.row_count, bool(result.error))
        return out
    except Exception as e:  # noqa: BLE001 - 工具内绝不抛，降级为带 error 的结果
        logger.error("[MCP-Server] query_business_data 失败：%s", e, exc_info=True)
        return {"sql": "", "columns": [], "rows": [], "summary": f"数据查询失败：{e}", "error": str(e)}


# 高层一站式问答工具的强制护栏常量
_TAX_QA_TIMEOUT_S = 30.0  # 硬超时：完整 pipeline 跑太久就降级，绝不无限等
_TAX_QA_DISCLAIMER = "AI生成，仅供参考，以官方口径为准"  # answer 强制附带的免责声明
# references 对外白名单字段：只透出引用编号/文档标识/标题/来源等"展示级" metadata，
# 不泄漏内部 Document 全量结构（content/片段/image_keys/向量得分明细等一律不出）。
_REFERENCE_WHITELIST = ("citation_index", "doc_id", "title", "kbase", "metadata")


def _sanitize_references(references: list) -> list[dict]:
    """把 pipeline 产出的 references 过滤成"只含白名单字段"的对外引用列表。

    背景：build_context 产出的 ref 里混有 content/片段、image_keys、image_urls、综合得分等
    内部结构，直接外泄既冗余又可能暴露实现细节。这里按 _REFERENCE_WHITELIST 做"投影"，
    把 kbase 作为"来源"暴露，其余一律丢弃。

    :param references: pipeline 最终态里的原始引用列表（list[dict]，可能为 None）。
    :return: 清洗后的引用列表 list[dict]，每条只含白名单字段；输入异常时降级为 []。
    """
    out: list[dict] = []
    for ref in references or []:
        if not isinstance(ref, dict):
            continue
        clean = {k: ref[k] for k in _REFERENCE_WHITELIST if k in ref}
        out.append(clean)
    return out


async def _run_pipeline_once(question: str, user_id: str, session_id: str) -> dict:
    """复用现有【编译好的 LangGraph 主图】跑一次完整 pipeline 的【非流式版】，收集最终态。

    关键约束（与流式编排同源同跑、绝不另写一套）：
    - 直接 build_pipeline_graph() 编译出与 Orchestrator 同一张图，再 graph.ainvoke(init_state)；
      不复制任何逐节点驱动逻辑，不碰图拓扑/节点名。
    - init_state 字段与 Orchestrator.astream 完全一致（query/user_id/session_id/top_k/filters），
      让非流式与流式行为对齐。

    :param question: 用户自然语言问题。
    :param user_id: 用户标识（用于多轮记忆，可空）。
    :param session_id: 会话标识（用于多轮记忆，可空）。
    :return: 从最终态投影出的对外 dict（answer/references/intent/route_type/sql/disclaimer）。
    """
    # 惰性导入：import server 模块本身不应触发 langgraph/图编译。
    from app.graph.pipeline import build_pipeline_graph
    from config.settings import settings as _settings

    graph = build_pipeline_graph()
    # init_state 与 Orchestrator.astream 同口径；top_k 缺省回退 settings.rerank_topk。
    init_state: dict[str, Any] = {
        "query": question,
        "user_id": user_id or "anonymous",
        "session_id": session_id or "default",
        "top_k": _settings.rerank_topk,
        "filters": {},
    }
    final_state = await graph.ainvoke(init_state)

    # 从最终态投影出对外字段（节点名/字段名均与 state.py 约定一致，不臆造）。
    qu = final_state.get("qu")
    intent = getattr(qu, "intent", "") if qu is not None else ""
    route_type = final_state.get("route_type", "") or ""
    answer = final_state.get("answer", "") or ""
    references = _sanitize_references(final_state.get("references"))

    out: dict[str, Any] = {
        "answer": answer,
        "references": references,
        "intent": intent,
        "route_type": route_type,
        "disclaimer": _TAX_QA_DISCLAIMER,
        "error": "",
    }
    # Text2SQL 分支才有 SQL：把生成的 SQL 一并透出（可选字段）。
    t2s = final_state.get("text2sql_result")
    if t2s is not None:
        out["sql"] = getattr(t2s, "sql", "") or ""
    return out


@mcp.tool()
async def tax_qa(question: str, user_id: str = "", session_id: str = "") -> dict:
    """税务【高层一站式】智能问答（跑完整 pipeline 的非流式版，一次性返回最终答案）。

    在链路里的位置：把平台【已编译的 LangGraph 主图】（确定性 IntentClassifier+SearchRouter 路由 ->
    RAG 召回/粗排/精排/重排/拼上下文/生成答案 或 Text2SQL）整条链路非流式跑完，
    收集最终态并投影成对外 dict。它【复用】build_pipeline_graph() 编译出的同一张图（graph.ainvoke），
    绝不另写一套 pipeline、绝不改图拓扑/节点名。

    与 query_business_data 的边界（调用方据此选工具）：
    - 纯【数据类】问题（查经营/财税明细、做聚合统计，需要确切 SQL/行数）优先用 query_business_data；
    - 需要法规解读、政策汇集、社保/编码、或问题类型不确定时，用本工具 tax_qa 一站式问答
      （它内部会按确定性意图自动路由到 RAG 或 Text2SQL）。

    重要约束（本工具内部一律直连、不得回调任何 MCP）：
    本工具运行所在的 server 进程已在模块顶层强制 settings.mcp_enabled=False，pipeline 内部各
    结构化 Agent 因此一律走【直连 MySQL】，绝不会再经 McpClient 回连本 server，从根上杜绝
    tax_qa -> pipeline -> 结构化 Agent -> McpClient -> 再拉起 server 的更深层自递归。

    :param question: 用户自然语言问题（必填；为空时下游会返回空答案）。
    :param user_id: 用户标识（可选，用于加载/写回多轮会话记忆）。
    :param session_id: 会话标识（可选，同上）。
    :return: dict(answer, references, intent, route_type, sql?, disclaimer, error)；
             references 仅含白名单字段；answer 附 disclaimer；超时/异常都收敛进 error，绝不抛。
    """
    logger.info("[MCP-Server] tax_qa question=%s user_id=%s session_id=%s",
                question, user_id, session_id)
    # 护栏(1)：入口断言本进程已强制直连，确保 pipeline 内部不会再回连 MCP 形成自递归。
    assert settings.mcp_enabled is False, "tax_qa 必须运行在已强制直连(mcp_enabled=False)的 server 进程内"

    # 超时/异常时的统一降级骨架（保证返回结构稳定）。
    fallback: dict[str, Any] = {
        "answer": "", "references": [], "intent": "", "route_type": "",
        "disclaimer": _TAX_QA_DISCLAIMER, "error": "",
    }
    try:
        # 护栏(2)：用 asyncio.wait_for 包硬超时；超时只降级不抛。
        result = await asyncio.wait_for(
            _run_pipeline_once(question, user_id, session_id),
            timeout=_TAX_QA_TIMEOUT_S,
        )
        logger.info("[MCP-Server] tax_qa 完成 intent=%s route_type=%s 引用数=%d",
                    result.get("intent"), result.get("route_type"),
                    len(result.get("references") or []))
        return result
    except asyncio.TimeoutError:
        logger.error("[MCP-Server] tax_qa 超时(%.0fs)，降级返回", _TAX_QA_TIMEOUT_S)
        return {**fallback, "error": f"超时（超过{int(_TAX_QA_TIMEOUT_S)}秒）"}
    except Exception as e:  # noqa: BLE001 - 工具内绝不抛，任何异常都收敛进 error
        logger.error("[MCP-Server] tax_qa 失败：%s", e, exc_info=True)
        return {**fallback, "error": str(e)}


if __name__ == "__main__":
    # 以 stdio 传输启动 MCP server：进程通过标准输入/输出与 MCP 客户端通信，
    # 适合被客户端（如 app.clients.mcp_client 的 stdio 分支）作为子进程拉起。
    logger.info("[MCP-Server] 启动 tax-tools（stdio 传输）")
    mcp.run()
