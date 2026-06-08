"""LangGraph 主链路节点定义。
本模块在整体链路里的位置：把 QU/路由/召回/粗排/精排/重排/摘要/Text2SQL/记忆 这些"能力组件"
封装成一个个标准 LangGraph 节点（async 函数：读 GraphState 的几个字段 -> 调组件 -> 写回几个字段）。
pipeline.py 再把这些节点连成图。

设计要点（为什么这么做）：
1. 节点只做"编排粘合"，不写业务细节——业务在各 core 组件里。节点保持薄，便于阅读整条链路。
2. 所有重组件（QueryUnderstanding/SearchRouter/各 Recaller/Ranker/ReRanker/Summarizer/
   Text2SQLAgent/SessionMemory）在模块级【惰性单例】构造：构造函数本身不连 infra，真正连接发生在
   首次调用时，保证 import 本模块不触发任何网络连接。
3. 每个节点入口/产出都有 INFO 日志（命中意图、召回数、排序数、SQL 等关键节点），便于链路观测。
4. 健壮性：召回/排序等外部依赖出错时，节点降级（写空列表/记 error 日志），不让单点故障打断全图。

风格对标 掌柜问数 LangGraph 节点（async def node(state)->dict 写回片段）。
"""
from __future__ import annotations

import math
from typing import Optional

from config.constants import Intent, RouteType
from config.logging_config import get_logger
from config.settings import settings
from app.graph.state import GraphState
from app.schemas.document import QUResult, RetrievalPlan

logger = get_logger(__name__)

# 联网兜底召回的"本地召回过少"阈值：本地召回结果数 <= 此值时才考虑联网补充。
# （联网是兜底来源，本地召回够多时不触网，省 token / 避免稀释权威召回。）
_WEB_RECALL_MIN_LOCAL = 2
# 允许触发"联网兜底召回"的意图白名单（确定性，不由 LLM 决定）：
# 仅通用问答类 / 政策汇集类——这两类信息开放、时效性强、可用公开网页补充；
# 精确法规/稽查案例/查社保/查商品编码等强权威或结构化意图绝不联网兜底。
_WEB_RECALL_INTENTS = frozenset({Intent.GENERAL_QA.value, Intent.POLICY_COLLECTION.value})

# ---------------------------------------------------------------------- #
# 惰性单例：组件构造不连 infra，首次调用才连。各组件由各自模块负责惰性连接。
# ---------------------------------------------------------------------- #
_qu_engine = None
_router = None
_recall_manager = None
_coarse_ranker = None
_fine_ranker = None
_reranker = None
_summarizer = None
_t2s_agent = None
_memory = None


def _get_qu():
    """惰性获取 QueryUnderstanding（意图分类+实体抽取+查询改写编排）。"""
    global _qu_engine
    if _qu_engine is None:
        from app.core.qu.understanding import QueryUnderstanding
        _qu_engine = QueryUnderstanding()
    return _qu_engine


def _get_router():
    """惰性获取 SearchRouter（意图->检索计划路由表）。"""
    global _router
    if _router is None:
        from app.core.router.search_router import SearchRouter
        _router = SearchRouter()
    return _router


def _get_recall_manager():
    """惰性获取 RecallManager（遍历 step.kbases 并行召回合并）。"""
    global _recall_manager
    if _recall_manager is None:
        from app.core.recall.manager import RecallManager
        _recall_manager = RecallManager()
    return _recall_manager


def _get_coarse_ranker():
    """惰性获取 CoarseRanker（多路 RRF 融合 + 去重）。"""
    global _coarse_ranker
    if _coarse_ranker is None:
        from app.core.rank.coarse_rank import CoarseRanker
        _coarse_ranker = CoarseRanker()
    return _coarse_ranker


def _get_fine_ranker():
    """惰性获取 FineRanker（按标题/正文做语义精排）。"""
    global _fine_ranker
    if _fine_ranker is None:
        from app.core.rank.fine_rank import FineRanker
        _fine_ranker = FineRanker()
    return _fine_ranker


def _get_reranker():
    """惰性获取 ReRanker（调 RerankerClient 做交叉编码重排）。"""
    global _reranker
    if _reranker is None:
        from app.core.rerank.rerank import ReRanker
        _reranker = ReRanker()
    return _reranker


def _get_summarizer():
    """惰性获取 Summarizer（意图化上下文拼装 + 流式摘要）。"""
    global _summarizer
    if _summarizer is None:
        from app.core.summarize.summarizer import Summarizer
        _summarizer = Summarizer()
    return _summarizer


def _get_t2s_agent():
    """惰性获取 Text2SQLAgent。"""
    global _t2s_agent
    if _t2s_agent is None:
        from app.agents.text2sql_agent import Text2SQLAgent
        _t2s_agent = Text2SQLAgent()
    return _t2s_agent


def _get_memory():
    """惰性获取 SessionMemory。"""
    global _memory
    if _memory is None:
        from app.memory.session import SessionMemory
        _memory = SessionMemory()
    return _memory


# ---------------------------------------------------------------------- #
# 节点函数（均为 async，签名 node(state)->dict 写回片段）
# ---------------------------------------------------------------------- #
async def load_memory(state: GraphState) -> dict:
    """节点：加载多轮会话历史，写入 state.history。

    :param state: 全局状态（读 user_id/session_id）。
    :return: {"history": [...]}（出错时由 SessionMemory 内部降级为空列表）。
    """
    user_id = state.get("user_id", "anonymous")
    session_id = state.get("session_id", "default")
    logger.info("[节点:load_memory] 进入 user_id=%s session_id=%s", user_id, session_id)
    history = await _get_memory().load(user_id, session_id)
    return {"history": history}


async def understand(state: GraphState) -> dict:
    """节点：Query Understanding —— 意图分类 + 实体抽取 + 查询改写/HyDE。

    :param state: 全局状态（读 query）。
    :return: {"qu": QUResult, "route_type": ...}（route_type 先占位，下个节点 route 再定）。
    """
    query = state.get("query", "")
    logger.info("[节点:understand] 进入 query=%s", query)
    try:
        qu: QUResult = await _maybe_await(_get_qu().understand(query))
    except Exception as e:  # noqa: BLE001 QU 失败时降级为"默认意图+原始query"
        logger.error("[节点:understand] QU 失败，降级为默认意图：%s", e, exc_info=True)
        qu = QUResult(raw_query=query, sub_queries=[query] if query else [])
    logger.info("[节点:understand] 命中意图=%s 子查询数=%d", qu.intent, len(qu.sub_queries))
    return {"qu": qu}


async def route(state: GraphState) -> dict:
    """节点：根据意图产出检索计划（RAG 多步 / Text2SQL），写入 state.plan 与 route_type。

    :param state: 全局状态（读 qu）。
    :return: {"plan": RetrievalPlan, "route_type": ...}。
    """
    qu: QUResult = state["qu"]
    logger.info("[节点:route] 进入 intent=%s", qu.intent)
    try:
        plan: RetrievalPlan = _get_router().route(qu)
    except Exception as e:  # noqa: BLE001 路由失败默认走 RAG 空计划，避免全图崩
        logger.error("[节点:route] 路由失败，降级为空 RAG 计划：%s", e, exc_info=True)
        plan = RetrievalPlan(route_type=RouteType.RAG.value, steps=[])
    logger.info("[节点:route] 路由类型=%s 召回步骤数=%d", plan.route_type, len(plan.steps))
    return {"plan": plan, "route_type": plan.route_type}


async def rag_recall(state: GraphState) -> dict:
    """节点：执行检索计划的召回，合并候选文档，写入 state.recalled。

    取数策略（方案C，对齐爱搜税的"结构化优先"）：
    - 若本意图配了结构化查表 Agent（plan.agent_name，社保/商品编码）：【优先】查表，
      命中就直接用查表结果（跳过文档召回——表里是权威精确值，不需要再混文档）；
      仅当查表为空（无表/无命中/未抽到实体）才【兜底】走文档 RAG。
    - 否则（普通意图，无结构化 Agent）：正常执行 plan.steps 的文档召回。

    :param state: 全局状态（读 qu/plan）。
    :return: {"recalled": list[Document]}。
    """
    qu: QUResult = state["qu"]
    plan: RetrievalPlan = state["plan"]
    structured_agent_name = getattr(plan, "agent_name", None)
    logger.info("[节点:rag_recall] 进入，步骤数=%d，结构化Agent=%s",
                len(plan.steps), structured_agent_name or "无")

    # ---- 优先：结构化查表 Agent（命中即用，跳过文档召回）----
    if structured_agent_name:
        adocs = await _run_structured_agent(structured_agent_name, qu)
        if adocs:
            logger.info("[节点:rag_recall] 结构化Agent %s 命中 %d 条 -> 优先采用，跳过文档召回",
                        structured_agent_name, len(adocs))
            return {"recalled": adocs}
        logger.info("[节点:rag_recall] 结构化Agent %s 无命中 -> 回退文档RAG兜底",
                    structured_agent_name)

    # ---- 兜底/常规：文档 RAG 召回 ----
    recalled: list = []
    manager = _get_recall_manager()
    for i, step in enumerate(plan.steps):
        try:
            docs = await _maybe_await(manager.recall(qu, step))
            logger.info("[节点:rag_recall] 第%d步召回 %d 条（kbases=%s）",
                        i + 1, len(docs or []), step.kbases)
            recalled.extend(docs or [])
        except Exception as e:  # noqa: BLE001 单步召回失败不影响其它步
            logger.error("[节点:rag_recall] 第%d步召回失败：%s", i + 1, e, exc_info=True)

    # ---- 兜底之兜底：联网搜索补充召回（隔离·标未核验·默认关）----
    # 纯加法：仅当【总开关开】且【本地召回太少】且【意图属于"可联网兜底"白名单】三者同时满足，
    # 才追加一段联网结果（带 kbase=web 标记，在 summarizer 里物理隔离、绝不进权威引用）。
    # 默认关(settings.websearch_mcp_enabled=False) => 整段直接跳过，默认行为完全不变。
    recalled = await _maybe_append_web_recall(qu, recalled)

    logger.info("[节点:rag_recall] 召回合计 %d 条", len(recalled))
    return {"recalled": recalled}


async def coarse_rank(state: GraphState) -> dict:
    """节点：多路 RRF 融合 + 按 doc_id 去重的粗排，写入 state.ranked。

    :param state: 全局状态（读 recalled/top_k 与 plan 推断 topk）。
    :return: {"ranked": list[Document]}。
    """
    recalled = state.get("recalled", []) or []
    # 粗排 topk 取计划里首个 step 的 coarse_topk，缺省回退到 settings 默认
    topk = _infer_coarse_topk(state)
    logger.info("[节点:coarse_rank] 进入，输入 %d 条，topk=%d", len(recalled), topk)
    try:
        ranked = _get_coarse_ranker().rank(recalled, topk)
    except Exception as e:  # noqa: BLE001
        logger.error("[节点:coarse_rank] 粗排失败，透传原列表：%s", e, exc_info=True)
        ranked = recalled[:topk]
    logger.info("[节点:coarse_rank] 粗排输出 %d 条", len(ranked))
    return {"ranked": ranked}


async def fine_rank(state: GraphState) -> dict:
    """节点：基于标题/正文的语义精排，覆盖写 state.ranked。

    :param state: 全局状态（读 query/ranked/plan）。
    :return: {"ranked": list[Document]}（精排后）。
    """
    query = state.get("query", "")
    docs = state.get("ranked", []) or []
    method, topk = _infer_fine_params(state)
    logger.info("[节点:fine_rank] 进入，输入 %d 条，method=%s topk=%d", len(docs), method, topk)
    try:
        ranked = _get_fine_ranker().rank(query, docs, method, topk)
    except Exception as e:  # noqa: BLE001
        logger.error("[节点:fine_rank] 精排失败，透传原列表：%s", e, exc_info=True)
        ranked = docs[:topk]
    logger.info("[节点:fine_rank] 精排输出 %d 条", len(ranked))
    return {"ranked": ranked}


async def rerank(state: GraphState) -> dict:
    """节点：交叉编码重排（调 RerankerClient），写入 state.reranked。

    :param state: 全局状态（读 query/ranked/top_k）。
    :return: {"reranked": list[Document]}。
    """
    query = state.get("query", "")
    docs = state.get("ranked", []) or []
    # 重排 topk：优先用 req.top_k（调用方可覆盖），否则回退 settings.rerank_topk
    # （与 _infer_coarse_topk/_infer_fine_params 回退 settings 的范式一致，让 RERANK_TOPK 真正生效）
    topk = int(state.get("top_k") or settings.rerank_topk)
    logger.info("[节点:rerank] 进入，输入 %d 条，topk=%d", len(docs), topk)
    try:
        reranked = _get_reranker().rerank(query, docs, topk)
    except Exception as e:  # noqa: BLE001 重排失败回退到精排结果，保证有内容可答
        logger.error("[节点:rerank] 重排失败，回退精排结果：%s", e, exc_info=True)
        reranked = docs[:topk]
    logger.info("[节点:rerank] 重排输出 %d 条", len(reranked))
    return {"reranked": reranked}


async def answerability_check(state: GraphState) -> dict:
    """节点（RAG分支·重排后·生成前）：可答性门控——用重排分判断"够不够答"，不够就不把弱证据喂给 LLM。

    动机（对标企业财税客服的「问答库可答判断」环节）：重排(cross-encoder)已对候选给出 [0,1] 相关性分。
    若 top1 仍很低，说明召回里没有真正相关的权威依据；此时若照常把弱证据喂给 LLM 生成，极易"看图说话"
    产生幻觉。这里在生成前加一道【确定性阈值门控】（与本系统"确定性路由层"一脉相承，不引入 LLM 裁决）：
      - top1 分 >= 阈值 -> 判"可答" -> 条件边走 build_context -> generate_answer（原链路不变）；
      - top1 分 <  阈值 -> 判"不可答" -> 条件边走 low_confidence_answer 给诚实兜底（不调 LLM、零编造）。

    信号与降级（健壮性，宁放行不误杀）：
      - 主信号：重排候选里最高的 rerank_score（reranker_client 已 sigmoid 归一到 [0,1]）。
      - 重排降级兜底：重排客户端异常时 ReRanker 会回退为"按上一阶段 score 排序"且【不写 rerank_score】
        （全为默认 0.0）。此时分数不可信 -> 改用"候选数 >= answerability_min_docs"这一弱信号判定，
        避免因重排服务抖动把本可回答的问题误杀成不可答。
      - 空候选：直接判不可答（empty_recall）。
      - 总开关 answerability_enabled=False：整段跳过、恒判可答，行为与未引入门控时完全一致。

    :param state: 全局状态（读 reranked / qu）。
    :return: {"answerable": bool, "answer_confidence": float, "answerability": dict}。
    """
    docs = state.get("reranked", []) or []
    intent = getattr(state.get("qu", None), "intent", None)

    # 总开关关：恒判可答，完全保留未引入门控时的原行为（空上下文仍由 generate_answer 内部兜底）
    if not settings.answerability_enabled:
        logger.info("[节点:answerability_check] 门控关闭 -> 恒判可答（行为同未引入门控）")
        return {"answerable": True, "answer_confidence": 1.0,
                "answerability": {"enabled": False, "intent": intent}}

    doc_count = len(docs)
    # 主信号：最高重排分。reranked 已按 rerank_score 降序，但用 max 更稳（不依赖排序假设）。
    # 收敛 NaN/inf 为 0.0：避免异常分数污染 max 与置信度（正常 sigmoid 路径不会触发，纯防御）。
    scores = []
    for d in docs:
        v = float(getattr(d, "rerank_score", 0.0))
        scores.append(v if math.isfinite(v) else 0.0)
    max_score = max(scores, default=0.0)
    # 重排降级探测：有候选但最高分仍 <= 0 -> rerank_score 不可信（重排服务回退/未真正打分）。
    # 注：ReRanker 成功打分会把结果钳到 _SCORED_FLOOR(>0)，故"恰好 0.0"只可能来自"未打分"，
    # 不会把极强不相关的真低分误判成降级（详见 app/core/rerank/rerank.py 的 _SCORED_FLOOR）。
    rerank_degraded = doc_count > 0 and max_score <= 0.0

    if doc_count == 0:
        answerable, confidence, reason = False, 0.0, "empty_recall"
    elif rerank_degraded:
        # 分数不可信 -> 用候选数兜底判定，宁放行不误杀
        answerable, confidence, reason = (
            doc_count >= settings.answerability_min_docs, 0.0, "rerank_degraded_doc_count")
    else:
        answerable, confidence, reason = (
            max_score >= settings.answerability_min_score, max_score, "rerank_score")

    signals = {
        "enabled": True,
        "intent": intent,
        "doc_count": doc_count,
        "top_rerank_score": round(max_score, 4),
        "threshold": settings.answerability_min_score,
        "rerank_degraded": rerank_degraded,
        "reason": reason,
    }
    logger.info("[节点:answerability_check] 可答=%s 置信=%.4f 候选=%d 依据=%s（阈值=%.2f）",
                answerable, confidence, doc_count, reason, settings.answerability_min_score)
    return {"answerable": bool(answerable), "answer_confidence": round(float(confidence), 4),
            "answerability": signals}


async def low_confidence_answer(state: GraphState) -> dict:
    """节点（RAG分支·可答性门控判"不可答"时的终点）：给出诚实兜底话术，不调 LLM、零编造。

    为什么不调 LLM：既然判为"召回里没有足够相关的权威依据"，再让 LLM 基于弱证据生成只会冒充权威、
    制造幻觉。这里直接产出一句诚实、可执行的引导话术（建议补充文号/政策名/地域），既守住"不编造"红线，
    又给用户下一步指引；零 token、零外部依赖。措辞按门控依据区分"空召回"与"弱相关"两种情形。

    :param state: 全局状态（读 answerability 信号里的 reason，决定措辞）。
    :return: {"answer": str}。
    """
    signals = state.get("answerability", {}) or {}
    reason = signals.get("reason", "")
    if reason == "empty_recall":
        answer = ("抱歉，未检索到与您问题直接相关的资料，暂时无法给出准确回答。"
                  "建议补充关键信息（如具体文号、政策名称、所属地域）后再试。")
    else:
        answer = ("抱歉，仅检索到与您问题相关性较低的资料，为避免给出不准确的回答，这里不做硬性作答。"
                  "建议补充关键信息（如具体文号、政策名称、所属地域），或换一种更具体的问法后再试。")
    logger.info("[节点:low_confidence_answer] 触发诚实兜底（reason=%s）", reason or "(空)")
    return {"answer": answer}


async def build_context(state: GraphState) -> dict:
    """节点：按意图把重排文档拼成上下文并产出引用，写入 state.context / state.references。

    :param state: 全局状态（读 qu/reranked/top_k）。
    :return: {"context": str, "references": list[dict]}。
    """
    qu: QUResult = state["qu"]
    docs = state.get("reranked", []) or []
    # 上下文条数与重排保持同一口径：优先 req.top_k，否则回退 settings.rerank_topk
    topk = int(state.get("top_k") or settings.rerank_topk)
    logger.info("[节点:build_context] 进入，文档 %d 条，topk=%d", len(docs), topk)
    context, references = _get_summarizer().build_context(qu, docs, topk)
    logger.info("[节点:build_context] 上下文 %d 字符，引用 %d 条", len(context), len(references))
    return {"context": context, "references": references}


async def generate_answer(state: GraphState) -> dict:
    """节点（RAG分支）：基于上下文生成最终答案，写入 state.answer。

    内部用 summarize_stream（底层 llm.astream）逐 token 生成——这些 token 会被 LangGraph 的
    "messages" 流模式实时透出给编排层（转成 SSE answer_delta，实现真·逐 token 流式）；
    本节点把它们累积成完整答案写回 state（供 save_memory 落库 / 空上下文兜底）。

    :param state: 全局状态（读 qu/query/context）。
    :return: {"answer": str}。
    """
    qu: QUResult = state["qu"]
    query = state.get("query", "")
    context = state.get("context", "")
    logger.info("[节点:generate_answer] 进入，context=%d 字符", len(context))
    parts: list[str] = []
    async for delta in _get_summarizer().summarize_stream(query, qu, context):
        parts.append(delta)
    answer = "".join(parts)
    logger.info("[节点:generate_answer] 答案生成完成 %d 字符", len(answer))
    return {"answer": answer}


async def text2sql(state: GraphState) -> dict:
    """节点：Text2SQL 分支入口，按 plan.agent_name 解析并运行替换式 Agent，写入 state.text2sql_result。

    :param state: 全局状态（读 query/qu/plan）。
    :return: {"text2sql_result": Text2SQLResult}。
    """
    query = state.get("query", "")
    qu: QUResult = state["qu"]
    plan: Optional[RetrievalPlan] = state.get("plan")
    agent_name = getattr(plan, "agent_name", None)
    logger.info("[节点:text2sql] 进入 query=%s agent=%s", query, agent_name or "(默认)")
    # 真正按 plan.agent_name 解析 agent（不再硬编码）；未知名/缺省回退默认 Text2SQL agent
    agent = _resolve_pipeline_agent(agent_name)
    result = await agent.arun(query, qu)
    logger.info("[节点:text2sql] 完成，SQL=%s 行数=%d", result.sql, result.row_count)
    # 把结论同时写进 answer，供 save_memory 落库（Text2SQL 结论非 LLM 流式，由编排层切块伪流式产出）
    return {"text2sql_result": result, "answer": result.summary or ""}


async def save_memory(state: GraphState) -> dict:
    """节点：把本轮问答（用户问题 + 助手答案）写回会话记忆。

    :param state: 全局状态（读 user_id/session_id/query/answer）。
    :return: {}（无状态变更，副作用是落库）。
    """
    user_id = state.get("user_id", "anonymous")
    session_id = state.get("session_id", "default")
    query = state.get("query", "")
    answer = state.get("answer", "")
    logger.info("[节点:save_memory] 进入 user_id=%s session_id=%s", user_id, session_id)
    memory = _get_memory()
    await memory.append(user_id, session_id, "user", query)
    if answer:
        await memory.append(user_id, session_id, "assistant", answer)
    return {}


# ---------------------------------------------------------------------- #
# 条件边判定
# ---------------------------------------------------------------------- #
def route_decider(state: GraphState) -> str:
    """条件边：route 节点后根据 route_type 分流到 RAG 链路或 Text2SQL 链路。

    :param state: 全局状态（读 route_type）。
    :return: "rag" 或 "text2sql"。
    """
    rt = state.get("route_type", RouteType.RAG.value)
    decision = "text2sql" if rt == RouteType.TEXT2SQL.value else "rag"
    logger.info("[条件边:route_decider] route_type=%s -> %s", rt, decision)
    return decision


def answerability_decider(state: GraphState) -> str:
    """条件边：answerability_check 之后，按可答性结论在 RAG 链路内二次分流。

    :param state: 全局状态（读 answerable）。
    :return: "answer"（可答 -> build_context -> generate_answer 原链路） /
             "insufficient"（不可答 -> low_confidence_answer 诚实兜底）。
    """
    answerable = bool(state.get("answerable", True))
    decision = "answer" if answerable else "insufficient"
    logger.info("[条件边:answerability_decider] answerable=%s -> %s", answerable, decision)
    return decision


# ---------------------------------------------------------------------- #
# 连接池回收：进程退出时把各 Agent 惰性持有的 MySQL 连接池优雅关闭
# ---------------------------------------------------------------------- #
async def close_all() -> None:
    """优雅回收各 Agent 惰性建立的 MySQL 连接池（仅在已建连时才关）。

    在链路里的位置：由 app/main.py 的 lifespan 在 yield 之后（应用关闭阶段）调用。
    因为所有客户端都是【惰性】持有的——只有真正查过表的 Agent 才会有非空 _mysql；
    没建连的（None）直接跳过，关闭某个池失败也不影响其它池与进程退出（逐个 try/except）。

    覆盖范围：
    - Text2SQL Agent 单例 _t2s_agent 的 _mysql；
    - structured_agents 模块里所有已实例化的结构化 Agent（_AGENT_SINGLETONS）的 _mysql。
    """
    # 汇总所有"可能持有 MySQL 连接池"的 Agent 实例（仅取已构造的单例，不触发新建）。
    agents: list = []
    if _t2s_agent is not None:
        agents.append(_t2s_agent)
    try:
        from app.agents.structured_agents import _AGENT_SINGLETONS
        agents.extend(_AGENT_SINGLETONS.values())
    except Exception as e:  # noqa: BLE001 取单例表失败不影响其它清理
        logger.error("[close_all] 读取结构化Agent单例失败：%s", e, exc_info=True)

    for agent in agents:
        mysql = getattr(agent, "_mysql", None)
        if mysql is None:
            continue  # 未建连，无需回收
        try:
            await mysql.close()
            logger.info("[close_all] 已关闭 %s 的 MySQL 连接池", type(agent).__name__)
        except Exception as e:  # noqa: BLE001 单个池关闭失败不影响其它池与退出
            logger.error("[close_all] 关闭 %s 的 MySQL 连接池失败：%s",
                         type(agent).__name__, e, exc_info=True)


# ---------------------------------------------------------------------- #
# 内部工具
# ---------------------------------------------------------------------- #
async def _run_structured_agent(agent_name: str, qu) -> list:
    """运行结构化查表 Agent（社保/商品编码），返回 Document 列表；未知名/异常/无命中均返回 []。

    :param agent_name: 结构化 Agent 名（来自 plan.agent_name）。
    :param qu: 查询理解结果（实体在 qu.entities）。
    :return: list[Document]；任何问题都降级为 []，由文档 RAG 兜底。
    """
    try:
        from app.agents.structured_agents import get_structured_agent
        agent = get_structured_agent(agent_name)
        if agent is None:
            return []
        return (await agent.search(qu)) or []
    except Exception as e:  # noqa: BLE001 结构化Agent失败不影响主流程
        logger.error("[节点:rag_recall] 结构化Agent %s 失败：%s", agent_name, e, exc_info=True)
        return []


async def _maybe_append_web_recall(qu, recalled: list) -> list:
    """联网兜底召回（隔离·标未核验·默认关）：在本地召回之后【纯加法】追加联网补充结果。

    触发条件（三者同时满足，缺一不补；任一不满足都原样返回 recalled）：
    1. settings.websearch_mcp_enabled 为真（总开关；默认 False => 整段跳过，行为不变）；
    2. 本地召回结果为空/过少（len(recalled) <= _WEB_RECALL_MIN_LOCAL）——联网只做兜底，
       本地够多时不触网，避免稀释权威召回、节省 token；
    3. 当前意图 ∈ 白名单 _WEB_RECALL_INTENTS（通用问答类 / 政策汇集类）——确定性触发，
       绝不由 LLM 决定，强权威/结构化意图绝不联网兜底。

    隔离保证：追加进来的是 WebSearchAgent 产出的 kbase=web、metadata.unverified=True 的
    Document（其内部已打"未核验·以官方为准"标记），summarizer 据此物理隔离、绝不并入
    权威法规引用([[citation:N]])。

    健壮性：任何异常/为空都降级——直接返回原 recalled，绝不打断主链路。

    :param qu: Query Understanding 结果（读 intent / raw_query）。
    :param recalled: 本地召回已合并的 Document 列表。
    :return: 追加联网结果后的列表（不满足条件或失败时即原列表）。
    """
    # 条件1：总开关（默认关 => 直接返回，零外部依赖、行为完全不变）
    if not settings.websearch_mcp_enabled:
        return recalled
    # 条件2：本地召回够多就不联网兜底
    if len(recalled) > _WEB_RECALL_MIN_LOCAL:
        logger.info("[节点:rag_recall] 本地召回 %d 条(>%d)，不触发联网兜底",
                    len(recalled), _WEB_RECALL_MIN_LOCAL)
        return recalled
    # 条件3：意图白名单（确定性触发，不由 LLM 决定）
    intent = getattr(qu, "intent", None)
    if intent not in _WEB_RECALL_INTENTS:
        logger.info("[节点:rag_recall] 意图=%s 不在联网兜底白名单，跳过联网兜底", intent)
        return recalled

    # 满足全部条件：调 WebSearchAgent 拿"未核验"补充结果（失败/为空即降级跳过）
    query = getattr(qu, "raw_query", "") or ""
    logger.info("[节点:rag_recall] 本地召回 %d 条(<=%d) 且意图=%s 命中白名单 -> 触发联网兜底召回",
                len(recalled), _WEB_RECALL_MIN_LOCAL, intent)
    try:
        from app.agents.web_search_agent import WebSearchAgent
        web_docs = await WebSearchAgent().search(query)
    except Exception as e:  # noqa: BLE001 联网是可选兜底链路，任何异常都降级，绝不打断主流程
        logger.error("[节点:rag_recall] 联网兜底召回失败，降级跳过：%s", e, exc_info=True)
        return recalled

    if not web_docs:
        logger.info("[节点:rag_recall] 联网兜底无补充结果，跳过")
        return recalled

    logger.info("[节点:rag_recall] 联网兜底追加 %d 条（kbase=web·未核验·物理隔离）", len(web_docs))
    return recalled + web_docs


def _resolve_pipeline_agent(name: Optional[str]):
    """route_type=text2sql 时按 plan.agent_name 解析"替换式" agent。

    目前仅 text2sql_agent；做成注册表是为了将来可扩展（如再加其它替换整条链路的 agent）。
    未知名 / None 一律回退默认 Text2SQL agent，保证分支永远有 agent 可用。

    :param name: agent 名（须与 router 的 TEXT2SQL_AGENT_NAME 一致）。
    :return: 可调用 .arun(query, qu) 的 agent 实例。
    """
    registry = {"text2sql_agent": _get_t2s_agent}
    getter = registry.get(name) or _get_t2s_agent
    return getter()


async def _maybe_await(value):
    """兼容"组件方法既可能是同步、也可能是异步"的情况：是协程则 await，否则原样返回。

    为什么需要：契约中 understand/recall 的返回有的标了 async、有的没标，为防止两种实现都能跑通，
    这里统一做一次"协程探测"。

    :param value: 可能是普通值，也可能是 awaitable。
    :return: await 后的结果（若不是协程则原值）。
    """
    import inspect
    if inspect.isawaitable(value):
        return await value
    return value


def _infer_coarse_topk(state: GraphState) -> int:
    """从计划首个 step 推断粗排 topk，缺省回退 settings.recall_topk。"""
    from config.settings import settings
    plan: Optional[RetrievalPlan] = state.get("plan")
    if plan and plan.steps:
        return plan.steps[0].coarse_topk
    return settings.recall_topk


def _infer_fine_params(state: GraphState) -> tuple[str, int]:
    """从计划首个 step 推断精排方式与 topk，缺省回退 (bge_content, settings.fine_topk)。"""
    from config.settings import settings
    plan: Optional[RetrievalPlan] = state.get("plan")
    if plan and plan.steps:
        step = plan.steps[0]
        return step.fine_rank_method, step.fine_topk
    return "bge_content", settings.fine_topk


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：只验证条件边判定逻辑，不触发任何组件连接。
    print("[nodes 自测] route_decider(rag) =>",
          route_decider({"route_type": RouteType.RAG.value}))
    print("[nodes 自测] route_decider(text2sql) =>",
          route_decider({"route_type": RouteType.TEXT2SQL.value}))
