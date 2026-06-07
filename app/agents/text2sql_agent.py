"""Text2SQL Agent（自然语言 -> SQL -> 查询数仓 -> 自然语言结论）。
本模块在整体链路里的位置：当顶层路由判定为"经营数据查询类(DATA_QUERY)"时，编排层不走 RAG，
而是把问题交给本 Agent。它用 LangGraph 把 Text2SQL 拆成可控的多步状态机（对标 掌柜问数）：

    schema_link（Qdrant 选相关表/字段/指标）
        -> generate_sql（LLM 据 schema + 问题生成 SQL）
        -> validate_fix（执行前静态校验；不通过则用 LLM 修，最多重试2次）
        -> execute（MySQL 真正执行，拿到列名+数据行）
        -> summarize（把结果转成自然语言结论）

为什么用 LangGraph 而非一段顺序代码：
1. SQL 生成天然需要"生成->校验->修正"的循环；用 StateGraph 的条件边表达"校验失败回到修正"
   既直观又可控（重试次数、分支一目了然），也便于将来插入更多节点（如指标过滤、JOIN 推断）。
2. 每个节点只读写 state 的一小片，单步可测、可观测（每步都有 INFO 日志）。

健壮性：所有外部依赖（Qdrant/LLM/MySQL）惰性获取 + try/except 中文报错；任何一步失败都把
错误写进 Text2SQLResult.error 并安全收尾，不让异常冒泡打断上层 SSE。

风格对标 掌柜问数 LangGraph 多步 Text2SQL，注释教学化。
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict

from config.logging_config import get_logger
from config.settings import settings
from app.schemas.document import QUResult, Text2SQLResult
from app.utils.prompt_loader import load_prompt
from app.agents.tools import (
    extract_sql,
    validate_sql_safety,
    ensure_limit,
    format_schema_context,
    rows_to_markdown,
)

logger = get_logger(__name__)

# validate_fix 失败时最多重试修正的次数（对标"最多重试2次"）。
_MAX_FIX_RETRY = 2
# schema linking 召回的元数据条数上限。
_SCHEMA_TOPK = 15

# 各 prompt 缺失时的兜底模板，保证无 prompt 文件也能跑通（仅早期/自测会走到）。
_FALLBACK_SCHEMA_LINK_PROMPT = (
    "下面是数仓可用的表与字段元数据：\n{schema}\n\n用户问题：{question}\n"
    "请判断回答该问题需要用到哪些表和字段（仅供参考）。"
)
_FALLBACK_GENERATE_PROMPT = (
    "你是资深数据分析工程师。请根据【表结构】和【用户问题】生成一条 MySQL 查询 SQL。\n"
    "要求：只输出一条 SELECT 语句，不要解释。\n\n"
    "【表结构】：\n{schema}\n\n【用户问题】：{question}\n\n【SQL】："
)
_FALLBACK_FIX_PROMPT = (
    "下面这条 SQL 校验未通过，请修正后只输出一条合法的只读 SELECT 语句，不要解释。\n"
    "【表结构】：\n{schema}\n\n【原SQL】：\n{sql}\n\n【错误原因】：{error}\n\n【修正后SQL】："
)


class T2SState(TypedDict, total=False):
    """Text2SQL 子图的内部状态（与全局 GraphState 解耦，仅本 Agent 使用）。"""
    question: str            # 用户问题（自然语言）
    qu: Any                  # QUResult，可用其实体辅助 schema linking
    schema_items: list       # schema linking 选出的表/字段/指标元数据
    schema_text: str         # 拼好的 schema 上下文文本
    sql: str                 # 当前 SQL
    error: str               # 校验/执行错误信息（None/"" 表示无错）
    fix_count: int           # 已修正次数
    give_up: bool            # 是否已放弃修正（达上限/无法修正），用于条件边导向收尾
    columns: list            # 执行结果列名
    rows: list               # 执行结果数据行
    summary: str             # 自然语言结论


class Text2SQLAgent:
    """Text2SQL 智能体：暴露编译后的 LangGraph 图(.graph) 与异步入口 arun。

    用法::

        agent = Text2SQLAgent()
        result = await agent.arun("2026年Q1总订单金额", qu)
        print(result.sql, result.summary)
    """

    def __init__(self) -> None:
        # 外部客户端全部惰性持有，import 阶段不连任何服务
        self._llm = None
        self._qdrant = None
        self._mysql = None
        self._embedding = None
        self._graph = None  # 编译后的 StateGraph，首次访问 .graph 时构建

    # ------------------------------------------------------------------ #
    # 惰性客户端
    # ------------------------------------------------------------------ #
    def _get_llm(self):
        """惰性获取 LLM 客户端。"""
        if self._llm is None:
            from app.clients.llm_client import get_llm
            self._llm = get_llm()
            logger.info("[Text2SQL] LLM 客户端惰性初始化完成")
        return self._llm

    def _get_qdrant(self):
        """惰性获取 Qdrant 元数据客户端（schema linking 用）。"""
        if self._qdrant is None:
            from app.clients.qdrant_meta_client import QdrantMetaClient
            self._qdrant = QdrantMetaClient()
            logger.info("[Text2SQL] Qdrant 元数据客户端惰性初始化完成")
        return self._qdrant

    def _get_mysql(self):
        """惰性获取 MySQL 客户端（execute 用）。"""
        if self._mysql is None:
            from app.clients.mysql_client import MySQLClient
            self._mysql = MySQLClient()
            logger.info("[Text2SQL] MySQL 客户端惰性初始化完成")
        return self._mysql

    def _get_embedding(self):
        """惰性获取 Embedding 客户端（把问题向量化以检索元数据）。"""
        if self._embedding is None:
            from app.clients.embedding_client import EmbeddingClient
            self._embedding = EmbeddingClient()
            logger.info("[Text2SQL] Embedding 客户端惰性初始化完成")
        return self._embedding

    # ------------------------------------------------------------------ #
    # 节点：schema_link
    # ------------------------------------------------------------------ #
    async def _node_schema_link(self, state: T2SState) -> dict:
        """节点1：根据问题在 Qdrant 里做语义检索，选出相关的表/字段/指标元数据。

        :param state: 子图状态（读 question/qu）。
        :return: 写回 {schema_items, schema_text}。
        """
        question = state["question"]
        logger.info("[Text2SQL][schema_link] 进入，question=%s", question)
        schema_items: list[dict] = []
        try:
            embedding = self._get_embedding()
            qdrant = self._get_qdrant()
            # 把问题向量化（embed_query 返回批量结构 {"dense":[[...]], "sparse":[{...}]}，单条放第 0 位）
            emb = embedding.embed_query(question)
            dense_list = emb.get("dense", []) if isinstance(emb, dict) else emb
            dense_vec = dense_list[0] if dense_list else []  # 取出该问题的稠密向量 list[float]
            if not dense_vec:
                logger.info("[Text2SQL][schema_link] 问题无稠密向量，跳过元数据召回")
                hits = []
            else:
                hits = qdrant.search(settings.qdrant_meta_collection, dense_vec, _SCHEMA_TOPK)
            # 命中条目的 payload 里携带表/字段/指标元数据
            for h in hits or []:
                payload = h.get("payload") or h.get("fields") or h
                if isinstance(payload, dict):
                    schema_items.append(payload)
            logger.info("[Text2SQL][schema_link] 召回元数据条数=%d", len(schema_items))
        except Exception as e:  # noqa: BLE001 schema linking 失败不致命，仍可让 LLM 凭空尝试
            logger.error("[Text2SQL][schema_link] 元数据召回失败：%s", e, exc_info=True)

        schema_text = format_schema_context(schema_items)
        # 可选：用 schema_link prompt 让 LLM 进一步过滤（此处保留 prompt 加载，作为教学示范）
        try:
            _ = self._render("text2sql_schema_link", _FALLBACK_SCHEMA_LINK_PROMPT,
                             schema=schema_text, question=question)
        except Exception:  # noqa: BLE001
            pass
        return {"schema_items": schema_items, "schema_text": schema_text}

    # ------------------------------------------------------------------ #
    # 节点：generate_sql
    # ------------------------------------------------------------------ #
    async def _node_generate_sql(self, state: T2SState) -> dict:
        """节点2：让 LLM 依据 schema 上下文 + 问题生成 SQL。

        :param state: 子图状态（读 question/schema_text）。
        :return: 写回 {sql, error, fix_count}（error 置空，fix_count 初始化为0）。
        """
        question = state["question"]
        schema_text = state.get("schema_text", "")
        logger.info("[Text2SQL][generate_sql] 进入")
        try:
            prompt = self._render("text2sql_generate", _FALLBACK_GENERATE_PROMPT,
                                  schema=schema_text, question=question)
            llm = self._get_llm()
            resp = await llm.ainvoke(prompt)
            sql = extract_sql(getattr(resp, "content", "") or "")
            logger.info("[Text2SQL][generate_sql] 生成SQL=%s", sql)
            return {"sql": sql, "error": "", "fix_count": 0}
        except Exception as e:  # noqa: BLE001
            logger.error("[Text2SQL][generate_sql] 生成失败：%s", e, exc_info=True)
            # 写 error，让后续 validate_fix 走修正/收尾分支
            return {"sql": "", "error": f"SQL生成失败：{e}", "fix_count": 0}

    # ------------------------------------------------------------------ #
    # 节点：validate_fix
    # ------------------------------------------------------------------ #
    async def _node_validate_fix(self, state: T2SState) -> dict:
        """节点3：执行前静态校验当前 SQL；不通过则用 LLM 修正一次（次数累加）。

        说明：本节点既做"校验"也做"一次修正"，配合条件边形成"校验->修正->再校验"的循环，
        直到通过或达到最大重试次数（对标掌柜问数 validate_sql/correct_sql 的组合）。

        :param state: 子图状态（读 sql/schema_text/fix_count）。
        :return: 写回 {sql, error, fix_count}；error 为空表示校验通过。
        """
        sql = state.get("sql", "")
        fix_count = state.get("fix_count", 0)
        logger.info("[Text2SQL][validate_fix] 进入，fix_count=%d sql=%s", fix_count, sql)

        ok, reason = validate_sql_safety(sql)
        if ok:
            # 通过：补 LIMIT 防全表，清空 error
            safe_sql = ensure_limit(sql)
            logger.info("[Text2SQL][validate_fix] 校验通过，最终SQL=%s", safe_sql)
            return {"sql": safe_sql, "error": ""}

        logger.info("[Text2SQL][validate_fix] 校验未通过：%s", reason)
        # 已达最大重试：置 give_up=True，由条件边导向收尾(summarize)
        if fix_count >= _MAX_FIX_RETRY:
            logger.error("[Text2SQL][validate_fix] 修正次数已达上限(%d)，放弃", _MAX_FIX_RETRY)
            return {"error": f"SQL 多次修正仍未通过：{reason}", "give_up": True}

        # 调 LLM 修正一次；修正后保持 error 非空(=reason)，让条件边回到本节点复验
        try:
            prompt = self._render(
                "text2sql_fix", _FALLBACK_FIX_PROMPT,
                schema=state.get("schema_text", ""), sql=sql, error=reason,
            )
            llm = self._get_llm()
            resp = await llm.ainvoke(prompt)
            fixed_sql = extract_sql(getattr(resp, "content", "") or "")
            logger.info("[Text2SQL][validate_fix] 第%d次修正结果=%s", fix_count + 1, fixed_sql)
            return {"sql": fixed_sql, "error": reason, "fix_count": fix_count + 1, "give_up": False}
        except Exception as e:  # noqa: BLE001
            # 修正调用本身异常：无法继续，直接放弃
            logger.error("[Text2SQL][validate_fix] 修正调用失败：%s", e, exc_info=True)
            return {"error": f"SQL 修正失败：{e}", "give_up": True}

    def _route_after_validate(self, state: T2SState) -> str:
        """条件边判定：校验通过->execute；放弃->summarize 收尾；否则->validate_fix 复验。

        判定只看两个显式信号，逻辑清晰无歧义：
        - error 为空：校验通过，去执行；
        - give_up=True：已放弃（达上限/修正异常），去收尾输出错误说明；
        - 其余：刚做了一次修正，回到 validate_fix 复验。

        :param state: 子图状态。
        :return: 下一节点名（"execute" / "summarize" / "validate_fix"）。
        """
        if not state.get("error"):
            return "execute"
        if state.get("give_up"):
            return "summarize"
        return "validate_fix"

    # ------------------------------------------------------------------ #
    # 节点：execute
    # ------------------------------------------------------------------ #
    async def _node_execute(self, state: T2SState) -> dict:
        """节点4：把校验通过的 SQL 打到 MySQL 执行，拿回列名与数据行。

        :param state: 子图状态（读 sql）。
        :return: 写回 {columns, rows, error}。
        """
        sql = state.get("sql", "")
        logger.info("[Text2SQL][execute] 执行SQL=%s", sql)
        try:
            mysql = self._get_mysql()
            columns, rows = await mysql.execute(sql)
            logger.info("[Text2SQL][execute] 执行成功，列数=%d 行数=%d",
                        len(columns or []), len(rows or []))
            return {"columns": columns or [], "rows": rows or [], "error": ""}
        except Exception as e:  # noqa: BLE001
            logger.error("[Text2SQL][execute] 执行失败：%s", e, exc_info=True)
            return {"columns": [], "rows": [], "error": f"SQL执行失败：{e}"}

    # ------------------------------------------------------------------ #
    # 节点：summarize
    # ------------------------------------------------------------------ #
    async def _node_summarize(self, state: T2SState) -> dict:
        """节点5：把查询结果（或错误）转成给用户看的自然语言结论。

        :param state: 子图状态（读 question/columns/rows/error）。
        :return: 写回 {summary}。
        """
        logger.info("[Text2SQL][summarize] 进入")
        error = state.get("error", "")
        if error:
            # 有错误：直接给出诚实的失败说明，不再调 LLM
            return {"summary": f"未能完成数据查询：{error}"}

        columns = state.get("columns", [])
        rows = state.get("rows", [])
        if not rows:
            return {"summary": "查询执行成功，但没有匹配到任何数据。"}

        table_md = rows_to_markdown(columns, rows)
        question = state["question"]
        # 用通用摘要思路：把结果表喂给 LLM 让其用自然语言概括（失败则降级为直接给表格）
        try:
            llm = self._get_llm()
            prompt = (
                "你是数据分析助手。请用简洁中文总结下面这条查询的结论，"
                "直接给关键数字与结论，不要复述SQL。\n\n"
                f"【用户问题】：{question}\n\n【查询结果】：\n{table_md}\n\n【结论】："
            )
            resp = await llm.ainvoke(prompt)
            summary = (getattr(resp, "content", "") or "").strip() or table_md
            logger.info("[Text2SQL][summarize] 结论生成完成")
            return {"summary": summary}
        except Exception as e:  # noqa: BLE001
            logger.error("[Text2SQL][summarize] 结论生成失败，降级返回表格：%s", e, exc_info=True)
            return {"summary": f"查询结果如下：\n{table_md}"}

    # ------------------------------------------------------------------ #
    # 工具：渲染 prompt（带兜底模板）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _render(name: str, fallback: str, **kwargs: Any) -> str:
        """加载并渲染指定 prompt，缺文件时用兜底模板。

        :param name: prompt 文件名（不含后缀）。
        :param fallback: 兜底模板字符串。
        :param kwargs: 模板占位符的实参。
        :return: 渲染后的 prompt 文本。
        """
        try:
            template = load_prompt(name)
        except FileNotFoundError:
            logger.error("[Text2SQL] 未找到 %s.prompt，使用内置兜底模板", name)
            template = fallback
        return template.format(**kwargs)

    # ------------------------------------------------------------------ #
    # 构图
    # ------------------------------------------------------------------ #
    def _build_graph(self):
        """用 langgraph.StateGraph 把五个节点连成图，并编译。

        图结构::

            START -> schema_link -> generate_sql -> validate_fix
            validate_fix --(通过)--> execute --> summarize -> END
            validate_fix --(需修正)--> validate_fix（最多2次）
            validate_fix --(放弃)--> summarize

        :return: 编译后的 CompiledStateGraph。
        :raise ImportError: 当 langgraph 未安装时由 import 抛出。
        """
        from langgraph.graph import StateGraph, START, END

        builder = StateGraph(T2SState)
        builder.add_node("schema_link", self._node_schema_link)
        builder.add_node("generate_sql", self._node_generate_sql)
        builder.add_node("validate_fix", self._node_validate_fix)
        builder.add_node("execute", self._node_execute)
        builder.add_node("summarize", self._node_summarize)

        builder.add_edge(START, "schema_link")
        builder.add_edge("schema_link", "generate_sql")
        builder.add_edge("generate_sql", "validate_fix")
        # 校验后条件分流：通过->执行，需修->复验(自环)，放弃->收尾
        builder.add_conditional_edges(
            "validate_fix",
            self._route_after_validate,
            {"execute": "execute", "validate_fix": "validate_fix", "summarize": "summarize"},
        )
        builder.add_edge("execute", "summarize")
        builder.add_edge("summarize", END)

        compiled = builder.compile()
        logger.info("[Text2SQL] LangGraph 子图编译完成")
        return compiled

    @property
    def graph(self):
        """惰性编译并返回 LangGraph 图（首次访问才构建，避免 import 期依赖 langgraph）。

        :return: 编译后的 LangGraph 图对象。
        """
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    # ------------------------------------------------------------------ #
    # 对外入口
    # ------------------------------------------------------------------ #
    async def arun(self, query: str, qu: QUResult) -> Text2SQLResult:
        """运行整条 Text2SQL 流程，返回结构化结果。

        :param query: 用户自然语言问题。
        :param qu: Query Understanding 结果（可携带实体辅助 schema linking）。
        :return: Text2SQLResult（含 sql/columns/rows/row_count/summary/error）。
        :raise: 不向外抛；任何异常都被收敛进 Text2SQLResult.error。
        """
        logger.info("[Text2SQL] arun 开始，query=%s", query)
        try:
            init_state: T2SState = {"question": query, "qu": qu, "fix_count": 0,
                                    "error": "", "give_up": False}
            final_state = await self.graph.ainvoke(init_state)
        except Exception as e:  # noqa: BLE001 兜底：连构图/执行都失败时也要给出可读结果
            logger.error("[Text2SQL] 流程整体异常：%s", e, exc_info=True)
            return Text2SQLResult(question=query, error=f"Text2SQL 流程异常：{e}",
                                  summary=f"数据查询失败：{e}")

        rows = final_state.get("rows", []) or []
        result = Text2SQLResult(
            question=query,
            sql=final_state.get("sql", ""),
            columns=final_state.get("columns", []) or [],
            rows=rows,
            row_count=len(rows),
            summary=final_state.get("summary", ""),
            error=final_state.get("error", "") or "",
        )
        logger.info("[Text2SQL] arun 结束，row_count=%d 有错误=%s",
                    result.row_count, bool(result.error))
        return result


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：不真连任何服务，只验证图能编译、tools 链路正确。
    agent = Text2SQLAgent()
    try:
        g = agent.graph  # 触发编译；若 langgraph 已装应成功
        print("[text2sql_agent 自测] 图编译成功 =>", type(g).__name__)
    except Exception as exc:  # noqa: BLE001
        print("[text2sql_agent 自测] 图编译跳过（缺 langgraph）：", exc)
