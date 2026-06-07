"""结构化查表 Agent（社保 / 商品编码）—— 对齐爱搜税 shebao_agent / product_agent。

本模块在整体链路里的位置：召回层的"结构化数据源"。查社保类/查商品编码类问题，答案是数据库表里的
【精确值】（社保缴费基数/比例、税收分类编码），不是政策长文——这类问题 RAG 只能找到"讲社保怎么算"
的文档、给不出"杭州2024基数下限4810元"。所以这里不检索文档，而是按 QU 抽取的实体(地域/商品名)去
【查 MySQL 表】，把命中的行格式化成 Document，并入文档召回结果，一起进入 粗排/精排/重排/摘要。

对齐爱搜税：爱搜税在 search_instance_mapping 里把 {"agent": "shebao_agent"} 当作一个召回步骤，
其结果与文档召回合并后统一 summarize；本模块就是它的等价实现（产物是 Document，自然并入召回池）。

健壮性（很重要）：MySQL 不可达 / 表不存在 / 没抽到实体 / 无命中 —— 一律返回 []（记日志），
由同意图配置的文档 RAG 步骤兜底，绝不抛异常打断链路。

数据源"槽位"：下方表名/字段名是合理默认，请按你数仓的真实表结构调整（这是接入点，不是硬编码业务数据）。
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from config.constants import KBase
from config.logging_config import get_logger
from config.settings import settings
from app.schemas.document import Document, QUResult

logger = get_logger(__name__)

# ============== 数据源槽位：按你数仓实际表名/字段名调整 ==============
# 社保缴费表：一行 = 某地域(某年度)的缴费基数/比例
_SHEBAO_TABLE = "dim_social_security"
_SHEBAO_REGION_COL = "region"
_SHEBAO_YEAR_COL = "year"
# 商品税收分类编码表：一行 = 某商品的编码/类别/税率
_PRODUCT_TABLE = "dim_tax_product_code"
_PRODUCT_NAME_COL = "product_name"

# 单个结构化 Agent 默认最多返回多少行（查表是精确命中，通常很少）
_DEFAULT_TOPK = 10


def _sql_str_literal(value: Any) -> str:
    """把值安全地包成 SQL 单引号字面量（转义单引号、去掉分号/反斜杠）。

    说明：MySQLClient.execute 只收原始 SQL 字符串、不暴露参数绑定，这里手工生成安全字面量。
    实体值来自正则/词典抽取(地域/商品名)，字符集受限、注入风险极低，这里仍做防御性转义。

    :param value: 待包装的值。
    :return: 形如 'xxx' 的安全 SQL 字面量。
    """
    s = "" if value is None else str(value)
    s = s.replace("\\", "").replace(";", "").replace("'", "''")
    return f"'{s}'"


class StructuredAgent:
    """结构化查表 Agent 基类：共享惰性 MySQL 客户端 + "行 -> Document"渲染。

    子类只需实现 search()：从 qu 取实体、拼 SQL、调 _query()、用 _rows_to_docs() 出结果。
    """

    # 子类覆盖：给产出的 Document 打"来源"标签(也作为 RRF 分路 / 摘要识别的 kbase)
    kbase = "structured"

    def __init__(self) -> None:
        """构造不连库（惰性）。"""
        self._mysql = None  # 惰性 MySQLClient

    def _get_mysql(self):
        """惰性获取 MySQL 客户端（首次查表才建连）。"""
        if self._mysql is None:
            from app.clients.mysql_client import MySQLClient
            self._mysql = MySQLClient()
            logger.info("[结构化Agent:%s] MySQL 客户端惰性初始化", self.kbase)
        return self._mysql

    async def search(self, qu: QUResult, topk: int = _DEFAULT_TOPK) -> list[Document]:
        """子类实现：按 qu 抽取的实体查表，返回 Document 列表。"""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # 共享工具
    # ------------------------------------------------------------------ #
    async def _query(self, sql: str) -> tuple[list[str], list[list]]:
        """执行查询；任何异常都吞掉并返回空（由文档 RAG 兜底）。

        :param sql: 只读 SELECT。
        :return: (columns, rows)；失败/无连接时返回 ([], [])。
        """
        try:
            return await self._get_mysql().execute(sql)
        except Exception as e:  # noqa: BLE001 - 查表失败不抛，降级为空
            logger.error("[结构化Agent:%s] 查表失败(由文档RAG兜底)：%s", self.kbase, e, exc_info=True)
            return [], []

    def _rows_to_docs(
        self,
        columns: list[str],
        rows: list[list],
        title_fn: Callable[[dict], str],
        query_from: str,
    ) -> list[Document]:
        """把查到的行渲染成 Document（content=可读"字段：值"，metadata=整行结构化值）。

        :param columns: 列名。
        :param rows: 数据行。
        :param title_fn: 入参为"列名->值"字典，产出该行的标题。
        :param query_from: 标记这些文档来自哪个查询(用于 RRF 分路)。
        :return: Document 列表。
        """
        docs: list[Document] = []
        for i, row in enumerate(rows):
            # 列名->值；非基本类型(Decimal/datetime 等)转成 str，保证后续 json 序列化(引用回传)不报错
            row_map: dict[str, Any] = {}
            for col, val in zip(columns, row):
                if val is None:
                    row_map[col] = ""
                elif isinstance(val, (int, float, str, bool)):
                    row_map[col] = val
                else:
                    row_map[col] = str(val)
            content = "；".join(f"{col}：{val}" for col, val in row_map.items())
            docs.append(Document(
                doc_id=f"{self.kbase}-{i}",
                title=title_fn(row_map),
                content=content,
                kbase=self.kbase,
                score=1.0,            # 结构化精确命中，给高分以在召回池中突出
                metadata=row_map,     # 整行结构化值，便于前端展示/溯源
                raw_query_from=query_from,
            ))
        return docs

    # ------------------------------------------------------------------ #
    # MCP 开关路径（可选外接层）
    # ------------------------------------------------------------------ #
    async def _search_via_mcp(
        self,
        tool_name: str,
        tool_args: dict,
        title_fn: Callable[[dict], str],
        query_from: str,
    ) -> Optional[list[Document]]:
        """当 settings.mcp_enabled=True 时，改经 MCP 调工具查表，把返回行包成 Document。

        在链路里的位置：这是"结构化查表"的【可选改道】。开关打开时，本方法去调一个 MCP server 暴露的
        税务工具（见 app/mcp/server.py）；返回的是一批结构化行（list[dict]，每个 dict=一行）。
        我们复用现有 _rows_to_docs 的渲染思路（content="字段：值"，metadata=整行）把它们包成 Document，
        与文档召回结果同构，自然并入召回池。

        为什么"返回 None"表示需要回退：
        - 把"未启用 / 调用失败 / 无命中"统一表达为 None，让子类 search() 用一个判断就回退到原直连 MySQL，
          实现优雅降级——MCP 这条可选链路绝不打断主链路。

        :param tool_name: MCP 工具名（"query_social_security" / "query_product_code"）。
        :param tool_args: 工具入参字典。
        :param title_fn: 行->标题函数（与直连路径共用，保证两条路径产物一致）。
        :param query_from: 标记这些文档来自哪个查询（用于 RRF 分路）。
        :return: Document 列表（命中且成功）；需要回退到直连时返回 None。
        """
        try:
            # 惰性导入：仅开关打开时才碰 MCP 客户端，import 本模块不牵连 mcp 依赖。
            from app.clients.mcp_client import McpClient

            logger.info("[结构化Agent:%s] MCP 路径：调用工具 %s args=%s", self.kbase, tool_name, tool_args)
            rows = await McpClient().call_tool(tool_name, tool_args)
            # call_tool 失败/未启用会返回 None；无命中可能返回 []；两种都回退到直连。
            if not rows:
                logger.info("[结构化Agent:%s] MCP 路径无结果，回退直连 MySQL", self.kbase)
                return None
            # 工具返回的每行是 dict（列名->值）；提取统一列序后复用 _rows_to_docs 渲染。
            row_dicts = [r for r in rows if isinstance(r, dict)]
            if not row_dicts:
                logger.info("[结构化Agent:%s] MCP 路径返回非预期结构，回退直连 MySQL", self.kbase)
                return None
            # 用所有行的并集列名作为 columns（不同行字段一致，这里仍稳妥取并集，保持顺序）。
            columns: list[str] = []
            for rd in row_dicts:
                for k in rd.keys():
                    if k not in columns:
                        columns.append(k)
            matrix = [[rd.get(c) for c in columns] for rd in row_dicts]
            docs = self._rows_to_docs(columns, matrix, title_fn=title_fn, query_from=query_from)
            logger.info("[结构化Agent:%s] MCP 路径命中 %d 行", self.kbase, len(docs))
            return docs
        except Exception as e:  # noqa: BLE001 - MCP 是可选链路，任何异常都回退直连
            logger.error("[结构化Agent:%s] MCP 路径异常，回退直连 MySQL：%s", self.kbase, e, exc_info=True)
            return None


class ShebaoAgent(StructuredAgent):
    """社保缴费查询 Agent：按地域(+年份)查社保基数/比例表（对齐爱搜税 shebao_agent）。"""

    kbase = KBase.SOCIAL_SECURITY.value

    async def search(self, qu: QUResult, topk: int = _DEFAULT_TOPK) -> list[Document]:
        regions = qu.entities.region or []
        year = qu.entities.year
        logger.info("[结构化Agent:社保] 进入 regions=%s year=%s", regions, year)
        # 没抽到地域就不强查（全表扫无意义），直接交回文档 RAG 兜底
        if not regions:
            logger.info("[结构化Agent:社保] 未抽到地域，跳过查表（由文档RAG兜底）")
            return []

        # MCP 开关路径：打开时优先经 MCP 调工具；失败/无命中返回 None 则回退下方直连逻辑（优雅降级）。
        if settings.mcp_enabled:
            mcp_docs = await self._search_via_mcp(
                tool_name="query_social_security",
                # 工具入参与直连查表语义对齐：取第一个地域 + 年份（工具按单地域设计）。
                tool_args={"region": regions[0], "year": year or ""},
                title_fn=lambda r: f"{r.get(_SHEBAO_REGION_COL, '')}社保缴费基数/比例",
                query_from=qu.raw_query,
            )
            if mcp_docs is not None:
                return mcp_docs

        conds = [f"{_SHEBAO_REGION_COL} IN ({', '.join(_sql_str_literal(r) for r in regions)})"]
        if year:
            conds.append(f"{_SHEBAO_YEAR_COL} = {_sql_str_literal(year)}")
        sql = f"SELECT * FROM {_SHEBAO_TABLE} WHERE {' AND '.join(conds)} LIMIT {int(topk)}"
        cols, rows = await self._query(sql)
        docs = self._rows_to_docs(
            cols, rows,
            title_fn=lambda r: f"{r.get(_SHEBAO_REGION_COL, '')}社保缴费基数/比例",
            query_from=qu.raw_query,
        )
        logger.info("[结构化Agent:社保] 命中 %d 行", len(docs))
        return docs


class ProductCodeAgent(StructuredAgent):
    """商品税收分类编码查询 Agent：按商品名查编码表（对齐爱搜税 product_agent）。"""

    kbase = KBase.PRODUCT_CODE.value

    async def search(self, qu: QUResult, topk: int = _DEFAULT_TOPK) -> list[Document]:
        names = qu.entities.product_name or []
        logger.info("[结构化Agent:商品编码] 进入 product_names=%s", names)
        if not names:
            logger.info("[结构化Agent:商品编码] 未抽到商品名，跳过查表（由文档RAG兜底）")
            return []

        # MCP 开关路径：打开时优先经 MCP 调工具；失败/无命中返回 None 则回退下方直连逻辑（优雅降级）。
        if settings.mcp_enabled:
            mcp_docs = await self._search_via_mcp(
                tool_name="query_product_code",
                # 工具按单商品名设计，取第一个商品名（直连路径仍支持多名 OR LIKE）。
                tool_args={"product_name": names[0]},
                title_fn=lambda r: f"{r.get(_PRODUCT_NAME_COL, '')} 税收分类编码",
                query_from=qu.raw_query,
            )
            if mcp_docs is not None:
                return mcp_docs

        # 商品名用 LIKE 模糊匹配（编码表里的名称未必与用户用词完全一致）
        likes = " OR ".join(
            f"{_PRODUCT_NAME_COL} LIKE {_sql_str_literal('%' + str(n) + '%')}" for n in names
        )
        sql = f"SELECT * FROM {_PRODUCT_TABLE} WHERE {likes} LIMIT {int(topk)}"
        cols, rows = await self._query(sql)
        docs = self._rows_to_docs(
            cols, rows,
            title_fn=lambda r: f"{r.get(_PRODUCT_NAME_COL, '')} 税收分类编码",
            query_from=qu.raw_query,
        )
        logger.info("[结构化Agent:商品编码] 命中 %d 行", len(docs))
        return docs


# ============== 工厂 + 惰性单例（对齐爱搜税 AgentService.func_mapping）==============
_AGENT_CLASSES: dict[str, type[StructuredAgent]] = {
    "shebao_agent": ShebaoAgent,
    "product_code_agent": ProductCodeAgent,
}
_AGENT_SINGLETONS: dict[str, StructuredAgent] = {}


def get_structured_agent(name: str) -> Optional[StructuredAgent]:
    """按名取结构化 Agent（进程内单例）；未知名返回 None。

    :param name: Agent 名（"shebao_agent" / "product_code_agent"）。
    :return: StructuredAgent 实例；未知名返回 None。
    """
    cls = _AGENT_CLASSES.get(name)
    if cls is None:
        logger.warning("[结构化Agent] 未知 Agent 名：%s", name)
        return None
    if name not in _AGENT_SINGLETONS:
        _AGENT_SINGLETONS[name] = cls()
    return _AGENT_SINGLETONS[name]


if __name__ == "__main__":
    # 自测：无 MySQL / 无实体时安全降级为 []（不报错）。
    import asyncio
    from app.schemas.document import Entities

    async def _demo():
        ag = get_structured_agent("shebao_agent")
        # 没抽到地域 -> 直接空（不查表）
        print("[structured_agents 自测] 无地域 =>", await ag.search(QUResult(raw_query="社保基数")))
        # 有地域但无 MySQL -> 查表异常降级空
        qu = QUResult(raw_query="杭州社保基数", entities=Entities(region=["杭州"]))
        print("[structured_agents 自测] 有地域无DB(降级) =>", await ag.search(qu))

    asyncio.run(_demo())
