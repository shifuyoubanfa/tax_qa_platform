"""联网搜索 Agent（WebSearch via MCP）—— 可选补充来源，默认关、物理隔离、失败降级。

本模块在整体链路里的位置：检索层之外的"可选补充来源底座"。
注意：本阶段【只建底座，不接主链路】（下一阶段才由确定性意图触发把它并入召回）。

为什么单独成一条链路、且与本地权威法规库【物理隔离】（这是硬约束）：
1. 联网结果是公开网页摘要，未经官方核验，权威性远低于本地法规库；
2. 因此它产出的 Document 一律打 kbase=KBase.WEB、metadata.source 标注"未核验·以官方为准"，
   并带 "unverified": True 标记——绝不进权威法规引用([[citation:N]])；
3. 总开关 settings.websearch_mcp_enabled 默认 False：关时本 Agent 直接返回 []、绝不触网，
   保证无网络/无 key/无 MCP server 时整条链路照常跑；
4. 是否触发由"确定性意图"决定（下一阶段接入），绝不交给 LLM 自行决定。

健壮性（很重要）：未开启 / 找不到搜索工具 / 调用失败 / 超时 —— 一律返回 []（记中文日志），
硬超时用 asyncio.wait_for(默认10s) 兜底，绝不抛异常、绝不重试、绝不打断主链路。

接口契约（INTERFACES）：
- async search(query: str) -> list[Document]   未开启/失败/超时一律返回 []
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from config.constants import KBase
from config.logging_config import get_logger
from config.settings import settings
from app.schemas.document import Document

logger = get_logger(__name__)

# 联网搜索硬超时（秒）：超过即放弃、返回 []，避免拖慢主链路。
_SEARCH_TIMEOUT_SEC = 10.0
# 单次联网搜索默认最多取多少条结果（补充来源，不宜过多以免稀释权威召回）。
_DEFAULT_TOPK = 5
# 联网结果统一的来源标注（物理隔离的核心标记，前端/摘要据此区分"未核验补充"）。
_UNVERIFIED_SOURCE = "联网·未核验·以官方为准"


class WebSearchAgent:
    """联网搜索 Agent：经 MCP（streamable http）调一个搜索工具，把网页结果包成"未核验"Document。

    用法::

        agent = WebSearchAgent()
        docs = await agent.search("2024年小微企业所得税优惠最新政策")  # 默认关时返回 []

    :return: list[Document]（kbase=web，标注未核验）；未开启/失败/超时返回 []。
    """

    def __init__(self) -> None:
        """构造仅读配置，不连任何服务（惰性 + 降级）。"""
        self._enabled = bool(settings.websearch_mcp_enabled)
        if not self._enabled:
            logger.info("[联网搜索] 未启用（WEBSEARCH_MCP_ENABLED=False），search 直接返回 []（绝不触网）")
        else:
            logger.info("[联网搜索] 已启用：transport=%s url=%s",
                        settings.websearch_mcp_transport, settings.websearch_mcp_url)

    async def search(self, query: str) -> list[Document]:
        """对外唯一入口：联网搜索一个查询，返回"未核验"补充 Document 列表。

        :param query: 用户查询（或子查询）文本。
        :return: list[Document]；未开启 / 无 query / 失败 / 超时一律返回 []（绝不抛）。
        """
        # 默认关：直接返回 []，绝不触网（最硬的安全闸）。
        if not self._enabled:
            logger.info("[联网搜索] 未启用，返回 []")
            return []
        if not (query or "").strip():
            logger.info("[联网搜索] 空查询，返回 []")
            return []

        # 硬超时兜底：整段联网逻辑放进 wait_for，超时即放弃。
        try:
            return await asyncio.wait_for(self._search_impl(query), timeout=_SEARCH_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            logger.warning("[联网搜索] 超时(%.0fs)，降级返回 []：query=%s", _SEARCH_TIMEOUT_SEC, query)
            return []
        except Exception as e:  # noqa: BLE001 - 联网是可选链路，任何异常都降级为空，绝不打断主链路
            logger.error("[联网搜索] 异常，降级返回 []：query=%s err=%s", query, e, exc_info=True)
            return []

    # ------------------------------------------------------------------ #
    # 内部实现
    # ------------------------------------------------------------------ #
    async def _search_impl(self, query: str) -> list[Document]:
        """真正执行：list_tools 找搜索工具 -> call_tool 取结果 -> 包成未核验 Document。

        :param query: 查询文本。
        :return: list[Document]；找不到工具/无结果时返回 []。
        """
        # 惰性导入：仅开关打开时才碰 MCP 客户端，import 本模块不牵连 mcp 依赖。
        from app.clients.mcp_client import McpClient

        client = McpClient()
        # 显式传入 url/transport/api_key 覆盖参数：让 McpClient 连联网搜索专用的 server，
        # 且因传了覆盖参数而不受结构化查表总开关 settings.mcp_enabled 限制。
        url = settings.websearch_mcp_url
        transport = settings.websearch_mcp_transport or "http"
        api_key = settings.dashscope_api_key

        # 1) 列工具，找到"搜索"工具（名字或描述里含 search）。
        tools = await client.list_tools(url=url, transport=transport, api_key=api_key)
        tool_name = self._pick_search_tool(tools)
        if not tool_name:
            logger.info("[联网搜索] 未找到含 'search' 的工具，返回 []（工具数=%d）", len(tools))
            return []

        # 2) 调工具拿结果。入参用通用键 "query"（多数搜索 MCP 工具用此键；不符合时由失败降级兜底）。
        logger.info("[联网搜索] 调用搜索工具：%s query=%s", tool_name, query)
        result = await client.call_tool(
            tool_name, {"query": query}, url=url, transport=transport, api_key=api_key,
        )
        if not result:
            logger.info("[联网搜索] 搜索工具无结果，返回 []：tool=%s", tool_name)
            return []

        # 3) 把结果包成"未核验"Document。
        items = self._normalize_results(result)
        docs = self._items_to_docs(items, query)
        logger.info("[联网搜索] 命中 %d 条（已标注未核验）", len(docs))
        return docs

    @staticmethod
    def _pick_search_tool(tools: list[dict]) -> Optional[str]:
        """从工具列表里挑一个"搜索"工具：名字或描述里含 'search' 即认。

        :param tools: list_tools 的返回（每项 {"name","description","input_schema"}）。
        :return: 命中的工具名；没有则 None。
        """
        for t in tools or []:
            name = (t.get("name") or "").lower()
            desc = (t.get("description") or "").lower()
            if "search" in name or "search" in desc:
                return t.get("name")
        return None

    @staticmethod
    def _normalize_results(result: Any) -> list[dict]:
        """把搜索工具的返回归一化成 list[dict]（每个 dict = 一条网页结果）。

        搜索 MCP 工具返回形态各异，这里尽量兜底：
        - 直接是 list：逐项保留 dict；
        - 是 dict：尝试取常见列表字段(results/data/items/list/webPages)；都没有就当单条结果；
        - 其他：返回 []。

        :param result: call_tool 解析后的返回值。
        :return: list[dict]（可能为空）。
        """
        if isinstance(result, list):
            return [r for r in result if isinstance(r, dict)]
        if isinstance(result, dict):
            for key in ("results", "data", "items", "list", "webPages", "pages"):
                val = result.get(key)
                if isinstance(val, list):
                    return [r for r in val if isinstance(r, dict)]
                # 有的 API 把列表再包一层 {"value": [...]}（如 bing webPages.value）
                if isinstance(val, dict) and isinstance(val.get("value"), list):
                    return [r for r in val["value"] if isinstance(r, dict)]
            # 没有列表字段，但本身像一条结果，就当单条
            return [result]
        return []

    def _items_to_docs(self, items: list[dict], query: str) -> list[Document]:
        """把归一化后的网页结果包成"未核验"Document（与本地权威库物理隔离）。

        每条 Document：
        - kbase=KBase.WEB.value（独立来源标识，绝不与权威库混淆）；
        - title=网页标题、content=摘要；
        - metadata 标注 source="联网·未核验·以官方为准" + url + unverified=True；
        - score 给低值，避免在召回池里压过本地权威召回。

        :param items: list[dict] 网页结果。
        :param query: 触发本次搜索的查询（记入 raw_query_from，用于 RRF 分路/溯源）。
        :return: list[Document]（最多 _DEFAULT_TOPK 条）。
        """
        docs: list[Document] = []
        for i, item in enumerate(items[:_DEFAULT_TOPK]):
            title = self._first_str(item, ("title", "name", "heading")) or "联网搜索结果"
            content = self._first_str(item, ("snippet", "summary", "content", "description", "text", "abstract"))
            url = self._first_str(item, ("url", "link", "href", "source_url"))
            docs.append(Document(
                doc_id=f"{KBase.WEB.value}-{i}",
                title=title,
                content=content,
                kbase=KBase.WEB.value,
                # 联网补充来源给低分：物理隔离 + 不抢本地权威召回的位置。
                score=0.1,
                metadata={
                    "source": _UNVERIFIED_SOURCE,   # 物理隔离的核心标注
                    "url": url,
                    "unverified": True,             # "未核验"标记：绝不进权威引用([[citation:N]])
                },
                raw_query_from=query,
            ))
        return docs

    @staticmethod
    def _first_str(item: dict, keys: tuple[str, ...]) -> str:
        """按候选键顺序取第一个非空字符串值（兼容不同搜索 API 的字段命名）。

        :param item: 单条结果 dict。
        :param keys: 候选字段名（按优先级）。
        :return: 命中的字符串；都没有返回 ""。
        """
        for k in keys:
            v = item.get(k)
            if v:
                return str(v)
        return ""


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：默认未启用，演示"绝不触网、安全返回 []"（无需任何服务/key/网络）。
    async def _demo():
        agent = WebSearchAgent()
        print("[web_search_agent 自测] search =>", await agent.search("小微企业所得税优惠最新政策"))

    asyncio.run(_demo())
