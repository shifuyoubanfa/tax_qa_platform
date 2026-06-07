"""MCP（Model Context Protocol）客户端（真实接入版：惰性、可降级）。
本模块在整体链路里的位置：基础设施层的"可选外接层"。MCP 是一套让程序/大模型以统一协议调用外部工具的协议。
本平台用它把结构化查询（社保 / 商品编码 / Text2SQL）的执行【可选地】改道到一个 MCP server
（既可以是本平台自己的 app.mcp.server，也可以是远程的百炼等 MCP 服务）。

与 app/mcp/server.py 的分工（一句话）：
- app.mcp.server = "服务端"：把本地税务能力暴露成 MCP 工具。
- 本模块(McpClient) = "客户端"：去连一个 MCP server，list_tools / call_tool 调它的工具。

设计要点（为什么是"惰性 + 安全降级"）：
- 默认 settings.mcp_enabled=False：list_tools 返回 []、call_tool 返回 None，绝不连接、绝不抛异常，
  保证无网络 / 无 MCP server 时整条链路照常跑（结构化 agent 自然回退到直连 MySQL）。
- 启用时按 settings.mcp_transport 选传输：
  - stdio：把 settings.mcp_server_command 拆成 command/args，作为子进程拉起一个 MCP server；
  - sse  ：连接 settings.mcp_sse_url 指向的远程 MCP server；
  - http ：streamable http 传输——连接一个远程 MCP server 的 http 端点（如百炼 DashScope），
           并带上 headers={"Authorization": f"Bearer {key}"}（key 来自传入或 settings.dashscope_api_key）。
- 惰性：import 本模块不连任何服务；即使 enabled=True，也是"每次调用临时建一个会话"（简单稳妥，
  不维护长连接，避免事件循环/进程生命周期管理的复杂度）。
- 未安装 mcp 依赖 / 连接失败 / 调用异常：一律记清晰中文日志并降级（list_tools->[]，call_tool->None）。
- 可覆盖参数：list_tools/call_tool 接受可选的 (url / transport / api_key)，用于临时连不同的 MCP server
  （例如联网搜索 WebSearchAgent 要连 settings.websearch_mcp_url，与结构化查表的 mcp_* 配置区分开）；
  不传时一律回退到构造时读取的 settings.mcp_* 默认值，保持向后兼容。

接口契约（INTERFACES，签名不可变；新增的覆盖参数均为可选关键字、默认 None，不破坏旧调用）：
- async list_tools(url=None, transport=None, api_key=None) -> list[dict]   每项形如 {"name","description","input_schema"}
- async call_tool(name, args, url=None, transport=None, api_key=None) -> Any 工具返回内容（解析后的结构化值）；未启用/失败返回 None
"""
from __future__ import annotations

from typing import Any, Optional

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class McpClient:
    """MCP 客户端（enabled=False 时安全降级，不连接、不报错）。

    用法::

        mcp = McpClient()
        tools = await mcp.list_tools()                       # 未启用时返回 []
        rows = await mcp.call_tool("query_social_security", {"region": "杭州"})  # 未启用时返回 None

    :return: 见各方法说明。
    """

    def __init__(self) -> None:
        """构造仅读配置，不连服务（惰性 + 降级）。"""
        self._enabled = bool(settings.mcp_enabled)
        self._transport = (settings.mcp_transport or "stdio").strip().lower()
        # 默认鉴权 key：http 传输用作 Bearer token；调用时可被 api_key 覆盖参数顶替。
        self._api_key = settings.dashscope_api_key or ""
        if not self._enabled:
            # 明确告知"已禁用、走降级"，便于排查"为什么没调到 MCP 工具"。
            logger.info("[MCP客户端] 未启用（MCP_ENABLED=False），list_tools/call_tool 走安全降级")
        else:
            logger.info("[MCP客户端] 已启用：transport=%s", self._transport)

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    async def list_tools(
        self,
        url: Optional[str] = None,
        transport: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> list[dict]:
        """列出 MCP server 暴露的工具。

        :param url: 可选，临时覆盖要连的 MCP server 地址（http/sse 用）；不传走 settings 默认。
        :param transport: 可选，临时覆盖传输方式（stdio/sse/http）；不传走构造时的 settings.mcp_transport。
        :param api_key: 可选，临时覆盖 http 传输的 Bearer 鉴权 key；不传走 settings.dashscope_api_key。
        :return: 工具描述列表 [{"name","description","input_schema"}, ...]；
                 未启用 / 失败时返回 []（调用方据此自然跳过工具调用）。
        :raise: 不向外抛；所有异常都被收敛为降级返回 []。
        """
        # 显式传入覆盖参数(url/transport)即视为"调用方自己负责开关"（如联网搜索已自行判 websearch 开关），
        # 此时不再受构造时的 settings.mcp_enabled 总开关限制；否则按结构化查表总开关降级。
        if not self._is_active(url, transport):
            logger.info("[MCP客户端] list_tools 降级返回空（未启用）")
            return []
        try:
            async with self._open_session(url, transport, api_key) as session:
                resp = await session.list_tools()
                tools = []
                for t in getattr(resp, "tools", []) or []:
                    tools.append({
                        "name": getattr(t, "name", ""),
                        "description": getattr(t, "description", "") or "",
                        # FastMCP 生成的入参 schema 字段名是 inputSchema（驼峰），对外统一成 input_schema。
                        "input_schema": getattr(t, "inputSchema", None) or {},
                    })
                logger.info("[MCP客户端] list_tools 成功，工具数=%d", len(tools))
                return tools
        except Exception as e:  # noqa: BLE001 - 列工具失败不致命，降级为空
            logger.error("[MCP客户端] list_tools 失败，降级返回空：%s", e, exc_info=True)
            return []

    async def call_tool(
        self,
        name: str,
        args: dict,
        url: Optional[str] = None,
        transport: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Any:
        """调用一个 MCP 工具并返回其结果。

        :param name: 工具名（如 "query_social_security"）。
        :param args: 工具入参字典。
        :param url: 可选，临时覆盖要连的 MCP server 地址（http/sse 用）；不传走 settings 默认。
        :param transport: 可选，临时覆盖传输方式（stdio/sse/http）；不传走构造时的 settings.mcp_transport。
        :param api_key: 可选，临时覆盖 http 传输的 Bearer 鉴权 key；不传走 settings.dashscope_api_key。
        :return: 工具返回内容（尽量解析成结构化值：优先 structuredContent，其次文本内容）；
                 未启用 / 失败时返回 None（调用方据此回退到本地直连逻辑）。
        :raise: 不向外抛；所有异常都被收敛为降级返回 None。
        """
        # 同 list_tools：显式 url/transport 覆盖时由调用方自管开关，不再受 settings.mcp_enabled 限制。
        if not self._is_active(url, transport):
            logger.info("[MCP客户端] call_tool 降级返回 None（未启用）：tool=%s", name)
            return None
        try:
            async with self._open_session(url, transport, api_key) as session:
                result = await session.call_tool(name, args or {})
                parsed = self._parse_tool_result(result)
                logger.info("[MCP客户端] call_tool 成功：tool=%s", name)
                return parsed
        except Exception as e:  # noqa: BLE001 - 调用失败不致命，降级为 None 让上层回退
            logger.error("[MCP客户端] call_tool 失败，降级返回 None：tool=%s err=%s",
                         name, e, exc_info=True)
            return None

    def _is_active(self, url: Optional[str], transport: Optional[str]) -> bool:
        """判断本次调用是否应真正连接 MCP server（而非降级）。

        规则：
        - 显式传入 url 或 transport 覆盖参数 => 视为"调用方自管开关"（如联网搜索已判过 websearch 开关），
          不再受构造时的 settings.mcp_enabled 总开关限制，允许连接；
        - 未传任何覆盖 => 沿用旧行为，仅当 settings.mcp_enabled=True 才连接。

        :param url: 调用时传入的覆盖地址（可能为 None）。
        :param transport: 调用时传入的覆盖传输方式（可能为 None）。
        :return: True 表示应连接；False 表示降级返回空。
        """
        if url or transport:
            return True
        return self._enabled

    # ------------------------------------------------------------------ #
    # 临时会话（每次调用临时建会话，简单稳妥）
    # ------------------------------------------------------------------ #
    def _open_session(
        self,
        url: Optional[str] = None,
        transport: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """按传输方式打开一个【临时】MCP 会话（异步上下文管理器）。

        为什么用上下文管理器：MCP 的 stdio/sse/http 传输与 ClientSession 都需要在异步 with 块里
        建立/释放，封装成一个 async context manager 后，list_tools/call_tool 用 `async with` 即可。

        :param url: 可选，覆盖 MCP server 地址；不传走 settings 默认。
        :param transport: 可选，覆盖传输方式；不传走构造时的 settings.mcp_transport。
        :param api_key: 可选，覆盖 http 传输的 Bearer 鉴权 key；不传走 settings.dashscope_api_key。
        :return: 一个异步上下文管理器，进入后产出已 initialize 的 ClientSession。
        :raise RuntimeError: 未安装 mcp 依赖，或传输配置不合法（具体由 _session_ctx 内部抛出后被上层降级）。
        """
        return self._session_ctx(url, transport, api_key)

    def _session_ctx(
        self,
        url: Optional[str] = None,
        transport: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """构造真正的会话上下文（用 contextlib.asynccontextmanager 包裹传输+会话的建立与释放）。

        :param url: 可选，覆盖 MCP server 地址；不传走 settings 默认。
        :param transport: 可选，覆盖传输方式；不传走构造时的 settings.mcp_transport。
        :param api_key: 可选，覆盖 http 传输的 Bearer 鉴权 key；不传走 settings.dashscope_api_key。
        :return: async context manager，产出 initialize 后的 ClientSession。
        :raise RuntimeError: 缺依赖 / 传输不支持 / 缺配置。
        """
        from contextlib import asynccontextmanager

        # 覆盖参数优先，否则回退构造时读取的 settings 默认值（保持向后兼容）。
        _transport = (transport or self._transport or "stdio").strip().lower()
        _api_key = api_key if api_key is not None else self._api_key

        @asynccontextmanager
        async def _ctx():
            # 惰性导入 mcp：未装时给出清晰中文报错（由上层 try/except 降级）。
            try:
                from mcp import ClientSession
            except ImportError as e:
                raise RuntimeError("[MCP客户端] 未安装 mcp，请 `pip install mcp`") from e

            if _transport == "stdio":
                # stdio：把 settings.mcp_server_command 拆成 command + args，作为子进程拉起 server。
                from mcp import StdioServerParameters
                from mcp.client.stdio import stdio_client

                parts = (settings.mcp_server_command or "").split()
                if not parts:
                    raise RuntimeError("[MCP客户端] stdio 模式缺少 MCP_SERVER_COMMAND 配置")
                params = StdioServerParameters(command=parts[0], args=parts[1:])
                logger.info("[MCP客户端] 建立 stdio 会话：command=%s args=%s", parts[0], parts[1:])
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        yield session

            elif _transport == "sse":
                # sse：连接远程 MCP server 的 SSE 地址（如百炼）。
                from mcp.client.sse import sse_client

                # 覆盖 url 优先，否则回退 settings 默认（mcp_sse_url / 百炼地址）。
                _url = url or settings.mcp_sse_url or settings.mcp_bailian_base_url
                if not _url:
                    raise RuntimeError("[MCP客户端] sse 模式缺少 MCP_SSE_URL 配置")
                logger.info("[MCP客户端] 建立 sse 会话：url=%s", _url)
                async with sse_client(_url) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        yield session

            elif _transport == "http":
                # http：streamable http 传输——连接远程 MCP server 的 http 端点（如百炼 DashScope）。
                # 用 mcp SDK 的 streamablehttp_client；带 Bearer 鉴权头（key 来自覆盖 api_key 或 settings.dashscope_api_key）。
                # 兼容不同 SDK 版本的函数名：新版重命名为 streamable_http_client，旧版为 streamablehttp_client。
                try:
                    from mcp.client.streamable_http import streamablehttp_client as _http_client
                except ImportError:
                    from mcp.client.streamable_http import streamable_http_client as _http_client  # type: ignore[attr-defined]

                # 覆盖 url 优先，否则回退 settings.websearch_mcp_url（联网搜索默认端点）。
                _url = url or settings.websearch_mcp_url
                if not _url:
                    raise RuntimeError("[MCP客户端] http 模式缺少 url（WEBSEARCH_MCP_URL）配置")
                # 仅在有 key 时才带 Authorization 头，避免拼出 "Bearer " 空值。
                headers = {"Authorization": f"Bearer {_api_key}"} if _api_key else None
                logger.info("[MCP客户端] 建立 streamable http 会话：url=%s 带鉴权=%s", _url, bool(headers))
                # streamablehttp_client 产出 (read, write, get_session_id)；只取前两个建会话，忽略其余。
                async with _http_client(_url, headers=headers) as transports:
                    read, write = transports[0], transports[1]
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        yield session

            else:
                raise RuntimeError(
                    f"[MCP客户端] 不支持的传输方式：{_transport}（应为 stdio / sse / http）"
                )

        return _ctx()

    @staticmethod
    def _parse_tool_result(result: Any) -> Any:
        """把 mcp 的 CallToolResult 解析成尽量"原生"的 Python 值。

        优先级：
        1. structuredContent：FastMCP 对返回 dict/list 的工具会带上结构化内容，直接返回它（最理想）；
        2. content 文本块：尝试 json.loads 解析成结构化值，解析失败则返回拼接后的纯文本；
        3. 都没有：返回 None。

        :param result: session.call_tool 的返回对象。
        :return: 解析后的结构化值 / 文本 / None。
        """
        import json

        # 1) 结构化内容（最可靠）
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            # FastMCP 对"返回 list 的工具"会包成 {"result": [...]}；解开这层方便上层直接用。
            if isinstance(structured, dict) and set(structured.keys()) == {"result"}:
                return structured["result"]
            return structured

        # 2) 文本内容块
        texts: list[str] = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                texts.append(text)
        if texts:
            joined = "\n".join(texts)
            try:
                return json.loads(joined)
            except (ValueError, TypeError):
                return joined

        # 3) 空
        return None


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：默认未启用，演示安全降级（无需任何服务、无需装 mcp）。
    import asyncio

    async def _demo():
        client = McpClient()
        print("[mcp_client 自测] list_tools =>", await client.list_tools())
        print("[mcp_client 自测] call_tool =>", await client.call_tool("query_social_security", {"region": "杭州"}))

    asyncio.run(_demo())
