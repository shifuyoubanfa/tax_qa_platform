"""MCP 接入的单元测试（不真连：只验证"默认关闭时安全降级" + "server 可导入"）。
本测试在测试体系里的位置：守护 MCP 这条【可选外接层】的两条底线——

1. 安全降级：settings.mcp_enabled=False（默认）时，McpClient.list_tools() 必须返回 []、
   call_tool() 必须返回 None，且全程不连任何服务、不抛异常。这是"MCP 默认关闭不影响现有链路"的回归保证。
2. 服务端可导入：app.mcp.server 能被 import（说明工具注册、FastMCP 装配无语法/装配错误）；
   当环境未安装 mcp 依赖时，用 importorskip 优雅跳过（不算失败），与项目"无 infra 也能跑测试"的约定一致。

测试策略：全程不打开 mcp_enabled、不连 stdio/sse，纯本地、零网络。
注：这里用 asyncio.run 直接驱动协程（而非 @pytest.mark.asyncio），这样不依赖 pytest-asyncio 插件是否安装，
与项目"无 infra / 缺可选依赖也能跑测试"的约定一致。
"""
from __future__ import annotations

import asyncio

import pytest


def test_list_tools_disabled_returns_empty(monkeypatch):
    """默认关闭(mcp_enabled=False)时 list_tools 应返回 []（安全降级，不连接、不抛异常）。"""
    from config.settings import settings
    monkeypatch.setattr(settings, "mcp_enabled", False, raising=False)

    from app.clients.mcp_client import McpClient
    client = McpClient()
    tools = asyncio.run(client.list_tools())
    assert tools == [], "未启用 MCP 时 list_tools 必须返回空列表"


def test_call_tool_disabled_returns_none(monkeypatch):
    """默认关闭(mcp_enabled=False)时 call_tool 应返回 None（安全降级，让上层回退直连）。"""
    from config.settings import settings
    monkeypatch.setattr(settings, "mcp_enabled", False, raising=False)

    from app.clients.mcp_client import McpClient
    client = McpClient()
    result = asyncio.run(client.call_tool("query_social_security", {"region": "杭州"}))
    assert result is None, "未启用 MCP 时 call_tool 必须返回 None"


def test_mcp_client_importable():
    """app.clients.mcp_client 应始终可导入（不依赖 mcp，import 阶段零副作用、不触网）。"""
    import importlib
    mod = importlib.import_module("app.clients.mcp_client")
    assert hasattr(mod, "McpClient"), "mcp_client 应暴露 McpClient 类"


def test_mcp_server_importable():
    """app.mcp.server 应可导入并暴露 FastMCP 实例 `mcp`；未装 mcp 依赖时优雅跳过。"""
    # server.py 顶层 import 了 mcp.server.fastmcp；缺依赖时 importorskip 直接跳过（不算失败）。
    server_mod = pytest.importorskip("app.mcp.server")
    assert hasattr(server_mod, "mcp"), "app.mcp.server 应暴露 FastMCP 实例 mcp"
    # 三个税务工具函数都应存在（被 @mcp.tool() 装饰后仍是可调用对象）。
    for fn_name in ("query_social_security", "query_product_code", "query_business_data"):
        assert hasattr(server_mod, fn_name), f"app.mcp.server 应定义工具 {fn_name}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
