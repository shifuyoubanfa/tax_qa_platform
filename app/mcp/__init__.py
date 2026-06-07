"""MCP（Model Context Protocol）服务端子包标识。
本包在整体链路里的位置：把本平台已有的"本地税务能力"（社保查表 / 商品编码查表 / Text2SQL 经营数据查询）
用 MCP 协议【对外暴露成一个 MCP server】，让任意支持 MCP 的大模型 / 客户端都能像调用标准工具一样调用它们。

与 app/clients/mcp_client.py 的分工（一句话）：
- 本包(app.mcp.server) = "服务端"：复用现有 agent，把能力包成 MCP 工具暴露出去。
- app.clients.mcp_client = "客户端"：在 mcp_enabled=True 时，反过来连一个 MCP server（可以就是本 server）去调工具。

注意：本 __init__ 故意"不"在导入期 import server，避免把 mcp 重依赖在 `import app.mcp` 时就拉进来；
需要时请 `from app.mcp.server import mcp` 或 `python -m app.mcp.server` 启动。
"""
