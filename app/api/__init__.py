"""HTTP 接口层（FastAPI 路由）包标识。
本包在整体链路里的位置：最外层"接入层"。它把内部的编排器(Orchestrator)产出的
SSEMessage 事件流，转换成标准 HTTP / SSE 响应回写给前端；自身不包含任何检索/排序/
生成的业务逻辑，只做"协议适配 + 异常兜底"。

包内文件：
- routes_chat.py  ：核心问答接口 POST /api/v1/chat（SSE 流式）。
- routes_health.py：健康检查 GET / 与 GET /health。
"""
