"""外部客户端层（clients）包标识。
本包在整体链路里的位置：最底层"基础设施适配层"，把所有外部服务
（LLM / Embedding / Reranker / Milvus / ES / Qdrant / MySQL / Mongo / MinIO / MCP）
封装成统一、惰性、可降级的客户端，供上层 召回/排序/重排/摘要/Agent/编排 调用。

设计总原则（所有子模块共同遵守）：
1. 惰性连接：import 本包不会触发任何网络连接，只有"首次真正使用"时才建连，
   保证没有 infra 的开发机上也能 import、能启动 FastAPI。
2. 配置全走 config.settings：禁止在代码里硬编码地址/账号/key。
3. 健壮性：外部调用一律 try/except + 清晰中文报错，关键网络调用配 tenacity 重试或超时。
4. 教学注释：每个客户端都写清"做什么 / 为什么这么写 / 怎么用"。

注意：本 __init__ 故意"不"在导入期 eager import 各客户端类，
避免把 pymilvus/elasticsearch 等重依赖在 import app.clients 时就拉进来。
需要时请按需 `from app.clients.milvus_client import MilvusClient`。
"""
