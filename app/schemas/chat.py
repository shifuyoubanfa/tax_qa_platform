"""对外HTTP接口的请求/响应模型（pydantic），以及SSE消息封装。"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Optional

class ChatRequest(BaseModel):
    """/api/v1/chat 请求体。"""
    query: str = Field(..., description="用户自然语言问题")
    user_id: str = Field("anonymous", description="用户ID，用于多轮会话记忆")
    session_id: str = Field("default", description="会话ID，用于多轮会话记忆")
    top_k: int = Field(5, description="最终参考片段数量")
    stream: bool = Field(True, description="是否SSE流式返回")
    filters: Optional[dict[str, Any]] = Field(None, description="可选的检索过滤条件")

class Reference(BaseModel):
    """返回给前端的单条引用。"""
    doc_id: str
    title: str = ""
    kbase: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = {}
    image_keys: list[str] = []
    # 由 summarizer 对 image_keys 逐个调 MinioClient.presigned_url 生成的临时可访问URL，
    # 供前端直接加载图片（image_keys 仅是对象键，浏览器无法直接访问）。MinIO 不可用时为空。
    image_urls: list[str] = []

class SSEMessage(BaseModel):
    """一条SSE消息：event=事件类型(SSEEvent值)，data=任意载荷。"""
    event: str
    data: dict[str, Any] = {}
