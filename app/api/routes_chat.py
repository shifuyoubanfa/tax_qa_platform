"""核心问答路由：POST /api/v1/chat（SSE 流式）。
本模块在整体链路里的位置：接入层的"主干道"。它接收前端的自然语言问题(ChatRequest)，
驱动编排器 Orchestrator.astream() 跑完"意图识别 -> 检索/Text2SQL -> 重排 -> 摘要生成"
整条链路，并把链路过程中产出的一连串 SSEMessage 实时转写成 SSE 文本帧，边算边推给前端。

================ SSE（Server-Sent Events）流式到底怎么工作？=================
1. SSE 是"服务器单向、持续地往浏览器推消息"的标准（基于一条长连接的 HTTP 响应）。
   与 WebSocket 相比它更轻：纯 HTTP、单向、浏览器原生 EventSource 即可消费。
2. 关键在响应头 Content-Type: text/event-stream。一旦是这个类型，浏览器就不会等
   整个响应结束，而是【收到一帧就触发一次事件】。
3. 每一帧的格式是固定的：
       event: <事件名>\n
       data:  <JSON载荷>\n
       \n                      <- 这个空行是"帧结束"标志，缺了浏览器不触发事件！
   本文件里这件事交给 app.utils.sse.format_sse() 统一完成。
4. 在 FastAPI 里，我们返回一个 StreamingResponse，它接收一个【异步生成器】：
   生成器每 yield 一段字符串，框架就立刻把这段字节冲刷给客户端。于是
   "Orchestrator 每算出一个中间结果 -> yield 一帧 -> 前端立刻看到进度"，
   实现"边想边答"的打字机效果，而不是干等几十秒后一次性返回。
5. 反向代理(Nginx等)默认会缓冲响应、破坏流式。所以本文件特地设置
   Cache-Control: no-cache 与 X-Accel-Buffering: no 两个响应头，关掉缓冲。
===========================================================================

设计要点（为什么这么做）：
- 业务异常【不抛 500】而是转成一帧 error 事件再正常结束流：因为 SSE 响应头已经发出、
  HTTP 状态码无法再改，此时唯一能告诉前端"出错了"的办法就是发一个 error 事件，
  这样前端能优雅提示用户，而不是连接被硬生生掐断、不知所措。
- 路由层只做"协议适配 + 异常兜底"，不写任何检索/生成逻辑，保持单一职责。

风格对标 掌柜智库/app/utils/sse_utils.py 的 SSE 生成器写法。
"""
from __future__ import annotations

from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from config.constants import SSEEvent
from config.logging_config import get_logger
from app.schemas.chat import ChatRequest
from app.utils.sse import format_sse

logger = get_logger(__name__)

# 该功能域的路由分组，统一前缀 /api/v1，便于将来做接口版本演进（/api/v2 等）。
router = APIRouter(prefix="/api/v1", tags=["chat"])

# StreamingResponse 的响应头：关掉各级缓冲，保证 SSE 真正"流式"而非攒齐再发。
# - Cache-Control: no-cache    -> 浏览器/中间层不要缓存这条流
# - Connection: keep-alive     -> 维持长连接
# - X-Accel-Buffering: no       -> 显式告诉 Nginx 不要缓冲（否则前端会卡到流结束才一次性收到）
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


async def _event_stream(req: ChatRequest) -> AsyncIterator[str]:
    """把 Orchestrator 产出的 SSEMessage 流，逐条转写成标准 SSE 文本帧。

    这是真正喂给 StreamingResponse 的【异步生成器】：它每 yield 一段字符串，
    FastAPI 就立刻冲刷给客户端，从而实现"边算边推"。

    :param req: 已解析校验过的用户请求体（ChatRequest）。
    :yield: 一帧帧的 SSE 文本（形如 "event: x\\ndata: {...}\\n\\n"）。
    :raise: 内部捕获所有异常并转成 error 事件，故对调用方不抛出。
    """
    # 进入关键链路节点：记录本次会话的核心入参，便于排障与链路追踪。
    logger.info(
        "[chat] 收到问答请求 user_id=%s session_id=%s top_k=%s stream=%s query=%s",
        req.user_id, req.session_id, req.top_k, req.stream, req.query,
    )

    # 延迟到函数内部再 import Orchestrator：
    # 1) 避免与编排层(graph.pipeline)产生模块级循环导入；
    # 2) 编排层会牵连一堆重客户端，模块加载期不触碰它能让 FastAPI 启动更快、更稳。
    from app.graph.pipeline import Orchestrator

    try:
        orchestrator = Orchestrator()
        # 异步迭代编排器的事件流：每拿到一个 SSEMessage 就立即转帧下发。
        async for msg in orchestrator.astream(req):
            yield format_sse(msg.event, msg.data)
        logger.info("[chat] 问答链路正常结束 session_id=%s", req.session_id)
    except Exception as exc:  # noqa: BLE001 —— 接入层必须兜住所有异常，绝不让连接裸崩
        # SSE 响应头已发出、HTTP 状态码改不了，只能用一帧 error 事件通知前端。
        logger.error("[chat] 问答链路异常 session_id=%s err=%s", req.session_id, exc, exc_info=True)
        yield format_sse(SSEEvent.ERROR.value, {"message": "服务处理异常，请稍后重试", "detail": str(exc)})
        # 再补一个 done 事件，让前端的流式状态机能干净收尾（关闭 loading 等）。
        yield format_sse(SSEEvent.DONE.value, {})


@router.post("/chat", summary="税务智能问答（SSE流式）")
async def chat(req: ChatRequest) -> StreamingResponse:
    """税务智能问答主接口。

    作用：接收用户问题，返回一条 text/event-stream 长连接，按 意图->检索->引用->答案增量->完成
    的顺序持续推送事件，前端据此实现进度提示与打字机式回答。

    :param req: 请求体 ChatRequest（query 必填，其余有默认值）。
    :return: StreamingResponse，media_type=text/event-stream，body 为 SSE 帧流。
    """
    # media_type 必须是 text/event-stream，浏览器才会按 SSE 协议逐帧解析。
    return StreamingResponse(
        _event_stream(req),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
