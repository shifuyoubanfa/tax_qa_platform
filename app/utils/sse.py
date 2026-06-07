"""SSE（Server-Sent Events）帧格式化工具。
本模块在整体链路里的位置：编排层(Orchestrator)产出一连串 SSEMessage，HTTP路由层把它们
逐条转成标准 SSE 文本帧写回浏览器，前端按 event 类型渲染"意图/检索进度/引用/答案增量/完成"。

设计要点（为什么这么做）：
1. SSE 帧的标准格式是 "event: <事件名>\\ndata: <载荷>\\n\\n"——两行内容 + 一个空行作为帧分隔。
   末尾的空行(\\n\\n)必不可少，缺了浏览器不会触发该事件。
2. data 用 json.dumps(..., ensure_ascii=False) 序列化：ensure_ascii=False 保证中文按原文输出，
   不被转义成 \\uXXXX，便于前端直接读取与调试。

风格对标 掌柜智库/app/utils/sse_utils.py 的 _sse_pack。
"""
from __future__ import annotations

import json
from typing import Any


def format_sse(event: str, data: dict[str, Any]) -> str:
    """把一个事件名 + 数据载荷格式化成标准 SSE 文本帧。

    :param event: 事件类型字符串（取 config.constants.SSEEvent 的值，如 "answer_delta"）。
    :param data: 任意可JSON序列化的字典载荷。
    :return: 形如 "event: {event}\\ndata: {json}\\n\\n" 的 SSE 帧字符串。
    """
    # ensure_ascii=False：中文不转义，前端/日志直接可读
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：打印一条带中文的 SSE 帧，注意结尾的空行。
    frame = format_sse("answer_delta", {"text": "您好，这是一条中文增量"})
    print(repr(frame))
