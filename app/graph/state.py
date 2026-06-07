"""LangGraph 全局状态。各节点读写其中字段，串起整条问答链路。
用 TypedDict(total=False) 表示所有字段都可选（节点逐步填充）。"""
from __future__ import annotations
from typing import TypedDict, Any

class GraphState(TypedDict, total=False):
    # 输入
    query: str
    user_id: str
    session_id: str
    top_k: int
    filters: dict
    # 过程
    history: list[dict]          # 多轮历史
    qu: Any                      # QUResult
    route_type: str             # RouteType 值
    plan: Any                    # RetrievalPlan
    recalled: list               # list[Document] 召回结果
    ranked: list                 # list[Document] 粗排+精排后
    reranked: list               # list[Document] 重排后
    context: str                 # 拼好的上下文
    references: list             # list[dict] 引用
    text2sql_result: Any         # Text2SQLResult
    # 输出
    answer: str
