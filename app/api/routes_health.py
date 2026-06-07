"""健康检查路由。
本模块在整体链路里的位置：接入层最简单的一组探针接口，供 K8s/负载均衡/运维脚本
判断服务"是否活着"。它【不触碰任何外部依赖】（不连 LLM/Milvus/MySQL...），
只要进程能响应 HTTP 就返回 ok——这正是"存活探针(liveness)"该有的行为：
轻、快、不因下游抖动而误判服务死亡。

设计要点（为什么这么做）：
1. 健康检查不做深度依赖检测，避免下游（如 Milvus）短暂不可用时探针失败、
   进而被编排系统反复重启本无故障的服务。若将来需要"就绪探针(readiness)"，
   可另开 /ready 去探活下游，与本文件的存活语义区分开。
2. 返回结构固定 {"service","status"}，前端/监控可稳定解析。
"""
from __future__ import annotations

from fastapi import APIRouter

from config.logging_config import get_logger

logger = get_logger(__name__)

# APIRouter 是 FastAPI 推荐的"路由分组"方式：每个功能域一个 router，
# 最后在 main.py 用 include_router 汇总挂载，避免把所有接口堆在一个文件里。
router = APIRouter(tags=["health"])

# 统一的健康响应体，避免两个接口各写一份、日后改字段时漏改。
_HEALTH_PAYLOAD = {"service": "tax-qa-platform", "status": "ok"}


# 注意：根路径 "/" 不在此注册——它留给前端静态页（main.py 把 StaticFiles 挂在 "/"）。
# 若把 GET "/" 注册成健康检查，会抢占 "/" 导致前端首页打不开（路由优先级高于 Mount）。
# 存活探针统一走下面的 /health（K8s liveness 标准路径）；负载均衡若默认探 "/"，会拿到前端页(200)亦可判活。


@router.get("/health", summary="标准健康检查")
async def health() -> dict:
    """标准健康检查接口（K8s liveness / 运维脚本常用路径）。

    作用：与根路径等价，提供约定俗成的 /health 路径，便于直接对接监控体系。

    :return: 固定字典 {"service": "tax-qa-platform", "status": "ok"}。
    """
    logger.debug("[health] 收到 /health 健康检查请求")
    return _HEALTH_PAYLOAD
