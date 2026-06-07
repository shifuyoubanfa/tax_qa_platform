"""函数耗时计时装饰器。
本模块在整体链路里的位置：横切工具。检索链路每一段（QU/召回/粗排/精排/重排/摘要/SQL）都很关心
耗时，给关键函数加上 @runtime / @aruntime 就能在日志里看到每段花了多少毫秒，方便定位瓶颈。

设计要点（为什么这么做）：
1. 同步函数用 @runtime，异步函数(async def)用 @aruntime——两者实现几乎一致，区别只在于
   异步版用 await 调用原函数。分开两个装饰器是为了语义清晰、避免运行时判断协程的复杂逻辑。
2. 用 time.perf_counter()（单调高精度计时器）而非 time.time()，避免系统时钟回拨影响测量。
3. 用 functools.wraps 保留原函数的 __name__/__doc__，不破坏被装饰函数的元信息。
4. 统一日志格式 "[计时] {func} 耗时 {ms}ms"，对标爱搜税 @runtime 的输出风格。

风格对标 爱搜税 seacore.utils.time_utils.runtime（@runtime 装饰器）。
"""
from __future__ import annotations

import functools
import time
from typing import Any, Callable

from config.logging_config import get_logger

logger = get_logger(__name__)


def runtime(func: Callable) -> Callable:
    """同步函数计时装饰器：执行结束后用 logger.info 打印耗时。

    :param func: 被装饰的同步函数。
    :return: 包装后的函数，行为与原函数一致，仅多打印一条耗时日志。
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()  # 高精度单调计时起点
        try:
            return func(*args, **kwargs)
        finally:
            # 放 finally：即使函数抛异常，也能记录到它"跑了多久才挂"
            cost_ms = (time.perf_counter() - start) * 1000
            logger.info("[计时] %s 耗时 %.2fms", func.__name__, cost_ms)

    return wrapper


def aruntime(func: Callable) -> Callable:
    """异步函数计时装饰器：await 原协程，结束后用 logger.info 打印耗时。

    :param func: 被装饰的异步函数（async def）。
    :return: 包装后的协程函数，行为与原函数一致，仅多打印一条耗时日志。
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return await func(*args, **kwargs)
        finally:
            cost_ms = (time.perf_counter() - start) * 1000
            logger.info("[计时] %s 耗时 %.2fms", func.__name__, cost_ms)

    return wrapper


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：演示同步/异步两种装饰器。
    import asyncio

    @runtime
    def _sync_demo():
        s = sum(i for i in range(100000))
        return s

    @aruntime
    async def _async_demo():
        await asyncio.sleep(0.05)
        return "ok"

    print("[timing 自测] 同步结果 =>", _sync_demo())
    print("[timing 自测] 异步结果 =>", asyncio.run(_async_demo()))
