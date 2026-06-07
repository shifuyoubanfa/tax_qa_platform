"""统一日志配置（标准库 logging 版）。
本模块在整体链路里的位置：横切"基础设施层"，被所有业务模块共享。
所有模块统一用::

    from config.logging_config import get_logger
    logger = get_logger(__name__)

设计要点（为什么这么做）：
1. 用标准库 logging 而非第三方（如 loguru），是为了零额外依赖、与 FastAPI/uvicorn 的
   日志体系天然兼容，作者学习时也最通用。
2. 双输出：StreamHandler(stdout) 让本地/容器日志直接可见；RotatingFileHandler 落盘并自动
   轮转（单文件20MB、保留5份），避免日志撑爆磁盘。
3. setup_logging() 做成"幂等"：用一个模块级标志位防止重复加 handler（重复加会导致同一条日志
   被打印多次），FastAPI 多次 import / reload 都安全。
4. 格式里带 时间/级别/模块名/行号/消息，定位问题时一眼能看到是哪行代码打的。

风格对标 掌柜智库/app/core/logger.py（配置驱动、中文注释、双输出）。
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.paths import PROJECT_ROOT
from config.settings import settings

# 日志格式：时间 | 级别 | 模块名:行号 | 消息
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 单个日志文件最大 20MB，超过后轮转
_MAX_BYTES = 20 * 1024 * 1024
# 最多保留 5 个历史文件（app.log.1 ~ app.log.5）
_BACKUP_COUNT = 5

# 幂等标志位：保证 setup_logging 多次调用只真正配置一次，避免 handler 重复叠加。
_CONFIGURED = False


def setup_logging() -> None:
    """初始化全局 root logger（进程启动时调用一次，通常在 FastAPI lifespan 里）。

    行为：
    - 级别取 settings.log_level（默认 INFO），保证 INFO 及以上一定输出。
    - 给 root logger 挂两个 handler：stdout 控制台 + 轮转文件 logs/app.log。
    - 幂等：重复调用不会重复加 handler。

    :return: 无返回值（副作用是配置好全局 logging）。
    :raise OSError: 当日志目录无法创建/无写权限时，由文件handler构造阶段抛出。
    """
    global _CONFIGURED
    # 已配置过则直接返回，避免重复挂 handler 导致日志重复打印
    if _CONFIGURED:
        return

    # 解析级别字符串（如 "INFO"）到 logging 常量；非法值时回退 INFO
    level = getattr(logging, str(settings.log_level).upper(), logging.INFO)

    # 确保日志目录存在；相对路径锚定到项目根，避免从不同目录运行时日志散落各处
    log_dir = Path(settings.log_dir)
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)

    # 控制台输出（stdout，便于本地/容器直接查看）
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # 文件输出（按大小轮转，utf-8 防中文乱码）
    file_handler = RotatingFileHandler(
        filename=str(log_file),
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _CONFIGURED = True
    root.info("[日志] 日志系统初始化完成：级别=%s，文件=%s", settings.log_level, log_file)


def get_logger(name: str) -> logging.Logger:
    """获取一个具名 logger（业务模块统一入口）。

    若全局尚未配置（极端情况下某模块在 setup_logging 之前就打日志），这里兜底调用一次
    setup_logging，保证拿到的 logger 一定有 handler、不会"日志丢失"。

    :param name: 通常传 __name__，让日志能显示来源模块。
    :return: 配置好的 logging.Logger 实例。
    """
    # 兜底：保证任何时候拿到的 logger 都已有 handler
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：验证双输出与级别。
    setup_logging()
    log = get_logger(__name__)
    log.debug("这是一条 DEBUG（默认INFO级下不显示）")
    log.info("这是一条 INFO，应同时出现在控制台和 logs/app.log")
    log.warning("这是一条 WARNING")
    try:
        1 / 0
    except ZeroDivisionError:
        log.error("这是一条带堆栈的 ERROR 示例", exc_info=True)
