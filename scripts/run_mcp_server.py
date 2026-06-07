"""启动税务能力 MCP Server 的便捷脚本。
本模块在整体链路里的位置：运维/调试侧的"一键启动入口"。它只做一件事——把 app.mcp.server 跑起来，
让平台已有的税务能力（社保查表 / 商品编码查表 / Text2SQL 经营数据查询）以 MCP 协议对外暴露。

为什么单独给一个脚本（而不是只用 python -m app.mcp.server）：
1. 统一入口：和 scripts/ 下其它离线脚本风格一致，便于在 README / 运行手册里指明"如何起 MCP server"。
2. 友好降级：在没装 mcp 依赖时给出清晰中文提示，而不是一串 ImportError 堆栈。

两种等价启动方式（任选其一）：
    python -m app.mcp.server            # 直接用模块入口（server.py 的 __main__ 调 mcp.run()）
    python scripts/run_mcp_server.py    # 本脚本（内部就是 import server 并调 mcp.run()）

传输方式：默认 stdio（进程经标准输入/输出与 MCP 客户端通信，通常由客户端作为子进程拉起；
直接手动运行时它会等待 stdio 上的 MCP 握手，属正常现象）。

注意：本脚本运行需要已安装 mcp 依赖（见 requirements.txt 的 `mcp` 行）。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 保证从任意目录运行都能 import 到 config / app（与其它脚本一致：把项目根放进 sys.path）。
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    """启动 MCP server（stdio 传输）；缺依赖时给出中文提示并以非零码退出。"""
    setup_logging()
    try:
        # 惰性导入：缺 mcp 时在这里捕获，给出可读提示而非裸 ImportError。
        from app.mcp.server import mcp
    except ImportError as e:
        logger.error("[run_mcp_server] 未安装 mcp 依赖，无法启动 MCP server：%s", e, exc_info=True)
        print("[run_mcp_server] 启动失败：未安装 mcp。请先 `pip install mcp`（见 requirements.txt）。")
        raise SystemExit(1)

    logger.info("[run_mcp_server] 启动 tax-tools MCP server（stdio 传输）……")
    mcp.run()


if __name__ == "__main__":
    main()
