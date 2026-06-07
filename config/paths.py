"""项目根路径锚点 —— 让 .env / 日志 / 资源 的定位与"从哪个目录运行"无关。

为什么需要它：
- 直接 `python 某文件.py` 或点 IDE 的"运行"按钮时，当前工作目录(cwd)往往不是项目根
  （可能是文件所在目录，或上层工作区目录）。
- 一旦 cwd 不对：settings 读不到 `.env`、日志写到别处、相对资源路径全错。
把这些路径统一锚定到"本文件推算出的项目根"，就能"在任何目录、用任何方式运行"都正确定位。

用法::

    from config.paths import PROJECT_ROOT
    env_path = PROJECT_ROOT / ".env"
"""
from __future__ import annotations

from pathlib import Path

# 本文件位于 <项目根>/config/paths.py：
#   parent      -> config/
#   parent.parent -> 项目根 tax_qa_platform/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
