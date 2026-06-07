"""提示词加载工具。
本模块在整体链路里的位置：QU(意图分类/改写/HyDE)、摘要、Text2SQL 各节点都需要读取对应的
.prompt 模板文件，统一从这里加载，做到"prompt与代码分离"——改提示词不用改代码。

设计要点（为什么这么做）：
1. 提示词集中放在 config/prompts/{name}.prompt（纯文本，utf-8），便于版本管理与团队协作。
2. 用 functools.lru_cache 缓存：同一个 prompt 进程内只读一次磁盘，避免高并发下重复IO。
3. 找不到文件时给"清晰中文报错"，直接告诉作者去哪个绝对路径放文件，降低排错成本。

风格对标 掌柜智库/app/core/load_prompt.py。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from config.logging_config import get_logger

logger = get_logger(__name__)

# prompts 目录：config/prompts。
# __file__ = app/utils/prompt_loader.py -> parents[2] = 项目根 tax_qa_platform
_PROMPT_DIR = Path(__file__).resolve().parents[2] / "config" / "prompts"


@lru_cache(maxsize=128)
def load_prompt(name: str) -> str:
    """读取并返回指定提示词模板的原始文本（带进程内缓存）。

    :param name: 提示词文件名（不带 .prompt 后缀），如 "intent_classify"。
    :return: 提示词全文字符串（未做变量渲染，渲染交由调用方 .format/.replace 处理）。
    :raise FileNotFoundError: 当 config/prompts/{name}.prompt 不存在时，给出绝对路径提示。
    """
    prompt_path = _PROMPT_DIR / f"{name}.prompt"
    # 不存在则给出"该去哪放文件"的明确中文报错，便于作者补齐模板
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"提示词文件不存在：{prompt_path.resolve()}。"
            f"请在 config/prompts/ 下创建 {name}.prompt（utf-8 纯文本）。"
        )
    text = prompt_path.read_text(encoding="utf-8")
    logger.info("[提示词] 已加载模板 name=%s（%d 字符）", name, len(text))
    return text


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：尝试加载一个不存在的模板，观察清晰报错。
    try:
        load_prompt("不存在的模板名")
    except FileNotFoundError as e:
        print("[prompt_loader 自测] 预期内的报错 =>", e)
