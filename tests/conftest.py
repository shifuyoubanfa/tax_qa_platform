"""pytest 公共夹具与路径配置。
本模块在测试体系里的位置：所有测试文件运行前都会先加载这里。

为什么需要它：
1. 项目没有用 pyproject/安装成包，直接 `import config`/`import app` 需要"项目根目录"在 sys.path 上。
   这里把项目根（tests 的上一级）插到 sys.path 最前面，保证无论从哪个目录调用 pytest 都能导入到。
2. 提供一份"假 LLM"夹具（fake_llm），让需要 LLM 的测试不必真连大模型服务。
3. pytest-asyncio 的事件循环由插件管理，这里不重复造轮子。

注：本文件只放"测试基础设施"，不放业务断言。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---- 1) 把项目根目录加入 sys.path，保证 import config / import app 可用 ----
# __file__ = tests/conftest.py -> parents[1] = 项目根 tax_qa_platform
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class _FakeMessage:
    """模拟 langchain LLM 返回的消息对象，只需有 .content 属性即可被上层读取。"""

    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    """极简假 LLM：把传入内容固定回成一段可预测的文本，避免真连大模型。

    支持 langchain ChatModel 常见调用方式：.invoke(...) 同步、.ainvoke(...) 异步。
    需要更复杂行为的测试可在用例内自定义。
    """

    def __init__(self, reply: str = "测试用固定回答") -> None:
        self._reply = reply

    def invoke(self, *args, **kwargs):  # noqa: D401 - 行为与签名对齐 langchain
        """同步调用，返回带 .content 的假消息。"""
        return _FakeMessage(self._reply)

    async def ainvoke(self, *args, **kwargs):
        """异步调用，返回带 .content 的假消息。"""
        return _FakeMessage(self._reply)


@pytest.fixture
def fake_llm() -> _FakeLLM:
    """提供一个假 LLM 实例，供需要 LLM 的用例注入，避免真实网络调用。

    :return: _FakeLLM 实例（.invoke/.ainvoke 返回固定内容）。
    """
    return _FakeLLM()
