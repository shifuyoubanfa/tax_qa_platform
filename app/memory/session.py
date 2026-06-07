"""多轮会话记忆（Session Memory）。
本模块在整体链路里的位置：编排层(Orchestrator/LangGraph)在"进入问答"前会先 load 一次历史，
回答结束后把"用户问题"与"助手答案"各 append 一条，下一轮就能带上上下文。底层落地在 MongoDB，
由 MongoHistoryClient(motor 异步) 负责真正读写。

设计要点（为什么这么做）：
1. 在 MongoHistoryClient 之上再包一层 SessionMemory，是为了给编排层一个"业务语义"接口
   （load/append），把"会话历史"这个概念与"具体存储后端"解耦——将来换 Redis/MySQL 只改这一层。
2. 客户端【惰性初始化】：第一次真正 load/append 时才创建 MongoHistoryClient（其内部再惰性连库），
   保证没有 Mongo 时 import 不报错、FastAPI 能照常启动。
3. 健壮性：Mongo 不可用时不让整条问答链路崩溃——load 失败返回空历史(降级为单轮)，
   append 失败只记 error 日志。会话记忆属于"锦上添花"，不该阻断主流程。

风格对标 掌柜智库 的客户端封装与教学注释。
"""
from __future__ import annotations

from typing import Optional

from config.logging_config import get_logger

logger = get_logger(__name__)


class SessionMemory:
    """会话记忆门面：基于 MongoHistoryClient 实现 load/append（按 user_id + session_id 隔离）。

    用法::

        memory = SessionMemory()
        history = await memory.load("u1", "s1")          # [{role, content}, ...]
        await memory.append("u1", "s1", "user", "你好")   # 追加一条
    """

    def __init__(self, max_history: int = 10) -> None:
        """初始化会话记忆。

        :param max_history: load 时默认拉取的最近历史条数，避免上下文过长撑爆 prompt。
        """
        self._max_history = max_history
        # 惰性持有的底层 Mongo 客户端；首次使用才创建，import 阶段不连库
        self._client: Optional[object] = None

    def _get_client(self):
        """惰性获取 MongoHistoryClient 实例（首次调用才创建）。

        :return: MongoHistoryClient 实例。
        :raise Exception: 当 motor/pymongo 未安装或客户端构造失败时，由 import/构造阶段抛出。
        """
        if self._client is None:
            # 延迟到此处 import，避免顶层 import 触发对 motor 的硬依赖
            from app.clients.mongo_client import MongoHistoryClient

            self._client = MongoHistoryClient()
            logger.info("[会话记忆] MongoHistoryClient 惰性初始化完成")
        return self._client

    async def load(self, user_id: str, session_id: str) -> list[dict]:
        """加载某会话最近的历史消息（最多 max_history 条）。

        :param user_id: 用户ID（多用户隔离）。
        :param session_id: 会话ID（同一用户的不同对话隔离）。
        :return: 历史消息列表，元素形如 {"role": "user"/"assistant", "content": "..."}；
                 出错时降级返回空列表（视为单轮对话），不抛异常以免阻断主链路。
        """
        logger.info("[会话记忆] 加载历史 user_id=%s session_id=%s limit=%s",
                    user_id, session_id, self._max_history)
        try:
            client = self._get_client()
            history = await client.load(user_id, session_id, limit=self._max_history)
            logger.info("[会话记忆] 历史加载完成，共 %d 条", len(history or []))
            return history or []
        except Exception as e:  # noqa: BLE001 记忆是增强项，失败不应中断问答
            # 降级：返回空历史，相当于本轮按"无上下文"处理
            logger.error("[会话记忆] 加载历史失败，降级为空历史：%s", e, exc_info=True)
            return []

    async def append(self, user_id: str, session_id: str, role: str, content: str) -> None:
        """向某会话追加一条消息（用户提问 / 助手回答各调一次）。

        :param user_id: 用户ID。
        :param session_id: 会话ID。
        :param role: 角色，约定取 "user" 或 "assistant"。
        :param content: 消息正文。
        :return: 无返回值。出错时只记日志、不抛异常（避免写记忆失败影响已生成的答案）。
        """
        # 空内容不落库，避免脏数据
        if not content:
            logger.info("[会话记忆] 跳过空内容写入 role=%s", role)
            return
        try:
            client = self._get_client()
            await client.append(user_id, session_id, role, content)
            logger.info("[会话记忆] 已追加一条消息 role=%s len=%d", role, len(content))
        except Exception as e:  # noqa: BLE001
            logger.error("[会话记忆] 追加消息失败：%s", e, exc_info=True)


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：不真连 Mongo，只演示降级行为不抛异常。
    import asyncio

    async def _demo():
        mem = SessionMemory()
        # 没有 Mongo 时，load 应安全返回空列表（走 except 降级分支）
        hist = await mem.load("u_demo", "s_demo")
        print("[session 自测] 无infra时 load =>", hist)
        # append 也应安全（只记 error 日志，不抛）
        await mem.append("u_demo", "s_demo", "user", "测试一条")
        print("[session 自测] append 调用未抛异常")

    asyncio.run(_demo())
