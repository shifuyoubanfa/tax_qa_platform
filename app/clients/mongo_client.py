"""MongoDB 会话历史客户端（motor 异步，惰性连接）。
本模块在整体链路里的位置：基础设施层，为"多轮对话记忆"提供持久化。
编排层(app/memory/session.py)在每轮问答前 load 最近历史拼进上下文，问答后把 user/assistant
两条消息 append 进来，下一轮就能"记住"上文。

为什么用 motor（异步 MongoDB 驱动）而非 pymongo：
- 整条链路是 async（FastAPI + LangGraph + SSE），用 motor 才不阻塞事件循环；
  pymongo 是同步的，在 async 里会卡住整个事件循环。
- 风格参考 掌柜智库 mongo_history_utils（同样是"按 session 存取最近 N 条"），但改成异步。

惰性：import 不连库；首次 await 调用才建 AsyncIOMotorClient。

接口契约（INTERFACES，async，签名不可变）：
- async load(user_id, session_id, limit=10) -> list[dict]   （按时间正序，可直接喂 LLM）
- async append(user_id, session_id, role, content) -> None
"""
from __future__ import annotations

import time
from typing import Any

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class MongoHistoryClient:
    """MongoDB 会话历史封装（惰性连接 + 读取最近 N 条 / 追加一条）。

    文档结构::

        {user_id, session_id, role("user"/"assistant"), content, ts(秒级时间戳)}

    用法::

        mh = MongoHistoryClient()
        await mh.append("u1", "s1", "user", "增值税税率是多少")
        history = await mh.load("u1", "s1", limit=10)

    :return: 见各方法说明。
    """

    def __init__(self) -> None:
        """构造仅保存配置，不建连接（惰性）。"""
        self._client = None        # AsyncIOMotorClient
        self._collection = None    # 历史集合
        self._index_ready = False  # 索引是否已确保（惰性建索引）
        logger.info("[Mongo客户端] 初始化（未连接）：uri=%s db=%s",
                    settings.mongo_uri, settings.mongo_db)

    # ---------------- 惰性连接 ----------------
    def _get_collection(self):
        """惰性建立 motor 连接并取到历史集合。

        :return: motor 的 collection 对象。
        :raise RuntimeError: 未安装 motor，或连接失败。
        """
        if self._collection is None:
            try:
                from motor.motor_asyncio import AsyncIOMotorClient
            except ImportError as e:
                raise RuntimeError(
                    "[Mongo客户端] 未安装 motor，请 `pip install motor`"
                ) from e
            try:
                # serverSelectionTimeoutMS：快速失败，避免无 Mongo 时长时间挂起。
                self._client = AsyncIOMotorClient(
                    settings.mongo_uri, serverSelectionTimeoutMS=5000
                )
                db = self._client[settings.mongo_db]
                self._collection = db[settings.mongo_history_collection]
                logger.info("[Mongo客户端] 连接对象已建立：db=%s coll=%s",
                            settings.mongo_db, settings.mongo_history_collection)
            except Exception as e:
                logger.error("[Mongo客户端] 连接失败：%s", e, exc_info=True)
                raise RuntimeError(f"[Mongo客户端] 连接 Mongo 失败：{e}") from e
        return self._collection

    async def _ensure_index(self) -> None:
        """惰性确保复合索引：(user_id, session_id, ts)，适配"按会话取最近记录"。

        create_index 自带幂等性，重复调用不会重复建；用标志位避免每次调用都发请求。
        """
        if self._index_ready:
            return
        coll = self._get_collection()
        try:
            # ts 升序：load 时按时间正序取，正好是 LLM 需要的上下文顺序。
            await coll.create_index([("user_id", 1), ("session_id", 1), ("ts", 1)])
            self._index_ready = True
            logger.info("[Mongo客户端] 历史集合索引已确保")
        except Exception as e:
            # 建索引失败不阻断读写（无索引只是慢），记录告警即可。
            logger.warning("[Mongo客户端] 建索引失败（不影响读写）：%s", e)

    # ---------------- 读取 ----------------
    async def load(self, user_id: str, session_id: str, limit: int = 10) -> list[dict]:
        """读取某会话最近 limit 条历史，按时间正序返回（可直接拼进 LLM 上下文）。

        :param user_id: 用户ID。
        :param session_id: 会话ID。
        :param limit: 最多返回条数（默认 10）。
        :return: list[dict(role, content)]；失败返回 []（不让记忆故障拖垮问答）。
        """
        await self._ensure_index()
        coll = self._get_collection()
        try:
            query = {"user_id": user_id, "session_id": session_id}
            # 先按 ts 降序取最近 limit 条（拿到的是"最新的几条"），再在内存里反转成正序。
            cursor = coll.find(query).sort("ts", -1).limit(limit)
            docs = await cursor.to_list(length=limit)
            docs.reverse()  # 翻成时间正序（旧->新），符合对话上下文顺序
            history = [{"role": d.get("role", ""), "content": d.get("content", "")} for d in docs]
            logger.info("[Mongo客户端] 读取历史完成：user=%s session=%s 共 %d 条",
                        user_id, session_id, len(history))
            return history
        except Exception as e:
            logger.error("[Mongo客户端] 读取历史失败：%s", e, exc_info=True)
            return []

    # ---------------- 追加 ----------------
    async def append(self, user_id: str, session_id: str, role: str, content: str) -> None:
        """追加一条会话记录。

        :param user_id: 用户ID。
        :param session_id: 会话ID。
        :param role: 角色，"user" 或 "assistant"。
        :param content: 消息内容。
        :return: 无返回；写入失败仅记录日志，不抛异常（记忆故障不应中断问答主流程）。
        """
        await self._ensure_index()
        coll = self._get_collection()
        doc = {
            "user_id": user_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "ts": time.time(),  # 秒级时间戳，用于排序
        }
        try:
            await coll.insert_one(doc)
            logger.info("[Mongo客户端] 追加历史完成：user=%s session=%s role=%s",
                        user_id, session_id, role)
        except Exception as e:
            # 写历史失败不抛：问答主流程比"记住历史"重要，降级即可。
            logger.error("[Mongo客户端] 追加历史失败：%s", e, exc_info=True)


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：需要 MongoDB 在线。
    import asyncio

    async def _demo():
        mh = MongoHistoryClient()
        try:
            await mh.append("u_test", "s_test", "user", "你好")
            await mh.append("u_test", "s_test", "assistant", "你好，我是税务助手")
            hist = await mh.load("u_test", "s_test", limit=5)
            print("[mongo_client 自测] 历史 =>", hist)
        except Exception as exc:
            print("[mongo_client 自测] 需要 Mongo 在线（属预期）=>", exc)

    asyncio.run(_demo())
