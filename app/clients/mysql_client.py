"""MySQL 异步客户端（SQLAlchemy async engine + aiomysql，惰性连接）。
本模块在整体链路里的位置：基础设施层，专为"经营数据查询类"意图的 Text2SQL 服务。
Text2SQL Agent 生成 SQL 后由本模块在数仓上执行，拿到列名+数据行；同时提供
list_tables / table_schema 给 schema linking 兜底与 SQL 校验使用。

为什么用异步(async) + SQLAlchemy async engine：
- 整条问答链路是 async（FastAPI + LangGraph + SSE 流式），DB 访问也用 async 才不阻塞事件循环。
- SQLAlchemy 的 create_async_engine 自带连接池(pool)，并发查询复用连接，比每次裸连 aiomysql 稳。
- 连接串走 settings.mysql_dsn（mysql+aiomysql://...），账号地址全部配置化，绝不硬编码。

安全注意：execute() 直接执行传入 SQL（Text2SQL 产物）。生产中应在上层(Text2SQL 的 validate 节点)
做"只读校验"——只允许 SELECT，禁止 DDL/DML。本客户端聚焦"连接与执行"，校验交给 Agent 层。

纵深防御（双层防护）：除了 Agent 层的 AST 只读校验，本客户端在 execute() 还支持"DB 侧只读账号"。
当 settings 配了 mysql_readonly_user（非空）时，execute() 会改用一个独立、惰性的"只读 engine"
（连接串走 settings.mysql_readonly_dsn）；该账号在 MySQL 里只被授予 SELECT 权限，即便上层校验被
绕过也无法 DDL/DML。默认 mysql_readonly_user 为空 => mysql_readonly_dsn 回退主 DSN，且本客户端
不会单独建只读 engine、直接复用主 engine，因此默认行为与现状【完全一致】。
（元数据方法 list_tables/table_schema 读 information_schema，仍走主 engine，不受只读开关影响。）

惰性：import 不连库；首次 await 调用才创建 async engine。

接口契约（INTERFACES，async，签名不可变）：
- async execute(sql) -> (columns: list[str], rows: list[list])
- async list_tables() -> list[str]
- async table_schema(name) -> list[dict]   （字段名/类型/注释等）
"""
from __future__ import annotations

from typing import Any

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class MySQLClient:
    """MySQL 异步客户端封装（惰性建 engine + 执行/取 schema）。

    用法::

        my = MySQLClient()
        cols, rows = await my.execute("SELECT * FROM dwd_invoice LIMIT 10")
        tables = await my.list_tables()
        schema = await my.table_schema("dwd_invoice")

    :return: 见各方法说明。
    """

    def __init__(self) -> None:
        """构造仅保存配置，不建 engine（惰性）。"""
        self._engine = None  # SQLAlchemy AsyncEngine（主账号）
        # 只读 engine：仅当 settings 配了只读账号时才会被惰性创建；默认 None 且永不创建。
        self._readonly_engine = None  # SQLAlchemy AsyncEngine（Text2SQL 只读账号）
        logger.info("[MySQL客户端] 初始化（未连接）：%s:%s db=%s",
                    settings.mysql_host, settings.mysql_port, settings.mysql_db)

    # ---------------- 惰性 engine ----------------
    def _create_engine(self, dsn: str):
        """按给定连接串惰性创建一个 SQLAlchemy async engine（带连接池）。

        抽成公共方法，供"主 engine"与"只读 engine"复用，避免连接池参数重复维护。

        :param dsn: 连接串（mysql_dsn 或 mysql_readonly_dsn）。
        :return: AsyncEngine 实例。
        :raise RuntimeError: 未安装 SQLAlchemy / aiomysql，或创建失败。
        """
        try:
            from sqlalchemy.ext.asyncio import create_async_engine
        except ImportError as e:
            raise RuntimeError(
                "[MySQL客户端] 未安装 SQLAlchemy(async)，请 `pip install 'SQLAlchemy>=2.0' aiomysql`"
            ) from e
        try:
            # pool_pre_ping：取连接前先 ping，自动剔除 MySQL 8h 超时断开的死连接。
            # pool_recycle：超过该秒数的连接主动回收，避免 "MySQL server has gone away"。
            engine = create_async_engine(
                dsn,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False,
            )
            logger.info("[MySQL客户端] async engine 已创建：%s:%s/%s",
                        settings.mysql_host, settings.mysql_port, settings.mysql_db)
            return engine
        except Exception as e:
            logger.error("[MySQL客户端] 创建 engine 失败：%s", e, exc_info=True)
            raise RuntimeError(f"[MySQL客户端] 创建 MySQL engine 失败：{e}") from e

    def _get_engine(self):
        """惰性创建/返回主账号 async engine（list_tables/table_schema 等都走它）。

        :return: AsyncEngine 实例。
        :raise RuntimeError: 未安装 SQLAlchemy / aiomysql，或创建失败。
        """
        if self._engine is None:
            self._engine = self._create_engine(settings.mysql_dsn)
        return self._engine

    def _get_exec_engine(self):
        """返回"执行 Text2SQL 用"的 engine——配了只读账号则用只读 engine，否则用主 engine。

        这是纵深防御的开关点：
        - 未配只读账号（settings.mysql_readonly_user 为空，默认）：直接复用主 engine，
          不创建任何额外 engine，行为与现状完全一致。
        - 配了只读账号：惰性创建独立的只读 engine（连接串走 settings.mysql_readonly_dsn），
          此后 execute() 都用只读账号连库，即便上层校验被绕过也只有 SELECT 权。

        :return: AsyncEngine 实例（只读或主）。
        :raise RuntimeError: 未安装 SQLAlchemy / aiomysql，或创建失败。
        """
        if not settings.mysql_readonly_user:
            # 默认路径：无只读账号配置，沿用主 engine（行为不变）。
            return self._get_engine()
        if self._readonly_engine is None:
            self._readonly_engine = self._create_engine(settings.mysql_readonly_dsn)
            logger.info("[MySQL客户端] 已启用 Text2SQL 只读账号 engine：user=%s（DB侧只读纵深防御）",
                        settings.mysql_readonly_user)
        return self._readonly_engine

    # ---------------- 执行 SQL ----------------
    async def execute(self, sql: str) -> tuple[list[str], list[list[Any]]]:
        """执行一条 SQL，返回 (列名列表, 数据行列表)。

        :param sql: 待执行 SQL（通常是 Text2SQL 生成且经校验的 SELECT）。
        :return: (columns, rows)；columns 为列名，rows 为 list[list]（每行一个 list）。
                 非查询语句(无返回行)时 columns/rows 为空列表。
        :raise RuntimeError: 执行失败（语法错/表不存在/连接异常）。
        """
        from sqlalchemy import text  # 惰性导入

        # 执行入口：配了只读账号则走只读 engine，否则走主 engine（默认，行为不变）。
        engine = self._get_exec_engine()
        logger.info("[MySQL客户端] 执行 SQL：%s", sql)
        try:
            async with engine.connect() as conn:
                result = await conn.execute(text(sql))
                # returns_rows：SELECT 等才有返回行；INSERT/UPDATE 等没有。
                if not result.returns_rows:
                    logger.info("[MySQL客户端] SQL 执行完成（无返回行）")
                    return [], []
                columns = list(result.keys())
                # 转成 list[list]，便于 JSON 序列化与前端表格渲染
                rows = [list(r) for r in result.fetchall()]
                logger.info("[MySQL客户端] SQL 执行完成：列=%d，行=%d", len(columns), len(rows))
                return columns, rows
        except Exception as e:
            logger.error("[MySQL客户端] SQL 执行失败：%s", e, exc_info=True)
            raise RuntimeError(f"[MySQL客户端] SQL 执行失败：{e}") from e

    # ---------------- 元数据 ----------------
    async def list_tables(self) -> list[str]:
        """列出当前库所有表名（schema linking 兜底用）。

        :return: 表名列表；失败返回 []。
        """
        from sqlalchemy import text

        engine = self._get_engine()
        # 用 information_schema 按当前库过滤，避免 SHOW TABLES 在多库环境下的歧义。
        sql = (
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = :db ORDER BY table_name"
        )
        try:
            async with engine.connect() as conn:
                result = await conn.execute(text(sql), {"db": settings.mysql_db})
                tables = [row[0] for row in result.fetchall()]
                logger.info("[MySQL客户端] 列出表完成：共 %d 张", len(tables))
                return tables
        except Exception as e:
            logger.error("[MySQL客户端] 列出表失败：%s", e, exc_info=True)
            return []

    async def table_schema(self, name: str) -> list[dict]:
        """获取指定表的字段结构（列名/类型/可空/注释），供 SQL 生成与校验参考。

        :param name: 表名。
        :return: list[dict(column, type, nullable, comment)]；失败返回 []。
        """
        from sqlalchemy import text

        engine = self._get_engine()
        sql = (
            "SELECT column_name, column_type, is_nullable, column_comment "
            "FROM information_schema.columns "
            "WHERE table_schema = :db AND table_name = :tbl ORDER BY ordinal_position"
        )
        try:
            async with engine.connect() as conn:
                result = await conn.execute(text(sql), {"db": settings.mysql_db, "tbl": name})
                cols = [
                    {
                        "column": row[0],
                        "type": row[1],
                        "nullable": row[2],
                        "comment": row[3] or "",  # 注释对 NL->SQL 很关键，缺失补空串
                    }
                    for row in result.fetchall()
                ]
                logger.info("[MySQL客户端] 取表结构完成：表=%s，字段=%d", name, len(cols))
                return cols
        except Exception as e:
            logger.error("[MySQL客户端] 取表结构失败：%s", e, exc_info=True)
            return []

    async def close(self) -> None:
        """释放连接池（FastAPI 关闭时调用，优雅收尾）。主 + 只读两个 engine 都要释放。"""
        if self._engine is not None:
            await self._engine.dispose()
            logger.info("[MySQL客户端] engine 已释放")
        if self._readonly_engine is not None:
            await self._readonly_engine.dispose()
            logger.info("[MySQL客户端] 只读 engine 已释放")


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：需要 MySQL 在线。
    import asyncio

    async def _demo():
        my = MySQLClient()
        try:
            tables = await my.list_tables()
            print("[mysql_client 自测] 表数量 =>", len(tables))
        except Exception as exc:
            print("[mysql_client 自测] 需要 MySQL 在线（属预期）=>", exc)
        finally:
            await my.close()

    asyncio.run(_demo())
