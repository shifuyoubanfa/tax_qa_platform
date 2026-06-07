"""FastAPI 应用入口。
本模块在整体链路里的位置：整个平台的"总开关"。它负责：
1. 创建 FastAPI 应用实例；
2. 通过 lifespan(生命周期钩子) 在进程启动时初始化日志、打印启动横幅、（可选）预热；
   并在关闭时打印收尾日志；
3. 注册 CORS 跨域中间件；
4. 用 include_router 汇总挂载各功能域路由（健康检查 + 问答）；
5. 提供 `python -m app.main` / 直接运行本文件 的本地启动方式。

设计要点（为什么这么做）：
- 用新式 lifespan(异步上下文管理器) 而非已弃用的 @app.on_event("startup")：
  它把"启动准备"与"关闭清理"写在同一个函数里，逻辑集中、官方推荐。
- 所有外部依赖一律【惰性初始化】（首次使用才连），故 lifespan 里不强连任何 infra，
  保证在没有 Milvus/ES/MySQL 的开发机上 FastAPI 也能正常起来——这正是本平台
  "import 不报错、空 infra 可启动"硬性要求的落点。
- setup_logging() 放在 lifespan 最前面执行：确保此后任何模块打的日志都有 handler、
  不丢日志，并与 uvicorn 的日志体系并存。

风格对标 掌柜智库 的配置驱动 + 爱搜税 app.py 的"轻入口、业务下沉"。
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config.paths import PROJECT_ROOT
from config.settings import settings
from config.logging_config import setup_logging, get_logger
from app.api import routes_chat, routes_health
from app.graph import nodes

# 注意：此处先不取 logger 实例的"使用"，只是声明；真正打日志在 lifespan 里
# （那时 setup_logging 已执行，handler 已就绪）。get_logger 内部有兜底配置，故安全。
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期钩子：yield 之前是"启动逻辑"，yield 之后是"关闭逻辑"。

    :param app: 当前 FastAPI 实例（FastAPI 调用时注入，本函数暂未使用，留作扩展）。
    :yield: 控制权交还给框架开始对外服务；框架收到关闭信号后继续执行 yield 之后的清理。
    """
    # ---------- 启动阶段 ----------
    # 第一件事就是初始化日志系统，确保之后所有模块的日志都能正常落盘+输出到控制台。
    setup_logging()
    logger.info("=" * 60)
    logger.info("税务智能问答平台 启动中 ...")
    logger.info("监听地址 => http://%s:%s", settings.app_host, settings.app_port)
    logger.info("LLM模型 => %s | 召回阈值 recall=%s fine=%s rerank=%s",
                settings.llm_model, settings.recall_topk, settings.fine_topk, settings.rerank_topk)
    logger.info("提示：所有外部依赖均惰性连接，无 infra 时服务仍可启动。")
    logger.info("=" * 60)
    # 可选预热位（留槽位，不强连）：
    # 如需在启动时预热 embedding / LLM 客户端以降低首请求延迟，可在此调用对应 client 的
    # 惰性 getter 触发一次构建。这里默认不做，以保证"空 infra 也能启动"的硬性要求。

    yield  # ====== 应用在此期间正常对外提供服务 ======

    # ---------- 关闭阶段 ----------
    # 优雅回收各 Agent 惰性建立的 MySQL 连接池（无 infra/未建连时为空操作）。
    # 用 try/except 包裹：清理异常绝不阻断进程正常退出。
    try:
        await nodes.close_all()
    except Exception as e:  # noqa: BLE001 关闭清理异常不影响退出
        logger.error("关闭阶段回收连接池异常（已忽略）：%s", e, exc_info=True)
    logger.info("税务智能问答平台 正在关闭，资源清理完成。")


def create_app() -> FastAPI:
    """工厂函数：创建并装配 FastAPI 应用。

    做成工厂函数（而非模块级直接建实例后立刻配置）的好处：测试时可重复创建干净实例、
    便于将来按环境差异化装配。

    :return: 装配完成（含中间件与路由）的 FastAPI 实例。
    """
    application = FastAPI(
        title="税务智能问答平台",
        description="融合 混合检索RAG + Text2SQL 的税务领域智能问答服务（SSE流式）。",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS 跨域中间件：开发期放开全部来源，方便前端本地联调。
    # 【生产环境务必收紧】：把 allow_origins 改为前端实际域名白名单，
    # 并视情况关闭 allow_credentials + "*" 的组合（浏览器规范不允许二者同时为通配）。
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],      # 生产应改为具体域名白名单
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 汇总挂载路由：健康检查 + 核心问答。
    application.include_router(routes_health.router)
    application.include_router(routes_chat.router)

    # ------- 同源部署：把前端静态站点挂到根路径 '/'（与 /api 同源，前端 fetch 用相对路径，无需 CORS）-------
    # 【为什么必须放在所有 include_router 之后】：StaticFiles 挂在 '/' 会贪婪匹配所有路径前缀，
    # 若先挂它，'/api/...' 也会被它吞掉导致 404。先注册 API 路由、再兜底挂静态，才能让
    # /api/* 命中接口、其余路径(/, /index.html, /static...)回落到前端。
    # html=True：访问 '/' 自动返回目录下的 index.html（单页应用入口）。
    # 用 PROJECT_ROOT 拼绝对路径，保证"从任何目录、任何方式启动"都能定位到 app/web。
    web_dir = PROJECT_ROOT / "app" / "web"
    if web_dir.is_dir():  # 惰性/健壮：目录不存在时不挂载，避免启动期 RuntimeError，不破坏现有路由
        application.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
        logger.info("[静态站点] 已挂载前端目录 => %s", web_dir)
    else:
        logger.warning("[静态站点] 未找到前端目录，跳过挂载 => %s", web_dir)

    return application


# 模块级应用实例：uvicorn 通过 "app.main:app" 这个导入路径加载它。
app = create_app()


if __name__ == "__main__":
    # 最小自测/本地启动块（仅供单文件学习运行）：
    # 直接 `python app/main.py` 即可起服务；生产部署一般用
    # `uvicorn app.main:app --host 0.0.0.0 --port 8000` 由进程管理器拉起。
    import uvicorn

    # 用 "app.main:app" 字符串而非直接传 app 对象，才能支持 reload 等特性。
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port)
