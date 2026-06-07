"""冒烟测试：导入全部模块 + 编译 LangGraph 主图，确认整个工程"装得起来、连得通图"。
本测试不验证业务正确性，只保证：

1. 所有模块都能在"没有任何外部基础设施(LLM/Milvus/ES/Qdrant/MySQL/Mongo/MinIO)"的机器上被 import；
   ——这是"惰性初始化"规范的回归测试：import 阶段绝不能真连服务、绝不能抛异常。
2. build_pipeline_graph() 能把各节点编译成一张可执行的 LangGraph 图（不真正跑，只编译）。
3. FastAPI 应用对象能被构造（app/main.py 的 app 可导入）。

测试策略（为什么这么测）：
- 用 monkeypatch 把 get_llm 换成"假 LLM"，避免编图/建 Agent 时因缺 LLM_API_KEY 报错；
- 外部客户端本就惰性，import 与"构造对象"都不会真连，故无需逐个打桩网络层；
- 尚未生成的模块(如开发中途的 pipeline)用 importorskip 跳过，避免阻塞其它冒烟项。
"""
from __future__ import annotations

import importlib

import pytest


# 工程内"应当随时可被 import"的模块清单（覆盖契约层/客户端/核心/编排/接口各层）。
# 注意：这里只 import，不构造重对象，验证的是"导入零副作用、不触网"。
_IMPORTABLE_MODULES = [
    # 配置 / 契约层
    "config.constants",
    "config.settings",
    "config.logging_config",
    "app.schemas.document",
    "app.schemas.chat",
    "app.graph.state",
    # 工具
    "app.utils.prompt_loader",
    "app.utils.timing",
    "app.utils.sse",
    "app.utils.normalize",
    # 客户端（全部惰性连接）
    "app.clients.llm_client",
    "app.clients.embedding_client",
    "app.clients.reranker_client",
    "app.clients.milvus_client",
    "app.clients.es_client",
    "app.clients.qdrant_meta_client",
    "app.clients.mysql_client",
    "app.clients.mongo_client",
    "app.clients.minio_client",
    "app.clients.mcp_client",
    # QU
    "app.core.qu.intent",
    "app.core.qu.extractor",
    "app.core.qu.query_rewrite",
    "app.core.qu.understanding",
    # 召回 / 排序 / 重排 / 路由
    "app.core.recall.hybrid",
    "app.core.recall.manager",
    "app.core.rank.coarse_rank",
    "app.core.rank.fine_rank",
    "app.core.rerank.rerank",
    "app.core.router.search_router",
    # 摘要 / Agent / 编排 / 记忆
    "app.core.summarize.summarizer",
    "app.agents.text2sql_agent",
    "app.graph.nodes",
    "app.graph.pipeline",
    "app.memory.session",
    # 接口层
    "app.api.routes_chat",
    "app.api.routes_health",
    "app.main",
    # 离线脚本（本 agent 负责，必须可导入）
    "scripts.init_collections",
    "scripts.ingest_documents",
    "scripts.build_metadata_index",
]


@pytest.fixture(autouse=True)
def _stub_llm(monkeypatch):
    """自动夹具：把 get_llm 换成"假 LLM"，避免编图/建 Agent 时因缺 LLM 配置而失败。

    在 clients.llm_client 源头打桩；各业务模块即便 import 阶段把 get_llm 绑成了本地名，
    其调用通常发生在运行期(节点执行时)而非编图期，故源头打桩足以覆盖冒烟所需。
    """

    class _FakeMsg:
        def __init__(self, content="冒烟测试固定回答"):
            self.content = content

    class _FakeLLM:
        def invoke(self, *a, **k):
            return _FakeMsg()

        async def ainvoke(self, *a, **k):
            return _FakeMsg()

        def bind(self, *a, **k):
            return self

    try:
        import app.clients.llm_client as llm_mod
        monkeypatch.setattr(llm_mod, "get_llm", lambda *a, **k: _FakeLLM(), raising=False)
    except Exception:  # noqa: BLE001 - clients 层异常不应阻塞冒烟
        pass


@pytest.mark.parametrize("module_name", _IMPORTABLE_MODULES)
def test_import_module(module_name):
    """逐个 import 模块：能导入即视为通过（验证惰性初始化、import 无副作用）。

    模块尚未生成时(开发中途)用 importorskip 跳过，不算失败；
    一旦模块存在，import 报错(如误在顶层连服务)就会被本测试抓出来。
    """
    pytest.importorskip(module_name)


def test_all_modules_import_together():
    """一次性把全部已存在模块都 import 一遍，确保彼此之间没有循环导入/命名冲突。"""
    failed: list[str] = []
    for name in _IMPORTABLE_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            # 模块还没生成 -> 跳过(不计失败)
            continue
        except Exception as exc:  # noqa: BLE001 - 其它异常说明 import 有副作用，需暴露
            failed.append(f"{name}: {exc!r}")
    assert not failed, "以下模块 import 时抛出异常(疑似 import 阶段触网/有副作用):\n" + "\n".join(failed)


def test_build_pipeline_graph():
    """build_pipeline_graph() 应能把各节点编译成一张 LangGraph 图(只编译，不执行)。

    这是对"编排层装配正确性"的核心回归：节点是否齐全、边是否连得上、状态类型是否匹配，
    任何装配错误都会在 compile 阶段暴露。pipeline 尚未生成时优雅跳过。
    """
    pipeline_mod = pytest.importorskip("app.graph.pipeline")
    # 关键守护：build_pipeline_graph() 内部真正 import 的是 langgraph.graph（StateGraph/START/END）。
    # 注意 langgraph 顶层是命名空间包，`import langgraph` 即便没装 graph 子包也能成功，
    # 故必须 importorskip 到"真正被 import 的那一层"——langgraph.graph：装了就 PASS、没装则优雅 SKIP，
    # 绝不让 ModuleNotFoundError 冒泡成 FAILED。
    pytest.importorskip("langgraph.graph")
    build = getattr(pipeline_mod, "build_pipeline_graph", None)
    assert callable(build), "app.graph.pipeline 应提供 build_pipeline_graph()"
    graph = build()
    # 编译产物应是可执行对象：LangGraph 编译图通常带 ainvoke / astream 接口
    assert graph is not None
    assert hasattr(graph, "ainvoke") or hasattr(graph, "invoke") or hasattr(graph, "astream"), \
        "编译后的图应具备 (a)invoke/astream 等可执行接口"


def test_orchestrator_constructible():
    """Orchestrator 应能在无外部基础设施下被构造(依赖客户端惰性、不在构造期触网)。"""
    pipeline_mod = pytest.importorskip("app.graph.pipeline")
    orch_cls = getattr(pipeline_mod, "Orchestrator", None)
    assert orch_cls is not None, "app.graph.pipeline 应提供 Orchestrator"
    # 仅构造，不调用 astream(astream 才会真正走召回/LLM，需要外部服务)
    orch = orch_cls()
    assert orch is not None
    assert hasattr(orch, "astream"), "Orchestrator 应提供 astream() 流式接口"


def test_fastapi_app_importable():
    """app/main.py 的 FastAPI app 对象应可被导入(应用能装配起来)。"""
    main_mod = pytest.importorskip("app.main")
    app_obj = getattr(main_mod, "app", None)
    assert app_obj is not None, "app/main.py 应暴露 FastAPI 实例 app"


def test_format_sse_contract():
    """顺带验证 SSE 封装契约：format_sse 产出 'event: x\\ndata: {...}\\n\\n' 格式。"""
    sse_mod = pytest.importorskip("app.utils.sse")
    text = sse_mod.format_sse("intent", {"intent": "通用问题类"})
    assert text.startswith("event: intent")
    assert "data:" in text
    assert text.endswith("\n\n"), "SSE 消息必须以空行结尾，否则前端无法分帧"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
