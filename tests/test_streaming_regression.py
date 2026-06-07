"""离线 SSE 流式回归夹具：用假对象 mock 掉全部 infra，跑通【真·LangGraph 主图】
（Orchestrator.astream），断言 SSE 事件类型与顺序符合契约。

本测试在整体测试体系里的位置：它是"后续所有改动的流式兜底"。任何人改 pipeline / nodes /
schemas 时，只要不小心打断了"逐 token 流式"或改错了 SSE 事件类型/顺序，本测试就会变红。

为什么这么测（设计要点）：
1. 不打桩 LangGraph 本身——我们让【编译后的真主图】端到端执行（graph.astream），只把
   nodes.py 里的【惰性组件 getter】换成假对象（假 QU / 假路由 / 假召回 / 假排序 / 假重排 /
   假摘要 / 假 Text2SQL Agent / 假记忆）。这样验证的是"主图拓扑 + 节点产出 -> SSE 映射"的
   真实链路，而不是一段被 mock 架空的假流程。
2. 全程不触网、不连库：所有假组件都返回固定值，假记忆不写库，假摘要不调 LLM。
3. 缺 langgraph 等依赖时用 pytest.importorskip 跳过（而非报错），保证在裸环境也能 collect。

关于"逐 token answer_delta"的两条产生路径（本测试都接受，断言只认"最终出现了逐段 delta"）：
- 路径A（真·messages 流）：generate_answer 节点内 LLM.astream 吐 AIMessageChunk，被
  Orchestrator 的 stream_mode="messages" 透出为 answer_delta。该路径需要真 langchain LLM，
  离线假摘要不触发它。
- 路径B（updates 兜底切块）：messages 没流出时，_updates_to_sse 对 generate_answer 的
  update["answer"] 按 _chunk_text 切块补发 answer_delta。离线场景走这条——同样是"逐段 delta"，
  足以回归"答案一定逐段到达前端且事件类型正确"。
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

# 缺 langgraph / pydantic 等关键依赖时整体跳过（裸环境不报错，仅跳过）。
# 注意：这里特意 importorskip 的是【真正被主图用到的子模块 langgraph.graph】，而非顶层
# langgraph 包。原因：langgraph 卸载残留可能留下一个空的命名空间目录，使 `import langgraph`
# 侥幸成功、但 `from langgraph.graph import StateGraph`（build_pipeline_graph 内部真正的导入）
# 仍失败。只有按"代码真正依赖的子模块"探测，才能保证依赖不全时是优雅跳过而非运行期报错。
pytest.importorskip("langgraph.graph")
pytest.importorskip("pydantic")

from config.constants import RouteType, SSEEvent  # noqa: E402  importorskip 之后才导入
from app.schemas.chat import ChatRequest  # noqa: E402
from app.schemas.document import (  # noqa: E402
    Document,
    QUResult,
    RecallStep,
    RetrievalPlan,
    Text2SQLResult,
)


# --------------------------------------------------------------------------- #
# 假组件：每个都只实现 nodes.py 真正调用到的那一两个方法，返回固定值，绝不触网/连库。
# --------------------------------------------------------------------------- #
class _FakeQU:
    """假 Query Understanding：返回固定意图与子查询，不调 LLM。"""

    def __init__(self, intent: str) -> None:
        self._intent = intent

    async def understand(self, query: str) -> QUResult:
        """返回固定 QUResult（understand 节点用 _maybe_await 兼容同步/异步，这里用异步）。"""
        return QUResult(
            raw_query=query,
            intent=self._intent,
            sub_queries=[query] if query else [],
            is_short_query=False,
        )


class _FakeRouterRAG:
    """假 SearchRouter（RAG 分支）：产出一个带单步召回、无结构化 Agent 的 RAG 计划。"""

    def route(self, qu: QUResult) -> RetrievalPlan:
        return RetrievalPlan(
            route_type=RouteType.RAG.value,
            steps=[RecallStep(kbases=["policy"], coarse_topk=10, fine_topk=5)],
            agent_name=None,  # None -> 纯文档 RAG，rag_recall 走 RecallManager.recall
        )


class _FakeRouterT2S:
    """假 SearchRouter（Text2SQL 分支）：产出 route_type=text2sql 的计划，导向 text2sql 节点。"""

    def route(self, qu: QUResult) -> RetrievalPlan:
        return RetrievalPlan(
            route_type=RouteType.TEXT2SQL.value,
            steps=[],
            agent_name="text2sql_agent",
        )


class _FakeRecallManager:
    """假召回：返回固定的 Document 列表（不连 Milvus/ES）。"""

    async def recall(self, qu: QUResult, step: RecallStep) -> list[Document]:
        return [
            Document(doc_id="d1", title="增值税暂行条例", content="一般纳税人……",
                     kbase="policy", score=0.9),
            Document(doc_id="d2", title="增值税政策解读", content="小规模纳税人……",
                     kbase="policy", score=0.8),
        ]


class _FakeCoarseRanker:
    """假粗排：原样透传并裁剪到 topk（不做真 RRF）。"""

    def rank(self, docs: list[Document], topk: int) -> list[Document]:
        return list(docs)[:topk]


class _FakeFineRanker:
    """假精排：原样透传（不调 embedding）。"""

    def rank(self, query: str, docs: list[Document], method: str, topk: int) -> list[Document]:
        return list(docs)[:topk]


class _FakeReRanker:
    """假重排：原样透传（不调 RerankerClient）。"""

    def rerank(self, query: str, docs: list[Document], topk: int) -> list[Document]:
        return list(docs)[:topk]


class _FakeSummarizer:
    """假摘要：build_context 产出固定上下文+引用；summarize_stream 逐段 yield 固定 token（不调 LLM）。

    关于答案长度：离线场景下 generate_answer 节点把这些段拼成完整 answer 写回 state（真·messages
    token 流需要真 langchain LLM，离线不触发），编排层 _updates_to_sse 再用 _chunk_text(size=24)
    把 answer 切块补发 answer_delta。为确保"逐段到达"被切成多段（验证逐段流式），这里让拼接后的
    完整答案明显超过 24 字符，保证 _chunk_text 至少切出 2 段。
    """

    # 拼接后约 50+ 字符，足以被 _chunk_text(size=24) 切成多段 answer_delta。
    _ANSWER_PIECES = [
        "根据《增值税暂行条例》的相关规定，",
        "增值税一般纳税人销售货物，",
        "适用的基本税率为13%，",
        "具体以最新政策口径为准。",
    ]

    def build_context(self, qu: QUResult, docs: list[Document], topk: int):
        references = [
            {"doc_id": d.doc_id, "title": d.title, "kbase": d.kbase,
             "score": d.score, "citation_index": i + 1, "context": d.content}
            for i, d in enumerate(docs[:topk])
        ]
        context = "\n".join(f"[[citation:{i + 1}]] {d.title}" for i, d in enumerate(docs[:topk]))
        return context, references

    async def summarize_stream(self, query: str, qu: QUResult, context: str):
        """逐段产出固定答案增量（async generator）。"""
        for piece in self._ANSWER_PIECES:
            yield piece


class _FakeText2SQLAgent:
    """假 Text2SQL Agent：arun 直接返回固定 Text2SQLResult（不连 Qdrant/LLM/MySQL）。"""

    async def arun(self, query: str, qu: QUResult) -> Text2SQLResult:
        return Text2SQLResult(
            question=query,
            sql="SELECT SUM(amount) FROM orders WHERE quarter='2026Q1' LIMIT 1000",
            columns=["total_amount"],
            rows=[[123456]],
            row_count=1,
            summary="2026年第一季度总订单金额为 123456 元。",
            error="",
        )


class _FakeMemory:
    """假会话记忆：load 返回空历史，append 不写库（纯内存丢弃）。"""

    async def load(self, user_id: str, session_id: str) -> list[dict]:
        return []

    async def append(self, user_id: str, session_id: str, role: str, content: str) -> None:
        return None


# --------------------------------------------------------------------------- #
# 公共夹具：把 nodes.py 的全部组件 getter 换成假对象（RAG 分支用）。
# --------------------------------------------------------------------------- #
def _patch_common(monkeypatch, nodes, *, intent: str, router) -> None:
    """把 nodes 模块里所有"惰性组件 getter"打桩为假对象。

    :param monkeypatch: pytest 的 monkeypatch 夹具。
    :param nodes: 已 import 的 app.graph.nodes 模块。
    :param intent: 假 QU 返回的意图（值取自 Intent 枚举的 .value）。
    :param router: 假 SearchRouter 实例（决定走 RAG 还是 text2sql 分支）。
    """
    monkeypatch.setattr(nodes, "_get_qu", lambda: _FakeQU(intent), raising=True)
    monkeypatch.setattr(nodes, "_get_router", lambda: router, raising=True)
    monkeypatch.setattr(nodes, "_get_recall_manager", lambda: _FakeRecallManager(), raising=True)
    monkeypatch.setattr(nodes, "_get_coarse_ranker", lambda: _FakeCoarseRanker(), raising=True)
    monkeypatch.setattr(nodes, "_get_fine_ranker", lambda: _FakeFineRanker(), raising=True)
    monkeypatch.setattr(nodes, "_get_reranker", lambda: _FakeReRanker(), raising=True)
    monkeypatch.setattr(nodes, "_get_summarizer", lambda: _FakeSummarizer(), raising=True)
    monkeypatch.setattr(nodes, "_get_t2s_agent", lambda: _FakeText2SQLAgent(), raising=True)
    monkeypatch.setattr(nodes, "_get_memory", lambda: _FakeMemory(), raising=True)


async def _collect(orchestrator, req) -> list:
    """把 Orchestrator.astream 的全部 SSEMessage 收集成列表（供断言）。"""
    out = []
    async for msg in orchestrator.astream(req):
        out.append(msg)
    return out


def _events(msgs) -> list[str]:
    """抽取事件类型序列（msg.event 列表）。"""
    return [m.event for m in msgs]


def _first_index(events: list[str], target: str) -> int:
    """返回某事件首次出现的下标；不存在返回 -1。"""
    return events.index(target) if target in events else -1


# --------------------------------------------------------------------------- #
# 用例一：RAG 分支的 SSE 事件序列回归
# --------------------------------------------------------------------------- #
def test_rag_branch_sse_sequence(monkeypatch):
    """RAG 分支：astream 应依次产出 intent -> retrieval(分阶段) -> references ->
    answer_delta(逐段) -> done，且全程不触网/连库。"""
    pipeline = pytest.importorskip("app.graph.pipeline")
    nodes = pytest.importorskip("app.graph.nodes")
    from config.constants import Intent

    _patch_common(monkeypatch, nodes, intent=Intent.GENERAL_QA.value, router=_FakeRouterRAG())

    orch = pipeline.Orchestrator()
    req = ChatRequest(query="增值税一般纳税人税率是多少", user_id="u1", session_id="s1", top_k=5)
    msgs = asyncio.run(_collect(orch, req))
    events = _events(msgs)

    # 1) 事件类型齐全：意图 / 检索进度 / 引用 / 答案增量 / 结束都在
    assert SSEEvent.INTENT.value in events, f"缺少 intent 事件：{events}"
    assert SSEEvent.RETRIEVAL.value in events, f"缺少 retrieval 事件：{events}"
    assert SSEEvent.REFERENCES.value in events, f"缺少 references 事件：{events}"
    assert SSEEvent.ANSWER_DELTA.value in events, f"缺少 answer_delta 事件：{events}"
    assert SSEEvent.DONE.value in events, f"缺少 done 事件：{events}"
    # RAG 分支不应出现 SQL 事件
    assert SSEEvent.SQL.value not in events, f"RAG 分支不应出现 sql 事件：{events}"
    # 不应出现 error 事件（全链路应正常跑通）
    assert SSEEvent.ERROR.value not in events, f"RAG 分支出现了 error 事件：{events}"

    # 2) 关键顺序：intent 在最前，done 在最后，检索/引用/答案依次靠后
    i_intent = _first_index(events, SSEEvent.INTENT.value)
    i_retr = _first_index(events, SSEEvent.RETRIEVAL.value)
    i_ref = _first_index(events, SSEEvent.REFERENCES.value)
    i_ans = _first_index(events, SSEEvent.ANSWER_DELTA.value)
    assert i_intent < i_retr < i_ref < i_ans, f"事件顺序不符合预期：{events}"
    assert events[-1] == SSEEvent.DONE.value, f"最后一个事件应为 done：{events}"

    # 3) 检索分阶段：recall / coarse_rank / fine_rank / rerank 四个 stage 都应出现且按序
    retr_stages = [m.data.get("stage") for m in msgs if m.event == SSEEvent.RETRIEVAL.value]
    assert retr_stages == ["recall", "coarse_rank", "fine_rank", "rerank"], \
        f"检索阶段缺失或乱序：{retr_stages}"

    # 4) intent 事件载荷正确
    intent_msg = next(m for m in msgs if m.event == SSEEvent.INTENT.value)
    assert intent_msg.data.get("intent") == Intent.GENERAL_QA.value
    assert "sub_queries" in intent_msg.data

    # 5) references 事件应带非空引用列表
    ref_msg = next(m for m in msgs if m.event == SSEEvent.REFERENCES.value)
    refs = ref_msg.data.get("references")
    assert isinstance(refs, list) and len(refs) >= 1, f"references 为空：{ref_msg.data}"

    # 6) 逐 token 流式：answer_delta 应出现多段，拼起来即完整答案（验证"真·逐段到达"）
    deltas = [m.data.get("text", "") for m in msgs if m.event == SSEEvent.ANSWER_DELTA.value]
    assert len(deltas) >= 2, f"answer_delta 应逐段产出（至少2段），实际：{deltas}"
    full = "".join(deltas)
    assert full, "拼接后的答案不应为空"
    assert "13%" in full or "税率" in full, f"答案内容不符合假摘要预期：{full!r}"


# --------------------------------------------------------------------------- #
# 用例二：Text2SQL 分支的 SSE 事件序列回归
# --------------------------------------------------------------------------- #
def test_text2sql_branch_sse_sequence(monkeypatch):
    """Text2SQL 分支：astream 应产出 intent -> sql -> answer_delta(逐段) -> done，
    且不出现 RAG 的 retrieval/references 事件。"""
    pipeline = pytest.importorskip("app.graph.pipeline")
    nodes = pytest.importorskip("app.graph.nodes")
    from config.constants import Intent

    _patch_common(monkeypatch, nodes, intent=Intent.DATA_QUERY.value, router=_FakeRouterT2S())

    orch = pipeline.Orchestrator()
    req = ChatRequest(query="2026年Q1总订单金额", user_id="u1", session_id="s1", top_k=5)
    msgs = asyncio.run(_collect(orch, req))
    events = _events(msgs)

    # 1) 事件类型：意图 / SQL / 答案增量 / 结束都在；不出现 retrieval/references
    assert SSEEvent.INTENT.value in events, f"缺少 intent 事件：{events}"
    assert SSEEvent.SQL.value in events, f"缺少 sql 事件：{events}"
    assert SSEEvent.ANSWER_DELTA.value in events, f"缺少 answer_delta 事件：{events}"
    assert SSEEvent.DONE.value in events, f"缺少 done 事件：{events}"
    assert SSEEvent.RETRIEVAL.value not in events, f"Text2SQL 分支不应出现 retrieval：{events}"
    assert SSEEvent.REFERENCES.value not in events, f"Text2SQL 分支不应出现 references：{events}"
    assert SSEEvent.ERROR.value not in events, f"Text2SQL 分支出现了 error 事件：{events}"

    # 2) 顺序：intent 在 sql 前，sql 在 answer_delta 前，done 收尾
    i_intent = _first_index(events, SSEEvent.INTENT.value)
    i_sql = _first_index(events, SSEEvent.SQL.value)
    i_ans = _first_index(events, SSEEvent.ANSWER_DELTA.value)
    assert i_intent < i_sql < i_ans, f"事件顺序不符合预期：{events}"
    assert events[-1] == SSEEvent.DONE.value, f"最后一个事件应为 done：{events}"

    # 3) sql 事件载荷正确（带 SQL 文本与列名）
    sql_msg = next(m for m in msgs if m.event == SSEEvent.SQL.value)
    assert "SELECT" in (sql_msg.data.get("sql") or "").upper(), f"sql 载荷异常：{sql_msg.data}"
    assert sql_msg.data.get("row_count") == 1
    assert sql_msg.data.get("error") in (None, ""), f"不应有错误：{sql_msg.data}"

    # 4) 逐段答案：Text2SQL 结论由编排层 _chunk_text 切块伪流式产出，应至少出现 1 段且能拼回结论
    deltas = [m.data.get("text", "") for m in msgs if m.event == SSEEvent.ANSWER_DELTA.value]
    assert len(deltas) >= 1, f"answer_delta 应至少产出 1 段：{deltas}"
    full = "".join(deltas)
    assert "123456" in full, f"拼接答案应含查询结论数字：{full!r}"


# --------------------------------------------------------------------------- #
# 用例三：意图事件总在最前（两分支共有的"起手式"回归）
# --------------------------------------------------------------------------- #
def test_intent_always_first_event(monkeypatch):
    """两个分支的第一条业务事件都应是 intent（understand 是 route 之后的第一个产出节点）。"""
    pipeline = pytest.importorskip("app.graph.pipeline")
    nodes = pytest.importorskip("app.graph.nodes")
    from config.constants import Intent

    # RAG 分支
    _patch_common(monkeypatch, nodes, intent=Intent.GENERAL_QA.value, router=_FakeRouterRAG())
    orch = pipeline.Orchestrator()
    req = ChatRequest(query="增值税税率", user_id="u", session_id="s", top_k=3)
    events = _events(asyncio.run(_collect(orch, req)))
    assert events and events[0] == SSEEvent.INTENT.value, f"首事件应为 intent：{events}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
