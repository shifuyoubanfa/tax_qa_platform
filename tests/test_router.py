"""SearchRouter 路由表单元测试。
覆盖对象：app/core/router/search_router.py 的 SearchRouter.route(qu) -> RetrievalPlan。

为什么要测路由：路由器是检索链路的"大脑"，决定每个意图查哪些库、dense/sparse 权重、走 RAG 还是 Text2SQL。
路由错了，后面召回再准也白搭。这块逻辑纯本地、无外部依赖，最适合做强断言回归。

测试要点（与 config/constants.py 契约对齐）：
1. 经营数据查询类 -> route_type=text2sql，且带 agent_name；
2. 其余 6 类 -> route_type=rag，steps 非空，且每个 step 的 kbases 都是合法 KBase 值；
3. 精确法规/政策汇集类 -> sparse 权重应高于 dense（要精确命中文号/政策名）；
4. 通用问题类 -> dense 权重应高于 sparse（口语化、重语义）；
5. 未知意图 -> 安全回退到通用问题类策略，不抛异常。
"""
from __future__ import annotations

import pytest

from config.constants import Intent, KBase, RouteType
from app.schemas.document import QUResult, RecallStep, RetrievalPlan


@pytest.fixture
def router():
    """构造一个 SearchRouter 实例（模块未就绪时优雅跳过）。"""
    mod = pytest.importorskip("app.core.router.search_router")
    return mod.SearchRouter()


def _make_qu(intent_value: str) -> QUResult:
    """造一个只关心 intent 的 QUResult（其余字段用默认值）。"""
    return QUResult(raw_query="测试问题", intent=intent_value)


# ============================================================
# 1) 顶层路由：结构化 vs 非结构化
# ============================================================
def test_data_query_routes_to_text2sql(router):
    """经营数据查询类应路由到 Text2SQL（route_type=text2sql 且有 agent_name）。"""
    plan = router.route(_make_qu(Intent.DATA_QUERY.value))
    assert isinstance(plan, RetrievalPlan)
    assert plan.route_type == RouteType.TEXT2SQL.value
    assert plan.agent_name, "Text2SQL 路由必须给出 agent_name"
    # 走 Text2SQL 时不应再产出 RAG 召回步骤
    assert not plan.steps


@pytest.mark.parametrize("intent", [
    Intent.PRECISE_REGULATION.value,
    Intent.POLICY_COLLECTION.value,
    Intent.GENERAL_QA.value,
    Intent.INSPECT_CASE.value,
    Intent.SOCIAL_SECURITY.value,
    Intent.PRODUCT_CODE.value,
])
def test_non_data_intents_route_to_rag(router, intent):
    """除经营数据查询类外的意图都应走 RAG，且产出至少一个召回步骤。"""
    plan = router.route(_make_qu(intent))
    assert plan.route_type == RouteType.RAG.value
    assert plan.steps, f"意图[{intent}] 未产出任何召回步骤"
    for step in plan.steps:
        assert isinstance(step, RecallStep)


# ============================================================
# 2) 召回步骤内容合法性
# ============================================================
_VALID_KBASES = {k.value for k in KBase}


@pytest.mark.parametrize("intent", [
    Intent.PRECISE_REGULATION.value,
    Intent.POLICY_COLLECTION.value,
    Intent.GENERAL_QA.value,
    Intent.INSPECT_CASE.value,
    Intent.SOCIAL_SECURITY.value,
    Intent.PRODUCT_CODE.value,
])
def test_steps_kbases_are_valid(router, intent):
    """每个召回步骤涉及的知识库都必须是合法 KBase 值（防止脏字符串）。"""
    plan = router.route(_make_qu(intent))
    for step in plan.steps:
        assert step.kbases, "召回步骤的 kbases 不能为空"
        for kb in step.kbases:
            assert kb in _VALID_KBASES, f"非法知识库标识：{kb}"


@pytest.mark.parametrize("intent", [
    Intent.PRECISE_REGULATION.value,
    Intent.POLICY_COLLECTION.value,
    Intent.GENERAL_QA.value,
])
def test_steps_weights_present(router, intent):
    """召回步骤应带 dense/sparse 权重（混合召回融合时要用）。"""
    plan = router.route(_make_qu(intent))
    for step in plan.steps:
        assert "dense" in step.weights and "sparse" in step.weights
        # 权重应为非负数
        assert step.weights["dense"] >= 0 and step.weights["sparse"] >= 0


# ============================================================
# 3) 权重方向：关键词 vs 语义（这套系统的核心设计）
# ============================================================
def test_precise_regulation_prefers_sparse(router):
    """精确法规类要精确命中文号/法规名 -> sparse 权重应高于 dense。"""
    plan = router.route(_make_qu(Intent.PRECISE_REGULATION.value))
    first = plan.steps[0]
    assert first.weights["sparse"] > first.weights["dense"], "精确法规类应偏关键词(sparse)"


def test_policy_collection_prefers_sparse(router):
    """政策汇集类围绕主题找政策合集，仍偏关键词 -> sparse 高于 dense。"""
    plan = router.route(_make_qu(Intent.POLICY_COLLECTION.value))
    first = plan.steps[0]
    assert first.weights["sparse"] > first.weights["dense"], "政策汇集类应偏关键词(sparse)"


def test_general_qa_prefers_dense(router):
    """通用问题类口语化、重语义 -> dense 权重应高于 sparse。"""
    plan = router.route(_make_qu(Intent.GENERAL_QA.value))
    first = plan.steps[0]
    assert first.weights["dense"] > first.weights["sparse"], "通用问题类应偏语义(dense)"


# ============================================================
# 4) 兜底：未知意图回退
# ============================================================
def test_unknown_intent_falls_back_to_rag(router):
    """传入路由表中不存在的意图，应安全回退到 RAG 策略而非抛异常。"""
    plan = router.route(_make_qu("不存在的意图XYZ"))
    assert plan.route_type == RouteType.RAG.value
    assert plan.steps, "未知意图回退后仍应有召回步骤"


def test_empty_intent_does_not_crash(router):
    """意图为空字符串时也应安全产出合法计划（默认意图兜底）。"""
    plan = router.route(_make_qu(""))
    assert plan.route_type in {RouteType.RAG.value, RouteType.TEXT2SQL.value}


def test_route_covers_all_intents(router):
    """全量意图都能被路由（不漏意图、不抛异常），保证路由表完整。"""
    for it in Intent:
        plan = router.route(_make_qu(it.value))
        assert isinstance(plan, RetrievalPlan)
        # 要么 RAG（有 steps），要么 Text2SQL（有 agent_name）
        if plan.route_type == RouteType.RAG.value:
            assert plan.steps
        else:
            assert plan.agent_name


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
