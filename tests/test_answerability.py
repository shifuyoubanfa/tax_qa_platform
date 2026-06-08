"""可答性门控（answerability gate）单元测试：纯函数级，不触网/不连库/不调 LLM。

本测试在整体测试体系里的位置：锁定"重排后、生成前"这道【确定性可答性门控】的判定逻辑——
它是本次新增能力的核心，必须把每条分支（高分可答 / 低分不可答 / 空召回 / 重排降级兜底 /
总开关关闭）都钉死，避免日后误改阈值或信号来源时悄悄回归。

为什么能纯函数测：answerability_check / low_confidence_answer / answerability_decider 都只读
GraphState 的几个字段、只读 settings 阈值，不依赖任何外部组件，故可直接构造 state 调用、断言产出。
"""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("pydantic")  # 仅依赖 schemas/settings，无 langgraph 也能跑

from config.constants import Intent  # noqa: E402
from config.settings import settings  # noqa: E402
from app.schemas.document import Document, QUResult  # noqa: E402
from app.graph import nodes  # noqa: E402


def _doc(doc_id: str, rerank_score: float, score: float = 0.5) -> Document:
    """构造一条带指定重排分的候选文档（其余字段给最小可用值）。"""
    return Document(doc_id=doc_id, title=f"标题{doc_id}", content=f"正文{doc_id}",
                    kbase="policy", score=score, rerank_score=rerank_score)


def _state(reranked, intent: str = Intent.GENERAL_QA.value) -> dict:
    """构造一个仅含门控所需字段的 GraphState 片段。"""
    return {"reranked": reranked, "qu": QUResult(raw_query="q", intent=intent)}


# --------------------------------------------------------------------------- #
# answerability_check：五条分支
# --------------------------------------------------------------------------- #
def test_high_score_is_answerable():
    """重排 top1 分明显高于阈值 -> 判可答，置信度=最高分，依据=rerank_score。"""
    out = asyncio.run(nodes.answerability_check(_state([_doc("1", 0.82), _doc("2", 0.40)])))
    assert out["answerable"] is True
    assert out["answer_confidence"] == pytest.approx(0.82, abs=1e-4)
    assert out["answerability"]["reason"] == "rerank_score"
    assert out["answerability"]["doc_count"] == 2
    assert out["answerability"]["rerank_degraded"] is False


def test_low_score_is_not_answerable():
    """所有候选重排分都低于阈值(默认0.15) -> 判不可答，依据=rerank_score。"""
    out = asyncio.run(nodes.answerability_check(_state([_doc("1", 0.06), _doc("2", 0.03)])))
    assert out["answerable"] is False
    assert out["answerability"]["reason"] == "rerank_score"
    assert out["answer_confidence"] == pytest.approx(0.06, abs=1e-4)


def test_empty_recall_is_not_answerable():
    """空候选 -> 判不可答，依据=empty_recall，置信度=0。"""
    out = asyncio.run(nodes.answerability_check(_state([])))
    assert out["answerable"] is False
    assert out["answerability"]["reason"] == "empty_recall"
    assert out["answer_confidence"] == 0.0
    assert out["answerability"]["doc_count"] == 0


def test_rerank_degraded_falls_back_to_doc_count():
    """有候选但 rerank_score 全为 0（重排服务降级·未打分）-> 不信分数，改用候选数兜底判可答（宁放行不误杀）。"""
    out = asyncio.run(nodes.answerability_check(_state([_doc("1", 0.0), _doc("2", 0.0)])))
    assert out["answerable"] is True  # doc_count=2 >= answerability_min_docs(默认1)
    assert out["answerability"]["reason"] == "rerank_degraded_doc_count"
    assert out["answerability"]["rerank_degraded"] is True
    assert out["answer_confidence"] == 0.0


def test_floored_score_is_scored_not_degraded():
    """关键消歧：ReRanker 成功打分会把结果钳到 _SCORED_FLOOR(1e-6>0)，故"极强不相关的真低分"恒 > 0，
    必须判为【已打分·不可答】(reason=rerank_score)，绝不能被当成"重排降级"而错误放行(fail-open)。"""
    out = asyncio.run(nodes.answerability_check(_state([_doc("1", 1e-6), _doc("2", 1e-6)])))
    assert out["answerability"]["rerank_degraded"] is False  # 1e-6>0 -> 不是降级
    assert out["answerability"]["reason"] == "rerank_score"
    assert out["answerable"] is False  # 1e-6 < 阈值(0.15) -> 不可答（正是门控该拦住的"无相关依据"场景）


def test_disabled_switch_always_answerable(monkeypatch):
    """总开关关闭 -> 恒判可答（行为同未引入门控），即便候选为空。"""
    monkeypatch.setattr(settings, "answerability_enabled", False)
    out = asyncio.run(nodes.answerability_check(_state([])))
    assert out["answerable"] is True
    assert out["answerability"]["enabled"] is False


def test_threshold_is_configurable(monkeypatch):
    """阈值可经 settings 调参：把阈值抬到 0.9 后，0.82 的候选应被判不可答。"""
    monkeypatch.setattr(settings, "answerability_min_score", 0.9)
    out = asyncio.run(nodes.answerability_check(_state([_doc("1", 0.82)])))
    assert out["answerable"] is False


# --------------------------------------------------------------------------- #
# answerability_decider：条件边
# --------------------------------------------------------------------------- #
def test_decider_routes_answer_and_insufficient():
    """条件边：answerable=True -> 'answer'；False -> 'insufficient'；缺省视为可答。"""
    assert nodes.answerability_decider({"answerable": True}) == "answer"
    assert nodes.answerability_decider({"answerable": False}) == "insufficient"
    assert nodes.answerability_decider({}) == "answer"  # 缺省宽松：不误杀


# --------------------------------------------------------------------------- #
# low_confidence_answer：诚实兜底措辞
# --------------------------------------------------------------------------- #
def test_low_confidence_answer_wording_by_reason():
    """兜底话术按依据区分措辞：空召回 vs 弱相关；二者都不调 LLM、都给补充关键信息的指引。"""
    empty = asyncio.run(nodes.low_confidence_answer({"answerability": {"reason": "empty_recall"}}))
    weak = asyncio.run(nodes.low_confidence_answer({"answerability": {"reason": "rerank_score"}}))
    assert "未检索到" in empty["answer"]
    assert "相关性较低" in weak["answer"]
    # 两种话术都应给出"补充关键信息"的可执行指引
    assert "补充" in empty["answer"] and "补充" in weak["answer"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
