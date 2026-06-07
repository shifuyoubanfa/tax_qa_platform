"""离线评测：用 data/eval/queries.jsonl 评 IntentClassifier【规则部分】的意图命中率。

本测试在整个体系里的位置：它既是"规则质量回归"（防止改规则时悄悄回退命中率），
也是后续"真跑 + 指标"的夹具来源（同一份 queries.jsonl 可喂给端到端评测、召回/答案指标）。

为什么只评"规则部分"、且不调 LLM：
    IntentClassifier.classify 在规则全不命中时会兜底调一次 LLM 做语义分类（见 intent.py）。
    线上 LLM 有成本、有网络、结果不稳定，不适合放进单测。这里有两层隔离保证"零触网、可复现"：
      1) 断言只针对【规则可判定】的样本——直接调用纯本地的 _classify_by_rules(q)，它不碰 LLM；
      2) 为了顺带打印"整库经 classify 的命中率"，把 get_llm 打桩成抛错，
         使 classify 在规则不中时安全回退 DEFAULT_INTENT（通用问题类），全程不触网。

样本契约（queries.jsonl 每行一个 JSON 对象）：
    query            : 用户原始问题（贴近真实税务）
    expected_intent  : 期望意图（取 config.constants.Intent 的中文枚举值）
    expected_route   : 期望顶层路由 text2sql / rag（与 SearchRouter 实际行为对齐：仅经营数据查询走 text2sql）
    expected_kbase   : 期望命中的数据源/结构化表（KBase 值；通用/数据查询可为 null）
    rule_decidable   : 该样本是否"靠本地规则即可判定"。仅这些样本纳入硬断言；
                       口语化/时效/多轮追问类(规则不中、需 LLM 或改写)只统计、不断言。
    note             : 该样本的标注理由（人类可读，便于维护）

依赖未就绪时（QU 模块或数据文件缺失）用 importorskip / skip 优雅跳过，绝不拖红整批测试。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from config.constants import Intent, RouteType

# data/eval/queries.jsonl 相对项目根定位：本文件 = tests/eval/xxx.py -> parents[2] = 项目根
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_QUERIES_PATH = _PROJECT_ROOT / "data" / "eval" / "queries.jsonl"

# 合法集合：用于校验标注本身不写错（标注的 intent/route 必须是枚举里的值）。
_VALID_INTENTS = {i.value for i in Intent}
_VALID_ROUTES = {r.value for r in RouteType}


def _load_samples() -> list[dict]:
    """读取并解析 queries.jsonl，返回样本字典列表。

    :return: 样本列表；文件不存在时返回空列表（调用方据此 skip）。
    """
    if not _QUERIES_PATH.exists():
        return []
    samples: list[dict] = []
    # 逐行解析 JSONL：跳过空行，保证对手写数据的容错。
    with _QUERIES_PATH.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:  # 标注文件有语法错误应立即暴露，便于修数据
                raise AssertionError(f"queries.jsonl 第 {line_no} 行不是合法 JSON: {e}") from e
    return samples


@pytest.fixture(scope="module")
def samples() -> list[dict]:
    """加载评测样本；文件缺失则整模块 skip（评测集尚未生成时不拖红）。"""
    data = _load_samples()
    if not data:
        pytest.skip(f"评测集不存在或为空：{_QUERIES_PATH}")
    return data


@pytest.fixture
def rule_only_classifier(monkeypatch):
    """构造一个"绝不触网"的 IntentClassifier：把 get_llm 打桩为抛错。

    这样 classify 在规则不命中时会落入 LLM 兜底分支，因 get_llm 抛错而安全回退 DEFAULT_INTENT，
    既保证零网络、又能复现地统计"整库命中率"。QU 模块尚未生成时优雅跳过。

    :return: IntentClassifier 实例（LLM 已被打桩为不可用）。
    """
    intent_mod = pytest.importorskip("app.core.qu.intent")

    def _boom(*args, **kwargs):
        raise RuntimeError("评测桩：禁止真实调用 LLM（只评规则部分）")

    # 双重打桩：clients 源头 + intent 模块内可能存在的本地绑定，确保运行期拿到的都是会抛错的 get_llm。
    try:
        import app.clients.llm_client as llm_mod
        monkeypatch.setattr(llm_mod, "get_llm", _boom, raising=False)
    except Exception:  # noqa: BLE001 - clients 层缺失不应阻塞
        pass
    monkeypatch.setattr(intent_mod, "get_llm", _boom, raising=False)

    return intent_mod.IntentClassifier()


def test_eval_set_annotations_wellformed(samples):
    """先校验评测集"标注本身"是规范的：字段齐全、枚举合法、覆盖全部意图。

    这是评测集的"元测试"——脏标注会让后续所有指标失真，必须先守住。
    """
    seen_intents: set[str] = set()
    for i, s in enumerate(samples):
        assert "query" in s and s["query"].strip(), f"样本{i} 缺少非空 query"
        ei = s.get("expected_intent")
        assert ei in _VALID_INTENTS, f"样本{i} expected_intent={ei!r} 不是合法 Intent 值"
        er = s.get("expected_route")
        assert er in _VALID_ROUTES, f"样本{i} expected_route={er!r} 不是合法 RouteType 值"
        # 路由自洽：仅"经营数据查询类"走 text2sql，其余意图均为 rag（与 SearchRouter 行为一致）。
        if ei == Intent.DATA_QUERY.value:
            assert er == RouteType.TEXT2SQL.value, f"样本{i} 经营数据查询类应走 text2sql"
        else:
            assert er == RouteType.RAG.value, f"样本{i} 非数据查询意图应走 rag"
        seen_intents.add(ei)

    # 样本量 >= 18，且 7 种意图全覆盖（评测集代表性的硬约束）。
    assert len(samples) >= 18, f"评测样本至少 18 条，当前 {len(samples)}"
    missing = _VALID_INTENTS - seen_intents
    assert not missing, f"评测集未覆盖以下意图：{missing}"


def test_intent_rule_accuracy(samples, rule_only_classifier):
    """评 IntentClassifier【规则部分】命中率，并对"规则可判定"样本做硬断言。

    断言口径：只断言 rule_decidable=True 的样本——直接调用纯本地的 _classify_by_rules(q)，
    它必须返回非 None 且等于 expected_intent（规则要真正覆盖到这些意图）。
    口语化/时效/多轮追问类(rule_decidable=False)只统计、不断言（它们本就交给 LLM/改写）。
    """
    clf = rule_only_classifier

    rule_total = 0          # 规则可判定样本数
    rule_hit = 0            # 其中规则判对的数
    rule_failures: list[str] = []

    overall_total = len(samples)
    overall_hit = 0         # 整库经 classify(含兜底) 的命中数（仅打印参考）
    by_intent = Counter()   # 各意图样本数（打印分布）

    for s in samples:
        q = s["query"]
        expected = s["expected_intent"]
        by_intent[expected] += 1

        # —— 整库命中率（经 classify，LLM 已打桩抛错 -> 规则不中即回退默认）——
        if clf.classify(q) == expected:
            overall_hit += 1

        # —— 规则部分命中率（只看纯规则层 _classify_by_rules）——
        if s.get("rule_decidable") is True:
            rule_total += 1
            rule_intent = clf._classify_by_rules(q)  # noqa: SLF001 - 评测刻意只测规则层
            if rule_intent == expected:
                rule_hit += 1
            else:
                rule_failures.append(
                    f"  [规则未命中或判错] query={q!r} 期望={expected} 规则判定={rule_intent!r}"
                )

    rule_acc = rule_hit / rule_total if rule_total else 0.0
    overall_acc = overall_hit / overall_total if overall_total else 0.0

    # 打印评测报告（-s 时可见；CI 里也会随失败信息一并输出，便于排查）。
    print("\n========== 意图分类离线评测 ==========")
    print(f"样本总数            : {overall_total}")
    print(f"意图分布            : {dict(by_intent)}")
    print(f"规则可判定样本      : {rule_total}")
    print(f"规则部分命中率      : {rule_hit}/{rule_total} = {rule_acc:.1%}")
    print(f"整库命中率(含兜底)  : {overall_hit}/{overall_total} = {overall_acc:.1%} "
          f"(规则不中的口语/时效/多轮样本回退默认意图，符合预期)")
    if rule_failures:
        print("规则判错明细：")
        print("\n".join(rule_failures))
    print("======================================")

    # 硬断言：规则可判定样本必须 100% 命中（规则是确定性的，达不到说明规则被改坏或标注有误）。
    assert rule_total > 0, "评测集应至少包含若干 rule_decidable=True 的样本"
    assert rule_hit == rule_total, (
        f"规则部分命中率应为 100%，实际 {rule_acc:.1%}；明细：\n" + "\n".join(rule_failures)
    )


if __name__ == "__main__":
    # 单文件自测：直接跑本评测（需先安装 pytest 与 QU 模块）。-s 可看到评测报告。
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
