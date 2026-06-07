"""QU（Query Understanding）规则部分单元测试。
覆盖对象：IntentClassifier（意图分类，规则正则部分）、EntityExtractor（实体抽取，纯本地）。

测试策略（为什么这么测）：
- 意图分类与实体抽取的"规则分支"完全是本地正则/关键词逻辑，无需任何外部服务，最适合做稳定单测。
- IntentClassifier 在规则命不中时可能回退调用 LLM；为了只验证"规则部分"且不触网，
  本测试把 LLM 工厂 get_llm 打桩成"一调用就抛错"，强制走规则分支（与 agent C 留下的逻辑自测同思路）。
- 所有期望值与 config/constants.py 的 Intent 枚举值（中文）对齐，避免魔法字符串。

依赖未就绪时（QU 模块尚未生成）用 importorskip 优雅跳过，不拖红整批测试。
"""
from __future__ import annotations

import pytest

from config.constants import Intent


# ---- 公共夹具：强制 IntentClassifier 走"规则分支"（LLM 打桩为抛错）----
@pytest.fixture
def rule_only_intent(monkeypatch):
    """把 LLM 工厂打桩成抛错，使意图分类只走本地规则、绝不触网。

    覆盖两处绑定：
    1) app.clients.llm_client.get_llm（模块属性，惰性调用方会读到）；
    2) app.core.qu.intent.get_llm（若该模块在导入时把 get_llm 绑成了本地名）。
    这样无论 intent.py 用哪种导入方式，运行期拿到的都是"会抛错的 get_llm"。
    """

    def _boom(*args, **kwargs):
        raise RuntimeError("测试桩：禁止真实调用 LLM")

    # 1) 打桩 clients 层
    try:
        import app.clients.llm_client as llm_mod
        monkeypatch.setattr(llm_mod, "get_llm", _boom, raising=False)
    except Exception:  # noqa: BLE001 - clients 层可能尚未生成，忽略
        pass
    # 2) 打桩 intent 模块内的本地绑定（若已 from ... import get_llm）
    try:
        import app.core.qu.intent as intent_mod
        monkeypatch.setattr(intent_mod, "get_llm", _boom, raising=False)
    except Exception:  # noqa: BLE001
        pass


# ============================================================
# 意图分类（规则部分）
# ============================================================
# 用例集：问题 -> 期望意图（取 Intent 枚举值）。
# 这些用例均由"规则正则/关键词"可判定，不依赖 LLM（与 agent C 的逻辑自测一致）。
_INTENT_CASES = [
    ("财税〔2024〕18号说了什么", Intent.PRECISE_REGULATION.value),
    ("《企业所得税法》第几条讲研发费用加计扣除", Intent.PRECISE_REGULATION.value),
    ("国家税务总局公告2023年第3号", Intent.PRECISE_REGULATION.value),
    ("有没有虚开发票的稽查案例", Intent.INSPECT_CASE.value),
    ("杭州市2024年社保缴费基数上限是多少", Intent.SOCIAL_SECURITY.value),
    ("餐饮服务的税收分类编码是多少", Intent.PRODUCT_CODE.value),
    ("各分公司本月销售额排名前十是哪些", Intent.DATA_QUERY.value),
    ("今年的营收同比增长多少", Intent.DATA_QUERY.value),
    ("小微企业税收优惠政策有哪些", Intent.POLICY_COLLECTION.value),
    ("公司注销需要哪些流程", Intent.GENERAL_QA.value),
]


@pytest.mark.parametrize("query, expected", _INTENT_CASES)
def test_intent_classify_rules(rule_only_intent, query, expected):
    """规则意图分类应与期望意图一致（LLM 已打桩为抛错，强制走规则）。"""
    intent_mod = pytest.importorskip("app.core.qu.intent")
    clf = intent_mod.IntentClassifier()
    got = clf.classify(query)
    assert got == expected, f"问题[{query}] 期望意图={expected} 实际={got}"


def test_intent_classify_returns_valid_enum_value(rule_only_intent):
    """无论命中哪条规则，返回值必须是 Intent 合法枚举值（防止脏字符串外泄）。"""
    intent_mod = pytest.importorskip("app.core.qu.intent")
    clf = intent_mod.IntentClassifier()
    valid = {i.value for i in Intent}
    for query, _ in _INTENT_CASES:
        assert clf.classify(query) in valid


def test_intent_empty_query_falls_back_default(rule_only_intent):
    """空/无意义 query 应安全回退到默认意图（通用问题类），不应抛异常。"""
    intent_mod = pytest.importorskip("app.core.qu.intent")
    clf = intent_mod.IntentClassifier()
    # 空字符串属于边界输入，分类器应优雅返回合法枚举值
    assert clf.classify("") in {i.value for i in Intent}


# ============================================================
# 实体抽取（纯本地，无需打桩）
# ============================================================
def test_extract_doc_no():
    """应能从问题中抽出标准文号（如 财税〔2024〕18号）。"""
    ext_mod = pytest.importorskip("app.core.qu.extractor")
    ext = ext_mod.EntityExtractor()
    e = ext.extract("财税〔2024〕18号关于研发费用加计扣除的规定")
    assert e.doc_no, "未抽到文号"
    assert "18" in e.doc_no and "2024" in e.doc_no


def test_extract_year():
    """应能抽出 4 位年份。"""
    ext_mod = pytest.importorskip("app.core.qu.extractor")
    ext = ext_mod.EntityExtractor()
    e = ext.extract("国家税务总局公告2023年第3号说了什么")
    assert e.year == "2023", f"年份抽取错误：{e.year}"


def test_extract_law_name():
    """应能从书名号中抽出法规名。"""
    ext_mod = pytest.importorskip("app.core.qu.extractor")
    ext = ext_mod.EntityExtractor()
    e = ext.extract("杭州市2024年《企业所得税法》怎么适用")
    assert e.law_name and "企业所得税法" in e.law_name


def test_extract_region():
    """应能识别省/市等地域词。"""
    ext_mod = pytest.importorskip("app.core.qu.extractor")
    ext = ext_mod.EntityExtractor()
    e = ext.extract("广东省深圳市的小微企业优惠")
    # region 为列表，至少应包含一个地域（深圳或广东）
    assert e.region, "未识别到地域"
    joined = "".join(e.region)
    assert ("深圳" in joined) or ("广东" in joined)


def test_extract_returns_entities_shape():
    """extract 必须返回 Entities 契约结构（字段齐全、类型正确），便于下游稳定消费。"""
    ext_mod = pytest.importorskip("app.core.qu.extractor")
    from app.schemas.document import Entities

    ext = ext_mod.EntityExtractor()
    e = ext.extract("一般纳税人如何认定")
    assert isinstance(e, Entities)
    # 列表型字段必须是 list（即便为空），避免下游对 None 迭代报错
    assert isinstance(e.region, list)
    assert isinstance(e.company, list)
    assert isinstance(e.product_name, list)


if __name__ == "__main__":
    # 单文件自测：直接用 pytest 跑本文件（需先安装 pytest 与 QU 模块）。
    raise SystemExit(pytest.main([__file__, "-v"]))
