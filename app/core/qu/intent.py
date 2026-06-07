"""意图分类器（IntentClassifier）。

本模块在整体链路里的位置：QU 的第一步。意图决定了 SearchRouter 之后走 RAG 还是 Text2SQL、
查哪几个知识库、用什么粗排/精排策略，是整条链路的"总开关"。

设计思路（对标爱搜税 kernel/qu.py 的 query_classification）：
    1. 【规则/正则优先】—— 税务领域的意图大多有很强的"表面特征"（书名号、文号、固定术语），
       用规则判定又快又稳、零成本、可解释，命中率高。规则按"优先级"逐条短路判断。
    2. 【LLM 兜底】—— 规则都没命中（多为自然语言长问句）时，才调一次 LLM（json_mode）做语义分类，
       既省 token 又保证覆盖率。
    3. 任何异常都安全回退到 DEFAULT_INTENT（通用问题类），保证链路永不中断。

为什么规则优先而不是直接上 LLM：
    线上检索 QPS 高、延迟敏感；规则能拦住绝大多数高频意图，把昂贵的 LLM 调用留给真正模糊的 query。
"""
from __future__ import annotations

import re

from config.constants import Intent, DEFAULT_INTENT
from config.logging_config import get_logger
from app.utils.prompt_loader import load_prompt

logger = get_logger(__name__)

# 合法意图值集合：LLM 返回的字符串必须落在这里面，否则视为非法、回退默认。
_VALID_INTENTS = {i.value for i in Intent}


class IntentClassifier:
    """把用户 query 分类到 Intent 七种意图之一。

    用法::

        clf = IntentClassifier()
        intent = clf.classify("企业所得税法第几条规定研发费用加计扣除？")  # -> "精确法规类"

    内部状态全部是预编译的正则/关键词表（在 __init__ 一次性构建），classify 是无副作用的纯计算，
    线程安全、可被并发调用。
    """

    def __init__(self) -> None:
        """初始化各意图的判定规则（关键词 + 预编译正则）。

        说明：规则集中放在这里集中维护，避免散落到 classify 里变成"魔法字符串"。
        预编译正则（re.compile）是为了在高频调用时省去每次编译的开销。
        """
        # 文号特征：财税〔2024〕18号 / 财政部税务总局公告2023年第19号 / 国家税务总局公告2023年第3号 / 国务院令第773号
        # 命中文号 -> 几乎可断定是"精确法规类"。
        self._re_doc_no = re.compile(
            r"(?:〔|\[|【)\s*(?:19|20)\d{2}\s*(?:〕|\]|】)\s*\d+\s*号"   # 财税〔2024〕18号
            r"|公告\s*(?:19|20)\d{2}\s*年第\s*\d+\s*号"                  # 公告2023年第19号
            r"|令第?\s*\d+\s*号"                                         # 国务院令第773号
        )
        # 书名号：《xxx办法》《xxx法》—— 用户点名了某部具体法规，归"精确法规类"。
        self._re_book_title = re.compile(r"《.+?》")
        # 以法规体裁词结尾（无书名号场景）：xxx管理办法 / xxx暂行条例 / xxx实施细则 / xxx税法
        self._re_regulation_suffix = re.compile(
            r"(?:管理办法|暂行条例|实施细则|管理法|税法|条例|公告|通知|批复|规定)\s*$"
        )

        # 政策汇集类：问"某主题下有哪些政策/优惠"，重在"合集"而非某一部具体法规。
        self._policy_collection_keywords = ["税收优惠", "优惠政策", "政策汇编", "政策合集", "相关政策"]
        # "xxx政策有哪些 / xxx政策哪些"这类典型汇集句式。
        self._re_policy_collection = re.compile(r".*政策.*?(?:有哪些|哪些|汇总|汇集|盘点)")

        # 稽查案例类：固定术语，直接关键词命中（对标爱搜税 inspect_case）。
        self._inspect_keywords = ["稽查案例", "处罚案例", "违法案例", "违规案例", "稽查热点", "稽查处罚"]

        # 查社保类：必须同时出现"社保/社会保险"和"基数/比例/上下限"，避免误伤普通社保问答。
        self._re_social_security = re.compile(r"(?:社保|社会保险|公积金).*?(?:基数|比例|上限|下限|最低|最高|缴费)")

        # 查商品编码类：商品/税收分类编码这类结构化查询术语。
        self._product_code_keywords = ["商品编码", "税收分类编码", "税收编码", "开票编码", "税务编码", "行业编码"]

        # 经营数据查询类（-> Text2SQL）：自然语言查经营/财税数仓数据，特征是"聚合/统计/排名/对比"类动词。
        # 这是与"通用问题类"区分的关键：通用问答是"政策怎么规定"，数据查询是"我司的销售额是多少"。
        # 只保留"强分析"信号词（这些在税务政策问句里几乎不会出现），刻意【不放】"销售额/营收/利润"
        # 等普通经营名词——否则像"月销售额多少以内免征增值税"这种【政策问句】会被误判成数据查询、
        # 错走 Text2SQL。真正的聚合分析问句(各省/各品类...多少)由下面的 _re_data_query 正则兜底命中。
        self._data_query_keywords = [
            "同比", "环比", "增长率", "占比", "排名", "排行", "前十", "top", "明细", "报表",
        ]
        # 聚合分析句式：各/每个 ... 的 销售额/数量；本月/上季度 ... 是多少
        self._re_data_query = re.compile(
            r"(?:各|每个|每月|每年|每季度|本月|上月|本季度|上季度|本年|去年|今年).*?(?:销售|营收|利润|金额|数量|统计|多少)"
        )

    def classify(self, query: str) -> str:
        """对单条 query 做意图分类。

        判定顺序（短路）：精确法规类 -> 稽查案例类 -> 查社保类 -> 查商品编码类
        -> 经营数据查询类 -> 政策汇集类 -> （规则全不中）LLM 兜底 -> 默认通用问题类。

        顺序设计的理由：把"特征最强、最不会误伤"的规则放前面（如文号/书名号、固定术语），
        把"语义型、易混"的（政策汇集 vs 通用问答）放后面，最后才交给 LLM。

        :param query: 用户原始问题（已可经过纠错），允许为空/None。
        :return: Intent 枚举的"值"（中文字符串），保证一定是 _VALID_INTENTS 之一。
        :raise: 不向外抛异常；任何内部异常都被捕获并回退到 DEFAULT_INTENT。
        """
        logger.info("意图分类开始，query=%s", query)
        # 入参兜底：空 query 直接返回默认意图，不必走任何规则。
        if not query or not query.strip():
            logger.info("意图分类：query 为空，回退默认意图=%s", DEFAULT_INTENT.value)
            return DEFAULT_INTENT.value

        q = query.strip()

        # ---------- 第一层：规则/正则（优先级从高到低，命中即返回） ----------
        rule_intent = self._classify_by_rules(q)
        if rule_intent is not None:
            logger.info("意图分类命中规则：intent=%s", rule_intent)
            return rule_intent

        # ---------- 第二层：LLM 兜底（仅当规则全不中） ----------
        logger.info("意图分类：规则未命中，调用 LLM 兜底")
        llm_intent = self._classify_by_llm(q)
        logger.info("意图分类结束：intent=%s", llm_intent)
        return llm_intent

    # ------------------------------------------------------------------ #
    # 内部方法
    # ------------------------------------------------------------------ #
    def _classify_by_rules(self, q: str) -> str | None:
        """纯规则判定。命中返回对应 Intent 值，全不命中返回 None（交给 LLM）。

        :param q: 已 strip 的非空 query。
        :return: Intent 值或 None。
        """
        # 1) 精确法规类：文号 / 书名号 / 法规体裁词结尾，命中其一即可。
        #    这是特征最强、误判率最低的意图，放在最前面短路。
        if self._re_doc_no.search(q) or self._re_book_title.search(q) or self._re_regulation_suffix.search(q):
            return Intent.PRECISE_REGULATION.value

        # 2) 稽查案例类：固定术语关键词，直接子串匹配。
        if any(kw in q for kw in self._inspect_keywords):
            return Intent.INSPECT_CASE.value

        # 3) 查社保类：必须"社保"+"基数/比例/上下限"组合命中，避免误伤"社保怎么交"这类通用问答。
        if self._re_social_security.search(q):
            return Intent.SOCIAL_SECURITY.value

        # 4) 查商品编码类：编码类术语关键词。
        if any(kw in q for kw in self._product_code_keywords):
            return Intent.PRODUCT_CODE.value

        # 5) 经营数据查询类（-> Text2SQL）：聚合/统计/排名/同比环比等数据分析意图。
        #    关键词命中 或 "各/本月...多少"句式命中，二者其一即可。
        if any(kw in q for kw in self._data_query_keywords) or self._re_data_query.search(q):
            return Intent.DATA_QUERY.value

        # 6) 政策汇集类：句式"xxx政策有哪些/汇总" 或 含"税收优惠/优惠政策"等汇集词。
        #    放在数据查询之后，避免"各地税收优惠政策有哪些"先被误判为数据类（其实它含统计词概率低）。
        if self._re_policy_collection.search(q) or any(kw in q for kw in self._policy_collection_keywords):
            return Intent.POLICY_COLLECTION.value

        # 全不命中：返回 None，交由 LLM 语义兜底。
        return None

    def _classify_by_llm(self, q: str) -> str:
        """LLM 语义兜底分类（json_mode 输出，便于稳定解析）。

        失败策略：任何异常（无 LLM 服务、网络错误、返回非法 JSON、意图值越界）都回退默认意图，
        保证"没有基础设施时也能跑通"且"线上永不因 QU 报错而中断"。

        :param q: 已 strip 的非空 query。
        :return: 合法的 Intent 值；异常或非法时回退 DEFAULT_INTENT。
        """
        try:
            # 延迟导入：避免本模块在没有 langchain/LLM 环境时 import 即失败（惰性原则）。
            import json
            from app.clients.llm_client import get_llm

            # 载入意图分类 prompt 模板（带缓存），并把 query 填进占位符。
            # 约定模板里用 {query} 占位、{intents} 列出候选意图，返回 {"intent": "xxx"} 的 JSON。
            template = load_prompt("intent_classify")
            prompt = template.format(query=q, intents="、".join(_VALID_INTENTS))

            llm = get_llm(json_mode=True)  # json_mode 强制返回 json_object，规避解析脏数据
            # langchain 的 ChatOpenAI 接受 [{"role":..,"content":..}] 形式的消息列表。
            resp = llm.invoke([{"role": "user", "content": prompt}])
            raw = getattr(resp, "content", "") or ""

            data = json.loads(raw)  # json_mode 下应为合法 JSON
            intent = str(data.get("intent", "")).strip()

            # 越界校验：LLM 可能臆造意图名，必须落回合法集合。
            if intent in _VALID_INTENTS:
                return intent
            logger.warning("LLM 返回非法意图值=%s，回退默认意图", intent)
            return DEFAULT_INTENT.value
        except Exception as e:  # 兜底所有异常，绝不让 QU 因分类失败而崩
            logger.error("LLM 意图分类异常，回退默认意图：%s", e, exc_info=True)
            return DEFAULT_INTENT.value


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行，不依赖任何外部服务，只验证规则层）。
    clf = IntentClassifier()
    cases = [
        "财税〔2024〕18号说了什么",                 # 精确法规类（文号）
        "《企业所得税法》第几条讲研发费用加计扣除",   # 精确法规类（书名号）
        "增值税小规模纳税人减免增值税政策的公告",     # 精确法规类（体裁词结尾）
        "有没有虚开发票的稽查案例",                 # 稽查案例类
        "杭州市2024年社保缴费基数上限是多少",       # 查社保类
        "餐饮服务的税收分类编码是多少",             # 查商品编码类
        "各分公司本月销售额排名前十是哪些",         # 经营数据查询类
        "小微企业税收优惠政策有哪些",               # 政策汇集类
        "公司注销需要哪些流程",                     # 规则不中 -> LLM 兜底（无服务时回退默认）
    ]
    for c in cases:
        print(f"{clf.classify(c):8s} <- {c}")
