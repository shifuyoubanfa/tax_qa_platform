"""实体抽取器（EntityExtractor）。

本模块在整体链路里的位置：QU 的第二步，与意图分类并列。
产出的 Entities 用于：
    1. 精确过滤——把文号/年份/法规名/地域下推到召回层做 Milvus expr / ES filter（精准命中）；
    2. 路由辅助——例如带文号时更应走"精确法规类"策略。

设计思路（对标爱搜税 kernel/qu.py 的 _find_year / standardize_query / get_company）：
    全部用正则 + 词典做"零成本、可解释"的抽取，不调 LLM——实体抽取规则化即可达到高准确率，
    没必要为它付出 LLM 延迟与费用。公司名借助 companynameparser（与爱搜税一致）。

健壮性：companynameparser 是可选重依赖，未安装时安全降级（公司名留空），不影响其它实体抽取。
"""
from __future__ import annotations

import re

from app.schemas.document import Entities
from config.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 预编译正则（模块级，避免每次调用重复编译）
# ---------------------------------------------------------------------------
# 文号：尽量覆盖税务领域常见写法。
#   财税〔2024〕18号 / 财政部 税务总局公告2023年第19号 / 国家税务总局公告2023年第3号 / 国务院令第773号
_RE_DOC_NO = re.compile(
    r"[一-龥]{0,15}?(?:〔|\[|【)\s*(?:19|20)\d{2}\s*(?:〕|\]|】)\s*\d+\s*号"  # 字轨〔年〕序号号
    r"|[一-龥]{0,15}?公告\s*(?:19|20)\d{2}\s*年第\s*\d+\s*号"               # xx公告20xx年第n号
    r"|[一-龥]{0,15}?令第?\s*\d+\s*号"                                       # xx令第n号
)
# 年份：19xx / 20xx，后面常跟"年"。
_RE_YEAR = re.compile(r"(?:19|20)\d{2}")
# 法规名（书名号内）：《企业所得税法》《xxx管理办法》
_RE_LAW_BOOK = re.compile(r"《(.+?)》")
# 法规名（无书名号、以体裁词结尾时的弱抽取）：xxx税法 / xxx管理办法 / xxx暂行条例 ...
_RE_LAW_SUFFIX = re.compile(
    r"([一-龥]{2,20}?(?:税法|管理办法|暂行条例|实施细则|条例|管理法))"
)

# 地域词典：省 + 直辖市 + 常见城市（轻量，仅做演示性抽取，生产可换更全的地名库）。
# 注意排序——长词在前，避免"黑龙江"被"龙江"之类短词截断（这里词都完整，故按长度倒序保险）。
_REGION_WORDS = [
    "北京", "天津", "上海", "重庆",
    "河北", "山西", "辽宁", "吉林", "黑龙江", "江苏", "浙江", "安徽", "福建", "江西",
    "山东", "河南", "湖北", "湖南", "广东", "广西", "海南", "四川", "贵州", "云南",
    "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆", "内蒙古", "香港", "澳门", "台湾",
    "深圳", "广州", "杭州", "南京", "苏州", "成都", "武汉", "西安", "厦门", "宁波",
    "青岛", "大连", "无锡", "佛山", "东莞",
]
_REGION_WORDS_SORTED = sorted(_REGION_WORDS, key=len, reverse=True)

# 商品名抽取的辅助句式："xxx的税率 / xxx税率 / xxx的税收分类编码 / xxx编码"。
# 仅做规则级弱抽取；强抽取（如多商品并列）由 LLM 在 PRODUCT_CODE 意图下另行处理。
# 用 [^...和与及或、]：禁止把连词/顿号纳入商品名内部，避免"和建筑服务"这类脏抽取。
_RE_PRODUCT = re.compile(
    r"([一-龥A-Za-z]{2,12}?)(?:的)?(?:税率|税收分类编码|商品编码|开票编码)"
)
# 商品名前缀里需要剥离的连词/标点（出现在抽取结果开头时去掉）。
_PRODUCT_LEADING_NOISE = ("和", "与", "及", "或", "、", "，", ",", "的")


class EntityExtractor:
    """从用户 query 中抽取结构化实体，产出 Entities。

    用法::

        ext = EntityExtractor()
        ents = ext.extract("杭州市财税〔2024〕18号关于《企业所得税法》的天然气税率")
        # ents.doc_no="财税〔2024〕18号", ents.year="2024", ents.law_name="企业所得税法",
        # ents.region=["杭州"], ents.product_name=["天然气"]

    extract 为无副作用纯计算，线程安全。
    """

    def extract(self, query: str) -> Entities:
        """抽取全部实体。

        各实体相互独立、互不阻塞：单类抽取出错只影响该类（置空/略过），不影响其它类。

        :param query: 用户原始问题，允许空/None。
        :return: Entities 实例；无任何命中时返回各字段为空的 Entities。
        :raise: 不向外抛异常；内部异常按字段就地降级。
        """
        logger.info("实体抽取开始，query=%s", query)
        ents = Entities()
        if not query or not query.strip():
            logger.info("实体抽取：query 为空，返回空实体")
            return ents

        q = query.strip()

        # 1) 文号（取第一个命中即可，通常一句话只点名一个文号）
        ents.doc_no = self._extract_doc_no(q)
        # 2) 年份（优先取文号里的年份；否则取句中第一个年份）
        ents.year = self._extract_year(q, ents.doc_no)
        # 3) 法规名（书名号优先，其次体裁词结尾弱抽取）
        ents.law_name = self._extract_law_name(q)
        # 4) 地域（词典子串匹配，去重保序）
        ents.region = self._extract_region(q)
        # 5) 商品名（句式弱抽取，去重保序）
        ents.product_name = self._extract_product_name(q)
        # 6) 公司名（companynameparser，可选依赖，降级安全）
        ents.company = self._extract_company(q)

        logger.info(
            "实体抽取结束：doc_no=%s, year=%s, law_name=%s, region=%s, product=%s, company=%s",
            ents.doc_no, ents.year, ents.law_name, ents.region, ents.product_name, ents.company,
        )
        return ents

    # ------------------------------------------------------------------ #
    # 各实体的内部抽取方法
    # ------------------------------------------------------------------ #
    def _extract_doc_no(self, q: str) -> str | None:
        """抽取文号（如 财税〔2024〕18号）。匹配多种写法，取第一个命中。"""
        m = _RE_DOC_NO.search(q)
        if not m:
            return None
        # 去掉内部多余空格，规整成紧凑文号（如 "财税 〔2024〕 18 号" -> "财税〔2024〕18号"）。
        return re.sub(r"\s+", "", m.group(0))

    def _extract_year(self, q: str, doc_no: str | None) -> str | None:
        """抽取年份：优先用文号里的年份（更可靠），否则取句中第一个 19xx/20xx。"""
        # 文号里通常自带年份，直接复用，避免句中其它数字干扰。
        if doc_no:
            m = _RE_YEAR.search(doc_no)
            if m:
                return m.group(0)
        m = _RE_YEAR.search(q)
        return m.group(0) if m else None

    def _extract_law_name(self, q: str) -> str | None:
        """抽取法规名：书名号《...》优先（最可靠），否则用体裁词结尾做弱抽取。"""
        m = _RE_LAW_BOOK.search(q)
        if m:
            return m.group(1).strip()
        m = _RE_LAW_SUFFIX.search(q)
        return m.group(1).strip() if m else None

    def _extract_region(self, q: str) -> list[str]:
        """抽取地域：词典子串匹配，去重且保持出现顺序。"""
        hits: list[str] = []
        for w in _REGION_WORDS_SORTED:
            if w in q and w not in hits:
                hits.append(w)
        return hits

    def _extract_product_name(self, q: str) -> list[str]:
        """抽取商品名：用"xxx税率/编码"句式弱抽取，去重保序。"""
        names: list[str] = []
        for m in _RE_PRODUCT.finditer(q):
            name = m.group(1).strip()
            # 剥离开头的连词/标点（如"和建筑服务"->"建筑服务"），可能叠加出现，循环剥到干净为止。
            changed = True
            while changed:
                changed = False
                for noise in _PRODUCT_LEADING_NOISE:
                    if name.startswith(noise) and len(name) > len(noise):
                        name = name[len(noise):]
                        changed = True
            # 过滤掉"商品/税收"这类前缀词本身被误抽的情况。
            if name and name not in {"商品", "税收", "开票"} and name not in names:
                names.append(name)
        return names

    def _extract_company(self, q: str) -> list[str]:
        """抽取公司名：用 companynameparser（与爱搜税一致）。

        companynameparser 为可选重依赖；未安装/解析异常时安全降级为空列表，不影响其它实体。
        """
        try:
            # 延迟导入：保证未安装该库时本模块仍可 import、其它实体仍可抽取。
            import companynameparser  # type: ignore

            parsed = companynameparser.parse(q)  # 返回 {brand/place/trade/suffix/symbol...}
            # 拼出"地名+字号+行业+后缀"作为公司全名；任一段缺失则跳过。
            place = parsed.get("place", "") or ""
            brand = parsed.get("brand", "") or ""
            trade = parsed.get("trade", "") or ""
            suffix = parsed.get("suffix", "") or ""
            full = f"{place}{brand}{trade}{suffix}".strip()
            # 至少要有"字号 + 后缀（如 有限公司）"才认为是有效公司名，避免把零散词当公司。
            if brand and suffix:
                return [full]
            return []
        except ImportError:
            logger.info("未安装 companynameparser，公司名抽取降级为空")
            return []
        except Exception as e:
            logger.error("公司名抽取异常，降级为空：%s", e, exc_info=True)
            return []


if __name__ == "__main__":
    # 最小自测块（仅验证正则层，不依赖 companynameparser 也能跑）。
    ext = EntityExtractor()
    samples = [
        "财税〔2024〕18号关于研发费用加计扣除的规定",
        "杭州市2024年《企业所得税法》怎么适用",
        "国家税务总局公告2023年第3号说了什么",
        "天然气的税率和建筑服务税率分别是多少",
        "广东省深圳市的小微企业优惠",
    ]
    for s in samples:
        e = ext.extract(s)
        print(f"{s}\n  -> doc_no={e.doc_no}, year={e.year}, law={e.law_name}, "
              f"region={e.region}, product={e.product_name}\n")
