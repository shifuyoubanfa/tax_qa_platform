"""查询理解编排器（QueryUnderstanding）：把意图/实体/改写三步串成一个 QUResult。

本模块在整体链路里的位置：QU 子包的"门面（Facade）"，对外只暴露一个 understand()。
下游（SearchRouter / 召回层 / 摘要层）只认 QUResult，不关心内部由几个子模块拼出来。
对标爱搜税 kernel/qu.py 里 query_classification + query_understanding + query_rewrite 的整体编排。

链路：
    query
      │
      ├─ IntentClassifier.classify  -> intent        （决定检索策略）
      ├─ EntityExtractor.extract    -> entities       （精确过滤/路由辅助）
      └─ QueryRewriter.rewrite      -> sub_queries, hyde_doc（提升召回）
      │
      └─> 组装成 QUResult（含 is_short_query 标记）

设计要点：
    - 三个子步骤逻辑上独立，这里按顺序串行调用（实现简单、可读；如需提速可在召回前并行化，
      但 QU 调用量级低、收益有限，故保持串行以便作者学习）。
    - 整体 try/except：任何子步骤异常都不让 understand() 崩，至少返回"只含原始 query+默认意图"
      的 QUResult，保证 FastAPI 链路不中断。
"""
from __future__ import annotations

from app.schemas.document import QUResult, Entities
from config.constants import DEFAULT_INTENT
from config.logging_config import get_logger
from app.core.qu.intent import IntentClassifier
from app.core.qu.extractor import EntityExtractor
from app.core.qu.query_rewrite import QueryRewriter
from app.utils.timing import runtime

logger = get_logger(__name__)


class QueryUnderstanding:
    """QU 子包门面：一次 understand() 产出完整 QUResult。

    用法::

        qu_engine = QueryUnderstanding()
        qu = qu_engine.understand("研发费用加计扣除政策有哪些")
        # qu.intent / qu.entities / qu.sub_queries / qu.hyde_doc / qu.is_short_query

    三个子组件在 __init__ 一次性构建并复用（它们都是无状态/带缓存的），understand 可并发调用。
    """

    def __init__(self) -> None:
        """构建三个子组件（意图分类 / 实体抽取 / 查询改写）。"""
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.query_rewriter = QueryRewriter()

    @runtime  # 计时装饰器：结束时 logger.info 打印本次 QU 总耗时（对标爱搜税 @runtime）
    def understand(self, query: str) -> QUResult:
        """对单条 query 做完整查询理解，产出 QUResult。

        :param query: 用户原始问题（可为经过纠错的 query），允许空/None。
        :return: QUResult。即便内部异常，也保证返回一个可用对象
                 （raw_query=原始query、intent=默认意图、sub_queries 至少含原始 query）。
        :raise: 不向外抛异常。
        """
        logger.info("查询理解(QU)开始，query=%s", query)
        # 入参兜底：空 query 直接返回最小可用 QUResult。
        if not query or not query.strip():
            logger.info("查询理解：query 为空，返回最小 QUResult")
            return QUResult(raw_query=query or "")

        raw = query.strip()
        try:
            # 1) 意图分类（决定后续走哪条策略）
            intent = self.intent_classifier.classify(raw)

            # 2) 实体抽取（文号/年份/法规名/地域/商品名/公司）
            entities = self.entity_extractor.extract(raw)

            # 3) 子查询扩写 + HyDE（提升召回）
            sub_queries, hyde_doc = self.query_rewriter.rewrite(raw, intent)

            # 4) 短查询标记（短查询走特殊摘要模板，见摘要层）
            is_short = QueryRewriter.is_short_query(raw)

            result = QUResult(
                raw_query=raw,
                intent=intent,
                sub_queries=sub_queries,
                hyde_doc=hyde_doc,
                entities=entities,
                is_short_query=is_short,
            )
            logger.info(
                "查询理解(QU)结束：intent=%s，子查询数=%d，is_short=%s",
                intent, len(sub_queries), is_short,
            )
            return result
        except Exception as e:
            # 整体兜底：任一步骤崩了也返回可用结果，保证检索链路继续。
            logger.error("查询理解(QU)异常，返回降级 QUResult：%s", e, exc_info=True)
            return QUResult(
                raw_query=raw,
                intent=DEFAULT_INTENT.value,
                sub_queries=[raw],          # 至少保留原始 query 供召回
                hyde_doc="",
                entities=Entities(),
                is_short_query=QueryRewriter.is_short_query(raw),
            )


if __name__ == "__main__":
    # 最小自测块：无外部服务时，意图/实体走规则、改写优雅降级，整体仍产出完整 QUResult。
    engine = QueryUnderstanding()
    for q in ["小微企业税收优惠政策有哪些", "财税〔2024〕18号", "社保基数"]:
        r = engine.understand(q)
        print(f"\nquery={q}")
        print(f"  intent={r.intent}, is_short={r.is_short_query}")
        print(f"  entities.doc_no={r.entities.doc_no}, year={r.entities.year}")
        print(f"  sub_queries={r.sub_queries}")
