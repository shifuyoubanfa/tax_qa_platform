"""搜索路由器（SearchRouter）——把"用户意图"翻译成"具体怎么检索"的检索计划。
本模块在整体链路里的位置：QU 之后的第一个岔路口，是整条检索链路的"大脑"。
它维护一张【意图 -> 检索策略】映射表(SEARCH_INSTANCE_MAPPING)，对标爱搜税的 search_instance_mapping：
  - 非结构化问题(法规/政策/问答/案例/社保/商品编码)：产出一组 RecallStep，告诉下游
    "查哪些知识库、dense/sparse 权重各多少、粗排取多少、用什么方式精排、精排取多少"；
  - 经营数据查询类(Intent.DATA_QUERY)：不走 RAG，返回 route_type=text2sql + agent_name="text2sql_agent"，
    交给 Text2SQL Agent 把自然语言翻成 SQL 查数仓。

为什么"按意图调 dense/sparse 权重"是这套系统的关键设计：
- 稀疏(BM25)擅长关键词精确命中，稠密(向量)擅长语义泛化，二者强弱场景互补；
- 【精确法规类】用户往往带文号/法规全名(财税〔2024〕18号、企业所得税法)，要的是"一字不差命中那一篇"，
  此时关键词权重必须压倒语义 -> dense 0.05 / sparse 0.95；
- 【政策汇集类】也偏关键词(围绕某主题的政策名)，同样 sparse 重；
- 【通用问题类】用户用口语/同义改写提问，要的是"语义最接近的解答"，关键词常对不上 ->
  dense 0.95 / sparse 0.05；
- 【稽查案例/社保/商品编码】偏语义检索为主，dense 重；其中社保/商品编码本质偏结构化，
  这里先用语义召回兜底(真实生产可再挂结构化 Agent)。
这套权重直接来自爱搜税生产配置(见 search.py search_instance_mapping)，是经验沉淀。

注意：知识库选择用 KBase 枚举值，不写魔法字符串；找不到意图时回退到通用问题类策略。

风格对标 爱搜税 search.py 的 search_instance_mapping(意图->搜索实例)。
"""
from __future__ import annotations

from config.constants import DEFAULT_INTENT, Intent, KBase, RouteType, FineRankMethod
from config.logging_config import get_logger
from config.settings import settings
from app.schemas.document import QUResult, RecallStep, RetrievalPlan
from app.utils.timing import runtime

logger = get_logger(__name__)

# Text2SQL Agent 的名字（与 app/agents/text2sql_agent.py 的注册名约定一致）
TEXT2SQL_AGENT_NAME = "text2sql_agent"

# 意图 -> 附加的"结构化查表 Agent"名（对齐爱搜税 shebao_agent / product_agent）。
# 这些意图的答案是表里的精确值（社保基数/比例、税收分类编码）：除了文档 RAG 兜底，
# 再挂一个结构化 Agent 去查 MySQL 表，命中行并入召回池、一起进排序/摘要。
# 复用 RetrievalPlan.agent_name 字段承载（route_type 仍是 rag；与"替换式"的 text2sql 靠 route_type 区分）。
_INTENT_STRUCTURED_AGENT = {
    Intent.SOCIAL_SECURITY.value: "shebao_agent",
    Intent.PRODUCT_CODE.value: "product_code_agent",
}

# ============== 意图 -> 检索策略 映射表（系统的"路由表"，对标爱搜税 search_instance_mapping）==============
# 每个意图对应一个 RecallStep 列表(可多步：先查政策库再查问答库等)。
# weights 里的 dense/sparse 之和不必为1，融合时会各自归一化后加权(见 hybrid.py)。
# 这里用 lambda 延迟构造，避免在 import 阶段就固化 settings 的 topk(便于测试时改配置后即时生效)。
def _build_mapping() -> dict[str, list[RecallStep]]:
    """构造意图->策略映射表（运行时构造，便于读取最新 settings.topk）。

    :return: {意图值: [RecallStep, ...]} 映射。
    """
    coarse = settings.recall_topk   # 单库粗召回条数
    fine = settings.fine_topk       # 精排后保留条数
    return {
        # 精确法规类：要精确命中文号/法规名 -> sparse 压倒 dense；用标题精排(标题信息密度最高)
        Intent.PRECISE_REGULATION.value: [
            RecallStep(
                kbases=[KBase.POLICY.value, KBase.DOC.value],
                weights={"dense": 0.05, "sparse": 0.95},
                coarse_topk=coarse,
                fine_rank_method=FineRankMethod.BGE_TITLE.value,
                fine_topk=fine,
            ),
        ],
        # 政策汇集类：围绕主题找政策合集，仍偏关键词 -> sparse 重；标题精排
        Intent.POLICY_COLLECTION.value: [
            RecallStep(
                kbases=[KBase.POLICY.value, KBase.DOC.value],
                weights={"dense": 0.05, "sparse": 0.95},
                coarse_topk=coarse,
                fine_rank_method=FineRankMethod.BGE_TITLE.value,
                fine_topk=fine,
            ),
        ],
        # 通用问题类：口语化/语义为主 -> dense 重；正文精排(答案在正文里)；
        # 第二步并查问答库(历史问答常能直接命中答案)，问答库直接透传不再语义精排。
        Intent.GENERAL_QA.value: [
            RecallStep(
                kbases=[KBase.POLICY.value, KBase.DOC.value],
                weights={"dense": 0.95, "sparse": 0.05},
                coarse_topk=coarse,
                fine_rank_method=FineRankMethod.BGE_CONTENT.value,
                fine_topk=fine,
            ),
            RecallStep(
                kbases=[KBase.QA.value],
                weights={"dense": 1.0, "sparse": 0.0},
                coarse_topk=max(coarse // 2, 10),
                fine_rank_method=FineRankMethod.DIRECT.value,
                fine_topk=max(fine // 2, 5),
            ),
        ],
        # 稽查案例类：找相似案情 -> 语义为主(dense 重)，专查稽查案例库，正文精排
        Intent.INSPECT_CASE.value: [
            RecallStep(
                kbases=[KBase.INSPECT.value],
                weights={"dense": 0.9, "sparse": 0.1},
                coarse_topk=coarse,
                fine_rank_method=FineRankMethod.BGE_CONTENT.value,
                fine_topk=fine,
            ),
        ],
        # 查社保类：本质偏结构化，这里先用政策/问答库做语义召回兜底(真实生产可再挂社保Agent)
        Intent.SOCIAL_SECURITY.value: [
            RecallStep(
                kbases=[KBase.POLICY.value, KBase.QA.value],
                weights={"dense": 0.9, "sparse": 0.1},
                coarse_topk=coarse,
                fine_rank_method=FineRankMethod.BGE_CONTENT.value,
                fine_topk=fine,
            ),
        ],
        # 查商品编码类：同上，先语义召回相关说明文档兜底
        Intent.PRODUCT_CODE.value: [
            RecallStep(
                kbases=[KBase.POLICY.value, KBase.DOC.value, KBase.QA.value],
                weights={"dense": 0.9, "sparse": 0.1},
                coarse_topk=coarse,
                fine_rank_method=FineRankMethod.BGE_CONTENT.value,
                fine_topk=fine,
            ),
        ],
        # 经营数据查询类不在此表(改走 Text2SQL)，见 route() 内的分支处理。
    }


class SearchRouter:
    """搜索路由器：意图 -> RetrievalPlan。

    用法::

        router = SearchRouter()
        plan = router.route(qu)
        if plan.route_type == RouteType.TEXT2SQL.value:
            ... # 交给 Text2SQL Agent
        else:
            for step in plan.steps: ... # 逐步召回

    映射表在实例化时构造一次(读取当时的 settings.topk)。
    """

    def __init__(self) -> None:
        self.mapping = _build_mapping()
        logger.info("SearchRouter 加载完成, 已注册意图路由: %s", list(self.mapping.keys()))

    @runtime
    def route(self, qu: QUResult) -> RetrievalPlan:
        """根据查询理解结果的意图，产出检索计划。

        :param qu: 查询理解结果（关键看 qu.intent）。
        :return: RetrievalPlan。经营数据查询类 -> route_type=text2sql；其余 -> route_type=rag + steps。
        """
        intent = qu.intent or DEFAULT_INTENT.value
        logger.info("命中意图: %s, 开始路由", intent)

        # 结构化数据查询：不走 RAG，改走 Text2SQL Agent
        if intent == Intent.DATA_QUERY.value:
            logger.info("意图=经营数据查询类 -> 路由到 Text2SQL Agent(%s)", TEXT2SQL_AGENT_NAME)
            return RetrievalPlan(
                route_type=RouteType.TEXT2SQL.value,
                steps=[],
                agent_name=TEXT2SQL_AGENT_NAME,
            )

        # 非结构化：按意图取召回步骤；未命中的意图回退到通用问题类策略(最安全的兜底)
        steps = self.mapping.get(intent)
        if steps is None:
            logger.info("意图 %s 未在路由表中，回退到通用问题类策略", intent)
            steps = self.mapping[DEFAULT_INTENT.value]

        # 附加结构化查表 Agent（社保/商品编码）：复用 agent_name 字段承载（route_type 仍是 rag），
        # 由 rag_recall 读取、把查表结果并入召回池；其余意图为 None=纯文档 RAG。
        structured_agent = _INTENT_STRUCTURED_AGENT.get(intent)
        plan = RetrievalPlan(route_type=RouteType.RAG.value, steps=steps, agent_name=structured_agent)
        logger.info(
            "路由完成: route_type=rag, 共 %s 个召回步骤, 涉及知识库=%s, 结构化Agent=%s",
            len(plan.steps),
            [s.kbases for s in plan.steps],
            structured_agent or "无",
        )
        return plan


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：逐个意图打印路由结果，直观看到权重/库随意图变化。
    router = SearchRouter()
    for it in Intent:
        plan = router.route(QUResult(raw_query="测试", intent=it.value))
        if plan.route_type == RouteType.TEXT2SQL.value:
            print(f"[router 自测] {it.value:8s} -> text2sql agent={plan.agent_name}")
        else:
            for i, step in enumerate(plan.steps):
                print(f"[router 自测] {it.value:8s} -> step{i} kbases={step.kbases} "
                      f"weights={step.weights} fine={step.fine_rank_method}")
