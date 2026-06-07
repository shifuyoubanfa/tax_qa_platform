"""QU（Query Understanding，查询理解）子包。

本包在整体链路里的位置：
    用户问题 --> 【QU(本包)】 --> SearchRouter(路由) --> 召回/排序/重排 --> 摘要/Agent --> 答案

本包做什么：
    把用户一句自然语言问题，加工成下游检索可直接消费的"结构化理解结果"（QUResult），包含：
      1. intent      意图分类（决定走哪条检索策略）          —— intent.py
      2. entities    结构化实体（文号/年份/法规名/地域/商品名/公司） —— extractor.py
      3. sub_queries 子查询扩写 + hyde_doc 假设性文档（提升召回） —— query_rewrite.py
      4. is_short_query 短查询标记（短查询走特殊摘要模板）

    四个子模块各司其职，由 understanding.py 的 QueryUnderstanding 统一编排。

设计蓝本：爱搜税 kernel/qu.py（正则优先的意图判定 + 文号/年份抽取 + LLM 兜底/改写）。
对外只暴露这四个类，下游统一从本包 import。
"""
from app.core.qu.intent import IntentClassifier
from app.core.qu.extractor import EntityExtractor
from app.core.qu.query_rewrite import QueryRewriter
from app.core.qu.understanding import QueryUnderstanding

__all__ = [
    "IntentClassifier",
    "EntityExtractor",
    "QueryRewriter",
    "QueryUnderstanding",
]
