"""路由子包（router）：检索链路的"决策中枢"。
本包在整体链路里的位置：QU(查询理解, 产出意图) -> 【SearchRouter 路由】 -> 召回/排序 或 Text2SQL。
- search_router.py：SearchRouter，依据 QU 的意图查"意图->检索策略"映射表(对标爱搜税
  search_instance_mapping)，产出 RetrievalPlan：非结构化问题给出多步召回计划(选哪些库/什么权重/
  粗排精排topk/精排方式)；经营数据查询类则改走 Text2SQL Agent。
"""
