"""召回子包（recall）：负责"从知识库里把候选文档捞出来"这一步。
本包在整体链路里的位置：QU(查询理解) -> 【召回】 -> 粗排 -> 精排 -> 重排 -> 摘要。
包含三个层次：
- base.py：召回器抽象基类 BaseRecaller，统一 recall(qu, step, kbase) 接口。
- hybrid.py：HybridRecaller，单库 dense(向量) + sparse(BM25) 混合召回并加权融合。
- manager.py：RecallManager，按检索步骤遍历多个知识库、对多个子查询并行召回再合并。
"""
