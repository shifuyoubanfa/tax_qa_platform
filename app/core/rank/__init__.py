"""排序子包（rank）：召回之后、重排之前的两段排序。
本包在整体链路里的位置：QU -> 召回 -> 【粗排 CoarseRanker -> 精排 FineRanker】 -> 重排 -> 摘要。
- coarse_rank.py：多路召回结果用 RRF(倒数排名融合) 融合 + 按 doc_id 去重，快速收敛候选集。
- fine_rank.py：对粗排候选用向量余弦相似(标题/正文)做精排，进一步提纯。
"""
