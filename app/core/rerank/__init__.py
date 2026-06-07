"""重排子包（rerank）：检索漏斗的最后一道、也是最精的一道排序。
本包在整体链路里的位置：QU -> 召回 -> 粗排 -> 精排 -> 【重排 ReRanker】 -> 摘要。
- rerank.py：ReRanker，调用 RerankerClient(cross-encoder, 如 bge-reranker-v2-m3) 对
  query 与每篇候选拼接精细打分，取最相关的前 rerank_topk 条作为最终引用上下文。
"""
