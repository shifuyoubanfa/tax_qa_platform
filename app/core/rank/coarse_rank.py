"""粗排器（CoarseRanker）——多路召回结果的 RRF 融合 + 按 doc_id 去重 + 取 topk。
本模块在整体链路里的位置：召回(可能来自多个知识库、多个子查询)之后第一道排序。
召回阶段每个 (知识库 × 子查询) 都会产出一个有序列表，同一篇文档可能在多个列表里以不同名次出现。
粗排的任务：把这些"多路有序列表"融合成一个统一有序列表，并把重复文档合成一条。

为什么用 RRF(Reciprocal Rank Fusion，倒数排名融合) 而不是直接比较各路原始分数：
1. 不同路的分数量纲/分布天然不可比（向量余弦 vs BM25 vs 不同子查询的难易），
   直接相加会被某一路主导；而 RRF 只看"名次"，对分数尺度不敏感、鲁棒性强。
2. RRF 公式：score(doc) = Σ_路 1 / (k + rank_路(doc))，rank 从 0 或 1 起算，k 是平滑常数(常取60)。
   一篇文档只要在多路里都排名靠前，累加后总分就高 —— 天然实现"多路投票、共识优先"。
3. 工业界(Elasticsearch/向量库)广泛采用，简单、无需训练、效果稳定。
   （对标爱搜税 seacore.utils.rrf.RRF 在 coarse_rank 里的用法。）

去重策略：按 doc_id 合并；合并时保留各路里更强的稠密/稀疏得分，RRF 分写入 score。

风格对标 爱搜税 kernel/rank/coarse_rank.py（多知识库召回融合 + duplicate 去重）。
"""
from __future__ import annotations

from collections import defaultdict

from config.logging_config import get_logger
from app.schemas.document import Document
from app.utils.timing import runtime

logger = get_logger(__name__)

# RRF 平滑常数：经验值 60（来自 Cormack 等人的原始论文），
# 作用是削弱头部名次的过度影响，让靠前但非第一的文档也有合理贡献。
RRF_K = 60


class CoarseRanker:
    """粗排器：把多路召回结果用 RRF 融合并按 doc_id 去重，取前 topk。

    用法::

        ranker = CoarseRanker()
        docs = ranker.rank(recalled_docs, topk=50)

    :param rrf_k: RRF 平滑常数，越大则名次差异被压得越平，默认 60。
    """

    def __init__(self, rrf_k: int = RRF_K) -> None:
        self.rrf_k = rrf_k

    @runtime
    def rank(self, docs: list[Document], topk: int) -> list[Document]:
        """对召回文档做 RRF 融合 + 去重，返回前 topk。

        :param docs: 召回阶段合并来的文档（可能含跨库/跨子查询的重复条目）。
        :param topk: 粗排后保留条数。
        :return: 按 RRF 综合分降序、去重后的前 topk 个 Document。
        """
        logger.info("进入粗排(RRF融合+去重): 输入文档 %s 条, topk=%s", len(docs), topk)
        if not docs:
            return []

        # 第一步：把"扁平的文档列表"还原成"多路有序子列表"。
        # 召回结果里每条 Document 带 raw_query_from(来源子查询) 与 kbase(来源库)，
        # 用 (kbase, raw_query_from) 作为"一路"的标识；每一路内部按当前 score 降序得到名次。
        groups: dict[tuple[str, str], list[Document]] = defaultdict(list)
        for d in docs:
            groups[(d.kbase, d.raw_query_from)].append(d)
        for key in groups:
            # 路内按召回综合分降序，得到该路的排名(rank=下标)
            groups[key].sort(key=lambda x: x.score, reverse=True)
        logger.info("粗排识别到 %s 路召回结果参与 RRF 融合", len(groups))

        # 第二步：RRF 累加 + 去重。用 best 字典保存每个 doc_id 的代表 Document。
        rrf_scores: dict[str, float] = defaultdict(float)
        best: dict[str, Document] = {}
        for _route_key, route_docs in groups.items():
            for rank, d in enumerate(route_docs):
                # RRF 核心公式：名次越靠前(rank 越小)贡献越大
                rrf_scores[d.doc_id] += 1.0 / (self.rrf_k + rank)
                self._keep_best(best, d)

        # 第三步：把 RRF 分写回 Document.score，按分排序取 topk
        fused: list[Document] = []
        for doc_id, doc in best.items():
            doc.score = rrf_scores[doc_id]
            fused.append(doc)
        fused.sort(key=lambda x: x.score, reverse=True)
        result = fused[:topk]
        logger.info("粗排完成: 去重后 %s 条, 返回前 %s 条", len(fused), len(result))
        return result

    @staticmethod
    def _keep_best(best: dict[str, Document], d: Document) -> None:
        """把文档并入去重字典；同一 doc_id 已存在时合并更强的稠密/稀疏得分与缺失字段。

        :param best: {doc_id: 代表Document}，原地修改。
        :param d: 待合并文档。
        """
        if not d.doc_id:
            return
        if d.doc_id not in best:
            best[d.doc_id] = d
            return
        exist = best[d.doc_id]
        # 保留两路中更强的信号，便于后续精排/调试观察
        exist.dense_score = max(exist.dense_score, d.dense_score)
        exist.sparse_score = max(exist.sparse_score, d.sparse_score)
        if not exist.title and d.title:
            exist.title = d.title
        if not exist.content and d.content:
            exist.content = d.content


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：构造两路召回，验证 RRF 让"两路都靠前"的文档胜出。
    route1 = [  # 来源: policy 库, 子查询 q1
        Document(doc_id="A", title="A", kbase="policy", raw_query_from="q1", score=0.9),
        Document(doc_id="B", title="B", kbase="policy", raw_query_from="q1", score=0.8),
    ]
    route2 = [  # 来源: doc 库, 子查询 q2
        Document(doc_id="B", title="B", kbase="doc", raw_query_from="q2", score=0.95),
        Document(doc_id="C", title="C", kbase="doc", raw_query_from="q2", score=0.7),
    ]
    ranker = CoarseRanker()
    out = ranker.rank(route1 + route2, topk=10)
    # B 在两路都靠前，RRF 累加后应排第一
    print("[coarse_rank 自测] 排序:", [(d.doc_id, round(d.score, 4)) for d in out])
