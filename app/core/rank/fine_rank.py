"""精排器（FineRanker）——对粗排候选用向量余弦相似(标题/正文)做二次提纯。
本模块在整体链路里的位置：粗排(RRF融合去重)之后、重排(cross-encoder)之前的"中段排序"。

为什么粗排之后还要精排（三段排序的分工）：
- 粗排(RRF)只看"名次"，快但粗，目的是从几百条召回里快速收敛到几十条候选；
- 精排(本模块)用 query 与 候选标题/正文 的【向量余弦相似度】重新打分，比 RRF 更贴合语义，
  但仍是"双塔/独立编码"——query 与 doc 各自编码后算相似，速度快、可批量；
- 重排(下一阶段 ReRanker)用 cross-encoder 把 query+doc 拼一起精细打分，最准但最慢，只对前若干条做。
这种"粗(快) -> 精(中) -> 重(准)"的漏斗，是检索系统兼顾效率与效果的经典套路。

method 三种取值（与 config.constants.FineRankMethod 对应）：
- bge_title  : 用"标题"算余弦相似（精确法规/政策汇集类常用——标题信息密度高）；
- bge_content: 用"正文"算余弦相似（通用问答/案例类常用——答案藏在正文里）；
- direct     : 不做语义精排，直接透传粗排顺序（如问答库已足够准、或不想再调向量服务时）。

EmbeddingClient 惰性初始化；向量服务异常时降级为"按粗排 score 排序"，不致整链路失败。

风格对标 爱搜税 kernel/rank/fine_rank.py（bge_with_only_title / bge_with_only_content / direct）。
"""
from __future__ import annotations

import math

from config.constants import FineRankMethod
from config.logging_config import get_logger
from app.schemas.document import Document
from app.utils.normalize import l2_normalize
from app.utils.timing import runtime

logger = get_logger(__name__)


class FineRanker:
    """精排器：按 method 用标题/正文向量余弦相似度对候选重排，取前 fine_topk。

    用法::

        ranker = FineRanker()
        docs = ranker.rank(query, coarse_docs, method="bge_content", topk=20)

    所有外部客户端惰性初始化，无 infra 时 import 不报错。
    """

    def __init__(self) -> None:
        self._embedding_client = None  # 惰性占位

    def _get_embedding_client(self):
        """惰性获取 Embedding 客户端（首次精排才构建）。

        :return: EmbeddingClient 实例。
        :raise RuntimeError: 构建失败时给出清晰中文报错。
        """
        if self._embedding_client is None:
            try:
                from app.clients.embedding_client import EmbeddingClient
                self._embedding_client = EmbeddingClient()
                logger.info("FineRanker 已惰性初始化 EmbeddingClient")
            except Exception as e:  # noqa: BLE001
                logger.error("初始化 EmbeddingClient 失败: %s", e, exc_info=True)
                raise RuntimeError(f"嵌入客户端初始化失败，无法做向量精排: {e}") from e
        return self._embedding_client

    @runtime
    def rank(self, query: str, docs: list[Document], method: str, topk: int) -> list[Document]:
        """对候选文档做精排，返回前 topk。

        :param query: 用于精排的查询文本（一般用原始 query，语义最准）。
        :param docs: 粗排后的候选 Document 列表。
        :param method: 精排方式（FineRankMethod 值：bge_title / bge_content / direct）。
        :param topk: 精排后保留条数。
        :return: 精排并截断后的 Document 列表。
        """
        logger.info("进入精排: method=%s, 输入 %s 条, topk=%s", method, len(docs), topk)
        if not docs:
            return []

        # direct：不做语义精排，直接按粗排现有 score 透传（仅保证有序 + 截断）
        if method == FineRankMethod.DIRECT.value:
            result = sorted(docs, key=lambda d: d.score, reverse=True)[:topk]
            logger.info("精排(direct透传)完成: 返回 %s 条", len(result))
            return result

        # bge_title / bge_content：取标题或正文作为待比较文本
        use_title = method == FineRankMethod.BGE_TITLE.value
        try:
            scored = self._semantic_rank(query, docs, use_title=use_title)
        except Exception as e:  # noqa: BLE001 - 向量服务异常则降级为粗排顺序
            logger.error("向量精排失败，降级为按粗排score排序: %s", e, exc_info=True)
            scored = sorted(docs, key=lambda d: d.score, reverse=True)

        result = scored[:topk]
        logger.info("精排完成: method=%s, 返回 %s 条", method, len(result))
        return result

    def _semantic_rank(self, query: str, docs: list[Document], use_title: bool) -> list[Document]:
        """用 query 与 标题/正文 的向量余弦相似度重新打分并排序。

        实现要点（为什么这么做）：
        - 一次性把 query 和所有候选文本批量送进 embed，减少网络往返(批处理比逐条快得多)；
        - 余弦相似 = 两个 L2 归一化向量的内积；先 l2_normalize 再点积，数值稳定且语义即"余弦"。

        :param query: 查询文本。
        :param docs: 候选文档。
        :param use_title: True 用标题，False 用正文。
        :return: 按余弦相似度降序的新列表（已把相似度写入 Document.score）。
        """
        emb = self._get_embedding_client()
        # 取每篇文档的比较文本；标题/正文为空时用另一者兜底，避免空文本影响相似度
        passages = [
            (d.title if use_title else d.content) or d.content or d.title or ""
            for d in docs
        ]
        # 向量化：embed/embed_query 都返回 {"dense": [[...], ...], "sparse": [...]}
        # embed_query 是单条，dense 里只有 1 个向量，取第 0 个才是该 query 的向量 list[float]
        q_dense = emb.embed_query(query).get("dense", [])
        q_vec = q_dense[0] if q_dense else []
        doc_vecs = emb.embed(passages).get("dense", [])  # 批量：每篇文档一个向量
        if not q_vec or not doc_vecs:
            logger.info("精排向量为空，降级为粗排顺序")
            return sorted(docs, key=lambda d: d.score, reverse=True)

        q_norm = l2_normalize(q_vec)
        for d, v in zip(docs, doc_vecs):
            # 写入余弦相似度作为新的精排 score（覆盖粗排 RRF 分，进入下一阶段）
            d.score = self._cosine(q_norm, l2_normalize(v))
        return sorted(docs, key=lambda d: d.score, reverse=True)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """两个【已 L2 归一化】向量的余弦相似度（即内积）。

        :param a: 已归一化向量。
        :param b: 已归一化向量。
        :return: 余弦相似度；长度不一致或为空时返回 0.0。
        """
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        # 已归一化时内积即余弦；防御性夹到 [-1,1] 避免浮点误差越界
        return max(-1.0, min(1.0, dot))


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：direct 模式不依赖向量服务，可直接验证排序与截断。
    sample = [
        Document(doc_id="A", title="A", score=0.3),
        Document(doc_id="B", title="B", score=0.9),
        Document(doc_id="C", title="C", score=0.6),
    ]
    ranker = FineRanker()
    out = ranker.rank("任意查询", sample, method=FineRankMethod.DIRECT.value, topk=2)
    print("[fine_rank 自测] direct透传 top2:", [(d.doc_id, d.score) for d in out])
    # 验证余弦计算
    print("[fine_rank 自测] cos([1,0],[1,0]) =", FineRanker._cosine([1.0, 0.0], [1.0, 0.0]))
    print("[fine_rank 自测] cos([1,0],[0,1]) =", FineRanker._cosine([1.0, 0.0], [0.0, 1.0]))
