"""重排器（ReRanker）——用 cross-encoder 重排模型对候选做最终精细打分。
本模块在整体链路里的位置：检索漏斗最后一段。精排(双塔余弦)把候选收敛到几十条后，
重排用更准的 cross-encoder(把 query 和 doc 拼在一起整体编码、输出相关性分) 重新打分，
取最相关的前 rerank_topk 条，作为喂给 LLM 的最终参考上下文。

为什么要单独一段重排（精排之后还重排的意义）：
- 精排是"双塔"：query 与 doc 各自独立编码再算余弦，速度快、可批量，但损失了二者的细粒度交互信息；
- 重排是"交互式(cross-encoder)"：query 和 doc 同时进模型，能捕捉词级对齐(谁回答了谁)，相关性判断最准；
- 但 cross-encoder 每对都要过一次模型，开销大，所以只对精排留下的少量候选做 —— 这正是漏斗的最后一档。

实现：调 RerankerClient.rerank(query, passages) 拿到每篇的相关性分，写入 Document.rerank_score，
按其降序取前 topk。RerankerClient 惰性初始化；服务异常时降级为"按上一阶段 score 排序"，不致整链路失败。

风格对标 爱搜税 kernel/rerank/re_rank.py（重排阶段日志 + 截断 + 兜底）。
"""
from __future__ import annotations

import math

from config.logging_config import get_logger
from app.schemas.document import Document
from app.utils.timing import runtime

logger = get_logger(__name__)

# "成功打分"的下限钳值：保证经过重排模型成功打分的文档 rerank_score 恒 > 0。
# 为什么需要（与下游"可答性门控"解耦）：bge-reranker 用 fp16 + sigmoid(normalize=True)，对【极不相关】
# 样本(logit≲-17)会下溢为【恰好 0.0】；而重排客户端异常【降级】时(见 rerank 的 except 分支)文档 rerank_score
# 保持默认 0.0(根本没打分)。两者都为 0.0 则无法区分——下游 answerability_check 会把"极强不相关的真低分"
# 误判成"重排降级"而错误放行(fail-open，恰与门控目标相反)。把成功打分的结果钳到 _SCORED_FLOOR 后：
# "成功打分"恒 ∈ [1e-6, 1]、"未打分"恒 = 0.0，0.0 被独占用于表示降级，二者彻底解耦。
_SCORED_FLOOR = 1e-6


class ReRanker:
    """重排器：用 RerankerClient 对候选打相关性分，取前 rerank_topk。

    用法::

        reranker = ReRanker()
        docs = reranker.rerank(query, fine_ranked_docs, topk=5)

    RerankerClient 惰性初始化，无 infra 时 import 不报错。
    """

    def __init__(self) -> None:
        self._reranker_client = None  # 惰性占位

    def _get_reranker_client(self):
        """惰性获取 Reranker 客户端（首次重排才构建）。

        :return: RerankerClient 实例。
        :raise RuntimeError: 构建失败时给出清晰中文报错。
        """
        if self._reranker_client is None:
            try:
                from app.clients.reranker_client import RerankerClient
                self._reranker_client = RerankerClient()
                logger.info("ReRanker 已惰性初始化 RerankerClient")
            except Exception as e:  # noqa: BLE001
                logger.error("初始化 RerankerClient 失败: %s", e, exc_info=True)
                raise RuntimeError(f"重排客户端初始化失败: {e}") from e
        return self._reranker_client

    @runtime
    def rerank(self, query: str, docs: list[Document], topk: int) -> list[Document]:
        """对候选文档做 cross-encoder 重排，返回前 topk。

        :param query: 用户查询（一般用原始 query，相关性判断最贴合用户真实意图）。
        :param docs: 精排后的候选 Document 列表。
        :param topk: 重排后最终保留条数（最终引用上下文规模）。
        :return: 按 rerank_score 降序、截断到 topk 的 Document 列表。
        """
        logger.info("进入重排: 输入 %s 条, topk=%s", len(docs), topk)
        if not docs:
            return []

        # 拼接每篇的待打分文本：标题 + 正文，信息更完整(cross-encoder 能同时看到二者)
        passages = [self._build_passage(d) for d in docs]
        try:
            client = self._get_reranker_client()
            scores = client.rerank(query, passages)  # -> list[float]，与 passages 一一对应
        except Exception as e:  # noqa: BLE001 - 重排服务异常则降级为上一阶段顺序
            logger.error("重排打分失败，降级为按上一阶段score排序: %s", e, exc_info=True)
            result = sorted(docs, key=lambda d: d.score, reverse=True)[:topk]
            return result

        # 把相关性分写入 rerank_score，并据此排序。
        # 钳到 _SCORED_FLOOR 并收敛 NaN/inf：成功打分恒为有限正数(>0)，与"降级·未打分(0.0)"区分，
        # 让下游可答性门控能可靠识别"重排降级"而不误杀极强不相关的真低分（见 _SCORED_FLOOR 注释）。
        for d, s in zip(docs, scores or []):
            sv = float(s)
            d.rerank_score = sv if (math.isfinite(sv) and sv > _SCORED_FLOOR) else _SCORED_FLOOR
        ranked = sorted(docs, key=lambda d: d.rerank_score, reverse=True)
        result = ranked[:topk]
        logger.info("重排完成: 返回 %s 条 (top1 rerank_score=%.4f)",
                    len(result), result[0].rerank_score if result else 0.0)
        return result

    @staticmethod
    def _build_passage(d: Document) -> str:
        """把一篇文档拼成给重排模型打分的文本（标题 + 正文）。

        为什么标题在前：标题信息密度高，cross-encoder 在长度受限截断时优先保住关键信息。

        :param d: 文档。
        :return: 用于重排的拼接文本。
        """
        title = (d.title or "").strip()
        content = (d.content or "").strip()
        if title and content:
            return f"{title}\n{content}"
        return title or content


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：用桩 client 验证"按 rerank_score 排序 + 截断"逻辑，不连真实服务。
    class _FakeReranker(ReRanker):
        def _get_reranker_client(self):  # 返回一个固定打分的假 client
            class _C:
                def rerank(self, query, passages):
                    # 故意让最后一条得分最高，验证重排能改变顺序
                    return [0.1 * (i + 1) for i in range(len(passages))]
            return _C()

    sample = [Document(doc_id=str(i), title=f"标题{i}", content=f"正文{i}", score=1.0) for i in range(3)]
    out = _FakeReranker().rerank("某税务问题", sample, topk=2)
    print("[rerank 自测] top2:", [(d.doc_id, round(d.rerank_score, 2)) for d in out])
