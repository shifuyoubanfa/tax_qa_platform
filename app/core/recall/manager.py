"""召回管理器（RecallManager）——把一个"召回步骤(RecallStep)"在多个知识库上并行执行并合并。
本模块在整体链路里的位置：召回层的"调度中枢"。SearchRouter 产出的检索计划里，每个 RecallStep
可能涉及多个知识库（如 ["policy","doc"]）；RecallManager 负责：
  1) 遍历 step.kbases，对每个库调用 HybridRecaller.recall（dense+sparse 融合）；
  2) 多个知识库【并行】召回（用线程池，对标爱搜税 hybrid.py 的 threading 并行），降低总时延；
  3) 把各库结果合并成一个大列表，交给下游粗排(CoarseRanker)做多路融合去重。

为什么用线程而不是 asyncio：
- 召回的瓶颈在"等待外部服务(Milvus/ES)返回"，是 IO 密集；底层客户端(pymilvus/elasticsearch)
  多为同步阻塞 API，用 ThreadPoolExecutor 并发最直接、改造成本最低（与爱搜税一致）。
- 真要全异步需要底层客户端也异步化，超出本层职责，故这里选线程池。

注意：本层只负责"召回并合并"，不做排序/去重——那是粗排层(CoarseRanker)的职责，
这样各层单一职责、便于学习与替换。

风格对标 爱搜税 kernel/recall/hybrid.py 的多 Query / 多知识库线程并行召回。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from config.logging_config import get_logger
from app.schemas.document import Document, QUResult, RecallStep
from app.utils.timing import runtime
from app.core.recall.hybrid import HybridRecaller

logger = get_logger(__name__)


class RecallManager:
    """召回管理器：按 RecallStep 在多个知识库上并行召回并合并结果。

    用法::

        manager = RecallManager()
        docs = manager.recall(qu, step)   # step.kbases 里的所有库并行召回后合并

    :param recaller: 可注入自定义召回器（便于测试时替换为假召回器）；默认用 HybridRecaller。
    :param max_workers: 线程池并发上限（按知识库数量并行）。
    """

    def __init__(self, recaller: HybridRecaller | None = None, max_workers: int = 8) -> None:
        # 默认混合召回器；测试时可注入桩对象，避免真连 infra
        self.recaller = recaller or HybridRecaller()
        self.max_workers = max_workers

    @runtime
    def recall(self, qu: QUResult, step: RecallStep) -> list[Document]:
        """对 step.kbases 中的每个知识库并行召回，合并为单个文档列表。

        :param qu: 查询理解结果。
        :param step: 召回步骤配置（含要查的知识库列表 kbases、权重、topk 等）。
        :return: 各知识库召回结果合并后的 Document 列表（未去重、未排序，留给粗排处理）。
        """
        kbases = step.kbases or []
        logger.info("进入召回管理: 本步知识库=%s, 权重=%s", kbases, step.weights)
        if not kbases:
            logger.info("召回步骤未配置任何知识库，返回空")
            return []

        merged: list[Document] = []
        # 线程池并发：每个知识库一个任务；worker 不超过库数量，避免浪费线程
        workers = min(self.max_workers, len(kbases))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # 提交时记录 future -> kbase，便于异常日志定位是哪个库挂了
            future_map = {
                pool.submit(self._recall_one, qu, step, kbase): kbase
                for kbase in kbases
            }
            for future in as_completed(future_map):
                kbase = future_map[future]
                try:
                    docs = future.result()
                    merged.extend(docs)
                    logger.info("知识库 %s 召回 %s 条", kbase, len(docs))
                except Exception as e:  # noqa: BLE001 - 单库失败不拖垮整体召回
                    logger.error("知识库 %s 召回异常(已跳过): %s", kbase, e, exc_info=True)

        logger.info("召回管理完成: 合并后共 %s 条(待粗排去重融合)", len(merged))
        return merged

    def _recall_one(self, qu: QUResult, step: RecallStep, kbase: str) -> list[Document]:
        """在单个知识库上召回（线程任务体）。

        独立成方法的原因：让线程池提交逻辑清晰，异常能精确归因到具体知识库。

        :param qu: 查询理解结果。
        :param step: 召回步骤配置。
        :param kbase: 目标知识库标识。
        :return: 该库召回到的 Document 列表。
        """
        return self.recaller.recall(qu, step, kbase)


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：注入"假召回器"，验证多库并行合并逻辑，不连真实 infra。
    class _FakeRecaller(HybridRecaller):
        def recall(self, qu, step, kbase):  # 覆盖真实召回，直接造数据
            return [Document(doc_id=f"{kbase}-1", title=f"{kbase}文档", kbase=kbase, score=0.5)]

    fake_qu = QUResult(raw_query="增值税税率", sub_queries=["增值税税率"])
    fake_step = RecallStep(kbases=["policy", "qa", "doc"])
    mgr = RecallManager(recaller=_FakeRecaller())
    out = mgr.recall(fake_qu, fake_step)
    print("[manager 自测] 合并条数:", len(out), "->", [d.doc_id for d in out])
