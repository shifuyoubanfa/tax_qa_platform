"""Reranker（重排）客户端 —— 本地加载 bge-reranker-v2-m3（FlagEmbedding，离线可用）。

本模块在整体链路里的位置：基础设施层。重排阶段(app/core/rerank/rerank.py)用它对"粗排+精排"
后的少量候选(几十条)做最终精细打分：把 (query, passage) 成对喂给 cross-encoder 重排模型，
拿到相关性分数，按分数取 top-k 作为送入大模型的最终上下文。

模型：**bge-reranker-v2-m3**（BAAI），cross-encoder 交叉编码——query 与 passage **一起进模型**
做深度交互，排序最准但最慢，所以只对少量候选做。

为什么本地加载（不走在线服务）：
- 没有现成的重排 HTTP 服务；用 FlagEmbedding 的 FlagReranker 直接把模型下到本地加载，离线可跑、可控。
- 模型不大(约 600MB)，没有 GPU 时用 CPU 也能跑（候选只有几十条，延迟可接受）；有 GPU 更快。

惰性 + 进程内单例：import 不加载模型；首次 rerank() 才加载，且**整个进程只加载一次**
（模型重，绝不能每次都重新加载）。

接口契约（INTERFACES，签名不可变）：
- rerank(query: str, passages: list[str]) -> list[float]   与 passages 等长、越大越相关。
"""
from __future__ import annotations

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class RerankerClient:
    """重排客户端（本地 FlagReranker，惰性加载 + 进程内单例）。

    用法::

        rc = RerankerClient()
        scores = rc.rerank("增值税税率", ["增值税基本税率13%...", "个税起征点5000..."])
        # scores 与 passages 一一对应，分数越大越相关
    """

    # 类级共享：进程内只保留一个已加载的模型实例。模型有数百MB，重复加载既慢又占内存。
    _shared_model = None

    def __init__(self) -> None:
        """仅记录配置，不加载模型（惰性）。"""
        logger.info("[Reranker客户端] 初始化（未加载模型）：model=%s local_path=%s",
                    settings.reranker_model, settings.reranker_local_path)

    def _get_model(self):
        """惰性加载本地 bge-reranker（首次 rerank 才加载，之后进程内复用）。

        模型来源优先级：
          RERANKER_LOCAL_PATH（本地已下好的目录）> RERANKER_MODEL（仓库名，FlagReranker 自动下载）。

        :return: FlagReranker 实例。
        :raise RuntimeError: 未安装 FlagEmbedding/torch，或模型加载失败。
        """
        if RerankerClient._shared_model is None:
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as e:
                raise RuntimeError(
                    "[Reranker客户端] 需要 FlagEmbedding + torch，请先 `pip install -U FlagEmbedding`"
                ) from e
            model_path = settings.reranker_local_path or settings.reranker_model
            if not model_path:
                raise RuntimeError("[Reranker客户端] 未配置 RERANKER_LOCAL_PATH / RERANKER_MODEL")
            logger.info("[Reranker客户端] 开始加载本地重排模型：%s（首次会自动下载，稍慢）", model_path)
            # use_fp16=True：半精度，省内存/显存、提速，对排序精度影响极小（CPU 上也能用）。
            RerankerClient._shared_model = FlagReranker(model_path, use_fp16=True)
            logger.info("[Reranker客户端] 重排模型加载完成")
        return RerankerClient._shared_model

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """对 (query, passage) 成对打分，返回与 passages 等长的相关性分数。

        :param query: 用户查询。
        :param passages: 候选文档文本列表。
        :return: list[float]，分数越大越相关；passages 为空时返回 []。
        :raise RuntimeError: 模型加载/打分失败（由上层 ReRanker 捕获并降级为精排顺序）。
        """
        if not passages:
            return []
        model = self._get_model()
        # 组装成对输入：[[query, p1], [query, p2], ...]，cross-encoder 要 query 和 doc 成对进
        pairs = [[query, p] for p in passages]
        try:
            # normalize=True：把模型原始 logits 经 sigmoid 压到 [0,1]，便于比较/阈值过滤
            scores = model.compute_score(pairs, normalize=True)
        except Exception as e:  # noqa: BLE001 - 统一转清晰中文报错；上层会降级
            logger.error("[Reranker客户端] 本地重排打分失败：%s", e, exc_info=True)
            raise RuntimeError(f"[Reranker客户端] 本地重排失败：{e}") from e
        # 只有 1 条候选时，compute_score 可能返回标量，这里统一成 list[float]
        if not isinstance(scores, list):
            scores = [scores]
        result = [float(s) for s in scores]
        logger.info("[Reranker客户端] 重排完成：候选=%d → 返回 %d 个分数", len(passages), len(result))
        return result


if __name__ == "__main__":
    # 自测：首次运行会自动下载 bge-reranker-v2-m3（约 600MB，需网络/梯子或镜像）。
    # 装好 FlagEmbedding+torch 后可直接跑；第一条应明显比第二条相关、分数更高。
    try:
        rc = RerankerClient()
        s = rc.rerank("增值税税率是多少", ["增值税基本税率为13%", "个人所得税起征点5000元"])
        print("[reranker_client 自测] 分数 =>", s)
    except Exception as exc:  # noqa: BLE001
        print("[reranker_client 自测] 需要 FlagEmbedding+torch 且能下载模型（属预期）=>", exc)
