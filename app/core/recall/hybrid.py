"""混合召回器（HybridRecaller）——单知识库内 dense(向量) + sparse(BM25) 双路召回并加权融合。
本模块在整体链路里的位置：召回层的"主力"。对一个知识库 kbase：
  1) 把查询(子查询们 + HyDE文档)向量化 -> 调 MilvusClient 做稠密(语义)召回；
  2) 用查询文本 -> 调 ESClient 做稀疏(BM25/全文)召回；
  3) 两路结果各自 min-max 归一化后，按 step.weights 里的 dense/sparse 权重加权求和，得到综合 score；
  4) 给每条 Document 标注 dense_score/sparse_score/score、kbase、raw_query_from。

为什么要 dense + sparse 混合（核心设计动机）：
- 稠密(向量)召回擅长"语义相近但用词不同"（同义改写、口语化问法），但对精确的文号/法规名不敏感；
- 稀疏(BM25)召回擅长"关键词精确命中"（财税〔2024〕18号、企业所得税法这类专有名词），但不懂语义。
两者互补，按意图调权重（精确法规类偏 sparse、通用问答偏 dense，权重由 SearchRouter 决定）。

为什么融合前要归一化：
- dense 得分通常是余弦/内积(0~1 左右)，sparse 是 BM25(可能十几到几十)，量纲完全不同，
  直接相加会被 BM25 淹没。先各自 min-max 归一化到 [0,1] 再按权重相加才有可比性。
  （对标爱搜税 hybrid_shui5.py 里 dense/sparse 用 boost 加权、本实现把加权显式化到分数层面。）

外部依赖（EmbeddingClient/MilvusClient/ESClient）一律【惰性初始化】：首次召回才真正构建客户端，
保证没有 infra 时 import 不报错、FastAPI 能正常启动；任一路召回失败都降级为空，不影响另一路。

风格对标 爱搜税 kernel/recall/hybrid.py、hybrid_shui5.py（多路召回 + 加权融合）。
"""
from __future__ import annotations

from config.logging_config import get_logger
from config.settings import settings
from app.schemas.document import Document, QUResult, RecallStep
from app.utils.normalize import minmax_normalize
from app.utils.timing import runtime
from app.core.recall.base import BaseRecaller

logger = get_logger(__name__)


class HybridRecaller(BaseRecaller):
    """单库混合召回器：dense + sparse 双路召回并按权重融合。

    用法::

        recaller = HybridRecaller()
        docs = recaller.recall(qu, step, kbase="policy")

    所有外部客户端惰性初始化（见各 _get_xxx 方法），无 infra 时也能 import。
    """

    def __init__(self) -> None:
        # 惰性占位：真正用到时才构建（避免 import 阶段连接外部服务）
        self._embedding_client = None
        self._milvus_client = None
        self._es_client = None

    # ---------------- 惰性客户端构建 ----------------
    def _get_embedding_client(self):
        """惰性获取 Embedding 客户端（首次调用才构建）。

        :return: EmbeddingClient 实例。
        :raise RuntimeError: clients 包尚未就绪或构建失败时给出中文报错。
        """
        if self._embedding_client is None:
            try:
                # 延迟到函数内 import：clients 由 B 负责，import 阶段不强依赖
                from app.clients.embedding_client import EmbeddingClient
                self._embedding_client = EmbeddingClient()
                logger.info("HybridRecaller 已惰性初始化 EmbeddingClient")
            except Exception as e:  # noqa: BLE001 - 统一转成清晰中文报错
                logger.error("初始化 EmbeddingClient 失败: %s", e, exc_info=True)
                raise RuntimeError(f"嵌入客户端初始化失败，无法做稠密召回: {e}") from e
        return self._embedding_client

    def _get_milvus_client(self):
        """惰性获取 Milvus 客户端（稠密召回用）。"""
        if self._milvus_client is None:
            try:
                from app.clients.milvus_client import MilvusClient
                self._milvus_client = MilvusClient()
                logger.info("HybridRecaller 已惰性初始化 MilvusClient")
            except Exception as e:  # noqa: BLE001
                logger.error("初始化 MilvusClient 失败: %s", e, exc_info=True)
                raise RuntimeError(f"Milvus 客户端初始化失败，无法做稠密召回: {e}") from e
        return self._milvus_client

    def _get_es_client(self):
        """惰性获取 ES 客户端（稀疏/BM25 召回用）。"""
        if self._es_client is None:
            try:
                from app.clients.es_client import ESClient
                self._es_client = ESClient()
                logger.info("HybridRecaller 已惰性初始化 ESClient")
            except Exception as e:  # noqa: BLE001
                logger.error("初始化 ESClient 失败: %s", e, exc_info=True)
                raise RuntimeError(f"ES 客户端初始化失败，无法做稀疏召回: {e}") from e
        return self._es_client

    # ---------------- 对外主入口 ----------------
    @runtime
    def recall(self, qu: QUResult, step: RecallStep, kbase: str) -> list[Document]:
        """在单个知识库 kbase 上做 dense + sparse 混合召回并融合。

        :param qu: 查询理解结果（子查询 + HyDE + 意图 + 实体）。
        :param step: 召回步骤配置（含 weights、coarse_topk）。
        :param kbase: 目标知识库标识（KBase 值）。
        :return: 融合并按 score 降序的 Document 列表。
        """
        logger.info(
            "进入混合召回: kbase=%s, 权重=%s, 粗召回topk=%s, 子查询数=%s",
            kbase, step.weights, step.coarse_topk, len(qu.sub_queries) or 1,
        )
        weights = step.weights or {}
        dense_w = float(weights.get("dense", 0.0))
        sparse_w = float(weights.get("sparse", 0.0))
        topk = step.coarse_topk or settings.recall_topk

        # 用 {doc_id: Document} 聚合两路结果，避免同一文档被两路重复成两条
        merged: dict[str, Document] = {}

        # ---- 稠密路（权重>0才走，省掉无谓的向量化/查询）----
        if dense_w > 0:
            try:
                dense_docs = self._dense_recall(qu, kbase, topk)
                self._merge(merged, dense_docs, channel="dense")
                logger.info("稠密召回 kbase=%s 命中 %s 条", kbase, len(dense_docs))
            except Exception as e:  # noqa: BLE001 - 单路失败不拖垮另一路
                logger.error("稠密召回失败(已降级跳过) kbase=%s: %s", kbase, e, exc_info=True)

        # ---- 稀疏路 ----
        if sparse_w > 0:
            try:
                sparse_docs = self._sparse_recall(qu, kbase, topk)
                self._merge(merged, sparse_docs, channel="sparse")
                logger.info("稀疏召回 kbase=%s 命中 %s 条", kbase, len(sparse_docs))
            except Exception as e:  # noqa: BLE001
                logger.error("稀疏召回失败(已降级跳过) kbase=%s: %s", kbase, e, exc_info=True)

        # ---- 加权融合：综合 score = dense_w*dense_norm + sparse_w*sparse_norm ----
        self._fuse_scores(merged, dense_w, sparse_w)
        fused = sorted(merged.values(), key=lambda d: d.score, reverse=True)
        logger.info("混合召回完成 kbase=%s, 去重融合后共 %s 条", kbase, len(fused))
        return fused

    # ---------------- 稠密召回 ----------------
    def _dense_recall(self, qu: QUResult, kbase: str, topk: int) -> list[Document]:
        """稠密(向量语义)召回：把每个子查询(+HyDE)向量化后到 Milvus 检索。

        为什么把 HyDE 文档也加进去召回（设计动机）：
        - HyDE(Hypothetical Document Embeddings) 让 LLM 先"假想一段答案文档"，再用它的向量去召回，
          对短问题/口语问法能显著拉近与正式政策文本的语义距离，提升稠密召回命中率。

        :param qu: 查询理解结果。
        :param kbase: 目标知识库。
        :param topk: 每个子查询的召回条数。
        :return: 来自稠密通道的 Document 列表（dense_score 已填）。
        """
        collection = self.resolve_milvus_collection(kbase)
        emb = self._get_embedding_client()
        milvus = self._get_milvus_client()

        # 待向量化文本：原始/扩写子查询 + HyDE 假设文档（去空、去重保序）
        texts = self._dense_query_texts(qu)
        docs: list[Document] = []
        for text in texts:
            try:
                # embed_query 返回批量结构 {"dense": [[...]], "sparse": [{...}]}（单条放在列表第 0 位）
                dense_list = emb.embed_query(text).get("dense", [])
                if not dense_list or not dense_list[0]:
                    logger.info("子查询无稠密向量，跳过: %s", text[:30])
                    continue
                vec = dense_list[0]  # 取出该条的稠密向量 list[float]，供 Milvus 单向量检索
                # 【按库过滤】只召回当前 kbase 的文档，消除跨库串味与后续 RRF 双计。
                # kbase 是 KBase 枚举受控值（非用户输入），仍对单引号做转义防御，避免表达式被破坏。
                expr = f"kbase == '{kbase.replace(chr(39), chr(39) * 2)}'"
                hits = milvus.search(collection, vec, topk, expr=expr)  # -> list[dict(id,score,fields)]
                docs.extend(self._hits_to_docs_dense(hits, kbase, raw_query_from=text))
            except Exception as e:  # noqa: BLE001 - 单条子查询失败不影响其它
                logger.error("稠密子查询召回失败 query=%s: %s", text[:30], e, exc_info=True)
        return docs

    @staticmethod
    def _dense_query_texts(qu: QUResult) -> list[str]:
        """组装稠密召回要用的查询文本集合：子查询(原始+扩写) + HyDE 文档。

        :param qu: 查询理解结果。
        :return: 去空去重(保序)后的文本列表；若全空则兜底用原始 query。
        """
        candidates: list[str] = list(qu.sub_queries or [])
        if qu.hyde_doc:
            candidates.append(qu.hyde_doc)
        if not candidates and qu.raw_query:
            candidates.append(qu.raw_query)  # 兜底：QU 没产出子查询时至少用原始问题
        # 去重保序
        seen: set[str] = set()
        result: list[str] = []
        for t in candidates:
            t = (t or "").strip()
            if t and t not in seen:
                seen.add(t)
                result.append(t)
        return result

    @staticmethod
    def _hits_to_docs_dense(hits: list[dict], kbase: str, raw_query_from: str) -> list[Document]:
        """把 Milvus 命中(dict)转成 Document，并写入 dense_score。

        :param hits: MilvusClient.search 返回的 [{"id","score","fields"}...]。
        :param kbase: 来源知识库。
        :param raw_query_from: 召回该文档的子查询文本。
        :return: Document 列表。
        """
        docs: list[Document] = []
        for h in hits or []:
            fields = h.get("fields", {}) or {}
            doc = Document(
                doc_id=str(h.get("id", fields.get("doc_id", ""))),
                title=fields.get("title", ""),
                content=fields.get("content", ""),
                kbase=kbase,
                dense_score=float(h.get("score", 0.0)),  # Milvus 返回的相似度即稠密得分
                metadata=fields.get("metadata", {}) or {},
                image_keys=fields.get("image_keys", []) or [],
                raw_query_from=raw_query_from,
            )
            docs.append(doc)
        return docs

    # ---------------- 稀疏召回 ----------------
    def _sparse_recall(self, qu: QUResult, kbase: str, topk: int) -> list[Document]:
        """稀疏(BM25/全文)召回：用查询文本到 ES 做全文检索。

        :param qu: 查询理解结果。
        :param kbase: 目标知识库。
        :param topk: 每个子查询的召回条数。
        :return: 来自稀疏通道的 Document 列表（sparse_score 已填）。
        """
        index = self.resolve_es_index(kbase)
        es = self._get_es_client()
        # 稀疏召回不需要 HyDE（HyDE 是给向量用的）；用原始+扩写子查询即可
        texts = [t for t in (qu.sub_queries or [qu.raw_query]) if (t or "").strip()]
        if not texts:
            texts = [qu.raw_query]
        docs: list[Document] = []
        for text in texts:
            try:
                # 【按库过滤】filter 子句精确限定 kbase，不参与 BM25 打分，
                # 同样消除跨库串味与 RRF 双计，与稠密路的 expr 过滤对齐。
                hits = es.search_bm25(index, text, topk, filters={"kbase": kbase})  # -> list[dict]
                docs.extend(self._hits_to_docs_sparse(hits, kbase, raw_query_from=text))
            except Exception as e:  # noqa: BLE001
                logger.error("稀疏子查询召回失败 query=%s: %s", text[:30], e, exc_info=True)
        return docs

    @staticmethod
    def _hits_to_docs_sparse(hits: list[dict], kbase: str, raw_query_from: str) -> list[Document]:
        """把 ES 命中(dict)转成 Document，并写入 sparse_score。

        【对齐契约】ESClient.search_bm25 实际返回的是【扁平结构】，与 Milvus 同构：
        每条 = {"id": 主键, "score": BM25分, "fields": _source(含 title/content/metadata/image_keys)}。
        因此这里的解析方式必须与 _hits_to_docs_dense 一致——doc_id 从顶层 id 取、
        其余业务字段统一从 fields 取；否则 doc_id 恒空会被 _merge 丢弃，导致 BM25 整路失效。

        :param hits: ESClient.search_bm25 返回的命中列表 [{"id","score","fields"}...]。
        :param kbase: 来源知识库。
        :param raw_query_from: 召回该文档的子查询文本。
        :return: Document 列表。
        """
        docs: list[Document] = []
        for h in hits or []:
            fields = h.get("fields", {}) or {}  # fields 即 ES 的 _source
            doc_id = str(h.get("id") or fields.get("doc_id", ""))
            score = float(h.get("score", 0) or 0)  # BM25 得分
            doc = Document(
                doc_id=doc_id,
                title=fields.get("title", ""),
                content=fields.get("content", ""),
                kbase=kbase,
                sparse_score=score,  # BM25 得分写到 sparse 通道
                metadata=fields.get("metadata", {}) or {},
                image_keys=fields.get("image_keys", []) or [],
                raw_query_from=raw_query_from,
            )
            docs.append(doc)
        return docs

    # ---------------- 合并与融合 ----------------
    @staticmethod
    def _merge(merged: dict[str, Document], docs: list[Document], channel: str) -> None:
        """把某一路(channel)的召回结果并入聚合字典(按 doc_id 去重)。

        同一 doc_id 已存在时，只把本路的得分(dense/sparse)补写到已有 Document 上，
        这样一篇被两路同时召回的文档会同时带上 dense_score 与 sparse_score。

        :param merged: 聚合字典 {doc_id: Document}，原地修改。
        :param docs: 本路召回的文档。
        :param channel: "dense" 或 "sparse"，决定补写哪个得分字段。
        """
        for d in docs:
            if not d.doc_id:
                continue
            if d.doc_id in merged:
                exist = merged[d.doc_id]
                if channel == "dense":
                    # 取较大值：同一文档可能被多个子查询稠密召回，保留最强信号
                    exist.dense_score = max(exist.dense_score, d.dense_score)
                else:
                    exist.sparse_score = max(exist.sparse_score, d.sparse_score)
                # 补全可能缺失的标题/正文（某一路只返回了 id 时）
                if not exist.title and d.title:
                    exist.title = d.title
                if not exist.content and d.content:
                    exist.content = d.content
                # 补全图片附件键：某一路缺 image_keys 时用另一路补齐，保证后续上下文拼接不丢图。
                if not exist.image_keys and d.image_keys:
                    exist.image_keys = d.image_keys
            else:
                merged[d.doc_id] = d

    @staticmethod
    def _fuse_scores(merged: dict[str, Document], dense_w: float, sparse_w: float) -> None:
        """对聚合后的文档做归一化加权融合，写入综合 score。

        步骤（为什么这么做）：
        1. 分别取出所有文档的 dense_score / sparse_score，各自 min-max 归一化到 [0,1]，
           消除两路量纲差异（dense 余弦 vs sparse BM25）。
        2. score = dense_w * dense_norm + sparse_w * sparse_norm。
        3. 权重通常已由 SearchRouter 按意图设定（精确法规偏 sparse、通用问答偏 dense）。

        :param merged: 聚合字典 {doc_id: Document}，原地写入 score。
        :param dense_w: 稠密权重。
        :param sparse_w: 稀疏权重。
        """
        if not merged:
            return
        docs = list(merged.values())
        dense_norm = minmax_normalize([d.dense_score for d in docs])
        sparse_norm = minmax_normalize([d.sparse_score for d in docs])
        for d, dn, sn in zip(docs, dense_norm, sparse_norm):
            d.score = dense_w * dn + sparse_w * sn


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：不连真实 infra，只验证"融合/合并/去重"纯逻辑。
    r = HybridRecaller()
    bucket: dict[str, Document] = {}
    # 模拟稠密路命中两条
    r._merge(bucket, [
        Document(doc_id="A", title="财税18号", dense_score=0.9, raw_query_from="q1"),
        Document(doc_id="B", title="增值税法", dense_score=0.4, raw_query_from="q1"),
    ], channel="dense")
    # 模拟稀疏路命中 A(重复) 和 C(新)
    r._merge(bucket, [
        Document(doc_id="A", title="财税18号", sparse_score=20.0, raw_query_from="q2"),
        Document(doc_id="C", title="所得税法", sparse_score=12.0, raw_query_from="q2"),
    ], channel="sparse")
    r._fuse_scores(bucket, dense_w=0.5, sparse_w=0.5)
    for doc in sorted(bucket.values(), key=lambda x: x.score, reverse=True):
        print(f"[hybrid 自测] {doc.doc_id} score={doc.score:.3f} "
              f"dense={doc.dense_score} sparse={doc.sparse_score}")
