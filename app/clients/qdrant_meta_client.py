"""Qdrant 元数据向量库客户端（qdrant-client，惰性连接）。
本模块在整体链路里的位置：基础设施层，专为 Text2SQL 的 schema linking 服务。
做法：把数仓里每张表、每个字段、每个业务指标的"自然语言描述"向量化后写进 Qdrant，
Text2SQL Agent 拿到用户问题时，先在这里做语义检索，召回"最相关的表/字段/指标"，
再据此让 LLM 生成 SQL。这样模型不必把整个库的 schema 都塞进 prompt，准且省 token。

为什么用 Qdrant 而非复用 Milvus：
- 职责隔离：文档检索(Milvus/ES) 与 schema 元数据检索(Qdrant) 是两套不同生命周期的数据，
  分库便于独立维护、独立重建；Qdrant 的 payload 过滤 + 轻量部署也很适合元数据这种小而精的场景。

惰性：import 不连库；首次用到才建客户端。检索失败返回空、不断链路。

接口契约（INTERFACES，签名不可变）：
- search(collection, vec, topk) -> list[dict]
- upsert(collection, points) -> int   （points: list[dict(id, vector, payload)]）
- ensure_collection(collection, dim=None) -> None  （幂等建集合）
"""
from __future__ import annotations

from typing import Optional

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class QdrantMetaClient:
    """Qdrant 元数据向量库封装（惰性连接 + 幂等建集合 + 检索/写入）。

    用法::

        qc = QdrantMetaClient()
        qc.ensure_collection(settings.qdrant_meta_collection)
        hits = qc.search(settings.qdrant_meta_collection, query_vec, topk=10)

    :return: 见各方法说明。
    """

    def __init__(self) -> None:
        """构造仅保存配置，不建连接（惰性）。"""
        self._client = None  # qdrant_client.QdrantClient 实例
        logger.info("[Qdrant客户端] 初始化（未连接）：%s:%s",
                    settings.qdrant_host, settings.qdrant_port)

    # ---------------- 惰性连接 ----------------
    def _get_client(self):
        """惰性建立 Qdrant 连接。

        :return: qdrant_client.QdrantClient 实例。
        :raise RuntimeError: 未安装 qdrant-client，或连接失败。
        """
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as e:
                raise RuntimeError(
                    "[Qdrant客户端] 未安装 qdrant-client，请 `pip install qdrant-client`"
                ) from e
            try:
                self._client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                    api_key=settings.qdrant_api_key or None,  # Cloud 才需要
                    timeout=30,
                )
                logger.info("[Qdrant客户端] 连接对象已建立：%s:%s",
                            settings.qdrant_host, settings.qdrant_port)
            except Exception as e:
                logger.error("[Qdrant客户端] 连接失败：%s", e, exc_info=True)
                raise RuntimeError(f"[Qdrant客户端] 连接 Qdrant 失败：{e}") from e
        return self._client

    # ---------------- 幂等建集合 ----------------
    def ensure_collection(self, collection: str, dim: Optional[int] = None) -> None:
        """幂等创建集合：已存在则跳过；不存在则建（COSINE 距离）。

        :param collection: 集合名。
        :param dim: 向量维度；默认取 settings.embedding_dim（与写入向量一致）。
        :raise RuntimeError: 建集合失败。
        """
        client = self._get_client()
        vec_dim = dim or settings.embedding_dim
        try:
            # collection_exists 幂等判断（新版 qdrant-client 提供）
            if client.collection_exists(collection):
                logger.info("[Qdrant客户端] 集合已存在，跳过创建：%s", collection)
                return
            from qdrant_client.models import Distance, VectorParams

            client.create_collection(
                collection_name=collection,
                # COSINE 与 bge-m3 dense 的相似度度量一致
                vectors_config=VectorParams(size=vec_dim, distance=Distance.COSINE),
            )
            logger.info("[Qdrant客户端] 集合创建完成：%s（dim=%d）", collection, vec_dim)
        except Exception as e:
            logger.error("[Qdrant客户端] 创建集合失败：%s", e, exc_info=True)
            raise RuntimeError(f"[Qdrant客户端] 创建集合【{collection}】失败：{e}") from e

    # ---------------- 写入 ----------------
    def upsert(self, collection: str, points: list[dict]) -> int:
        """批量写入/更新点（按 id upsert）。

        :param collection: 集合名。
        :param points: list[dict]，每条含 id(主键)、vector(向量)、payload(表名/字段名/描述等元信息)。
        :return: 实际写入条数。
        :raise RuntimeError: 写入失败。
        """
        if not points:
            return 0
        client = self._get_client()
        try:
            from qdrant_client.models import PointStruct

            structs = [
                PointStruct(id=p["id"], vector=p["vector"], payload=p.get("payload", {}))
                for p in points
            ]
            client.upsert(collection_name=collection, points=structs)
            logger.info("[Qdrant客户端] upsert 完成：集合=%s，写入=%d 条", collection, len(structs))
            return len(structs)
        except Exception as e:
            logger.error("[Qdrant客户端] upsert 失败：%s", e, exc_info=True)
            raise RuntimeError(f"[Qdrant客户端] 写入集合【{collection}】失败：{e}") from e

    # ---------------- 检索 ----------------
    def search(self, collection: str, vec: list[float], topk: int) -> list[dict]:
        """向量检索（schema linking：召回最相关的表/字段/指标）。

        :param collection: 集合名。
        :param vec: 查询向量。
        :param topk: 返回条数。
        :return: list[dict(id, score, payload)]；失败/空返回 []。
        """
        client = self._get_client()
        try:
            results = client.search(
                collection_name=collection,
                query_vector=vec,
                limit=topk,
                with_payload=True,
            )
        except Exception as e:
            # 检索失败不断链路，记录并返回空。
            logger.error("[Qdrant客户端] 检索失败（集合=%s）：%s", collection, e, exc_info=True)
            return []

        out: list[dict] = []
        for r in results:
            out.append({
                "id": r.id,
                "score": float(r.score),
                "payload": r.payload or {},
            })
        logger.info("[Qdrant客户端] 检索完成：集合=%s，命中=%d 条", collection, len(out))
        return out


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：需要 Qdrant 在线。
    try:
        qc = QdrantMetaClient()
        qc.ensure_collection(settings.qdrant_meta_collection)
        print("[qdrant_meta_client 自测] ensure_collection 完成（需 Qdrant 在线）")
    except Exception as exc:
        print("[qdrant_meta_client 自测] 需要 Qdrant 在线（属预期）=>", exc)
