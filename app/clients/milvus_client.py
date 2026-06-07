"""Milvus 稠密向量库客户端（pymilvus，惰性连接）。
本模块在整体链路里的位置：基础设施层。文档库(policy/qa/doc/inspect_case)的"稠密(向量)召回"
走 Milvus——把 bge-m3 的 dense 向量存进来，召回时用查询向量做 ANN 检索拿 top-k。
（稀疏/全文召回走 ES，两路在 hybrid 层融合。）

为什么用 pymilvus 的 MilvusClient（而非旧的 connections+Collection API）：
- MilvusClient 是 2.4+ 推荐的高层封装，建表/插入/检索 API 更简洁，惰性建连也更好控制。

惰性：import 不连库；首次用到时才 connect。所有方法对"连接失败/集合不存在"做容错，
返回空结果或抛清晰中文异常，绝不让 import 阶段崩溃。

接口契约（INTERFACES，签名不可变）：
- search(collection, dense_vec, topk, expr=None) -> list[dict(id, score, fields)]
- upsert(collection, rows) -> int                （rows: list[dict]，含主键/向量/标量字段）
- ensure_collection(collection, dim=None)         （幂等建集合 + 建索引 + load）
"""
from __future__ import annotations

from typing import Any, Optional

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class MilvusClient:
    """Milvus 客户端封装（惰性连接 + 幂等建集合 + 稠密检索/写入）。

    用法::

        mc = MilvusClient()
        mc.ensure_collection(settings.milvus_doc_collection)
        hits = mc.search(settings.milvus_doc_collection, query_vec, topk=50,
                         expr="kbase == 'policy'")

    :return: 见各方法说明。
    """

    # 集合的标量字段（除主键 id 与向量 vector 外）。入库脚本与召回层共用这套字段名。
    # 设计：把检索/展示需要的元信息直接冗余进 Milvus，召回后无需再回查别的库即可拼上下文。
    # image_keys：与 ES 一致，存该文档片段关联的图片键（字符串数组），多模态展示时回查图床用。
    _SCALAR_FIELDS = ["title", "content", "kbase", "doc_no", "metadata", "image_keys"]

    def __init__(self) -> None:
        """构造仅保存配置，不建连接（惰性）。"""
        self._client = None  # pymilvus.MilvusClient 实例
        logger.info("[Milvus客户端] 初始化（未连接）：%s:%s db=%s",
                    settings.milvus_host, settings.milvus_port, settings.milvus_db)

    # ---------------- 惰性连接 ----------------
    def _get_client(self):
        """惰性建立 Milvus 连接。

        :return: pymilvus.MilvusClient 实例。
        :raise RuntimeError: 未安装 pymilvus，或连接失败。
        """
        if self._client is None:
            try:
                from pymilvus import MilvusClient as _PyMilvusClient
            except ImportError as e:
                raise RuntimeError("[Milvus客户端] 未安装 pymilvus，请 `pip install pymilvus`") from e
            uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
            try:
                # token：开启鉴权时用 "user:password"，否则空串。
                token = (
                    f"{settings.milvus_user}:{settings.milvus_password}"
                    if settings.milvus_user else ""
                )
                self._client = _PyMilvusClient(
                    uri=uri, token=token, db_name=settings.milvus_db or "default"
                )
                logger.info("[Milvus客户端] 连接成功：%s db=%s", uri, settings.milvus_db)
            except Exception as e:
                logger.error("[Milvus客户端] 连接失败：%s", e, exc_info=True)
                raise RuntimeError(f"[Milvus客户端] 连接 Milvus 失败（{uri}）：{e}") from e
        return self._client

    # ---------------- 幂等建集合 ----------------
    def ensure_collection(self, collection: str, dim: Optional[int] = None) -> None:
        """幂等创建集合：已存在则跳过；不存在则建 schema + 向量索引并 load。

        schema：主键 id(VARCHAR) + vector(FLOAT_VECTOR, dim) + 若干标量字段。
        向量索引：HNSW + COSINE（与 bge-m3 dense 的相似度度量一致；COSINE 直观、效果稳）。

        :param collection: 集合名。
        :param dim: 向量维度；默认取 settings.embedding_dim（需与写入向量一致）。
        :raise RuntimeError: 建集合/建索引失败。
        """
        client = self._get_client()
        vec_dim = dim or settings.embedding_dim
        try:
            # has_collection 幂等判断
            if client.has_collection(collection):
                logger.info("[Milvus客户端] 集合已存在，跳过创建：%s", collection)
                return

            from pymilvus import DataType  # 惰性导入，建表才用到

            logger.info("[Milvus客户端] 开始创建集合：%s（dim=%d）", collection, vec_dim)
            schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=vec_dim)
            schema.add_field("title", DataType.VARCHAR, max_length=1024)
            schema.add_field("content", DataType.VARCHAR, max_length=65535)
            schema.add_field("kbase", DataType.VARCHAR, max_length=64)
            schema.add_field("doc_no", DataType.VARCHAR, max_length=256)
            # metadata 用 JSON 存放灵活的 clause_path/region/publish_date 等
            schema.add_field("metadata", DataType.JSON)
            # image_keys 用 JSON 存字符串数组（与 ES 字段一致），承载片段关联的图片键，供多模态回查图床
            schema.add_field("image_keys", DataType.JSON)

            # 向量索引：HNSW + COSINE
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 200},
            )
            client.create_collection(
                collection_name=collection, schema=schema, index_params=index_params
            )
            client.load_collection(collection)  # load 到内存后才能检索
            logger.info("[Milvus客户端] 集合创建并加载完成：%s", collection)
        except Exception as e:
            logger.error("[Milvus客户端] 创建集合失败：%s", e, exc_info=True)
            raise RuntimeError(f"[Milvus客户端] 创建集合【{collection}】失败：{e}") from e

    # ---------------- 写入 ----------------
    def upsert(self, collection: str, rows: list[dict]) -> int:
        """批量写入/更新（按主键 id upsert）。

        :param collection: 集合名。
        :param rows: 记录列表，每条须含 id 与 vector，及可选标量字段。
        :return: 实际写入条数。
        :raise RuntimeError: 写入失败。
        """
        if not rows:
            return 0
        client = self._get_client()
        try:
            client.upsert(collection_name=collection, data=rows)
            logger.info("[Milvus客户端] upsert 完成：集合=%s，写入=%d 条", collection, len(rows))
            return len(rows)
        except Exception as e:
            logger.error("[Milvus客户端] upsert 失败：%s", e, exc_info=True)
            raise RuntimeError(f"[Milvus客户端] 写入集合【{collection}】失败：{e}") from e

    # ---------------- 检索 ----------------
    def search(
        self,
        collection: str,
        dense_vec: list[float],
        topk: int,
        expr: Optional[str] = None,
    ) -> list[dict]:
        """稠密向量 ANN 检索。

        :param collection: 集合名。
        :param dense_vec: 查询向量（与建库 dim 一致）。
        :param topk: 返回条数。
        :param expr: 可选的标量过滤表达式，如 "kbase == 'policy'"（精确召回/按库过滤用）。
        :return: list[dict(id, score, fields)]；fields 为标量字段字典。失败/空返回 []。
        """
        client = self._get_client()
        try:
            results = client.search(
                collection_name=collection,
                data=[dense_vec],
                limit=topk,
                filter=expr or "",  # 空串表示不过滤
                output_fields=self._SCALAR_FIELDS,
                search_params={"metric_type": "COSINE", "params": {"ef": max(topk, 64)}},
            )
        except Exception as e:
            # 检索失败不抛断链路（某个库挂了不该让整条问答崩），记录并返回空，由上层多路兜底。
            logger.error("[Milvus客户端] 检索失败（集合=%s）：%s", collection, e, exc_info=True)
            return []

        # pymilvus 返回 [[hit, hit, ...]]（每个查询向量一组），这里只有一个查询向量取 [0]
        hits = results[0] if results else []
        out: list[dict] = []
        for h in hits:
            # 兼容不同 pymilvus 版本的 hit 取值方式
            entity = h.get("entity", {}) if isinstance(h, dict) else getattr(h, "entity", {})
            hid = h.get("id") if isinstance(h, dict) else getattr(h, "id", "")
            score = h.get("distance") if isinstance(h, dict) else getattr(h, "distance", 0.0)
            out.append({
                "id": str(hid),
                "score": float(score),
                "fields": {k: entity.get(k) for k in self._SCALAR_FIELDS} if entity else {},
            })
        logger.info("[Milvus客户端] 检索完成：集合=%s，命中=%d 条", collection, len(out))
        return out


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：需要 Milvus 在线。
    # 没服务时建连会抛清晰中文异常，属预期。
    try:
        mc = MilvusClient()
        mc.ensure_collection(settings.milvus_doc_collection)
        print("[milvus_client 自测] ensure_collection 完成（需 Milvus 在线）")
    except Exception as exc:
        print("[milvus_client 自测] 需要 Milvus 在线（属预期）=>", exc)
