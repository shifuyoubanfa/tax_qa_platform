"""Elasticsearch 全文/BM25 客户端（惰性连接）。
本模块在整体链路里的位置：基础设施层。文档库的"稀疏(BM25/全文)召回"走 ES——
它擅长关键词精确命中（法规名、文号、专有名词），与 Milvus 的稠密语义召回互补，
两路在 hybrid 层加权融合。

为什么稀疏召回单独放 ES 而不是只用 Milvus 的 sparse：
- ES 的全文检索成熟（分词、BM25、布尔/短语查询、聚合过滤），中文可配 ik 分词，
  对"精确法规类"意图的关键词命中尤其重要；与 Milvus 形成"语义+关键词"双引擎。

惰性：import 不连库；首次用到才建 Elasticsearch 客户端。检索失败返回空、不断链路。

接口契约（INTERFACES，签名不可变）：
- search_bm25(index, text, topk, filters=None) -> list[dict]
- index_doc(index, doc_id, body) -> bool
- ensure_index(index) -> None   （幂等建索引 + mapping）
"""
from __future__ import annotations

from typing import Any, Optional

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class ESClient:
    """Elasticsearch 客户端封装（惰性连接 + 幂等建索引 + BM25 检索/写入）。

    用法::

        es = ESClient()
        es.ensure_index(settings.es_doc_index)
        hits = es.search_bm25(settings.es_doc_index, "增值税税率", topk=50,
                              filters={"kbase": "policy"})

    :return: 见各方法说明。
    """

    def __init__(self) -> None:
        """构造仅保存配置，不建连接（惰性）。"""
        self._client = None  # elasticsearch.Elasticsearch 实例
        logger.info("[ES客户端] 初始化（未连接）：hosts=%s", settings.es_hosts)

    # ---------------- 惰性连接 ----------------
    def _get_client(self):
        """惰性建立 ES 连接。

        :return: elasticsearch.Elasticsearch 实例。
        :raise RuntimeError: 未安装 elasticsearch，或连接失败。
        """
        if self._client is None:
            try:
                from elasticsearch import Elasticsearch
            except ImportError as e:
                raise RuntimeError(
                    "[ES客户端] 未安装 elasticsearch，请 `pip install 'elasticsearch>=8.13,<9'`"
                ) from e
            # es_hosts 支持逗号分隔多节点
            hosts = [h.strip() for h in settings.es_hosts.split(",") if h.strip()]
            # 开启鉴权时才传 basic_auth，避免空账号导致报错
            basic_auth = (
                (settings.es_user, settings.es_password) if settings.es_user else None
            )
            try:
                self._client = Elasticsearch(
                    hosts=hosts,
                    basic_auth=basic_auth,
                    request_timeout=30,           # 单次请求超时
                    verify_certs=False,           # 本地/自签证书场景；生产应配真实证书
                )
                logger.info("[ES客户端] 连接对象已建立：%s", hosts)
            except Exception as e:
                logger.error("[ES客户端] 连接失败：%s", e, exc_info=True)
                raise RuntimeError(f"[ES客户端] 连接 ES 失败（{hosts}）：{e}") from e
        return self._client

    # ---------------- 幂等建索引 ----------------
    def ensure_index(self, index: str) -> None:
        """幂等创建索引：已存在则跳过；不存在则建带 mapping 的新索引。

        mapping：title/content 用 text(可被 BM25 分词检索)，kbase/doc_no 用 keyword(精确过滤)，
        metadata 用 object(灵活存放)。

        :param index: 索引名。
        :raise RuntimeError: 建索引失败。
        """
        client = self._get_client()
        try:
            if client.indices.exists(index=index):
                logger.info("[ES客户端] 索引已存在，跳过创建：%s", index)
                return
            mapping = {
                "mappings": {
                    "properties": {
                        # text 类型参与全文分词与 BM25 打分；中文可在此处接 ik 分词器
                        "title": {"type": "text"},
                        "content": {"type": "text"},
                        # keyword 用于精确过滤(term)，不分词
                        "kbase": {"type": "keyword"},
                        "doc_no": {"type": "keyword"},
                        "metadata": {"type": "object", "enabled": True},
                        # 图片对象键列表：仅随 _source 回显给前端取图、不参与检索，
                        # 显式声明 keyword 与权威 mapping(scripts/init_collections.py)对齐，避免依赖动态映射。
                        "image_keys": {"type": "keyword"},
                    }
                }
            }
            client.indices.create(index=index, body=mapping)
            logger.info("[ES客户端] 索引创建完成：%s", index)
        except Exception as e:
            logger.error("[ES客户端] 创建索引失败：%s", e, exc_info=True)
            raise RuntimeError(f"[ES客户端] 创建索引【{index}】失败：{e}") from e

    # ---------------- 写入 ----------------
    def index_doc(self, index: str, doc_id: str, body: dict) -> bool:
        """写入/更新单条文档（按 doc_id 幂等，重复 id 覆盖）。

        :param index: 索引名。
        :param doc_id: 文档主键（与 Milvus 主键保持一致，便于多路融合按 id 去重）。
        :param body: 文档内容(title/content/kbase/doc_no/metadata)。
        :return: 是否成功。
        :raise RuntimeError: 写入失败。
        """
        client = self._get_client()
        try:
            client.index(index=index, id=doc_id, document=body)
            logger.info("[ES客户端] 写入文档完成：index=%s id=%s", index, doc_id)
            return True
        except Exception as e:
            logger.error("[ES客户端] 写入文档失败：%s", e, exc_info=True)
            raise RuntimeError(f"[ES客户端] 写入索引【{index}】失败：{e}") from e

    # ---------------- 检索 ----------------
    def search_bm25(
        self,
        index: str,
        text: str,
        topk: int,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """BM25 全文检索（title^2 + content，title 权重更高）。

        :param index: 索引名。
        :param text: 查询文本。
        :param topk: 返回条数。
        :param filters: 可选精确过滤（term），如 {"kbase": "policy"}；走 filter 子句不影响打分。
        :return: list[dict(id, score, fields)]；fields 为 _source。失败/空返回 []。
        """
        client = self._get_client()
        # bool 查询：must 做 BM25 打分，filter 做精确过滤(不参与打分、可缓存)。
        must = [{
            "multi_match": {
                "query": text,
                "fields": ["title^2", "content"],  # 标题命中更重要，权重 ^2
            }
        }]
        filter_clauses = []
        if filters:
            for k, v in filters.items():
                filter_clauses.append({"term": {k: v}})
        query_body = {
            "size": topk,
            "query": {"bool": {"must": must, "filter": filter_clauses}},
        }
        try:
            resp = client.search(index=index, body=query_body)
        except Exception as e:
            # 检索失败不断链路（某路挂了由其它路兜底），记录并返回空。
            logger.error("[ES客户端] BM25 检索失败（index=%s）：%s", index, e, exc_info=True)
            return []

        hits = resp.get("hits", {}).get("hits", [])
        out: list[dict] = []
        for h in hits:
            out.append({
                "id": str(h.get("_id", "")),
                "score": float(h.get("_score", 0.0)),
                "fields": h.get("_source", {}) or {},
            })
        logger.info("[ES客户端] BM25 检索完成：index=%s，命中=%d 条", index, len(out))
        return out


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：需要 ES 在线。
    try:
        es = ESClient()
        es.ensure_index(settings.es_doc_index)
        print("[es_client 自测] ensure_index 完成（需 ES 在线）")
    except Exception as exc:
        print("[es_client 自测] 需要 ES 在线（属预期）=>", exc)
