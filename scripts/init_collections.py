"""初始化各存储的集合/索引（建库建表脚本，幂等）。
本模块在整体链路里的位置：离线侧"第0步"。在线检索依赖三套存储：
1. Milvus —— 文档稠密向量集合（向量召回），维度必须 == settings.embedding_dim。
2. Elasticsearch —— 文档全文索引（BM25/稀疏召回），mapping 用中文分词器。
3. Qdrant —— 数仓表/字段元数据向量集合（Text2SQL 的 schema linking）。

为什么单独成脚本：建集合/建索引是"一次性、需谨慎"的运维动作，不该混在数据导入里反复执行；
做成幂等（已存在则跳过）后，重复跑也安全，CI/部署可放心调用。

依赖关系：本脚本只调用 app/clients 下的客户端（MilvusClient/ESClient/QdrantMetaClient），
这些客户端均为"惰性连接"——脚本 import 阶段不会真连基础设施，只有真正调用 ensure_* 时才连接。

风格对标 standard/src/main/create_test_index.py（argparse + 幂等 + 详细中文日志）。
"""
from __future__ import annotations

import argparse
import sys

from config.logging_config import get_logger, setup_logging
from config.settings import settings

logger = get_logger(__name__)


# ============================================================
# ES 文档索引 mapping —— 仅作"推荐参考"（不由本脚本下发）
# ------------------------------------------------------------
# 说明（为什么放在这里却不直接使用）：
# 真正的建索引/下发 mapping 由 app/clients/es_client.py 的 ESClient.ensure_index() 内部完成
# （它内置了一份基础 mapping：title/content=text、kbase/doc_no=keyword、metadata=object）。
# 本脚本只负责"触发幂等建索引"，把 mapping 的细节收口在客户端，避免脚本与客户端两处 mapping 打架。
# 下面这份是"更细粒度的推荐 mapping"，留给作者学习参考：若要中文召回更好，可把它搬进
# ESClient.ensure_index——把 title/content 的 analyzer 改成 ik_max_word（需 ES 装 analysis-ik 插件），
# 并为 doc_no/policy_type/region/effective_status 建 keyword、publish_date 建 date 以支持精确过滤。
# 字段与 app/schemas/document.py 的 Document.metadata 约定保持一致。
# ============================================================
ES_DOC_MAPPING_REFERENCE: dict = {
    "settings": {
        "number_of_shards": 1,        # 单分片单副本：开发/学习足够，生产按量调大
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "doc_id": {"type": "keyword"},                 # 文档主键，精确匹配/去重
            "title": {                                      # 标题：建议中文分词 + 子keyword便于聚合
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "content": {                                    # 正文：中文分词，BM25 主战场
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart",
            },
            "kbase": {"type": "keyword"},                   # 来源知识库标识（KBase 值）
            "doc_no": {"type": "keyword"},                  # 文号，如 财税〔2024〕18号
            "policy_type": {"type": "keyword"},             # 政策类型
            "region": {"type": "keyword"},                  # 地域
            "effective_status": {"type": "keyword"},        # 生效状态（现行有效/已废止等）
            "publish_date": {                               # 发布日期，容错多种格式
                "type": "date",
                "format": "yyyy-MM-dd||yyyy/MM/dd||yyyy-MM-dd HH:mm:ss||epoch_millis",
            },
            "clause_path": {"type": "keyword"},             # 条款路径（第X条/第X款）
            "image_keys": {"type": "keyword"},              # MinIO 图片对象键数组（仅回显）
        }
    },
}


def init_milvus() -> None:
    """幂等创建 Milvus 文档稠密向量集合（维度取 settings.embedding_dim）。

    :return: 无返回值，副作用是确保集合存在。
    :raise RuntimeError: 当连接 Milvus 或建集合失败时，抛出带中文上下文的错误。
    """
    # 延迟导入：把"重依赖 pymilvus"的导入推迟到真正用时，保证脚本被 -h 查看帮助时也不报错
    from app.clients.milvus_client import MilvusClient

    collection = settings.milvus_doc_collection
    dim = settings.embedding_dim
    logger.info("[init][Milvus] 准备确保集合存在 collection=%s dim=%s", collection, dim)
    try:
        client = MilvusClient()
        # ensure_collection 是幂等的：不存在则按 dim 建集合+索引并 load，存在则直接跳过。
        # 维度必须与 embedding_dim 一致，否则后续写入/检索会因维度不匹配报错。
        client.ensure_collection(collection=collection, dim=dim)
        logger.info("[init][Milvus] 集合就绪 collection=%s", collection)
    except Exception as exc:  # noqa: BLE001 - 统一兜底，给出清晰中文报错
        logger.error("[init][Milvus] 创建/检查集合失败：%s", exc, exc_info=True)
        raise RuntimeError(
            f"初始化 Milvus 集合失败（collection={collection}）。"
            f"请检查 settings.milvus_host/port 是否可达、维度 embedding_dim={dim} 是否正确。原始错误：{exc}"
        ) from exc


def init_es() -> None:
    """幂等创建 Elasticsearch 文档全文索引（mapping 由 ESClient.ensure_index 内置下发）。

    :return: 无返回值。
    :raise RuntimeError: 连接 ES 或建索引失败时抛出。
    """
    from app.clients.es_client import ESClient

    index = settings.es_doc_index
    logger.info("[init][ES] 准备确保索引存在 index=%s", index)
    try:
        client = ESClient()
        # ensure_index 幂等：不存在则用客户端内置 mapping 创建，存在则跳过。
        # （若需上面 ES_DOC_MAPPING_REFERENCE 的中文分词细化，请改 ESClient.ensure_index 内的 mapping。）
        client.ensure_index(index=index)
        logger.info("[init][ES] 索引就绪 index=%s", index)
    except Exception as exc:  # noqa: BLE001
        logger.error("[init][ES] 创建/检查索引失败：%s", exc, exc_info=True)
        raise RuntimeError(
            f"初始化 ES 索引失败（index={index}）。"
            f"请检查 settings.es_hosts 是否可达；若改用 IK 中文分词，请确认 ES 已装 analysis-ik 插件。原始错误：{exc}"
        ) from exc


def init_qdrant() -> None:
    """幂等创建 Qdrant 元数据向量集合（Text2SQL schema linking 用，维度取 embedding_dim）。

    :return: 无返回值。
    :raise RuntimeError: 连接 Qdrant 或建集合失败时抛出。
    """
    from app.clients.qdrant_meta_client import QdrantMetaClient

    collection = settings.qdrant_meta_collection
    dim = settings.embedding_dim
    logger.info("[init][Qdrant] 准备确保集合存在 collection=%s dim=%s", collection, dim)
    try:
        client = QdrantMetaClient()
        # 元数据向量与文档向量同源（同一个 embedding 模型），维度同样取 embedding_dim。
        client.ensure_collection(collection=collection, dim=dim)
        logger.info("[init][Qdrant] 集合就绪 collection=%s", collection)
    except Exception as exc:  # noqa: BLE001
        logger.error("[init][Qdrant] 创建/检查集合失败：%s", exc, exc_info=True)
        raise RuntimeError(
            f"初始化 Qdrant 集合失败（collection={collection}）。"
            f"请检查 settings.qdrant_host/port 是否可达、embedding_dim={dim} 是否正确。原始错误：{exc}"
        ) from exc


def init_all(targets: list[str]) -> None:
    """按指定目标逐个初始化存储。

    :param targets: 目标列表，元素取值 {"milvus","es","qdrant"}。
    :return: 无返回值。
    """
    logger.info("[init] 开始初始化存储，目标=%s", targets)
    # 用映射表把目标名 -> 初始化函数，便于扩展，避免一长串 if/elif
    dispatch = {"milvus": init_milvus, "es": init_es, "qdrant": init_qdrant}
    for name in targets:
        fn = dispatch.get(name)
        if fn is None:
            logger.warning("[init] 跳过未知目标：%s（可选 milvus/es/qdrant）", name)
            continue
        fn()
    logger.info("[init] 全部目标初始化完成")


def _build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器（独立成函数，便于测试与复用）。

    :return: 配置好的 ArgumentParser。
    """
    parser = argparse.ArgumentParser(
        description="幂等初始化 Milvus 集合 / ES 索引 / Qdrant 集合（建库建表，跑一次即可）。"
    )
    parser.add_argument(
        "--target",
        choices=["all", "milvus", "es", "qdrant"],
        default="all",
        help="要初始化的目标存储；all=全部（默认）。",
    )
    # 说明：当前各客户端的 ensure_* 仅"幂等创建"（已存在则跳过），不提供"先删后建"。
    # 如确需重置，请到对应控制台/客户端手动删除集合或索引后再跑本脚本。
    return parser


def main() -> None:
    """脚本入口：解析参数 -> 初始化日志 -> 按目标建库建表。

    :return: 无返回值。
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    # 进程级日志初始化（脚本独立运行，需要自己 setup 一次）
    setup_logging()

    # Windows 控制台默认编码可能不是 utf-8，重配一下避免中文日志乱码（对标 standard 脚本）
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001 - reconfigure 在少数环境不可用，忽略即可
            pass

    targets = ["milvus", "es", "qdrant"] if args.target == "all" else [args.target]
    init_all(targets)


if __name__ == "__main__":
    # 学习提示：
    #   py scripts/init_collections.py                 # 初始化全部存储（幂等，已存在则跳过）
    #   py scripts/init_collections.py --target es     # 只建 ES 索引
    # 注意：需先在 .env 配好对应基础设施地址，且服务可达。
    main()
