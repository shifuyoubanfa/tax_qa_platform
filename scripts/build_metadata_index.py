"""数仓元数据向量化入库脚本（Text2SQL 的 schema linking 基石）。
本模块在整体链路里的位置：离线侧，专为 Text2SQL Agent 服务。

为什么需要它：Text2SQL 把"自然语言问题"翻译成 SQL，第一步是 schema linking —— 在几十上百张表里
找到"和问题相关的表/字段"。直接把所有表结构塞进 LLM 提示词既超长又低效；正确做法是先把每张表、
每个字段的"语义描述"向量化存进 Qdrant，问题来了先做向量召回，只把最相关的少数表/字段喂给 LLM。
这个脚本就是把 MySQL 数仓的元数据（表名/表注释/字段名/字段注释/指标）拼成描述、向量化、写入 Qdrant。

完整流程：
    1) 用 MySQLClient（异步）列出所有表 -> 读每张表的字段结构。
    2) 把 表+字段 拼成一段自然语言"语义描述"（便于向量召回命中口语化问法）。
    3) EmbeddingClient.embed(描述) -> 向量。
    4) QdrantMetaClient.upsert 写入元数据集合（payload 带 table/column 等原始信息，召回后可还原）。
    --dry-run 时只读元数据并打印拼好的描述，不向量化、不写 Qdrant。

设计要点：MySQLClient 是异步客户端，故主流程用 async + asyncio.run 驱动；
所有外部连接惰性初始化，脚本 import 阶段不连库。

风格对标 掌柜问数（Text2SQL：多路召回选表/字段/指标）的 schema linking 思路。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

from config.logging_config import get_logger, setup_logging
from config.settings import settings

logger = get_logger(__name__)

# schema_meta.json 的默认路径（相对项目根：tax_qa_platform/data/metadata/schema_meta.json）。
# 为什么算成绝对路径：脚本可能从任意工作目录被调用，用 __file__ 反推到 data/ 才稳妥。
# 本文件位于 scripts/ 下，parents[1] 即 tax_qa_platform 项目根。
_DEFAULT_SCHEMA_JSON = Path(__file__).resolve().parents[1] / "data" / "metadata" / "schema_meta.json"

# 用一个固定命名空间给"表名"生成稳定的 UUID 主键。
# 为什么：Qdrant 的点 id 只接受 无符号整数 或 UUID，不能用 "table::xxx" 这类任意字符串；
# 用 uuid5(命名空间, 表名) 能由同一个表名稳定推出同一个 UUID，从而实现"按表幂等覆盖"。
_QDRANT_ID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def _table_point_id(table: str) -> str:
    """把表名映射成 Qdrant 可接受的稳定 UUID 字符串（同名表恒得同一 id，便于幂等覆盖）。

    :param table: 表名。
    :return: uuid5 生成的 UUID 字符串。
    """
    return str(uuid.uuid5(_QDRANT_ID_NAMESPACE, table))


def _column_point_id(table: str, column: str) -> str:
    """把"表.字段"映射成 Qdrant 可接受的稳定 UUID 字符串（同一表同一字段恒得同一 id，便于幂等覆盖）。

    为什么和 _table_point_id 分开：从 JSON 走"字段级"point，每条是一个 (table, column) 粒度，
    要用 (table, column) 联合键推 UUID，避免同一张表的多个字段相互覆盖。

    :param table: 表名。
    :param column: 字段名。
    :return: uuid5 生成的 UUID 字符串。
    """
    return str(uuid.uuid5(_QDRANT_ID_NAMESPACE, f"{table}::{column}"))


def build_column_description(table: str, table_comment: str, column: str,
                            column_comment: str, sample: str = "") -> str:
    """把一条"字段级"元数据拼成一段"语义描述"文本（供向量召回）。

    与 build_table_description 的区别：JSON 源是"一行一个字段"的扁平结构（来自掌柜问数式的
    schema 明细），所以这里以"字段"为最小召回单元。把 表注释 + 字段注释 + 样例值 都拼进去，
    口语化问法（如"各省销售额"）更容易命中到 dim_region.province 这一类具体字段。

    :param table: 表名。
    :param table_comment: 表中文注释（帮助把问法对到"哪张表"）。
    :param column: 字段名。
    :param column_comment: 字段中文注释（字段语义的核心，召回主要靠它）。
    :param sample: 样例值（可空，给一个具体取值帮助向量更"接地气"）。
    :return: 一段用于向量化的中文描述文本。
    """
    # 字段主体：字段名(注释)；没注释就只用字段名。
    field_desc = f"{column}（{column_comment}）" if column_comment else column
    parts = [f"数据表 {table}"]
    if table_comment:
        parts.append(f"（{table_comment}）")
    parts.append(f" 的字段 {field_desc}")
    if sample:
        parts.append(f"，样例值：{sample}")
    parts.append("。")
    return "".join(parts)


def collect_metadata_from_json(json_path: str | Path = _DEFAULT_SCHEMA_JSON) -> list[dict]:
    """从 schema_meta.json 读取"字段级"元数据，组装成与"从 MySQL 读"一致的下游结构。

    为什么需要它：兑现"无库也能建索引"——data/metadata/schema_meta.json 是离线静态元数据，
    没有 MySQL 也能据此向量化建索引。本函数把 JSON（数组，每条
    {table, table_comment, column, column_comment, sample}）转换成下游统一结构，
    后续走同一向量化 + upsert Qdrant 流程。

    与 tools.format_schema_context 的兼容：每条作为"字段级"point，payload 直接携带
    table / column / type / description(=column_comment) / table_comment —— 召回后该 payload
    会被原样塞进 schema_items，format_schema_context 即可渲染成"表.字段 (类型): 描述"。

    :param json_path: schema_meta.json 路径，默认 data/metadata/schema_meta.json。
    :return: 列表，每项形如
             {"id":..., "table":..., "description":..., "payload": {...}}（与 collect_metadata 同构）。
    :raise RuntimeError: 文件不存在/解析失败/格式非数组时抛出带中文上下文的错误。
    """
    path = Path(json_path)
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except FileNotFoundError as exc:
        logger.error("[元数据] 未找到元数据 JSON：%s", path, exc_info=True)
        raise RuntimeError(
            f"未找到元数据 JSON 文件：{path}。请确认 data/metadata/schema_meta.json 存在，或用 --from-json 指定路径。"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("[元数据] 解析元数据 JSON 失败：%s", exc, exc_info=True)
        raise RuntimeError(f"解析元数据 JSON 失败（{path}）：{exc}") from exc

    if not isinstance(data, list):
        raise RuntimeError(
            f"元数据 JSON 格式应为数组（每条 {{table, table_comment, column, column_comment, sample}}），实际为 {type(data).__name__}：{path}"
        )

    logger.info("[元数据] 从 JSON 读取到 %d 条字段级元数据：%s", len(data), path)
    items: list[dict] = []
    for row in data:
        if not isinstance(row, dict):
            logger.warning("[元数据] 跳过非对象条目：%r", row)
            continue
        table = (row.get("table") or "").strip()
        column = (row.get("column") or "").strip()
        if not table or not column:
            logger.warning("[元数据] 跳过缺 table/column 的条目：%r", row)
            continue
        table_comment = row.get("table_comment") or ""
        column_comment = row.get("column_comment") or ""
        sample = row.get("sample")
        sample_str = "" if sample is None else str(sample)
        description = build_column_description(table, table_comment, column, column_comment, sample_str)
        items.append({
            # Qdrant 主键：由 (table, column) 稳定推导的 UUID，同字段幂等覆盖
            "id": _column_point_id(table, column),
            "table": table,
            "description": description,
            # payload：召回命中后原样塞进 schema_items，供 format_schema_context 渲染。
            # type 留空（JSON 源不含类型，contract 允许可空/留空）；description 取字段注释，便于直接展示语义。
            "payload": {
                "table": table,
                "column": column,
                "type": "",
                "description": column_comment or description,
                "table_comment": table_comment,
                "sample": sample_str,
            },
        })
    logger.info("[元数据] JSON 共组装字段级条目 %d 条", len(items))
    return items


def build_table_description(table: str, columns: list[dict]) -> str:
    """把一张表的元数据拼成一段"语义描述"文本（供向量召回）。

    为什么要拼成自然语言：用户问"上个月华东区销售额"，不会出现表名 `dwd_sales_detail`；
    把"表注释 + 各字段中文注释"组织成可读描述，向量召回才更容易把口语问法和这张表对上号。

    :param table: 表名。
    :param columns: 字段元数据列表，每项形如 MySQLClient.table_schema 的返回：
                    {"column": 字段名, "type": 类型, "nullable": 是否可空, "comment": 中文注释}
                    （对其它键名如 name/Field 也做容错，便于复用）。
    :return: 一段用于向量化的中文描述文本。
    """
    # 字段行：优先用"字段名(注释)"，没有注释就只用字段名；附带类型帮助 LLM 判断可比较/可聚合。
    # 键名优先匹配 MySQLClient.table_schema 的 "column"，并兼容 name/Field 等其它来源。
    col_lines = []
    for col in columns:
        name = col.get("column") or col.get("name") or col.get("Field") or ""
        comment = col.get("comment") or col.get("Comment") or ""
        ctype = col.get("type") or col.get("Type") or ""
        if not name:
            continue
        desc = f"{name}（{comment}）" if comment else name
        col_lines.append(f"{desc}[{ctype}]" if ctype else desc)
    cols_text = "、".join(col_lines) if col_lines else "（无字段信息）"
    # 表描述模板：表名 + 字段清单。简单直接，便于学习者理解 schema linking 的"被检索对象"长什么样。
    return f"数据表 {table}，包含字段：{cols_text}。"


async def collect_metadata(mysql_client) -> list[dict]:
    """从 MySQL 读取所有表及其字段，组装成待入库的元数据条目列表。

    :param mysql_client: 已注入的 MySQLClient（异步）。
    :return: 列表，每项形如
             {"id":..., "table":..., "description":..., "payload": {...}}。
    :raise RuntimeError: 读取元数据失败时抛出带中文上下文的错误。
    """
    try:
        tables = await mysql_client.list_tables()  # 约定返回 list[str]
    except Exception as exc:  # noqa: BLE001
        logger.error("[元数据] 列出数据表失败：%s", exc, exc_info=True)
        raise RuntimeError(
            f"读取 MySQL 表清单失败。请检查 settings.mysql_* 配置与连通性（db={settings.mysql_db}）。原始错误：{exc}"
        ) from exc

    logger.info("[元数据] 共发现数据表 %d 张", len(tables))
    items: list[dict] = []
    for table in tables:
        try:
            columns = await mysql_client.table_schema(table)  # 约定返回 list[dict]
        except Exception as exc:  # noqa: BLE001
            logger.error("[元数据] 读取表结构失败 table=%s：%s", table, exc, exc_info=True)
            continue  # 单表失败不影响整体，跳过继续
        description = build_table_description(table, columns or [])
        items.append({
            # Qdrant 主键：由表名稳定推导的 UUID（Qdrant 不接受任意字符串 id），同名表幂等覆盖
            "id": _table_point_id(table),
            "table": table,
            "description": description,
            # payload：召回命中后用于还原原始结构，交给 LLM 生成 SQL；额外存 table 便于反查
            "payload": {"table": table, "columns": columns or [], "description": description},
        })
        logger.info("[元数据] 已组装 table=%s 字段数=%d", table, len(columns or []))
    return items


async def build_index(dry_run: bool = False, from_json: str | Path | None = None,
                     mysql_client=None, emb_client=None, qdrant_client=None) -> int:
    """主流程：读元数据 -> 拼描述 ->（向量化 -> 写 Qdrant）。

    :param dry_run: 干跑时只读元数据并打印描述，不向量化、不写库。
    :param from_json: 指定时从该 JSON 路径读"字段级"元数据（无需 MySQL）；None 时维持原 MySQL 实时读行为。
    :param mysql_client/emb_client/qdrant_client: 可注入客户端（便于测试/复用），None 时惰性新建。
    :return: 入库（或干跑时组装）的元数据条目数。
    :raise RuntimeError: 向量化或写 Qdrant 失败时抛出。
    """
    logger.info("[元数据] 开始构建数仓元数据索引 dry_run=%s from_json=%s", dry_run, from_json)

    # 元数据来源二选一：
    #   - from_json 非空：从静态 JSON 读字段级元数据（兑现"无库也能建索引"，不连 MySQL）。
    #   - 否则：维持原行为，从 MySQL 实时读表/字段结构。
    # 两条路最终都产出同构的 items（{id, table, description, payload}），后续向量化+写库流程完全复用。
    if from_json is not None:
        items = collect_metadata_from_json(from_json)
        if not items:
            logger.warning("[元数据] JSON 未读到任何字段元数据，结束（请检查 %s 内容）", from_json)
            return 0
    else:
        mysql_client = mysql_client or _lazy_mysql()
        items = await collect_metadata(mysql_client)
        if not items:
            logger.warning("[元数据] 未读到任何表元数据，结束（请确认数仓 %s 中有表）", settings.mysql_db)
            return 0

    if dry_run:
        for it in items[:5]:  # 干跑只预览前 5 条拼好的描述（用表名展示，比 UUID 直观）
            logger.info("[元数据][dry-run] table=%s => %s", it["table"], it["description"])
        logger.info("[元数据][dry-run] 共组装 %d 条（未向量化、未写 Qdrant）", len(items))
        return len(items)

    # ---- 向量化 ----
    emb_client = emb_client or _lazy_embedding()
    descriptions = [it["description"] for it in items]
    try:
        emb = emb_client.embed(descriptions)        # {"dense": [...], "sparse": [...]}
        dense_list = emb.get("dense", [])
    except Exception as exc:  # noqa: BLE001
        logger.error("[元数据] 向量化失败：%s", exc, exc_info=True)
        raise RuntimeError(f"元数据向量化失败：{exc}") from exc

    # ---- 写 Qdrant ----
    qdrant_client = qdrant_client or _lazy_qdrant()
    points: list[dict] = []
    for i, it in enumerate(items):
        vec = dense_list[i] if i < len(dense_list) else []
        points.append({"id": it["id"], "vector": vec, "payload": it["payload"]})
    try:
        qdrant_client.upsert(collection=settings.qdrant_meta_collection, points=points)
    except Exception as exc:  # noqa: BLE001
        logger.error("[元数据] 写 Qdrant 失败：%s", exc, exc_info=True)
        raise RuntimeError(
            f"写入 Qdrant 失败（collection={settings.qdrant_meta_collection}）：{exc}"
        ) from exc

    logger.info("[元数据] 完成 写入元数据条目=%d -> Qdrant.%s", len(points), settings.qdrant_meta_collection)
    return len(points)


def _lazy_mysql():
    from app.clients.mysql_client import MySQLClient
    return MySQLClient()


def _lazy_embedding():
    from app.clients.embedding_client import EmbeddingClient
    return EmbeddingClient()


def _lazy_qdrant():
    from app.clients.qdrant_meta_client import QdrantMetaClient
    return QdrantMetaClient()


def _build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。

    :return: 配置好的 ArgumentParser。
    """
    parser = argparse.ArgumentParser(
        description="读取 MySQL 数仓表/字段元数据 -> 向量化 -> 写 Qdrant（供 Text2SQL schema linking）。"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="干跑：只读元数据并打印拼好的描述，不向量化、不写 Qdrant。")
    # --from-json：从静态 JSON 读字段级元数据（无需 MySQL，兑现"无库也能建索引"）。
    #   - 不带该参数：维持原 MySQL 实时读行为不变。
    #   - 带参数不带值：用默认路径 data/metadata/schema_meta.json。
    #   - 带参数带值：用指定路径。
    # nargs="?" + const 实现"可选值"语义；default=None 用于区分"未指定该参数"。
    parser.add_argument(
        "--from-json", nargs="?", const=str(_DEFAULT_SCHEMA_JSON), default=None,
        metavar="PATH",
        help=f"从 JSON 读字段级元数据建索引（无需 MySQL）。不带值时默认用 {_DEFAULT_SCHEMA_JSON}。",
    )
    return parser


def main() -> None:
    """脚本入口：解析参数 -> 初始化日志 -> asyncio 驱动异步主流程。"""
    parser = _build_arg_parser()
    args = parser.parse_args()

    setup_logging()
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    # MySQLClient 为异步客户端，用 asyncio.run 驱动整条异步链路。
    # from_json 走 JSON 静态读时不连 MySQL；为 None 时维持原 MySQL 实时读行为。
    asyncio.run(build_index(dry_run=args.dry_run, from_json=args.from_json))


if __name__ == "__main__":
    # 学习提示：
    #   py scripts/build_metadata_index.py --dry-run                 # 从 MySQL 读，看每张表拼成什么描述
    #   py scripts/build_metadata_index.py                           # 从 MySQL 读，正式写入 Qdrant
    #   py scripts/build_metadata_index.py --from-json --dry-run     # 无库：从默认 JSON 读字段级元数据预览
    #   py scripts/build_metadata_index.py --from-json               # 无库：从默认 JSON 建索引写入 Qdrant
    #   py scripts/build_metadata_index.py --from-json 路径.json     # 无库：指定 JSON 路径
    # 从 MySQL 读需先在 .env 配好 MySQL 数仓与 Qdrant；数仓表最好带中文注释，召回质量更高。
    # --from-json 兑现"无库也能建索引"：仅需 data/metadata/schema_meta.json + Qdrant（非 dry-run 时）。
    main()
