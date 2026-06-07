"""查询改写器（QueryRewriter）：子查询扩写 + HyDE 假设性文档。

本模块在整体链路里的位置：QU 的第三步。它的两个产物都服务于"提升召回率"：
    1. sub_queries（子查询扩写）—— 用户原话往往口语化、信息稀疏；扩写成多个语义更完整的检索式，
       多路召回再融合（RRF），能显著提升召回覆盖面。对标爱搜税 query_rewrite_with_llm。
    2. hyde_doc（HyDE，Hypothetical Document Embeddings）—— 先让 LLM 写一段"假设性答案范文"，
       再用这段范文去做稠密向量召回。原理：答案文本与库中政策原文在向量空间里更接近，
       比用"问句"直接检索更准。对标掌柜智库 hyde_prompt。

设计要点（为什么这么做）：
    - sub_queries 永远包含【原始 query】（保证不丢原意），LLM 扩写只是"附加"。
    - 短查询（len<8）信息量太少，最值得扩写；这里也据此打 is_short_query 标记给摘要层用。
    - 所有 LLM 调用 try/except 包裹：失败时退化为"只有原始 query、hyde 为空"，链路照常走。
"""
from __future__ import annotations

from config.logging_config import get_logger
from app.utils.prompt_loader import load_prompt

logger = get_logger(__name__)

# 短查询阈值：长度 < 该值视为"短查询"（与爱搜税 is_short_query 的 <8 一致）。
SHORT_QUERY_THRESHOLD = 8
# LLM 扩写子查询的最大保留条数，防止异常返回撑爆下游召回路数。
_MAX_SUB_QUERIES = 5


class QueryRewriter:
    """对 query 做扩写与 HyDE 生成。

    用法::

        rw = QueryRewriter()
        sub_queries, hyde_doc = rw.rewrite("研发费用加计扣除", intent="通用问题类")

    rewrite 内部只读取无状态的 prompt 模板与 LLM 客户端（均带缓存），可并发调用。
    """

    def rewrite(self, query: str, intent: str) -> tuple[list[str], str]:
        """生成子查询列表与 HyDE 文档。

        :param query: 用户原始问题。
        :param intent: 已分类出的意图值（透传给 prompt，便于按意图微调扩写风格）。
        :return: (sub_queries, hyde_doc)
                 - sub_queries: 始终以"原始 query"开头，后接 LLM 扩写（去重、限量）；
                   query 为空时返回 [""] 之外的安全空结构（见下）。
                 - hyde_doc: HyDE 假设性文档字符串，失败/空 query 时为 ""。
        :raise: 不向外抛异常；任一步失败都降级，保证至少返回原始 query。
        """
        logger.info("查询改写开始，query=%s，intent=%s", query, intent)
        # 入参兜底：空 query 不调 LLM，直接返回空结构。
        if not query or not query.strip():
            logger.info("查询改写：query 为空，返回空结构")
            return [], ""

        raw = query.strip()
        # sub_queries 必含原始 query（契约要求），后续扩写在此基础上"加"。
        sub_queries: list[str] = [raw]

        # 1) LLM 扩写子查询
        expanded = self._expand_sub_queries(raw, intent)
        for q in expanded:
            # 去重：避免扩写出与原始/彼此重复的查询，导致多路召回浪费。
            if q and q not in sub_queries:
                sub_queries.append(q)
        # 限量：最多保留 _MAX_SUB_QUERIES 条（含原始 query）。
        sub_queries = sub_queries[:_MAX_SUB_QUERIES]

        # 2) HyDE 假设性文档
        hyde_doc = self._generate_hyde(raw)

        logger.info("查询改写结束：子查询数=%d，hyde长度=%d", len(sub_queries), len(hyde_doc))
        return sub_queries, hyde_doc

    @staticmethod
    def is_short_query(query: str) -> bool:
        """判断是否为短查询（信息稀疏，需重点扩写/走短查询摘要模板）。

        :param query: 用户原始问题。
        :return: 长度 < SHORT_QUERY_THRESHOLD 为 True。
        """
        return bool(query) and len(query.strip()) < SHORT_QUERY_THRESHOLD

    # ------------------------------------------------------------------ #
    # 内部方法
    # ------------------------------------------------------------------ #
    def _expand_sub_queries(self, query: str, intent: str) -> list[str]:
        """调 LLM 用 query_rewrite.prompt 生成扩写子查询（按行切分）。

        失败安全：无 LLM/网络/模板异常时返回 []（上层只用原始 query 召回）。
        """
        try:
            # 延迟导入：无 LLM 环境时不影响本模块 import。
            from app.clients.llm_client import get_llm

            template = load_prompt("query_rewrite")  # 带缓存读取模板
            prompt = template.format(query=query, intent=intent)

            llm = get_llm()  # 扩写无需 json_mode，普通文本即可
            resp = llm.invoke([{"role": "user", "content": prompt}])
            text = getattr(resp, "content", "") or ""

            # 约定：模板要求每行一个扩写问题，用换行分隔（与爱搜税 query_rewrite_with_llm 一致）。
            lines = [ln.strip(" -·\t") for ln in text.splitlines()]
            return [ln for ln in lines if ln]
        except Exception as e:
            logger.error("子查询扩写异常，降级为仅原始query：%s", e, exc_info=True)
            return []

    def _generate_hyde(self, query: str) -> str:
        """调 LLM 用 hyde.prompt 生成假设性答案文档（用于稠密召回）。

        失败安全：异常时返回 ""（稠密召回退回用原始 query 的向量）。
        """
        try:
            from app.clients.llm_client import get_llm

            template = load_prompt("hyde")
            prompt = template.format(query=query)

            llm = get_llm()
            resp = llm.invoke([{"role": "user", "content": prompt}])
            return (getattr(resp, "content", "") or "").strip()
        except Exception as e:
            logger.error("HyDE 文档生成异常，降级为空：%s", e, exc_info=True)
            return ""


if __name__ == "__main__":
    # 最小自测块：无 LLM 服务时，应优雅降级为"仅原始query + 空 hyde"。
    rw = QueryRewriter()
    subs, hyde = rw.rewrite("研发费用加计扣除", intent="通用问题类")
    print("子查询：", subs)
    print("HyDE：", hyde[:80])
    print("是否短查询：", QueryRewriter.is_short_query("社保基数"))
