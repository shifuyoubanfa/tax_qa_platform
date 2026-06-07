"""摘要/上下文拼装层（Summarizer）。
本模块在整体链路里的位置：处于"重排(ReRanker) -> 摘要 -> LLM 生成答案"之间，是 RAG 链路的
"最后一公里"。它做两件事：
1. build_context：把重排后的 Document 列表，按【意图】拼成喂给 LLM 的 context 文本，并同时
   产出给前端展示的 references（带 [[citation:N]] 编号，便于答案里引用、前端高亮）。
2. summarize_stream：用 get_llm + load_prompt('summarize') 的 .astream 逐 token 流式产出答案，
   交给编排层包装成 SSE 的 answer_delta 事件。

为什么"按意图拼上下文"（对标爱搜税 summarize.get_summary_reference）：
不同意图关心的资料结构不同——
- 稽查案例类：要先放"直指政策"，再补"案例库"片段，最后才用通用政策兜底；
- 通用问题类(无地域)：要把"地方法规"过滤掉，避免地方口径污染全国性回答；
- 其它意图：按综合得分顺序直接取 topk。
把这套"取材策略"集中在这里，链路其它层就不必关心意图差异。

风格对标 爱搜税 summarize.py（意图化 reference 选取 + [[citation]] 上下文拼装）、
掌柜智库 lm_utils（LLM 客户端用法）。
"""
from __future__ import annotations

from typing import AsyncIterator

from config.constants import Intent, KBase
from config.logging_config import get_logger
from app.schemas.document import Document, QUResult
from app.utils.prompt_loader import load_prompt
from app.utils.timing import runtime

logger = get_logger(__name__)

# 单条文档片段截断长度：第一条给足正文，其余给摘要级片段，控制总 token。
_FIRST_DOC_MAX_CHARS = 500
_OTHER_DOC_MAX_CHARS = 300
# 稽查案例类：案例片段最多并入几条（对标爱搜税 max_inspect_number）。
_MAX_INSPECT_NUMBER = 3
# 短查询"逐子查询取材"时，每个子查询最多并入几条（对标爱搜税 max_each_query_num）。
_MAX_PER_SUBQUERY = 3
# 联网补充区块的显式标注（物理隔离的核心文案）：联网结果单独成块、绝不混入权威引用池，
# 并明确告知 LLM/前端"未核验·以官方为准"，避免被当成权威法规来引用。
_WEB_BLOCK_HEADER = "【联网补充（未核验·以官方为准，请勿作为权威法规引用）】"
# 联网补充单条片段截断长度（与正文片段同量级，控制总 token）。
_WEB_DOC_MAX_CHARS = 300
# 摘要 prompt 缺失时的兜底模板（保证无 prompt 文件也能跑通自测）。
_FALLBACK_SUMMARIZE_PROMPT = (
    "你是专业的税务问答助手。请仅依据【参考资料】回答用户问题，"
    "并用 [[citation:序号]] 标注引用来源；资料不足时如实说明。\n\n"
    "【用户问题】：{query}\n\n【参考资料】：\n{context}\n\n【回答】："
)


class Summarizer:
    """意图化上下文拼装 + 流式摘要生成器。

    用法::

        summarizer = Summarizer()
        context, references = summarizer.build_context(qu, docs, topk=5)
        async for delta in summarizer.summarize_stream(query, qu, context):
            ...  # 逐 token 输出
    """

    def __init__(self) -> None:
        # LLM 客户端惰性获取（首次 summarize_stream 才创建），import 阶段不连服务
        self._llm = None
        # MinIO 客户端惰性单例（首次 build_context 需要生成图片URL时才创建）：
        # 持有一个实例供本 Summarizer 复用，避免对每条 reference / 每个 image_key 都新建客户端。
        self._minio = None

    def _get_minio(self):
        """惰性获取 MinIO 客户端单例（首次需要生成图片URL时才创建）。

        为何惰性 + 单例：import 阶段不连服务；同一次 build_context 可能有多条 reference、
        每条又有多个 image_key，复用同一个客户端避免反复建连。MinioClient() 构造本身不建连，
        真正连接发生在首次 presigned_url 内部，且其已自带 try/except 降级返回空串。

        :return: MinioClient 实例。
        """
        if self._minio is None:
            from app.clients.minio_client import MinioClient

            self._minio = MinioClient()
            logger.info("[摘要] MinIO 客户端惰性初始化完成（用于生成图片预签名URL）")
        return self._minio

    # ------------------------------------------------------------------ #
    # 上下文拼装
    # ------------------------------------------------------------------ #
    @runtime
    def build_context(
        self, qu: QUResult, docs: list[Document], topk: int
    ) -> tuple[str, list[dict]]:
        """按意图把文档拼成 LLM 上下文，并产出前端引用列表。

        :param qu: Query Understanding 结果（提供 intent / entities，决定取材策略）。
        :param docs: 重排后的候选文档（通常已按综合得分降序）。
        :param topk: 最终参与摘要的文档数量上限。
        :return: (context, references) —— context 为带 [[citation:N]] 的拼接文本；
                 references 为 list[dict]，每条在 Document.to_reference 基础上补 citation_index 与片段 context。
        :raise: 不主动抛异常；docs 为空时返回 ("", [])，由上层决定"无资料"话术。
        """
        logger.info("[摘要] 开始构建上下文 intent=%s 候选文档数=%d topk=%d",
                    qu.intent, len(docs or []), topk)

        # 0. 物理隔离联网补充来源：把 kbase=web（联网·未核验）的文档先剥离出来，
        #    绝不参与下面的权威取材与 [[citation:N]] 编号；它们只进单独的"联网补充"区块/字段。
        #    这样权威法规引用的编号与取材逻辑【完全不变】（与未引入联网时一致）。
        docs = docs or []
        web_docs = [d for d in docs if d.kbase == KBase.WEB.value]
        docs = [d for d in docs if d.kbase != KBase.WEB.value]
        if web_docs:
            logger.info("[摘要] 剥离联网补充来源 %d 条（物理隔离，不进权威引用池）", len(web_docs))

        # 1. 挑选要进上下文的文档（对标爱搜税 get_context 的分流）：
        #    短查询 -> "逐子查询取材"：短问题被扩写成多个子查询，每个子查询各取几条，保证每个
        #              扩写角度都被覆盖，而不是被某一个子查询的强命中刷屏；
        #    长查询 -> "按意图取材"(_select_by_intent)：稽查案例分层、通用问答过滤地方法规等。
        if qu.is_short_query:
            selected = self._select_for_each_subquery(qu, docs, topk)
        else:
            selected = self._select_by_intent(qu, docs, topk)
        if not selected and not web_docs:
            # 既无权威文档、也无联网补充：返回空上下文，由上层给"无资料"话术。
            logger.info("[摘要] 无可用文档，返回空上下文")
            return "", []

        # 2. 逐条拼接：标题 + 片段，并打 [[citation:N]] 编号
        context_parts: list[str] = []
        references: list[dict] = []
        for idx, doc in enumerate(selected, start=1):
            # 第一条给较长正文（直接命中概率高），其余给较短片段，控制总长度
            limit = _FIRST_DOC_MAX_CHARS if idx == 1 else _OTHER_DOC_MAX_CHARS
            snippet = (doc.content or "").strip()[:limit]
            title = (doc.title or "").strip()

            part = f"[[citation:{idx}]]\n【标题】：{title}\n【片段】：{snippet}"
            context_parts.append(part)

            # 引用条目：复用 Document.to_reference，再补 citation 编号与实际入参片段
            ref = doc.to_reference()
            ref["citation_index"] = idx
            ref["context"] = snippet

            # 图片URL：ref 里的 image_keys 只是对象键，前端无法直接加载。这里对每个 key 调
            # MinioClient.presigned_url 生成临时可访问URL，写入 ref["image_urls"]（保留原 image_keys）。
            # presigned_url 内部已 try/except、失败返回空串；这里过滤掉空串，MinIO 不可用时
            # image_urls 为空列表，不影响主链路（降级）。
            image_keys = ref.get("image_keys") or []
            image_urls: list[str] = []
            if image_keys:
                minio = self._get_minio()
                for key in image_keys:
                    url = minio.presigned_url(key)
                    if url:  # 过滤空串（presigned_url 失败时返回 ""）
                        image_urls.append(url)
            ref["image_urls"] = image_urls
            references.append(ref)

        context = "\n\n".join(context_parts)

        # 3. 联网补充（物理隔离）：把剥离出来的 kbase=web 结果拼成【独立区块】追加到 context 末尾，
        #    并作为【独立 references 字段】返回——绝不分配 [[citation:N]] 编号、绝不混入权威引用池。
        #    上面 selected/references/编号逻辑完全不受影响：本地权威引用编号、取材逻辑保持原样。
        if web_docs:
            web_block, web_refs = self._build_web_supplement(web_docs)
            # 区块拼接：权威上下文在前、联网补充在后，并以显式标题分隔（即便权威为空也带标题）。
            context = f"{context}\n\n{web_block}" if context else web_block
            references.extend(web_refs)
            logger.info("[摘要] 已追加联网补充区块：补充条数=%d（未核验·物理隔离）", len(web_refs))

        logger.info("[摘要] 上下文构建完成：权威入选=%d，联网补充=%d，上下文长度=%d 字符",
                    len(selected), len(web_docs), len(context))
        return context, references

    def _build_web_supplement(
        self, web_docs: list[Document]
    ) -> tuple[str, list[dict]]:
        """把联网（未核验）文档拼成【物理隔离】的补充区块与补充引用条目。

        与权威引用的关键区别（硬约束）：
        - 区块用 _WEB_BLOCK_HEADER 显式标注"未核验·以官方为准·勿作权威引用"；
        - 区块内【不打 [[citation:N]] 编号】，只用项目符号列出，避免被 LLM 当权威来源引用；
        - 每条 reference 标 source="web" / unverified=True / 不含 citation_index，
          与权威 references（带 citation_index）从字段层面区分，前端可单独成"联网补充"列表。

        :param web_docs: 已从主候选里剥离的 kbase=web 文档列表。
        :return: (web_block, web_refs) —— web_block 为带标题的补充文本；web_refs 为独立引用条目。
        """
        lines: list[str] = [_WEB_BLOCK_HEADER]
        web_refs: list[dict] = []
        for doc in web_docs:
            title = (doc.title or "").strip()
            snippet = (doc.content or "").strip()[:_WEB_DOC_MAX_CHARS]
            url = (doc.metadata or {}).get("url", "")
            # 不打 citation 编号，用项目符号；附 URL 便于核验"以官方为准"。
            line = f"- 【{title}】{snippet}"
            if url:
                line += f"（来源：{url}）"
            lines.append(line)

            # 独立引用条目：复用 to_reference 拿基础字段，再补"未核验"标记，但【不加 citation_index】。
            ref = doc.to_reference()
            ref["context"] = snippet
            ref["unverified"] = True              # 未核验标记（前端据此显示"未核验"角标）
            ref["source"] = KBase.WEB.value       # 与权威 references 区分的来源标识
            ref["image_urls"] = []                # 联网补充不处理图片，保持字段结构一致
            web_refs.append(ref)

        return "\n".join(lines), web_refs

    def _select_by_intent(
        self, qu: QUResult, docs: list[Document], topk: int
    ) -> list[Document]:
        """按意图选取参与摘要的文档（对标爱搜税 get_summary_reference 的路由逻辑）。

        :param qu: QU 结果（读 intent 与 entities.region）。
        :param docs: 全部候选文档。
        :param topk: 数量上限。
        :return: 选中的文档列表（长度 <= topk）。
        """
        intent = qu.intent

        # 稽查案例类：优先"直指政策(policy)"，再补"案例库(inspect_case)"，最后通用兜底。
        if intent == Intent.INSPECT_CASE.value:
            policy_docs = [d for d in docs if d.kbase == "policy"]
            inspect_docs = [d for d in docs if d.kbase == "inspect_case"]
            others = [d for d in docs if d.kbase not in ("policy", "inspect_case")]
            merged = policy_docs + inspect_docs[:_MAX_INSPECT_NUMBER]
            # 不足 topk 时用其它库（如通用政策/文档）补齐
            if len(merged) < topk:
                merged += others[: topk - len(merged)]
            logger.info("[摘要] 稽查案例类取材：政策=%d 案例=%d 兜底=%d",
                        len(policy_docs), len(inspect_docs), len(others))
            return merged[:topk]

        # 通用问题类且未指定地域：过滤掉"地方法规"，避免地方口径污染全国性回答。
        if intent == Intent.GENERAL_QA.value and not qu.entities.region:
            filtered = [
                d for d in docs
                if d.metadata.get("policy_type") != "地方法规"
            ]
            logger.info("[摘要] 通用问题类(无地域)过滤地方法规：%d -> %d",
                        len(docs), len(filtered))
            return filtered[:topk]

        # 查社保类 / 查商品编码类：结构化 Agent 查表的精确结果(kbase=social_security/product_code)
        # 优先进上下文，文档召回兜底其后（对齐爱搜税：直指/agent 结果排在文档召回之前）。
        if intent in (Intent.SOCIAL_SECURITY.value, Intent.PRODUCT_CODE.value):
            structured_kbases = {KBase.SOCIAL_SECURITY.value, KBase.PRODUCT_CODE.value}
            structured = [d for d in docs if d.kbase in structured_kbases]
            others = [d for d in docs if d.kbase not in structured_kbases]
            logger.info("[摘要] %s 取材：结构化Agent结果=%d 文档兜底=%d",
                        intent, len(structured), len(others))
            return (structured + others)[:topk]

        # 其它意图：按既有顺序（综合得分）直接取 topk。
        return docs[:topk]

    def _select_for_each_subquery(
        self, qu: QUResult, docs: list[Document], topk: int
    ) -> list[Document]:
        """短查询取材：按"每个子查询各取若干条"组装（对标爱搜税 get_summary_reference_for_each_query）。

        动机：短查询(is_short_query)信息稀薄、被扩写成多个子查询。若仍按全局综合得分取 topk，
        很可能被"某一个子查询的强命中"占满，丢掉其它扩写角度。这里按"召回来源子查询(raw_query_from)"
        分桶，各桶轮流取、单桶最多 _MAX_PER_SUBQUERY 条，保证每个扩写角度都进上下文。

        :param qu: QU 结果（此处主要用 docs 上的 raw_query_from；qu 保留以便扩展）。
        :param docs: 重排后的候选（已按综合得分降序）。
        :param topk: 数量上限。
        :return: 选中的文档列表（长度 <= topk）。
        """
        # 1) 按"由哪个子查询召回"分桶，桶内保持原(得分)顺序
        buckets: dict[str, list[Document]] = {}
        for d in docs:
            buckets.setdefault(d.raw_query_from or "_", []).append(d)
        order = list(buckets.keys())
        taken = {k: 0 for k in order}

        # 2) 轮转：各子查询桶轮流取一条，单桶最多 _MAX_PER_SUBQUERY 条，直到凑满 topk
        selected: list[Document] = []
        progressed = True
        while len(selected) < topk and progressed:
            progressed = False
            for k in order:
                if len(selected) >= topk:
                    break
                if taken[k] < _MAX_PER_SUBQUERY and taken[k] < len(buckets[k]):
                    selected.append(buckets[k][taken[k]])
                    taken[k] += 1
                    progressed = True

        # 3) 若子查询太少/候选不够而没凑满，按原综合得分顺序补齐
        if len(selected) < topk:
            chosen = {d.doc_id for d in selected}
            for d in docs:
                if len(selected) >= topk:
                    break
                if d.doc_id not in chosen:
                    selected.append(d)
                    chosen.add(d.doc_id)

        logger.info("[摘要] 短查询逐子查询取材：子查询桶=%d，入选=%d", len(buckets), len(selected))
        return selected[:topk]

    # ------------------------------------------------------------------ #
    # 流式摘要
    # ------------------------------------------------------------------ #
    def _get_llm(self):
        """惰性获取 LLM 客户端（首次调用才创建）。

        :return: langchain_openai.ChatOpenAI 实例。
        :raise Exception: LLM 客户端配置缺失/构造失败时由 get_llm 抛出。
        """
        if self._llm is None:
            from app.clients.llm_client import get_llm

            self._llm = get_llm()
            logger.info("[摘要] LLM 客户端惰性初始化完成")
        return self._llm

    def _render_prompt(self, query: str, context: str) -> str:
        """渲染摘要 prompt：优先用 config/prompts/summarize.prompt，缺失时用兜底模板。

        :param query: 用户原始问题。
        :param context: build_context 产出的带引用上下文。
        :return: 渲染后的完整 prompt 文本。
        """
        try:
            template = load_prompt("summarize")
        except FileNotFoundError:
            # 模板文件缺失时不崩，用内置兜底模板（仅自测/早期开发会走到）
            logger.error("[摘要] 未找到 summarize.prompt，使用内置兜底模板")
            template = _FALLBACK_SUMMARIZE_PROMPT
        # 模板里约定用 {query} 与 {context} 两个占位符
        return template.format(query=query, context=context)

    async def summarize_stream(
        self, query: str, qu: QUResult, context: str
    ) -> AsyncIterator[str]:
        """基于上下文流式生成答案，逐 token 产出（供编排层包成 SSE answer_delta）。

        :param query: 用户原始问题。
        :param qu: QU 结果（保留入参以便将来按意图切换 prompt，本实现暂只用于日志）。
        :param context: build_context 产出的上下文；为空时直接产出"无资料"话术并结束。
        :return: 异步迭代器，逐段 yield 答案增量字符串。
        :raise: 内部捕获 LLM 异常并 yield 一句中文报错，不向外抛，避免中断 SSE 流。
        """
        logger.info("[摘要] 进入流式摘要 intent=%s context_len=%d", qu.intent, len(context or ""))

        # 无上下文：不调 LLM，直接给出诚实话术（省 token、避免幻觉）
        if not context:
            logger.info("[摘要] 上下文为空，返回兜底话术，不调用 LLM")
            yield "抱歉，未检索到与您问题直接相关的资料，暂时无法给出准确回答。建议补充关键词（如具体文号、政策名称）后再试。"
            return

        prompt = self._render_prompt(query, context)
        try:
            llm = self._get_llm()
            # langchain_openai.ChatOpenAI 支持 .astream，逐 chunk 返回 AIMessageChunk
            async for chunk in llm.astream(prompt):
                # chunk.content 为本次增量文本；空增量跳过
                delta = getattr(chunk, "content", "") or ""
                if delta:
                    yield delta
            logger.info("[摘要] 流式摘要生成完成")
        except Exception as e:  # noqa: BLE001 不让 LLM 异常中断 SSE 流
            logger.error("[摘要] 流式摘要生成失败：%s", e, exc_info=True)
            yield "（生成回答时发生异常，请稍后重试）"


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：只验证 build_context 的意图分流，不调 LLM。
    docs_demo = [
        Document(doc_id="1", title="增值税暂行条例", content="一般纳税人...", kbase="policy",
                 metadata={"policy_type": "中央法规"}, score=0.9),
        Document(doc_id="2", title="某地补充规定", content="本省执行口径...", kbase="policy",
                 metadata={"policy_type": "地方法规"}, score=0.8),
    ]
    qu_demo = QUResult(raw_query="增值税怎么算", intent=Intent.GENERAL_QA.value)
    ctx, refs = Summarizer().build_context(qu_demo, docs_demo, topk=5)
    print("[summarizer 自测] 长查询(按意图)引用数(应过滤地方法规=1) =>", len(refs))
    print("[summarizer 自测] 上下文预览 =>", ctx[:80])

    # 短查询：逐子查询取材——两个子查询桶各取若干条，保证两个角度都被覆盖
    short_docs = [
        Document(doc_id=str(i), title=f"片段{i}", content=f"正文{i}", kbase="policy",
                 raw_query_from=("社保基数" if i < 4 else "社保缴费基数标准"),
                 score=1.0 - i * 0.1)
        for i in range(6)
    ]
    qu_short = QUResult(raw_query="社保基数", intent=Intent.SOCIAL_SECURITY.value,
                        sub_queries=["社保基数", "社保缴费基数标准"], is_short_query=True)
    _, refs2 = Summarizer().build_context(qu_short, short_docs, topk=5)
    print("[summarizer 自测] 短查询(逐子查询)入选(应跨两个子查询桶) =>",
          [(r["doc_id"], short_docs[int(r["doc_id"])].raw_query_from) for r in refs2])
