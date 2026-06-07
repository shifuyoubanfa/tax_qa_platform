"""Agent 门面层的"能力接口"定义（P1 门面 · 纯类型注解 + 教学注释，零运行逻辑）。

本模块在整体链路里的位置：它【不参与任何执行】，只是给已经存在的三类子能力起一个统一的
"对外名字 + 接口形状"，方便阅读、类型检查与文档对照。它【不抽取、不改写、不统一】任何现有逻辑。

为什么"不强行统一成一个 BaseAgent"（本模块最重要的设计立场）：
    现实里这三类子能力的接口本来就不一样，硬塞进同一个抽象基类只会逼出"假统一"——要么参数名/返回
    类型对不上要被迫加无意义的形参，要么用 Any 把类型信息抹平、反而看不清各自真实契约。所以这里
    用 typing.Protocol（结构化子类型 / 鸭子类型）【如实记录三种不同的接口形状】，谁长得像谁就是谁，
    不要求任何现有类显式继承——现有代码一行都不用改，也不引入运行期耦合。

    具体而言，三类子能力的接口"天生不同"：

    1) 替换式子图 Agent（ReplacingSubgraphAgent）—— 如 Text2SQLAgent：
       当顶层确定性路由判定为"经营数据查询类(DATA_QUERY)"时，它【替换】整条 RAG pipeline，
       自己用一张内部 LangGraph 子图把问题翻成 SQL 查数仓。
       接口：``async arun(query: str, qu: QUResult) -> Text2SQLResult``
       —— 入参是"原始问题 + QU 结果"，出参是结构化的 Text2SQLResult（含 sql/rows/summary/error）。

    2) 召回源 Agent（RecallSourceAgent）—— 如 ShebaoAgent / ProductCodeAgent：
       它【不是】与 QA 平级的"子智能体"，而是召回层的一个【结构化数据源】：按 QU 抽取的实体查
       MySQL 表，把命中行包成 Document，并入文档召回池，一起进入 粗排/精排/重排/摘要。
       接口：``async search(qu: QUResult, topk: int = ...) -> list[Document]``
       —— 入参是 QU 结果，出参是与文档召回同构的 Document 列表（失败/无命中返回 []，由文档 RAG 兜底）。

    3) 主图本体问答（QaAgentLike）—— 即 QA：
       它【就是主图本身】(load_memory→understand→route→rag_recall→...→generate_answer→save_memory)。
       它不是"一个被调用的子图"，而是整条链路的本体；QaAgent 只是给它一个对外的 agent 名字（薄门面）。
       接口：``astream(req) -> AsyncIterator[SSEMessage]``
       —— 出参是真·逐 token 的 SSE 事件流（intent/retrieval/references/sql/answer_delta/done/error）。

    => 这三者：返回类型不同（Text2SQLResult vs list[Document] vs 流）、调用语义不同（替换 vs 并入召回
       vs 主图本体）、生命周期不同。把它们画成"平级三兄弟"是失真的；本模块用三个独立 Protocol
       分别刻画，避免任何"假统一"。

健壮性：本模块只含类型声明与常量字符串，import 无任何副作用、不连接任何外部服务、不依赖 langgraph。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator, Protocol, runtime_checkable

from config.logging_config import get_logger

# 仅类型检查期导入，运行期不触发——保持本模块 import 零副作用、零外部依赖。
if TYPE_CHECKING:  # pragma: no cover - 仅供静态类型检查
    from app.schemas.document import Document, QUResult, Text2SQLResult
    from app.schemas.chat import ChatRequest, SSEMessage

logger = get_logger(__name__)


# ====================================================================== #
# 三类子能力各自的接口契约（用 Protocol 如实刻画，互不强行统一）
# ====================================================================== #
@runtime_checkable
class ReplacingSubgraphAgent(Protocol):
    """【替换式子图 Agent】接口（如 Text2SQLAgent）。

    语义：当确定性路由判定为某结构化意图时，本 Agent 用自己的内部子图【替换】整条 RAG pipeline，
    端到端产出一个结构化结果。它不是召回源、也不并入召回池，而是另一条独立的回答路径。

    现有实现：``app/agents/text2sql_agent.py::Text2SQLAgent``（无需显式继承本 Protocol）。
    """

    async def arun(self, query: str, qu: "QUResult") -> "Text2SQLResult":
        """运行整条替换式子图，返回结构化结果。

        :param query: 用户自然语言问题。
        :param qu: Query Understanding 结果（可携带实体辅助子图内部步骤）。
        :return: 结构化结果（如 Text2SQLResult，含 sql/columns/rows/summary/error）。
        """
        ...


@runtime_checkable
class RecallSourceAgent(Protocol):
    """【召回源 Agent】接口（如 ShebaoAgent / ProductCodeAgent）。

    语义：它是召回层的一个【结构化数据源】，按 QU 实体查表并把命中行包成 Document，
    并入文档召回池，一起进入 粗排/精排/重排/摘要。它【不是】与 QA 平级的子智能体，
    而是"召回的一路输入"；无命中/失败时返回 []，由文档 RAG 兜底。

    现有实现：``app/agents/structured_agents.py`` 里的 StructuredAgent 子类（无需显式继承本 Protocol）。
    """

    async def search(self, qu: "QUResult", topk: int = 10) -> "list[Document]":
        """按 QU 抽取的实体查表，返回与文档召回同构的 Document 列表。

        :param qu: Query Understanding 结果（读其 entities）。
        :param topk: 最多返回多少行（结构化精确命中通常很少）。
        :return: Document 列表；无命中/失败返回 []（绝不抛异常打断召回）。
        """
        ...


@runtime_checkable
class QaAgentLike(Protocol):
    """【主图本体问答】接口（即 QA，由 QaAgent 薄门面对外暴露）。

    语义：QA 就是主图本身，astream 产出真·逐 token 的 SSE 事件流。它不是"被调用的子图"，
    而是整条链路的本体；QaAgent 只是给它一个对外的 agent 名字（见 qa_agent.py），不重写 RAG。

    现有实现：``app/graph/pipeline.py::Orchestrator``（QaAgent 直接委托它，无需显式继承本 Protocol）。
    """

    def astream(self, req: "ChatRequest") -> "AsyncIterator[SSEMessage]":
        """驱动主图执行，逐条产出 SSEMessage（含 answer_delta 的逐 token 答案流）。

        :param req: 对外请求体。
        :return: 异步迭代器，逐条产出 SSEMessage。
        """
        ...


# 教学用别名：当你只想说"任意一类子能力"时可用它表意（不做强约束，仅作文档/类型提示）。
# 注意：这只是"三选一"的联合表述，并非把三者统一成同一接口——它们的方法名/签名依旧各不相同。
AnyAgent = "ReplacingSubgraphAgent | RecallSourceAgent | QaAgentLike"


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：不连任何服务，只确认三个 Protocol 可被 isinstance 识别（结构化子类型）。
    # 这里用一个"长得像召回源"的临时对象演示 runtime_checkable 的鸭子类型判定。
    class _FakeRecall:
        async def search(self, qu, topk: int = 10):  # noqa: D401 - 仅演示形状
            return []

    print("[base 自测] _FakeRecall 是召回源 Agent 吗 =>", isinstance(_FakeRecall(), RecallSourceAgent))
    print("[base 自测] _FakeRecall 是替换式子图 Agent 吗 =>",
          isinstance(_FakeRecall(), ReplacingSubgraphAgent))
