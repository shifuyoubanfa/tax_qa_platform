"""核心数据结构（跨层共享的"数据契约"）：检索文档、实体、QU结果、检索计划、Text2SQL结果。
全部用 dataclass，轻量、可直接 asdict 序列化。任何模块都从这里导入，禁止各自重定义。"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from config.constants import DEFAULT_INTENT, RouteType

@dataclass
class Document:
    """一条召回/排序的最小单元，贯穿 召回->粗排->精排->重排->摘要 全链路。"""
    doc_id: str
    title: str = ""
    content: str = ""
    kbase: str = ""                                  # 来源知识库（KBase值）
    score: float = 0.0                               # 当前阶段综合得分
    dense_score: float = 0.0                         # 稠密(向量)得分
    sparse_score: float = 0.0                        # 稀疏(BM25/全文)得分
    rerank_score: float = 0.0                        # 重排得分
    metadata: dict[str, Any] = field(default_factory=dict)  # doc_no/clause_path/policy_type/region/effective_status/publish_date...
    image_keys: list[str] = field(default_factory=list)     # MinIO对象键（多模态图片）
    raw_query_from: str = ""                         # 该文档由哪个子查询召回而来
    def to_reference(self) -> dict:
        """转成给前端展示的引用条目。"""
        return {"doc_id": self.doc_id, "title": self.title, "kbase": self.kbase,
                "score": round(self.score, 4), "metadata": self.metadata, "image_keys": self.image_keys}

@dataclass
class Entities:
    """从问题里抽取的结构化实体（用于精确过滤/路由）。"""
    doc_no: Optional[str] = None                     # 文号，如 财税〔2024〕18号
    year: Optional[str] = None
    law_name: Optional[str] = None                   # 法规名，如 企业所得税法
    region: list[str] = field(default_factory=list)  # 地域
    company: list[str] = field(default_factory=list) # 公司名
    product_name: list[str] = field(default_factory=list)  # 商品名

@dataclass
class QUResult:
    """Query Understanding 的产物：意图 + 子查询(原始+扩写+HyDE) + 实体。"""
    raw_query: str
    intent: str = DEFAULT_INTENT.value
    sub_queries: list[str] = field(default_factory=list)
    hyde_doc: str = ""                               # HyDE假设性文档（用于稠密召回）
    entities: Entities = field(default_factory=Entities)
    is_short_query: bool = False

@dataclass
class RecallStep:
    """一个召回步骤的配置（对标爱搜税 search_instance_mapping 里的一项）。"""
    kbases: list[str]                                # 本步要查的知识库
    weights: dict[str, float] = field(default_factory=lambda: {"dense": 0.5, "sparse": 0.5})
    coarse_topk: int = 50
    fine_rank_method: str = "bge_content"
    fine_topk: int = 20

@dataclass
class RetrievalPlan:
    """SearchRouter 根据意图产出的完整检索计划。"""
    route_type: str = RouteType.RAG.value
    steps: list[RecallStep] = field(default_factory=list)  # route_type=rag 时使用
    # 本路由命中的 Agent 名（单个，用途靠 route_type 区分）：
    #   route_type=text2sql -> 替换整条 pipeline 的 agent（如 text2sql_agent）；
    #   route_type=rag      -> 与文档召回并存的"结构化查表 agent"（社保/商品编码），由 rag_recall 并入召回池；
    #   None                -> 纯文档 RAG。
    agent_name: Optional[str] = None

@dataclass
class Text2SQLResult:
    """Text2SQL Agent 的产物。"""
    question: str = ""
    sql: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    row_count: int = 0
    summary: str = ""
    error: str = ""
    def to_dict(self) -> dict:
        return asdict(self)
