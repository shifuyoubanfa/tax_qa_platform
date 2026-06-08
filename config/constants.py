"""全局常量与枚举：意图、知识库、路由类型、SSE事件类型。
本模块是整个平台的"词汇表"，被 QU/路由/召回/编排各层共享，禁止散落魔法字符串。"""
from enum import Enum

class Intent(str, Enum):
    """用户问题意图分类（决定后续走哪条检索策略）。值为中文，便于日志直读。"""
    PRECISE_REGULATION = "精确法规类"   # 指名某部法规/文号，重关键词精确匹配
    POLICY_COLLECTION = "政策汇集类"     # 某主题下的政策合集
    GENERAL_QA = "通用问题类"            # 普通税务问答，重语义
    INSPECT_CASE = "稽查案例类"          # 稽查/处罚案例
    SOCIAL_SECURITY = "查社保类"         # 社保基数/比例等结构化查询
    PRODUCT_CODE = "查商品编码类"        # 商品/税收分类编码结构化查询
    DATA_QUERY = "经营数据查询类"        # 自然语言查经营/财税数据 -> 走 Text2SQL

DEFAULT_INTENT = Intent.GENERAL_QA

class KBase(str, Enum):
    """数据源/检索通道标识（标记一条 Document 来自哪个来源）。

    前 4 个 = 文档知识库（存 Milvus + ES、走文档召回）；
    后 2 个 = 结构化数据源（存 MySQL 表、由结构化 Agent 查表产出，不走 Milvus/ES）。
    统一登记进枚举，避免散落的魔法字符串。"""
    # —— 文档知识库（Milvus 稠密 + ES 稀疏 召回）——
    POLICY = "policy"          # 政策法规条款库（对应 standard 思路）
    QA = "qa"                  # 历史问答库
    DOC = "doc"                # 多模态文档/操作手册库（对应 掌柜智库 思路）
    INSPECT = "inspect_case"   # 稽查案例库
    # —— 结构化数据源（MySQL 表，由结构化 Agent 查表产出；不参与 Milvus/ES 文档召回）——
    SOCIAL_SECURITY = "social_security"   # 社保缴费表（ShebaoAgent）
    PRODUCT_CODE = "product_code"         # 商品税收分类编码表（ProductCodeAgent）
    # —— 联网补充来源（WebSearchAgent 经 MCP 联网搜索产出；与本地权威库【物理隔离】）——
    # 关键约束：WEB 来源只作"未核验"补充，绝不进权威法规引用([[citation:N]])，默认关、由确定性意图触发。
    WEB = "web"                           # 联网搜索补充来源（WebSearchAgent，未核验·以官方为准）

class RouteType(str, Enum):
    """顶层路由：非结构化走RAG，结构化走Text2SQL。"""
    RAG = "rag"
    TEXT2SQL = "text2sql"

class FineRankMethod(str, Enum):
    """精排方式。"""
    BGE_TITLE = "bge_title"      # 用标题算语义相似
    BGE_CONTENT = "bge_content"  # 用正文算语义相似
    DIRECT = "direct"           # 不精排，直接透传

class SSEEvent(str, Enum):
    """SSE流式事件类型（前端按此渲染进度）。"""
    INTENT = "intent"
    RETRIEVAL = "retrieval"
    ANSWERABILITY = "answerability"   # 可答性门控结论（重排后、生成前的"能不能答"判定）
    REFERENCES = "references"
    SQL = "sql"
    ANSWER_DELTA = "answer_delta"
    DONE = "done"
    ERROR = "error"
