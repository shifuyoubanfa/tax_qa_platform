"""召回器抽象基类（BaseRecaller）。
本模块在整体链路里的位置：召回层的"接口契约"。所有具体召回器（混合召回 HybridRecaller、
将来可能新增的纯向量召回 / 纯全文召回 / 第三方API召回）都继承它，对外暴露统一的 recall 方法。

为什么要抽象基类（而不是直接写一个函数）：
1. 让上层 RecallManager 不关心"具体怎么召回"，只按统一接口调用，便于以后替换/扩展召回策略
   （对标爱搜税里每个知识库一个 HybridXxxRecallService，但都提供同名 get_docs 入口）。
2. 把"知识库标识 -> 底层 collection/index 名"的映射这类公共逻辑沉淀到基类，子类复用。

风格对标 爱搜税 kernel/recall/hybrid.py 的多知识库召回服务设计。
"""
from __future__ import annotations

import abc

from config.constants import KBase
from config.logging_config import get_logger
from config.settings import settings
from app.schemas.document import Document, QUResult, RecallStep

logger = get_logger(__name__)


class BaseRecaller(abc.ABC):
    """召回器抽象基类：定义"给定查询理解结果 + 召回步骤配置 + 目标知识库 -> 返回文档列表"的契约。

    子类只需实现 :meth:`recall`，即可被 RecallManager 统一调度。
    """

    @abc.abstractmethod
    def recall(self, qu: QUResult, step: RecallStep, kbase: str) -> list[Document]:
        """在单个知识库 kbase 上执行召回。

        :param qu: 查询理解结果（含原始/扩写子查询、HyDE文档、意图、实体）。
        :param step: 当前召回步骤配置（权重、粗召回条数等）。
        :param kbase: 目标知识库标识（KBase 枚举值，如 "policy"/"qa"/"doc"/"inspect_case"）。
        :return: 召回到的 Document 列表（已标注来源知识库、各路得分、来源子查询）。
        :raise: 由具体子类决定；网络/服务异常应捕获后给出清晰中文报错。
        """
        raise NotImplementedError

    # ---------------- 公共工具：知识库 -> 底层存储名 映射 ----------------
    @staticmethod
    def resolve_milvus_collection(kbase: str) -> str:
        """把知识库标识解析成 Milvus 集合名（稠密召回用）。

        说明（为什么这么做）：当前 demo 阶段 policy/qa/doc/inspect 四个知识库
        共用同一个文档集合 settings.milvus_doc_collection；真实生产部署应"一库一集合"，
        在这里按 kbase 返回各自集合名即可，调用方无需改动。

        :param kbase: 知识库标识（KBase 值）。
        :return: Milvus 集合名。
        """
        # TODO(部署): 生产环境建议为每个知识库建独立集合，如
        #   {KBase.POLICY.value: settings.milvus_policy_collection, ...}
        # 当前统一落到文档集合，保证无 infra 也能跑通示例。
        _ = kbase  # 占位：当前未按库区分，保留参数语义清晰
        return settings.milvus_doc_collection

    @staticmethod
    def resolve_es_index(kbase: str) -> str:
        """把知识库标识解析成 Elasticsearch 索引名（稀疏/BM25 召回用）。

        :param kbase: 知识库标识（KBase 值）。
        :return: ES 索引名。
        """
        # TODO(部署): 同上，生产应一库一索引；当前统一落到文档索引。
        _ = kbase
        return settings.es_doc_index

    @staticmethod
    def is_known_kbase(kbase: str) -> bool:
        """校验 kbase 是否为已知知识库标识，便于上层快速拦截非法配置。

        :param kbase: 待校验的知识库标识。
        :return: 是否属于 KBase 枚举。
        """
        return kbase in {k.value for k in KBase}


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：演示基类的映射工具与抽象约束。
    print("[base 自测] policy -> milvus:", BaseRecaller.resolve_milvus_collection("policy"))
    print("[base 自测] qa -> es:", BaseRecaller.resolve_es_index("qa"))
    print("[base 自测] 是否已知库 doc:", BaseRecaller.is_known_kbase("doc"))
    print("[base 自测] 是否已知库 xxx:", BaseRecaller.is_known_kbase("xxx"))
    try:
        BaseRecaller()  # 抽象类不可实例化，应抛 TypeError
    except TypeError as e:
        print("[base 自测] 抽象类不可实例化（符合预期）:", e)
