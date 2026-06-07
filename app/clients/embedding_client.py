"""Embedding（文本向量化）客户端 —— 只对接一个 HTTP 嵌入服务（OpenAI 兼容协议）。

本模块在整体链路里的位置：基础设施层。混合召回里"稠密(dense)语义召回"那一路、HyDE、
入库脚本、Text2SQL 的 schema linking，都靠它把文本变成向量。

设计（刻意做简单，只有一条路，好懂好排错）：
- 只走 HTTP，对接 settings.embedding_base_url 指向的嵌入服务（OpenAI 兼容 /embeddings，如百炼 text-embedding / 自建 bge-m3 等）。
- 不内置本地模型加载、不做多格式兼容、不算稀疏向量——这些对当前用法是冗余。
- 请求体只发 {"model","input"}，解析 OpenAI 标准返回 {"data":[{"embedding":[...]}]}，
  与 standard 项目里验证过能跑通的写法完全一致。
- 惰性建连：import 本模块不连服务，首次真正调用 embed() 时才建 httpx.Client。

接口契约（不可变，被 recall/fine_rank/text2sql 等调用）：
- embed(texts: list[str])  -> {"dense": list[list[float]], "sparse": list[dict]}
- embed_query(text: str)   -> 同结构，单条（dense 含 1 个向量）
说明：该嵌入服务只产 dense；返回里的 sparse 恒为"与 dense 等长的空 dict 列表"，仅为保持
返回结构稳定（下游按下标取用不出错），真正的稀疏/关键词召回由 ES 的 BM25 负责。
"""
from __future__ import annotations

from typing import Any

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class EmbeddingClient:
    """文本向量化客户端（HTTP，惰性连接）。

    用法::

        ec = EmbeddingClient()
        vecs = ec.embed(["增值税税率", "小型微利企业"])["dense"]   # list[list[float]]，每条一个向量
        qv = ec.embed_query("小规模纳税人免税额度")["dense"][0]     # list[float]，单条向量
    """

    # 单次请求最多发送的文本条数：百炼 text-embedding 批量上限为 10（超过会 400 InvalidParameter），故按 10 切批。
    _MAX_CHUNK = 10

    def __init__(self) -> None:
        """仅记录配置，不建任何连接（惰性）。"""
        self._http_client = None   # httpx.Client，首次调用才建
        self._endpoint = None      # 最终请求地址，首次调用才算
        logger.info("[Embedding客户端] 初始化（未连接）：模型=%s 服务=%s",
                    settings.embedding_model, settings.embedding_base_url)

    def _get_http_client(self):
        """惰性创建 httpx.Client，并算好最终的 /embeddings 请求地址。

        :return: httpx.Client 实例（复用，避免每次新建连接池）。
        :raise RuntimeError: 未安装 httpx。
        """
        if self._http_client is None:
            try:
                import httpx
            except ImportError as e:
                raise RuntimeError("[Embedding客户端] 未安装 httpx，请 `pip install httpx`") from e
            # 兼容两种配置写法：填到 ".../v1"（自动补 /embeddings）或直接填 ".../v1/embeddings"（原样用）。
            base = (settings.embedding_base_url or "").rstrip("/")
            self._endpoint = base if base.endswith("/embeddings") else f"{base}/embeddings"
            headers = {"Content-Type": "application/json"}
            # 鉴权头：服务免鉴权时 api_key 为空，则不发该头。
            # 防御：HTTP 头必须可 ASCII 编码。若 api_key 被 .env "空值行的行内注释"污染成中文
            # (如 "# 无需鉴权" 之类中文)，直接拼进 Authorization 会让 httpx 编码头时抛
            # UnicodeEncodeError、整条向量化崩。故 strip + 跳过 '#' 开头 + ASCII 校验，
            # 任何异常值都安全降级为"不发鉴权头"，绝不让一个配置注释打断召回。
            api_key = (settings.embedding_api_key or "").strip()
            if api_key and not api_key.startswith("#"):
                try:
                    api_key.encode("ascii")
                    headers["Authorization"] = f"Bearer {api_key}"
                except UnicodeEncodeError:
                    logger.warning("[Embedding客户端] api_key 含非 ASCII 字符（疑似 .env 行内注释污染），"
                                   "已忽略 Authorization 头")
            self._http_client = httpx.Client(headers=headers, timeout=httpx.Timeout(60.0, connect=10.0))
            logger.info("[Embedding客户端] HTTP 客户端已建立 endpoint=%s", self._endpoint)
        return self._http_client

    def embed(self, texts: list[str]) -> dict[str, list]:
        """批量把文本转成 dense 向量。

        :param texts: 待向量化的文本列表。
        :return: {"dense": list[list[float]], "sparse": list[dict]}；sparse 恒为等长空 dict 列表。
        :raise RuntimeError: 调用嵌入服务失败时抛出（带清晰中文上下文）。
        """
        if not texts:
            return {"dense": [], "sparse": []}
        client = self._get_http_client()
        dense: list[list[float]] = []
        # 按 _MAX_CHUNK 分批请求，缩小单次失败影响面
        for i in range(0, len(texts), self._MAX_CHUNK):
            chunk = texts[i:i + self._MAX_CHUNK]
            payload = {"model": settings.embedding_model, "input": chunk}
            try:
                resp = client.post(self._endpoint, json=payload)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:  # noqa: BLE001 - 统一转成清晰中文报错
                logger.error("[Embedding客户端] 向量化失败：%s", e, exc_info=True)
                raise RuntimeError(f"[Embedding客户端] 调用嵌入服务失败（{self._endpoint}）：{e}") from e
            # 解析 OpenAI 标准返回：{"data":[{"embedding":[...]}, ...]}，按返回顺序与输入一一对应
            items = data.get("data", []) if isinstance(data, dict) else []
            for it in items:
                dense.append(self._to_float_list(it.get("embedding", [])))
        logger.info("[Embedding客户端] 向量化完成：%d 条文本 -> %d 个向量", len(texts), len(dense))
        # 该服务不产稀疏向量；返回等长空 dict 仅为保持结构稳定，稀疏检索由 ES BM25 负责。
        return {"dense": dense, "sparse": [{} for _ in dense]}

    def embed_query(self, text: str) -> dict[str, list]:
        """单条文本向量化（召回前对 query / 子查询 / HyDE 文档向量化常用）。

        :param text: 单条文本。
        :return: 与 embed 同结构，dense 含 1 个向量（取用时是 result["dense"][0]）。
        """
        return self.embed([text])

    @staticmethod
    def _to_float_list(vec: Any) -> list[float]:
        """把返回的向量统一成 list[float]（防止个别服务把数字返回成字符串）。

        :param vec: 服务返回的单条 embedding（一般已是 float 列表）。
        :return: list[float]。
        """
        return [float(x) for x in (vec or [])]


if __name__ == "__main__":
    # 自测：需要能连到 settings.embedding_base_url 指向的嵌入服务
    #（需嵌入服务在线，如百炼/自建端点）。连不到会抛清晰中文异常，属预期。
    try:
        ec = EmbeddingClient()
        out = ec.embed_query("增值税小规模纳税人征收率")
        print("[embedding_client 自测] 返回向量维度 =>", len(out["dense"][0]) if out["dense"] else 0)
    except Exception as exc:  # noqa: BLE001
        print("[embedding_client 自测] 需要嵌入服务在线（属预期）=>", exc)
