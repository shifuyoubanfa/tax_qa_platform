"""LLM 客户端（OpenAI 兼容协议）。
本模块在整体链路里的位置：基础设施层。QU（意图分类/改写/HyDE）、摘要生成、Text2SQL
的"生成/校正"等所有"调大模型"的环节，都通过本模块拿到一个 langchain_openai.ChatOpenAI 实例。

为什么用 langchain_openai.ChatOpenAI 而不是直接用 openai SDK：
- LangGraph / LangChain 生态里链路编排(.ainvoke / .astream)统一走 Runnable 接口，
  用 ChatOpenAI 能无缝接进编排图，少写胶水代码。
- 走 OpenAI 兼容协议，千问/DeepSeek/即梦/vLLM 自部署都能用，只改 base_url+model 即可。

为什么要缓存（_llm_client_cache）：
- 同一个(模型, JSON模式)组合反复创建客户端是浪费；缓存后全局复用一个实例，
  减少初始化开销，也便于统一管理。键里带 json_mode 是因为"是否强制JSON输出"会改变
  client 的构造参数（response_format），必须区分缓存。

惰性：本模块 import 时不会建任何连接，只有调用 get_llm() 时才构造 ChatOpenAI（其本身也是
惰性的，真正发请求时才连服务）。

风格严格对标 掌柜智库/app/lm/lm_utils.py（缓存 + 配置校验 + 精准异常 + 教学注释）。
"""
from __future__ import annotations

from typing import Optional

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

# 全局缓存：键为 (模型名, 是否JSON模式) 元组，值为 ChatOpenAI 实例。
# 作用：避免重复初始化客户端，提升性能、统一实例管理（对标智库 _llm_client_cache）。
_llm_client_cache: dict[tuple[str, bool], object] = {}


def get_llm(model: Optional[str] = None, json_mode: bool = False):
    """获取带全局缓存的 LangChain ChatOpenAI 客户端实例。

    适配所有 OpenAI 兼容 API（千问 / DeepSeek / vLLM 自部署等）。支持自定义模型名与
    "强制 JSON 输出"模式（意图分类、Text2SQL 等需要结构化结果时用）。

    :param model: 模型名；优先级 传入参数 > settings.llm_model。
    :param json_mode: 是否开启 JSON 输出模式；开启后强制返回 json_object，便于下游 json.loads。
    :return: 初始化完成的 langchain_openai.ChatOpenAI 实例（命中缓存则直接复用）。
    :raise ValueError: 缺失 llm_api_key / llm_base_url 等核心配置。
    :raise RuntimeError: 未安装 langchain-openai 依赖，或客户端初始化失败。
    """
    # 1. 确定目标模型（保证非空，便于做缓存键）
    target_model = model or settings.llm_model
    cache_key = (target_model, json_mode)

    # 2. 缓存命中：直接返回，避免重复构造
    if cache_key in _llm_client_cache:
        logger.debug("[LLM客户端] 缓存命中：模型=%s，JSON模式=%s", target_model, json_mode)
        return _llm_client_cache[cache_key]

    # 3. 核心配置校验：提前拦截缺失的 key/base_url，给出明确中文提示
    if not settings.llm_api_key:
        raise ValueError("[LLM客户端] 配置缺失：请在 .env 中配置 LLM_API_KEY（大模型API密钥）")
    if not settings.llm_base_url:
        raise ValueError("[LLM客户端] 配置缺失：请在 .env 中配置 LLM_BASE_URL（API基础地址）")

    # 4. 惰性导入 langchain_openai：放到函数内而非模块顶部，
    #    保证"没装 langchain-openai 时 import 本模块不报错"，符合可跑通规范。
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:  # 依赖未安装时给清晰提示
        raise RuntimeError(
            "[LLM客户端] 未安装 langchain-openai，请先 `pip install langchain-openai`"
        ) from e

    logger.info("[LLM客户端] 初始化新实例：模型=%s，JSON模式=%s，地址=%s",
                target_model, json_mode, settings.llm_base_url)

    # 5. 组装参数：
    #    - extra_body：千问(Qwen3)专属私有参数 enable_thinking=False（关闭思考链）。
    #      ⚠️ 关键：OpenAI 官方 API 对未知请求参数会直接报 400，所以默认【不注入】，
    #      只有 settings.llm_disable_thinking=True（确认指向千问时）才带上，避免误伤 OpenAI/DeepSeek。
    #    - model_kwargs：OpenAI 通用参数；json_mode 时强制 json_object 输出。
    extra_body: dict = {}
    if settings.llm_disable_thinking:
        extra_body["enable_thinking"] = False
    model_kwargs: dict = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    # 6. 构造客户端：捕获异常并抛出更友好的中文提示
    try:
        client = ChatOpenAI(
            model=target_model,
            temperature=settings.llm_temperature,   # 低温度保证税务问答的确定性
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout,           # 单次请求超时，避免卡死链路
            max_tokens=settings.llm_max_tokens,     # 单次生成最大 token
            extra_body=extra_body,
            model_kwargs=model_kwargs,
        )
    except Exception as e:
        logger.error("[LLM客户端] 模型【%s】初始化失败：%s", target_model, e, exc_info=True)
        raise RuntimeError(f"[LLM客户端] 模型【{target_model}】初始化失败：{e}") from e

    # 7. 写入缓存供复用
    _llm_client_cache[cache_key] = client
    logger.info("[LLM客户端] 实例初始化成功并缓存：模型=%s，JSON模式=%s", target_model, json_mode)
    return client


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：需要 .env 配好 LLM_* 且模型服务在线才能真正调用。
    # 这里只演示"获取实例 + 缓存命中"，不发真实请求，避免没服务时报错。
    try:
        c1 = get_llm()
        c2 = get_llm()  # 同参数应命中缓存
        print("[llm_client 自测] 两次获取是否同一实例(缓存) =>", c1 is c2)
        c3 = get_llm(json_mode=True)
        print("[llm_client 自测] JSON模式实例是否独立 =>", c3 is not c1)
    except Exception as exc:
        # 没配 key 时会走到这里，属正常（演示配置校验）
        print("[llm_client 自测] 未配置或初始化失败（属预期，先配 .env 的 LLM_*）=>", exc)
