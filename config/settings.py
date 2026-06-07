"""全局配置中心（基于 pydantic-settings）。
本模块在整体链路里的位置：最底层"基础设施层"。所有外部依赖（LLM/embedding/reranker/
Milvus/ES/Qdrant/MySQL/Mongo/MinIO/MinerU/MCP）的连接地址、账号、密钥都集中在这里，
通过环境变量(.env)注入，任何业务模块只 `from config.settings import settings` 即可读取，
绝不在代码里硬编码地址或key。

设计要点（为什么这么做）：
1. 字段全部扁平、小写命名，pydantic-settings 默认大小写不敏感，会自动映射到 .env 里的大写名，
   例如字段 `llm_base_url` 对应 .env 中的 `LLM_BASE_URL`，读起来直观、改起来无歧义。
2. 每个字段都给了"能让程序不报错启动"的本地默认值（localhost + 标准端口），
   这样在没有任何 infra 的开发机上也能 import settings、启动 FastAPI（连接采用惰性初始化）。
3. 密钥类字段默认空字符串，强制走 .env 注入，避免把真实key写进仓库。
4. extra="ignore"：.env 里多出来的、本类没声明的变量不报错，方便多人协作时各自加临时变量。

风格对标 掌柜智库/app/conf/*_config.py：集中、带教学注释、字段分组清晰。
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from config.paths import PROJECT_ROOT


class Settings(BaseSettings):
    """平台全局配置。字段按"基础设施分组"排列，便于查找与维护。

    用法::

        from config.settings import settings
        url = settings.llm_base_url

    :return: 模块级单例 `settings`，全局共享一份配置实例。
    """

    # pydantic-settings 行为配置：
    # - env_file=".env"：从项目根目录的 .env 读取（占位见 .env.example）
    # - case_sensitive=False：字段名与环境变量大小写不敏感匹配
    # - extra="ignore"：忽略 .env 里未声明的多余变量，避免启动报错
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),   # 绝对路径：无论从哪个目录/方式运行都能读到同一个 .env
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ============== 应用与日志 ==============
    # FastAPI 监听地址/端口；日志级别与日志目录（logging_config 会用到）
    app_host: str = "0.0.0.0"          # 服务监听网卡，0.0.0.0 表示对外可访问
    app_port: int = 8000               # 服务端口
    log_level: str = "INFO"            # 日志级别，INFO 及以上必须输出
    log_dir: str = "logs"              # 日志文件目录（相对项目根，运行时自动创建）

    # ============== 大模型（LLM，OpenAI兼容） ==============
    # 走 OpenAI 兼容协议（千问/即梦/DeepSeek 等均可），由 llm_client 惰性构建客户端
    llm_base_url: str = "http://localhost:8000/v1"  # OpenAI兼容API基础地址
    llm_api_key: str = ""              # API密钥（务必走 .env，勿硬编码）
    llm_model: str = "qwen2.5-32b-instruct"          # 默认对话模型名
    llm_temperature: float = 0.1       # 采样温度，低温度保证税务问答的确定性
    llm_timeout: int = 60              # 单次请求超时（秒）
    llm_max_tokens: int = 2048         # 单次生成最大token数
    # 是否向请求体注入千问专属的 extra_body={"enable_thinking": False}（关闭千问思考链）。
    # 默认 False：OpenAI 官方 API 对未知请求参数会报 400，所以用 OpenAI/DeepSeek 时必须保持 False；
    # 仅当 LLM 指向千问(Qwen3)系列、想关闭其思考输出时才设为 True。
    llm_disable_thinking: bool = False

    # ============== Embedding（向量化，支持 http / local 两种模式） ==============
    # embedding_mode=http：调远程嵌入服务；=local：本地 FlagEmbedding bge-m3
    # use_sparse：bge-m3 可同时产出 dense + sparse(稀疏词权重)，用于混合召回
    embedding_mode: str = "http"       # http | local
    embedding_base_url: str = "http://localhost:9001"  # http模式的嵌入服务地址
    embedding_api_key: str = ""        # http模式的鉴权key（如服务需要）
    embedding_model: str = "bge-m3"    # 嵌入模型名
    embedding_dim: int = 1024          # 向量维度（bge-m3 为1024，需与Milvus集合一致）
    embedding_local_path: str = ""     # local模式的本地模型权重路径
    embedding_use_sparse: bool = True  # 是否同时产出稀疏向量（混合召回需要）

    # ============== Reranker（重排，仅本地 FlagReranker） ==============
    # reranker 客户端只走本地 FlagReranker，不存在 http 重排服务，故不设 mode/base_url/api_key，
    # 避免误导成"有 http 重排服务"。仅保留模型名与本地权重路径两项。
    reranker_model: str = "bge-reranker-v2-m3"         # 重排模型名
    reranker_local_path: str = ""      # 本地模型权重路径（FlagReranker 加载）

    # ============== Milvus（稠密向量库，文档稠密召回） ==============
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_user: str = ""              # 开启鉴权时填写
    milvus_password: str = ""
    milvus_db: str = "default"         # Milvus 数据库名
    milvus_doc_collection: str = "tax_doc"             # 文档稠密向量集合名

    # ============== Elasticsearch（全文/BM25 稀疏召回） ==============
    es_hosts: str = "http://localhost:9200"            # 多个用逗号分隔，由 es_client 解析
    es_user: str = ""
    es_password: str = ""
    es_doc_index: str = "tax_doc"      # 文档全文索引名

    # ============== Qdrant（数仓元数据向量库，Text2SQL schema linking 用） ==============
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""           # Qdrant Cloud 等需要时填写
    qdrant_meta_collection: str = "schema_meta"        # 表/字段元数据向量集合名

    # ============== MySQL（经营/财税数仓，Text2SQL 执行库） ==============
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_db: str = "tax_dw"           # 数仓库名
    mysql_charset: str = "utf8mb4"     # 字符集，utf8mb4 支持完整中文与emoji
    # —— Text2SQL 只读账号（纵深防御第二层）——
    # 在链路里的位置：Text2SQL Agent 层已做 AST 校验(只放行 SELECT)；这里再叠一层"DB 侧只读权限"，
    # 即便上层校验被绕过，数据库账号本身也只有 SELECT 权，无法 DDL/DML，形成双层防护。
    # 默认留空：表示"沿用主账号"，行为与现有完全一致（不破坏任何现状）；
    # 一旦在 .env 填了只读账号，Text2SQL 的 execute 就会改用只读账号连库。
    mysql_readonly_user: str = ""      # Text2SQL 专用只读账号；空=沿用主账号(mysql_user)
    mysql_readonly_password: str = ""  # 只读账号密码；mysql_readonly_user 为空时此项被忽略

    # ============== MongoDB（多轮会话历史存储） ==============
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "tax_qa"
    mongo_history_collection: str = "chat_history"     # 会话历史集合名

    # ============== MinIO（多模态图片对象存储） ==============
    minio_endpoint: str = "localhost:9000"             # 注意：MinIO客户端要的是 host:port，不带scheme
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "tax-doc-images"               # 存放文档图片的桶名
    minio_secure: bool = False         # True 走 https，本地开发一般 False

    # ============== MinerU（PDF/文档解析，支持 http / local） ==============
    mineru_mode: str = "http"          # http | local
    mineru_base_url: str = "http://localhost:9003"     # http模式的解析服务地址
    mineru_token: str = ""             # 解析服务鉴权token

    # ============== MCP（Model Context Protocol，可选外部工具调用） ==============
    # 在链路里的位置：结构化查询(社保/产品编码/Text2SQL)的"可选外接层"。
    # mcp_enabled=False 时，McpClient 安全降级（不真连），结构化查询走原生 agent，
    # 保证无网络/无 MCP server 也能跑；=True 时结构化查询改为经 MCP 调用我们暴露的工具。
    mcp_enabled: bool = False          # 总开关：True 时结构化查询经 MCP 调工具，False 走原生 agent（默认关，优雅降级）
    # 传输方式：stdio=本地以子进程方式拉起我们自己的 MCP server；sse=连远程 MCP server（如百炼）。
    mcp_transport: str = "stdio"       # stdio | sse
    # stdio 模式专用：启动我们自己 MCP server 的命令（作为子进程被 MCP 客户端拉起）。
    mcp_server_command: str = "python -m app.mcp.server"
    # sse 模式专用：远程 MCP server 的 SSE 地址（如百炼）。stdio 模式下留空即可。
    mcp_sse_url: str = ""
    # 百炼 MCP 兼容字段（历史保留，sse 模式连百炼时可用其地址/key；不要删，避免 .env 旧变量失配）。
    mcp_bailian_base_url: str = ""     # 百炼MCP服务地址
    mcp_bailian_api_key: str = ""      # 百炼MCP鉴权key

    # ============== 联网搜索（WebSearch via MCP，可选补充来源，默认关、物理隔离） ==============
    # 在链路里的位置：检索层之外的"可选补充来源"。与本地权威法规库【物理隔离】——
    # 联网结果只作未核验补充、绝不进权威引用([[citation:N]])，由确定性意图触发、不由 LLM 决定。
    # websearch_mcp_enabled=False（默认关）时 WebSearchAgent 直接返回 []、绝不触网，保证零外部依赖也能跑。
    websearch_mcp_enabled: bool = False    # 联网搜索总开关：True 才允许 WebSearchAgent 经 MCP 触网（默认关）
    # 传输方式：联网搜索这条链路独立于结构化查表的 mcp_*，固定走 streamable http（百炼等远程 MCP server）。
    websearch_mcp_transport: str = "http"  # http（streamable http）；预留可扩展，当前只用 http
    # 联网搜索 MCP server 的地址（如百炼 DashScope 的 MCP 端点）。留空 + 总开关关 => 永不触网。
    websearch_mcp_url: str = ""             # 联网搜索 MCP server 的 streamable http 地址
    # 百炼 DashScope 鉴权 key（作为 Bearer token 注入 http 请求头 Authorization）。务必走 .env，勿硬编码。
    dashscope_api_key: str = ""            # 百炼 DashScope API Key（联网搜索 MCP 鉴权用）

    # ============== 检索链路通用阈值（被召回/排序/重排各层读取） ==============
    recall_topk: int = 50              # 单库粗召回条数
    fine_topk: int = 20                # 精排后保留条数
    rerank_topk: int = 5               # 重排后最终保留条数

    @property
    def mysql_dsn(self) -> str:
        """组装 SQLAlchemy 异步连接串（aiomysql 驱动）。

        之所以做成 property 而非普通字段：连接串是由多个独立字段拼出来的"派生值"，
        放成 property 既能随时反映最新字段，又避免在 .env 里重复维护一长串。

        :return: 形如 "mysql+aiomysql://user:pwd@host:port/db?charset=utf8mb4" 的连接串。
        """
        return (
            f"mysql+aiomysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}"
            f"?charset={self.mysql_charset}"
        )

    @property
    def mysql_readonly_dsn(self) -> str:
        """组装 Text2SQL 专用"只读账号"连接串（仿 mysql_dsn，仅换账号）。

        纵深防御第二层的连接入口：host/port/db/charset 都与主 DSN 相同，只把账号密码
        换成只读账号。设计成 property 的理由同 mysql_dsn——派生值随字段实时反映，无需在
        .env 重复维护整串连接。

        关键的"优雅降级"语义：当 mysql_readonly_user 为空（默认）时，直接回退主 DSN，
        因此未配置只读账号的环境下 Text2SQL 行为与现状完全一致；只有真正填了只读账号，
        才会切到只读账号连库。

        :return: 只读账号连接串；只读账号为空时回退 self.mysql_dsn。
        """
        if not self.mysql_readonly_user:
            return self.mysql_dsn
        return (
            f"mysql+aiomysql://{self.mysql_readonly_user}:{self.mysql_readonly_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}"
            f"?charset={self.mysql_charset}"
        )


# 模块级单例：全局共享同一份配置实例。
# 任何模块 `from config.settings import settings` 拿到的都是这一个对象。
settings = Settings()


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：打印几组关键配置，确认 .env 注入是否生效。
    print("[settings 自测] 服务监听 =>", f"{settings.app_host}:{settings.app_port}")
    print("[settings 自测] LLM模型 =>", settings.llm_model, "| 地址 =>", settings.llm_base_url)
    print("[settings 自测] MySQL DSN =>", settings.mysql_dsn)
    print("[settings 自测] 召回阈值 => recall_topk=%s fine_topk=%s rerank_topk=%s"
          % (settings.recall_topk, settings.fine_topk, settings.rerank_topk))
