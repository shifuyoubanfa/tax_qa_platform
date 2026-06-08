# tax_qa_platform · 财税智能问答平台（应用端）

> 一套"教学级"的财税 RAG + Text2SQL 问答平台。把生产级检索/RAG 平台的核心套路
> （意图路由、多库混合召回、粗排/精排/重排、HyDE、摘要引用、LangGraph 编排、自然语言查数）
> 浓缩成一份**结构清晰、注释详尽、能跑通**的工程，便于照着学、照着改。

本项目定位为财税技术"三柱"中的 **应用端**：
- **数据端**：政策爬取/清洗/结构化入库（上游，提供原料）。
- **算法端**：embedding / reranker / Text2SQL 模型与服务（被本平台以 HTTP 或本地方式调用）。
- **应用端（本项目）**：把用户的一句话问题，经过"理解 → 路由 → 检索 → 排序 → 生成"，
  变成带引用、可溯源的专业回答；结构化诉求则走自然语言查数（Text2SQL）。

---

## 1. 它能做什么

输入：用户一句自然语言税务问题（可多轮）。
输出：流式（SSE）返回 **意图 → 检索进度 → 参考片段 → 最终回答（带 [citation:x] 引用）**；
若问题是"查自己经营数据"，则自动改走 Text2SQL，返回 **SQL + 数据表 + 数据解读**。

支持 7 类意图（见 `config/constants.py` 的 `Intent`）：

| 意图 | 说明 | 检索策略侧重 |
|------|------|--------------|
| 精确法规类 | 点名法规/带文号 | 重稀疏(BM25)精确匹配，按文号过滤 |
| 政策汇集类 | 某主题政策合集 | 政策库为主，召回面更宽 |
| 通用问题类 | 普通税务问答 | 重稠密(向量)语义召回 |
| 稽查案例类 | 稽查/处罚案例 | 案例库 + 政策库联合 |
| 查社保类 | 社保基数/比例 | 结构化向问答/文档库 |
| 查商品编码类 | 税收分类编码 | 结构化向文档库 |
| 经营数据查询类 | 查自有经营数据 | **走 Text2SQL 分支** |

---

## 2. 整体架构（链路图）

```text
                              ┌─────────────────────────────────────────────┐
   用户问题 ──HTTP/SSE──▶     │            FastAPI (app/main.py)              │
                              │      POST /api/v1/chat → Orchestrator         │
                              └───────────────────────┬─────────────────────┘
                                                       │
                                          ┌────────────▼────────────┐
                                          │  Orchestrator (LangGraph)│  app/graph/pipeline.py
                                          └────────────┬────────────┘
                                                       │
                       ┌───────────────────────────────▼───────────────────────────────┐
                       │ 1) QueryUnderstanding  app/core/qu/                             │
                       │    意图分类 IntentClassifier（规则正则优先，兜底调 LLM）        │
                       │    实体抽取 EntityExtractor（文号/年份/法规名/地域/公司/商品）   │
                       │    查询改写 QueryRewriter（子查询扩写 + HyDE 假设文档）          │
                       └───────────────────────────────┬───────────────────────────────┘
                                                       │ QUResult(intent, sub_queries, hyde, entities)
                                          ┌────────────▼────────────┐
                                          │ 2) SearchRouter          │ app/core/router/
                                          │  意图 → RetrievalPlan     │ (对标爱搜税 search_instance_mapping)
                                          └──────┬──────────────┬────┘
                              route=rag         │              │   route=text2sql
              ┌──────────────────────────────────▼──┐        ┌──▼───────────────────────────────────┐
              │ 3) 多路混合召回 RecallManager          │        │  Text2SQLAgent  app/agents/            │
              │    HybridRecaller: dense(Milvus)       │        │   schema_link(Qdrant 选表/字段)        │
              │    + sparse(ES BM25) 加权融合          │        │     → generate_sql(LLM)                │
              │ 4) 粗排 CoarseRanker（多路 RRF 去重）   │        │     → validate_fix(报错回灌重试 N 次)  │
              │ 5) 精排 FineRanker（bge 标题/正文相似） │        │     → execute(MySQL 只读)              │
              │ 6) 重排 ReRanker（reranker 交叉打分）   │        │     → summarize(LLM 解读数据)          │
              └───────────────────┬───────────────────┘        └──────────────────┬────────────────────┘
                                  │ top-k Documents                                 │ Text2SQLResult
                       ┌──────────▼──────────┐                                      │
                       │ 7) Summarizer        │  app/core/summarize/                │
                       │  拼上下文 + 流式生成  │  （依据片段作答 + [citation] 引用） │
                       └──────────┬───────────┘                                      │
                                  └───────────────────────┬────────────────────────┘
                                                          │ SSE 事件流
                                          intent / retrieval / references / sql / answer_delta / done
                                                          ▼
                                                      前端渲染

   依赖的外部基础设施（全部惰性连接，缺失也不影响 import / 启动）：
   LLM(OpenAI兼容) · Embedding(bge-m3 http/local) · Reranker(bge http/local)
   Milvus(稠密) · Elasticsearch(BM25) · Qdrant(数仓元数据) · MySQL(数仓) · MongoDB(会话) · MinIO(图片) · MinerU(解析) · MCP(可选)
```

> 想深入每一层"为什么这么设计"，看 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

---

## 3. 技术栈

| 分类 | 选型 | 作用 |
|------|------|------|
| Web/接口 | FastAPI + uvicorn + SSE | 对外 HTTP，流式返回回答 |
| 编排 | LangGraph | 主问答 pipeline、Text2SQL 多步 agent |
| 大模型 | OpenAI 兼容 (langchain-openai) | 意图兜底、改写、HyDE、摘要、SQL 生成/校正 |
| 向量化 | bge-m3（http 或本地 FlagEmbedding） | dense + sparse 混合召回 |
| 重排 | bge-reranker-v2-m3（http 或本地） | 交叉编码精排 |
| 稠密库 | Milvus | 文档向量召回 |
| 稀疏库 | Elasticsearch | BM25/全文召回 |
| 元数据库 | Qdrant | Text2SQL schema linking |
| 数仓 | MySQL（SQLAlchemy + aiomysql） | Text2SQL 只读查询 |
| 会话 | MongoDB（motor） | 多轮历史 |
| 对象存储 | MinIO | 多模态图片 |
| 文档解析 | MinerU（http 或本地 magic-pdf） | PDF → 结构化 + 图片 |
| 配置 | pydantic-settings + .env | 全部地址/密钥走配置 |

---

## 4. 目录结构

```text
tax_qa_platform/
├── README.md                     本文件（总览 + 运行手册）
├── requirements.txt              依赖（重依赖注释为可选）
├── .env.example                  所有环境变量占位（复制为 .env 使用）
├── config/
│   ├── settings.py               全局配置（扁平字段 + mysql_dsn）
│   ├── constants.py              意图/知识库/路由/SSE 枚举（词汇表）
│   ├── logging_config.py         统一日志（stdout + 滚动文件）
│   └── prompts/*.prompt          7 个提示词模板（意图/改写/HyDE/摘要/Text2SQL×3）
├── app/
│   ├── main.py                   FastAPI 入口（lifespan 初始化日志、挂路由、CORS）
│   ├── api/                      路由：/api/v1/chat（SSE）、/health
│   ├── schemas/                  数据契约：Document/QUResult/RetrievalPlan/ChatRequest...
│   ├── clients/                  各基础设施客户端（惰性连接，http/local 切换）
│   ├── core/
│   │   ├── qu/                   意图分类 / 实体抽取 / 查询改写+HyDE / 编排
│   │   ├── recall/              单库混合召回 + 多库并行召回
│   │   ├── rank/                粗排(RRF) + 精排(bge)
│   │   ├── rerank/             重排(reranker)
│   │   ├── router/             意图→检索计划路由表
│   │   └── summarize/         上下文拼装 + 流式摘要
│   ├── agents/                 Text2SQL agent（LangGraph 多步）
│   ├── graph/                  全局 State、节点、主 pipeline
│   ├── memory/                 多轮会话记忆
│   └── utils/                  prompt 加载 / SSE / 归一化 / 计时
├── scripts/                    init_collections / ingest_documents / build_metadata_index
├── data/                       数据槽位（见 data/README.md，仓库不放真实数据）
├── docs/ARCHITECTURE.md        架构详解（照着学的教程）
├── logs/                       运行时日志（自动生成）
└── tests/                      QU / 路由 / 冒烟测试（不真连外部服务）
```

---

## 5. 怎么把它跑通（运行手册）

> 目标：从零到 `curl /api/v1/chat` 拿到流式回答。下面以 Linux/macOS 为例；
> Windows 用 PowerShell 时把激活命令换成 `.venv\Scripts\Activate.ps1` 即可。

### 第 0 步 · 先决条件
- Python 3.10+（建议 3.11/3.12）
- 一个 OpenAI 兼容的 LLM 服务地址 + key（千问/DeepSeek/vLLM/Ollama 均可）
- 一个 embedding 服务、一个 reranker 服务（HTTP 模式最省事；没有就用本地模式，需装重依赖）
- Docker（用来一键起 Milvus/ES/Qdrant/MySQL/Mongo/MinIO，最方便）

### 第 1 步 · 建 conda 环境并安装依赖
```bash
cd tax_qa_platform
conda create -n tax_qa python=3.11 -y     # 3.11 兼容性最好（torch/FlagEmbedding/pymilvus 都稳）
conda activate tax_qa
python -m pip install -U pip
pip install -r requirements.txt
# 若要用本地 embedding/rerank/PDF 解析，再解开 requirements.txt 末尾的可选重依赖后安装
```

### 第 2 步 · 起基础设施（一键 docker compose）
> 本项目已带 `docker-compose.yml`，端口/账号都对齐了 .env 默认值，一条命令拉起全部服务：
> Milvus(含 etcd/minio 依赖) + Elasticsearch + Qdrant + MySQL + MongoDB。
> 内存紧张时可只留 milvus + elasticsearch 跑基础 RAG，其余按需开。

前置：装好 Docker Desktop（Windows 走 WSL2 后端），并在 Settings → Resources 给够内存（建议 6~8GB）。

```bash
cd tax_qa_platform
docker compose up -d          # 首次会下载镜像，较慢；之后秒起
docker compose ps             # 看状态：Milvus 约需 30~60s 才 healthy
docker compose logs -f milvus # 看某服务日志（milvus 可换成 es/qdrant/mysql/mongo/minio）
# 关闭：docker compose down（保留数据）；docker compose down -v（连数据一起删）
```

验证关键服务起来了：
```bash
curl http://localhost:9091/healthz          # Milvus 健康
curl http://localhost:9200                   # Elasticsearch
curl http://localhost:6333/readyz            # Qdrant
# MinIO 控制台：浏览器开 http://localhost:9001 （账号 minioadmin / minioadmin）
```

### 第 3 步 · 填配置
```bash
cp .env.example .env
# 编辑 .env：至少填 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL；
# 把 EMBEDDING_BASE_URL、RERANKER_BASE_URL 指向你的服务（或改为 local 模式）；
# 把 Milvus/ES/Qdrant/MySQL/Mongo/MinIO 的 host/port/账号填成第 2 步起的服务。
```

### 第 4 步 · 初始化索引/集合（幂等，可重复跑）
```bash
python scripts/init_collections.py    # 创建 Milvus collection / ES index / Qdrant collection
```

### 第 5 步 · 导入数据（先按 data/README.md 准备数据）
```bash
# 文档入库：解析(MinerU)→切分→embedding→写 Milvus+ES；图片→MinIO
python scripts/ingest_documents.py

# 数仓元数据入库：供 Text2SQL schema linking（写 Qdrant）
python scripts/build_metadata_index.py
```

### 第 6 步 · 启动服务
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 或：python -m app.main
```

### 第 7 步 · 验证
```bash
# 健康检查
curl http://localhost:8000/health

# 流式问答（RAG）
curl -N -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"小微企业有哪些增值税优惠政策","stream":true}'

# 自然语言查数（Text2SQL，会命中"经营数据查询类"意图）
curl -N -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"今年每个月的销售额是多少","stream":true}'
```

返回是 SSE 流，依次出现 `event: intent / retrieval / references / (sql) / answer_delta ... / done`。

---

## 6. 没有 infra 也能跑的部分（学习友好）

所有外部连接都是**惰性初始化**（首次使用才连），因此：
- `python -c "import app.main"`、`uvicorn app.main:app` 可以正常启动，不会因为缺 Milvus/ES 等而崩。
- 跑 `pytest`（`tests/`）只验证 QU 规则、路由表、模块可导入与图可编译，**不真连**外部服务。

```bash
pytest -q
```

---

## 7. 学习路线建议（按链路顺序读）

1. `config/constants.py` + `app/schemas/document.py` —— 先掌握"词汇表"和"数据契约"。
2. `app/core/qu/` —— 理解一句话怎么被理解成 意图+子查询+HyDE+实体。
3. `app/core/router/search_router.py` —— 意图如何映射到检索策略（对标爱搜税路由表）。
4. `app/core/recall/` → `rank/` → `rerank/` —— 召回到排序的完整漏斗。
5. `app/core/summarize/summarizer.py` —— 上下文拼装与带引用的生成。
6. `app/agents/text2sql_agent.py` + `app/graph/pipeline.py` —— LangGraph 多步编排。

> 每个 `.py` 文件顶部都写了"本模块做什么/在链路里的位置"，函数有完整 docstring，
> 关键行有"为什么这么做"的行内注释。很多文件结尾还带 `if __name__ == "__main__"` 自测块，
> 可单文件运行学习。

---

## 8. 许可与免责
本项目用于学习与工程参考。回答内容由检索资料 + 大模型生成，**仅供参考、不构成正式税务意见**，
具体执行以现行有效法规原文与主管税务机关口径为准。请勿将真实密钥/业务数据提交进仓库。
