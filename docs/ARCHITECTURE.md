# 架构详解（照着学的教程）

本文逐层拆解 `tax_qa_platform` 的设计，重点不是"代码写了什么"，而是 **"为什么这么设计"**。
读完你应该能回答：一句话问题进来，到底经过了哪些处理、每一步解决了什么问题、换个场景该怎么改。

> 配合阅读：链路总图见 [README.md](../README.md) 第 2 节；数据结构契约见 `app/schemas/document.py`；
> 词汇表（意图/库/路由枚举）见 `config/constants.py`。

---

## 0. 设计哲学（三条主线）

1. **契约先行**：跨层共享的数据结构（`Document`/`QUResult`/`RetrievalPlan`/`Text2SQLResult`）
   和接口签名先固定下来，各层只依赖契约、不依赖彼此实现。这样多人并行开发、替换某一层实现都不破坏全局。
2. **配置与连接分离、惰性初始化**：所有地址/密钥走 `config/settings.py`（`.env` 注入），
   所有外部连接"首次使用才建"。好处：没有 infra 也能 `import`、能启动 FastAPI、能跑单测，学习门槛低。
3. **意图驱动一切**：先把问题"理解"成意图，再由意图决定走 RAG 还是 Text2SQL、走哪些库、稀疏/稠密怎么配权重。
   这是检索质量的总开关，思路对标爱搜税的 `search_instance_mapping` 路由表。

---

## 1. 入口层：FastAPI + SSE

**位置**：`app/main.py`、`app/api/routes_chat.py`。

- 对外只暴露一个核心接口 `POST /api/v1/chat`，返回 `text/event-stream`（SSE 流）。
- **为什么用 SSE 而不是一次性返回？** 问答链路较长（理解→检索→排序→生成），一次性等全部完成再返回，
  用户要干等好几秒、体验差。SSE 让我们边算边推：先推"识别到的意图"，再推"检索进度/参考片段"，
  最后逐 token 推"答案"。前端可即时渲染进度条与引用卡片。
- SSE 消息格式由 `app/utils/sse.py::format_sse(event, data)` 统一生成
  （`event: x\ndata: {json}\n\n`），事件类型固定在 `SSEEvent` 枚举里：
  `intent / retrieval / answerability / references / sql / answer_delta / done / error`
  （`answerability` 透出"可答性门控"结论，见 §6.4）。
- 入口层只做"取请求 → 调 `Orchestrator.astream` → 把每个 `SSEMessage` 格式化下发"，**不含业务逻辑**，
  保证关注点分离。

---

## 2. 编排层：LangGraph Orchestrator

**位置**：`app/graph/`（`state.py` 全局状态、`nodes.py` 节点、`pipeline.py` 编排）。

- 用 **LangGraph** 把整条链路画成一张有向图：每个节点读写同一个 `GraphState`（TypedDict），
  按边的方向流转。思路对标"掌柜问数"的 LangGraph 多步 Text2SQL。
- **为什么用图而不是一长串函数调用？**
  1. 可读：链路一目了然，节点职责单一，便于教学与维护。
  2. 可控：能在节点间做**条件分支**（路由到 RAG 还是 Text2SQL）、**循环**（SQL 校正重试）。
  3. 可观测：每个节点进出都打日志（命中意图、召回数量、排序结果数、SQL、异常），排错快。
- `GraphState`（见 `app/graph/state.py`）用 `total=False` 让所有字段可选，节点**逐步填充**：
  输入 `query` → `qu`（理解结果）→ `plan`（检索计划）→ `recalled/ranked/reranked`
  → `answerable/answer_confidence`（可答性门控，见 §6.4）→ `context/references`
  →（或 `text2sql_result`）→ `answer`。
- `Orchestrator.astream(req)` 是对外的异步生成器：驱动图执行，并在关键节点产出 `SSEMessage` 流。

---

## 3. 查询理解层（QU：Query Understanding）

**位置**：`app/core/qu/`。产物是 `QUResult(intent, sub_queries, hyde_doc, entities, is_short_query)`。

这一层把"模糊的一句话"加工成"检索友好的结构化信息"，是后续所有环节的输入。分三件事：

### 3.1 意图分类 `IntentClassifier.classify`
- **规则正则优先，LLM 兜底**。为什么？意图类别里有很多"强信号"：带文号（`财税〔2024〕18号`）大概率是
  精确法规类、含"社保/缴费基数"是查社保类、含"编码/税收分类编码"是查商品编码类、含"销售额/利润/本月"
  指向经营数据查询类。这些用正则**又快又准又省钱**，命中就直接定意图；只有规则都不命中时才调 LLM
  （`config/prompts/intent_classify.prompt`，要求输出 `{"intent": 中文意图}`）。
- **为什么意图值用中文？** 直接出现在日志里可读性最好（对标爱搜税）。枚举集中在 `Intent`，杜绝魔法字符串。

### 3.2 实体抽取 `EntityExtractor.extract`
- 抽 `doc_no / year / law_name / region / company / product_name`（见 `Entities`）。
- 文号、年份用**正则**抽取（思路对标爱搜税 `kernel/qu.py` 的文号/年份正则）；
  公司名可用 `companynameparser`；地域可用词典/规则。
- **为什么要抽实体？** 给召回做**精确过滤**：精确法规类命中文号后，可在 ES 用 `doc_no` 过滤，
  把"差不多相关"压成"精确命中"，大幅提升 top1 准确率。

### 3.3 查询改写 + HyDE `QueryRewriter.rewrite`
- 产出两样东西：`sub_queries`（2~3 条扩写子查询）和 `hyde_doc`（一段假设性答案文档）。
- **为什么改写？** 用户问题常常短、口语化（短查询 `is_short_query=True`），直接检索召回差。
  扩写成多个语义完整、角度互补的子查询做**多路召回**，能显著提高召回率（`query_rewrite.prompt`）。
- **为什么要 HyDE（Hypothetical Document Embeddings）？** 问题向量和"答案型文档"向量在语义空间里
  常常隔得远。先让 LLM 生成一段"像标准答案/政策正文"的范文（`hyde.prompt`），再用它做**稠密召回**，
  因为它和库里条款的风格/用词更接近，更容易召回到真正相关的正文。注意 HyDE 文档只用于检索、不展示给用户。

---

## 4. 路由层：意图 → 检索计划

**位置**：`app/core/router/search_router.py`。产物是 `RetrievalPlan(route_type, steps, agent_name)`。

- **顶层路由**：`route_type` 决定走 `RAG` 还是 `TEXT2SQL`。
  "经营数据查询类"→ `TEXT2SQL`（`agent_name` 指向 Text2SQL agent）；其余 6 类 → `RAG`。
  这是"非结构化检索 vs 结构化查数"的总岔路。
- **RAG 子路由**：用一张"意图 → `RecallStep[]`"映射表（对标爱搜税 `search_instance_mapping`）决定：
  - 查哪些知识库（`KBase`：政策库/问答库/文档库/案例库），一个意图可路由到多个库；
  - 每个库的 **dense/sparse 权重**（混合召回融合用）；
  - 粗召回条数 `coarse_topk`、精排方式 `fine_rank_method`、精排保留 `fine_topk`。
- **为什么按意图配不同权重？** 不同意图对"精确 vs 语义"的需求不同：
  - **精确法规类**：重 **sparse(BM25)**，因为要按法规名/文号精确匹配（关键词命中比语义更重要）。
  - **通用问题类**：重 **dense(向量)**，因为问题和答案是语义相关而非字面相同。
  - **稽查案例类**：案例库 + 政策库联合召回，案例为主、政策兜底（对标爱搜税 summarize 的意图化拼装）。
  这种"一张表把策略写清楚"的做法，让新增意图/调权重只改配置、不改链路代码。

---

## 5. 召回层：多路混合召回

**位置**：`app/core/recall/`（`hybrid.py` 单库、`manager.py` 多库）。产物是 `list[Document]`。

- **单库混合召回 `HybridRecaller.recall`**：对一个库同时做：
  - **dense**：用 HyDE/子查询的向量去 **Milvus** 查 top-k；
  - **sparse**：用文本去 **Elasticsearch** 做 BM25/全文查 top-k；
  - 然后按 `step.weights` 把两路分数**归一化后加权融合**（归一化用 `app/utils/normalize.py`，
    避免 dense 与 sparse 分数量纲不同导致一边压倒另一边）。
- **为什么 dense + sparse 一起上（hybrid）？** 两者互补：
  - dense 擅长"语义相近但用词不同"（同义改写、口语化问题）；
  - sparse 擅长"关键词/专有名词/文号精确命中"（这正是政策检索的命门）。
  单用任一种都有明显盲区，混合召回是政策/RAG 检索的主流稳妥方案（思路对标爱搜税 `kernel/recall/hybrid.py`、
  以及你自己做的 standard 政策条款混合召回）。
- **多库并行召回 `RecallManager.recall`**：遍历 `step.kbases`，对每个库调 `HybridRecaller` 并**并行**执行、
  合并结果。每条 `Document` 用 `raw_query_from` 记下"由哪个子查询召回而来"，便于后续去重与归因。
- **召回阶段重"广"不重"准"**：这里要的是**高召回率**（别漏），精度交给后面的排序漏斗。
  所以 topk 较大（`recall_topk` 默认 50）。

---

## 6. 排序漏斗：粗排 → 精排 → 重排

召回拿到的是"又多又杂"的候选，排序漏斗负责"逐级提纯"，最终留下少而精的 top-k。

### 6.1 粗排 `CoarseRanker.rank`（多路 RRF 融合 + 去重）
**位置**：`app/core/rank/coarse_rank.py`。
- 多个子查询 / 多个库会召回**重叠**的文档，且各路分数不可直接比较。用 **RRF（Reciprocal Rank Fusion，
  倒数排名融合）** 把"多个排序列表"融合成一个：每个文档得分 = Σ 1/(k + rank)。
- **为什么用 RRF 而不是直接加分数？** RRF 只看**排名**不看绝对分值，天然规避了"不同检索器分数量纲不一致"
  的问题，简单、稳健、无需调参，是多路融合的经典做法。
- 同时按 `doc_id` **去重**（保留融合得分最高的那条），把候选压到 `topk`。

### 6.2 精排 `FineRanker.rank`（bge 语义相似）
**位置**：`app/core/rank/fine_rank.py`。
- 用 `FineRankMethod` 选择：`bge_title`（拿标题算语义相似）、`bge_content`（拿正文算）、`direct`（不精排透传）。
- **为什么粗排之后还要精排？** 粗排是基于"召回时的检索分/排名"，比较粗。精排用 embedding 对
  `query` 与候选的标题/正文重新算语义相似度，做一次更细的重排序，把语义最贴的提上来。
- **为什么按意图选 title 还是 content？** 精确法规类更看"标题/法规名"是否对上，用 `bge_title`；
  通用问答更看"正文"是否答到点，用 `bge_content`。由路由层在 `RecallStep.fine_rank_method` 指定。

### 6.3 重排 `ReRanker.rerank`（cross-encoder 交叉打分）
**位置**：`app/core/rerank/rerank.py`，调 `RerankerClient`（bge-reranker-v2-m3）。
- 把 `(query, 每条候选正文)` 成对喂给 **cross-encoder 重排模型**，得到精细的相关性分，取 top-k（`rerank_topk` 默认 5）。
- **为什么重排是"最后一锤"、最关键的一步？** 召回/精排用的是 **bi-encoder**（query 和 doc 各自独立编码再算相似度），
  速度快但精度有限；**cross-encoder** 让 query 和 doc 在模型内部充分交互，相关性判断准得多，
  但太慢、只能用于少量候选。所以工程上是"前面用 bi-encoder 快速筛到几十条 → 最后用 cross-encoder 精排到 5 条"，
  兼顾速度与精度。这正是 RAG 检索质量的分水岭。

> 漏斗一句话总结：**召回(50) 求全 → 粗排(RRF 去重) 融合 → 精排(bge) 提纯 → 重排(cross-encoder, 5) 定稿
> → 可答性门控(够不够答)**。

### 6.4 可答性门控 `answerability_check`（漏斗出口的"能不能答"判定）
**位置**：`app/graph/nodes.py::answerability_check` + `answerability_decider`（条件边），阈值在 `config/settings.py`。

- **解决什么问题？** 漏斗只负责"把召回里相对最相关的 5 条挑出来"，但**不保证这 5 条真的相关**。
  如果用户问的东西库里根本没有，漏斗照样会返回 5 条"矮子里拔将军"的弱证据；若照常喂给 LLM 生成，
  模型会"看图说话"硬凑一个**带文号、带数字、却没有依据**的答案——这是财税场景最危险的幻觉。
- **怎么判？** 重排用的 cross-encoder 已经给每条候选打了 **[0,1] 的相关性分**（sigmoid 归一）。
  取重排 top1 分作主信号，与 `answerability_min_score`（默认 0.15）比：
  - **≥ 阈值 → 判"可答"**：条件边走原链路 `build_context → generate_answer`，行为完全不变；
  - **< 阈值 / 空召回 → 判"不可答"**：转 `low_confidence_answer` 节点，**不调 LLM**、直接给一句诚实兜底话术
    （"未检索到足够相关的权威依据，建议补充文号/政策名/地域……"），从源头杜绝"没依据也硬答"。
- **为什么是确定性阈值、而不是再问一次 LLM？** 与本系统"确定性路由层"（见 [能力地图.md](能力地图.md) §1）
  一脉相承：这种高频、对延迟敏感、错一次就放幻觉出去的关键岔口，用**可单测、可复现、零额外 token** 的
  阈值判定，比把它交给又慢又不可复现的 LLM 黑盒更稳。
- **一个隐藏的工程坑（值得记）**：重排客户端用 **fp16 + sigmoid**，对**极不相关**样本（logit ≲ -17）会
  **下溢成恰好 0.0**；而重排服务**降级**（未真正打分）时分数也是默认 0.0。两者都为 0.0 就**无法区分**——
  会把"极强不相关的真低分"误判成"重排降级"而**错误放行**（fail-open，恰好与门控目标相反）。
  解法：`ReRanker` 成功打分时把结果**钳到一个极小正数** `_SCORED_FLOOR`(1e-6)，让"成功打分"恒 > 0、
  "未打分"恒 = 0.0，`0.0` 被**独占**用来表示降级，二者彻底解耦（见 `app/core/rerank/rerank.py`）。
  降级时（分数全 0）改用"候选数"这一弱信号判定，**宁放行不误杀**。
- **可观测**：门控结论以 `answerability` SSE 事件透出（`answerable / confidence / signals`），
  前端在时间线上对"判不可答"打一枚"依据不足·诚实兜底"角标（只呈现门控自身字段，**不编造任何指标**）。

> 这一环对标企业财税客服系统里的「问答库可答判断（阈值 + 大模型 + 规则）」环节——本实现取其"确定性阈值"
> 内核、并刻意**不引入 LLM 裁决**，与系统整体设计立场一致。

---

## 7. 摘要层：上下文拼装 + 带引用的流式生成

**位置**：`app/core/summarize/summarizer.py`。

- `build_context(qu, docs, topk)`：把 top-k `Document` 拼成给 LLM 的**上下文**，每条以 `[citation:序号]` 开头，
  同时产出给前端展示的 `references`（`Document.to_reference()`）。
- **为什么要按意图化拼装上下文？** 不同意图需要的"证据组合"不同（对标爱搜税 `summarize.py`）：
  稽查案例类 = 直指政策 + 若干案例 + 政策兜底；通用问答（无地域）可过滤掉地方法规只留全国性政策。
  拼对上下文，答案质量比"无脑塞 top-k"高很多。
- `summarize_stream(query, qu, context)`：用 `summarize.prompt` 让 LLM **依据片段作答**，逐 token 异步产出
  （映射成 `answer_delta` 事件流）。提示词里写死三条纪律：**只依据检索内容、不杜撰、句末标 [citation:x]**，
  并在结尾附**免责声明**。这是财税场景的底线——宁可说"资料不足无法确切回答"，也不编造文号和数字。

---

## 8. Text2SQL 分支：自然语言查数

**位置**：`app/agents/text2sql_agent.py`（LangGraph 多步），命中"经营数据查询类"时由路由触发。

四步流水线（每步都有日志，校正步带重试循环）：

1. **schema_link（模式链接）**：用问题去 **Qdrant** 召回相关"表/字段元数据"，再用 LLM
   （`text2sql_schema_link.prompt`）从候选里**裁剪**出真正相关的少数表与字段。
   - **为什么先裁剪？** 数仓表多字段多，全量 schema 塞给模型既超上下文又干扰判断、还费 token。
     先缩小到几张表，SQL 准确率明显提升。
2. **generate_sql**：基于选定 schema，用 LLM（`text2sql_generate.prompt`）生成 **只读 SELECT**。
   提示词里硬性禁止任何写操作（防注入、防误删），并要求合理 LIMIT、区间时间过滤等。
3. **validate_fix（校正）**：先做静态校验（只允许 SELECT、表/字段是否存在），执行报错时把
   **真实报错信息回灌**给 LLM（`text2sql_fix.prompt`）定向修复，**最多重试 N 次**。
   - **为什么要"报错回灌重试"？** 一次生成难免有字段名/聚合错误。把数据库的真实报错喂回去让模型定向改，
     比盲目重生成稳得多，是 Text2SQL 工程化鲁棒性的关键技巧（对标掌柜问数的 SQL 校正环节）。
4. **execute**：用 `MySQLClient`（SQLAlchemy async + aiomysql，**只读**）执行，拿到 `columns/rows`。
5. **summarize**：把数据表 + 问题给 LLM，生成一句**自然语言数据解读**（如"今年销售额整体上升，3 月最高"）。

产物是 `Text2SQLResult(question, sql, columns, rows, row_count, summary, error)`，
经 SSE 以 `sql` + `answer_delta` 事件返回前端。

---

## 9. 记忆层：多轮会话

**位置**：`app/memory/session.py`（底层 `MongoHistoryClient`，motor 异步）。
- `load(user_id, session_id)` 取最近若干轮历史；`append(...)` 落库本轮问答。
- **为什么要历史？** 多轮里用户常用代词（"它/这个/上面那条"），QU 的改写/指代消解需要历史才能补全成
  独立完整的问题（对标掌柜智库的 `rewritten_query` 思路）。

---

## 10. 客户端层：统一惰性连接 + http/local 双模

**位置**：`app/clients/`。
- LLM / Embedding / Reranker / Milvus / ES / Qdrant / MySQL / Mongo / MinIO / MinerU / MCP 各一个封装。
- **三条统一规范**：
  1. **惰性初始化**：连接对象第一次用到才建。保证缺 infra 时 `import` 不报错、FastAPI 能起、单测能跑。
  2. **http/local 双模**：embedding/reranker/mineru 可在"调远程服务"与"本地模型"间用 `*_mode` 切换；
     本地模式重依赖（FlagEmbedding/torch/magic-pdf）默认不装，按需解开。
  3. **配置驱动 + 健壮性**：地址/密钥全走 `settings`；外部调用 `try/except` + 清晰中文报错，
     网络调用配 `tenacity` 重试/超时。MCP 在 `mcp_enabled=False` 时**安全降级**（不真连）。

---

## 11. 离线脚本：把数据灌进来

**位置**：`scripts/`（均**幂等**，可重复跑）。
- `init_collections.py`：创建 Milvus collection / ES index / Qdrant collection（建库建表）。
- `ingest_documents.py`：文档解析(MinerU 槽位)→切分→embedding→写 Milvus + ES；图片→MinIO。
- `build_metadata_index.py`：读 MySQL 数仓元数据→embedding→写 Qdrant（供 Text2SQL schema linking）。

---

## 12. 一句话回顾全链路

> **理解(QU) → 路由(意图表) → 召回(dense+sparse 多库) → 粗排(RRF) → 精排(bge) → 重排(cross-encoder)
> → 可答性门控(够不够答) → 摘要(带引用)**，若是"查数据"则改走
> **Text2SQL（选 schema → 生成 SQL → 报错校正 → 执行 → 解读）**，
> 全程由 **LangGraph** 编排、**SSE** 流式吐给前端，所有外部依赖**惰性连接、配置驱动**。

把这条链路吃透，你就掌握了一套可迁移到任何垂直领域的 RAG + Text2SQL 工程范式。
