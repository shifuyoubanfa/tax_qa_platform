# data/ 数据目录说明（本平台用到的所有数据类型 + 准备与灌库步骤）

本目录是平台的"离线数据入口"：所有要进检索/数仓的原料都放这里，再由 `scripts/` 下的脚本灌进
在线存储（Milvus / ES / Qdrant / MinIO / MySQL）。在线问答服务（`app/`）只**读**这些存储。

> 说明：本平台是"应用端"。仓库里给的是**学习演示数据**（自洽假数据，非真实业务/密钥），
> 用于把端到端链路跑通；正式数据由上游（政策爬取、数仓、文档库）提供，按同样格式替换即可。

## 目录结构总览

```text
data/
├── README.md                本说明文件
├── documents/               待入库的政策/文档原文（md / txt / pdf），由 ingest_documents.py 读取
│   ├── *.md / *.txt          纯文本（无需 MinerU，可直接切分入库，适合先跑通链路）★仓库已含 14 篇演示政策
│   ├── *.pdf                 原始 PDF（走 pypdf 本地抽取或 MinerU；图片走 MinIO）
│   └── pdf/                  ★仓库已含 14 篇预生成演示 PDF（由上面的 md 渲染而来，验证 PDF 入库用）
├── qa/                      历史问答库原料（入库时 --kbase qa）
│   └── qa_pairs.jsonl        每行一条 {question, answer, tags}  ★仓库已含 20 条
├── metadata/                数仓表/字段元数据（Text2SQL schema linking 用）
│   └── schema_meta.json      [{table, table_comment, column, column_comment, sample}, ...]  ★仓库已含
└── sql/                     数仓建库脚本（灌进 MySQL）
    └── seed_tax_dw.sql       建表 + 插数（经营星型表 + 社保/编码结构化源）  ★仓库已含
```

> `documents/pdf/` 已含 14 篇预生成 PDF；（运行时）`logs/` 由程序自动创建，无需手动建。
> `logs/` 在项目根目录下，不在 data/ 内。

---

## 平台用到的所有数据类型（逐类说明）

### 1. 政策/文档（`documents/`）—— 走 `ingest_documents.py` → Milvus + ES（+ MinIO 图片）
检索侧的"非结构化知识库"。支持三种格式：
- **md / txt**：纯文本，直接读取→切分→向量化入库，无需任何外部解析服务，**最适合先跑通链路**。
  仓库已放 14 篇演示政策（小规模增值税、研发加计扣除、社保费率、虚开案例等）。
- **pdf**：默认走 `pypdf` 本地抽纯文本即可入库；想要版面/OCR/图片，把 `.env` 的 `MINERU_MODE` 切到
  `http` 接 MinerU 服务，解析出的图片会写入 MinIO（对象键回填到 `Document.image_keys`）。
- 入库后落点：稠密向量→Milvus，全文→ES；按 `--kbase` 标记来源（policy / doc / inspect_case 等）。

> 仓库已含 14 篇**预生成的中文 PDF**（在 `documents/pdf/`，由演示 md 渲染而来），可直接用于验证 PDF 入库链路。

### 2. 历史问答（`qa/qa_pairs.jsonl`）—— 走 `ingest_documents.py --kbase qa` → Milvus + ES
作为"历史问答库"召回通道。JSONL，每行一条，字段：
```json
{"question": "增值税小规模纳税人月销售额多少以内可以免征增值税？", "answer": "……", "tags": ["增值税", "小规模纳税人", "免税"]}
```
- `question`：用户历史问法（入库可作 title）；`answer`：标准答复（作 content）；`tags`：标签数组（便于过滤/展示）。
- 最简入库方式：直接用 `ingest_documents.py --kbase qa`（解析层把 question 作 title、answer 作 content）。

### 3. 数仓元数据（`metadata/schema_meta.json`）—— 走 `build_metadata_index.py` → Qdrant
供 Text2SQL 的 **schema linking**：把 `table_comment + column_comment` 向量化写入 Qdrant，
生成 SQL 时据用户问题召回相关表/字段。JSON 数组，每条：
```json
{"table": "fact_sales_order", "table_comment": "销售订单事实表……", "column": "sales_amount", "column_comment": "销售额(不含税,元)", "sample": "85600.00"}
```
- 覆盖经营数据星型模型：`fact_sales_order` + `dim_date / dim_region / dim_product / dim_customer`，
  另含结构化源表 `dim_social_security / dim_tax_product_code`，字段释义与 `seed_tax_dw.sql` 完全一致。
> 注意：正式构建时 `build_metadata_index.py` 可**直接从 MySQL 实时读取**表/字段元数据，
> 本 JSON 仅供离线参考/无库时使用。

### 4. 数仓数据（`sql/seed_tax_dw.sql`）—— 直接执行进 MySQL（库 `tax_dw`）
Text2SQL 与结构化查表 Agent 真正查询的关系库。脚本含建表 + 插数：
- **经营数据（星型模型，供 Text2SQL）**：
  `fact_sales_order`（20 行，覆盖 2024 全年 12 月）+ `dim_date`/`dim_region`/`dim_product`/`dim_customer`。
  可跑通「各月销售额」「各省销售额排名」「各品类销售额」等典型自然语言查询。
- **结构化源（供 `app/agents/structured_agents.py` 精确查表）**：
  - `dim_social_security`：列名 **`region` / `year`** 与 `ShebaoAgent` 常量一致；含 **杭州/北京 2024** 等行。
  - `dim_tax_product_code`：列名 **`product_name`** 与 `ProductCodeAgent` 常量一致；含 **天然气 / 建筑服务** 等行。

### 5. 图片（MinIO，仅含图 PDF 才有）
PDF 经 MinerU（http 模式）解析出的图片，由 `ingest_documents.py` 写入 MinIO，
对象键约定 `{文档id}/img_{序号}.png`，回填到 `Document.image_keys`，前端凭此取图。
纯 md/txt 与本地 pypdf 抽取无图片，不涉及 MinIO。

---

## 脚本一览（`scripts/`）

| 脚本 | 作用 | 产出落点 |
| --- | --- | --- |
| `init_collections.py` | 建 Milvus 集合 / ES 索引 / Qdrant 集合（幂等） | Milvus / ES / Qdrant |
| `ingest_documents.py` | 文档/问答：解析→切分→向量化→入库 | Milvus + ES（图片→MinIO） |
| `build_metadata_index.py` | 数仓元数据→向量化 | Qdrant |
| `seed_tax_dw.sql`（非脚本，SQL 文件） | 建表+插数，喂 Text2SQL/结构化 Agent | MySQL 库 `tax_dw` |

> 注：示例 PDF（`documents/pdf/` 下 14 篇）已**预生成并随仓库提供**，生成脚本已移除——直接用于验证 PDF 入库即可。

---

## 从零准备数据并灌库（命令顺序）

```bash
# 0) 建在线存储的集合/索引（幂等，跑一次）
py scripts/init_collections.py

# 1)（可选）验证 PDF 入库：仓库已含预生成 PDF（data/documents/pdf/），可单独干跑看切分
py scripts/ingest_documents.py --input-dir data/documents/pdf --dry-run

# 2) 文档入库（先干跑看切分质量，再正式入库）
py scripts/ingest_documents.py --input-dir data/documents --dry-run
py scripts/ingest_documents.py --input-dir data/documents --kbase doc

# 3) 历史问答入库（历史问答库通道）
py scripts/ingest_documents.py --input-dir data/qa --kbase qa

# 4) 数仓建库灌数（MySQL，库 tax_dw；供 Text2SQL 与结构化查表 Agent）
mysql -h <host> -u <user> -p < data/sql/seed_tax_dw.sql

# 5) 数仓元数据 → Qdrant（Text2SQL schema linking；可读 schema_meta.json 或从 MySQL 实时读）
py scripts/build_metadata_index.py --dry-run
py scripts/build_metadata_index.py
```

数据流总览：
```text
原始资料(data/)
   │  documents/  ──ingest_documents.py──▶ Milvus(稠密) + ES(全文)   ; 含图PDF的图片 ▶ MinIO
   │  qa/         ──ingest_documents.py──▶ Milvus + ES (--kbase qa)
   │  sql/        ──mysql 执行───────────▶ MySQL(tax_dw)  ◀── Text2SQL / 结构化Agent 查询
   └  metadata/   ──build_metadata_index─▶ Qdrant(表/字段向量, Text2SQL schema linking)
                                  ▲
                          init_collections.py 先把上面这些集合/索引建好
```

## 运行时产物（不要手动塞数据）
- `logs/`：日志目录（程序自动创建，按 20MB×5 滚动），在项目根目录下。
- Milvus / ES / Qdrant / MinIO / Mongo / MySQL：均为外部服务，数据存各自服务里，不落本目录。
