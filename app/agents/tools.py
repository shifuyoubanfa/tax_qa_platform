"""Text2SQL Agent 的可复用工具函数集合。
本模块在整体链路里的位置：被 text2sql_agent.py 的各个 LangGraph 节点调用，集中放"纯函数"
工具（无外部连接、可独立单测）：从 LLM 回复里抽 SQL、SQL 静态安全校验、把 schema/结果拼成文本。
把这些细节抽出来，节点函数才能保持"读状态->调工具->写状态"的清爽结构。

设计要点（为什么这么做）：
1. LLM 产出的 SQL 常被 ```sql ``` 代码块或解释文字包裹，extract_sql 负责"去壳取净 SQL"。
2. validate_sql_safety 做"执行前静态体检"：只允许只读 SELECT、拦截 DROP/DELETE/UPDATE 等写操作，
   这是 Text2SQL 必须的安全闸门（防止 LLM 生成破坏性语句直接打到数仓）。
   它采用"AST 白名单 + 关键词黑名单"双保险，详见函数内注释。
3. 工具全部为同步纯函数，便于 tests/ 离线测试，不依赖任何 infra。
"""
from __future__ import annotations

import re

from config.logging_config import get_logger

logger = get_logger(__name__)

# 允许执行的语句开头（只读分析场景，仅放行查询类）——仅作第二层关键词兜底用
_ALLOWED_PREFIXES = ("SELECT", "WITH")
# 明令禁止的危险关键词（出现即拦截，防止写/删/改/提权）——第二层关键词兜底用
_FORBIDDEN_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "REPLACE", "GRANT", "REVOKE", "MERGE", "CALL", "EXEC",
)

# 禁止访问的系统库（提权/信息泄露的常见入口；库名小写后比对）
_FORBIDDEN_SCHEMAS = frozenset({
    "information_schema",  # 元数据库：可枚举所有库表结构
    "mysql",              # 系统库：含 user 表等账号/权限信息
    "performance_schema",  # 性能库：可探测服务内部状态
    "sys",                # sys 库：performance_schema 的视图封装
})
# 禁止调用的危险函数（耗时/读写文件/执行系统命令/网络外联/提权/盲注；函数名小写后比对）
#   下面按"危害类别"分组列出。判定口径：凡是能让 SQL 越过"纯读数仓"边界、去碰
#   操作系统/文件系统/网络/服务内部状态，或被用于时间盲注/报错注入的函数，一律拦掉。
#   注意：sqlglot 把这些非内建函数解析成 Anonymous 节点，.name 即裸函数名（如
#   master.dbo.xp_cmdshell 也会在树里暴露出内层 Anonymous name='xp_cmdshell'），
#   故只需按"裸函数名"小写比对即可同时覆盖带库名/schema 限定的调用形式。
_FORBIDDEN_FUNCS = frozenset({
    # —— 耗时 / 时间盲注 / DoS ——
    "sleep", "benchmark", "pg_sleep", "pg_sleep_for", "pg_sleep_until",
    "waitfor", "dbms_lock", "dbms_pipe", "get_lock",
    # —— 直接读/写服务器本地文件（等于任意文件读写）——
    "load_file", "loadfile",
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "lo_import", "lo_export",          # PostgreSQL 大对象读/写本地文件
    "utl_file",                        # Oracle 文件读写包
    # —— 执行操作系统命令（最高危：RCE）——
    "xp_cmdshell",                     # SQL Server 执行 shell 命令
    "sys_eval", "sys_exec",            # MySQL lib_mysqludf_sys UDF 执行命令
    "shell", "system",                 # 部分引擎/UDF 的系统命令入口
    # —— 网络外联 / SSRF / 带外数据渗出 ——
    "utl_inaddr", "get_host_address",  # Oracle DNS 解析（常用于带外渗出/SSRF）
    "utl_http", "httpget", "http_get", "inet_aton",  # 发起网络请求/外联（保守拦截）
    "dblink", "dblink_connect",        # PostgreSQL 跨库/外部连接
    # —— 加载扩展 / 提权 ——
    "load_extension",                  # SQLite 加载本地动态库（可致 RCE）
    # —— MySQL 报错注入常用函数（用于把敏感信息塞进报错回显）——
    "extractvalue", "updatexml",
})

# sqlglot 惰性导入句柄：模块导入时先置 None，首次校验时再真正 import，
# 这样"缺包"不会拖垮整个模块的导入（其它纯函数仍可用），仅在用到时给出清晰中文报错。
_sqlglot = None


def _load_sqlglot():
    """惰性加载 sqlglot 解析库（缺包时抛出清晰中文 ImportError）。

    为什么惰性导入：本模块还提供 extract_sql / format_schema_context 等不依赖 sqlglot
    的纯函数；若在文件顶部 import sqlglot，一旦环境没装就会让整个模块 import 失败，
    连带拖垮无关功能。改为"首次校验时才导入"，把依赖影响范围收敛到最小。

    :return: 已导入的 sqlglot 模块对象。
    :raises ImportError: 未安装 sqlglot 时，附带安装指引的中文提示。
    """
    global _sqlglot
    if _sqlglot is None:
        try:
            import sqlglot  # 局部导入：仅在真正需要 AST 校验时才触发
        except ImportError as exc:  # pragma: no cover - 取决于运行环境是否装包
            raise ImportError(
                "缺少 sqlglot 依赖：validate_sql_safety 需用它做 SQL AST 白名单校验。"
                "请先安装：pip install sqlglot（已在 requirements.txt 中声明）。"
            ) from exc
        _sqlglot = sqlglot
    return _sqlglot


def extract_sql(llm_text: str) -> str:
    """从 LLM 的自由文本回复里抽取出纯 SQL 语句。

    处理三种常见形态：
    1. 被 ```sql ... ``` 或 ``` ... ``` 代码块包裹；
    2. 前后带自然语言解释；
    3. 末尾多余分号/空白。

    :param llm_text: LLM 原始回复文本。
    :return: 去壳后的单条 SQL（去掉首尾空白与多余分号）；抽不到时返回空字符串。
    """
    if not llm_text:
        return ""
    text = llm_text.strip()

    # 1. 优先抓 ```sql ... ``` / ``` ... ``` 代码块里的内容
    fence = re.search(r"```(?:sql)?\s*(.+?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # 2. 从第一个 SELECT/WITH 起截取（丢掉前面的解释性文字）
    m = re.search(r"(?is)\b(select|with)\b", text)
    if m:
        text = text[m.start():].strip()

    # 3. 去掉结尾分号与多余空白，便于后续统一加 LIMIT 等处理
    text = text.rstrip().rstrip(";").strip()
    return text


def validate_sql_safety(sql: str) -> tuple[bool, str]:
    """对 SQL 做"执行前"静态安全校验（只允许只读查询）。

    双保险设计（为什么不只用关键词黑名单）：
    -------------------------------------------------------------------
    第一层【AST 白名单，主力】：用 sqlglot 把 SQL 解析成抽象语法树（AST），
      只放行"顶层是 SELECT / 集合查询(UNION 等)"的只读语句，再遍历整棵树拦截危险节点。
      为什么 AST 比"大写关键词正则黑名单"可靠得多：
        1) 大小写绕过：黑名单要把 SQL 转大写再匹配，攻击者可用 `DrOp`、全角等花样；
           AST 由解析器规范化节点类型，`select`/`SELECT`/`SeLeCt` 都是同一个 Select 节点。
        2) 注释绕过：`SELECT 1 /*!50000 ; DROP TABLE t */` 这类"内联注释藏语句"或
           `-- xxx\n; DROP ...`，正则很难穷举；而 sqlglot 解析时会忽略注释、按真实语法
           拆出多条语句，多语句一票否决，注释里的伪装失效。
        3) INTO OUTFILE / DUMPFILE：`SELECT ... INTO OUTFILE '/x'` 是写磁盘的危险操作，
           但整句以 SELECT 开头、又不含黑名单里的写关键词，正则会漏放；AST 里它是独立的
           Into 节点（或在本方言下直接解析失败），可被精确识别。
        4) 子查询里的写操作、系统库访问(information_schema/mysql)、耗时函数(SLEEP/BENCHMARK)、
           读文件函数(LOAD_FILE) —— 都能在树里逐节点精确命中，而非靠字符串碰运气。
    第二层【关键词黑名单，兜底】：保留原有的大写关键词扫描与多语句检查。万一某条 SQL
      恰好是 sqlglot 不认识/解析放行的边角语法，这层仍能挡住明显的写/DDL 关键词，双保险。

    :param sql: 待校验的 SQL。
    :return: (是否通过, 失败原因)；通过时原因为空字符串。函数签名保持 (sql)->(bool,str) 不变。
    """
    if not sql or not sql.strip():
        return False, "SQL 为空"

    # ---------------- 第一层：sqlglot AST 白名单校验（主力） ----------------
    ok, reason = _validate_by_ast(sql)
    if not ok:
        return False, reason

    # ---------------- 第二层：关键词黑名单兜底（双保险） ----------------
    return _validate_by_keywords(sql)


def _validate_by_ast(sql: str) -> tuple[bool, str]:
    """用 sqlglot 把 SQL 解析成 AST 做白名单校验（只放行只读查询）。

    :param sql: 待校验 SQL。
    :return: (是否通过, 失败原因)。
    """
    sqlglot = _load_sqlglot()
    from sqlglot import exp  # 局部导入表达式节点类型，避免顶部硬依赖
    from sqlglot.errors import SqlglotError  # 解析/分词类错误的统一基类（跨版本稳定）

    # ① 解析：用 mysql 方言（本平台数仓为 MySQL 兼容）。解析失败=语法非法/被刻意构造，直接拒。
    #    注意 INTO OUTFILE 等在本方言下会直接抛 ParseError（SqlglotError 子类），这里一并兜住。
    try:
        statements = sqlglot.parse(sql, read="mysql")
    except SqlglotError as exc:
        return False, f"SQL 解析失败（疑似语法非法或恶意构造）：{exc}"
    except Exception as exc:  # 解析器内部异常也按"不可信"处理，宁可错杀
        return False, f"SQL 解析异常，拒绝执行：{exc}"

    # ② 过滤掉解析出的空语句（空串/纯注释会解析成 None）
    statements = [s for s in statements if s is not None]
    if not statements:
        return False, "SQL 为空或仅含注释，无可执行的查询"

    # ③ 多语句拦截：parse 返回 >1 条说明用 ; 串接了多条（注释绕过的多语句也在此被拆出来）
    if len(statements) > 1:
        return False, "检测到多条语句，仅允许执行单条只读查询"

    top = statements[0]

    # ④ 顶层必须是只读查询：SELECT（含 WITH...SELECT，sqlglot 里 WITH 查询顶层仍是 Select）
    #    或集合查询 UNION/EXCEPT/INTERSECT（SetOperation，同样只读）。其余（Insert/Update/
    #    Delete/Drop/Create/Truncate/...）一律拒绝。
    if not isinstance(top, (exp.Select, exp.SetOperation)):
        return False, f"仅允许只读查询（SELECT/UNION），拒绝执行 {type(top).__name__} 语句"

    # ⑤ 遍历整棵 AST，逐节点拦截危险构造（含所有子查询/嵌套层级）
    for node in top.walk():
        # 5.1 SELECT ... INTO（OUTFILE/DUMPFILE/@变量）：写磁盘或导出，非纯读，拒。
        if isinstance(node, exp.Into):
            return False, "检测到 INTO OUTFILE/DUMPFILE/变量 导出操作，拒绝执行（仅允许纯查询）"

        # 5.2 子查询/CTE 里夹带写操作或 DDL（防御 sqlglot 容忍的边角写语法）
        if isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Drop,
                             exp.Create, exp.Alter, exp.TruncateTable, exp.Command)):
            return False, f"检测到嵌套的写/DDL 操作：{type(node).__name__}，拒绝执行"

        # 5.3 系统库访问：information_schema / mysql / performance_schema / sys
        if isinstance(node, exp.Table) and node.db and node.db.lower() in _FORBIDDEN_SCHEMAS:
            return False, f"检测到对系统库 {node.db} 的访问，拒绝执行（防信息泄露/提权）"

        # 5.4 危险函数：SLEEP/BENCHMARK(耗时DoS/盲注)、LOAD_FILE(任意文件读)等。
        #     sqlglot 把这些不内建的函数解析成 Anonymous 节点，.name 保留原始大小写，
        #     这里统一小写后比对——大小写绕过(sLeEp)在此失效。
        if isinstance(node, exp.Anonymous):
            fname = (node.name or "").lower()
            if fname in _FORBIDDEN_FUNCS:
                return False, f"检测到禁止调用的危险函数：{node.name}，拒绝执行"

    return True, ""


def _validate_by_keywords(sql: str) -> tuple[bool, str]:
    """第二层兜底：保留原有的大写关键词黑名单 + 多语句检查（双保险）。

    :param sql: 待校验 SQL（已过 AST 校验）。
    :return: (是否通过, 失败原因)。
    """
    upper = sql.strip().upper()

    # 必须是查询类语句开头
    if not upper.startswith(_ALLOWED_PREFIXES):
        return False, "仅允许 SELECT/WITH 查询语句，拒绝执行非只读语句"

    # 含危险关键词直接拦截（用单词边界匹配，避免误伤列名如 created_at）
    for kw in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            return False, f"检测到禁止的写/DDL 关键词：{kw}"

    # 多语句拦截（防止用 ; 串接第二条语句注入）
    # 去掉结尾分号后，正文中不应再出现分号
    if ";" in sql.strip().rstrip(";"):
        return False, "检测到多条语句（包含分号分隔），仅允许单条查询"

    return True, ""


def ensure_limit(sql: str, default_limit: int = 200) -> str:
    """给没有 LIMIT 的查询补一个默认上限，防止误拉全表。

    :param sql: 已通过安全校验的 SELECT/WITH 查询。
    :param default_limit: 默认行数上限。
    :return: 末尾带 LIMIT 的 SQL（已有 LIMIT 则原样返回）。
    """
    if re.search(r"(?is)\blimit\b\s+\d+", sql):
        return sql
    return f"{sql.rstrip().rstrip(';')} LIMIT {default_limit}"


def format_schema_context(schema_items: list[dict]) -> str:
    """把 schema linking 选出的"表/字段/指标"元数据拼成喂 LLM 的文本上下文。

    兼容两种 payload 形状（这是本函数的关键）：
    1. "每表一条、列信息嵌在 columns 里"（build_metadata_index 写入 Qdrant 的真实形状）：
        {"table": "fact_order", "table_comment"/"description": "订单事实表",
         "columns": [{"column": "order_amount", "type": "float", "nullable": ...,
                      "comment"/"column_comment": "订单金额"}, ...]}
       —— 此时要遍历 columns，逐字段输出一行，否则字段级信息永远进不了 LLM 提示词。
    2. "每字段一条"的扁平形状（历史/单测兼容）：
        {"table": "fact_order", "column": "order_amount", "type": "float",
         "description": "订单金额", "role": "measure"}（字段可缺省）。

    :param schema_items: 元数据条目列表（形状见上，两种皆可混用）。
    :return: 多行文本，每行描述一个表/字段/指标，供 generate_sql 的 prompt 注入。
    """
    if not schema_items:
        return "（未召回到相关表/字段元数据）"
    lines: list[str] = []
    for item in schema_items:
        table = item.get("table", "")
        columns = item.get("columns")
        # 形状1：嵌套结构——每表一条，列信息在 columns: list[dict] 里
        if isinstance(columns, list) and columns:
            # 表注释键兼容 table_comment / description（不同写入方命名不一）
            table_desc = item.get("table_comment", "") or item.get("description", "")
            # 先输出表级标题，再逐列输出字段行
            lines.append(f"# 表 {table}: {table_desc}")
            for col in columns:
                if not isinstance(col, dict):
                    # 防御：columns 里混入非字典元素时跳过，避免整段渲染崩掉
                    continue
                # 列名键兼容 column；类型键兼容 type；注释键兼容 comment / column_comment
                col_name = col.get("column", "")
                col_type = col.get("type", "")
                col_desc = col.get("comment", "") or col.get("column_comment", "")
                # 字段级：- 表名.列名 (类型): 注释
                lines.append(f"- {table}.{col_name} ({col_type}): {col_desc}")
            continue
        # 形状2：扁平结构——每字段一条（保留原有渲染分支，向后兼容）
        column = item.get("column", "")
        ctype = item.get("type", "")
        role = item.get("role", "")
        desc = item.get("description", "")
        if column:
            # 字段级：表.字段 (类型, 角色) - 描述
            lines.append(f"- {table}.{column} ({ctype}{'/' + role if role else ''}): {desc}")
        else:
            # 表级：表 - 描述
            lines.append(f"# 表 {table}: {desc}")
    return "\n".join(lines)


def rows_to_markdown(columns: list[str], rows: list[list], max_rows: int = 20) -> str:
    """把查询结果（列名 + 行）渲染成简洁的 Markdown 表格文本（供 summarize 节点喂 LLM）。

    :param columns: 列名列表。
    :param rows: 数据行列表（每行是与 columns 等长的值列表）。
    :param max_rows: 最多渲染多少行，避免结果过大撑爆 prompt。
    :return: Markdown 表格字符串；无数据时返回提示文本。
    """
    if not columns:
        return "（查询无结果列）"
    if not rows:
        return "（查询结果为空）"
    header = "| " + " | ".join(str(c) for c in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body_lines = []
    for r in rows[:max_rows]:
        body_lines.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    table = "\n".join([header, sep, *body_lines])
    if len(rows) > max_rows:
        table += f"\n（仅展示前 {max_rows} 行，共 {len(rows)} 行）"
    return table


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：覆盖抽取 + 安全校验的典型场景。
    raw = "好的，以下是查询：\n```sql\nSELECT SUM(order_amount) FROM fact_order;\n```"
    sql = extract_sql(raw)
    print("[tools 自测] 抽取SQL =>", sql)
    print("[tools 自测] 安全校验(应True) =>", validate_sql_safety(sql))
    # 下面这些都应被 AST 白名单拦下（演示注释/大小写/INTO OUTFILE/系统库/耗时函数等绕过都失效）
    print("[tools 自测] DROP(应False) =>", validate_sql_safety("DROP TABLE fact_order"))
    print("[tools 自测] 多语句(应False) =>",
          validate_sql_safety("SELECT 1 -- x\n; DROP TABLE fact_order"))
    print("[tools 自测] INTO OUTFILE(应False) =>",
          validate_sql_safety("SELECT * FROM fact_order INTO OUTFILE '/tmp/x'"))
    print("[tools 自测] 系统库(应False) =>",
          validate_sql_safety("SELECT * FROM mysql.user"))
    print("[tools 自测] 大小写耗时函数(应False) =>",
          validate_sql_safety("SELECT SlEeP(5)"))
    print("[tools 自测] 读文件函数(应False) =>",
          validate_sql_safety("SELECT LOAD_FILE('/etc/passwd')"))
    print("[tools 自测] 补LIMIT =>", ensure_limit(sql))
