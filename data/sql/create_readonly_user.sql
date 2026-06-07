-- =====================================================================
-- create_readonly_user.sql  ——  Text2SQL 只读账号建账脚本（纵深防御第二层）
-- ---------------------------------------------------------------------
-- 在链路里的位置：
--   Text2SQL 链路的"双层只读防护"中的【DB 侧那一层】。
--   第一层：Agent 层(app/agents/text2sql_agent.py 的 validate 节点)用 sqlglot 做 AST 校验，
--           只放行 SELECT，禁止 DDL/DML。
--   第二层（本脚本）：在 MySQL 里建一个只有 SELECT 权限的只读账号 tax_ro，
--           让 Text2SQL 的 SQL 执行(app/clients/mysql_client.py 的 execute)走这个账号。
--           这样即便上层 AST 校验被绕过，数据库账号本身也无法 DDL/DML，形成纵深防御。
--
-- 用法（只需执行一次）：
--   1) 把下面的 <改我> 换成一个强密码（务必与下一步 .env 里填的密码一致）。
--   2) 在 MySQL 里用有授权权限的账号(如 root)执行本脚本：
--        mysql -h <host> -P <port> -u root -p < data/sql/create_readonly_user.sql
--      或在客户端里选中执行。
--   3) 把账号/密码填进项目根的 .env：
--        MYSQL_READONLY_USER=tax_ro
--        MYSQL_READONLY_PASSWORD=<与上面 <改我> 相同的密码>
--      填好后，Text2SQL 的 execute 会自动改用只读账号连库（settings.mysql_readonly_dsn）。
--      若 .env 留空，则 Text2SQL 仍用主账号，行为与现状完全一致（本脚本不影响默认行为）。
--
-- 说明：
--   - 库名 tax_dw 须与 .env 的 MYSQL_DB 一致；若改过库名，请同步修改下面的 GRANT ... ON tax_dw.*。
--   - 'tax_ro'@'%' 允许任意主机连接；如需收紧，可把 '%' 换成具体网段(如 '10.199.%')。
--   - 只授予 SELECT：只读账号无法 INSERT/UPDATE/DELETE/DROP 等，这正是第二层防护的核心。
-- =====================================================================

-- 1) 建只读账号（IF NOT EXISTS 幂等：已存在则不报错；首次执行即创建）。
CREATE USER IF NOT EXISTS 'tax_ro'@'%' IDENTIFIED BY '<改我>';

-- 2) 只授予数仓库 tax_dw 的 SELECT 权限（只读，无任何写/改/删/建权限）。
GRANT SELECT ON tax_dw.* TO 'tax_ro'@'%';

-- 3) 刷新权限表，使授权立即生效。
FLUSH PRIVILEGES;
