-- =====================================================================
-- seed_tax_dw.sql  ——  税务问答平台 演示数仓建库脚本（可在 MySQL 直接执行）
-- ---------------------------------------------------------------------
-- 在链路里的位置：
--   1) 经营数据星型模型(fact_sales_order + dim_date/region/product/customer)
--      供 Text2SQL Agent(app/agents/text2sql_agent.py)把"各月销售额/各省排名/各品类"
--      之类的自然语言问题翻译成 SQL 并执行。
--   2) 结构化源两张维表(dim_social_security / dim_tax_product_code)
--      供 结构化查表 Agent(app/agents/structured_agents.py)按 地域+年度 / 商品名 精确查表。
--      ★ 列名必须与 structured_agents.py 顶部常量一致：region / year / product_name。
--
-- 字段口径与 data/metadata/schema_meta.json 完全对应（Text2SQL schema linking 用同一套释义）。
-- 所有数据均为学习演示用的自洽假数据，便于跑通端到端链路；非真实业务数据。
--
-- 用法：
--   mysql -h <host> -u <user> -p < data/sql/seed_tax_dw.sql
-- 或在客户端里选中执行。库名 tax_dw 若与 .env 的 MYSQL_DATABASE 不一致，请同步修改。
-- =====================================================================

CREATE DATABASE IF NOT EXISTS tax_dw DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
USE tax_dw;

-- 幂等：先删后建，方便反复灌库（演示环境用，生产勿直接 DROP）。
DROP TABLE IF EXISTS fact_sales_order;
DROP TABLE IF EXISTS dim_date;
DROP TABLE IF EXISTS dim_region;
DROP TABLE IF EXISTS dim_product;
DROP TABLE IF EXISTS dim_customer;
DROP TABLE IF EXISTS dim_social_security;
DROP TABLE IF EXISTS dim_tax_product_code;

-- =====================================================================
-- 一、经营数据：维度表
-- =====================================================================

-- 日期维度：按年/季/月/年月标签做时间汇总
CREATE TABLE dim_date (
  date_key    INT          NOT NULL COMMENT '日期主键(yyyymmdd 整数)',
  full_date   DATE         NOT NULL COMMENT '完整日期',
  year        INT          NOT NULL COMMENT '年份',
  quarter     TINYINT      NOT NULL COMMENT '季度(1-4)',
  month       TINYINT      NOT NULL COMMENT '月份(1-12)',
  year_month_label  VARCHAR(7)   NOT NULL COMMENT '年月标签(yyyy-MM)',
  PRIMARY KEY (date_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='日期维度表';

-- 地区维度：省/市/大区，做各省排名与区域对比
CREATE TABLE dim_region (
  region_key  INT          NOT NULL COMMENT '地区主键',
  province    VARCHAR(32)  NOT NULL COMMENT '省份名称',
  city        VARCHAR(32)  NOT NULL COMMENT '城市名称',
  area        VARCHAR(16)  NOT NULL COMMENT '所属大区',
  PRIMARY KEY (region_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='地区维度表';

-- 产品维度：产品/品类/品牌/单价，做各品类销售统计
CREATE TABLE dim_product (
  product_key  INT          NOT NULL COMMENT '产品主键',
  product_name VARCHAR(64)  NOT NULL COMMENT '产品名称',
  category     VARCHAR(32)  NOT NULL COMMENT '产品品类',
  brand        VARCHAR(32)  NOT NULL COMMENT '品牌',
  unit_price   DECIMAL(12,2) NOT NULL COMMENT '单价(不含税,元)',
  PRIMARY KEY (product_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='产品维度表';

-- 客户维度：客户/行业/类型/纳税人身份，做客户与行业分析
CREATE TABLE dim_customer (
  customer_key  INT          NOT NULL COMMENT '客户主键',
  customer_name VARCHAR(64)  NOT NULL COMMENT '客户名称',
  industry      VARCHAR(32)  NOT NULL COMMENT '所属行业',
  customer_type VARCHAR(16)  NOT NULL COMMENT '客户类型(企业/个体工商户)',
  taxpayer_type VARCHAR(16)  NOT NULL COMMENT '纳税人身份(一般纳税人/小规模纳税人)',
  PRIMARY KEY (customer_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客户维度表';

-- =====================================================================
-- 二、经营数据：事实表
-- =====================================================================
CREATE TABLE fact_sales_order (
  order_id     INT           NOT NULL COMMENT '订单ID(主键)',
  date_key     INT           NOT NULL COMMENT '日期外键 -> dim_date.date_key',
  region_key   INT           NOT NULL COMMENT '地区外键 -> dim_region.region_key',
  product_key  INT           NOT NULL COMMENT '产品外键 -> dim_product.product_key',
  customer_key INT           NOT NULL COMMENT '客户外键 -> dim_customer.customer_key',
  quantity     INT           NOT NULL COMMENT '销售数量',
  sales_amount DECIMAL(14,2) NOT NULL COMMENT '销售额(不含税,元)',
  tax_amount   DECIMAL(14,2) NOT NULL COMMENT '税额(元)',
  total_amount DECIMAL(14,2) NOT NULL COMMENT '价税合计(元)=sales_amount+tax_amount',
  PRIMARY KEY (order_id),
  KEY idx_date (date_key),
  KEY idx_region (region_key),
  KEY idx_product (product_key),
  KEY idx_customer (customer_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='销售订单事实表';

-- =====================================================================
-- 三、结构化源维表（供结构化查表 Agent；列名对齐 structured_agents.py 常量）
-- =====================================================================

-- 社保缴费维度（region/year 与 ShebaoAgent 的 _SHEBAO_REGION_COL/_SHEBAO_YEAR_COL 一致）
CREATE TABLE dim_social_security (
  region                     VARCHAR(32)  NOT NULL COMMENT '地区(城市名,如 杭州/北京)',
  year                       INT          NOT NULL COMMENT '年度',
  base_lower                 DECIMAL(10,2) NOT NULL COMMENT '缴费基数下限(元/月)',
  base_upper                 DECIMAL(10,2) NOT NULL COMMENT '缴费基数上限(元/月)',
  pension_company_rate       DECIMAL(5,4) NOT NULL COMMENT '养老-单位比例',
  pension_personal_rate      DECIMAL(5,4) NOT NULL COMMENT '养老-个人比例',
  medical_company_rate       DECIMAL(5,4) NOT NULL COMMENT '医疗(含生育)-单位比例',
  medical_personal_rate      DECIMAL(5,4) NOT NULL COMMENT '医疗-个人比例',
  unemployment_company_rate  DECIMAL(5,4) NOT NULL COMMENT '失业-单位比例',
  unemployment_personal_rate DECIMAL(5,4) NOT NULL COMMENT '失业-个人比例',
  injury_company_rate        DECIMAL(5,4) NOT NULL COMMENT '工伤-单位比例(个人不缴)',
  PRIMARY KEY (region, year)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='社保缴费维度表(结构化源)';

-- 商品税收分类编码维度（product_name 与 ProductCodeAgent 的 _PRODUCT_NAME_COL 一致）
CREATE TABLE dim_tax_product_code (
  product_name VARCHAR(64)  NOT NULL COMMENT '商品/服务名称',
  code         VARCHAR(32)  NOT NULL COMMENT '税收分类编码(19位)',
  category     VARCHAR(64)  NOT NULL COMMENT '所属类别',
  rate         DECIMAL(5,4) NOT NULL COMMENT '适用增值税税率',
  PRIMARY KEY (product_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='商品税收分类编码维度表(结构化源)';

-- =====================================================================
-- 四、插入演示数据
-- =====================================================================

-- 日期维度：2024 全年 12 个月各取一天（按月统计够用）
INSERT INTO dim_date (date_key, full_date, year, quarter, month, year_month_label) VALUES
(20240115, '2024-01-15', 2024, 1,  1, '2024-01'),
(20240218, '2024-02-18', 2024, 1,  2, '2024-02'),
(20240312, '2024-03-12', 2024, 1,  3, '2024-03'),
(20240410, '2024-04-10', 2024, 2,  4, '2024-04'),
(20240520, '2024-05-20', 2024, 2,  5, '2024-05'),
(20240615, '2024-06-15', 2024, 2,  6, '2024-06'),
(20240708, '2024-07-08', 2024, 3,  7, '2024-07'),
(20240819, '2024-08-19', 2024, 3,  8, '2024-08'),
(20240916, '2024-09-16', 2024, 3,  9, '2024-09'),
(20241011, '2024-10-11', 2024, 4, 10, '2024-10'),
(20241114, '2024-11-14', 2024, 4, 11, '2024-11'),
(20241220, '2024-12-20', 2024, 4, 12, '2024-12');

-- 地区维度：6 个省/市，覆盖多个大区（支持各省排名）
INSERT INTO dim_region (region_key, province, city, area) VALUES
(1, '浙江省',   '杭州市', '华东'),
(2, '北京市',   '北京市', '华北'),
(3, '广东省',   '深圳市', '华南'),
(4, '江苏省',   '南京市', '华东'),
(5, '四川省',   '成都市', '西南'),
(6, '湖北省',   '武汉市', '华中');

-- 产品维度：覆盖 3 个品类（支持各品类销售统计）
INSERT INTO dim_product (product_key, product_name, category, brand, unit_price) VALUES
(1, '笔记本电脑', '电子产品', '联想',   5800.00),
(2, '智能手机',   '电子产品', '华为',   4200.00),
(3, '办公桌',     '办公用品', '震旦',    980.00),
(4, '办公椅',     '办公用品', '永艺',    560.00),
(5, '空气净化器', '家居',     '小米',   1200.00),
(6, '台式服务器', '电子产品', '浪潮',  18800.00);

-- 客户维度：覆盖多行业、两种纳税人身份
INSERT INTO dim_customer (customer_key, customer_name, industry, customer_type, taxpayer_type) VALUES
(1, '杭州智云科技有限公司',   '信息技术', '企业',     '一般纳税人'),
(2, '北京远大制造集团',       '制造业',   '企业',     '一般纳税人'),
(3, '深圳鹏程贸易有限公司',   '批发零售', '企业',     '一般纳税人'),
(4, '南京方圆咨询事务所',     '商务服务', '企业',     '小规模纳税人'),
(5, '成都好味餐饮个体店',     '住宿餐饮', '个体工商户', '小规模纳税人');

-- 事实表：20 行，覆盖 12 个月、6 个省、6 个产品、5 个客户
-- 约定 tax_amount=round(sales_amount*0.13,2)，total=sales+tax，便于核对。
INSERT INTO fact_sales_order
(order_id, date_key, region_key, product_key, customer_key, quantity, sales_amount, tax_amount, total_amount) VALUES
(1001, 20240115, 1, 1, 1,  20, 116000.00, 15080.00, 131080.00),
(1002, 20240115, 2, 6, 2,   3,  56400.00,  7332.00,  63732.00),
(1003, 20240218, 3, 2, 3,  30, 126000.00, 16380.00, 142380.00),
(1004, 20240218, 1, 3, 4,  50,  49000.00,  6370.00,  55370.00),
(1005, 20240312, 4, 1, 1,  10,  58000.00,  7540.00,  65540.00),
(1006, 20240312, 5, 5, 5,  40,  48000.00,  6240.00,  54240.00),
(1007, 20240410, 2, 2, 2,  25, 105000.00, 13650.00, 118650.00),
(1008, 20240410, 6, 4, 3, 100,  56000.00,  7280.00,  63280.00),
(1009, 20240520, 1, 6, 1,   5,  94000.00, 12220.00, 106220.00),
(1010, 20240520, 3, 1, 4,  15,  87000.00, 11310.00,  98310.00),
(1011, 20240615, 4, 5, 5,  60,  72000.00,  9360.00,  81360.00),
(1012, 20240708, 2, 3, 2,  80,  78400.00, 10192.00,  88592.00),
(1013, 20240708, 1, 2, 1,  20,  84000.00, 10920.00,  94920.00),
(1014, 20240819, 5, 4, 3, 120,  67200.00,  8736.00,  75936.00),
(1015, 20240916, 3, 6, 4,   4,  75200.00,  9776.00,  84976.00),
(1016, 20241011, 1, 1, 1,  18, 104400.00, 13572.00, 117972.00),
(1017, 20241011, 6, 2, 2,  22,  92400.00, 12012.00, 104412.00),
(1018, 20241114, 4, 5, 5,  35,  42000.00,  5460.00,  47460.00),
(1019, 20241220, 2, 6, 3,   6, 112800.00, 14664.00, 127464.00),
(1020, 20241220, 1, 3, 4,  70,  68600.00,  8918.00,  77518.00);

-- 社保缴费维度（含 杭州/北京 2024，列名 region/year 对齐 ShebaoAgent）
INSERT INTO dim_social_security
(region, year, base_lower, base_upper,
 pension_company_rate, pension_personal_rate,
 medical_company_rate, medical_personal_rate,
 unemployment_company_rate, unemployment_personal_rate,
 injury_company_rate) VALUES
('杭州', 2024, 4812.00, 24930.00, 0.1600, 0.0800, 0.0950, 0.0200, 0.0050, 0.0050, 0.0040),
('北京', 2024, 6821.00, 35283.00, 0.1600, 0.0800, 0.0900, 0.0200, 0.0050, 0.0050, 0.0040),
('深圳', 2024, 2360.00, 27927.00, 0.1400, 0.0800, 0.0520, 0.0200, 0.0070, 0.0030, 0.0050),
('成都', 2024, 4246.00, 21228.00, 0.1600, 0.0800, 0.0680, 0.0200, 0.0040, 0.0040, 0.0030),
('杭州', 2023, 4462.00, 22311.00, 0.1600, 0.0800, 0.0950, 0.0200, 0.0050, 0.0050, 0.0040);

-- 商品税收分类编码维度（含 天然气/建筑服务，列名 product_name 对齐 ProductCodeAgent）
INSERT INTO dim_tax_product_code (product_name, code, category, rate) VALUES
('天然气',       '1070201010000000000', '成品油及其他能源',       0.0900),
('建筑服务',     '3050000000000000000', '建筑服务',               0.0900),
('居民用管道天然气', '1070201020000000000', '成品油及其他能源',   0.0900),
('笔记本电脑',   '1090112020000000000', '计算机、通信和其他电子设备', 0.1300),
('信息技术服务', '3040201000000000000', '信息技术服务',           0.0600),
('餐饮服务',     '3070101000000000000', '餐饮服务',               0.0600),
('房屋租赁服务', '3110201020000000000', '不动产经营租赁服务',     0.0900);

-- =====================================================================
-- 五、自检示例（可手动执行，验证 Text2SQL 常见问法均可跑通）
-- ---------------------------------------------------------------------
-- 各月销售额：
--   SELECT d.year_month_label, SUM(f.sales_amount) AS 销售额
--   FROM fact_sales_order f JOIN dim_date d ON f.date_key=d.date_key
--   GROUP BY d.year_month_label ORDER BY d.year_month_label;
-- 各省销售额排名：
--   SELECT r.province, SUM(f.sales_amount) AS 销售额
--   FROM fact_sales_order f JOIN dim_region r ON f.region_key=r.region_key
--   GROUP BY r.province ORDER BY 销售额 DESC;
-- 各品类销售额：
--   SELECT p.category, SUM(f.sales_amount) AS 销售额
--   FROM fact_sales_order f JOIN dim_product p ON f.product_key=p.product_key
--   GROUP BY p.category ORDER BY 销售额 DESC;
-- 杭州2024社保基数：
--   SELECT * FROM dim_social_security WHERE region='杭州' AND year=2024;
-- 天然气税收分类编码：
--   SELECT * FROM dim_tax_product_code WHERE product_name LIKE '%天然气%';
-- =====================================================================
