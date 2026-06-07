"""数据脚本包（离线运维工具集）。
本包在整体链路里的位置：属于"离线侧"，与在线问答服务(app/)解耦。
负责把原始资料"喂"进各个存储，让在线检索链路有数据可查：

- init_collections.py   ：幂等创建 Milvus 集合 / ES 索引 / Qdrant 集合（建库建表，跑一次即可）。
- ingest_documents.py   ：解析文档->切分->向量化->写 Milvus(稠密) + ES(全文)；图片写 MinIO。
- build_metadata_index.py：读 MySQL 数仓元数据->向量化->写 Qdrant，供 Text2SQL 的 schema linking 用。

设计原则：所有外部连接都走 app/clients 下的客户端（惰性初始化），脚本本身不直接连基础设施；
所有脚本提供 argparse CLI 与详细中文日志，关键脚本支持 --dry-run 干跑（只演示不落库）。
"""
