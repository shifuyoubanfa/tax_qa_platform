"""测试包标识。
本包收纳平台的单元/冒烟测试，全部基于 pytest（异步用 pytest-asyncio）。

设计原则（为什么这么测）：
- 纯逻辑优先：意图分类、实体抽取、路由表这些"无外部依赖"的逻辑直接断言，跑得快、最稳。
- 外部依赖一律打桩：LLM/embedding/Milvus/ES/Qdrant/MySQL/Mongo/MinIO 等用 monkeypatch 替身，
  保证在"没有任何基础设施"的开发机/CI 上也能全绿。
- 尚未就绪的模块用 pytest.importorskip 优雅跳过，避免单个模块缺失把整批测试拖红。
"""
