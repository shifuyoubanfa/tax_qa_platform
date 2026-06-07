"""文档入库脚本：解析 -> 切分 -> 向量化 -> 写 Milvus(稠密) + ES(全文)，图片写 MinIO。
本模块在整体链路里的位置：离线侧"数据生产线"。在线检索能查到什么，全靠这里灌进去什么。

完整流水线（每一步对应一段函数，便于学习）：
    1) 遍历 --input-dir 下的源文件（PDF/Markdown/txt）。
    2) 解析：PDF 走 MinerU（按 settings.mineru_mode 选 http/local，当前为"待接入"槽位）；
       md/txt 直接读文本。解析产出 (正文文本, 图片字节列表)。
    3) 切分：用带教学注释的"中文友好"简单切分器，把长文切成带重叠的片段。
       说明：PDF 默认走本地 pypdf 纯文本抽取（parse_pdf_local，无需任何外部服务即可入库）；
       若想要版面/图片等结构化解析，再把 settings.mineru_mode 切到 http 接 MinerU 服务。
    4) 向量化：EmbeddingClient.embed(片段) -> dense(+sparse) 向量。
    5) 落库：MilvusClient.upsert 写稠密向量；ESClient.index_doc 写全文；图片 MinioClient.put_object。
    --dry-run 时只走到"切分+打印统计"，不向量化、不落库，方便先验证切分效果。

设计要点：
- 所有外部连接走 app/clients（惰性）；本脚本不直接连基础设施。
- MinerU 解析为"留槽位"：不编造解析实现，默认抛出清晰中文提示，引导作者接入真实服务/库。
- 切分器刻意写得朴素好懂（按段落聚合 + 字符长度 + 重叠），方便作者理解 RAG 切分的本质。

风格对标 standard/src/main 下的导入脚本（argparse + dry-run + 详细中文日志）。
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Iterable

from config.constants import KBase
from config.logging_config import get_logger, setup_logging
from config.settings import settings

logger = get_logger(__name__)

# 支持的源文件后缀（小写）。PDF 走 MinerU/pypdf，md/txt 按纯文本处理，
# .jsonl 走"历史问答(QA)入库"专用分支（每行一问一答=一条独立召回单元，不切分）。
_SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt", ".jsonl"}


# ============================================================
# 第2步：解析（MinerU 槽位）
# ============================================================
def parse_pdf_with_mineru(pdf_path: Path) -> tuple[str, list[bytes]]:
    """用 MinerU 解析 PDF，产出 (正文文本, 图片字节列表)。【当前为待接入槽位】

    设计说明（为什么留槽位而不编造）：MinerU 的 http 服务接口/本地 magic-pdf 调用都依赖真实环境，
    不同部署差异很大；编造一份"看似能跑"的实现反而误导学习者。这里按 settings.mineru_mode 分流，
    两条路径都抛出"清晰的中文待接入提示"，告诉作者该去哪接、接什么。

    :param pdf_path: 待解析 PDF 的路径。
    :return: 二元组 (markdown/纯文本正文, [图片二进制, ...])。
    :raise NotImplementedError: 解析能力尚未接入时抛出，提示接入方式。
    :raise RuntimeError: mineru_mode 配置非法时抛出。
    """
    mode = settings.mineru_mode
    logger.info("[解析] 进入 MinerU 解析槽位 mode=%s file=%s", mode, pdf_path.name)
    if mode == "http":
        # 接入指引：调用 settings.mineru_base_url 提供的解析服务（带 settings.mineru_token 鉴权），
        # 上传 pdf 字节，拿回 markdown 正文与图片资源，再返回 (text, images)。
        raise NotImplementedError(
            f"MinerU(http 模式) 解析能力待接入。请在 parse_pdf_with_mineru() 中调用 "
            f"settings.mineru_base_url={settings.mineru_base_url!r} 的解析接口完成 PDF->文本/图片。"
        )
    if mode == "local":
        # 接入指引：本地安装 magic-pdf(MinerU)，加载模型对 pdf 做版面/OCR 解析。
        raise NotImplementedError(
            "MinerU(local 模式) 解析能力待接入。请安装 magic-pdf 并在本函数内完成本地解析。"
        )
    raise RuntimeError(f"非法的 mineru_mode={mode!r}，应为 'http' 或 'local'。")


def parse_pdf_local(path: Path) -> tuple[str, list[bytes]]:
    """用 pypdf 本地抽取 PDF 纯文本，产出 (正文文本, 空图片列表)。

    在链路里的位置：这是 PDF 入库的"零依赖默认路径"。只要本地装了 pypdf，
    就能把 PDF 当纯文本灌进知识库，不必先把 MinerU 服务跑起来。

    它和 MinerU 的本质区别（理解这点才知道何时该升级）：
    - pypdf 只做"取文字"：逐页 extract_text() 把 PDF 里的文本流拼成正文，
      不识别版面结构、不做表格还原、不抽图片、对扫描件(图片型 PDF)抽不出字。
    - MinerU(http) 做"结构化解析"：版面分析 + OCR + 表格/公式/图片抽取，
      产出更接近原排版的 markdown 与图片资源。需要这些时再把 mineru_mode 切到 http。
    因此本函数 images 恒为 []（本地纯文本抽取，不取图）。

    :param path: 待解析 PDF 的路径。
    :return: 二元组 (拼接后的纯文本正文, [] 空图片列表)。
    :raise RuntimeError: pypdf 未安装、或 PDF 打开/抽取失败时抛出（带清晰中文上下文）。
    """
    # 惰性导入：import 阶段不强依赖 pypdf，缺包时给出可操作的中文提示
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "未安装 pypdf，无法本地抽取 PDF 文本。请先 `pip install pypdf`"
            "（已在 requirements.txt 中声明），或把 settings.mineru_mode 切到 'http' 接 MinerU。"
        ) from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001
        # 打不开通常是文件损坏/加密/不是真正的 PDF，原样带出底层原因便于排查
        raise RuntimeError(f"pypdf 打开 PDF 失败（file={path.name}）：{exc}") from exc

    # 逐页抽文本：单页抽取失败（个别页损坏）不应让整篇报废，降级为跳过该页并记日志
    page_texts: list[str] = []
    for page_no, page in enumerate(reader.pages):
        try:
            page_texts.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[解析] pypdf 抽取单页失败，跳过该页 file=%s 第%d页：%s",
                           path.name, page_no + 1, exc)
            page_texts.append("")

    # 用换行拼接各页，正好衔接 split_text 的"按换行粗分段落"口径
    text = "\n".join(page_texts).strip()
    if not text:
        # 抽不到任何字：极可能是扫描件/图片型 PDF —— 这正是该上 MinerU(OCR) 的信号
        logger.warning(
            "[解析] pypdf 未抽到任何文本 file=%s（可能是扫描件/图片型 PDF）。"
            "如需 OCR/版面解析，请把 settings.mineru_mode 切到 'http' 接 MinerU。", path.name)
    logger.info("[解析] pypdf 本地抽取完成 file=%s 页数=%d 字符数=%d",
                path.name, len(reader.pages), len(text))
    return text, []


def parse_text_file(path: Path) -> tuple[str, list[bytes]]:
    """解析纯文本类文件（md/markdown/txt）：直接读取为正文，无图片。

    :param path: 文本文件路径。
    :return: (正文文本, 空图片列表)。
    :raise OSError: 文件读取失败时抛出（编码问题用 errors='ignore' 容错）。
    """
    # 用 utf-8 读，遇到个别坏字节忽略，保证一个坏字符不至于让整篇导入失败
    text = path.read_text(encoding="utf-8", errors="ignore")
    logger.info("[解析] 文本文件读取完成 file=%s 字符数=%d", path.name, len(text))
    return text, []


def parse_file(path: Path) -> tuple[str, list[bytes]]:
    """按后缀分流解析单个源文件。

    PDF 分流策略（按 settings.mineru_mode 选解析后端）：
    - local（建议默认）：走 parse_pdf_local，用 pypdf 本地纯文本抽取，零外部依赖即可入库；
    - http：走 parse_pdf_with_mineru，调 MinerU 服务做版面/OCR/图片等结构化解析（更强但需先把服务跑起来）。
    一句话：本地有 pypdf 就能把 PDF 当纯文本入库；要版面/图片再切到 http 接 MinerU。

    :param path: 源文件路径。
    :return: (正文文本, 图片字节列表)。
    :raise NotImplementedError: PDF 走 http(MinerU) 槽位（待接入）时抛出。
    :raise RuntimeError: PDF 本地抽取失败、或 mineru_mode 配置非法时抛出。
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        mode = settings.mineru_mode
        if mode == "http":
            # http：保留为 MinerU 结构化解析槽位（版面/OCR/图片）
            return parse_pdf_with_mineru(path)
        if mode == "local":
            # local：pypdf 本地纯文本抽取，无需任何外部服务
            return parse_pdf_local(path)
        raise RuntimeError(f"非法的 mineru_mode={mode!r}，应为 'http'(MinerU) 或 'local'(pypdf)。")
    return parse_text_file(path)


# ============================================================
# 第3步：切分（教学式简单切分器）
# ============================================================
def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """把长文本切成带重叠的片段（中文友好的简单切分器）。

    为什么这样切（RAG 切分的核心权衡）：
    - 片段太长：召回粒度粗、向量语义被稀释，且超出 embedding 上下文；
    - 片段太短：上下文不完整，答非所问；
    - 重叠(overlap)：避免关键句正好被切在两片交界处而"语义断裂"。
    实现思路：先按"段落/换行"粗分，再贪心地把相邻段落拼到接近 chunk_size，
    超长的单段再按字符硬切；相邻片段之间保留 overlap 个字符的重叠。

    :param text: 原始正文。
    :param chunk_size: 单片目标最大字符数（中文按字符计）。
    :param overlap: 相邻片段的重叠字符数（应小于 chunk_size）。
    :return: 切好的片段列表（已去除空白片段）。
    """
    text = (text or "").strip()
    if not text:
        return []
    if overlap >= chunk_size:
        # 防御：重叠不该大于等于片长，否则会原地踏步切不动
        overlap = chunk_size // 5

    # 先按换行粗分成"段落"，过滤空段
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks: list[str] = []
    buf = ""  # 当前正在累积的片段缓冲区
    for para in paragraphs:
        # 单段就超长：先把缓冲区落地，再对超长段按字符窗口硬切
        if len(para) > chunk_size:
            if buf:
                chunks.append(buf)
                buf = ""
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunks.append(para[start:end])
                # 下一个窗口回退 overlap，制造重叠
                start = end - overlap
            continue
        # 普通段：能拼进缓冲区就拼，拼不下就先落地再开新片
        if len(buf) + len(para) + 1 <= chunk_size:
            buf = f"{buf}\n{para}" if buf else para
        else:
            if buf:
                chunks.append(buf)
            # 新片以"上一片尾部 overlap 字符"开头，保持上下文连续
            tail = chunks[-1][-overlap:] if chunks and overlap > 0 else ""
            buf = f"{tail}\n{para}" if tail else para
    if buf:
        chunks.append(buf)

    # 去掉切完仍为空白的片段
    result = [c.strip() for c in chunks if c.strip()]
    logger.info("[切分] 完成 原始字符=%d -> 片段数=%d (chunk_size=%d overlap=%d)",
                len(text), len(result), chunk_size, overlap)
    return result


# ============================================================
# 工具：遍历源文件
# ============================================================
def iter_source_files(input_dir: Path) -> Iterable[Path]:
    """递归遍历输入目录下所有受支持的源文件。

    :param input_dir: 文档根目录。
    :return: 逐个产出受支持的文件路径（生成器）。
    :raise FileNotFoundError: 目录不存在时抛出，提示作者放数据的位置。
    """
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(
            f"输入目录不存在：{input_dir.resolve()}。请把待入库文档放到 data/documents/ 下，"
            f"或用 --input-dir 指定目录（格式见 data/README.md）。"
        )
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES:
            yield path


# ============================================================
# 第5步：图片落 MinIO
# ============================================================
def _upload_images(images: list[bytes], doc_id: str, dry_run: bool) -> list[str]:
    """把一篇文档解析出的图片写入 MinIO，返回对象键列表。

    :param images: 图片二进制列表。
    :param doc_id: 所属文档 id，用于拼对象键前缀。
    :param dry_run: 干跑时只生成键名、不真正上传。
    :return: MinIO 对象键列表（写入 Document.image_keys，前端凭此取图）。
    """
    if not images:
        return []
    keys: list[str] = []
    client = None if dry_run else _lazy_minio()
    for idx, img in enumerate(images):
        key = f"{doc_id}/img_{idx}.png"  # 对象键约定：{文档id}/img_{序号}.png
        keys.append(key)
        if not dry_run:
            client.put_object(key, img)  # 真正上传二进制
    logger.info("[图片] doc_id=%s 图片数=%d 已%s", doc_id, len(images),
                "生成键名(干跑)" if dry_run else "上传 MinIO")
    return keys


def _lazy_minio():
    """惰性构造 MinioClient（推迟重依赖导入）。"""
    from app.clients.minio_client import MinioClient
    return MinioClient()


# ============================================================
# 主流程：单文件入库
# ============================================================
def ingest_file(
    path: Path,
    kbase: str,
    chunk_size: int,
    overlap: int,
    dry_run: bool,
    emb_client=None,
    milvus_client=None,
    es_client=None,
) -> int:
    """处理单个源文件：解析->切分->(向量化->落库)。

    :param path: 源文件路径。
    :param kbase: 该文件归属的知识库（KBase 值，如 'doc'/'policy'）。
    :param chunk_size: 切分片长。
    :param overlap: 切分重叠。
    :param dry_run: 干跑（只切分+统计，不向量化不落库）。
    :param emb_client/milvus_client/es_client: 可注入的客户端（便于复用/测试），为 None 时惰性新建。
    :return: 本文件成功入库（或干跑时切出的）片段数量。
    :raise RuntimeError: 向量化或落库失败时抛出带中文上下文的错误。
    """
    logger.info("[入库] 开始处理文件 file=%s kbase=%s", path.name, kbase)
    # 顶部按后缀分流：.jsonl 是"历史问答"，每行一问一答本身就是一个召回单元，
    # 不适合套 parse_file/split_text(切分)那套（强行切会把一问一答拆散）。
    # 故走独立分支 ingest_qa_jsonl：解析→(每行)向量化→写 Milvus+ES，复用同一组客户端。
    if path.suffix.lower() == ".jsonl":
        return ingest_qa_jsonl(
            path, kbase, dry_run,
            emb_client=emb_client, milvus_client=milvus_client, es_client=es_client,
        )

    # 其余后缀(pdf/md/txt)：走原"解析→切分"流水线。
    text, images = parse_file(path)
    chunks = split_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        logger.warning("[入库] 文件无有效内容，跳过 file=%s", path.name)
        return 0

    # 文档级 id：用文件名 stem + 短 uuid，保证可读且不撞车
    base_id = f"{path.stem}-{uuid.uuid4().hex[:8]}"
    image_keys = _upload_images(images, base_id, dry_run)

    if dry_run:
        # 干跑：打印前 2 片预览，帮助作者直观判断切分质量，然后结束
        for i, c in enumerate(chunks[:2]):
            preview = c[:60].replace("\n", " ")
            logger.info("[入库][dry-run] 片段#%d 预览：%s ...", i, preview)
        logger.info("[入库][dry-run] file=%s 共切 %d 片（未向量化、未落库）", path.name, len(chunks))
        return len(chunks)

    # ---- 向量化（惰性新建客户端）----
    emb_client = emb_client or _lazy_embedding()
    try:
        emb = emb_client.embed(chunks)  # 约定返回 {"dense": [[...]], "sparse": [{...}]}
        dense_list = emb.get("dense", [])
        sparse_list = emb.get("sparse", [None] * len(chunks))
    except Exception as exc:  # noqa: BLE001
        logger.error("[入库] 向量化失败 file=%s：%s", path.name, exc, exc_info=True)
        raise RuntimeError(f"向量化失败（file={path.name}）：{exc}") from exc

    # ---- 落库（Milvus 稠密 + ES 全文）----
    milvus_client = milvus_client or _lazy_milvus()
    es_client = es_client or _lazy_es()

    milvus_rows: list[dict] = []
    for i, chunk in enumerate(chunks):
        doc_id = f"{base_id}-{i}"  # 片段级 id
        dense_vec = dense_list[i] if i < len(dense_list) else []
        metadata = {
            "source_file": path.name,
            "chunk_index": i,
        }
        # Milvus：主键 + 向量 + 必要标量字段（具体字段以 MilvusClient.ensure_collection 的 schema 为准）
        milvus_rows.append({
            "id": doc_id,
            "vector": dense_vec,
            "kbase": kbase,
            "title": path.stem,
            "content": chunk,
            "doc_no": metadata.get("doc_no", ""),   # schema 定义了该字段→必须提供(md 暂无文号则空串)，否则 Milvus 入库报缺字段
            "image_keys": image_keys,                # 与 ES 的 image_keys、Milvus schema 的 JSON 字段三处口径一致：图片对象键数组
            "metadata": metadata,                    # 嵌套元数据(JSON)，与 ES 的 metadata、与召回端读取口径一致
        })
        # ES：写一条全文文档（doc_id 作为 _id，便于幂等覆盖）
        try:
            es_client.index_doc(
                index=settings.es_doc_index,
                doc_id=doc_id,
                body={
                    "doc_id": doc_id,
                    "title": path.stem,
                    "content": chunk,
                    "kbase": kbase,
                    "doc_no": metadata.get("doc_no", ""),   # 顶层 keyword，便于按文号精确过滤(term)
                    "image_keys": image_keys,
                    "metadata": metadata,                    # 嵌套对象，与 mapping(object) 及召回端 source.get("metadata") 一致
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[入库] ES 写入失败 doc_id=%s：%s", doc_id, exc, exc_info=True)
            raise RuntimeError(f"ES 写入失败（doc_id={doc_id}）：{exc}") from exc

    try:
        milvus_client.upsert(collection=settings.milvus_doc_collection, rows=milvus_rows)
    except Exception as exc:  # noqa: BLE001
        logger.error("[入库] Milvus 写入失败 file=%s：%s", path.name, exc, exc_info=True)
        raise RuntimeError(f"Milvus 写入失败（file={path.name}）：{exc}") from exc

    logger.info("[入库] 完成 file=%s 写入片段=%d (Milvus+ES)", path.name, len(chunks))
    return len(chunks)


# ============================================================
# 历史问答(.jsonl) 入库：每行一问一答 = 一条独立记录（不切分）
# ============================================================
def parse_qa_jsonl(path: Path) -> list[dict]:
    """按行解析 .jsonl 历史问答文件，产出一条条 {question, answer, tags} 记录。

    为什么单独写一个解析器（不复用 parse_file/parse_text_file）：
    - parse_file 的契约是返回 (整篇正文, 图片)，再交给 split_text 切片——那是"长文档"的口径；
    - 而一问一答里"一行"本身就是一个完整的语义单元(召回粒度)，切片只会破坏它。
    因此这里逐行 json.loads，把每行直接当成一条记录返回，由 ingest_qa_jsonl 逐条向量化落库。

    容错策略（一行坏不毁全篇）：空行跳过；某行 JSON 解析失败/缺 question 与 answer 时
    记 warning 并跳过该行，不让整个文件入库中断。

    :param path: .jsonl 文件路径（每行一个 JSON 对象：{question, answer, tags}）。
    :return: 记录列表，每条形如 {"question": str, "answer": str, "tags": Any, "line_no": int}。
    :raise OSError: 文件读取失败时抛出（编码问题用 errors='ignore' 容错）。
    """
    records: list[dict] = []
    # 逐行读：用 utf-8 容错读，避免个别坏字节让整篇报废
    raw = path.read_text(encoding="utf-8", errors="ignore")
    for line_no, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue  # 跳过空行
        try:
            obj = json.loads(line)
        except Exception as exc:  # noqa: BLE001 - 单行坏 JSON 不该毁全篇
            logger.warning("[入库][QA] 第%d行 JSON 解析失败，跳过 file=%s：%s",
                           line_no, path.name, exc)
            continue
        if not isinstance(obj, dict):
            logger.warning("[入库][QA] 第%d行不是 JSON 对象，跳过 file=%s", line_no, path.name)
            continue
        question = str(obj.get("question", "")).strip()
        answer = str(obj.get("answer", "")).strip()
        # question 作 title、answer 作 content；两者全空则这行没有召回价值，跳过
        if not question and not answer:
            logger.warning("[入库][QA] 第%d行缺 question/answer，跳过 file=%s", line_no, path.name)
            continue
        records.append({
            "question": question,
            "answer": answer,
            "tags": obj.get("tags"),   # 原样保留(可能是 list/str/None)，放进 metadata
            "line_no": line_no,
        })
    logger.info("[入库][QA] 解析完成 file=%s 有效记录=%d", path.name, len(records))
    return records


def ingest_qa_jsonl(
    path: Path,
    kbase: str,
    dry_run: bool,
    emb_client=None,
    milvus_client=None,
    es_client=None,
) -> int:
    """历史问答(.jsonl)入库分支：每行一条记录，question 作 title、answer 作 content，tags 进 metadata。

    与 ingest_file 主流程的区别（核心）：不走 split_text 切分——一问一答本身就是一个召回单元，
    切片只会破坏它。流程：解析(每行)→(整批)向量化→逐条组装 Milvus 行 + 写 ES。

    向量化口径：用 "question\\nanswer" 作为被向量化的文本（问+答都进向量，召回更稳）；
    若某行 answer 为空则只向量化 question。

    :param path: .jsonl 文件路径。
    :param kbase: 该批问答归属的知识库（默认上层会传 'qa'）。
    :param dry_run: 干跑时只统计条数，不向量化、不落库。
    :param emb_client/milvus_client/es_client: 可注入客户端（复用/测试），为 None 时惰性新建。
    :return: 本文件成功入库（或干跑统计）的记录条数。
    :raise RuntimeError: 向量化或落库失败时抛出带中文上下文的错误。
    """
    records = parse_qa_jsonl(path)
    if not records:
        logger.warning("[入库][QA] 文件无有效问答记录，跳过 file=%s", path.name)
        return 0

    if dry_run:
        # 干跑：只统计条数 + 预览前 2 条，不向量化、不落库
        for i, rec in enumerate(records[:2]):
            preview = (rec["question"] or rec["answer"])[:60].replace("\n", " ")
            logger.info("[入库][QA][dry-run] 记录#%d 预览：%s ...", i, preview)
        logger.info("[入库][QA][dry-run] file=%s 共 %d 条问答（未向量化、未落库）", path.name, len(records))
        return len(records)

    # ---- 向量化（惰性新建客户端）----
    emb_client = emb_client or _lazy_embedding()
    # 被向量化的文本：问+答拼接(答为空则只用问)，让召回既能命中问法也能命中答案内容
    embed_texts = [
        (f"{rec['question']}\n{rec['answer']}".strip() if rec["answer"] else rec["question"])
        for rec in records
    ]
    try:
        emb = emb_client.embed(embed_texts)  # 约定返回 {"dense": [[...]], "sparse": [...]}
        dense_list = emb.get("dense", [])
    except Exception as exc:  # noqa: BLE001
        logger.error("[入库][QA] 向量化失败 file=%s：%s", path.name, exc, exc_info=True)
        raise RuntimeError(f"向量化失败（file={path.name}）：{exc}") from exc

    # ---- 落库（Milvus 稠密 + ES 全文）----
    milvus_client = milvus_client or _lazy_milvus()
    es_client = es_client or _lazy_es()

    milvus_rows: list[dict] = []
    for i, rec in enumerate(records):
        # doc_id：文件 stem + 行号 + 短 uuid，保证可读且不撞车（一问一答天然就是一条 doc）
        doc_id = f"{path.stem}-L{rec['line_no']}-{uuid.uuid4().hex[:8]}"
        dense_vec = dense_list[i] if i < len(dense_list) else []
        # tags 放进 metadata，与文档库的 metadata(JSON) 口径一致，召回端按同样方式读取
        metadata = {
            "source_file": path.name,
            "line_no": rec["line_no"],
            "tags": rec["tags"],
        }
        # Milvus：question 作 title、answer 作 content；image_keys 历史问答无图，给空数组保持三处口径一致
        milvus_rows.append({
            "id": doc_id,
            "vector": dense_vec,
            "kbase": kbase,
            "title": rec["question"],
            "content": rec["answer"],
            "doc_no": "",                 # 历史问答无文号，给空串(schema 要求该字段必填)
            "image_keys": [],             # 与 ES 的 image_keys、Milvus schema 的 JSON 字段三处口径一致
            "metadata": metadata,
        })
        # ES：写一条全文文档（doc_id 作为 _id，便于幂等覆盖）
        try:
            es_client.index_doc(
                index=settings.es_doc_index,
                doc_id=doc_id,
                body={
                    "doc_id": doc_id,
                    "title": rec["question"],
                    "content": rec["answer"],
                    "kbase": kbase,
                    "doc_no": "",
                    "image_keys": [],
                    "metadata": metadata,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[入库][QA] ES 写入失败 doc_id=%s：%s", doc_id, exc, exc_info=True)
            raise RuntimeError(f"ES 写入失败（doc_id={doc_id}）：{exc}") from exc

    try:
        milvus_client.upsert(collection=settings.milvus_doc_collection, rows=milvus_rows)
    except Exception as exc:  # noqa: BLE001
        logger.error("[入库][QA] Milvus 写入失败 file=%s：%s", path.name, exc, exc_info=True)
        raise RuntimeError(f"Milvus 写入失败（file={path.name}）：{exc}") from exc

    logger.info("[入库][QA] 完成 file=%s 写入问答=%d (Milvus+ES)", path.name, len(records))
    return len(records)


def _lazy_embedding():
    from app.clients.embedding_client import EmbeddingClient
    return EmbeddingClient()


def _lazy_milvus():
    from app.clients.milvus_client import MilvusClient
    return MilvusClient()


def _lazy_es():
    from app.clients.es_client import ESClient
    return ESClient()


def ingest_dir(input_dir: Path, kbase: str, chunk_size: int, overlap: int, dry_run: bool) -> int:
    """遍历目录并逐个入库，返回累计片段数。

    :param input_dir: 文档根目录。
    :param kbase: 知识库归属（KBase 值）。
    :param chunk_size: 切分片长。
    :param overlap: 切分重叠。
    :param dry_run: 是否干跑。
    :return: 累计处理的片段总数。
    """
    logger.info("[入库] 扫描目录 dir=%s kbase=%s dry_run=%s", input_dir, kbase, dry_run)
    # 干跑时只新建一次客户端引用为空即可；非干跑时复用同一组客户端，避免每个文件都重连
    emb_client = milvus_client = es_client = None
    if not dry_run:
        emb_client = _lazy_embedding()
        milvus_client = _lazy_milvus()
        es_client = _lazy_es()

    total = 0
    file_count = 0
    for path in iter_source_files(input_dir):
        file_count += 1
        try:
            total += ingest_file(
                path, kbase, chunk_size, overlap, dry_run,
                emb_client=emb_client, milvus_client=milvus_client, es_client=es_client,
            )
        except NotImplementedError as exc:
            # MinerU 槽位未接入：跳过该 PDF 并提示，不让整批中断
            logger.warning("[入库] 跳过（解析能力待接入）file=%s：%s", path.name, exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("[入库] 文件处理失败 file=%s：%s", path.name, exc, exc_info=True)
    logger.info("[入库] 全部完成 文件数=%d 片段总数=%d", file_count, total)
    return total


def _build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。

    :return: 配置好的 ArgumentParser。
    """
    parser = argparse.ArgumentParser(
        description="文档入库：解析->切分->向量化->写 Milvus+ES，图片写 MinIO。"
    )
    parser.add_argument("--input-dir", default="data/documents",
                        help="待入库文档目录（默认 data/documents）。")
    parser.add_argument("--kbase", default=KBase.DOC.value,
                        choices=[k.value for k in KBase],
                        help="这批文档归属的知识库（默认 doc）。")
    parser.add_argument("--chunk-size", type=int, default=500, help="切分片长（字符数，默认500）。")
    parser.add_argument("--overlap", type=int, default=80, help="相邻片段重叠字符数（默认80）。")
    parser.add_argument("--dry-run", action="store_true",
                        help="干跑：只解析+切分并打印统计，不向量化、不落库。")
    return parser


def main() -> None:
    """脚本入口：解析参数 -> 初始化日志 -> 执行入库。"""
    parser = _build_arg_parser()
    args = parser.parse_args()

    setup_logging()
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    ingest_dir(
        input_dir=Path(args.input_dir),
        kbase=args.kbase,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    # 学习提示：
    #   py scripts/ingest_documents.py --input-dir data/documents --dry-run   # 先看切分效果
    #   py scripts/ingest_documents.py --input-dir data/documents --kbase doc # 正式入库
    # PDF 入库：mineru_mode=local(默认建议) 用 pypdf 本地抽纯文本，开箱即用；
    #           mineru_mode=http 走 MinerU 服务做版面/OCR/图片(见 parse_pdf_with_mineru 待接入提示)。md/txt 始终可直接入库。
    main()
