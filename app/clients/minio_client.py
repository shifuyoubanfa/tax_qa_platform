"""MinIO 对象存储客户端（多模态图片，惰性连接）。
本模块在整体链路里的位置：基础设施层。多模态文档(掌柜智库式)解析后，正文里的图片/表格截图
存进 MinIO，文档元数据里只存 object key(image_keys)。前端展示引用时，由本模块按 key
换成"带签名的临时可访问 URL"(presigned_url) 给浏览器加载。

为什么图片走对象存储而非塞进库：
- 图片是二进制大对象，放向量库/ES 既臃肿又拖慢检索。对象存储专门干这个：便宜、可水平扩展、
  能生成有时效的预签名 URL（无需把桶设为公开也能临时访问，更安全）。

惰性：import 不连服务；首次用到才建 Minio 客户端，并按需幂等建桶。

接口契约（INTERFACES，签名不可变）：
- presigned_url(key) -> str
- put_object(key, data) -> None     （data: bytes 或 file-like）
- get_object(key) -> bytes
"""
from __future__ import annotations

import io
from datetime import timedelta

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class MinioClient:
    """MinIO 客户端封装（惰性连接 + 幂等建桶 + 取/存对象 + 预签名URL）。

    用法::

        mio = MinioClient()
        mio.put_object("doc1/img1.png", image_bytes)
        url = mio.presigned_url("doc1/img1.png")  # 临时可访问URL

    :return: 见各方法说明。
    """

    def __init__(self) -> None:
        """构造仅保存配置，不建连接（惰性）。"""
        self._client = None        # minio.Minio 实例
        self._bucket_ready = False  # 桶是否已确保（惰性建桶）
        logger.info("[MinIO客户端] 初始化（未连接）：endpoint=%s bucket=%s",
                    settings.minio_endpoint, settings.minio_bucket)

    # ---------------- 惰性连接 ----------------
    def _get_client(self):
        """惰性建立 Minio 客户端。

        :return: minio.Minio 实例。
        :raise RuntimeError: 未安装 minio，或创建失败。
        """
        if self._client is None:
            try:
                from minio import Minio
            except ImportError as e:
                raise RuntimeError("[MinIO客户端] 未安装 minio，请 `pip install minio`") from e
            try:
                # endpoint 是 host:port（不带 scheme）；secure 决定 http/https。
                self._client = Minio(
                    endpoint=settings.minio_endpoint,
                    access_key=settings.minio_access_key,
                    secret_key=settings.minio_secret_key,
                    secure=settings.minio_secure,
                )
                logger.info("[MinIO客户端] 客户端已建立：%s（secure=%s）",
                            settings.minio_endpoint, settings.minio_secure)
            except Exception as e:
                logger.error("[MinIO客户端] 创建客户端失败：%s", e, exc_info=True)
                raise RuntimeError(f"[MinIO客户端] 创建 MinIO 客户端失败：{e}") from e
        return self._client

    def _ensure_bucket(self) -> None:
        """惰性幂等建桶：桶不存在则创建（首次写/读对象前确保）。"""
        if self._bucket_ready:
            return
        client = self._get_client()
        bucket = settings.minio_bucket
        try:
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
                logger.info("[MinIO客户端] 桶不存在，已创建：%s", bucket)
            else:
                logger.info("[MinIO客户端] 桶已存在：%s", bucket)
            self._bucket_ready = True
        except Exception as e:
            logger.error("[MinIO客户端] 确保桶失败：%s", e, exc_info=True)
            raise RuntimeError(f"[MinIO客户端] 确保桶【{bucket}】失败：{e}") from e

    # ---------------- 预签名URL ----------------
    def presigned_url(self, key: str, expires_seconds: int = 3600) -> str:
        """生成对象的临时可访问 URL（默认 1 小时有效）。

        :param key: 对象键（如 "doc1/img1.png"）。
        :param expires_seconds: 有效期秒数（默认 3600）。
        :return: 预签名 URL；失败返回空串（前端拿到空串则不渲染该图，不报错）。
        """
        try:
            client = self._get_client()
            url = client.presigned_get_object(
                settings.minio_bucket, key, expires=timedelta(seconds=expires_seconds)
            )
            logger.info("[MinIO客户端] 生成预签名URL：key=%s", key)
            return url
        except Exception as e:
            logger.error("[MinIO客户端] 生成预签名URL失败：key=%s err=%s", key, e, exc_info=True)
            return ""

    # ---------------- 写对象 ----------------
    def put_object(self, key: str, data) -> None:
        """上传对象（图片等二进制）。

        :param key: 对象键。
        :param data: bytes 或 file-like（有 read 方法）。
        :return: 无。
        :raise RuntimeError: 上传失败。
        """
        self._ensure_bucket()
        client = self._get_client()
        try:
            # 统一成 (stream, length)：bytes 包成 BytesIO；file-like 需调用方保证可 seek 取长度。
            if isinstance(data, (bytes, bytearray)):
                stream = io.BytesIO(data)
                length = len(data)
            else:
                # file-like：尝试求长度（seek 到末尾再回到开头）
                data.seek(0, io.SEEK_END)
                length = data.tell()
                data.seek(0)
                stream = data
            client.put_object(settings.minio_bucket, key, stream, length=length)
            logger.info("[MinIO客户端] 上传对象完成：key=%s，size=%d", key, length)
        except Exception as e:
            logger.error("[MinIO客户端] 上传对象失败：key=%s err=%s", key, e, exc_info=True)
            raise RuntimeError(f"[MinIO客户端] 上传对象【{key}】失败：{e}") from e

    # ---------------- 读对象 ----------------
    def get_object(self, key: str) -> bytes:
        """下载对象内容为 bytes。

        :param key: 对象键。
        :return: 对象字节内容；失败返回 b""。
        """
        client = self._get_client()
        resp = None
        try:
            resp = client.get_object(settings.minio_bucket, key)
            content = resp.read()
            logger.info("[MinIO客户端] 下载对象完成：key=%s，size=%d", key, len(content))
            return content
        except Exception as e:
            logger.error("[MinIO客户端] 下载对象失败：key=%s err=%s", key, e, exc_info=True)
            return b""
        finally:
            # MinIO 的响应对象需显式 close/release_conn 归还连接，放 finally 保证一定释放。
            if resp is not None:
                try:
                    resp.close()
                    resp.release_conn()
                except Exception:
                    pass


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：需要 MinIO 在线。
    try:
        mio = MinioClient()
        mio.put_object("selftest/hello.txt", b"hello minio")
        data = mio.get_object("selftest/hello.txt")
        print("[minio_client 自测] 读回内容 =>", data)
        print("[minio_client 自测] 预签名URL =>", mio.presigned_url("selftest/hello.txt"))
    except Exception as exc:
        print("[minio_client 自测] 需要 MinIO 在线（属预期）=>", exc)
