"""向量与分数归一化工具。
本模块在整体链路里的位置：混合召回时，稠密(向量内积/余弦)得分与稀疏(BM25)得分量纲完全不同，
直接加权相加没有意义；需要先把各路分数做归一化(min-max)再融合。向量入库前也常做 L2 归一化，
让"内积"等价于"余弦相似度"。

设计要点（为什么这么做）：
1. l2_normalize：把向量缩放到单位长度。零向量时直接原样返回，避免除以0。
2. minmax_normalize：把一组分数线性映射到 [0,1]。对"空列表""单元素""全相等"等边界情况
   做了安全处理（避免 max==min 时除以0），保证调用方不必到处写 try。

被 app/core/recall/hybrid.py（混合融合）、入库脚本等复用。
"""
from __future__ import annotations

import math
from typing import Sequence


def l2_normalize(vec: Sequence[float]) -> list[float]:
    """对单个向量做 L2 归一化（缩放到单位长度）。

    :param vec: 原始向量（任意可迭代的浮点序列）。
    :return: 归一化后的新列表；输入为空或为零向量时按原值返回，避免除0。
    """
    if not vec:
        return list(vec)
    # 计算 L2 范数 = sqrt(sum(x^2))
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    # 零向量（范数为0）无法归一化，原样返回，避免除以0
    if norm == 0.0:
        return [float(x) for x in vec]
    return [float(x) / norm for x in vec]


def minmax_normalize(scores: Sequence[float]) -> list[float]:
    """对一组分数做 min-max 归一化，线性映射到 [0,1]。

    边界处理（为什么）：
    - 空列表：返回空列表。
    - 单元素 / 所有分数相等(max==min)：无法拉伸，统一映射为 1.0（视为同等重要），避免除0。

    :param scores: 一组原始分数。
    :return: 归一化后的新列表，长度与输入一致。
    """
    if not scores:
        return []
    values = [float(s) for s in scores]
    lo, hi = min(values), max(values)
    span = hi - lo
    # max==min（含单元素）时无法做线性拉伸，统一给 1.0
    if span == 0.0:
        return [1.0 for _ in values]
    return [(v - lo) / span for v in values]


if __name__ == "__main__":
    # 最小自测块（仅供单文件学习运行）：覆盖常规与边界场景。
    print("[normalize 自测] L2 =>", l2_normalize([3.0, 4.0]))          # 期望 [0.6, 0.8]
    print("[normalize 自测] L2 零向量 =>", l2_normalize([0.0, 0.0]))    # 期望 [0.0, 0.0]
    print("[normalize 自测] minmax =>", minmax_normalize([1, 2, 3]))    # 期望 [0.0, 0.5, 1.0]
    print("[normalize 自测] minmax 单元素 =>", minmax_normalize([5]))   # 期望 [1.0]
    print("[normalize 自测] minmax 空 =>", minmax_normalize([]))        # 期望 []
