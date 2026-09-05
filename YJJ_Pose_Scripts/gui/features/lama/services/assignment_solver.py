"""最小 O(N^3) 线性分配（匈牙利 / Kuhn-Munkres）求解器。

存在理由
--------
关键点 ID assignment 原本用 ``itertools.permutations(N)`` 全排列枚举。
N=7 时 7! = 5040 尚可接受，但关键点数量改为 UI 可配置（1~32）之后，
10! / 20! 会直接不可用（阶乘爆炸）。因此所有一对一 ID assignment 统一走本求解器。

选型：项目不引入 scipy（服务层保持零第三方依赖），自行实现
e-maxx 版 Jonker-Volgenant 变体，方阵 O(N^3)，N<=32 时开销可忽略。

只做"求解最小代价一对一分配"，不含任何业务语义（不认识关键点 / mask / reference）。
"""
from __future__ import annotations

import numpy as np


def solve_linear_assignment(cost: np.ndarray) -> list[int] | None:
    """求解总代价最小的一对一分配（方阵）。

    参数：
        cost: (N, N) 浮点代价矩阵，cost[i, j] = 把第 i 行分配给第 j 列的代价。
              值必须全部有限（不允许 NaN / Inf）。

    返回：
        list[int]，长度 N；result[i] = j 表示第 i 行分配给第 j 列。
        输入非法（None / 空 / 非方阵 / 含非有限值）时返回 None。
    """
    if cost is None or cost.size == 0:
        return None
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        return None

    n = int(cost.shape[0])
    if n <= 0:
        return None

    c = np.asarray(cost, dtype=np.float64)
    if not np.all(np.isfinite(c)):
        return None

    inf = float("inf")
    # 下标 1..N 为真实行列，下标 0 是增广路哨兵
    u = np.zeros(n + 1, dtype=np.float64)      # 行势
    v = np.zeros(n + 1, dtype=np.float64)      # 列势
    p = np.zeros(n + 1, dtype=np.int64)        # p[j] = 当前分配给列 j 的行（1..N，0 表示空）
    way = np.zeros(n + 1, dtype=np.int64)      # 增广路前驱

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, inf, dtype=np.float64)
        used = np.zeros(n + 1, dtype=bool)
        # ---- 为第 i 行寻找增广路（Dijkstra 式松弛）----
        while True:
            used[j0] = True
            i0 = int(p[j0])
            row = c[i0 - 1]
            delta = inf
            j1 = -1
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = float(row[j - 1]) - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            # ---- 势更新（对偶变量）----
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # ---- 沿增广路翻转匹配 ----
        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # p[j] = 分配给列 j 的行 -> 转成 result[i] = 分配给行 i 的列
    result = [0] * n
    for j in range(1, n + 1):
        result[int(p[j]) - 1] = j - 1
    return result
