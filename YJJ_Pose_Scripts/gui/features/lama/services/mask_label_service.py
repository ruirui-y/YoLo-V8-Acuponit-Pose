"""Mask -> YOLO Pose Label 生成服务（Phase 2）。

职责（与 C++ LamaErasure 对齐）：
1. 从 Final Mask 提取 valid connected components
2. component count 必须 == 本次 session 的关键点数量 N（否则 BLOCK）
3. 每个 component：center / bbox / area
4. stable ID assignment：Reference N 点整体几何一致性（Option 2 防线）
5. 根据 N 个 final mask components 的 union bbox 生成 BoundingBox（比例扩张）
6. 输出 YOLO Pose 标签：
   class cx cy w h  kp0x kp0y vis  ...  kp(N-1)x kp(N-1)y vis
   每行 1 + 4 + N*3 个数值；坐标归一化 [0,1]；vis 固定为 2

关键点数量 N
------------
N 不再硬编码为 7，由调用方（Controller 从 UI 的 Keypoints QSpinBox 取得）显式传入。
本服务所有分配 / 校验 / 输出均以 N 为准：
- component count != N            -> BLOCK
- predicted_centers keys 必须严格 == {0..N-1}
- reference_points 长度必须 == N
- ordered_points 长度 == N

Assignment 不再使用全排列枚举
-----------------------------
N 可配置后 10!/20! 会阶乘爆炸，因此所有一对一分配统一走
``assignment_solver.solve_linear_assignment``（自实现 O(N^3) 匈牙利，零第三方依赖）。
Reference 几何一致性在此基础上用 similarity ICP 迭代收敛，
歧义护栏改为对最终 assignment 做两两 swap 求次优（O(N^2)）。

参考：
- C++ AlignToReference::ExtractMaskCenters（component 提取）
- C++ AlignToReference::BuildLocalTracks（min area 过滤 + 排序）
- C++ MainWindow::onQuickInpaint（占位 bbox = 7 点 min/max + padding）
- C++ tools/generate_bbox_from_keypoints.py（标签格式）

关键约束：
- 每个 ID 只能匹配一个 Component（一对一）
- 匹配不可靠时 BLOCK（返回 None），绝不写错误标签
- 不为关键点引入大型第三方框架（自实现匈牙利 + 阈值检查）
- bbox 必须来自 N 个 MaskComponent.bbox 的 union，不能用 keypoint 的 min/max

本服务不依赖 QWidget / PySide6，纯算法 + 文件 IO，方便自动测试。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .assignment_solver import solve_linear_assignment


# ============================================================ 数据结构
@dataclass
class MaskComponent:
    """单个连通区域。"""
    center: tuple[float, float]            # (cx, cy) 浮点质心（原图像素坐标）
    bbox: tuple[int, int, int, int]        # (x, y, w, h) 整数包围盒
    area: int                              # 像素面积


@dataclass
class LabelResult:
    """Q 流程的标签生成结果。"""
    bbox: tuple[int, int, int, int]                       # (x, y, w, h) 像素 bbox（已 clamp 到图内）
    ordered_points: list[tuple[float, float]]             # [(x, y), ...] N 个点，索引 0..N-1 对应 stable ID
    image_size: tuple[int, int]                           # (w, h)
    components: list[MaskComponent] = field(default_factory=list)  # 调试用：原始 component 列表


# ============================================================ MaskLabelService
class MaskLabelService:
    """Final Mask -> YOLO Pose Label 的纯算法服务。"""

    # C++ 默认参数：minMaskComponentArea=50
    DEFAULT_MIN_AREA = 50

    # P0-5: stable ID assignment 单个匹配距离上限（像素）
    # 超过此阈值视为不可靠 -> BLOCK，绝不写错误标签
    DEFAULT_MAX_ASSIGNMENT_DISTANCE = 80.0

    # Option 2 防御：Reference N 点整体几何一致性 assignment 的歧义护栏。
    # best 与 second-best 的几何 RMS 残差过于接近时（相对 gap <= 此值），
    # 视为无法明确区分的近似对称 / ID 交换歧义，返回 None -> Q BLOCK（绝不猜 ID）。
    DEFAULT_AMBIGUITY_GAP_RATIO = 0.20

    # ---- bbox 扩张规则 ----
    # 原来是固定 20px padding，导致 bbox 基本被最外围关键点锁死（贴边）。
    # 改为 component bbox union 再按尺寸比例向外扩张，并保留最小 padding 保底：
    #   pad_x = max(min_padding, union_w * ratio_x)
    #   pad_y = max(min_padding, union_h * ratio_y)
    DEFAULT_MIN_BBOX_PADDING = 20
    DEFAULT_BBOX_PADDING_RATIO_X = 0.12
    DEFAULT_BBOX_PADDING_RATIO_Y = 0.18

    # similarity（平移 + 旋转 + 等比缩放）共 4 个自由度，
    # 至少需要 3 个点才会留下可判别的残差；N < 3 时几何完全退化
    #（任意 assignment 都能被精确拟合），此时不做歧义 BLOCK。
    MIN_COUNT_FOR_GEOMETRY_GUARD = 3

    # similarity ICP 最大迭代次数（正常几轮即收敛）
    MAX_ICP_ITERATIONS = 16

    # ============================================================ 公开 API
    def make_label(self,
                  final_mask: np.ndarray,
                  predicted_centers: dict[int, tuple[float, float]],
                  image_size: tuple[int, int],
                  *,
                  expected_count: int,
                  min_area: int = DEFAULT_MIN_AREA,
                  max_assignment_distance: float = DEFAULT_MAX_ASSIGNMENT_DISTANCE,
                  reference_points: list[tuple[float, float]] | None = None,
                  ) -> LabelResult | None:
        """从 final mask + 预测中心生成完整标签结果。

        参数：
            final_mask:        二值 mask（HxW，uint8，0/255）
            predicted_centers: dict {track_id: (cx, cy)} 来自 current_prediction
            image_size:        (width, height)
            expected_count:    本次 session 的关键点数量 N（由 UI 决定，>= 1）
            min_area:          最小 component 面积（过滤噪点）
            max_assignment_distance: stable ID 单个匹配距离上限（超过 -> BLOCK）
            reference_points:  当前合法 Reference 的 canonical ordered points（Option 2 防线）

        返回：
            LabelResult 或 None（component 数 != N 或 stable ID 匹配失败 / 距离过大时 BLOCK）
        """
        if expected_count <= 0:
            return None

        # ---- 第一步：提取 components ----
        components = self.extract_components(final_mask, min_area)
        if len(components) != expected_count:
            return None     # 数量不对必须停下，绝不写错误标签

        # ---- 第二步：一对一 stable ID 匹配（Reference 几何一致性 + 阈值）----
        ordered_points = self.assign_stable_ids(
            components, predicted_centers,
            expected_count=expected_count,
            reference_points=reference_points,
            max_distance=max_assignment_distance,
        )
        if ordered_points is None:
            return None     # 匹配不可靠或距离过大 -> BLOCK

        # ---- 第三步：生成 BoundingBox（component bbox union + 比例 padding + clamp）----
        # bbox 必须来自 N 个 MaskComponent.bbox 的 union，不能用 keypoint 的 min/max
        # keypoint 和 bbox 必须来自同一次 finalMask 分析
        bbox = self.build_bbox(components, image_size)

        return LabelResult(
            bbox=bbox,
            ordered_points=ordered_points,
            image_size=image_size,
            components=components,
        )

    def extract_components(self,
                          mask_gray8: np.ndarray,
                          min_area: int = DEFAULT_MIN_AREA
                          ) -> list[MaskComponent]:
        """从二值 mask 提取所有有效 connected component。

        与 C++ AlignToReference::BuildLocalTracks 一致：
        - cv2.connectedComponentsWithStats 8-连通
        - 跳过 area < min_area 的噪点
        - 返回 MaskComponent 列表（未排序，由后续步骤匹配 stable ID）

        返回：list[MaskComponent]
        """
        if mask_gray8 is None or mask_gray8.size == 0:
            return []

        # ---- 双保险二值化（与 C++ BuildLocalTracks 一致）----
        bin_mask = mask_gray8.copy()
        _, bin_mask = cv2.threshold(bin_mask, 127, 255, cv2.THRESH_BINARY)

        # ---- connectedComponentsWithStats：labels / stats / centroids ----
        # n: 包含背景的 component 总数；i=0 是背景
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bin_mask, connectivity=8, ltype=cv2.CV_32S
        )
        if n <= 1:
            return []

        components = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx = float(centroids[i][0])
            cy = float(centroids[i][1])
            components.append(MaskComponent(
                center=(cx, cy),
                bbox=(x, y, w, h),
                area=area,
            ))
        return components

    def assign_stable_ids(self,
                          components: list[MaskComponent],
                          predicted_centers: dict[int, tuple[float, float]],
                          *,
                          expected_count: int,
                          reference_points: list[tuple[float, float]] | None = None,
                          max_distance: float = DEFAULT_MAX_ASSIGNMENT_DISTANCE,
                          ambiguity_gap_ratio: float = DEFAULT_AMBIGUITY_GAP_RATIO,
                          ) -> list[tuple[float, float]] | None:
        """给 N 个 final mask component 分配 stable ID，输出 ordered_points。

        Option 2 防御（Pose stable ID tracking / reference assignment）：
        - 若提供了合法 Reference 的 ordered reference_points（Q 流程总是提供），
          则走 ``_assign_by_reference_geometry``：用匈牙利 + similarity ICP 求
          “Reference N 点整体几何一致性”最优的 ID 映射，再做两两 swap 歧义校验。
          predicted_centers 仅作辅助/交叉校验信号，不决定最终 ID。
        - 若未提供 reference_points（reference_points is None，非 Q 调用 / 历史兜底），
          退回 predicted_centers 的匈牙利最小距离 assignment（_assign_by_predicted）。
        - 若提供了 reference_points 但长度 != N：直接 BLOCK（返回 None），
          绝不静默降级到 predicted assignment——predicted 可能已被 ID swap 污染，
          降级会重新引入“写错 ID label”的风险。

        ambiguity guard：best 与 second-best 几何残差过于接近时，不自动猜 ID，
        返回 None -> Q BLOCK（绝不写错误标签）。

        返回：
            ordered_points: [(x, y), ...] 长度 N，索引 = canonical track_id（0..N-1）
            或 None（数量不对 / 几何歧义 / 匹配不可靠）
        """
        n = expected_count
        if n <= 0 or len(components) != n:
            return None

        if reference_points is None:
            # ---- 兜底：无 Reference（非 Q 调用 / 历史行为）时走 predicted 最小距离 assignment ----
            return self._assign_by_predicted(
                components, predicted_centers,
                expected_count=n, max_distance=max_distance)

        # ---- Option 2 防线：调用方显式给了 Reference 但数量 != N ----
        # 绝不静默降级到 predicted assignment：那会重新引入 stable-ID swap 污染
        # label 的风险（predicted 可能本身就被 swap 污染），直接 BLOCK。
        if len(reference_points) != n:
            print(f"[MaskLabelService] BLOCK: reference_points 数量 "
                  f"{len(reference_points)} != 期望 {n}")
            return None

        # ---- Option 2：Reference 整体几何一致性定 ID（最后一道防线）----
        geometry = self._assign_by_reference_geometry(
            reference_points, components, n, ambiguity_gap_ratio)
        # 辅助信号：若与原 predicted 结果不一致，说明 tracking 可能发生了 ID 交换，
        # 已被几何方法纠正（predicted 仅参考、不决定最终 ID）。
        predicted = self._assign_by_predicted(
            components, predicted_centers,
            expected_count=n, max_distance=max_distance)
        if predicted is not None and geometry is not None and predicted != geometry:
            print("[MaskLabelService] WARN: reference geometry overrode predicted "
                  "assignment (possible stable-ID swap detected)")
        return geometry

    def _assign_by_predicted(self,
                             components: list[MaskComponent],
                             predicted_centers: dict[int, tuple[float, float]],
                             expected_count: int,
                             max_distance: float = DEFAULT_MAX_ASSIGNMENT_DISTANCE,
                             ) -> list[tuple[float, float]] | None:
        """P0-5 实现（predicted_centers 最小距离 assignment），作为兜底/辅助保留。

        N 可配置后不再枚举 N!，改用 O(N^3) 匈牙利求全局最小总距离。
        保留为 reference_points 缺失时的历史行为，并作为 Option 2 的交叉校验信号。
        """
        n = expected_count
        if n <= 0 or len(components) != n:
            return None
        if set(predicted_centers.keys()) != set(range(n)):
            return None

        # ---- 构造 cost 矩阵：cost[i][j] = ID i 与 component j 中心的距离 ----
        cost = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            pc = predicted_centers[i]
            for j in range(n):
                dx = pc[0] - components[j].center[0]
                dy = pc[1] - components[j].center[1]
                cost[i, j] = (dx * dx + dy * dy) ** 0.5

        # ---- 匈牙利求全局最小一对一分配（替代 N! 枚举）----
        perm = solve_linear_assignment(cost)
        if perm is None:
            return None

        # ---- 可靠性检查：每个 assignment 距离不能超过 max_distance ----
        # 任一点距离过大 -> 整体不可靠 -> BLOCK（绝不写错误标签）
        for i in range(n):
            if cost[i, perm[i]] > max_distance:
                return None

        # ---- 输出 ordered_points[track_id] = component.center ----
        return [components[perm[i]].center for i in range(n)]

    def _assign_by_reference_geometry(self,
                                      reference_points: list[tuple[float, float]],
                                      components: list[MaskComponent],
                                      expected_count: int,
                                      ambiguity_gap_ratio: float,
                                      ) -> list[tuple[float, float]] | None:
        """Reference N 点整体几何一致性 assignment（Option 2 核心）。

        可扩展方案（不再枚举 N!）：
        1. 用“形状签名”（每点到其余点的距离排序，平移/旋转/等比缩放不变）
           构造代价矩阵 -> 匈牙利求初始 assignment（避免陷入局部最优）
        2. Similarity ICP 迭代：拟合 similarity -> 按变换后位置重新匈牙利分配
           -> 重复到 assignment 稳定
        3. 歧义护栏：对最终 assignment 做所有两两 swap，取其中最小残差作为
           second-best；与 best 过于接近则 BLOCK（O(N^2)，不是 O(N!)）

        不依赖任何图像坐标排序，可处理任意两个 canonical ID 的 swap。
        """
        n = expected_count
        if n <= 0 or len(reference_points) != n or len(components) != n:
            return None

        ref = np.array(reference_points, dtype=np.float64)                    # (N,2) canonical id
        cand = np.array([c.center for c in components], dtype=np.float64)      # (N,2)

        # ---- 1. 形状签名初始化（姿态无关）----
        perm = self._initial_assignment_by_signature(ref, cand)
        if perm is None:
            return None

        # ---- 2. similarity ICP 收敛 ----
        perm = self._refine_by_similarity_icp(ref, cand, perm)
        if perm is None:
            return None

        best_res = self._residual_of(ref, cand, perm)

        # ---- 3. 歧义护栏（两两 swap 求次优）----
        if n >= self.MIN_COUNT_FOR_GEOMETRY_GUARD:
            second_res = self._best_swap_residual(ref, cand, perm)
            if (second_res - best_res) <= max(1e-3, best_res * ambiguity_gap_ratio):
                print(f"[MaskLabelService] BLOCK: ambiguous reference geometry "
                      f"(best={best_res:.4f}, second={second_res:.4f}, "
                      f"gap_ratio={ambiguity_gap_ratio})")
                return None

        return [components[perm[i]].center for i in range(n)]

    # ============================================================ 内部：几何分配工具
    @staticmethod
    def _shape_signature(pts: np.ndarray) -> np.ndarray:
        """每个点的“到其余点距离排序”签名：平移 / 旋转 / 等比缩放均不变。

        用于给匈牙利提供与手姿态无关的初始代价，
        避免 ICP 在大旋转 / 大缩放下陷入局部最优。
        """
        n = int(pts.shape[0])
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        denom = max(1, n * (n - 1))
        scale = float(np.sqrt((d ** 2).sum() / denom))
        if scale <= 1e-12:
            scale = 1.0
        return np.sort(d / scale, axis=1)

    @classmethod
    def _initial_assignment_by_signature(cls, ref: np.ndarray,
                                          cand: np.ndarray) -> list[int] | None:
        """形状签名代价矩阵 + 匈牙利，得到初始 assignment。"""
        sig_ref = cls._shape_signature(ref)
        sig_cand = cls._shape_signature(cand)
        n = int(ref.shape[0])
        cost = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            diff = sig_cand - sig_ref[i]
            cost[i] = np.sqrt((diff ** 2).sum(axis=1))
        return solve_linear_assignment(cost)

    @classmethod
    def _refine_by_similarity_icp(cls, ref: np.ndarray, cand: np.ndarray,
                                   perm: list[int]) -> list[int] | None:
        """ICP：拟合 similarity -> 按变换后位置重新匈牙利分配 -> 迭代到稳定。"""
        cur = list(perm)
        n = int(ref.shape[0])
        for _ in range(cls.MAX_ICP_ITERATIONS):
            R, s, t = cls._fit_similarity_2d(ref, cand[np.array(cur, dtype=int)])
            mapped = (s * (R @ ref.T)).T + t
            cost = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                diff = cand - mapped[i]
                cost[i] = np.sqrt((diff ** 2).sum(axis=1))
            nxt = solve_linear_assignment(cost)
            if nxt is None:
                return cur
            if nxt == cur:
                return cur
            cur = nxt
        return cur

    @classmethod
    def _residual_of(cls, ref: np.ndarray, cand: np.ndarray, perm: list[int]) -> float:
        """按给定 assignment 拟合 similarity 后的 RMS 残差。"""
        idx = np.array(perm, dtype=int)
        R, s, t = cls._fit_similarity_2d(ref, cand[idx])
        mapped = (s * (R @ ref.T)).T + t
        diff = mapped - cand[idx]
        return float(np.sqrt((diff ** 2).sum(axis=1).mean()))

    @classmethod
    def _best_swap_residual(cls, ref: np.ndarray, cand: np.ndarray,
                             perm: list[int]) -> float:
        """最终 assignment 的所有两两 swap 中最小的残差（second-best 估计）。

        O(N^2) 次 similarity 拟合，替代 O(N!) 全排列枚举。
        N=1 时不存在可交换对，返回 inf（不触发歧义 BLOCK）。
        """
        best = float("inf")
        n = int(ref.shape[0])
        for i in range(n):
            for j in range(i + 1, n):
                swapped = list(perm)
                swapped[i], swapped[j] = swapped[j], swapped[i]
                r = cls._residual_of(ref, cand, swapped)
                if r < best:
                    best = r
        return best

    @staticmethod
    def _fit_similarity_2d(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """Umeyama 2D similarity 拟合：translation + rotation + uniform scale（禁止反射）。

        返回 (R 2x2, scale float, t (2,)) 使 src 经 T(x)=scale*R*x+t 后最接近 dst。
        """
        n = src.shape[0]
        mu_src = src.mean(axis=0)
        mu_dst = dst.mean(axis=0)
        src_c = src - mu_src
        dst_c = dst - mu_dst
        cov = (dst_c.T @ src_c) / n             # (2,2) maps src -> dst（Umeyama: Y'^T X'）
        U, D, Vt = np.linalg.svd(cov)
        d = np.ones(2, dtype=np.float64)
        if np.linalg.det(cov) < 0:
            d[1] = -1.0                          # 强制 proper rotation（det=+1，禁反射）
        R = U @ np.diag(d) @ Vt
        var_src = float((src_c ** 2).sum() / n)
        scale = 1.0 if var_src <= 1e-12 else float((D * d).sum() / var_src)
        t = mu_dst - scale * (R @ mu_src)
        return R, scale, t

    def build_bbox(self,
                  components: list[MaskComponent],
                  image_size: tuple[int, int],
                  min_padding: int = DEFAULT_MIN_BBOX_PADDING,
                  ratio_x: float = DEFAULT_BBOX_PADDING_RATIO_X,
                  ratio_y: float = DEFAULT_BBOX_PADDING_RATIO_Y,
                  ) -> tuple[int, int, int, int]:
        """由 N 个 final MaskComponent.bbox 的 union + 比例 padding 生成 BoundingBox。

        bbox 必须来自 component bbox 的 union，不能用 keypoint 的 min/max。
        keypoint 和 bbox 都来自同一次 finalMask 的 connectedComponentsWithStats 分析。

        padding 规则（避免 bbox 被最外围关键点锁死而贴边）：
            pad_x = max(min_padding, union_w * ratio_x)
            pad_y = max(min_padding, union_h * ratio_y)
        四个方向各自外扩，最后 clamp 到图片范围。

        返回：(x, y, w, h) 像素 bbox，已 clamp 到图内
        """
        if not components:
            return (0, 0, 0, 0)

        w_img, h_img = image_size
        # component bbox 的 union：取所有 bbox 的最小 left/top 和最大 right/bottom
        min_x = min(c.bbox[0] for c in components)
        min_y = min(c.bbox[1] for c in components)
        max_x = max(c.bbox[0] + c.bbox[2] for c in components)
        max_y = max(c.bbox[1] + c.bbox[3] for c in components)

        # ---- 比例扩张（小范围退化为 min_padding 保底）----
        union_w = max_x - min_x
        union_h = max_y - min_y
        pad_x = max(min_padding, union_w * ratio_x)
        pad_y = max(min_padding, union_h * ratio_y)

        min_x = max(0, min_x - pad_x)
        min_y = max(0, min_y - pad_y)
        max_x = min(w_img, max_x + pad_x)
        max_y = min(h_img, max_y + pad_y)
        return (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))

    def make_label_content(self,
                           bbox: tuple[int, int, int, int],
                           ordered_points: list[tuple[float, float]],
                           image_size: tuple[int, int]
                           ) -> str:
        """生成 YOLO Pose 标签内容字符串（不写文件）。

        供 Q 原子 Commit 使用：Controller 先写 image.tmp + label.tmp，
        都成功后再 rename，避免半提交。

        格式与 save_label 一致（按 K0..K(N-1) 顺序输出）：
            class cx cy w h  kp0x kp0y vis  ...  kp(N-1)x kp(N-1)y vis
        """
        w_img, h_img = image_size
        if w_img <= 0 or h_img <= 0:
            return ""

        bx, by, bw, bh = bbox
        cx = (bx + bw / 2.0) / w_img
        cy = (by + bh / 2.0) / h_img
        nw = bw / w_img
        nh = bh / h_img
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))

        kp_tokens = []
        for (px, py) in ordered_points:
            kx = max(0.0, min(1.0, px / w_img))
            ky = max(0.0, min(1.0, py / h_img))
            kp_tokens.append(f"{kx:.6f}")
            kp_tokens.append(f"{ky:.6f}")
            kp_tokens.append("2")

        tokens = ["0",
                  f"{cx:.6f}", f"{cy:.6f}", f"{nw:.6f}", f"{nh:.6f}",
                  *kp_tokens]
        return " ".join(tokens)

    def save_label(self,
                  path: str,
                  bbox: tuple[int, int, int, int],
                  ordered_points: list[tuple[float, float]],
                  image_size: tuple[int, int]
                  ) -> bool:
        """写出 YOLO Pose 标签文件。

        格式（与 C++ tools/generate_bbox_from_keypoints.py 完全一致）：
            class cx cy w h  kp0x kp0y vis  ...  kp(N-1)x kp(N-1)y vis
            每行 1 + 4 + N*3 个数值；归一化 [0,1]；vis 固定为 2

        返回：True 写入成功 / False 失败
        """
        content = self.make_label_content(bbox, ordered_points, image_size)
        if not content:
            return False

        tmp_path = path + ".tmp"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(content + "\n")
            os.replace(tmp_path, path)
            return True
        except OSError:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            return False
