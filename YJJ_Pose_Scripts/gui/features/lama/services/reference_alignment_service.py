"""Reference Alignment 多点局部模板追踪服务（Phase 3）。

忠实迁移 C++ LamaErasure/AlignToReference.cpp/.h 的核心算法：
- SetReference: 拆分独立 Mask + 生成局部模板 + 一对一匹配继承 canonical ID
- predict (TestOne 严格 / A 宽松): 两阶段匹配
    1. Coarse 大范围粗匹配 -> 全局位移中位数
    2. Refined 在预测位置附近小范围精匹配 + 校验 + 重复占用检测

关键约束（来自 LamaController 工作流要求）：
- TestOne 与 A 必须走同一个 predict() API
- stable Track ID：第一次 SetRef 时由 Controller 传入 (y,x) 升序的 ordered_points，
  后续 rolling reference 调用 set_reference 时通过一对一匹配继承 canonical ID，
  绝不重新按 (y,x) 重排。
- predict() 输出的 TrackPrediction.id 始终是 canonical ID（不被 rolling reference 改变）。
- A 键允许部分 Track 成功，并平移参考 mask patch；TestOne 严格，任一失败整张失败
- 不依赖 QWidget / PySide6，纯 numpy + cv2 算法，方便自动测试

参考 C++ AlignToReference:
- SetReference / BuildLocalTracks / CoarseMatchTrack / RefinedMatchTrack
- RunTwoStageMatch / ComposeFinalMask / MakeAssistMaskFor
- ExtractMaskCenters（静态工具）

参数对齐 C++ AlignToReference::Params 默认值（MainWindow::makeAlignParams）。
"""

from __future__ import annotations
import copy

from dataclasses import dataclass, field

import cv2
import numpy as np

from .assignment_solver import solve_linear_assignment


# ============================================================ 数据结构
@dataclass
class LocalTrack:
    """单个独立 Mask 的局部追踪信息（对应 C++ AlignToReference::LocalTrack）。"""
    id: int                                       # canonical Track 编号（0..N-1，由 ordered_points 一对一匹配继承）
    mask_rect: tuple[int, int, int, int]          # 参考图非零包围盒 (x, y, w, h)
    template_rect: tuple[int, int, int, int]      # 扩边后模板区域 (x, y, w, h)
    mask_patch: np.ndarray | None = None          # 二值 Mask 补丁（CV_8UC1 0/255）
    template_gray: np.ndarray | None = None       # 参考灰度模板
    ref_center: tuple[float, float] = (0.0, 0.0)  # mask 中心（参考图坐标）


@dataclass
class TrackPrediction:
    """单个 Track 在目标图的匹配结果（对应 C++ LocalMatchResult 的精简版）。"""
    id: int                                       # canonical Track 编号（与 LocalTrack.id 一致）
    ref_center: tuple[float, float] = (0.0, 0.0)  # 参考图 mask 中心
    current_center: tuple[float, float] = (0.0, 0.0)  # 目标图 mask 中心 = ref_center + (dx, dy)，与 assist mask 一致
    dx: int = 0                                   # X 位移
    dy: int = 0                                   # Y 位移
    score: float = 0.0                            # 匹配分数
    ok: bool = False                              # 是否成功
    fail_reason: str = ""                         # 失败原因


@dataclass
class PredictionResult:
    """predict() 统一返回（A 与 TestOne 共用）。"""
    mask: np.ndarray | None                       # 最终 mask（HxW uint8 0/255），None 表示失败
    tracks: list[TrackPrediction] = field(default_factory=list)  # list[TrackPrediction]
    success: int = 0
    total: int = 0
    global_dx: int = 0
    global_dy: int = 0


@dataclass
class _PreparedReference:
    """内部：预验证通过但尚未 apply 的 Reference State（供 Q 原子 Commit 使用）。

    由 prepare_reference() 生成，apply_prepared_reference() 消费。
    不可被外部直接构造。
    """
    ref_gray: np.ndarray                          # 参考灰度图（已 copy）
    ref_mask: np.ndarray                          # 参考 mask（已 copy）
    ref_points: list[tuple[float, float]]         # ordered_points（已 copy）
    ref_rect: object                             # 参考 BoundingBox（可选）
    local_tracks: list[LocalTrack]                # list[LocalTrack]，ID 已是 canonical
    ready: bool = True


# ============================================================ ReferenceAlignmentService
class ReferenceAlignmentService:
    """多点局部模板追踪：rolling reference + predict。

    用法：
        svc = ReferenceAlignmentService()
        svc.set_reference(ref_rgb, ref_mask_gray, ordered_points)
        # A 键（宽松）：允许部分失败，画圆
        result_a = svc.predict(target_rgb, mode="assist")
        # TestOne（严格）：任一失败整张失败，合成实际 mask 补丁
        result_t = svc.predict(target_rgb, mode="strict")
    """

    # ---- C++ AlignToReference::Params 默认值 ----
    DEFAULT_LOCAL_TEMPLATE_PADDING = 35
    DEFAULT_SEARCH_RADIUS_X = 400
    DEFAULT_SEARCH_RADIUS_Y = 220
    DEFAULT_MIN_LOCAL_SCORE = 0.60
    DEFAULT_MIN_MASK_COMPONENT_AREA = 50
    DEFAULT_DILATE_R = 2
    DEFAULT_ASSIST_MASK_RADIUS = 18
    DEFAULT_REFINE_RADIUS_X = 120
    DEFAULT_REFINE_RADIUS_Y = 90
    DEFAULT_MAX_LOCAL_DEVIATION_X = 140
    DEFAULT_MAX_LOCAL_DEVIATION_Y = 110
    DEFAULT_MIN_TRACK_CENTER_DISTANCE = 35

    def __init__(self,
                 local_template_padding: int = DEFAULT_LOCAL_TEMPLATE_PADDING,
                 search_radius_x: int = DEFAULT_SEARCH_RADIUS_X,
                 search_radius_y: int = DEFAULT_SEARCH_RADIUS_Y,
                 min_local_score: float = DEFAULT_MIN_LOCAL_SCORE,
                 min_mask_component_area: int = DEFAULT_MIN_MASK_COMPONENT_AREA,
                 dilate_r: int = DEFAULT_DILATE_R,
                 assist_mask_radius: int = DEFAULT_ASSIST_MASK_RADIUS,
                 refine_radius_x: int = DEFAULT_REFINE_RADIUS_X,
                 refine_radius_y: int = DEFAULT_REFINE_RADIUS_Y,
                 max_local_deviation_x: int = DEFAULT_MAX_LOCAL_DEVIATION_X,
                 max_local_deviation_y: int = DEFAULT_MAX_LOCAL_DEVIATION_Y,
                 min_track_center_distance: int = DEFAULT_MIN_TRACK_CENTER_DISTANCE,
                 ) -> None:
        self.local_template_padding: int = local_template_padding
        self.search_radius_x: int = search_radius_x
        self.search_radius_y: int = search_radius_y
        self.min_local_score: float = min_local_score
        self.min_mask_component_area: int = min_mask_component_area
        self.dilate_r: int = dilate_r
        self.assist_mask_radius: int = assist_mask_radius
        self.refine_radius_x: int = refine_radius_x
        self.refine_radius_y: int = refine_radius_y
        self.max_local_deviation_x: int = max_local_deviation_x
        self.max_local_deviation_y: int = max_local_deviation_y
        self.min_track_center_distance: int = min_track_center_distance

        # ---- 内部状态 ----
        self._ready: bool = False
        self._ref_gray_a: np.ndarray | None = None             # 参考灰度图（CV_8UC1）
        self._ref_mask_a: np.ndarray | None = None             # 参考 mask（CV_8UC1 0/255）
        self._ref_points_a: list[tuple[float, float]] = []     # canonical ordered_points [(x, y), ...]
        self._ref_rect_a: object | None = None                 # 参考 BoundingBox（可选）
        self._local_tracks: list[LocalTrack] = []              # list[LocalTrack]，id 已是 canonical

    # ============================================================ 公开 API
    def is_ready(self) -> bool:
        return self._ready

    def local_tracks(self) -> list[LocalTrack]:
        return self._local_tracks

    def reference_points(self) -> list[tuple[float, float]]:
        """返回当前 Reference 的 canonical ordered points 副本。

        index 0..N-1 = stable ID 0..N-1（N 为本次 session 的关键点数量），
        与 SetRef/rolling ref 建立的 canonical 顺序一致。
        绝不直接返回内部 list 原对象，避免外部意外修改内部状态。
        """
        return list(self._ref_points_a)

    def prepare_reference(self,
                          ref_rgb: np.ndarray,
                          ref_mask_gray: np.ndarray,
                          ordered_points: list[tuple[float, float]] | None = None,
                          ref_rect: object | None = None,
                          expected_count: int | None = None) -> _PreparedReference | None:
        """预构建并验证新的 Reference State，但不修改当前状态（供 Q 原子 Commit 使用）。

        与 set_reference 算法一致，但只返回 _PreparedReference 或 None，
        不写入 self._local_tracks / self._ready，由 apply_prepared_reference 消费。

        用途：Q 流程先 prepare_reference() 预验证能否成功，再启动后台推理，
        最后文件写成功后再 apply_prepared_reference() 应用，避免半提交。

        参数：
            ordered_points: canonical ordered_points [(x, y), ...]
            expected_count: 本次 session 的关键点数量 N（由 Controller 从 UI 取得）。
                            非 None 时要求 mask component 数 == N 且
                            ordered_points 长度 == N，否则 BLOCK。
                            None 表示沿用“component 数由 mask 自身决定”的历史行为。

        返回：_PreparedReference 或 None（失败）
        """
        if ref_rgb is None or ref_mask_gray is None:
            return None

        # ---- 1. 准备参考灰度图 + mask（拷贝，避免外部修改）----
        ref_gray = self._to_gray(ref_rgb).copy()
        ref_mask = self._to_gray_mask_u8(ref_mask_gray).copy()
        if ref_mask.shape[:2] != ref_gray.shape[:2]:
            return None

        # ---- 2. mask 必须非空 ----
        nz = cv2.findNonZero(ref_mask)
        if nz is None:
            return None

        # ---- 3. 拆分 components + 生成 local tracks（尚未赋 ID）----
        tracks_unsorted = self._build_local_tracks_into(ref_gray, ref_mask)
        if not tracks_unsorted:
            return None

        # ---- 3.5 关键点数量校验：N 由本次 session（UI）决定，不再固定 7 ----
        if expected_count is not None and expected_count > 0:
            if len(tracks_unsorted) != expected_count:
                return None

        # ---- 4. 赋 canonical ID ----
        # 关键约束（本修复后）：
        # - 只要调用方显式传了 ordered_points（ordered_points is not None）：
        #   * 空列表 / 长度 != tracks_unsorted / 一对一匹配失败 → 一律 BLOCK (None)
        #   * 绝不退化为 y/x 排序重新定义 ID，避免跨图 stable ID 被静默篡改
        # - 仅当 ordered_points is None（首次 SetRef 没传 ordered_points 的历史兜底）
        #   才允许 y/x 升序建立 canonical ID。
        if ordered_points is not None:
            # 调用方明确传了 ordered_points：长度必须对得上，否则 BLOCK
            ref_points = list(ordered_points)
            if not ref_points:
                return None
            if len(ref_points) != len(tracks_unsorted):
                return None
            # 一对一最小距离匹配：ordered_points[i] -> 找最近的 component，继承 ID = i
            local_tracks = self._assign_canonical_ids_by_match(
                tracks_unsorted, ref_points)
            if local_tracks is None:
                return None     # 一对一匹配失败 -> BLOCK
        else:
            # ordered_points is None：退化为 (y,x) 升序，仅首次 SetRef 兜底
            ref_points = []
            local_tracks = self._assign_canonical_ids_by_yx(tracks_unsorted)

        return _PreparedReference(
            ref_gray=ref_gray,
            ref_mask=ref_mask,
            ref_points=ref_points,
            ref_rect=ref_rect,
            local_tracks=local_tracks,
            ready=True,
        )

    def apply_prepared_reference(self, prepared: _PreparedReference) -> bool:
        """应用 prepare_reference 预验证过的 Reference State。

        本方法只做赋值，不做任何校验或 I/O，因此不会失败（除非 prepared 为 None）。
        供 Q 原子 Commit 在文件写成功之后调用，保证 ref 更新与文件落盘原子一致。
        """
        if prepared is None:
            return False
        self._ref_gray_a = prepared.ref_gray
        self._ref_mask_a = prepared.ref_mask
        self._ref_points_a = list(prepared.ref_points)
        self._ref_rect_a = prepared.ref_rect
        self._local_tracks = list(prepared.local_tracks)
        self._ready = True
        return True

    def set_reference(self,
                      ref_rgb: np.ndarray,
                      ref_mask_gray: np.ndarray,
                      ordered_points: list[tuple[float, float]] | None = None,
                      ref_rect: object | None = None,
                      expected_count: int | None = None) -> bool:
        """设置基准图与 mask，构建 local tracks（prepare + apply 的便捷封装）。

        算法：
        1. 保存参考灰度图 + mask
        2. 拆分 connected components，过滤噪点
        3. 赋 canonical ID：
           - 有 ordered_points（N 个）：一对一最小距离匹配，每个 component 继承
             ordered_points[i] 对应的 canonical ID = i
           - 无 ordered_points：退化为 (y,x) 升序（仅首次 SetRef 兜底）
        4. 为每个 component 生成局部模板（mask_rect + padding -> template_rect）

        参数：
            ref_rgb:          参考原图（HxWx3 uint8，RGB 或 BGR 均可）
            ref_mask_gray:    参考 mask（HxW uint8，0/255）
            ordered_points:   canonical ordered_points [(x, y), ...]（必须 N 个）
                              首次 SetRef 由 Controller 按 (y,x) 升序传入；
                              后续 rolling ref 由 Q 流程传入上一轮继承的 canonical 顺序
            ref_rect:         参考 BoundingBox (x, y, w, h)（可选，目前未使用）
            expected_count:   本次 session 的关键点数量 N（由 Controller 从 UI 取得）

        返回：True 成功 / False 失败（一对一匹配失败 / 数量 != N / mask 为空）
        """
        prepared = self.prepare_reference(
            ref_rgb, ref_mask_gray, ordered_points, ref_rect, expected_count)
        if prepared is None:
            self._ready = False
            self._local_tracks = []
            return False
        return self.apply_prepared_reference(prepared)

    def predict(self,
                target_rgb: np.ndarray,
                mode: str = "strict") -> PredictionResult:
        """统一 predict API（A 与 TestOne 共用同一算法）。

        与 C++ 一致：
        - mode="strict"：RunTwoStageMatch + 严格校验（任一失败整张失败）
          + 合成实际 mask 补丁（PasteMaskPatch）+ dilate
          对应 C++ MakeMaskFor（TestOne / Batch 使用）
        - mode="assist"：RunTwoStageMatch + 宽松（允许部分成功）
            + 将成功 Track 的参考 mask patch 平移到预测位置
          对应 C++ MakeAssistMaskFor（A 键使用）

        返回 PredictionResult：
            mask:    strict 失败时为 None；assist 至少 1 个成功才有 mask
            tracks:  每个 track 的匹配详情（id/current_center/dx/dy/score/ok/fail_reason）
            success/total/global_dx/global_dy
        """
        empty = PredictionResult(mask=None, tracks=[], success=0, total=0)
        if not self._ready or target_rgb is None:
            return empty

        gray_b = self._to_gray(target_rgb)
        h, w = gray_b.shape[:2]
        # V1：参考图与目标图尺寸必须一致（与 C++ 一致，不做自动缩放）
        if gray_b.shape[:2] != self._ref_gray_a.shape[:2]:
            return empty

        n = len(self._local_tracks)
        tracks, gdx, gdy = self._run_two_stage_match(gray_b, (w, h))
        success = sum(1 for t in tracks if t.ok)

        if mode == "strict":
            # 严格：任一失败整张失败
            if success != n:
                return PredictionResult(mask=None, tracks=tracks,
                                        success=success, total=n,
                                        global_dx=gdx, global_dy=gdy)
            # 合成 finalMask（实际 mask 补丁 + dilate）
            final = self._compose_final_mask(tracks, [True] * n, h, w)
            if self.dilate_r > 0:
                k = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (self.dilate_r * 2 + 1, self.dilate_r * 2 + 1),
                )
                final = cv2.dilate(final, k, iterations=1)
            return PredictionResult(mask=final, tracks=tracks,
                                    success=success, total=n,
                                    global_dx=gdx, global_dy=gdy)

        # ---- assist 模式：继承参考图实际 mask 形状 ----
        if success == 0:
            return PredictionResult(mask=None, tracks=tracks,
                                    success=0, total=n,
                                    global_dx=gdx, global_dy=gdy)

        use = [t.ok for t in tracks]
        final = self._compose_final_mask(tracks, use, h, w)

        return PredictionResult(mask=final, tracks=tracks,
                                success=success, total=n,
                                global_dx=gdx, global_dy=gdy)

    # ============================================================ 静态工具
    @staticmethod
    def extract_mask_centers(mask_gray8: np.ndarray,
                             min_component_area: int = 50) -> list[tuple[float, float]]:
        """从最终 Mask 重新提取 Component 中心（Q 流程唯一真相来源）。

        与 C++ AlignToReference::ExtractMaskCenters 一致：
        - connectedComponentsWithStats 8 连通
        - 跳过 area < min_component_area 的噪点
        - 返回 [(cx, cy), ...]（质心浮点）

        注意：本方法不排序。排序（canonical y/x）由 set_reference 在第一次 SetRef 时做。
        """
        if mask_gray8 is None:
            return []
        m = ReferenceAlignmentService._to_gray_mask_u8_static(mask_gray8)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            m, connectivity=8, ltype=cv2.CV_32S
        )
        if n <= 1:
            return []
        centers = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_component_area:
                continue
            cx = float(centroids[i][0])
            cy = float(centroids[i][1])
            centers.append((cx, cy))
        return centers

    # ============================================================ 内部：构建 local tracks
    def _build_local_tracks_into(self, ref_gray: np.ndarray,
                                  ref_mask: np.ndarray) -> list[LocalTrack]:
        """拆分 components + 生成模板（与 C++ BuildLocalTracks 一致），不赋 ID。

        返回 list[LocalTrack]，每个 track.id 临时为 -1（待 _assign_canonical_ids_* 赋值）。
        不修改 self._local_tracks。返回空 list 表示失败。
        """
        tracks = []

        bin_mask = ref_mask.copy()
        _, bin_mask = cv2.threshold(bin_mask, 127, 255, cv2.THRESH_BINARY)

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bin_mask, connectivity=8, ltype=cv2.CV_32S
        )
        if n <= 1:
            return []

        h, w = ref_gray.shape[:2]
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_mask_component_area:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            mask_rect = (x, y, bw, bh)
            ref_center = (x + bw / 2.0, y + bh / 2.0)

            # mask_rect 四周加 padding -> template_rect，裁剪到图内
            tx = x - self.local_template_padding
            ty = y - self.local_template_padding
            tw = bw + self.local_template_padding * 2
            th = bh + self.local_template_padding * 2
            # clamp 到 [0, W] / [0, H]
            tx = max(0, tx)
            ty = max(0, ty)
            tx2 = min(w, tx + tw)
            ty2 = min(h, ty + th)
            template_rect = (tx, ty, tx2 - tx, ty2 - ty)
            if template_rect[2] < 4 or template_rect[3] < 4:
                continue

            # 模板从参考灰度图截取（绝不用带 overlay 的预览图）
            template_gray = ref_gray[
                ty:ty + template_rect[3], tx:tx + template_rect[2]
            ].copy()
            mask_patch = ref_mask[y:y + bh, x:x + bw].copy()
            if template_gray.size == 0:
                continue

            tracks.append(LocalTrack(
                id=-1,                          # 待赋 canonical ID
                mask_rect=mask_rect,
                template_rect=template_rect,
                mask_patch=mask_patch,
                template_gray=template_gray,
                ref_center=ref_center,
            ))

        return tracks

    def _assign_canonical_ids_by_match(self,
                                        tracks: list[LocalTrack],
                                        ordered_points: list[tuple[float, float]]) -> list[LocalTrack] | None:
        """一对一最小距离 assignment：每个 ordered_points[i] 匹配唯一一个 track，
        track 继承 canonical ID = i。

        算法：构造 N×N 距离代价矩阵，用 O(N^3) 匈牙利求全局最小总距离。
        关键点数量改为 UI 可配置后，N 可达 32，原先的 permutations(N) 全排列
        会阶乘爆炸，因此统一改用匈牙利求解器（N=7 结果与枚举一致）。

        无距离阈值（rolling reference 由用户已确认 final mask，component 应在合理位置）。
        若数量不匹配或无法一对一，返回 None（BLOCK）。
        """
        n = len(ordered_points)
        if n != len(tracks) or n == 0:
            return None

        # 构造 cost 矩阵：cost[i][j] = ordered_points[i] 与 tracks[j].ref_center 距离
        cost = np.zeros((n, n), dtype=np.float64)
        for i, pt in enumerate(ordered_points):
            for j, tr in enumerate(tracks):
                dx = pt[0] - tr.ref_center[0]
                dy = pt[1] - tr.ref_center[1]
                cost[i, j] = (dx * dx + dy * dy) ** 0.5

        # 匈牙利求全局最小一对一分配：best_perm[i] = 第 i 个 ordered_point 匹配的 track 索引
        best_perm = solve_linear_assignment(cost)
        if best_perm is None:
            return None

        # 第 i 个 ordered_point 对应的 track 继承 ID = i
        out = [None] * n
        for i in range(n):
            tr = tracks[best_perm[i]]
            out[i] = LocalTrack(
                id=i,
                mask_rect=tr.mask_rect,
                template_rect=tr.template_rect,
                mask_patch=tr.mask_patch,
                template_gray=tr.template_gray,
                ref_center=tr.ref_center,
            )
        return out

    def _assign_canonical_ids_by_yx(self, tracks: list[LocalTrack]) -> list[LocalTrack]:
        """退化路径：没有 ordered_points 时按 (y, x) 升序赋 ID 0..N-1。

        仅用于首次 SetRef 兜底（Controller 正常路径总会传 ordered_points）。
        后续 rolling reference 不允许走此路径。
        """
        sorted_tracks = sorted(
            tracks, key=lambda t: (t.ref_center[1], t.ref_center[0]))
        out = []
        for i, tr in enumerate(sorted_tracks):
            out.append(LocalTrack(
                id=i,
                mask_rect=tr.mask_rect,
                template_rect=tr.template_rect,
                mask_patch=tr.mask_patch,
                template_gray=tr.template_gray,
                ref_center=tr.ref_center,
            ))
        return out

    # ============================================================ 内部：两阶段匹配
    def _run_two_stage_match(self, gray_b: np.ndarray, img_size: tuple[int, int]) -> tuple[list[TrackPrediction], int, int]:
        """两阶段匹配：Coarse -> 全局中位数 -> Refined -> 校验 -> 重复检测。

        与 C++ RunTwoStageMatch 一致。
        返回 (tracks: list[TrackPrediction], global_dx, global_dy)
        """
        n = len(self._local_tracks)
        tracks = [TrackPrediction(id=t.id, ref_center=t.ref_center)
                  for t in self._local_tracks]
        coarse_valid = [False] * n

        # ---- 第一阶段：Coarse 大范围粗匹配 ----
        ok_coarse_dx = []
        ok_coarse_dy = []
        for i, t in enumerate(self._local_tracks):
            ok, dx, dy, score = self._coarse_match_track(t, gray_b, img_size, tracks[i])
            coarse_valid[i] = ok
            if ok and score >= self.min_local_score:
                ok_coarse_dx.append(dx)
                ok_coarse_dy.append(dy)

        # ---- 全局位移 = 粗位移中位数（抗单点串点）----
        gdx = self._median_int(ok_coarse_dx)
        gdy = self._median_int(ok_coarse_dy)

        # ---- 第二阶段：Refined 在预测位置附近小范围精匹配 ----
        for i, t in enumerate(self._local_tracks):
            if not coarse_valid[i]:
                tracks[i].ok = False
                tracks[i].fail_reason = "coarse_failed"
                continue
            ok = self._refined_match_track(t, gray_b, img_size, gdx, gdy, tracks[i])
            if not ok:
                tracks[i].ok = False
                tracks[i].fail_reason = "refine_window_invalid"
                continue

            # 校验 1：分数过低
            if tracks[i].score < self.min_local_score:
                tracks[i].ok = False
                tracks[i].fail_reason = "low_score"
                continue

            # 校验 2：局部位移相对全局位移偏差过大（疑似串点/离群）
            if (abs(tracks[i].dx - gdx) > self.max_local_deviation_x or
                abs(tracks[i].dy - gdy) > self.max_local_deviation_y):
                tracks[i].ok = False
                tracks[i].fail_reason = "excessive_deviation"
                continue

            tracks[i].ok = True

        # ---- 重复占用检测：两个 Track 当前中心过近 -> 较低分者判失败 ----
        for i in range(n):
            if not tracks[i].ok:
                continue
            for j in range(i + 1, n):
                if not tracks[j].ok:
                    continue
                ddx = tracks[i].current_center[0] - tracks[j].current_center[0]
                ddy = tracks[i].current_center[1] - tracks[j].current_center[1]
                dist = (ddx * ddx + ddy * ddy) ** 0.5
                if dist < self.min_track_center_distance:
                    keep, rej = (i, j) if tracks[i].score >= tracks[j].score else (j, i)
                    tracks[rej].ok = False
                    tracks[rej].fail_reason = "duplicate"

        return tracks, gdx, gdy

    def _coarse_match_track(self, track: LocalTrack, gray_b: np.ndarray,
                            img_size: tuple[int, int], out: TrackPrediction) -> tuple[bool, int, int, float]:
        """第一阶段粗匹配（与 C++ CoarseMatchTrack 一致）。

        返回 (ok, dx, dy, score)；只更新 out 的 coarse_* 字段（这里简化为只返回值）。
        """
        w, h = img_size
        tx, ty, tw, th = track.template_rect
        if tw <= 0 or th <= 0:
            return False, 0, 0, 0.0

        # 粗匹配二维搜索窗口（局部）
        sx1 = max(0, tx - self.search_radius_x)
        sy1 = max(0, ty - self.search_radius_y)
        sx2 = min(w, tx + tw + self.search_radius_x)
        sy2 = min(h, ty + th + self.search_radius_y)
        if sx2 - sx1 < tw or sy2 - sy1 < th:
            return False, 0, 0, 0.0

        search_gray = gray_b[sy1:sy2, sx1:sx2]
        if search_gray.size == 0:
            return False, 0, 0, 0.0

        result = cv2.matchTemplate(search_gray, track.template_gray,
                                   cv2.TM_CCOEFF_NORMED)
        if result.size == 0:
            return False, 0, 0, 0.0
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        cur_x = sx1 + max_loc[0]
        cur_y = sy1 + max_loc[1]
        dx = cur_x - tx
        dy = cur_y - ty
        out.dx = dx
        out.dy = dy
        out.score = float(max_val)
        # P1-5: current_center 必须是预测的 mask component center = ref_center + (dx, dy)
        # 与 predict() assist 模式画圆使用的中心完全一致，避免 template clamping 导致偏差
        out.current_center = (track.ref_center[0] + dx,
                               track.ref_center[1] + dy)
        return True, dx, dy, float(max_val)

    def _refined_match_track(self, track: LocalTrack, gray_b: np.ndarray,
                             img_size: tuple[int, int], gdx: int, gdy: int,
                             out: TrackPrediction) -> bool:
        """第二阶段精匹配（与 C++ RefinedMatchTrack 一致）。"""
        w, h = img_size
        tx, ty, tw, th = track.template_rect
        if tw <= 0 or th <= 0:
            return False

        # 预测模板位置 = 参考模板位置 + 全局位移
        pred_x = tx + gdx
        pred_y = ty + gdy
        sx1 = max(0, pred_x - self.refine_radius_x)
        sy1 = max(0, pred_y - self.refine_radius_y)
        sx2 = min(w, pred_x + tw + self.refine_radius_x)
        sy2 = min(h, pred_y + th + self.refine_radius_y)
        if sx2 - sx1 < tw or sy2 - sy1 < th:
            return False

        search_gray = gray_b[sy1:sy2, sx1:sx2]
        if search_gray.size == 0:
            return False
        result = cv2.matchTemplate(search_gray, track.template_gray,
                                   cv2.TM_CCOEFF_NORMED)
        if result.size == 0:
            return False
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        cur_x = sx1 + max_loc[0]
        cur_y = sy1 + max_loc[1]
        out.dx = cur_x - tx
        out.dy = cur_y - ty
        out.score = float(max_val)
        # P1-5: current_center = ref_center + (dx, dy)，与 assist mask 画圆中心一致
        out.current_center = (track.ref_center[0] + out.dx,
                               track.ref_center[1] + out.dy)
        return True

    # ============================================================ 内部：合成 mask
    def _compose_final_mask(self, tracks: list[TrackPrediction], use: list[bool],
                            h: int, w: int) -> np.ndarray:
        """合并 use[i]=True 的 track maskPatch 到 finalMask（与 C++ ComposeFinalMask 一致）。"""
        final = np.zeros((h, w), dtype=np.uint8)
        for i, t in enumerate(tracks):
            if i >= len(use) or not use[i]:
                continue
            track = self._local_tracks[i]
            dx, dy = t.dx, t.dy
            self._paste_mask_patch(final, track.mask_patch,
                                    track.mask_rect[0] + dx,
                                    track.mask_rect[1] + dy)
        return final

    @staticmethod
    def _paste_mask_patch(final_mask: np.ndarray, patch: np.ndarray,
                          dst_x: int, dst_y: int) -> None:
        """单个 mask 补丁合并到 finalMask（边界裁剪 + bitwise OR）。"""
        if patch is None or final_mask is None:
            return
        if patch.size == 0 or final_mask.size == 0:
            return
        fh, fw = final_mask.shape[:2]
        ph, pw = patch.shape[:2]
        sx0 = sy0 = 0
        dx0, dy0 = dst_x, dst_y
        cw, ch = pw, ph
        if dx0 < 0:
            sx0 = -dx0
            cw -= sx0
            dx0 = 0
        if dy0 < 0:
            sy0 = -dy0
            ch -= sy0
            dy0 = 0
        if dx0 + cw > fw:
            cw = fw - dx0
        if dy0 + ch > fh:
            ch = fh - dy0
        if cw <= 0 or ch <= 0:
            return
        dst_roi = final_mask[dy0:dy0 + ch, dx0:dx0 + cw]
        src_roi = patch[sy0:sy0 + ch, sx0:sx0 + cw]
        cv2.bitwise_or(dst_roi, src_roi, dst_roi)

    # ============================================================ 内部：工具
    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        """转灰度图（与 C++ ToGray 一致）。"""
        if img.ndim == 2:
            return img
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        return img

    @staticmethod
    def _to_gray_mask_u8(mask_gray: np.ndarray) -> np.ndarray:
        """mask 转灰度 0/255 二值化（与 C++ ToGrayMaskU8 一致）。"""
        if mask_gray.ndim == 3:
            if mask_gray.shape[2] == 3:
                m = cv2.cvtColor(mask_gray, cv2.COLOR_RGB2GRAY)
            elif mask_gray.shape[2] == 4:
                m = cv2.cvtColor(mask_gray, cv2.COLOR_RGBA2GRAY)
            else:
                m = mask_gray[..., 0]
        else:
            m = mask_gray
        out = m.copy()
        cv2.threshold(out, 127, 255, cv2.THRESH_BINARY, out)
        return out

    @staticmethod
    def _to_gray_mask_u8_static(mask_gray: np.ndarray) -> np.ndarray:
        """静态版（extract_mask_centers 用）。"""
        return ReferenceAlignmentService._to_gray_mask_u8(mask_gray)

    @staticmethod
    def _median_int(values: list[int]) -> int:
        """计算中位数（与 C++ MedianInt 一致）。"""
        if not values:
            return 0
        s = sorted(values)
        n = len(s)
        if n % 2 == 1:
            return int(s[n // 2])
        return int((s[n // 2 - 1] + s[n // 2]) / 2)
