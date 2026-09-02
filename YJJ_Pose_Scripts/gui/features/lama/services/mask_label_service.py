"""Mask -> YOLO Pose Label 生成服务（Phase 2）。

职责（与 C++ LamaErasure 对齐）：
1. 从 Final Mask 提取 valid connected components
2. component count 必须 == 7（否则 BLOCK，不允许写错误标签）
3. 每个 component：center / bbox / area
4. stable ID assignment：基于 current_prediction 的 ID -> predictedCenter 做一对一匹配
5. 根据 7 个 final mask components 的联合 bbox 生成 BoundingBox（+ padding 20px）
6. 输出 YOLO Pose 标签：
   class cx cy w h  kp1x kp1y vis  kp2x kp2y vis  ...  kp7x kp7y vis
   每行 1 + 4 + 7*3 = 26 个数值；坐标归一化 [0,1]；vis 固定为 2

参考：
- C++ AlignToReference::ExtractMaskCenters（component 提取）
- C++ AlignToReference::BuildLocalTracks（min area 过滤 + 排序）
- C++ MainWindow::onQuickInpaint（占位 bbox = 7 点 min/max + 20px padding）
- C++ tools/generate_bbox_from_keypoints.py（标签格式）

关键约束（来自 LamaController Q 工作流要求）：
- N 固定只有 7
- 每个 ID 只能匹配一个 Component（一对一）
- 匹配不可靠时 BLOCK（返回 None），绝不写错误标签
- 不为 7 个点引入大型第三方框架（自己实现枚举 permutation + 阈值检查）
- bbox 必须来自 7 个 MaskComponent.bbox 的 union，不能用 keypoint 的 min/max

本服务不依赖 QWidget / PySide6，纯算法 + 文件 IO，方便自动测试。
"""
import os
from dataclasses import dataclass, field
from itertools import permutations
from typing import Optional

import cv2
import numpy as np


# ============================================================ 数据结构
@dataclass
class MaskComponent:
    """单个连通区域。"""
    center: tuple            # (cx, cy) 浮点质心（原图像素坐标）
    bbox: tuple              # (x, y, w, h) 整数包围盒
    area: int                # 像素面积


@dataclass
class LabelResult:
    """Q 流程的标签生成结果。"""
    bbox: tuple                       # (x, y, w, h) 像素 bbox（已 clamp 到图内）
    ordered_points: list              # [(x, y), ...] 7 个点，索引 0..6 对应 stable ID
    image_size: tuple                 # (w, h)
    components: list = field(default_factory=list)  # 调试用：原始 component 列表


# ============================================================ MaskLabelService
class MaskLabelService:
    """Final Mask -> YOLO Pose Label 的纯算法服务。"""

    # C++ 默认参数：minMaskComponentArea=50, bbox padding=20
    DEFAULT_MIN_AREA = 50
    DEFAULT_BBOX_PADDING = 20
    EXPECTED_COMPONENT_COUNT = 7      # 工作流固定 7 个 keypoint

    # P0-5: stable ID assignment 单个匹配距离上限（像素）
    # 超过此阈值视为不可靠 -> BLOCK，绝不写错误标签
    DEFAULT_MAX_ASSIGNMENT_DISTANCE = 80.0

    # ============================================================ 公开 API
    def make_label(self,
                  final_mask: np.ndarray,
                  predicted_centers: dict,
                  image_size: tuple,
                  min_area: int = DEFAULT_MIN_AREA,
                  bbox_padding: int = DEFAULT_BBOX_PADDING,
                  max_assignment_distance: float = DEFAULT_MAX_ASSIGNMENT_DISTANCE,
                  ) -> Optional[LabelResult]:
        """从 final mask + 预测中心生成完整标签结果。

        参数：
            final_mask:        二值 mask（HxW，uint8，0/255）
            predicted_centers: dict {track_id: (cx, cy)} 来自 current_prediction
            image_size:        (width, height)
            min_area:          最小 component 面积（过滤噪点）
            bbox_padding:      bbox 在 4 个方向的 padding 像素数
            max_assignment_distance: stable ID 单个匹配距离上限（超过 -> BLOCK）

        返回：
            LabelResult 或 None（component 数 != 7 或 stable ID 匹配失败/距离过大时 BLOCK）
        """
        # ---- 第一步：提取 components ----
        components = self.extract_components(final_mask, min_area)
        if len(components) != self.EXPECTED_COMPONENT_COUNT:
            return None     # 数量不对必须停下，绝不写错误标签

        # ---- 第二步：一对一 stable ID 匹配（全局最小 + 阈值）----
        # predicted_centers: {id: (cx, cy)}；目标是给每个 ID 找到唯一的 component
        ordered_points = self.assign_stable_ids(
            components, predicted_centers, max_assignment_distance)
        if ordered_points is None:
            return None     # 匹配不可靠或距离过大 -> BLOCK

        # ---- 第三步：生成 BoundingBox（7 个 component bbox union + padding + clamp）----
        # P0-4: bbox 必须来自 7 个 MaskComponent.bbox 的 union，不能用 keypoint 的 min/max
        # keypoint 和 bbox 必须来自同一次 finalMask 分析
        bbox = self.build_bbox(components, bbox_padding, image_size)

        return LabelResult(
            bbox=bbox,
            ordered_points=ordered_points,
            image_size=image_size,
            components=components,
        )

    def extract_components(self,
                          mask_gray8: np.ndarray,
                          min_area: int = DEFAULT_MIN_AREA
                          ) -> list:
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
                          components: list,
                          predicted_centers: dict,
                          max_distance: float = DEFAULT_MAX_ASSIGNMENT_DISTANCE,
                          ) -> Optional[list]:
        """一对一最小距离 assignment：每个 ID -> 唯一一个 Component（全局最优）。

        算法（P0-5）：
        - N 固定 7，7! = 5040，直接枚举所有 permutations，选总距离最小的一组
        - 优于贪心：贪心在两点接近/交叉时可能错配，全局最优更稳定
        - 增加可靠性阈值：每个 assignment 距离不能超过 max_distance，否则 None（BLOCK）

        参数：
            components:        list[MaskComponent]
            predicted_centers: dict {track_id(int): (cx, cy)}
            max_distance:      单个匹配距离上限（超过 -> BLOCK）

        返回：
            ordered_points: [(x, y), ...] 长度 7，索引 = track_id（0..6）
            或 None（匹配不可靠 / 距离过大 / 数量不对）
        """
        # ---- 校验：predicted_centers 必须有 7 个 ID（0..6）----
        expected_ids = set(range(self.EXPECTED_COMPONENT_COUNT))
        if set(predicted_centers.keys()) != expected_ids:
            return None

        if len(components) != self.EXPECTED_COMPONENT_COUNT:
            return None

        n = self.EXPECTED_COMPONENT_COUNT

        # ---- 构造 cost 矩阵：cost[i][j] = ID i 与 component j 中心的距离 ----
        cost = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            pc = predicted_centers[i]
            for j in range(n):
                dx = pc[0] - components[j].center[0]
                dy = pc[1] - components[j].center[1]
                cost[i, j] = (dx * dx + dy * dy) ** 0.5

        # ---- 枚举所有 perm：perm[i] = ID i 匹配的 component 索引 ----
        # 7! = 5040，纯 Python 循环也足够快（一次 Q 提交只跑一次）
        best_perm = None
        best_total = float("inf")
        for perm in permutations(range(n)):
            total = 0.0
            for i in range(n):
                total += cost[i, perm[i]]
            if total < best_total:
                best_total = total
                best_perm = perm

        if best_perm is None:
            return None

        # ---- 可靠性检查：每个 assignment 距离不能超过 max_distance ----
        # 任一点距离过大 -> 整体不可靠 -> BLOCK（绝不写错误标签）
        for i in range(n):
            if cost[i, best_perm[i]] > max_distance:
                return None

        # ---- 输出 ordered_points[track_id] = component.center ----
        ordered_points = [None] * n
        for i in range(n):
            ordered_points[i] = components[best_perm[i]].center

        return ordered_points

    def build_bbox(self,
                  components: list,
                  padding: int,
                  image_size: tuple
                  ) -> tuple:
        """由 7 个 final MaskComponent.bbox 的 union + padding 生成 BoundingBox。

        P0-4: 必须使用 7 个 component 的 bbox union，不能用 keypoint 的 min/max。
        keypoint 和 bbox 都来自同一次 finalMask 的 connectedComponentsWithStats 分析。

        与 C++ MainWindow::onQuickInpaint 的占位 bbox 逻辑一致：
        - 7 个 component bbox union + 20px padding
        - clamp 到图片范围 [0, W] / [0, H]

        返回：(x, y, w, h) 像素 bbox，已 clamp 到图内
        """
        if not components:
            return (0, 0, 0, 0)

        w_img, h_img = image_size
        # 7 个 component bbox 的 union：取所有 bbox 的最小 left/top 和最大 right/bottom
        min_x = min(c.bbox[0] for c in components)
        min_y = min(c.bbox[1] for c in components)
        max_x = max(c.bbox[0] + c.bbox[2] for c in components)
        max_y = max(c.bbox[1] + c.bbox[3] for c in components)
        # padding
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding
        # clamp 到图片范围
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(w_img, max_x)
        max_y = min(h_img, max_y)
        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def make_label_content(self,
                           bbox: tuple,
                           ordered_points: list,
                           image_size: tuple
                           ) -> str:
        """生成 YOLO Pose 标签内容字符串（不写文件）。

        供 Q 原子 Commit 使用：Controller 先写 image.tmp + label.tmp，
        都成功后再 rename，避免半提交。

        格式与 save_label 一致：
            class cx cy w h  kp1x kp1y vis  ...  kp7x kp7y vis
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
                  bbox: tuple,
                  ordered_points: list,
                  image_size: tuple
                  ) -> bool:
        """写出 YOLO Pose 标签文件。

        格式（与 C++ tools/generate_bbox_from_keypoints.py 完全一致）：
            class cx cy w h  kp1x kp1y vis  kp2x kp2y vis  ...  kp7x kp7y vis
            每行 1 + 4 + 7*3 = 26 个数值；归一化 [0,1]；vis 固定为 2

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
