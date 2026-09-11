import numpy as np
import cv2
from scipy import ndimage


# ============================================================
# 默认阈值（仅作为默认值，实际由调用方 / CLI 传入）
# ============================================================

DEFAULT_DEPTH_LOW = 1100.0
DEFAULT_DEPTH_HIGH = 1850.0
DEFAULT_MAX_HOLE_RATIO = 0.5
DEFAULT_MIN_VALID_RANGE_RATIO = 0.3

# 有效像素过少时的兜底阈值（避免极小样本被误判为有效）
MIN_VALID_PIXELS = 100

# 原始 Depth 的无效哨兵值（uint16 饱和值）
INVALID_DEPTH_SENTINEL = 65535


# ============================================================
# 无效 Depth 判定（全局统一）
# ============================================================

def invalid_depth_mask(depth: np.ndarray) -> np.ndarray:
    """统一的无效 Depth 判定。

    无效条件（满足任一即为无效）：
        0 / 负数 / 65535 / NaN / Inf

    evaluate_depth / fill_depth_nearest / fill_depth_guided / fill_zihang_filter
    必须共用本函数，不要再各自写一套条件。
    """
    depth_float = np.asarray(depth, dtype=np.float32)

    return (
        (~np.isfinite(depth_float))
        | (depth_float <= 0)
        | (depth_float == INVALID_DEPTH_SENTINEL)
    )


def cast_to_dtype(values: np.ndarray, dtype) -> np.ndarray:
    """把 float 结果安全转回目标 dtype。

    整数 dtype：先 round 再 clip 到该 dtype 的可表示范围，避免溢出 / 截断偏差。
    浮点 dtype：直接转。
    """
    target_dtype = np.dtype(dtype)

    values = np.asarray(values, dtype=np.float32)

    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)

        values = np.clip(np.round(values), info.min, info.max)

    return values.astype(target_dtype)


def sanitize_depth(values: np.ndarray) -> np.ndarray:
    """兜底清理：保证结果里不出现 NaN / Inf，且不出现负值。"""
    values = np.asarray(values, dtype=np.float32)

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    return np.maximum(values, 0.0)


# ============================================================
# 质量评估
# ============================================================

def evaluate_depth(
        depth: np.ndarray,
        depth_low: float = DEFAULT_DEPTH_LOW,
        depth_high: float = DEFAULT_DEPTH_HIGH,
        max_hole_ratio: float = DEFAULT_MAX_HOLE_RATIO,
        min_valid_range_ratio: float = DEFAULT_MIN_VALID_RANGE_RATIO,
) -> tuple[bool, dict]:
    """评估一张 Depth NPY 的质量。

    无效 Depth 由 invalid_depth_mask() 统一判定（0 / 65535 / NaN / Inf）。

    返回：
        (is_valid, metrics)

        metrics:
            total_pixels        总像素数
            invalid_pixels      无效像素数
            hole_ratio          无效像素 / 总像素
            valid_pixels        有效像素数
            valid_range_pixels  有效且落在 [depth_low, depth_high] 的像素数
            valid_range_ratio   valid_range_pixels / valid_pixels
            median_depth_mm     有效像素的中位数（无有效像素时为 None）
    """
    depth_float = np.asarray(depth, dtype=np.float32)

    total_pixels = int(depth_float.size)

    invalid_mask = invalid_depth_mask(depth_float)

    invalid_pixels = int(np.count_nonzero(invalid_mask))

    hole_ratio = (
        invalid_pixels / total_pixels
        if total_pixels > 0
        else 1.0
    )

    valid_values = depth_float[~invalid_mask]

    valid_pixels = int(valid_values.size)

    if valid_pixels > 0:
        valid_range_mask = (
            (valid_values >= depth_low)
            & (valid_values <= depth_high)
        )

        valid_range_pixels = int(np.count_nonzero(valid_range_mask))

        median_depth_mm = float(np.median(valid_values))
    else:
        valid_range_pixels = 0
        median_depth_mm = None

    valid_range_ratio = (
        valid_range_pixels / valid_pixels
        if valid_pixels > 0
        else 0.0
    )

    is_valid = (
        valid_pixels >= MIN_VALID_PIXELS
        and valid_range_ratio >= min_valid_range_ratio
        and hole_ratio <= max_hole_ratio
    )

    metrics = {
        "total_pixels": total_pixels,
        "invalid_pixels": invalid_pixels,
        "hole_ratio": hole_ratio,
        "valid_pixels": valid_pixels,
        "valid_range_pixels": valid_range_pixels,
        "valid_range_ratio": valid_range_ratio,
        "median_depth_mm": median_depth_mm,
    }

    return is_valid, metrics


# ============================================================
# 修补
# ============================================================

def fill_depth_nearest(depth: np.ndarray) -> np.ndarray:
    """最近邻填充（用距离变换找最近的有效像素值）。

    depth: (H, W)，单位任意（本流程为 mm）
    """
    original_dtype = np.asarray(depth).dtype

    depth_float = np.asarray(depth, dtype=np.float32)

    invalid_mask = invalid_depth_mask(depth_float)

    # 全图无效：没有可参考的值，安全返回全 0
    if bool(invalid_mask.all()):
        return np.zeros(depth_float.shape, dtype=original_dtype)

    if not bool(invalid_mask.any()):
        return cast_to_dtype(depth_float, original_dtype)

    indices = ndimage.distance_transform_edt(
        invalid_mask,
        return_distances=False,
        return_indices=True
    )

    filled = depth_float[tuple(indices)]

    depth_float[invalid_mask] = filled[invalid_mask]

    depth_float = sanitize_depth(depth_float)

    return cast_to_dtype(depth_float, original_dtype)


def fill_depth_guided(depth: np.ndarray, guide: np.ndarray) -> np.ndarray:
    """引导滤波填充。

    depth: 深度图
    guide: RGB 图（同尺寸）
    """
    original_dtype = np.asarray(depth).dtype

    depth_float = np.asarray(depth, dtype=np.float32)

    invalid_mask = invalid_depth_mask(depth_float)

    # 全图无效：没有可参考的值，安全返回全 0
    if bool(invalid_mask.all()):
        return np.zeros(depth_float.shape, dtype=original_dtype)

    # 简单填充初值
    depth_filled = np.asarray(
        fill_depth_nearest(depth_float),
        dtype=np.float32,
    )

    guide_gray = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)

    if guide_gray.dtype != np.uint8:
        guide_gray = (guide_gray * 255).astype(np.uint8)

    # 引导滤波（需要 opencv-contrib）
    gf = cv2.ximgproc.createGuidedFilter(
        guide=guide_gray,
        radius=7,
        eps=50
    )

    filtered = np.asarray(gf.filter(depth_filled), dtype=np.float32)

    depth_float[invalid_mask] = filtered[invalid_mask]

    depth_float = sanitize_depth(depth_float)

    return cast_to_dtype(depth_float, original_dtype)


def fill_zihang_filter(depth: np.ndarray) -> np.ndarray:
    """空洞修补流程：

        原始 depth
        -> 0 / 65535 / NaN / Inf 统一判为无效（invalid_depth_mask）
        -> 无效位置置 0
        -> cv2.inpaint
        -> bilateralFilter
        -> 输出修补后的 depth

    保证：
        shape 与输入一致
        dtype 与输入一致（整数输入 cast 回前先 round）
        输出不含 NaN / Inf / 负值
        整张图没有任何有效深度时返回全 0，不崩溃
    """
    depth_input = np.asarray(depth)

    original_dtype = depth_input.dtype

    depth_float = depth_input.astype(np.float32)

    invalid_mask = invalid_depth_mask(depth_float)

    # 整张图没有有效深度：inpaint 无源可依，安全返回全 0
    if bool(invalid_mask.all()):
        return np.zeros(depth_input.shape, dtype=original_dtype)

    depth_fill = depth_float.copy()
    depth_fill[invalid_mask] = 0.0

    mask = invalid_mask.astype(np.uint8)

    depth_fill = cv2.inpaint(
        depth_fill,
        mask,
        3,
        cv2.INPAINT_TELEA
    )

    depth_fill = np.asarray(depth_fill, dtype=np.float32)

    depth_blur = cv2.bilateralFilter(depth_fill, 9, 50, 50)

    depth_blur = sanitize_depth(depth_blur)

    return cast_to_dtype(depth_blur, original_dtype)
