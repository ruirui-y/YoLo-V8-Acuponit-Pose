import sys
import csv
from pathlib import Path

import cv2
import numpy as np

# 把项目根目录加入 Python 搜索路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


# ============================================================
# 基础计算
# ============================================================

def calculate_iou(box1, box2) -> float:
    """
    计算两个 xyxy 格式框的 IoU。
    box = [x1, y1, x2, y2]
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    inter_w = max(0.0, x2_inter - x1_inter)
    inter_h = max(0.0, y2_inter - y1_inter)
    inter_area = inter_w * inter_h

    box1_area = max(0.0, x2_1 - x1_1) * max(0.0, y2_1 - y1_1)
    box2_area = max(0.0, x2_2 - x1_2) * max(0.0, y2_2 - y1_2)

    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def calculate_bbox_diagonal(box) -> float:
    """
    GT bbox 对角线长度。

    NME 使用它作为归一化尺度：
        NME = keypoint_pixel_error / GT_bbox_diagonal
    """
    x1, y1, x2, y2 = box

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    return float(np.sqrt(width ** 2 + height ** 2))


# ============================================================
# GT
# ============================================================

def load_pose_label(
        label_path: Path,
        image_width: int,
        image_height: int
) -> dict:
    """
    读取单目标 YOLO Pose 标签。

    格式：
        class cx cy w h k0x k0y v0 k1x k1y v1 ...

    所有计算坐标保持 float，不进行 int 截断。
    """
    if not label_path.exists():
        raise FileNotFoundError(f"GT label 不存在: {label_path}")

    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(lines) != 1:
        raise ValueError(
            f"当前单张 NME 测试只支持 1 个 GT 实例，"
            f"实际发现 {len(lines)} 个: {label_path}"
        )

    parts = list(map(float, lines[0].split()))

    if len(parts) < 8:
        raise ValueError(f"Pose label 字段数量不足: {label_path}")

    cls_id, cx, cy, bw, bh = parts[:5]

    keypoint_values = parts[5:]

    if len(keypoint_values) % 3 != 0:
        raise ValueError(
            f"关键点字段无法按 x y v 分组: {label_path}"
        )

    keypoints = np.asarray(
        keypoint_values,
        dtype=np.float64
    ).reshape(-1, 3)

    # YOLO normalized bbox -> pixel bbox
    x1 = (cx - bw / 2.0) * image_width
    y1 = (cy - bh / 2.0) * image_height
    x2 = (cx + bw / 2.0) * image_width
    y2 = (cy + bh / 2.0) * image_height

    gt_box = np.asarray(
        [x1, y1, x2, y2],
        dtype=np.float64
    )

    gt_keypoints = keypoints.copy()

    gt_keypoints[:, 0] *= image_width
    gt_keypoints[:, 1] *= image_height

    return {
        "class_id": int(cls_id),
        "box": gt_box,
        "keypoints": gt_keypoints,
    }


# ============================================================
# Depth NPY（复用 01_data 的 color_ / depth_ 时间戳配对约定）
# ============================================================

# 与 prepare_pose_rgbd_dataset.py / rebuild_cross_subject_testset.py 保持一致：
#     RGB   图 : color_<timestamp>.<ext>
#     Depth 项 : depth_<timestamp>.npy    (uint16, 单位 mm, invalid = 0 / 65535)
RGB_PREFIX = "color_"
DEPTH_PREFIX = "depth_"

# depth npy 候选目录（相对 dataset_root，按顺序查找，取第一个命中的文件）。
# 说明：发布后的数据集只保留 rgb/ 与 rgbd/（4 通道 PNG），原始 npy 不随数据集发布，
# 所以这里同时覆盖「数据集自带 npy」与「同级/上级 session 的 out_npy」两种现有布局。
DEPTH_NPY_RELATIVE_DIRS = (
    "npy",
    "depth",
    "out_npy",
    "rgbd/npy",
    "rgb/npy",
    "../out_npy",
    "../../out_npy",
)


def extract_timestamp(stem: str, prefix: str) -> str:
    """复用 01_data/ts_of 的时间戳约定：去掉前缀后的剩余部分即完整时间戳。"""
    if stem.startswith(prefix):
        return stem[len(prefix):]

    return stem


def build_depth_npy_name(image_stem: str) -> str:
    """固定映射规则（不 nearest、不容差、不按序号）：

        color_<完整时间戳>.<ext>   ->   depth_<完整时间戳>.npy

    例：
        color_2026_07_28_20_03_48_116464
        -> depth_2026_07_28_20_03_48_116464.npy
    """
    timestamp = extract_timestamp(image_stem, RGB_PREFIX)

    return f"{DEPTH_PREFIX}{timestamp}.npy"


def count_depth_npy_files(depth_dir: Path) -> int:
    """统计目录内 depth_*.npy 文件数量（批量开始时的提示信息用）。"""
    if not depth_dir.is_dir():
        return 0

    return sum(
        1
        for path in depth_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".npy"
        and path.stem.startswith(DEPTH_PREFIX)
    )


def collect_depth_npy_dirs(dataset_root: Path, depth_npy_dir=None) -> list:
    """返回可用的 depth npy 候选目录列表。

    优先级（显式最高）：
        1. 传入 depth_npy_dir（--depth-dir）时，只使用该目录，不再自动猜测；
        2. 否则 fallback：收集 dataset_root 下（含上级）实际存在的约定目录。
    全部不存在时返回空列表。
    """
    if depth_npy_dir is not None:
        return [Path(depth_npy_dir)]

    depth_dirs = []

    for relative in DEPTH_NPY_RELATIVE_DIRS:
        candidate = dataset_root / relative

        if candidate.is_dir():
            depth_dirs.append(candidate)

    return depth_dirs


def resolve_depth_npy_path(depth_dirs, image_stem: str):
    """按候选目录顺序查找 depth_<timestamp>.npy。

    目标文件名由 build_depth_npy_name() 决定（固定映射规则）。
    找不到返回 None，由调用方降级为 N/A，不抛异常。
    """
    depth_name = build_depth_npy_name(image_stem)

    for depth_dir in depth_dirs:
        npy_path = depth_dir / depth_name

        if npy_path.is_file():
            return npy_path

    return None


def calculate_depth_stats_in_box(gt_box, depth: np.ndarray) -> dict:
    """在 GT bbox 范围内统计有效 Depth 值（单位 mm）。

    bbox 先裁剪到图像边界；valid_ratio 的分母 = 裁剪后 bbox 的总像素数。

    有效 Depth 定义（本脚本固定）：
        np.isfinite(depth) and depth > 0

    返回：
        median_mm   : 有效 Depth 的中位数（无有效值时 None）
        valid_ratio : 有效像素 / 裁剪后 bbox 总像素（无有效值时 0.0）
    """
    image_height, image_width = depth.shape[:2]

    x1 = int(np.clip(np.floor(gt_box[0]), 0, image_width))
    y1 = int(np.clip(np.floor(gt_box[1]), 0, image_height))
    x2 = int(np.clip(np.ceil(gt_box[2]), 0, image_width))
    y2 = int(np.clip(np.ceil(gt_box[3]), 0, image_height))

    region = depth[y1:y2, x1:x2]

    total_pixels = int(region.size)

    if total_pixels <= 0:
        return {
            "median_mm": None,
            "valid_ratio": 0.0,
        }

    valid_mask = np.isfinite(region) & (region > 0)

    valid_count = int(np.count_nonzero(valid_mask))

    if valid_count == 0:
        return {
            "median_mm": None,
            "valid_ratio": 0.0,
        }

    return {
        "median_mm": float(np.median(region[valid_mask])),
        "valid_ratio": valid_count / total_pixels,
    }


def load_gt_bbox_depth_stats(gt_box, npy_path) -> dict:
    """读取 depth npy 并统计 GT bbox 内 Depth。

    npy 缺失 / 读取失败 / 维度异常 -> 返回 None 值，
    绝不中断批量评估。
    """
    empty_stats = {
        "median_mm": None,
        "valid_ratio": 0.0,
    }

    if npy_path is None:
        return empty_stats

    try:
        depth = np.load(str(npy_path))
    except Exception as exc:
        print(
            f"[depth] 读取失败，该图片 Depth 记为 N/A: "
            f"{npy_path} ({exc})"
        )

        return empty_stats

    if depth.ndim == 3:
        depth = depth[..., 0]

    if depth.ndim != 2:
        print(
            f"[depth] 维度异常(期望 2D)，该图片 Depth 记为 N/A: "
            f"{npy_path} shape={depth.shape}"
        )

        return empty_stats

    return calculate_depth_stats_in_box(gt_box, depth)


# ============================================================
# 模型推理
# ============================================================

def run_pose_model(
        model: YOLO,
        image: np.ndarray,
        conf_threshold: float,
        imgsz: int
):
    """
    保持当前项目的基础推理方式。
    """
    results = model(
        image,
        conf=conf_threshold,
        imgsz=imgsz,
        rect=False,
        augment=False,
        verbose=False,
    )

    return results[0]


def find_best_prediction(
        result,
        gt_box,
        match_iou_threshold: float
) -> dict:
    """
    不再默认取第 0 个 prediction。

    将所有预测框与 GT bbox 计算 IoU，
    选择 IoU 最大的预测实例。

    boxes 与 keypoints 按相同 index 对应。
    """
    output = {
        "matched": False,
        "index": None,
        "box": None,
        "keypoints": None,
        "confidence": None,
        "iou": 0.0,
    }

    if result.boxes is None or len(result.boxes) == 0:
        return output

    if result.keypoints is None or len(result.keypoints) == 0:
        return output

    pred_boxes = result.boxes.xyxy.cpu().numpy().astype(np.float64)

    best_index = None
    best_iou = -1.0

    for index, pred_box in enumerate(pred_boxes):
        current_iou = calculate_iou(gt_box, pred_box)

        if current_iou > best_iou:
            best_iou = current_iou
            best_index = index

    if best_index is None:
        return output

    output["index"] = int(best_index)
    output["box"] = pred_boxes[best_index]
    output["iou"] = float(best_iou)

    if result.boxes.conf is not None:
        output["confidence"] = float(
            result.boxes.conf[best_index].cpu().item()
        )

    if best_iou < match_iou_threshold:
        return output

    keypoint_data = (
        result.keypoints.data[best_index]
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    output["matched"] = True
    output["keypoints"] = keypoint_data

    return output


# ============================================================
# NME
# ============================================================

def calculate_nme(
        gt_box,
        gt_keypoints,
        pred_keypoints
) -> dict:
    """
    NME 定义：

        d_k = sqrt(
            (pred_x - gt_x)^2 +
            (pred_y - gt_y)^2
        )

        nme_k = d_k / GT_bbox_diagonal

        overall_nme = mean(valid nme_k)

    GT visibility > 0 的点才参与计算。
    """
    if len(gt_keypoints) != len(pred_keypoints):
        raise ValueError(
            f"GT / Pred 关键点数量不同: "
            f"{len(gt_keypoints)} != {len(pred_keypoints)}"
        )

    bbox_diagonal = calculate_bbox_diagonal(gt_box)

    if bbox_diagonal <= 0:
        raise ValueError("GT bbox diagonal <= 0，无法计算 NME")

    pixel_errors = []
    normalized_errors = []
    valid_indices = []

    for index in range(len(gt_keypoints)):
        gt_x, gt_y, visibility = gt_keypoints[index]

        # v <= 0 不参加
        if visibility <= 0:
            continue

        pred_x = pred_keypoints[index][0]
        pred_y = pred_keypoints[index][1]

        pixel_error = float(
            np.sqrt(
                (pred_x - gt_x) ** 2 +
                (pred_y - gt_y) ** 2
            )
        )

        normalized_error = pixel_error / bbox_diagonal

        valid_indices.append(index)
        pixel_errors.append(pixel_error)
        normalized_errors.append(normalized_error)

    if not normalized_errors:
        raise ValueError("没有可用于 NME 的有效 GT 关键点")

    return {
        "bbox_diagonal": bbox_diagonal,
        "valid_indices": valid_indices,
        "pixel_errors": pixel_errors,
        "normalized_errors": normalized_errors,
        "mean_pixel_error": float(np.mean(pixel_errors)),
        "nme": float(np.mean(normalized_errors)),
    }


# ============================================================
# 单点像素误差（K0..K{N-1}）
#
# 直接复用 calculate_nme() 已经算好的 pixel_errors，
# 与控制台打印的 "Kx: xx.xx px" 同源，不重新定义误差公式。
# ============================================================

def extract_keypoint_pixel_errors(nme_result, keypoint_count: int) -> list:
    """从已有 NME 结果中取出每个关键点的像素误差。

    返回长度固定为 keypoint_count 的列表：
        有效点（GT visibility > 0 且成功匹配） -> float 像素误差
        未匹配 / visibility <= 0              -> None

    未匹配时 nme_result 为 None，整列返回 None；
    缺失值一律用 None 表示，不使用 0 代替。
    """
    pixel_errors = [None] * keypoint_count

    if nme_result is None:
        return pixel_errors

    for index, pixel_error in zip(
            nme_result["valid_indices"],
            nme_result["pixel_errors"],
    ):
        if 0 <= index < keypoint_count:
            pixel_errors[index] = float(pixel_error)

    return pixel_errors


def build_keypoint_pixel_fieldnames(keypoint_count: int) -> list:
    """按关键点数量生成字段名，避免写死 7 组重复字段。

    顺序：
        rgb_k0_px .. rgb_k{N-1}_px
        rgbd_k0_px .. rgbd_k{N-1}_px
        delta_k0_px .. delta_k{N-1}_px
    """
    fieldnames = []

    for index in range(keypoint_count):
        fieldnames.append(f"rgb_k{index}_px")

    for index in range(keypoint_count):
        fieldnames.append(f"rgbd_k{index}_px")

    for index in range(keypoint_count):
        fieldnames.append(f"delta_k{index}_px")

    return fieldnames


def build_keypoint_pixel_fields(
        rgb_keypoint_px: list,
        rgbd_keypoint_px: list,
) -> dict:
    """生成单点像素误差字段。

    delta_kX_px = rgbd_kX_px - rgb_kX_px
        delta < 0 => RGBD 更准
        delta > 0 => RGB 更准

    只有 RGB / RGBD 两侧该点都有有效误差时才计算 delta，
    否则 delta_kX_px = None。
    """
    keypoint_count = max(
        len(rgb_keypoint_px),
        len(rgbd_keypoint_px),
    )

    fields = {}

    for index in range(keypoint_count):
        rgb_px = (
            rgb_keypoint_px[index]
            if index < len(rgb_keypoint_px)
            else None
        )

        rgbd_px = (
            rgbd_keypoint_px[index]
            if index < len(rgbd_keypoint_px)
            else None
        )

        fields[f"rgb_k{index}_px"] = rgb_px
        fields[f"rgbd_k{index}_px"] = rgbd_px

        if rgb_px is not None and rgbd_px is not None:
            fields[f"delta_k{index}_px"] = rgbd_px - rgb_px
        else:
            fields[f"delta_k{index}_px"] = None

    return fields


# ============================================================
# 可视化
# ============================================================

def draw_gt(
        image: np.ndarray,
        gt_box,
        gt_keypoints
) -> np.ndarray:
    canvas = image.copy()

    x1, y1, x2, y2 = np.round(gt_box).astype(int)

    cv2.rectangle(
        canvas,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2,
    )

    for index, (x, y, visibility) in enumerate(gt_keypoints):
        if visibility <= 0:
            continue

        px = int(round(x))
        py = int(round(y))

        cv2.circle(
            canvas,
            (px, py),
            6,
            (255, 0, 255),
            -1,
        )

        cv2.putText(
            canvas,
            f"GT{index}",
            (px + 7, py - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2,
        )

    return canvas


def draw_prediction(
        image: np.ndarray,
        prediction: dict,
        prefix: str,
        nme_result: dict = None
) -> np.ndarray:
    canvas = image.copy()

    if prediction["box"] is not None:
        x1, y1, x2, y2 = np.round(
            prediction["box"]
        ).astype(int)

        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

    if not prediction["matched"]:
        cv2.putText(
            canvas,
            f"{prefix}: UNMATCHED",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        return canvas

    for index, keypoint in enumerate(prediction["keypoints"]):
        x = int(round(keypoint[0]))
        y = int(round(keypoint[1]))

        cv2.circle(
            canvas,
            (x, y),
            5,
            (0, 255, 255),
            -1,
        )

        cv2.putText(
            canvas,
            f"K{index}",
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )

    text = f"{prefix} IoU: {prediction['iou']:.4f}"

    if nme_result is not None:
        text += f"  NME: {nme_result['nme']:.4f}"

    cv2.putText(
        canvas,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )

    return canvas


def draw_combined(
        image: np.ndarray,
        gt_keypoints,
        rgb_prediction: dict,
        rgbd_prediction: dict,
        rgb_nme: dict = None,
        rgbd_nme: dict = None
) -> np.ndarray:
    """
    同一张 RGB 图显示：

        GT   = 洋红圆
        RGB  = 红色叉
        RGBD = 黄色十字

    同时从 GT 向两个预测点画误差连线。
    """
    canvas = image.copy()

    for index, gt_kpt in enumerate(gt_keypoints):
        gt_x, gt_y, visibility = gt_kpt

        if visibility <= 0:
            continue

        gx = int(round(gt_x))
        gy = int(round(gt_y))

        # GT
        cv2.circle(
            canvas,
            (gx, gy),
            6,
            (255, 0, 255),
            -1,
        )

        # RGB
        if rgb_prediction["matched"]:
            pred = rgb_prediction["keypoints"][index]

            rx = int(round(pred[0]))
            ry = int(round(pred[1]))

            cv2.line(
                canvas,
                (gx, gy),
                (rx, ry),
                (0, 0, 255),
                1,
            )

            cv2.drawMarker(
                canvas,
                (rx, ry),
                (0, 0, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=12,
                thickness=2,
            )

        # RGBD
        if rgbd_prediction["matched"]:
            pred = rgbd_prediction["keypoints"][index]

            dx = int(round(pred[0]))
            dy = int(round(pred[1]))

            cv2.line(
                canvas,
                (gx, gy),
                (dx, dy),
                (0, 255, 255),
                1,
            )

            cv2.drawMarker(
                canvas,
                (dx, dy),
                (0, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2,
            )

        cv2.putText(
            canvas,
            f"K{index}",
            (gx + 7, gy - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

    y = 30

    cv2.putText(
        canvas,
        "GT: magenta  RGB: red X  RGBD: yellow +",
        (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    y += 28

    if rgb_nme is not None:
        cv2.putText(
            canvas,
            f"RGB NME: {rgb_nme['nme']:.4f}",
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )

        y += 25

    if rgbd_nme is not None:
        cv2.putText(
            canvas,
            f"RGBD NME: {rgbd_nme['nme']:.4f}",
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

    return canvas


def make_depth_visualization(rgbd_image: np.ndarray) -> np.ndarray:
    """
    将第 4 通道仅用于人眼观察。
    不参与 NME 计算。
    """
    depth = rgbd_image[:, :, 3]

    depth_float = depth.astype(np.float32)

    min_value = float(depth_float.min())
    max_value = float(depth_float.max())

    if max_value > min_value:
        depth_norm = (
            (depth_float - min_value)
            / (max_value - min_value)
            * 255.0
        ).astype(np.uint8)
    else:
        depth_norm = np.zeros_like(depth, dtype=np.uint8)

    return cv2.applyColorMap(
        depth_norm,
        cv2.COLORMAP_TURBO,
    )


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    canvas = image.copy()

    cv2.rectangle(
        canvas,
        (0, 0),
        (canvas.shape[1], 45),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        canvas,
        title,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    return canvas


def resize_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# 单张 RGB vs RGBD
# ============================================================

def evaluate_single_rgb_vs_rgbd(
        rgb_model_path: str,
        rgbd_model_path: str,
        rgb_image_path: str,
        rgbd_image_path: str,
        gt_label_path: str,
        conf_threshold: float = 0.25,
        imgsz: int = 640,
        match_iou_threshold: float = 0.50,
        save_path: str = "rgb_vs_rgbd_nme.jpg",
) -> dict:

    rgb_image_path = Path(rgb_image_path)
    rgbd_image_path = Path(rgbd_image_path)
    gt_label_path = Path(gt_label_path)

    # --------------------------------------------------------
    # 读取图像
    # --------------------------------------------------------

    rgb_image = cv2.imread(
        str(rgb_image_path),
        cv2.IMREAD_COLOR,
    )

    if rgb_image is None:
        raise FileNotFoundError(
            f"RGB 图像读取失败: {rgb_image_path}"
        )

    rgbd_image = cv2.imread(
        str(rgbd_image_path),
        cv2.IMREAD_UNCHANGED,
    )

    if rgbd_image is None:
        raise FileNotFoundError(
            f"RGBD 图像读取失败: {rgbd_image_path}"
        )

    if rgbd_image.ndim != 3 or rgbd_image.shape[2] != 4:
        raise ValueError(
            f"RGBD 图像必须为 4 通道，实际 shape={rgbd_image.shape}"
        )

    if rgb_image.shape[:2] != rgbd_image.shape[:2]:
        raise ValueError(
            "RGB / RGBD 图像尺寸不同"
        )

    height, width = rgb_image.shape[:2]

    # --------------------------------------------------------
    # GT
    # --------------------------------------------------------

    gt = load_pose_label(
        gt_label_path,
        width,
        height,
    )

    print("=" * 70)
    print("GT")
    print("=" * 70)
    print(f"Image size     : {width} x {height}")
    print(f"GT bbox        : {gt['box']}")
    print(f"Keypoint count : {len(gt['keypoints'])}")
    print(
        f"BBox diagonal  : "
        f"{calculate_bbox_diagonal(gt['box']):.4f} px"
    )

    # --------------------------------------------------------
    # 模型
    # --------------------------------------------------------

    print("\n加载 RGB 模型...")
    rgb_model = YOLO(
        rgb_model_path,
        task="pose",
    )

    print("加载 RGBD 模型...")
    rgbd_model = YOLO(
        rgbd_model_path,
        task="pose",
    )

    # --------------------------------------------------------
    # 推理
    # --------------------------------------------------------

    rgb_result = run_pose_model(
        rgb_model,
        rgb_image,
        conf_threshold,
        imgsz,
    )

    rgbd_result = run_pose_model(
        rgbd_model,
        rgbd_image,
        conf_threshold,
        imgsz,
    )

    # --------------------------------------------------------
    # GT instance matching
    # --------------------------------------------------------

    rgb_prediction = find_best_prediction(
        rgb_result,
        gt["box"],
        match_iou_threshold,
    )

    rgbd_prediction = find_best_prediction(
        rgbd_result,
        gt["box"],
        match_iou_threshold,
    )

    # --------------------------------------------------------
    # NME
    # --------------------------------------------------------

    rgb_nme = None
    rgbd_nme = None

    if rgb_prediction["matched"]:
        rgb_nme = calculate_nme(
            gt["box"],
            gt["keypoints"],
            rgb_prediction["keypoints"],
        )

    if rgbd_prediction["matched"]:
        rgbd_nme = calculate_nme(
            gt["box"],
            gt["keypoints"],
            rgbd_prediction["keypoints"],
        )

    # --------------------------------------------------------
    # 打印结果
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RGB")
    print("=" * 70)

    print(f"Matched : {rgb_prediction['matched']}")
    print(f"Box IoU : {rgb_prediction['iou']:.4f}")

    if rgb_nme is not None:
        for index, pixel_error, normalized_error in zip(
                rgb_nme["valid_indices"],
                rgb_nme["pixel_errors"],
                rgb_nme["normalized_errors"]
        ):
            print(
                f"K{index}: "
                f"{pixel_error:.2f} px, "
                f"NME={normalized_error:.6f}"
            )

        print(
            f"RGB mean pixel error : "
            f"{rgb_nme['mean_pixel_error']:.4f} px"
        )

        print(
            f"RGB Overall NME      : "
            f"{rgb_nme['nme']:.6f}"
        )

    print("\n" + "=" * 70)
    print("RGBD")
    print("=" * 70)

    print(f"Matched : {rgbd_prediction['matched']}")
    print(f"Box IoU : {rgbd_prediction['iou']:.4f}")

    if rgbd_nme is not None:
        for index, pixel_error, normalized_error in zip(
                rgbd_nme["valid_indices"],
                rgbd_nme["pixel_errors"],
                rgbd_nme["normalized_errors"]
        ):
            print(
                f"K{index}: "
                f"{pixel_error:.2f} px, "
                f"NME={normalized_error:.6f}"
            )

        print(
            f"RGBD mean pixel error : "
            f"{rgbd_nme['mean_pixel_error']:.4f} px"
        )

        print(
            f"RGBD Overall NME      : "
            f"{rgbd_nme['nme']:.6f}"
        )

    # --------------------------------------------------------
    # 公平成对比较
    # --------------------------------------------------------

    delta_nme = None
    winner = None

    if rgb_nme is not None and rgbd_nme is not None:
        # 固定定义：
        # delta = RGBD - RGB
        delta_nme = rgbd_nme["nme"] - rgb_nme["nme"]

        if delta_nme < 0:
            winner = "RGBD"
        elif delta_nme > 0:
            winner = "RGB"
        else:
            winner = "TIE"

        print("\n" + "=" * 70)
        print("RGB vs RGBD")
        print("=" * 70)

        print(f"RGB NME   : {rgb_nme['nme']:.6f}")
        print(f"RGBD NME  : {rgbd_nme['nme']:.6f}")
        print(f"Delta NME : {delta_nme:+.6f}")
        print(f"Winner    : {winner}")

    else:
        print("\n无法进行公平 NME 成对比较：")
        print(
            "必须 RGB 和 RGBD 都成功匹配 GT 实例。"
        )

    # --------------------------------------------------------
    # 可视化
    # --------------------------------------------------------

    gt_vis = draw_gt(
        rgb_image,
        gt["box"],
        gt["keypoints"],
    )

    rgb_vis = draw_prediction(
        rgb_image,
        rgb_prediction,
        "RGB",
        rgb_nme,
    )

    rgbd_vis = draw_prediction(
        rgbd_image[:, :, :3],
        rgbd_prediction,
        "RGBD",
        rgbd_nme,
    )

    combined_vis = draw_combined(
        rgb_image,
        gt["keypoints"],
        rgb_prediction,
        rgbd_prediction,
        rgb_nme,
        rgbd_nme,
    )

    depth_vis = make_depth_visualization(
        rgbd_image
    )

    original_vis = rgb_image.copy()

    # 统一面板尺寸
    panel_width = 640
    panel_height = 480

    panels = [
        add_title(original_vis, "Original RGB"),
        add_title(depth_vis, "Depth Channel"),
        add_title(gt_vis, "Ground Truth"),
        add_title(rgb_vis, "RGB Prediction"),
        add_title(rgbd_vis, "RGBD Prediction"),
        add_title(combined_vis, "GT vs RGB vs RGBD"),
    ]

    panels = [
        resize_panel(
            panel,
            panel_width,
            panel_height,
        )
        for panel in panels
    ]

    row1 = np.hstack(
        panels[:3]
    )

    row2 = np.hstack(
        panels[3:]
    )

    final_vis = np.vstack(
        [row1, row2]
    )

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(save_path),
        final_vis,
    )

    print("\n" + "=" * 70)
    print(f"可视化已保存: {save_path}")

    return {
        "image": rgb_image_path.name,

        "gt": gt,

        "rgb": {
            "prediction": rgb_prediction,
            "nme": rgb_nme,
        },

        "rgbd": {
            "prediction": rgbd_prediction,
            "nme": rgbd_nme,
        },

        "comparison": {
            "common_matched": (
                rgb_nme is not None
                and rgbd_nme is not None
            ),
            "delta_nme": delta_nme,
            "winner": winner,
        },
    }


def find_file_by_stem(directory: Path, stem: str, suffixes) -> Path:
    """
    根据 stem 自动寻找文件，不要求 RGB / RGBD 扩展名相同。
    """
    directory = Path(directory)

    for suffix in suffixes:
        path = directory / f"{stem}{suffix}"
        if path.exists():
            return path

    raise FileNotFoundError(
        f"找不到文件: directory={directory}, stem={stem}, suffixes={suffixes}"
    )

def evaluate_all_test(
        dataset_root: str,
        rgb_model_path: str,
        rgbd_model_path: str,
        conf_threshold: float = 0.001,
        imgsz: int = 640,
        match_iou_threshold: float = 0.50,
        output_dir: str = "runs/pose_nme/all_test",
        depth_npy_dir=None,
):
    dataset_root = Path(dataset_root)

    rgb_dir = dataset_root / "rgb" / "images" / "test"
    rgbd_dir = dataset_root / "rgbd" / "images" / "test"
    label_dir = dataset_root / "rgb" / "labels" / "test"

    output_dir = Path(output_dir)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Depth NPY 目录
    #   --depth-dir 显式指定时优先使用它（不再根据 dataset_root 猜测）
    #   未指定时 fallback 到 dataset_root 下的约定目录
    # --------------------------------------------------------

    depth_dirs = collect_depth_npy_dirs(
        dataset_root,
        depth_npy_dir,
    )

    print()
    print("=" * 70)
    print("DEPTH NPY")
    print("=" * 70)

    if depth_dirs:
        for depth_dir in depth_dirs:
            print(f"Depth dir       : {depth_dir}")
            print(f"Depth NPY count : {count_depth_npy_files(depth_dir)}")

            if not depth_dir.is_dir():
                print("                  (目录不存在，Depth 统计将全部为 N/A)")
    else:
        print("Depth dir       : (未指定，且 dataset_root 下无约定目录)")
        print("Depth NPY count : 0")
        print("Depth 统计将全部为 N/A")

    rgb_files = sorted([
        p for p in rgb_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ])

    rows = []

    depth_npy_found_count = 0

    # GT 关键点数量（由实际数据决定，不写死 7）；同时用于生成 CSV 字段名
    keypoint_count_seen = 0

    for index, rgb_file in enumerate(rgb_files, 1):
        stem = rgb_file.stem

        print("\n")
        print("#" * 70)
        print(f"[{index}/{len(rgb_files)}] {stem}")
        print("#" * 70)

        try:
            rgbd_file = find_file_by_stem(
                rgbd_dir,
                stem,
                [".png", ".tif", ".tiff"],
            )

            label_file = label_dir / f"{stem}.txt"

            result = evaluate_single_rgb_vs_rgbd(
                rgb_model_path=rgb_model_path,
                rgbd_model_path=rgbd_model_path,
                rgb_image_path=str(rgb_file),
                rgbd_image_path=str(rgbd_file),
                gt_label_path=str(label_file),
                conf_threshold=conf_threshold,
                imgsz=imgsz,
                match_iou_threshold=match_iou_threshold,
                save_path=str(vis_dir / f"{stem}.jpg"),
            )

            rgb_nme_result = result["rgb"]["nme"]
            rgbd_nme_result = result["rgbd"]["nme"]

            rgb_nme = (
                rgb_nme_result["nme"]
                if rgb_nme_result is not None
                else None
            )

            rgbd_nme = (
                rgbd_nme_result["nme"]
                if rgbd_nme_result is not None
                else None
            )

            common_matched = (
                rgb_nme is not None
                and rgbd_nme is not None
            )

            delta = None
            winner = ""

            if common_matched:
                delta = rgbd_nme - rgb_nme

                if delta < 0:
                    winner = "RGBD"
                elif delta > 0:
                    winner = "RGB"
                else:
                    winner = "TIE"

            rgb_matched = result["rgb"]["prediction"]["matched"]
            rgbd_matched = result["rgbd"]["prediction"]["matched"]

            rgb_box_iou = result["rgb"]["prediction"]["iou"]
            rgbd_box_iou = result["rgbd"]["prediction"]["iou"]

            # --------------------------------------------------
            # GT bbox 与真实 Depth（mm）统计
            # --------------------------------------------------

            gt_box = result["gt"]["box"]

            gt_bbox_diagonal = calculate_bbox_diagonal(gt_box)

            depth_npy_path = resolve_depth_npy_path(
                depth_dirs,
                stem,
            )

            if depth_npy_path is not None:
                depth_npy_found_count += 1

            # 首张图片额外打印一次，用于确认入参目录与映射是否生效
            if index == 1:
                print(f"RGB stem          : {stem}")
                print(
                    "Matched Depth NPY : "
                    f"{depth_npy_path if depth_npy_path is not None else 'N/A'}"
                )

            depth_stats = load_gt_bbox_depth_stats(
                gt_box,
                depth_npy_path,
            )

            # --------------------------------------------------
            # 单点像素误差（K0..K{N-1}）
            # 直接取 calculate_nme 的结果，与控制台打印同源
            # --------------------------------------------------

            gt_keypoints = result["gt"]["keypoints"]

            keypoint_count_seen = max(
                keypoint_count_seen,
                len(gt_keypoints),
            )

            rgb_keypoint_px = extract_keypoint_pixel_errors(
                result["rgb"]["nme"],
                keypoint_count_seen,
            )

            rgbd_keypoint_px = extract_keypoint_pixel_errors(
                result["rgbd"]["nme"],
                keypoint_count_seen,
            )

            row = {
                "image": rgb_file.name,

                "rgb_matched": rgb_matched,
                "rgbd_matched": rgbd_matched,

                "rgb_box_iou": rgb_box_iou,
                "rgbd_box_iou": rgbd_box_iou,

                "gt_bbox_diagonal": gt_bbox_diagonal,
                "hand_depth_median_mm": depth_stats["median_mm"],
                "hand_depth_valid_ratio": depth_stats["valid_ratio"],

                "rgb_nme": rgb_nme,
                "rgbd_nme": rgbd_nme,

                # 固定定义：
                # delta = RGBD - RGB
                # delta < 0 表示 RGBD 更好
                "delta_nme": delta,

                "winner": winner,
                "common_matched": common_matched,
            }

            row.update(
                build_keypoint_pixel_fields(
                    rgb_keypoint_px,
                    rgbd_keypoint_px,
                )
            )

            rows.append(row)

        except Exception as exc:
            print(f"ERROR: {stem}: {exc}")

            error_row = {
                "image": rgb_file.name,

                "rgb_matched": False,
                "rgbd_matched": False,

                "rgb_box_iou": None,
                "rgbd_box_iou": None,

                "gt_bbox_diagonal": None,
                "hand_depth_median_mm": None,
                "hand_depth_valid_ratio": 0.0,

                "rgb_nme": None,
                "rgbd_nme": None,
                "delta_nme": None,

                "winner": "ERROR",
                "common_matched": False,
            }

            # 未匹配 / 异常：整列写 None（不用 0 代替缺失值）
            error_row.update(
                build_keypoint_pixel_fields(
                    [None] * keypoint_count_seen,
                    [None] * keypoint_count_seen,
                )
            )

            rows.append(error_row)

    # ========================================================
    # 只使用 RGB 和 RGBD 都成功匹配的图片做公平平均
    # ========================================================

    valid_rows = [
        row for row in rows
        if row["common_matched"]
    ]

    rgb_matched_count = sum(
        bool(row["rgb_matched"])
        for row in rows
    )

    rgbd_matched_count = sum(
        bool(row["rgbd_matched"])
        for row in rows
    )

    rgb_only_count = sum(
        bool(row["rgb_matched"])
        and not bool(row["rgbd_matched"])
        for row in rows
    )

    rgbd_only_count = sum(
        bool(row["rgbd_matched"])
        and not bool(row["rgb_matched"])
        for row in rows
    )

    neither_count = sum(
        not bool(row["rgb_matched"])
        and not bool(row["rgbd_matched"])
        for row in rows
    )

    if valid_rows:
        rgb_mean = float(np.mean([
            row["rgb_nme"]
            for row in valid_rows
        ]))

        rgbd_mean = float(np.mean([
            row["rgbd_nme"]
            for row in valid_rows
        ]))

        rgb_wins = sum(
            row["winner"] == "RGB"
            for row in valid_rows
        )

        rgbd_wins = sum(
            row["winner"] == "RGBD"
            for row in valid_rows
        )

        ties = sum(
            row["winner"] == "TIE"
            for row in valid_rows
        )

        print("\n")
        print("=" * 70)
        print("FINAL NME SUMMARY")
        print("=" * 70)

        print(f"Total images      : {len(rows)}")
        print(f"RGB matched       : {rgb_matched_count}")
        print(f"RGBD matched      : {rgbd_matched_count}")
        print(f"Common matched    : {len(valid_rows)}")
        print(f"RGB only matched  : {rgb_only_count}")
        print(f"RGBD only matched : {rgbd_only_count}")
        print(f"Neither matched   : {neither_count}")

        print()
        print(f"RGB Overall NME  : {rgb_mean:.6f}")
        print(f"RGBD Overall NME : {rgbd_mean:.6f}")
        print(f"Delta NME        : {rgbd_mean - rgb_mean:+.6f}")

        print()
        print(f"RGB wins  : {rgb_wins}")
        print(f"RGBD wins : {rgbd_wins}")
        print(f"Ties      : {ties}")

        if rgbd_mean < rgb_mean:
            print("\nOverall Winner: RGBD")
        elif rgb_mean < rgbd_mean:
            print("\nOverall Winner: RGB")
        else:
            print("\nOverall Winner: TIE")

    else:
        print("没有 RGB / RGBD 同时成功匹配的图片。")

    # ========================================================
    # 单点像素误差统计（K0..K{N-1}）
    # 只统计 common matched 图片，且该 K 点 RGB / RGBD 两侧都有有效误差
    # ========================================================

    # 明显领先阈值（px）：rgb_kX_px - rgbd_kX_px >= threshold
    KEYPOINT_LEAD_THRESHOLDS = (5.0, 10.0, 20.0)

    print("\n")
    print("=" * 70)
    print("KPOINT PIXEL ERROR SUMMARY")
    print("=" * 70)

    if keypoint_count_seen == 0:
        print("没有可用的关键点像素误差（未获取到 GT 关键点数量）。")

    for keypoint_index in range(keypoint_count_seen):
        rgb_key = f"rgb_k{keypoint_index}_px"
        rgbd_key = f"rgbd_k{keypoint_index}_px"

        # 该 K 点两侧都有有效误差的 common matched 图片
        pair_rows = [
            row for row in valid_rows
            if row.get(rgb_key) is not None
            and row.get(rgbd_key) is not None
        ]

        print()
        print(f"K{keypoint_index}:")
        print(f"{'Count':<16}: {len(pair_rows)}")

        if not pair_rows:
            continue

        rgb_mean_px = float(np.mean([
            row[rgb_key]
            for row in pair_rows
        ]))

        rgbd_mean_px = float(np.mean([
            row[rgbd_key]
            for row in pair_rows
        ]))

        # 完全相等才算 tie
        rgb_wins = sum(
            row[rgb_key] < row[rgbd_key]
            for row in pair_rows
        )

        rgbd_wins = sum(
            row[rgbd_key] < row[rgb_key]
            for row in pair_rows
        )

        ties = len(pair_rows) - rgb_wins - rgbd_wins

        print(f"{'RGB mean px':<16}: {rgb_mean_px:.4f}")
        print(f"{'RGBD mean px':<16}: {rgbd_mean_px:.4f}")
        print(
            f"{'Delta mean px':<16}: "
            f"{rgbd_mean_px - rgb_mean_px:+.4f}"
        )
        print(f"{'RGB wins':<16}: {rgb_wins}")
        print(f"{'RGBD wins':<16}: {rgbd_wins}")
        print(f"{'Ties':<16}: {ties}")

        for threshold in KEYPOINT_LEAD_THRESHOLDS:
            rgbd_better_count = sum(
                row[rgb_key] - row[rgbd_key] >= threshold
                for row in pair_rows
            )

            label = f"RGBD better >= {threshold:g} px"

            print(f"{label:<22}: {rgbd_better_count}")

        for threshold in KEYPOINT_LEAD_THRESHOLDS:
            rgb_better_count = sum(
                row[rgbd_key] - row[rgb_key] >= threshold
                for row in pair_rows
            )

            label = f"RGB better >= {threshold:g} px"

            print(f"{label:<22}: {rgb_better_count}")

    # ========================================================
    # 距离分析（探索性：按当前样本真实 Depth 三等分，不写死距离阈值）
    # 只使用 common_matched=True 且 Depth 有效的图片
    # ========================================================

    depth_valid_rows = sorted(
        [
            row for row in valid_rows
            if row["hand_depth_median_mm"] is not None
        ],
        key=lambda row: row["hand_depth_median_mm"],
    )

    print("\n")
    print("=" * 70)
    print("NME BY DEPTH")
    print("=" * 70)
    print(f"Depth NPY found   : {depth_npy_found_count} / {len(rows)}")
    print(f"Depth usable      : {len(depth_valid_rows)}")

    if depth_valid_rows:
        # 按 depth 从近到远排序后三等分
        depth_groups = np.array_split(
            np.arange(len(depth_valid_rows)),
            3,
        )

        depth_sections = [
            ("Near", depth_groups[0]),
            ("Mid", depth_groups[1]),
            ("Far", depth_groups[2]),
        ]

        for section_name, section_indices in depth_sections:
            section_rows = [
                depth_valid_rows[int(i)]
                for i in section_indices
            ]

            print()
            print(f"{section_name}:")
            print(f"Count          : {len(section_rows)}")

            if not section_rows:
                continue

            section_mean_depth = float(np.mean([
                row["hand_depth_median_mm"]
                for row in section_rows
            ]))

            section_rgb_mean = float(np.mean([
                row["rgb_nme"]
                for row in section_rows
            ]))

            section_rgbd_mean = float(np.mean([
                row["rgbd_nme"]
                for row in section_rows
            ]))

            section_rgb_wins = sum(
                row["winner"] == "RGB"
                for row in section_rows
            )

            section_rgbd_wins = sum(
                row["winner"] == "RGBD"
                for row in section_rows
            )

            print(f"Mean Depth     : {section_mean_depth:.2f} mm")
            print(f"RGB NME        : {section_rgb_mean:.6f}")
            print(f"RGBD NME       : {section_rgbd_mean:.6f}")
            print(
                f"Delta NME      : "
                f"{section_rgbd_mean - section_rgb_mean:+.6f}"
            )
            print(f"RGB wins       : {section_rgb_wins}")
            print(f"RGBD wins      : {section_rgbd_wins}")

    # ========================================================
    # RGBD NME 更优图片排序
    # ========================================================

    rgbd_better_rows = sorted(
        [
            row for row in valid_rows
            if row["winner"] == "RGBD"
        ],
        key=lambda row: row["delta_nme"]
    )

    print("\n")
    print("=" * 70)
    print("TOP RGBD BETTER IMAGES")
    print("=" * 70)

    for index, row in enumerate(rgbd_better_rows[:20], 1):
        depth_text = (
            f"{row['hand_depth_median_mm']:.1f} mm"
            if row["hand_depth_median_mm"] is not None
            else "N/A"
        )

        print(
            f"{index:02d}. "
            f"{row['image']}  "
            f"Depth={depth_text}  "
            f"BBoxDiag={row['gt_bbox_diagonal']:.1f}  "
            f"RGB={row['rgb_nme']:.6f}  "
            f"RGBD={row['rgbd_nme']:.6f}  "
            f"Delta={row['delta_nme']:+.6f}"
        )

    # ========================================================
    # 保存 CSV
    # ========================================================

    csv_path = output_dir / "per_image.csv"

    # 基础字段（原有字段，顺序不变、不删除、不重命名）
    base_fieldnames = [
        "image",
        "rgb_matched",
        "rgbd_matched",
        "rgb_box_iou",
        "rgbd_box_iou",
        "gt_bbox_diagonal",
        "hand_depth_median_mm",
        "hand_depth_valid_ratio",
        "rgb_nme",
        "rgbd_nme",
        "delta_nme",
        "winner",
        "common_matched",
    ]

    # 完整字段 = 原有字段 + 单点像素误差字段
    #   顺序：原有字段 -> rgb_k0_px..rgb_k{N-1}_px
    #                 -> rgbd_k0_px..rgbd_k{N-1}_px
    #                 -> delta_k0_px..delta_k{N-1}_px
    #
    # per_image.csv 与 rgbd_better.csv 共用同一份（单一数据源），
    # 避免以后再次出现「行里有字段、fieldnames 里没有」导致的
    # ValueError: dict contains fields not in fieldnames
    csv_fieldnames = (
        base_fieldnames
        + build_keypoint_pixel_fieldnames(keypoint_count_seen)
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=csv_fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"CSV saved: {csv_path}")

    # --------------------------------------------------------
    # RGBD 优势最大图片单独保存
    # 只包含 common_matched=True 且 winner="RGBD"
    # 按 delta_nme 从小到大（RGBD 优势最大在前）
    # 字段与 per_image.csv 完全一致（含全部 K 点字段）
    # --------------------------------------------------------

    rgbd_better_csv_path = output_dir / "rgbd_better.csv"

    if not rgbd_better_rows:
        print("(没有 winner=RGBD 的图片，rgbd_better.csv 只写表头)")

    with open(
        rgbd_better_csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=csv_fieldnames,
        )

        writer.writeheader()
        writer.writerows(rgbd_better_rows)

    print(f"RGBD better CSV saved: {rgbd_better_csv_path}")

    print(f"Visualizations: {vis_dir}")

# ============================================================
# 临时单张测试入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="单张 RGB vs RGBD 关键点 NME 对比"
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
        help="数据集根目录，例如 hand_cross_subject_v2",
    )

    parser.add_argument(
        "--stem",
        default=None,
        help="单张测试图片 stem，不带扩展名",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="测试 test 目录中的全部图片",
    )

    parser.add_argument(
        "--rgb-model",
        required=True,
        help="RGB 模型 best.pt",
    )

    parser.add_argument(
        "--rgbd-model",
        required=True,
        help="RGBD 模型 best.pt",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="推理置信度阈值，默认 0.25",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="推理尺寸，默认 640",
    )

    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.50,
        help="预测实例与 GT 匹配所需最低 IoU，默认 0.50",
    )

    parser.add_argument(
        "--output",
        default="runs/pose_nme/single_rgb_vs_rgbd_nme.jpg",
        help="可视化输出路径",
    )

    parser.add_argument(
        "--depth-dir",
        "--depth-npy-dir",
        dest="depth_npy_dir",
        default=None,
        help=(
            "原始 Depth NPY 目录（depth_<完整时间戳>.npy, uint16/mm），"
            "仅 --all 批量模式使用；显式指定时优先使用该目录，"
            "不再根据 dataset-root 自动猜测；未指定时 fallback 到 "
            "dataset-root 下的约定目录自动查找"
        ),
    )

    args = parser.parse_args()

    if args.all:
        evaluate_all_test(
            dataset_root=args.dataset_root,
            rgb_model_path=args.rgb_model,
            rgbd_model_path=args.rgbd_model,
            conf_threshold=args.conf,
            imgsz=args.imgsz,
            match_iou_threshold=args.match_iou,
            output_dir="runs/pose_nme/all_test",
            depth_npy_dir=args.depth_npy_dir,
        )

        sys.exit(0)

    if not args.stem:
        parser.error("单张模式需要 --stem，批量测试请使用 --all")

    dataset_root = Path(args.dataset_root)
    stem = args.stem

    rgb_image_path = find_file_by_stem(
        dataset_root / "rgb" / "images" / "test",
        stem,
        [".jpg", ".jpeg", ".png", ".bmp"],
    )

    rgbd_image_path = find_file_by_stem(
        dataset_root / "rgbd" / "images" / "test",
        stem,
        [".png", ".tif", ".tiff"],
    )

    gt_label_path = dataset_root / "rgb" / "labels" / "test" / f"{stem}.txt"

    if not gt_label_path.exists():
        raise FileNotFoundError(
            f"GT label 不存在: {gt_label_path}"
        )

    print("=" * 70)
    print("输入解析")
    print("=" * 70)
    print(f"Dataset : {dataset_root}")
    print(f"Stem    : {stem}")
    print(f"RGB     : {rgb_image_path}")
    print(f"RGBD    : {rgbd_image_path}")
    print(f"Label   : {gt_label_path}")
    print(f"RGB PT  : {args.rgb_model}")
    print(f"RGBD PT : {args.rgbd_model}")

    result = evaluate_single_rgb_vs_rgbd(
        rgb_model_path=args.rgb_model,
        rgbd_model_path=args.rgbd_model,
        rgb_image_path=str(rgb_image_path),
        rgbd_image_path=str(rgbd_image_path),
        gt_label_path=str(gt_label_path),
        conf_threshold=args.conf,
        imgsz=args.imgsz,
        match_iou_threshold=args.match_iou,
        save_path=args.output,
    )