"""RGBD Depth 消融评估 (验证已训练的 RGBD Pose 模型是否真实利用第 4 通道 Depth)

仅用于已训练好的 RGBD 4ch 模型：在同一批 RGBD test 图片上，构造三种 Depth 输入，
比较检测/关键点指标，判断模型是否真的在用 Depth。

六种变体（共用同一批 12 个 test ID 与同一份 labels，同一权重，同一评估参数；blurred 含 11/31/51 三尺度，
flat/fixed/full-constant 为信息剥离系列，scaled_true_depth 为幅度扫描系列，
flat_scaled_depth 为配对幅度控制系列，共二十六个具体 variant）：
  1. true_depth           原始 4 通道 RGBD 不变
  2. zero_depth           RGB 三通道不变，第 4 通道(Depth)全部置 0
  3. shuffled_depth       RGB 三通道不变，第 4 通道(Depth)取自其它 test 图（跨图随机，
                          derangement 不允许回配自己），可复现 —— Cross-image Shuffle
  4. spatial_shuffle_depth RGB 三通道不变，第 4 通道用「本图自身 Depth」，仅对像素位置做
                          固定 seed 的随机 permutation（flatten→permute→reshape），破坏与 RGB
                          的空间对应关系，但保留 min/max/mean/std/histogram —— Spatial-shuffled
  5. blurred_depth        RGB 三通道不变，第 4 通道用「本图自身 Depth」做 masked Gaussian
                          blur（11x11，0=invalid 不参与平均），验证模型需要 Depth 粗尺度
                          空间结构还是局部高频细节 —— Blurred Depth
                          （另有 blurred_depth_31=31x31 / blurred_depth_51=51x51 验证模型
                          依赖多大尺度的 Depth 结构；三者共用同一 masked blur 函数，仅 kernel 不同）
  6. flat_valid_depth     RGB 三通道不变，第 4 通道用「本图自身 Depth」：有效区域(>0)全部
                          替换为该图有效区域 Depth 均值（单一常数值），0=invalid 严格保持。
                          信息剥离：只保留 valid mask + 整体距离水平，删掉连续局部 Depth 几何 /
                          梯度 / 纹理，验证模型依赖真实连续 Depth 还是主要依赖 valid 轮廓 —— Flat Valid Depth
  7. fixed_valid_depth    RGB 三通道不变，第 4 通道用「本图自身 Depth」：有效区域(>0)全部
                          替换为整 split 共用的固定值 global_valid_mean（所有图同一值），0=invalid
                          严格保持。在 flat_valid_depth 基础上进一步删除「每张图自己的整体 Depth 水平」，
                          只留 valid mask / 空间轮廓，验证每张图整体距离水平是否被模型利用 —— Fixed Valid Depth
  8. full_constant_depth  RGB 三通道不变，第 4 通道所有像素（含原 invalid）全部填成同一 fixed_value，
                          故意删除 valid mask / 空间轮廓。与 Fixed Valid 共用同一 fixed_value：
                          Full Constant vs Fixed Valid 区分「valid mask 空间形状」作用；
                          Full Constant vs Zero（全0）区分「第4通道绝对非零基线」作用 —— Full Constant Depth
  9. full_constant_depth_1 / _64 / _128 / _224 / _255
                          full_constant_depth 的常数值扫描点：第 4 通道所有像素（含原 invalid）全部填成
                          指定常数值（1 / 64 / 128 / 224 / 255），故意删除 valid mask。叠加 zero_depth=0
                          与 full_constant_depth=~170，构成 0/1/64/128/170/224/255 扫描，验证模型仅对
                          「0/非0」敏感，还是随第4通道绝对数值大小发生系统性变化 —— Constant Sweep
 10. fixed_valid_depth_1 / _64 / _128 / _224 / _255
                          fixed_valid_depth 的常数值扫描点：有效区域(>0)填成指定常数、invalid 严格保持 0，
                          保留 valid mask / 空间轮廓（与 Full Constant Sweep 同幅度、不同 mask 形状）。叠加
                          zero_depth=0 与 fixed_valid_depth=~170，构成 0/1/64/128/170/224/255 扫描，用于判断
                          「第4通道绝对幅度效应是否依赖 valid mask / 空间结构」 —— Fixed Valid Sweep
 11. scaled_true_depth_25 / _50 / _75
                          保留真实 Depth 空间排列 + valid mask + 像素相对高低（排序），仅把第4通道整体数值
                          幅度按 25% / 50% / 75% 缩放；原始 valid 像素缩放后钳到 1~255 防归零，invalid=0
                          严格保持。用于测试「真实 Depth 空间模式还在、但整体幅度降低后 Pose 是否比原始
                          True Depth 改善」—— Scaled True Depth（注意：幅度降低、空间模式与排序保留，
                          并非物理 Depth 几何完全不变）
 12. flat_scaled_depth_25 / _50 / _75
                          配对控制实验：严格复用 scaled_true_depth 的 scale / round / clip / invalid
                          处理生成 scaled，但随后把当前图片有效区域全部替换为该区域平均值（单一常数值），
                          故意删除有效区域内部的真实 Depth 空间变化；幅度与 valid mask 与对应
                          scaled_true_depth 几乎完全一致（仅整数常数引入 <=0.5 舍入误差）。用于回答
                          「在相同幅度 + 相同 valid mask 下，真实 Depth 空间变化是否有独立贡献」
                          —— Flat Scaled Depth（Scaled True vs Flat Scaled 的唯一差异即有效区域内部
                          是否保留真实 Depth 空间变化）

输出：
  - 每个变体 Box / Pose 的 P / R / mAP50 / mAP50-95
  - 对比表 + True-Zero / True-CrossShuffle / True-SpatialShuffle 差值

生成期逐图校验（打印 + 断言）：
  - shape 必须为 (H, W, 4)
  - dtype 必须为 uint8
  - 第 4 通道 min / max / mean / std
  - zero_depth 第 4 通道必须全部为 0
  - shuffled_depth 第 4 通道必须与原样本 Depth 不同、且来自其它图
  - spatial_shuffle_depth 第 4 通道必须与原样本 Depth 不同，但排序后像素集合必须完全一致

约束：
  - 不重新训练；不覆盖原始 dataset；在临时目录生成消融数据
  - 不改任何 train 脚本
  - 原始 RGB / RGBD test 结果不受影响（只读取，不写入）

调用示例：
  python YJJ_Pose_Scripts/04_eval/ablate_rgbd_depth.py
  python YJJ_Pose_Scripts/04_eval/ablate_rgbd_depth.py --no-eval   # 仅生成临时数据 + 校验，不跑 val
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# ---- 默认路径（本机 Resource 数据集；如换机用 --rgbd-yaml 指定）----
_DEFAULT_RGBD_YAML = (
    r"H:/YJJ/Yolo_RGBD/Resource/session1_200146/dataset/hand/rgbd/data_rgbd.yaml"
)
_DEFAULT_WEIGHTS = r"runs/pose/train_pose_rgbd_4ch/weights/best.pt"
_DEFAULT_OUT = r"runs/ablation/rgbd_depth"  # runs/ 已被 .gitignore 忽略，作为安全临时目录

VARIANTS = ("true_depth", "zero_depth", "shuffled_depth", "spatial_shuffle_depth",
            "blurred_depth", "blurred_depth_31", "blurred_depth_51",
            "flat_valid_depth", "fixed_valid_depth", "full_constant_depth",
            "full_constant_depth_1", "full_constant_depth_64", "full_constant_depth_128",
            "full_constant_depth_224", "full_constant_depth_255",
            "fixed_valid_depth_1", "fixed_valid_depth_64", "fixed_valid_depth_128",
            "fixed_valid_depth_224", "fixed_valid_depth_255",
            "scaled_true_depth_25", "scaled_true_depth_50", "scaled_true_depth_75",
            "flat_scaled_depth_25", "flat_scaled_depth_50", "flat_scaled_depth_75")

# 变体显示名（最终表格列头用）：shuffled_depth = 跨图随机 Depth；
# spatial_shuffle_depth = 同图 Depth 像素位置随机打乱（保留统计量，仅破坏空间结构）。
VARIANT_LABELS = {
    "true_depth": "True Depth",
    "zero_depth": "Zero Depth",
    "shuffled_depth": "Cross-image Shuffled",
    "spatial_shuffle_depth": "Spatial-shuffled",
    "blurred_depth": "Blurred Depth",
    "blurred_depth_31": "Blurred Depth 31x31",
    "blurred_depth_51": "Blurred Depth 51x51",
    "flat_valid_depth": "Flat Valid Depth",
    "fixed_valid_depth": "Fixed Valid Depth",
    "full_constant_depth": "Full Constant Depth",
    "full_constant_depth_1": "Full Constant Depth 1",
    "full_constant_depth_64": "Full Constant Depth 64",
    "full_constant_depth_128": "Full Constant Depth 128",
    "full_constant_depth_224": "Full Constant Depth 224",
    "full_constant_depth_255": "Full Constant Depth 255",
    "fixed_valid_depth_1": "Fixed Valid Depth 1",
    "fixed_valid_depth_64": "Fixed Valid Depth 64",
    "fixed_valid_depth_128": "Fixed Valid Depth 128",
    "fixed_valid_depth_224": "Fixed Valid Depth 224",
    "fixed_valid_depth_255": "Fixed Valid Depth 255",
    "scaled_true_depth_25": "Scaled True Depth 25%",
    "scaled_true_depth_50": "Scaled True Depth 50%",
    "scaled_true_depth_75": "Scaled True Depth 75%",
    "flat_scaled_depth_25": "Flat Scaled Depth 25%",
    "flat_scaled_depth_50": "Flat Scaled Depth 50%",
    "flat_scaled_depth_75": "Flat Scaled Depth 75%",
}

# full_constant_depth 扫描点：第 4 通道全部（含原 invalid）填成指定常数值（删 valid mask 形状）。
# zero_depth=0 与 full_constant_depth=global fixed value(~170) 已在别处，此处补齐 1/64/128/170/224/255 扫描。
CONSTANT_SWEEP_VALUES = {
    "full_constant_depth_1": 1,
    "full_constant_depth_64": 64,
    "full_constant_depth_128": 128,
    "full_constant_depth_224": 224,
    "full_constant_depth_255": 255,
}

# fixed_valid_depth 扫描点：有效区域(>0)填成指定常数、invalid(<0)严格保持 0（保留 valid mask 形状）。
# zero_depth=0 与 fixed_valid_depth=global fixed value(~170) 已在别处，此处补齐 1/64/128/170/224/255 扫描，
# 用于在保留 valid mask / 空间轮廓的前提下，判断第4通道绝对幅度效应是否仍然存在（与 Full Constant Sweep 对应比较）。
FIXED_VALID_SWEEP_VALUES = {
    "fixed_valid_depth_1": 1,
    "fixed_valid_depth_64": 64,
    "fixed_valid_depth_128": 128,
    "fixed_valid_depth_224": 224,
    "fixed_valid_depth_255": 255,
}

# scaled_true_depth 扫描点：保留真实 Depth 空间排列 + valid mask + 相对高低关系，
# 仅把第4通道整体数值幅度按指定比例缩放（25% / 50% / 75%）；原始 valid 像素缩放后
# 限制到 1~255 防止归零。用于测试「真实 Depth 空间模式还在、但整体幅度降低后 Pose 是否改善」。
SCALED_TRUE_DEPTH_SCALES = {
    "scaled_true_depth_25": 0.25,
    "scaled_true_depth_50": 0.50,
    "scaled_true_depth_75": 0.75,
}

# flat_scaled_depth 配对点：在 scaled_true_depth（保留真实 Depth 空间变化）基础上，
# 复用完全相同的 scale / round / clip / invalid 处理生成 scaled，但随后把当前图片有效区域
# 全部替换为该区域平均值（单一常数值），故意删除有效区域内部的 Depth 空间变化；幅度与
# valid mask 与 scaled_true_depth 几乎完全一致（仅整数常数引入 <=0.5 的舍入误差）。
# 用于严格配对控制，回答「在相同幅度 + 相同 valid mask 下，真实 Depth 空间变化是否有独立贡献」。
FLAT_SCALED_DEPTH_SCALES = {
    "flat_scaled_depth_25": 0.25,
    "flat_scaled_depth_50": 0.50,
    "flat_scaled_depth_75": 0.75,
}


# --------------------------------------------------------------------------- 参数
def parse_args():
    p = argparse.ArgumentParser(description="RGBD Depth 消融评估 (true / zero / cross-image shuffled / spatial-shuffled / blurred depth)")
    p.add_argument("--weights", default=_DEFAULT_WEIGHTS, help="RGBD 4ch 评估权重 .pt")
    p.add_argument("--rgbd-yaml", default=_DEFAULT_RGBD_YAML,
                   help="原始 RGBD data yaml（提供 path / nc / kpt_shape / channels / rgbd）")
    p.add_argument("--out", default=_DEFAULT_OUT, help="临时消融数据输出目录")
    p.add_argument("--imgsz", type=int, default=640, help="推理图像尺寸（三组必须一致）")
    p.add_argument("--batch", type=int, default=4, help="batch size（三组必须一致）")
    p.add_argument("--device", default=None, help="设备，默认自动选择 (cuda 优先)")
    p.add_argument("--seed", type=int, default=42, help="shuffled_depth 的 derangement 随机种子")
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS),
                   help="指定要执行的 Depth 变体子集（默认全部 26 个）；GUI 调用时传 true_depth zero_depth")
    p.add_argument("--split", default="test", help="评估切分（消融固定 test）")
    p.add_argument("--no-eval", action="store_true",
                   help="只生成临时数据 + 逐图校验，不调用 model.val（快速自检用）")
    return p.parse_args()


# --------------------------------------------------------------------------- 工具
def read_rgba(path: Path) -> np.ndarray:
    """读取 4 通道融合 PNG（BGRA），返回 (H, W, 4) uint8。"""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"图像读取失败: {path}")
    if im.ndim != 3 or im.shape[2] != 4:
        raise ValueError(f"图像不是 4 通道: {path} shape={im.shape}")
    if im.dtype != np.uint8:
        raise ValueError(f"图像 dtype 非 uint8: {path} dtype={im.dtype}")
    return im


def ch4_stats(ch4: np.ndarray) -> dict:
    return {
        "min": float(ch4.min()),
        "max": float(ch4.max()),
        "mean": round(float(ch4.mean()), 4),
        "std": round(float(ch4.std()), 4),
    }


def derangement(n: int, seed: int) -> np.ndarray:
    """返回 [0, n) 的一个排列，且无固定点 (perm[i] != i)。固定 seed 可复现。"""
    rng = np.random.default_rng(seed)
    # 真实随机排列，循环直到得到 derangement（n=12 时几乎一次命中，且完全可复现）
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            break
    assert set(perm.tolist()) == set(range(n)), "derangement 必须是合法排列"
    assert not np.any(perm == np.arange(n)), "derangement 不能有固定点"
    return perm


def spatial_shuffle_depth_channel(ch4: np.ndarray, seed: int) -> np.ndarray:
    """同图 Depth 通道：仅对像素位置做固定 seed 的随机 permutation，保留全部像素值。

    flatten -> rng.permutation -> reshape 回 (H, W)。因此 min/max/mean/std/histogram
    与原 Depth 完全相同，唯一被破坏的是像素的空间排列（与 RGB 的对应关系）。
    固定 seed 可复现；seed = 全局 seed + 图片下标。
    """
    rng = np.random.default_rng(seed)
    flat = ch4.ravel()
    perm = rng.permutation(flat.size)  # flat.size 个像素位置的随机排列
    return flat[perm].reshape(ch4.shape).astype(ch4.dtype)


def blurred_depth_channel(ch4: np.ndarray, kernel_size: int) -> np.ndarray:
    """同图 Depth 通道：仅对第 4 通道做 masked Gaussian blur（固定 kernel_size×kernel_size 核）。

    目的：验证模型需要多大尺度的 Depth 空间结构（11 / 31 / 51）。
    使用 RGBD 第 4 通道原始 0~255 uint8 Depth；0 = invalid，绝对不能直接参与平均，
    否则 invalid 区域会污染有效 Depth 边缘。

    实现：对 (depth * valid) 与 valid mask 分别做 GaussianBlur，再相除归一化，
    最后把原始 depth==0 的位置严格重设回 0。固定 kernel=(kernel_size, kernel_size)，
    sigma=0（由核尺寸推导），kernel_size 由调用方显式传入，不自动计算。
    输出 dtype=uint8，shape 与原 Depth 完全一致，范围 0~255。
    """
    k = (int(kernel_size), int(kernel_size))       # 固定核，不随图变化
    valid = (ch4 > 0).astype(np.float32)          # 0 = invalid mask
    depth = ch4.astype(np.float32)

    weighted = cv2.GaussianBlur(depth * valid, k, 0)
    weights = cv2.GaussianBlur(valid, k, 0)

    # 仅 valid 权重足够处做除法，避免 0/0；其余保持 0
    blurred = np.divide(
        weighted,
        weights,
        out=np.zeros_like(weighted),
        where=weights > 1e-6,
    )

    # 原始 invalid（depth==0）位置严格重设为 0，invariant mask 完全保持
    blurred[ch4 == 0] = 0

    return np.clip(np.rint(blurred), 0, 255).astype(np.uint8)


def flat_valid_depth_channel(ch4: np.ndarray) -> np.ndarray:
    """同图 Depth 通道信息剥离：有效区域全部替换为该图有效区域 Depth 的均值（单常数值）。

    目的：剥离「连续局部 Depth 几何 / 梯度 / 纹理」，仅保留 valid mask + 整体距离水平，
    验证模型依赖的是真实连续 Depth 几何，还是主要依赖 Depth 有效区域的空间轮廓。
    使用当前样本自己的 Depth，不取其它图。

    original_depth == 0 -> 继续严格保持 0（invalid 不污染均值，也不参与均值计算）；
    original_depth > 0  -> 全部置为该图有效区域 Depth 的平均值（四舍五入，限定 1~255）。
    输出 dtype=uint8，shape 与原 Depth 完全一致；有效区域内部应只有一个唯一 Depth 值。
    """
    valid_mask = ch4 > 0
    flat = np.zeros_like(ch4, dtype=np.uint8)
    if not np.any(valid_mask):
        return flat  # 无有效像素：直接返回全 0
    mean_value = float(ch4[valid_mask].mean())
    rounded_mean = int(np.clip(np.rint(mean_value), 1, 255))
    flat[valid_mask] = rounded_mean
    return flat


def fixed_valid_depth_channel(ch4: np.ndarray, fixed_value: int) -> np.ndarray:
    """同图 Depth 通道信息剥离：有效区域(>0)全部替换为整 split 共用的固定值 fixed_value。

    目的：在 flat_valid_depth（每张图自己的均值）基础上进一步删除「每张图整体 Depth 水平」，
    只保留 valid mask / 空间轮廓 + 一个所有图共用的固定非零值，验证每张图自己的平均 Depth
    是否被模型利用。fixed_value 由调用方用整 split 有效像素均值算一次后传入，不在此重算。
    original_depth == 0 严格保持 0；输出 dtype=uint8，shape 与原 Depth 完全一致；
    有效区域内部只有一个唯一值 = fixed_value。
    """
    flat = np.zeros_like(ch4, dtype=np.uint8)
    valid_mask = ch4 > 0
    flat[valid_mask] = np.uint8(fixed_value)
    return flat


def full_constant_depth_channel(ch4: np.ndarray, fixed_value: int) -> np.ndarray:
    """同图 Depth 通道：第 4 通道所有像素（无论原 valid / invalid）全部填同一个 fixed_value。

    目的：在 fixed_valid_depth（保留 valid mask 形状）基础上进一步删除「valid mask / 空间轮廓」，
    只留第 4 通道的非零常数基线，验证 valid mask 前景轮廓本身是否影响模型。fixed_value 复用
    fixed_valid_depth 整 split 共用值，保证两者使用完全相同的非零值。
    故意不保留 invalid mask（全图统一值，这是实验定义，非 bug）；输出 dtype=uint8，
    shape 与原 Depth 完全一致，整张第 4 通道只有一个唯一值 = fixed_value。
    """
    return np.full_like(ch4, np.uint8(fixed_value), dtype=np.uint8)


def scaled_true_depth_channel(ch4: np.ndarray, scale: float) -> np.ndarray:
    """同图自身真实 Depth 幅度缩放：保留空间排列 + valid mask + 相对高低关系，仅降低整体数值幅度。

    目的：在保留真实 Depth 空间模式、valid mask 与像素间相对高低（排序）的前提下，验证把第4通道
    整体数值幅度降低后 Pose 是否比原始 True Depth 改善（区分「高幅度本身是否正在掩盖真实 Depth
    空间信息收益」）。
    实现：valid_mask = ch4 > 0；result[valid_mask] = clip(rint(ch4[valid_mask] * scale), 1, 255)，
    原始 depth==0 严格保持 0；缩放后有效像素下限钳到 1 防止原始有效像素归零（不污染 invalid）。
    输出 dtype=uint8，shape 与原 Depth 完全一致；不复制三套代码，三个 scale 仅参数不同。
    """
    result = np.zeros_like(ch4, dtype=np.uint8)
    valid_mask = ch4 > 0
    # 仅对有效像素缩放；先转 float 再 rint 避免 uint8 乘法溢出/截断，最后 clip 到 1~255
    scaled = np.rint(ch4[valid_mask].astype(np.float32) * scale)
    scaled = np.clip(scaled, 1, 255)
    result[valid_mask] = scaled.astype(np.uint8)
    return result


def flat_scaled_depth_channel(ch4: np.ndarray, scale: float) -> np.ndarray:
    """同图真实 Depth 幅度缩放后再拍平（配对控制「有效区域内部是否保留真实 Depth 空间变化」）。

    目的：在 scaled_true_depth（保留真实 Depth 空间变化）基础上做严格配对——本 variant 用
    完全相同的 scale / round / clip / invalid 处理生成 scaled，但随后把当前图片有效区域
    (original_depth>0) 全部替换为该区域平均值（单一常数值），故意删除有效区域内部的
    Depth 空间变化；幅度（≈ scaled 平均）与 valid mask 与 scaled_true_depth 几乎完全一致
    （仅整数常数引入 <=0.5 的舍入误差）。用于回答「在相同幅度 + 相同 valid mask 下，
    真实 Depth 空间变化是否有独立贡献」。
    实现：先 scaled = scaled_true_depth_channel(ch4, scale)（严格复用现有 Scaled True 逻辑，
    不复制缩放公式）；再 valid_mask = ch4>0；mean_value = scaled[valid_mask].mean()；
    flat_value = int(clip(rint(mean_value), 1, 255))；flat_scaled[valid_mask] = flat_value。
    original_depth == 0 严格保持 0（scaled 已保证 invalid=0，valid_mask 与 scaled 严格一致）；
    输出 dtype=uint8，shape 与原 Depth 完全一致；有效区域内部只有一个唯一值 = flat_value。
    无有效像素时直接返回全 0（与 scaled_true_depth_channel 一致）。
    """
    scaled = scaled_true_depth_channel(ch4, scale)  # 严格复用现有 Scaled True 逻辑，不复制
    valid_mask = ch4 > 0
    flat = np.zeros_like(ch4, dtype=np.uint8)
    if not np.any(valid_mask):
        return flat  # 无有效像素：直接返回全 0
    mean_value = float(scaled[valid_mask].mean())
    flat_value = int(np.clip(np.rint(mean_value), 1, 255))
    flat[valid_mask] = np.uint8(flat_value)
    return flat


def build_variant_yaml(src_yaml: Path, variant_root: Path) -> Path:
    """复制源 yaml，仅把 path 改写为变体目录（其余 nc/kpt_shape/channels/rgbd 不变）。"""
    text = src_yaml.read_text(encoding="utf-8")
    new_path = str(variant_root).replace("\\", "/")
    test_split = "images/test"  # 消融仅生成 test 拆分
    # 替换 path: 的值（保留行内注释），并把 train/val/test 统一指向已生成的 test 拆分，
    # 否则 check_det_dataset 会因缺 train/val 目录而报 FileNotFoundError。
    lines = []
    path_replaced = False
    for line in text.splitlines():
        m = re.match(r"^(\s*path\s*:\s*)(.+?)(\s*#.*)?$", line)
        if m and not path_replaced:
            lines.append(f"{m.group(1)}{new_path}{m.group(3) or ''}")
            path_replaced = True
            continue
        m2 = re.match(r"^(\s*(?:train|val|test)\s*:\s*)(.+?)(\s*#.*)?$", line)
        if m2:
            lines.append(f"{m2.group(1)}{test_split}{m2.group(3) or ''}")
            continue
        lines.append(line)
    if not path_replaced:
        raise ValueError("源 yaml 未找到 path: 字段，无法改写")
    out = variant_root / "data_rgbd_ablation.yaml"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------------- 生成 + 校验
def generate_ablation_data(src_yaml: Path, out_root: Path, seed: int, selected):
    """读取源 RGBD test 集，生成指定变体子集的临时数据，返回 (variants_info, checks)。"""
    import yaml  # 延迟导入，--no-eval 之外的路径也安全

    cfg = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
    ds_root = Path(cfg["path"])
    test_img_dir = ds_root / cfg.get("test", "images/test")
    test_lbl_dir = ds_root / "labels" / "test"

    img_paths = sorted(test_img_dir.glob("*"))
    img_paths = [p for p in img_paths if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    if not img_paths:
        raise FileNotFoundError(f"未找到 test 图片: {test_img_dir}")
    n = len(img_paths)
    print(f"[生成] test 图片数 = {n}，来源: {test_img_dir}")

    # 读取所有 4 通道原图 + 对应 labels
    orig = [read_rgba(p) for p in img_paths]
    depths = [im[:, :, 3].copy() for im in orig]  # 第 4 通道 = Depth
    label_paths = []
    for p in img_paths:
        lp = test_lbl_dir / (p.stem + ".txt")
        if not lp.exists():
            raise FileNotFoundError(f"缺少 label: {lp}")
        label_paths.append(lp)

    # fixed_valid_depth / full_constant_depth 共用的全局有效均值：仅统计 depth>0，
    # 整 split 只算一次（不再逐图重算），两者复用同一 fixed_value。
    fixed_value = None
    if "fixed_valid_depth" in selected or "full_constant_depth" in selected:
        valid_arrays = [d[d > 0] for d in depths if np.any(d > 0)]
        if not valid_arrays:
            raise RuntimeError("当前 split 无任何有效 Depth(全为0)，无法计算 fixed_valid_depth 全局值")
        global_valid_mean = float(np.concatenate(valid_arrays).mean())
        fixed_value = int(np.clip(np.rint(global_valid_mean), 1, 255))
        print(f"[生成] Fixed Valid Depth global value = {fixed_value}  "
              f"(global_valid_mean={global_valid_mean:.4f}, 来自 {len(valid_arrays)} 张有效图)")

    # full_constant_depth 扫描点日志：明确打印每个 variant 对应的常数值（便于实验记录）。
    # 顺序：1 / 64 / 128 / 全局 fixed value(~170) / 224 / 255。仅打印被选中的 variant。
    _sweep_print_order = ["full_constant_depth_1", "full_constant_depth_64",
                          "full_constant_depth_128", "full_constant_depth",
                          "full_constant_depth_224", "full_constant_depth_255"]
    for _v in _sweep_print_order:
        if _v in selected:
            if _v == "full_constant_depth":
                if fixed_value is not None:
                    print(f"[生成] {VARIANT_LABELS[_v]:22s} -> value={fixed_value}  (global fixed value)")
            elif _v in CONSTANT_SWEEP_VALUES:
                print(f"[生成] {VARIANT_LABELS[_v]:22s} -> value={CONSTANT_SWEEP_VALUES[_v]}")

    # fixed_valid_depth 扫描点日志：明确打印每个 variant 对应的常数值（保留 valid mask 形状）。
    # 顺序：1 / 64 / 128 / 全局 fixed value(~170) / 224 / 255。仅打印被选中的 variant。
    _fixed_sweep_print_order = ["fixed_valid_depth_1", "fixed_valid_depth_64",
                                "fixed_valid_depth_128", "fixed_valid_depth",
                                "fixed_valid_depth_224", "fixed_valid_depth_255"]
    for _v in _fixed_sweep_print_order:
        if _v in selected:
            if _v == "fixed_valid_depth":
                if fixed_value is not None:
                    print(f"[生成] {VARIANT_LABELS[_v]:22s} -> value={fixed_value}  (global fixed value)")
            elif _v in FIXED_VALID_SWEEP_VALUES:
                print(f"[生成] {VARIANT_LABELS[_v]:22s} -> value={FIXED_VALID_SWEEP_VALUES[_v]}")

    # scaled_true_depth 扫描点日志：明确打印每个 variant 对应的缩放比例（保留真实 Depth 空间模式、仅降幅度）。
    for _v in ("scaled_true_depth_25", "scaled_true_depth_50", "scaled_true_depth_75"):
        if _v in selected:
            print(f"[生成] {VARIANT_LABELS[_v]:22s} -> scale={SCALED_TRUE_DEPTH_SCALES[_v]:.2f}  "
                  f"(保留真实 Depth 空间模式 + valid mask，仅降低第4通道整体幅度)")

    # flat_scaled_depth 配对点日志：复用 scaled_true_depth 的 scale / round / clip / invalid 处理，
    # 但随后把当前图片有效区域拍平为区域均值（删有效区域内部 Depth 空间变化；幅度与 mask 与 Scaled True 几乎一致）。
    for _v in ("flat_scaled_depth_25", "flat_scaled_depth_50", "flat_scaled_depth_75"):
        if _v in selected:
            print(f"[生成] {VARIANT_LABELS[_v]:22s} -> scale={FLAT_SCALED_DEPTH_SCALES[_v]:.2f}  "
                  f"(复用 Scaled True 缩放逻辑 + 当前图有效区域拍平为均值，删有效区域内部 Depth 空间变化)")

    # shuffled_depth 的 derangement（固定 seed）
    perm = derangement(n, seed)

    checks = []  # 逐图校验记录
    variant_roots = {}
    for vname in selected:
        vroot = out_root / vname / "rgbd"
        img_out = vroot / "images" / "test"
        lbl_out = vroot / "labels" / "test"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        variant_roots[vname] = vroot

        if vname == "true_depth":
            src_depths = depths
        elif vname == "zero_depth":
            src_depths = [np.zeros_like(d) for d in depths]
        elif vname == "spatial_shuffle_depth":
            # 同图自身 Depth，仅随机打乱像素位置（seed = 全局 seed + 图片下标，可复现）
            src_depths = [spatial_shuffle_depth_channel(depths[i], seed + i) for i in range(n)]
        elif vname == "blurred_depth":
            # 同图自身原始 Depth，仅对第 4 通道做 masked Gaussian blur（11x11）。
            # 不取 shuffled Depth：RGB_i + Blur(Depth_i, 11)。
            src_depths = [blurred_depth_channel(depths[i], 11) for i in range(n)]
        elif vname == "blurred_depth_31":
            # 同图自身原始 Depth，masked Gaussian blur（31x31）：RGB_i + Blur(Depth_i, 31)。
            src_depths = [blurred_depth_channel(depths[i], 31) for i in range(n)]
        elif vname == "blurred_depth_51":
            # 同图自身原始 Depth，masked Gaussian blur（51x51）：RGB_i + Blur(Depth_i, 51)。
            src_depths = [blurred_depth_channel(depths[i], 51) for i in range(n)]
        elif vname == "flat_valid_depth":
            # 同图自身 Depth，有效区域全部替换为有效区域均值（剥离连续几何）：
            # RGB_i + FlatValid(Depth_i)。不取其它图。
            src_depths = [flat_valid_depth_channel(depths[i]) for i in range(n)]
        elif vname == "fixed_valid_depth":
            # 同图自身 Depth，有效区域全部替换为整 split 共用的固定值 fixed_value（再删整体水平）：
            # RGB_i + FixedValid(Depth_i, fixed_value)。所有图共用同一值。
            src_depths = [fixed_valid_depth_channel(depths[i], fixed_value) for i in range(n)]
        elif vname == "full_constant_depth":
            # 同图自身 Depth，第 4 通道全部（含原 invalid）填成同一 fixed_value（删 valid mask 形状）：
            # RGB_i + FullConstant(Depth_i, fixed_value)。复用与 Fixed Valid 相同的 fixed_value。
            src_depths = [full_constant_depth_channel(depths[i], fixed_value) for i in range(n)]
        elif vname in CONSTANT_SWEEP_VALUES:
            # 同图自身 Depth，第 4 通道全部（含原 invalid）填成指定常数值（删 valid mask 形状）：
            # RGB_i + FullConstant(Depth_i, value)。value 来自 CONSTANT_SWEEP_VALUES，不复用 global mean。
            src_depths = [full_constant_depth_channel(depths[i], CONSTANT_SWEEP_VALUES[vname]) for i in range(n)]
        elif vname in FIXED_VALID_SWEEP_VALUES:
            # 同图自身 Depth，有效区域填成指定常数、invalid 保持 0（保留 valid mask 形状）：
            # RGB_i + FixedValid(Depth_i, value)。value 来自 FIXED_VALID_SWEEP_VALUES，不复用 global mean。
            # 复用 fixed_valid_depth_channel（该函数已保证 ch4>0→value、ch4==0→0）。
            src_depths = [fixed_valid_depth_channel(depths[i], FIXED_VALID_SWEEP_VALUES[vname]) for i in range(n)]
        elif vname in SCALED_TRUE_DEPTH_SCALES:
            # 同图自身真实 Depth，仅把第4通道整体幅度按 scale 缩放（保留空间模式 + valid mask + 相对高低）：
            # RGB_i + Scale(TrueDepth_i, scale)。复用 scaled_true_depth_channel，不复制代码。
            src_depths = [scaled_true_depth_channel(depths[i], SCALED_TRUE_DEPTH_SCALES[vname]) for i in range(n)]
        elif vname in FLAT_SCALED_DEPTH_SCALES:
            # 配对控制：同图自身真实 Depth，先按相同 scale 缩放（严格复用 Scaled True 逻辑），
            # 再把当前图有效区域拍平为区域均值（删有效区域内部 Depth 空间变化）：
            # RGB_i + FlatScaled(TrueDepth_i, scale)。复用 flat_scaled_depth_channel，不复制代码。
            src_depths = [flat_scaled_depth_channel(depths[i], FLAT_SCALED_DEPTH_SCALES[vname]) for i in range(n)]
        else:  # shuffled_depth：跨图随机 Depth（derangement，不取自己）
            src_depths = [depths[perm[i]] for i in range(n)]

        for i, p in enumerate(img_paths):
            im = orig[i].copy()
            new_ch4 = src_depths[i]
            # 形状对齐（极少数尺寸不一致时 resize 到原图尺寸，保持公平）
            if new_ch4.shape != im[:, :, 3].shape:
                new_ch4 = cv2.resize(new_ch4, (im.shape[1], im.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            im[:, :, 3] = new_ch4.astype(np.uint8)
            dst = img_out / p.name
            ok = cv2.imwrite(str(dst), im, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            if not ok:
                raise RuntimeError(f"写图失败: {dst}")
            shutil.copy2(label_paths[i], lbl_out / (p.stem + ".txt"))

            # 逐图校验
            ch4 = im[:, :, 3]
            rec = {
                "variant": vname,
                "image": p.name,
                "shape": list(im.shape),
                "dtype": str(im.dtype),
                "ch4": ch4_stats(ch4),
            }
            if vname == "zero_depth":
                assert ch4.max() == 0 and ch4.min() == 0, f"zero_depth 第4通道非全0: {p.name}"
            if vname == "shuffled_depth":
                assert not np.array_equal(ch4, depths[i]), \
                    f"shuffled_depth 第4通道与原样本相同(回配自己): {p.name}"
                # 必须确实来自某张其它图
                assert any(np.array_equal(ch4, d) for d in depths), \
                    f"shuffled_depth 第4通道不匹配任何样本: {p.name}"
            if vname == "spatial_shuffle_depth":
                # 必须与原样本 Depth 不同（确实被打乱）
                assert not np.array_equal(ch4, depths[i]), \
                    f"spatial_shuffle_depth 第4通道与原样本相同(未被打乱): {p.name}"
                # 排序后像素集合必须与原 Depth 完全一致 -> 统计量必然一致
                assert np.array_equal(np.sort(ch4.ravel()), np.sort(depths[i].ravel())), \
                    f"spatial_shuffle_depth 像素集合与原 Depth 不一致(非纯 permutation): {p.name}"
                # 双重校验：逐图统计相等（permutation 必然成立）
                assert ch4_stats(ch4) == ch4_stats(depths[i]), \
                    f"spatial_shuffle_depth 第4通道统计与原 Depth 不一致: {p.name}"
            if vname in ("blurred_depth", "blurred_depth_31", "blurred_depth_51",
                         "flat_valid_depth", "fixed_valid_depth"):
                # 1) shape 必须与原图一致（H, W, 4）
                assert im.shape == orig[i].shape, \
                    f"{vname} 图像 shape 改变: {p.name}"
                # 2) dtype 必须为 uint8
                assert im.dtype == np.uint8, \
                    f"{vname} dtype 非 uint8: {p.name}"
                # 3) invalid mask 严格保持：原始 depth==0 与 flat/blurred==0 必须完全一致
                assert np.array_equal(ch4 == 0, depths[i] == 0), \
                    f"{vname} 改变了 invalid=0 区域(0=invalid 被污染): {p.name}"
                # 4) RGB 三通道必须完全不变（只动第 4 通道）
                assert np.array_equal(im[:, :, :3], orig[i][:, :, :3]), \
                    f"{vname} 改变了 RGB 三通道: {p.name}"
                # 5) label 必须完全不变（逐文件内容比对拷贝结果）
                lbl_dst = lbl_out / (p.stem + ".txt")
                assert label_paths[i].read_text(encoding="utf-8") == lbl_dst.read_text(encoding="utf-8"), \
                    f"{vname} 改变了 label: {p.name}"
                # 6) flat/fixed_valid_depth：有效区域内部必须只有一个唯一 Depth 值
                if vname in ("flat_valid_depth", "fixed_valid_depth") and np.any(depths[i] > 0):
                    uniq = np.unique(ch4[depths[i] > 0])
                    assert uniq.size == 1, \
                        f"{vname} 有效区域非单一常数值: {p.name} uniq={uniq.tolist()}"
                    # fixed_valid_depth：该唯一值必须等于整 split 共用的 fixed_value（即所有图共用同一值）
                    if vname == "fixed_valid_depth":
                        assert int(uniq[0]) == fixed_value, \
                            f"fixed_valid_depth 唯一值 != 全局 fixed_value: {p.name} " \
                            f"uniq={int(uniq[0])} fixed={fixed_value}"
                # 7) 仅 valid 区域（original_depth > 0）计算 Depth MAE；invalid=0 不参与
                orig_f = depths[i].astype(np.float32)
                valid_mask = depths[i] > 0
                if np.any(valid_mask):
                    mae = float(np.mean(np.abs(ch4.astype(np.float32) - orig_f)[valid_mask]))
                else:
                    mae = 0.0
                rec["depth_mae"] = mae
            if vname == "full_constant_depth":
                # 1) shape 必须与原图一致（H, W, 4）
                assert im.shape == orig[i].shape, \
                    f"full_constant_depth 图像 shape 改变: {p.name}"
                # 2) dtype 必须为 uint8
                assert im.dtype == np.uint8, \
                    f"full_constant_depth dtype 非 uint8: {p.name}"
                # 3) 整张第4通道所有像素只有一个唯一值
                uniq = np.unique(ch4)
                assert uniq.size == 1, \
                    f"full_constant_depth 第4通道非单一常数值: {p.name} uniq={uniq.tolist()}"
                # 4) 唯一值必须等于统一的 fixed_value（所有图共用同一值）
                assert int(uniq[0]) == fixed_value, \
                    f"full_constant_depth 唯一值 != 全局 fixed_value: {p.name} " \
                    f"uniq={int(uniq[0])} fixed={fixed_value}"
                # 5) RGB 三通道必须完全不变（只动第 4 通道）
                assert np.array_equal(im[:, :, :3], orig[i][:, :, :3]), \
                    f"full_constant_depth 改变了 RGB 三通道: {p.name}"
                # 6) label 必须完全不变（逐文件内容比对拷贝结果）
                lbl_dst = lbl_out / (p.stem + ".txt")
                assert label_paths[i].read_text(encoding="utf-8") == lbl_dst.read_text(encoding="utf-8"), \
                    f"full_constant_depth 改变了 label: {p.name}"
                # 注：full_constant_depth 故意删除 invalid mask（全图统一 fixed_value），这是实验定义，不做 mask 一致性断言
                # 7) Depth MAE：仍只在原始 valid 区域（original_depth > 0）计算，保持与已有变体可比；invalid=0 不参与
                orig_f = depths[i].astype(np.float32)
                valid_mask = depths[i] > 0
                if np.any(valid_mask):
                    mae = float(np.mean(np.abs(ch4.astype(np.float32) - orig_f)[valid_mask]))
                else:
                    mae = 0.0
                rec["depth_mae"] = mae
            if vname in CONSTANT_SWEEP_VALUES:
                expected_val = CONSTANT_SWEEP_VALUES[vname]
                # 1) shape 必须与原图一致（H, W, 4）
                assert im.shape == orig[i].shape, \
                    f"{vname} 图像 shape 改变: {p.name}"
                # 2) dtype 必须为 uint8
                assert im.dtype == np.uint8, \
                    f"{vname} dtype 非 uint8: {p.name}"
                # 3) 整张第4通道所有像素（含原 invalid）只有一个唯一值
                uniq = np.unique(ch4)
                assert uniq.size == 1, \
                    f"{vname} 第4通道非单一常数值: {p.name} uniq={uniq.tolist()}"
                # 4) 唯一值必须严格等于该 variant 指定的常数值
                assert int(uniq[0]) == expected_val, \
                    f"{vname} 唯一值 != 指定常数值: {p.name} uniq={int(uniq[0])} expected={expected_val}"
                # 5) RGB 三通道必须完全不变（只动第 4 通道）
                assert np.array_equal(im[:, :, :3], orig[i][:, :, :3]), \
                    f"{vname} 改变了 RGB 三通道: {p.name}"
                # 6) label 必须完全不变（逐文件内容比对拷贝结果）
                lbl_dst = lbl_out / (p.stem + ".txt")
                assert label_paths[i].read_text(encoding="utf-8") == lbl_dst.read_text(encoding="utf-8"), \
                    f"{vname} 改变了 label: {p.name}"
                # 注：constant sweep 故意删除 invalid mask（全图统一常数值），这是实验定义，不做 mask 一致性断言
                # 7) Depth MAE：仍只在原始 valid 区域（original_depth > 0）计算，保持与已有变体可比；invalid=0 不参与
                orig_f = depths[i].astype(np.float32)
                valid_mask = depths[i] > 0
                if np.any(valid_mask):
                    mae = float(np.mean(np.abs(ch4.astype(np.float32) - orig_f)[valid_mask]))
                else:
                    mae = 0.0
                rec["depth_mae"] = mae
            if vname in FIXED_VALID_SWEEP_VALUES:
                expected_val = FIXED_VALID_SWEEP_VALUES[vname]
                # 1) shape 必须与原图一致（H, W, 4）
                assert im.shape == orig[i].shape, \
                    f"{vname} 图像 shape 改变: {p.name}"
                # 2) dtype 必须为 uint8
                assert im.dtype == np.uint8, \
                    f"{vname} dtype 非 uint8: {p.name}"
                # 3) invalid mask 严格保持：原始 depth==0 与 generated==0 必须完全一致（保留 valid mask 形状）
                assert np.array_equal(ch4 == 0, depths[i] == 0), \
                    f"{vname} 改变了 invalid=0 区域(0=invalid 被污染): {p.name}"
                # 4) valid 区域（original_depth>0）只能有一个唯一非零值，且必须等于指定值
                if np.any(depths[i] > 0):
                    uniq = np.unique(ch4[depths[i] > 0])
                    assert uniq.size == 1, \
                        f"{vname} 有效区域非单一常数值: {p.name} uniq={uniq.tolist()}"
                    assert int(uniq[0]) == expected_val, \
                        f"{vname} 有效区域唯一值 != 指定常数值: {p.name} uniq={int(uniq[0])} expected={expected_val}"
                # 5) RGB 三通道必须完全不变（只动第 4 通道）
                assert np.array_equal(im[:, :, :3], orig[i][:, :, :3]), \
                    f"{vname} 改变了 RGB 三通道: {p.name}"
                # 6) label 必须完全不变（逐文件内容比对拷贝结果）
                lbl_dst = lbl_out / (p.stem + ".txt")
                assert label_paths[i].read_text(encoding="utf-8") == lbl_dst.read_text(encoding="utf-8"), \
                    f"{vname} 改变了 label: {p.name}"
                # 7) Depth MAE：仍只在原始 valid 区域（original_depth > 0）计算，保持与已有变体可比；invalid=0 不参与
                orig_f = depths[i].astype(np.float32)
                valid_mask = depths[i] > 0
                if np.any(valid_mask):
                    mae = float(np.mean(np.abs(ch4.astype(np.float32) - orig_f)[valid_mask]))
                else:
                    mae = 0.0
                rec["depth_mae"] = mae
            if vname in SCALED_TRUE_DEPTH_SCALES:
                # 1) shape 必须与原图一致（H, W, 4）
                assert im.shape == orig[i].shape, \
                    f"{vname} 图像 shape 改变: {p.name}"
                # 2) dtype 必须为 uint8
                assert im.dtype == np.uint8, \
                    f"{vname} dtype 非 uint8: {p.name}"
                # 3) invalid mask 严格保持：原始 depth==0 与 scaled==0 必须完全一致
                assert np.array_equal(ch4 == 0, depths[i] == 0), \
                    f"{vname} 改变了 invalid=0 区域(0=invalid 被污染): {p.name}"
                # 4) 所有原始 valid 像素缩放后必须仍然 > 0（clip 下限 1 保证）
                if np.any(depths[i] > 0):
                    assert ch4[depths[i] > 0].min() >= 1, \
                        f"{vname} 有效像素缩放后存在 <=0（应钳到 >=1）: {p.name} " \
                        f"min={int(ch4[depths[i] > 0].min())}"
                # 5) RGB 三通道必须完全不变（只动第 4 通道）
                assert np.array_equal(im[:, :, :3], orig[i][:, :, :3]), \
                    f"{vname} 改变了 RGB 三通道: {p.name}"
                # 6) label 必须完全不变（逐文件内容比对拷贝结果）
                lbl_dst = lbl_out / (p.stem + ".txt")
                assert label_paths[i].read_text(encoding="utf-8") == lbl_dst.read_text(encoding="utf-8"), \
                    f"{vname} 改变了 label: {p.name}"
                # 7) scaled depth 与 true depth 必须不完全相同（正常存在幅度差异的图）
                if np.any(depths[i] > 0):
                    assert not np.array_equal(ch4[depths[i] > 0], depths[i][depths[i] > 0]), \
                        f"{vname} 有效区域与 true depth 完全相同（未缩放）: {p.name}"
                # 8) 仅 valid 区域（original_depth > 0）统计平均 Depth + 计算 Depth MAE；invalid=0 不参与
                orig_f = depths[i].astype(np.float32)
                valid_mask = depths[i] > 0
                if np.any(valid_mask):
                    vmean = float(ch4[valid_mask].mean())
                    mae = float(np.mean(np.abs(ch4.astype(np.float32) - orig_f)[valid_mask]))
                else:
                    vmean = 0.0
                    mae = 0.0
                rec["valid_depth_mean"] = vmean
                rec["depth_mae"] = mae
            if vname in FLAT_SCALED_DEPTH_SCALES:
                scale = FLAT_SCALED_DEPTH_SCALES[vname]
                # 1) shape 必须与原图一致（H, W, 4）
                assert im.shape == orig[i].shape, \
                    f"{vname} 图像 shape 改变: {p.name}"
                # 2) dtype 必须为 uint8
                assert im.dtype == np.uint8, \
                    f"{vname} dtype 非 uint8: {p.name}"
                # 3) invalid mask 严格保持：原始 depth==0 与 flat_scaled==0 必须完全一致
                #    （同时保证 Scaled True 与 Flat Scaled 使用完全相同的 valid mask）
                assert np.array_equal(ch4 == 0, depths[i] == 0), \
                    f"{vname} 改变了 invalid=0 区域(0=invalid 被污染 / valid mask 不一致): {p.name}"
                # 4) 所有原始 valid 像素在 flat_scaled 中必须 > 0
                if np.any(depths[i] > 0):
                    assert ch4[depths[i] > 0].min() >= 1, \
                        f"{vname} 有效像素存在 <=0: {p.name} min={int(ch4[depths[i] > 0].min())}"
                # 5) 当前图片 valid 区域只能有一个唯一非零值（拍平为区域均值常数）
                if np.any(depths[i] > 0):
                    uniq = np.unique(ch4[depths[i] > 0])
                    assert uniq.size == 1, \
                        f"{vname} 有效区域非单一常数值: {p.name} uniq={uniq.tolist()}"
                # 6) RGB 三通道必须完全不变（只动第 4 通道）
                assert np.array_equal(im[:, :, :3], orig[i][:, :, :3]), \
                    f"{vname} 改变了 RGB 三通道: {p.name}"
                # 7) label 必须完全不变（逐文件内容比对拷贝结果）
                lbl_dst = lbl_out / (p.stem + ".txt")
                assert label_paths[i].read_text(encoding="utf-8") == lbl_dst.read_text(encoding="utf-8"), \
                    f"{vname} 改变了 label: {p.name}"
                # 8) 平均幅度匹配校验（本轮最关键控制条件）：复用完全相同的 scaled_true_depth 逻辑
                #    生成 scaled，再对比 flat_scaled 的有效区域平均 Depth；整数常数仅引入 <=0.5 舍入误差。
                valid_mask = depths[i] > 0
                if np.any(valid_mask):
                    scaled = scaled_true_depth_channel(depths[i], scale)  # 严格复用 Scaled True 逻辑
                    scaled_valid_mean = float(scaled[valid_mask].mean())
                    flat_valid_mean = float(ch4[valid_mask].mean())
                    assert abs(scaled_valid_mean - flat_valid_mean) <= 0.5, \
                        f"{vname} 平均幅度与 Scaled True 偏差 >0.5: {p.name} " \
                        f"scaled_mean={scaled_valid_mean:.4f} flat_mean={flat_valid_mean:.4f} " \
                        f"diff={abs(scaled_valid_mean - flat_valid_mean):.4f}"
                    # 9) 仅 valid 区域（original_depth > 0）统计平均 Depth + 计算 Depth MAE；invalid=0 不参与
                    orig_f = depths[i].astype(np.float32)
                    vmean = flat_valid_mean
                    mae = float(np.mean(np.abs(ch4.astype(np.float32) - orig_f)[valid_mask]))
                else:
                    vmean = 0.0
                    mae = 0.0
                rec["valid_depth_mean"] = vmean
                rec["depth_mae"] = mae
            checks.append(rec)

        build_variant_yaml(src_yaml, vroot)
        print(f"[生成] 变体完成: {vname} -> {vroot}")

    # 控制台汇总校验
    print("\n=== 逐图生成校验（第4通道 Depth 统计）===")
    for i, p in enumerate(img_paths):
        line = f"  {p.name:40s}"
        for vname in selected:
            rec = next(c for c in checks if c["variant"] == vname and c["image"] == p.name)
            s = rec["ch4"]
            line += f" | {vname[:4]}:{s['min']:5.0f}/{s['max']:5.0f}/{s['mean']:6.1f}/{s['std']:5.1f}"
        print(line)
    print("\n[校验] 全部 shape=(H,W,4) / dtype=uint8 / zero 全0 / shuffled 异于原样本 / spatial 排序一致：通过")

    # Depth MAE 汇总（blur 系列 + flat/fixed/full-constant + 两个 sweep + scaled_true_depth；
    # invalid=0 区域不参与，逐图平均全 test 集）
    blur_variants = [v for v in ("blurred_depth", "blurred_depth_31", "blurred_depth_51",
                                 "flat_valid_depth", "fixed_valid_depth", "full_constant_depth",
                                 "full_constant_depth_1", "full_constant_depth_64",
                                 "full_constant_depth_128", "full_constant_depth_224",
                                 "full_constant_depth_255",
                                 "fixed_valid_depth_1", "fixed_valid_depth_64",
                                 "fixed_valid_depth_128", "fixed_valid_depth_224",
                                 "fixed_valid_depth_255",
                                 "scaled_true_depth_25", "scaled_true_depth_50",
                                 "scaled_true_depth_75",
                                 "flat_scaled_depth_25", "flat_scaled_depth_50",
                                 "flat_scaled_depth_75") if v in selected]
    if blur_variants:
        print("\n=== Depth MAE mean（仅 valid 区域，全 test 集平均）===")
        for vname in blur_variants:
            maes = [c["depth_mae"] for c in checks if c["variant"] == vname and "depth_mae" in c]
            if maes:
                print(f"  {VARIANT_LABELS[vname]:22s} Depth MAE mean = {float(np.mean(maes)):.4f}  (n={len(maes)})")

    # Scaled True Depth / Flat Scaled Depth：额外输出「valid 区域平均 Depth」，观察整 split 整体幅度
    # （Flat Scaled 与 Scaled True 配对，应几乎一致，仅 <=0.5 舍入误差）
    scaled_variants = [v for v in ("scaled_true_depth_25", "scaled_true_depth_50",
                                   "scaled_true_depth_75",
                                   "flat_scaled_depth_25", "flat_scaled_depth_50",
                                   "flat_scaled_depth_75") if v in selected]
    if scaled_variants:
        print("\n=== Scaled True / Flat Scaled Depth：valid 区域平均 Depth（全 test 集平均）===")
        for vname in scaled_variants:
            vmeans = [c["valid_depth_mean"] for c in checks if c["variant"] == vname and "valid_depth_mean" in c]
            if vmeans:
                print(f"  {VARIANT_LABELS[vname]:22s} Valid Depth Mean = {float(np.mean(vmeans)):.2f}  (n={len(vmeans)})")

    (out_root / "_ablation_checks.json").write_text(
        json.dumps({"n": n, "seed": seed, "checks": checks}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    return variant_roots, n


# --------------------------------------------------------------------------- 评估
def evaluate_variant(yaml_path: Path, weights: str, imgsz: int, batch: int, device, split="test"):
    # 4 通道 RGBD 加载逻辑只存在于仓库根目录的 vendored ultralytics 副本中，
    # 必须把仓库根插入 sys.path 最前，避免误用 conda site-packages 里的原生 ultralytics。
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import ultralytics  # noqa: F401  (触发路径生效后再 import YOLO)
    from ultralytics import YOLO

    print(f"[ultralytics] {ultralytics.__file__}")
    model = YOLO(weights, task="pose")
    val_kwargs = dict(
        data=str(yaml_path),
        split=split,
        imgsz=imgsz,
        batch=batch,
        verbose=False,
        plots=False,
        save_json=False,
    )
    if device is not None:
        val_kwargs["device"] = device
    metrics = model.val(**val_kwargs)
    d = metrics.results_dict
    return {
        "box_p": float(d["metrics/precision(B)"]),
        "box_r": float(d["metrics/recall(B)"]),
        "box_map50": float(d["metrics/mAP50(B)"]),
        "box_map": float(d["metrics/mAP50-95(B)"]),
        "pose_p": float(d["metrics/precision(P)"]),
        "pose_r": float(d["metrics/recall(P)"]),
        "pose_map50": float(d["metrics/mAP50(P)"]),
        "pose_map": float(d["metrics/mAP50-95(P)"]),
    }


def _ablation_conclusion(results: dict, selected: list) -> str:
    """保守自动结论：仅当 true/zero 同时可用时比较 Pose mAP50-95。

    明确不推断“全部性能提升都来自真实三维几何信息”，因为 Zero Depth 已明显改变
    输入分布（第4通道全0），属分布外消融。
    """
    if "true_depth" not in results or "zero_depth" not in results:
        return "本次未同时执行 True/Zero 变体，跳过自动结论。"
    td = results["true_depth"]
    zd = results["zero_depth"]
    if td["pose_map"] > zd["pose_map"]:
        return (f"Zero Depth 后 Pose mAP50-95 下降（True={td['pose_map']:.4f} > "
                f"Zero={zd['pose_map']:.4f}），表明当前 RGBD 模型实际利用了第4通道提供的 "
                f"Depth 信息，而非完全忽略 Depth 通道。")
    if td["pose_map"] < zd["pose_map"]:
        return (f"Zero Depth 后 Pose mAP50-95 不降反升（True={td['pose_map']:.4f} < "
                f"Zero={zd['pose_map']:.4f}），属非预期，建议检查消融数据生成与评估一致性。")
    return (f"True Depth 与 Zero Depth 的 Pose mAP50-95 基本持平 "
            f"（True={td['pose_map']:.4f} ≈ Zero={zd['pose_map']:.4f}），"
            f"无法据此判断 Depth 是否被利用。")


def print_table(results: dict, out_root: Path, selected: list):
    """按 selected 变体（保持 VARIANTS 顺序）输出逐指标表 + True-Zero 差值 + 简洁对照表
    + 保守自动结论，并写出 ablation_result.json 与一行 DEPTH_ABLATION_JSON（供 GUI 捕获）。"""
    keys = ("box_p", "box_r", "box_map50", "box_map", "pose_p", "pose_r", "pose_map50", "pose_map")
    metric_labels = {
        "box_p": "Box P", "box_r": "Box R", "box_map50": "Box mAP50",
        "box_map": "Box mAP50-95", "pose_p": "Pose P", "pose_r": "Pose R",
        "pose_map50": "Pose mAP50", "pose_map": "Pose mAP50-95",
    }
    col_w = 20
    hdr = f"{'Metric':14s}"
    for v in selected:
        hdr += f" | {VARIANT_LABELS[v]:>{col_w}s}"
    sep = "-" * (14 + 3 + col_w * len(selected))
    print("\n" + "=" * len(sep))
    print(hdr)
    print(sep)
    for k in keys:
        line = f"{metric_labels[k]:14s}"
        for v in selected:
            line += f" | {results[v][k]:{col_w}.4f}"
        print(line)
    print(sep)

    td = results.get("true_depth")
    zd = results.get("zero_depth")
    if td is not None and zd is not None:
        print("\n=== True Depth vs Zero Depth ===")
        print(f"Box  mAP50-95 : True={td['box_map']:.4f}  Zero={zd['box_map']:.4f}  "
              f"True-Zero={td['box_map'] - zd['box_map']:+.4f}")
        print(f"Pose mAP50-95 : True={td['pose_map']:.4f}  Zero={zd['pose_map']:.4f}  "
              f"True-Zero={td['pose_map'] - zd['pose_map']:+.4f}")

    # 简洁对照表
    print("\n=== 简洁对照表 ===")
    print(f"{'Variant':20s} {'Box mAP50-95':>14s} {'Pose mAP50-95':>16s}")
    print("-" * 52)
    for v in selected:
        print(f"{VARIANT_LABELS[v]:20s} {results[v]['box_map']:14.4f} {results[v]['pose_map']:16.4f}")
    if td is not None and zd is not None:
        print(f"{'True-Zero':20s} {td['box_map'] - zd['box_map']:14.4f} "
              f"{td['pose_map'] - zd['pose_map']:16.4f}")

    conclusion = _ablation_conclusion(results, selected)
    print("\n[结论] " + conclusion)
    print("=" * len(sep))

    (out_root / "ablation_result.json").write_text(
        json.dumps({"results": results, "selected": selected, "conclusion": conclusion},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[结果] 已写出 {out_root / 'ablation_result.json'}")
    # 机器可读摘要（供 GUI 实时捕获）
    print("DEPTH_ABLATION_JSON " + json.dumps(
        {"selected": selected, "results": results, "conclusion": conclusion},
        ensure_ascii=False))


# --------------------------------------------------------------------------- main
def main():
    args = parse_args()
    src_yaml = Path(args.rgbd_yaml)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # 选定变体（保持 VARIANTS 原有顺序，仅保留请求子集）
    selected = [v for v in VARIANTS if v in set(args.variants)]
    if not selected:
        print("[错误] --variants 为空或不含合法变体")
        sys.exit(2)

    print("=" * 64)
    print("RGBD Depth 消融评估")
    print(f"weights : {args.weights}")
    print(f"rgbd    : {src_yaml}")
    print(f"out     : {out_root}")
    print(f"imgsz  : {args.imgsz}   batch: {args.batch}   seed: {args.seed}")
    print(f"split   : {args.split}")
    print(f"variants: {selected}")
    print("=" * 64)

    variant_roots, n = generate_ablation_data(src_yaml, out_root, args.seed, selected)

    if args.no_eval:
        print("\n[no-eval] 已生成临时数据并完成校验，跳过 model.val。运行去掉 --no-eval 执行评估。")
        return

    results = {}
    for vname in selected:
        yaml_path = variant_roots[vname] / "data_rgbd_ablation.yaml"
        print(f"\n>>> 评估变体: {vname}  ({yaml_path})")
        results[vname] = evaluate_variant(
            yaml_path, args.weights, args.imgsz, args.batch, args.device, args.split)

    print_table(results, out_root, selected)


if __name__ == "__main__":
    main()
