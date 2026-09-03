"""LamaInferenceService 自动测试（Phase 5）。

不依赖真实 ONNX 模型，测试：
1. 服务创建 / is_loaded / load_model 失败处理
2. run() 无模型 -> 空数组
3. 预处理：_image_to_float_chw_hole_zero01（CHW 布局 + 洞区置 0 + 0..1）
4. 后处理：_out_chw_to_hwc（0..1 检测 + 乘 255 + clamp + HWC）
5. 合成：_composite_by_mask（洞区=fill，非洞区=base）
6. 完整 pipeline（用 mock session 模拟 ONNX Run）

启动：
    python YJJ_Pose_Scripts/gui/tests/test_lama_inference.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

GUI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GUI_DIR))

import numpy as np                  # noqa: E402
import cv2                          # noqa: E402

from features.lama.services.lama_inference_service import (   # noqa: E402
    LamaInferenceService,
    _INFERENCE_SIZE,
    _DILATE_RADIUS,
    _BINARIZE_THRESH,
    _OUTPUT_01_THRESHOLD,
)


def _check(name: str, cond: bool, failures: list[str], detail: str = "") -> None:
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}  {detail}")
        if failures is not None:
            failures.append(name)


class _MockSession:
    """模拟 onnxruntime.InferenceSession，输出 = 输入 image（恒等映射）。"""

    def __init__(self) -> None:
        self._input_names = ["image", "mask"]
        self._output_names = ["output"]

    def get_inputs(self) -> list["_MockSession._IO"]:
        class _IO:
            def __init__(self, name: str) -> None:
                self.name = name
        return [_IO(n) for n in self._input_names]

    def get_outputs(self) -> list["_MockSession._IO"]:
        class _IO:
            def __init__(self, name: str) -> None:
                self.name = name
        return [_IO(n) for n in self._output_names]

    def run(self, output_names: list[str] | None, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        """输出 = 输入 image（恒等映射），形状 (1,3,H,W)。"""
        img = feed["image"]      # (1,3,H,W) float32 0..1
        # 直接返回 image 作为 output（模拟模型输出 0..1）
        return [img.copy()]


def main() -> None:
    failures = []

    # ================================================================ 1. 服务创建 + 无模型
    print("[1] 服务创建 + is_loaded / load_model 失败处理")
    svc = LamaInferenceService()
    _check("1a. is_loaded() == False（未加载模型）",
           not svc.is_loaded(), failures)

    # load_model 不存在的路径
    ok = svc.load_model("/nonexistent/path/model.onnx")
    _check("1b. load_model 不存在路径 -> False", not ok, failures)
    _check("1c. 加载失败后 is_loaded() == False",
           not svc.is_loaded(), failures)

    # run 无模型 -> 空数组
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    result = svc.run(dummy, dummy[..., 0])
    _check("1d. 无模型 run -> 空数组",
           result.size == 0, failures)

    # run None 输入 -> 空数组
    _check("1e. None image -> 空数组",
           svc.run(None, None).size == 0, failures)

    # ================================================================ 2. 预处理 _image_to_float_chw_hole_zero01
    print("[2] 预处理：_image_to_float_chw_hole_zero01")
    # 造一张 512x512 的图，值全 128，mask 在左上角 100x100 画白
    img = np.full((_INFERENCE_SIZE, _INFERENCE_SIZE, 3), 128, dtype=np.uint8)
    mask = np.zeros((_INFERENCE_SIZE, _INFERENCE_SIZE), dtype=np.uint8)
    mask[:100, :100] = 255
    mask_hw = (mask > 0).astype(np.float32)

    chw = LamaInferenceService._image_to_float_chw_hole_zero01(img, mask_hw)
    _check("2a. 输出形状 (1,3,512,512)",
           chw.shape == (1, 3, _INFERENCE_SIZE, _INFERENCE_SIZE), failures,
           f"got {chw.shape}")
    _check("2b. 数据类型 float32",
           chw.dtype == np.float32, failures)
    # 0..1 范围
    _check("2c. 值在 [0,1] 范围",
           float(chw.min()) >= 0.0 and float(chw.max()) <= 1.0, failures,
           f"min={float(chw.min())} max={float(chw.max())}")
    # 洞区（mask>0）应该全部为 0
    # chw[0, :, :100, :100] 应该全 0
    hole_region = chw[0, :, :100, :100]
    _check("2d. 洞区全为 0",
           np.all(hole_region == 0.0), failures)
    # 非洞区应该为 128/255 ≈ 0.502
    non_hole_val = chw[0, 0, 200, 200]
    expected = 128.0 / 255.0
    _check("2e. 非洞区值 ≈ 128/255",
           abs(float(non_hole_val) - expected) < 0.01, failures,
           f"got {float(non_hole_val)} expected {expected}")

    # ================================================================ 3. 后处理 _out_chw_to_hwc
    print("[3] 后处理：_out_chw_to_hwc")
    # case A: output 在 0..1 范围（max <= 1.5）-> 乘 255
    out01 = np.zeros((3, 4, 4), dtype=np.float32)
    out01[0, :, :] = 0.5     # R = 0.5 -> 127.5 -> 128
    out01[1, :, :] = 1.0     # G = 1.0 -> 255
    out01[2, :, :] = 0.0     # B = 0.0 -> 0
    hwc_a = LamaInferenceService._out_chw_to_hwc(out01)
    _check("3a. 0..1 output -> HWC 形状 (4,4,3)",
           hwc_a.shape == (4, 4, 3), failures, f"got {hwc_a.shape}")
    _check("3b. 0..1 output -> 乘 255 后 R≈128",
           abs(int(hwc_a[0, 0, 0]) - 128) <= 1, failures,
           f"got {int(hwc_a[0, 0, 0])}")
    _check("3c. 0..1 output -> 乘 255 后 G=255",
           int(hwc_a[0, 0, 1]) == 255, failures)
    _check("3d. 0..1 output -> 乘 255 后 B=0",
           int(hwc_a[0, 0, 2]) == 0, failures)

    # case B: output 已在 0..255 范围（max > 1.5）-> 不乘 255
    out255 = np.zeros((3, 4, 4), dtype=np.float32)
    out255[0, :, :] = 128.0    # R = 128
    out255[1, :, :] = 255.0    # G = 255
    out255[2, :, :] = 0.0      # B = 0
    hwc_b = LamaInferenceService._out_chw_to_hwc(out255)
    _check("3e. 0..255 output -> 不乘 255，R=128",
           int(hwc_b[0, 0, 0]) == 128, failures,
           f"got {int(hwc_b[0, 0, 0])}")
    _check("3f. 0..255 output -> G=255",
           int(hwc_b[0, 0, 1]) == 255, failures)

    # clamp 测试：超出 255 的值被裁剪
    out_over = np.full((3, 4, 4), 300.0, dtype=np.float32)
    hwc_over = LamaInferenceService._out_chw_to_hwc(out_over)
    _check("3g. 超出 255 的值被 clamp 到 255",
           int(hwc_over[0, 0, 0]) == 255, failures)

    # ================================================================ 4. 合成 _composite_by_mask
    print("[4] 合成：_composite_by_mask")
    base = np.full((4, 4, 3), 10, dtype=np.uint8)
    fill = np.full((4, 4, 3), 200, dtype=np.uint8)
    mask_c = np.zeros((4, 4), dtype=np.uint8)
    mask_c[0, 0] = 255        # 只有一个洞像素
    mask_c[1, 1] = 255
    mask_c[2, 2] = 255

    out_c = LamaInferenceService._composite_by_mask(base, fill, mask_c)
    _check("4a. 洞区(0,0)用 fill=200",
           int(out_c[0, 0, 0]) == 200, failures)
    _check("4b. 非洞区(0,1)用 base=10",
           int(out_c[0, 1, 0]) == 10, failures)
    _check("4c. 洞区(1,1)用 fill=200",
           int(out_c[1, 1, 0]) == 200, failures)
    _check("4d. 非洞区(3,3)用 base=10",
           int(out_c[3, 3, 0]) == 10, failures)

    # ================================================================ 5. 完整 pipeline（mock session）
    print("[5] 完整 pipeline（mock session，恒等映射）")
    svc2 = LamaInferenceService()
    # 注入 mock session
    svc2._session = _MockSession()
    svc2._input_names = ["image", "mask"]
    svc2._output_names = ["output"]
    _check("5a. mock 注入后 is_loaded() == True",
           svc2.is_loaded(), failures)

    # 造一张 640x480 的图 + mask
    orig_img = np.full((480, 640, 3), 100, dtype=np.uint8)
    orig_img[100:200, 100:200] = 200      # 一块不同颜色
    orig_mask = np.zeros((480, 640), dtype=np.uint8)
    orig_mask[100:200, 100:200] = 255     # 洞区

    result = svc2.run(orig_img, orig_mask)
    _check("5b. run 返回非空数组",
           result.size > 0, failures)
    _check("5c. run 返回原尺寸 (480,640,3)",
           result.shape == (480, 640, 3), failures,
           f"got {result.shape}")
    _check("5d. run 返回 uint8",
           result.dtype == np.uint8, failures)
    # 恒等映射模型：输出 ≈ 输入（在非洞区）
    # 注意：由于缩放到 512x512 再还原，可能有插值误差，容差 20
    non_hole_pixel = result[0, 0]
    expected_val = 100      # 原图非洞区
    _check("5e. 非洞区像素接近原图（恒等映射模型）",
           abs(int(non_hole_pixel[0]) - expected_val) <= 20, failures,
           f"got {int(non_hole_pixel[0])} expected ≈{expected_val}")

    # ================================================================ 6. 空 / None 边界
    print("[6] 空 / None 边界")
    _check("6a. None image -> 空", svc2.run(None, None).size == 0, failures)
    _check("6b. None mask -> 空",
           svc2.run(orig_img, None).size == 0, failures)
    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    _check("6c. 空图 -> 空",
           svc2.run(empty_img, empty_img[..., 0]).size == 0, failures)

    # ---------------------------------------------------------------- 总结
    print("-" * 60)
    if failures:
        print(f"LamaInferenceService 测试失败：{len(failures)} 项 -> {failures}")
        sys.exit(1)
    else:
        print("LamaInferenceService 测试全部通过")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("测试抛出异常：")
        traceback.print_exc()
        sys.exit(1)
