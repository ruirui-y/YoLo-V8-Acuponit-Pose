"""LaMa ONNX Runtime 推理服务（Phase 5）。

忠实迁移 C++ LamaErasure/LamaOrt.cpp/.h 的推理流程：
1. 原图 + mask -> 缩放到 512x512（IgnoreAspectRatio）
2. mask 二值化(thresh=127) + 膨胀(radius=4)
3. image: 0..1 归一化 + 洞区置 0（CHW 布局）
4. mask: 0/1 浮点（HW 布局 -> 1x1xHxW）
5. ONNX Run(image=[1,3,512,512], mask=[1,1,512,512]) -> output=[1,3,512,512]
6. output CHW -> HWC，检测是否 0..1（max<=1.5 -> 乘 255）
7. 合成：洞区用 output，非洞区用 img512
8. 还原原尺寸（SmoothTransformation 等价）

输入输出约定（与 LamaController 对接）：
    run(rgb_image, binary_mask) -> inpainted_rgb
    - rgb_image:  HxWx3 uint8 RGB
    - binary_mask: HxW uint8 0/255（255=洞）
    - 返回:       HxWx3 uint8 RGB（原尺寸）

不依赖 QWidget / PySide6，纯 onnxruntime + numpy + cv2。
模型路径由外部注入（load_model），不写死绝对路径。

参考 C++ LamaOrt:
- processCore / RunOrt / ImageToFloatCHW_HoleZero01 / MaskToFloatHW
- BinarizeInPlace / DilateGray8 / OutputIs01 / CompositeByMask
"""
import numpy as np

try:
    import onnxruntime as ort
    _HAS_ORT = True
except ImportError:
    ort = None
    _HAS_ORT = False

import cv2


# 推理固定尺寸（与 C++ LamaOrt::kSize 一致）
_INFERENCE_SIZE = 512

# mask 膨胀半径（与 C++ dilateR = 4 一致）
_DILATE_RADIUS = 4

# 二值化阈值（与 C++ BinarizeInPlace thresh=127 一致）
_BINARIZE_THRESH = 127

# 输出 0..1 检测阈值（与 C++ OutputIs01 max<=1.5 一致）
_OUTPUT_01_THRESHOLD = 1.5


class LamaInferenceService:
    """LaMa ONNX 推理服务：RGB image + binary mask -> inpainted image。

    用法：
        svc = LamaInferenceService()
        svc.load_model("/path/to/lama.onnx")
        result = svc.run(rgb_image, binary_mask)
    """

    def __init__(self):
        self._session = None
        self._input_names = None
        self._output_names = None

    # ============================================================ 公开 API
    def is_loaded(self) -> bool:
        """模型是否已加载。"""
        return self._session is not None

    def load_model(self, model_path: str) -> bool:
        """加载 ONNX 模型。

        与 C++ LamaOrt 构造函数一致：
        - IntraOpNumThreads = 1
        - GraphOptimizationLevel = ORT_ENABLE_ALL
        - 输入名: image, mask
        - 输出名: output

        返回：True 成功 / False 失败（onnxruntime 未安装或模型加载失败）
        """
        if not _HAS_ORT:
            return False
        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            available = ort.get_available_providers()

            providers = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

            self._session = ort.InferenceSession(
                model_path,
                so,
                providers=providers
            )

            # 与 C++ 一致：输入 image/mask，输出 output
            self._input_names = [i.name for i in self._session.get_inputs()]
            self._output_names = [o.name for o in self._session.get_outputs()]
            return True
        except Exception:
            self._session = None
            self._input_names = None
            self._output_names = None
            return False

    def run(self, rgb_image: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
        """执行 LaMa 推理：RGB image + binary mask -> inpainted RGB image。

        与 C++ LamaOrt::processCore 一致：
        1. 缩放到 512x512
        2. mask 二值化 + 膨胀
        3. image 归一化 0..1 + 洞区置 0（CHW）
        4. ONNX 推理
        5. output CHW -> HWC + 0..1 检测
        6. 合成（洞区用 output，非洞区用 img512）
        7. 还原原尺寸

        参数：
            rgb_image:    HxWx3 uint8 RGB
            binary_mask:  HxW uint8 0/255（255=洞）

        返回：
            HxWx3 uint8 RGB（原尺寸），失败返回空数组
        """
        if not self.is_loaded():
            return np.zeros((0, 0, 3), dtype=np.uint8)
        if rgb_image is None or binary_mask is None:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        if rgb_image.size == 0 or binary_mask.size == 0:
            return np.zeros((0, 0, 3), dtype=np.uint8)

        orig_h, orig_w = rgb_image.shape[:2]
        # ---- 1. 缩放到 512x512 ----
        # image 用 smooth（INTER_LINEAR），mask 用 fast（INTER_NEAREST）
        img512 = cv2.resize(rgb_image, (_INFERENCE_SIZE, _INFERENCE_SIZE),
                            interpolation=cv2.INTER_LINEAR)
        m512 = cv2.resize(binary_mask, (_INFERENCE_SIZE, _INFERENCE_SIZE),
                          interpolation=cv2.INTER_NEAREST)

        # ---- 2. mask 二值化 + 膨胀 ----
        _, m512 = cv2.threshold(m512, _BINARIZE_THRESH, 255, cv2.THRESH_BINARY)
        if _DILATE_RADIUS > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (_DILATE_RADIUS * 2 + 1, _DILATE_RADIUS * 2 + 1),
            )
            m512 = cv2.dilate(m512, k, iterations=1)

        # ---- 3. 准备输入张量 ----
        # mask: 0/1 浮点 (1,1,H,W)
        mask_hw = (m512 > 0).astype(np.float32)            # (H,W) 0.0/1.0
        # image: 0..1 归一化 + 洞区置 0，CHW 布局 (1,3,H,W)
        img_chw = self._image_to_float_chw_hole_zero01(img512, mask_hw)

        mask_nchw = mask_hw[np.newaxis, np.newaxis, :, :]   # (1,1,H,W)

        # ---- 4. ONNX 推理 ----
        feed = {
            self._input_names[0]: img_chw,      # image
            self._input_names[1]: mask_nchw,    # mask
        }
        try:
            outputs = self._session.run(self._output_names, feed)
        except Exception:
            return np.zeros((0, 0, 3), dtype=np.uint8)

        out = outputs[0]          # (1,3,H,W) float32
        # ---- 5. output CHW -> HWC + 0..1 检测 ----
        out_hwc = self._out_chw_to_hwc(out[0])          # (H,W,3) uint8

        # ---- 6. 合成：洞区用 out，非洞区用 img512 ----
        final512 = self._composite_by_mask(img512, out_hwc, m512)

        # ---- 7. 还原原尺寸 ----
        result = cv2.resize(final512, (orig_w, orig_h),
                            interpolation=cv2.INTER_LINEAR)
        return result

    # ============================================================ 内部：预处理
    @staticmethod
    def _image_to_float_chw_hole_zero01(img512: np.ndarray,
                                         mask_hw: np.ndarray) -> np.ndarray:
        """image 转 0..1 + 洞区置 0 + CHW 布局（与 C++ ImageToFloatCHW_HoleZero01 一致）。

        返回 (1,3,H,W) float32
        """
        h, w = img512.shape[:2]
        # 0..1 归一化
        img_f = img512.astype(np.float32) / 255.0
        # 洞区置 0（mask>0 的位置）
        hole = mask_hw > 0.5
        img_f[hole] = 0.0
        # HWC -> CHW
        chw = np.transpose(img_f, (2, 0, 1))           # (3,H,W)
        return chw[np.newaxis, :, :, :]                 # (1,3,H,W)

    # ============================================================ 内部：后处理
    @staticmethod
    def _out_chw_to_hwc(out_chw: np.ndarray) -> np.ndarray:
        """output CHW -> HWC uint8（与 C++ OutCHWToRgb32 + OutputIs01 一致）。

        检测 output 是否在 0..1 范围（max<=1.5），若是则乘 255。
        """
        # out_chw: (3,H,W) float32
        # 检测是否 0..1
        mx = float(out_chw.max()) if out_chw.size > 0 else 0.0
        out01 = mx <= _OUTPUT_01_THRESHOLD

        out_f = out_chw.copy()
        if out01:
            out_f *= 255.0
        # clamp 到 [0, 255]
        out_f = np.clip(out_f, 0.0, 255.0)
        # CHW -> HWC
        out_hwc = np.transpose(out_f, (1, 2, 0))        # (H,W,3)
        return out_hwc.astype(np.uint8)

    @staticmethod
    def _composite_by_mask(base: np.ndarray, fill: np.ndarray,
                           mask_gray: np.ndarray) -> np.ndarray:
        """合成：洞区(255)用 fill，非洞区(0)用 base（与 C++ CompositeByMask 一致）。

        所有输入必须同尺寸。
        """
        # mask>0 的位置用 fill，否则用 base
        hole = mask_gray > 127
        out = base.copy()
        out[hole] = fill[hole]
        return out
