"""LaMa 工作流协调控制器。

负责整个 Open -> SetRef -> TestOne -> A -> 修正 -> Q -> Next 的顺序工作流。
维护：
- image_files / current_index / reference_index / busy
- 当前 reference 状态 / prediction 状态

连接 LamaPage signals，但算法调用必须交给 service。

依赖方向：
    LamaPage -> LamaController -> services / InpaintCanvas
    LamaController 连接 Page signals，Page 不调用 Controller。
    Service 不依赖 QWidget。

参考 C++ LamaErasure/MainWindow.cpp 的工作流槽函数（onSetAsReference/
onAssistMask/onQuickInpaint/onTestOneImage/showNextImage/showPrevImage），
但裁剪掉 Batch / AutoMask / Inpaint Preview / SaveAs 等历史功能。

Phase 1（已完成）：Open / Prev / Next / ClearMask + 占位信号槽
Phase 2（已完成）：MaskLabelService（Q 流程标签生成，本文件暂未接入）
Phase 3（已完成）：ReferenceAlignmentService（SetRef / predict / stable ID）
Phase 4（已完成）：A + TestOne 走同一 predict() API
Phase 5（已完成）：LamaInferenceService 接入 + TestOne 可选推理预览
Phase 6（本文件）：Q 原子 Commit 状态机（Image+Label+Rolling Ref+Next）

正确性修复（P0/P1）：
- P0-2: Q 不再有 y/x fallback；必须有 N 个 OK 预测，否则 BLOCK；failed track 不允许
        直接用未位移的 ref_center 作为正常预测（_get_predicted_centers 仅返回 ok=True）。
        N = 本次 session 的关键点数量（UI Keypoints，SetRef 时锁定）。
- P0-3: Q 真正原子提交：
        * 预先 prepare_reference 预验证 Reference State，inference 前就确保可 apply
        * 写 image.tmp.<ext> + label.tmp.txt（保留真实扩展名，cv2 才能正确编码）
        * 旧正式文件先备份 -> rename temp -> 正式；任一失败恢复备份不破坏旧文件
        * 文件全部成功后才 apply_prepared_reference（仅赋值不会失败）
- P1-1: TestOne 主预测必须与 A 同样使用 predict(mode="assist")。
- P1-2: TestOne 推理走后台 worker（_test_one_worker），不阻塞 GUI 主线程。
- P1-4: _InpaintWorker 完成后 deleteLater + 清空 self._inpaint_worker / _test_one_worker；
        成功/失败都清理 _commit_snapshot；窗口关闭时 cleanup_worker 显式 wait / terminate。
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from PySide6.QtCore import QObject, Qt, QPointF, QThread, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QColor, QFont, QPixmap
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QLabel, QPushButton, QVBoxLayout,
)

from .services import (
    MaskLabelService, ReferenceAlignmentService, PredictionResult,
    LamaInferenceService,
)
from .widgets.inpaint_canvas import _make_mask_overlay_rgba
from core.settings_store import SettingsStore

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .lama_page import LamaPage

# 支持的图片扩展名（与 C++ 一致）
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


# ============================================================ Q 原子 Commit 快照
@dataclass
class _CommitSnapshot:
    """Q 流程开始时的独立快照（与 C++ onQuickInpaint 的快照一致）。

    后面 show_next_image() 会换掉 Canvas，异步 callback 里绝不能再读 canvas / current_index。
    """
    current_index: int                          # 快照时的图片索引
    current_path: str                           # 快照时的图片绝对路径
    original_rgb: np.ndarray                    # 原图 RGB（numpy，已 copy）
    final_mask: np.ndarray                      # 最终人工确认 mask（numpy，已 copy）
    ordered_points: list[tuple[float, float]]   # stable ID -> (cx, cy) N 个点（canonical 顺序）
    bbox: tuple[int, int, int, int]             # 占位 BoundingBox (x, y, w, h)
    image_size: tuple[int, int]                 # (width, height)
    output_path: str                            # out/ 目录下的输出图片路径
    label_path: str                             # labels/ 目录下的标签路径
    # P0-3: 预验证过的 Reference State，文件写成功后 apply，避免半提交
    prepared_ref: object | None = None          # _PreparedReference（opaque）


# ============================================================ 后台推理 Worker
class _InpaintWorker(QThread):
    """在后台线程执行 LaMa 推理，避免 GUI 卡死。

    与 C++ LamaOrt::RunAsync + QMetaObject::invokeMode 一致：
    - 在独立线程跑 processCore
    - 通过 finished 信号把结果回到主线程

    生命周期（P1-4）：
    - parent 必须为 None，避免 page 销毁时 Qt 自动 destroy 仍 running 的 thread
    - 调用方在 finished slot 中 deleteLater + 清空引用
    - 窗口关闭时调用 cleanup_worker() 显式 wait / terminate
    """
    finished_with_result = Signal(object)     # np.ndarray（空数组表示失败）

    def __init__(self, service: LamaInferenceService, image: np.ndarray,
                 mask: np.ndarray, parent: QObject | None = None) -> None:
        # P1-4: parent 强制 None，避免 page 销毁时 "QThread: Destroyed while still running"
        super().__init__(None)
        self._service: LamaInferenceService = service
        self._image: np.ndarray = image
        self._mask: np.ndarray = mask

    def run(self) -> None:
        """在后台线程执行推理。"""
        try:
            result = self._service.run(self._image, self._mask)
        except Exception:
            result = np.zeros((0, 0, 3), dtype=np.uint8)
        self.finished_with_result.emit(result)


class _ModelLoadWorker(QThread):
    """在后台线程加载 LaMa ONNX 模型（GPU-only），避免 GUI 卡死。

    职责只有：service.load_model(model_path)。
    通过 finished_with_result 信号把结果回到主线程：
        (success: bool, model_path: str, error: str)
    """
    finished_with_result = Signal(bool, str, str)

    def __init__(self, service: LamaInferenceService, model_path: str,
                 parent: QObject | None = None) -> None:
        # parent 强制 None，避免 page 销毁时 "QThread: Destroyed while still running"
        super().__init__(None)
        self._service: LamaInferenceService = service
        self._model_path: str = model_path

    def run(self) -> None:
        """在后台线程加载模型。"""
        try:
            ok = self._service.load_model(self._model_path)
        except Exception as e:
            ok = False
            err = str(e)
        else:
            # load_model 返回 False 时真实错误在 service.last_error()
            err = self._service.last_error() if not ok else ""
        self.finished_with_result.emit(ok, self._model_path, err)


class LamaController(QObject):
    """LaMa 工作流协调器：连接 Page 信号 -> 调用 services / 更新 Page。"""

    def __init__(self, page: LamaPage, log_sink: Callable[[str], None] | None = None) -> None:
        super().__init__(page)
        self._page: LamaPage = page

        # ---- 应用级共享日志 sink（SharedLogPanel.append_log）----
        # 与 PoseController 用的是同一个 callable；状态栏仍走 page.set_status_text。
        self._log_sink: Callable[[str], None] = (
            log_sink if callable(log_sink) else (lambda _text: None)
        )

        # ---- 状态 ----
        self._image_files: list[str] = []     # 当前文件夹所有图片绝对路径
        self._current_index: int = -1         # 当前显示的图片索引
        self._reference_index: int = -1       # 当前 Reference 对应的图片索引（rolling reference）
        self._busy: bool = False              # Q 流程进行中（避免状态被破坏）

        # ---- 本次 session 的关键点数量 N ----
        # 0  = 尚未 SetRef，数量仍以 UI Keypoints 当前值为准（允许随时改）
        # >0 = SetRef 成功时锁定的 N，整个 rolling reference session 固定不变
        self._expected_keypoint_count: int = 0

        # ---- services ----
        self._mask_label_service: MaskLabelService = MaskLabelService()
        self._reference_alignment_service: ReferenceAlignmentService = ReferenceAlignmentService()
        self._lama_inference_service: LamaInferenceService = LamaInferenceService()

        # ---- SettingsStore：仅用于记住最近一次 Open/TestOne 的图片目录 ----
        # 模型路径是固定的（features/lama/models/lama_fp32.onnx），这里绝不恢复
        # 之前删除的 lama/model_path / YJJ_LAMA_MODEL / 手动 LoadModel。
        self._settings_store: SettingsStore = SettingsStore()

        # ---- 当前 prediction（A 键结果，供 Q 流程一对一匹配）----
        self._current_prediction: PredictionResult | None = None       # PredictionResult

        # ---- Q 原子 Commit 状态 ----
        self._commit_snapshot: _CommitSnapshot | None = None          # _CommitSnapshot（Q 流程进行中持有）
        self._inpaint_worker: _InpaintWorker | None = None           # _InpaintWorker（Q 后台推理线程）

        # ---- 启动期异步模型加载状态（GPU-only）----
        self._model_load_worker: _ModelLoadWorker | None = None        # _ModelLoadWorker（后台加载 ONNX）
        self._model_load_failed: bool = False       # 模型后台加载最终失败标记

        # ---- TestOne 异步推理状态（P1-2）----
        # _test_one_pending 持有 (base_qimg, prediction_result) 供 callback 使用
        self._test_one_worker: _InpaintWorker | None = None          # _InpaintWorker（TestOne 后台推理）
        self._test_one_pending: tuple[QImage, PredictionResult] | None = None         # (base_qimg, result) or None

        # ---- 信号连接 ----
        self._connect_signals()
        self._refresh_status()

        # ---- 启动时异步加载 LaMa 模型（固定 GPU-only 路径，不阻塞 GUI 主线程）----
        self._start_model_load()

    # ================================================================ 共享日志
    def _log(self, text: str) -> None:
        """写应用级共享日志（带 [LaMa] 前缀）。

        只在关键事件写：模型加载 / ORT providers / Open / SetRef / TestOne /
        Assist / Q 全流程 / 输出路径 / rolling reference 更新。
        refresh_status 之类的高频状态刷新绝不写日志。
        """
        self._log_sink(f"[LaMa] {text}")

    # ================================================================ 关键点数量 N（session 生命周期）
    def ExpectedKeypointCount(self) -> int:
        """当前 session 生效的关键点数量 N（对外只读）。"""
        return self._current_expected_count()

    def ClearReference(self) -> None:
        """清除 Reference 并解锁关键点数量（Reset / Clear Reference 流程入口）。

        清除后用户可以重新选择 Keypoints，再重新 SetRef 建立新的 session。
        """
        self._reset_reference_session()
        self._refresh_status()
        self._log("Reference 已清除，关键点数量已解锁")

    def _current_expected_count(self) -> int:
        """当前生效的关键点数量 N。

        SetRef 成功后锁定为 _expected_keypoint_count；
        尚未 SetRef 时直接取 UI Keypoints 的当前值（允许随时修改）。
        """
        if self._expected_keypoint_count > 0:
            return self._expected_keypoint_count
        return self._page.KeypointCount()

    def _reset_reference_session(self) -> None:
        """重置整个 rolling reference session 并解锁关键点数量（唯一入口）。

        Open 新序列 / ClearReference 都走这里，保证
        “清 reference + 解锁 N” 只有一处实现，避免两处状态不一致。
        """
        self._reference_index = -1
        self._current_prediction = None
        self._reference_alignment_service = ReferenceAlignmentService()
        self._expected_keypoint_count = 0
        self._page.SetKeypointCountEditable(True)

    # ================================================================ Worker 生命周期（P1-4）
    def cleanup_worker(self) -> None:
        """窗口关闭 / Page 销毁前调用：等待或终止仍运行的后台 worker。

        防止 "QThread: Destroyed while thread is still running" 警告。
        成功/失败都清理 self._inpaint_worker / self._commit_snapshot。
        """
        # Q worker
        if self._inpaint_worker is not None:
            w = self._inpaint_worker
            self._inpaint_worker = None
            self._cleanup_running_worker(w)
        self._commit_snapshot = None
        # Model load worker（窗口关闭时若仍在加载，等待/终止，避免 QThread destroyed while running）
        if self._model_load_worker is not None:
            w = self._model_load_worker
            self._model_load_worker = None
            self._cleanup_running_worker(w)
        # TestOne worker
        if self._test_one_worker is not None:
            w = self._test_one_worker
            self._test_one_worker = None
            self._test_one_pending = None
            self._cleanup_running_worker(w)

    @staticmethod
    def _cleanup_running_worker(w: _InpaintWorker) -> None:
        """等待 / 终止单个 worker 并 deleteLater。"""
        if w is None:
            return
        if w.isRunning():
            # 给最多 5 秒自然完成（LaMa 推理一般几秒）
            w.wait(5000)
            if w.isRunning():
                # 仍未完成 -> 强制 terminate（不推荐但避免警告）
                w.terminate()
                w.wait(1000)
        try:
            w.deleteLater()
        except Exception:
            pass

    # ================================================================ 信号连接
    def _connect_signals(self) -> None:
        p = self._page
        p.openRequested.connect(self._on_open)
        p.clearMaskRequested.connect(self._on_clear_mask)
        p.setRefRequested.connect(self._on_set_ref)
        p.testOneRequested.connect(self._on_test_one)
        p.assistMaskRequested.connect(self._on_assist_mask)
        p.commitRequested.connect(self._on_commit)
        p.helpRequested.connect(self._on_help)
        p.prevRequested.connect(self._on_prev)
        p.nextRequested.connect(self._on_next)
        p.brushRadiusChanged.connect(self._on_brush_changed)

    # ================================================================ 最近目录记忆（lama/last_image_dir）
    def _read_last_image_dir(self) -> str:
        """从 SettingsStore 读取上次 Open/TestOne 选择的图片目录。

        仅当保存的路径仍然存在时返回它；未保存或目录已不存在返回 ""。
        """
        saved = self._settings_store.get("lama/last_image_dir", "")
        if saved and isinstance(saved, str) and os.path.isdir(saved):
            return saved
        return ""

    def _save_last_image_dir(self, image_path: str) -> None:
        """把所选图片的父目录持久化到 SettingsStore（key: lama/last_image_dir）。

        用户取消 QFileDialog 时不要调用本方法。必须 sync() 才落盘，
        确保关闭程序重新启动后仍能读取。
        """
        parent = str(Path(image_path).resolve().parent)
        try:
            self._settings_store.set("lama/last_image_dir", parent)
            self._settings_store.sync()
        except Exception:
            # 持久化失败不影响本次打开，下次回到默认行为即可
            pass

    # ================================================================ Phase 1: Open / Prev / Next / ClearMask
    def _on_open(self) -> None:
        """打开单张图片：自动加载同目录所有图片并按名称排序。"""
        if self._busy:
            self._page.set_status_text("Busy，请等待 Q 完成")
            return
        # 默认目录：若已打开图片序列，优先当前图片所在目录；
        # 否则读取上次保存的 lama/last_image_dir（存在才用）。
        if self._current_index >= 0 and self._image_files:
            default_dir = str(Path(self._image_files[self._current_index]).parent)
        else:
            default_dir = self._read_last_image_dir()

        f, _ = QFileDialog.getOpenFileName(
            self._page, "Open Image", default_dir,
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not f:
            # 用户取消：不要修改 last_image_dir
            return

        # 记住所选图片的父目录，下次启动直接从此目录打开
        self._save_last_image_dir(f)

        # 收集同目录所有图片并按名称排序（与 C++ QDir::Name 一致）
        path = Path(f)
        files = [p for p in path.parent.iterdir()
                 if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
        files.sort(key=lambda x: x.name)
        if not files:
            return

        # ---- 重置上一轮 LaMa 标注 session（新序列绝不能继承旧 rolling reference）----
        # 只重建 ReferenceAlignmentService，不动 MaskLabelService / LamaInferenceService，
        # 避免自动加载好的 ONNX 模型被丢掉；同时解锁关键点数量选择。
        self._reset_reference_session()

        self._image_files = [str(p) for p in files]
        try:
            self._current_index = self._image_files.index(str(path))
        except ValueError:
            self._current_index = 0

        self._load_current_image()
        self._refresh_status()
        self._log(
            f"Open: {path.parent} | {len(self._image_files)} 张图片 | "
            f"当前 {self._current_index + 1}/{len(self._image_files)} "
            f"({Path(self._image_files[self._current_index]).name}) | rolling reference 已重置"
        )

    def _on_clear_mask(self) -> None:
        """清空当前 mask。"""
        if self._busy:
            return
        self._page.canvas.clear_mask()
        # 清空 mask 同时清掉当前 prediction（草稿已弃用）
        self._current_prediction = None
        self._refresh_status()

    def _on_prev(self) -> None:
        """切到上一张（不再循环）。"""
        if self._busy:
            return
        if not self._image_files or self._current_index < 0:
            return
        if self._current_index == 0:
            self._page.set_status_text("Already at first image")
            self._refresh_status()
            return
        self._current_index -= 1
        self._load_current_image()
        self._refresh_status()

    def _on_next(self) -> None:
        """切到下一张（不再循环，最后一张 Q 后停留此处）。"""
        if self._busy:
            return
        if not self._image_files or self._current_index < 0:
            return
        n = len(self._image_files)
        if self._current_index == n - 1:
            self._page.set_status_text("All images completed / 已到最后一张")
            self._refresh_status()
            return
        self._current_index += 1
        self._load_current_image()
        self._refresh_status()

    def _on_brush_changed(self, radius: int) -> None:
        """画笔半径变化，更新状态。"""
        self._page.set_status_text(f"Brush={radius}")

    # ---- Phase 1 内部辅助 ----
    def _load_current_image(self) -> None:
        """加载 self._current_index 指向的图片到 canvas。"""
        if not self._image_files or self._current_index < 0:
            return
        f = self._image_files[self._current_index]
        if self._page.canvas.load_image(f):
            # 切图后清空当前 prediction（旧 prediction 已失效）
            self._current_prediction = None
            self._page.set_status_text(
                f"Image {self._current_index + 1}/{len(self._image_files)}"
            )

    # ================================================================ 模型加载（GPU-only 固定路径，异步）
    def _start_model_load(self) -> None:
        """启动后台模型加载（GPU-only，固定模型路径，异步非阻塞 GUI）。

        唯一模型路径：<本文件所在目录>/models/lama_fp32.onnx
        （__file__ 即 lama_controller.py -> features/lama/models/lama_fp32.onnx）

        不在源码写 Windows 绝对路径（固定为 features/lama/models/lama_fp32.onnx）。
        """
        model_path = str(
            Path(__file__).resolve().parent / "models" / "lama_fp32.onnx"
        )
        self._model_load_failed = False
        self._log("正在后台加载模型...")
        self._model_load_worker = _ModelLoadWorker(
            self._lama_inference_service, model_path, parent=None
        )
        self._model_load_worker.finished_with_result.connect(self._on_model_loaded)
        self._model_load_worker.start()

    def _on_model_loaded(self, success: bool, model_path: str, error: str) -> None:
        """模型后台加载完成回调（主线程，由 worker 信号触发）。

        无论成功失败都清理 worker，并在 GUI 日志显示真实结果。
        """
        worker = self._model_load_worker
        self._model_load_worker = None
        if worker is not None:
            worker.deleteLater()

        if success:
            self._model_load_failed = False
            self._page.set_status_text(f"Model loaded: {Path(model_path).name}")
            self._log(f"模型加载成功: {Path(model_path).name}")
            self._log_ort_providers()
        else:
            self._model_load_failed = True
            self._page.set_status_text("LaMa CUDA 模型加载失败，请查看运行日志。")
            self._log(f"模型加载失败: {error if error else '(未知错误)'}")
            self._log_ort_providers()

    def _check_lama_model(self) -> tuple[str, str]:
        """检查 LaMa 模型当前状态，供 Q / TestOne 在真正需要推理前调用。

        返回 (state, message):
            ("ready",   "")           模型已加载，可推理
            ("loading", "...")        后台仍在加载，需等待
            ("failed",  "...")        加载失败（CUDA/DLL/模型错误）
        """
        if self._lama_inference_service.is_loaded():
            return ("ready", "")
        if self._model_load_worker is not None:
            return ("loading", "LaMa 模型正在后台加载，请稍候。")
        if self._model_load_failed:
            return ("failed", "LaMa CUDA 模型加载失败，请查看运行日志。")
        # 理论不应发生（启动时即进入 loading）：兜底视为加载中
        return ("loading", "LaMa 模型正在后台加载，请稍候。")

    def _log_ort_providers(self) -> None:
        """记录 ORT available / active providers（只读诊断信息）。"""
        available = self._lama_inference_service.available_providers()
        active = self._lama_inference_service.active_providers()
        self._log(f"ORT available providers: {available if available else '(none)'}")
        self._log(f"ORT active providers: {active if active else '(none)'}")

    # ================================================================ Phase 4: SetRef / A / TestOne
    def _on_set_ref(self) -> None:
        """SetRef：将当前图 + 最终 mask 设为基准（rolling reference 入口 1/2）。

        与 C++ MainWindow::onSetAsReference 一致：
        1. 检查 canvas 有图 + mask 非空
        2. 从 UI 读取本次 session 的关键点数量 N，finalMask 必须恰好 N 个 component
        3. 从 finalMask 提取 centers，按 (y,x) 升序建立 canonical ID 0..N-1
        4. 调 ReferenceAlignmentService.set_reference(..., expected_count=N)
        5. 成功后锁定 N（UI SpinBox disable）+ 更新 reference_index / 清空 current_prediction
        """
        if self._busy:
            self._page.set_status_text("Busy，请等待 Q 完成")
            return
        canvas = self._page.canvas
        src = canvas.source_image()
        if src.isNull():
            self._page.set_status_text("No image on canvas")
            return
        mask = canvas.mask_image()
        if mask.isNull():
            n_hint = self._current_expected_count()
            self._page.set_status_text(
                f"Mask is empty, please draw {n_hint} masks first"
            )
            return

        # ---- 转 numpy ----
        ref_rgb = self._qimage_to_rgb_np(src)
        ref_mask = self._mask_qimage_to_np(mask)

        # ---- 关键点数量在 SetRef 这一刻从 UI 读取并锁定 ----
        n = self._page.KeypointCount()

        # ---- 从 finalMask 提取 centers（与 C++ ExtractMaskCenters 一致）----
        centers = ReferenceAlignmentService.extract_mask_centers(
            ref_mask, min_component_area=50
        )
        if len(centers) != n:
            self._page.set_status_text(
                f"Expected {n} components, got {len(centers)}"
            )
            self._log(
                f"SetRef 失败：期望 {n} 个 mask component，"
                f"实际 {len(centers)} 个"
            )
            return

        # ---- 按 (y, x) 升序建立 canonical ID 0..N-1（stable ID 关键）----
        centers.sort(key=lambda c: (c[1], c[0]))

        # ---- 调 service（component 数与 canonical 点数都必须 == N）----
        ok = self._reference_alignment_service.set_reference(
            ref_rgb, ref_mask, ordered_points=centers, ref_rect=None,
            expected_count=n,
        )
        if not ok:
            self._page.set_status_text("Reference set FAILED")
            self._log("SetRef 失败：ReferenceAlignmentService.set_reference 返回 False")
            return

        # ---- SetRef 成功：锁定 N，之后整个 rolling session 固定不变 ----
        self._expected_keypoint_count = n
        self._page.SetKeypointCountEditable(False)
        self._reference_index = self._current_index
        self._current_prediction = None     # 清空旧 prediction
        self._refresh_status()
        self._page.set_status_text("Reference set OK")
        cur_name = (Path(self._image_files[self._current_index]).name
                    if 0 <= self._current_index < len(self._image_files) else "-")
        self._log(
            f"SetRef 成功：index={self._current_index} ({cur_name})，"
            f"canonical points={len(centers)}，Keypoints 已锁定为 {n}"
        )

    def _on_assist_mask(self) -> None:
        """A 键：基于基准预测 mask 草稿（mode="assist"）。

        与 C++ MainWindow::onAssistMask 一致：
        1. 前提：Reference Ready + 当前图不是 Reference 本身 + busy == false
        2. 调 predict(target_rgb, mode="assist")
        3. 把 prediction.mask 写回 canvas
        4. 保存 current_prediction（供 Q 流程 ID -> predictedCenter 匹配）
        5. 显示 Prediction: x/N（N = 本次 session 关键点数量）
        """
        if self._busy:
            return
        if not self._reference_alignment_service.is_ready():
            self._page.set_status_text("Reference not set. Click SetRef first")
            return
        canvas = self._page.canvas
        src = canvas.source_image()
        if src.isNull():
            self._page.set_status_text("No image on canvas")
            return
        if self._current_index == self._reference_index:
            self._page.set_status_text("Current is reference itself, no need to predict")
            return

        # ---- 调用统一 predict API（A 与 TestOne 共用同一方法）----
        target_rgb = self._qimage_to_rgb_np(src)
        result = self._reference_alignment_service.predict(target_rgb, mode="assist")

        # ---- prediction track 数必须等于本次 session 的关键点数量 N ----
        expected_n = self._current_expected_count()
        if result.total != expected_n:
            self._current_prediction = None
            self._page.set_prediction_info(0, expected_n)
            self._page.set_status_text(
                f"AssistMask BLOCK: prediction tracks {result.total} != {expected_n}"
            )
            self._log(
                f"Assist BLOCK: prediction tracks={result.total}，期望 {expected_n}"
            )
            return

        self._current_prediction = result

        # ---- 显示预测结果到 canvas ----
        if result.mask is None:
            self._page.set_prediction_info(0, result.total)
            self._page.set_status_text(
                f"AssistMask: predicted 0/{result.total}, please draw manually"
            )
            self._log(f"Assist: 0/{result.total}（无可用 mask，需人工绘制）")
            return

        mask_qimg = self._np_mask_to_qimage(result.mask)
        canvas.set_mask(mask_qimg)
        self._page.set_prediction_info(result.success, result.total)
        self._log(f"Assist: {result.success}/{result.total}")
        if result.success < result.total:
            self._page.set_status_text(
                f"AssistMask: predicted {result.success}/{result.total}, please fix manually"
            )
        else:
            self._page.set_status_text(
                f"AssistMask: predicted {result.success}/{result.total}"
            )

    def _on_test_one(self) -> None:
        """TestOne：测试基准追踪效果（不保存、不修改 reference、不切图）。

        P1-1 修订：主预测必须与 A 同样使用 predict(mode="assist")，因为 TestOne
        的目的就是验证实际 A workflow（A 模式画固定半径圆，允许部分失败）。
        历史的 strict 模式只在调试需要时才用，不再作为主结果。

        P1-2 修订：推理不得阻塞 GUI 主线程。若 LaMa 模型已加载，启动后台 worker，
        完成后回调 _on_test_one_inpaint_done 显示预览窗。

        与 C++ MainWindow::onTestOneImage 一致：
        1. 前提：Reference Ready
        2. 选择测试图片
        3. 调 predict(target_rgb, mode="assist")  <- 与 A 同一 API
        4. 失败：显示失败原因
        5. 成功：显示 overlay 预览 + track 调试框
        6. 若 LaMa 模型已加载，启动后台 worker 推理（不阻塞 GUI）
        """
        if self._busy:
            return
        if not self._reference_alignment_service.is_ready():
            self._page.set_status_text("Reference not set. Click SetRef first")
            return
        # 上一轮 TestOne worker 仍在运行就不允许新一轮
        if self._test_one_worker is not None:
            self._page.set_status_text("TestOne worker still running, please wait")
            return

        # ---- 选择测试图片（起始目录同样使用 lama/last_image_dir，避免每次从工作目录开始）----
        f, _ = QFileDialog.getOpenFileName(
            self._page, "Select one image (B)", self._read_last_image_dir(),
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not f:
            # 用户取消：不要修改 last_image_dir
            return
        # 记住所选图片的父目录，Open 与 TestOne 共用同一个记忆
        self._save_last_image_dir(f)
        img_b = QImage(f)
        if img_b.isNull():
            self._page.set_status_text("Failed to load image")
            self._log(f"TestOne 失败：图片无法加载 {f}")
            return

        self._log(f"TestOne 开始: {f}")

        # ---- P1-1: 调用统一 predict API，mode 与 A 一致 ----
        target_rgb = self._qimage_to_rgb_np(img_b)
        result = self._reference_alignment_service.predict(target_rgb, mode="assist")

        if result.mask is None:
            # ---- 失败：统计 + 第一个失败 track 的原因 ----
            fail_msgs = []
            for t in result.tracks:
                if not t.ok:
                    fail_msgs.append(f"track {t.id}: {t.fail_reason} (score={t.score:.2f})")
            head = "; ".join(fail_msgs[:3]) if fail_msgs else "no track succeeded"
            self._page.set_status_text(
                f"Local match failed: {result.success}/{result.total} | {head}"
            )
            self._log(f"TestOne 失败: {result.success}/{result.total} | {head}")
            return

        self._page.set_status_text(
            f"Local match OK: {result.success}/{result.total}"
        )
        self._log(f"TestOne 匹配成功: {result.success}/{result.total}")

        # ---- 预先生成 base preview（mask overlay + track 调试框）----
        base_qimg = self._build_test_one_preview(img_b, result)

        # ---- 仅当模型真正就绪时后台推理，避免阻塞 GUI（P1-2）----
        # 模型仍在加载 / 加载失败：只显示 tracking + mask preview，不跑 LaMa 擦除预览
        model_state, model_msg = self._check_lama_model()
        if model_state == "ready":
            self._page.set_status_text("TestOne: 推理中（后台）...")
            self._test_one_pending = (base_qimg, result)
            self._test_one_worker = _InpaintWorker(
                self._lama_inference_service, target_rgb, result.mask, parent=None
            )
            self._test_one_worker.finished_with_result.connect(
                self._on_test_one_inpaint_done
            )
            self._test_one_worker.start()
            return

        if model_state == "loading":
            self._log(f"TestOne: {model_msg}")
        else:
            self._log(
                "TestOne: LaMa CUDA 模型加载失败，跳过擦除预览。详情："
                + self._lama_inference_service.last_error()
            )
        # ---- 模型未就绪：直接显示 base preview ----
        self._show_test_one_preview_dialog(base_qimg, result, None)

    def _on_test_one_inpaint_done(self, result_arr: np.ndarray) -> None:
        """TestOne 后台推理完成回调（P1-2）。

        在主线程执行，由 worker 信号触发。从 _test_one_pending 取出 base_qimg + prediction，
        显示对比预览窗，最后清理 worker。
        """
        pending = self._test_one_pending
        # P1-4: 无论成功失败都清理 worker 与 pending
        worker = self._test_one_worker
        self._test_one_worker = None
        self._test_one_pending = None
        if worker is not None:
            worker.deleteLater()

        if pending is None:
            self._page.set_status_text("TestOne: pending state lost")
            return

        base_qimg, result = pending
        inpainted = result_arr if (result_arr is not None and result_arr.size > 0) else None

        if inpainted is not None:
            self._page.set_status_text(
                f"Local match OK: {result.success}/{result.total} | Inpaint done"
            )
            self._log(f"TestOne 推理完成: {result.success}/{result.total}")
        else:
            self._page.set_status_text(
                f"Local match OK: {result.success}/{result.total} | Inpaint failed"
            )
            self._log("TestOne 推理失败（inpaint 返回空结果）")

        self._show_test_one_preview_dialog(base_qimg, result, inpainted)

    def _build_test_one_preview(self, base_img: QImage,
                                 result: PredictionResult) -> QImage:
        """生成 TestOne 预览 base 图：原图 + mask overlay + track 调试框 + ID。"""
        preview = base_img.convertToFormat(QImage.Format.Format_RGB888).copy()
        p = QPainter(preview)
        # ---- mask overlay 半透明红（与 C++ MakeOverlayPreview 一致）----
        if result.mask is not None:
            mask_q = self._np_mask_to_qimage(result.mask)
            overlay = _make_mask_overlay_rgba(mask_q, alpha=120)
            p.drawImage(0, 0, overlay)
        # ---- track 调试框：绿色圆 + ID（与 C++ 绿框 + 编号一致）----
        p.setPen(QPen(QColor(0, 255, 0), 2))
        p.setFont(QFont("monospace", 11))
        for t in result.tracks:
            if not t.ok:
                continue
            cx, cy = t.current_center
            p.drawEllipse(QPointF(cx, cy), 16, 16)
            p.drawText(int(cx) - 6, int(cy) - 20, str(t.id))
        p.end()
        return preview

    def _show_test_one_preview_dialog(self, base_qimg: QImage,
                                       result: PredictionResult,
                                       inpainted: np.ndarray | None = None) -> None:
        """弹出 TestOne 预览窗：base + （可选）擦除结果对比。"""
        dlg = QDialog(self._page)
        dlg.setWindowTitle(f"TestOne: {result.success}/{result.total} ok")
        dlg.resize(1200, 600)
        lay = QVBoxLayout(dlg)
        # 预览图
        lbl = QLabel()
        lbl.setPixmap(QPixmap.fromImage(base_qimg))
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lbl)
        # 擦除结果（如果有）
        if inpainted is not None and inpainted.size > 0:
            inpainted_q = self._np_rgb_to_qimage(inpainted)
            lbl2 = QLabel()
            lbl2.setPixmap(QPixmap.fromImage(inpainted_q))
            lbl2.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(lbl2)
        btn = QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        lay.addWidget(btn)
        dlg.exec()

    # ================================================================ Phase 6: Q 原子 Commit 状态机
    def _on_commit(self) -> None:
        """Q 键：原子 Commit 状态机（与 C++ MainWindow::onQuickInpaint 一致）。

        真正原子提交流程（P0-3）：
        1. 安全检查（busy / canvas / current_index / mask 非空）
        2. 快照 original_rgb + final_mask + current_path + image_size
        3. 检查 current_prediction：必须 N 个 OK 预测，否则 BLOCK（P0-2）
           禁止用 finalMask centers 按 y/x 重新建 ID；
           禁止 failed track 用未位移的 ref_center 当作正常预测。
        4. MaskLabelService 分析 final_mask（N component + 全局 assignment + bbox）
           失败 -> BLOCK，不写文件
        5. 预验证 Reference State（prepare_reference），失败 BLOCK
        6. 生成输出路径 out/ + labels/
        7. 检查 LaMa 模型已加载
        8. busy = true，禁用所有按钮
        9. 后台 worker 执行 LaMa 推理（避免 GUI freeze）
        10. callback _on_inpaint_done：
            - 写 image.tmp.<ext> + label.tmp.txt
            - 都成功 -> 备份旧正式文件 -> rename 两个 temp 到正式路径
              任一失败 -> 删除 temp + 恢复旧正式文件，busy=false，留当前图
            - apply 预验证过的 Reference State（不会失败）
            - busy = false，清空 worker
            - show_next_image()（Q 成功以前绝不切下一张）
        """
        # ---- 1. 安全检查 ----
        if self._busy:
            self._page.set_status_text("Busy，请等待 Q 完成")
            return
        canvas = self._page.canvas
        src = canvas.source_image()
        if src.isNull():
            self._page.set_status_text("No image on canvas")
            return
        if self._current_index < 0 or self._current_index >= len(self._image_files):
            self._page.set_status_text("当前没有打开的文件路径，请先 Open")
            self._log("Q BLOCK: 当前没有打开的文件路径，请先 Open")
            return
        mask = canvas.mask_image()
        if mask.isNull():
            n_hint = self._current_expected_count()
            self._page.set_status_text(
                f"Mask is empty, please draw {n_hint} masks first"
            )
            self._log(f"Q BLOCK: mask 为空（本次 session 需要 {n_hint} 个 component）")
            return

        # ---- 2. 快照（与 C++ 一样，异步 callback 里绝不能再读 canvas/current_index）----
        current_path = self._image_files[self._current_index]
        self._log(f"Q start: index={self._current_index} | {current_path}")
        original_rgb = self._qimage_to_rgb_np(src).copy()
        final_mask = self._mask_qimage_to_np(mask).copy()
        image_size = (src.width(), src.height())
        snapshot_index = self._current_index

        # ---- 3. 确定 predicted_centers：当前图是 Reference 还是后续图 ----
        # Case 1: 当前图 == 当前 Reference 且 ref 已经建立
        #         SetRef 之后用户可以直接 Q，不需要 A。
        #         此时 stable ID 来自 Reference canonical points，
        #         MaskLabelService 会让 finalMask 的 N 个 component
        #         与 canonical reference points 做一对一匹配，
        #         即使用户在 SetRef 后又轻微修了 Mask 也能正确 assignment。
        # Case 2: 当前图 != Reference
        #         必须有 A 的稳定预测（N 个 OK stable ID），否则 BLOCK。
        #         禁止对后续图退回到 finalMask centers 按 y/x 重排建 ID，
        #         因为那会破坏跨图的 stable ID 一致性。
        is_current_ref = (
            self._current_index == self._reference_index
            and self._reference_alignment_service.is_ready()
        )
        # ---- 本次 session 的关键点数量 N（SetRef 后已锁定）----
        n = self._current_expected_count()
        if is_current_ref:
            # ---- Case 1：Reference 自己直接 Q ----
            ref_points = self._reference_alignment_service.reference_points()
            if len(ref_points) != n:
                self._page.set_status_text(
                    f"Q BLOCK: Reference canonical points 不足 {n} 个"
                )
                self._log(
                    f"Q BLOCK: Reference canonical points={len(ref_points)}，需 {n} 个"
                )
                return
            predicted_centers = {i: ref_points[i] for i in range(n)}
        else:
            # ---- Case 2：后续图必须来自 A prediction ----
            predicted_centers = self._get_predicted_centers()
            if len(predicted_centers) != n:
                self._page.set_status_text(
                    "当前图片尚未执行 Assist Mask，请先按 A。"
                )
                self._log(
                    f"Q BLOCK: 有效预测 {len(predicted_centers)}/{n}，请先按 A"
                )
                return
            # rolling reference 必须基于已建立且合法的 Reference 几何
            ref_points = self._reference_alignment_service.reference_points()
            if len(ref_points) != n:
                self._page.set_status_text(
                    f"Q BLOCK: 缺少合法 Reference（{n} 点），请先 SetRef"
                )
                self._log(
                    f"Q BLOCK: rolling reference 缺 Reference canonical points="
                    f"{len(ref_points)}，需 {n} 个"
                )
                return

        # ---- 4. MaskLabelService 分析 final_mask（component 数 != N 或 assignment 失败 -> BLOCK）----
        # Option 2 防御：把当前合法 Reference 的 ordered reference_points 传入，
        # 由 assign_stable_ids 做“Reference N 点整体几何一致性”定 ID，
        # 防止 A/tracking 中任意两个 stable ID 交换污染最终 label。
        label_result = self._mask_label_service.make_label(
            final_mask, predicted_centers, image_size,
            expected_count=n,
            reference_points=ref_points,
        )
        if label_result is None:
            self._page.set_status_text(
                f"Q BLOCK: mask label validation FAILED ({n} components / stable ID / distance)"
            )
            self._log(f"Q BLOCK: mask label 校验失败（{n} component / stable ID / 距离）")
            return

        # ---- 5. P0-3: 预验证 Reference State（不修改当前状态）----
        # 文件写成功后才 apply，避免半提交（image/label 已写但 ref 更新失败）
        prepared_ref = self._reference_alignment_service.prepare_reference(
            original_rgb, final_mask,
            ordered_points=list(label_result.ordered_points),
            ref_rect=None,
            expected_count=n,
        )
        if prepared_ref is None:
            self._page.set_status_text(
                f"Q BLOCK: rolling reference 预验证失败（component 数 != {n} 或一对一匹配失败）"
            )
            self._log(
                f"Q BLOCK: rolling reference 预验证失败（component != {n} 或匹配失败）"
            )
            return

        # ---- 6. 生成输出路径（与 C++ 一致：sourceDir/out/ + sourceDir/labels/）----
        path_obj = Path(current_path)
        source_dir = path_obj.parent
        out_dir = source_dir / "out"
        labels_dir = source_dir / "labels"
        out_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / path_obj.name)
        label_path = str(labels_dir / (path_obj.stem + ".txt"))

        # ---- 7. 检查模型状态（GPU-only 异步加载）----
        model_state, model_msg = self._check_lama_model()
        if model_state == "loading":
            self._page.set_status_text(model_msg)
            self._log("Q BLOCK: " + model_msg)
            return
        if model_state == "failed":
            self._page.set_status_text(model_msg)
            self._log(
                "Q BLOCK: " + model_msg
                + " 详情：" + self._lama_inference_service.last_error()
            )
            return

        # ---- 保存快照（供 callback 使用，含预验证过的 prepared_ref）----
        self._commit_snapshot = _CommitSnapshot(
            current_index=snapshot_index,
            current_path=current_path,
            original_rgb=original_rgb,
            final_mask=final_mask,
            ordered_points=list(label_result.ordered_points),
            bbox=label_result.bbox,
            image_size=image_size,
            output_path=output_path,
            label_path=label_path,
            prepared_ref=prepared_ref,
        )

        # ---- 8. busy = true，禁用所有按钮 ----
        self._busy = True
        self._page.set_busy(True)
        self._page.set_status_text("Q: 正在推理...")

        # ---- 9. 后台 worker 执行 LaMa 推理 ----
        # P1-4: parent 强制 None；callback 内完成后 deleteLater + 清空 self._inpaint_worker
        self._inpaint_worker = _InpaintWorker(
            self._lama_inference_service, original_rgb, final_mask, parent=None
        )
        self._inpaint_worker.finished_with_result.connect(self._on_inpaint_done)
        self._inpaint_worker.start()

    def _on_inpaint_done(self, result: np.ndarray) -> None:
        """LaMa 推理完成回调（在主线程执行，由 worker 信号触发）。

        真正原子提交（P0-3）：
        1. 推理失败 -> busy=false，清理 worker，不保存，不更新 ref，不切图
        2. 写 image.tmp.<ext> + label.tmp.txt（保持真实扩展名，cv2 才能正确编码）
        3. 都成功 -> 备份旧正式文件（image.bak.<ext> + label.bak.txt）
                 -> os.replace temp -> 正式路径
                 -> 任一失败 -> 删除 temp + 恢复备份，busy=false，留当前图
        4. apply 预验证过的 Reference State（apply_prepared_reference 仅赋值不会失败）
        5. busy = false，deleteLater worker + 清空 self._inpaint_worker + _commit_snapshot
        6. show_next_image()（Q 成功以前绝不切下一张）
        """
        snap = self._commit_snapshot
        # P1-4: 无论成功失败都清理 worker 与 snapshot
        worker = self._inpaint_worker
        self._inpaint_worker = None
        if worker is not None:
            worker.deleteLater()

        def _abort(msg: str):
            """统一失败路径：恢复 busy=false + 清理 snapshot，不切图。"""
            self._busy = False
            self._page.set_busy(False)
            self._commit_snapshot = None
            self._page.set_status_text(msg)
            self._log(f"Q fail: {msg}")

        # ---- 1. 推理失败 ----
        if result is None or result.size == 0 or snap is None:
            _abort("Q: Inpaint failed. Not saved.")
            return

        # ---- 2. 写 image.tmp.<ext> + label.tmp.txt ----
        # P0-3: temp 必须保留真实图片扩展名（cv2.im* 按扩展名选编码器）
        path_obj = Path(snap.output_path)
        tmp_img = str(path_obj.parent / (path_obj.stem + ".tmp" + path_obj.suffix))
        # image: RGB -> BGR for cv2.imwrite
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        img_ok = cv2.imwrite(tmp_img, result_bgr)

        # label content
        label_content = self._mask_label_service.make_label_content(
            snap.bbox, snap.ordered_points, snap.image_size
        )
        tmp_label = snap.label_path + ".tmp.txt"
        label_ok = False
        if label_content:
            try:
                os.makedirs(os.path.dirname(tmp_label), exist_ok=True)
                with open(tmp_label, "w", encoding="utf-8") as f:
                    f.write(label_content + "\n")
                label_ok = True
            except OSError:
                label_ok = False

        # ---- 任一写失败：清理 temp + 不破坏旧正式文件 ----
        if not (img_ok and label_ok):
            self._cleanup_tmp(tmp_img, tmp_label)
            fail_reason = []
            if not img_ok:
                fail_reason.append("image write")
            if not label_ok:
                fail_reason.append("label write")
            _abort(f"Q: Save failed ({', '.join(fail_reason)}). Not committed.")
            return

        # ---- 3. 备份旧正式文件 + rename temp -> 正式 ----
        # 如果正式文件已存在：先备份，rename 成功后删除备份，失败恢复
        # 如果正式文件不存在：直接 rename
        backup_img = snap.output_path + ".bak"
        backup_label = snap.label_path + ".bak"
        had_img = os.path.exists(snap.output_path)
        had_label = os.path.exists(snap.label_path)

        try:
            # ---- 备份旧文件 ----
            if had_img:
                os.replace(snap.output_path, backup_img)
            if had_label:
                os.replace(snap.label_path, backup_label)

            # ---- rename temp -> 正式 ----
            os.replace(tmp_img, snap.output_path)
            os.replace(tmp_label, snap.label_path)

            # ---- 正式文件成功后删除备份 ----
            if had_img and os.path.exists(backup_img):
                os.remove(backup_img)
            if had_label and os.path.exists(backup_label):
                os.remove(backup_label)

        except OSError:
            # ---- 任一 rename 失败：恢复旧正式文件 + 清理 temp ----
            # 先把可能已经替换的正式文件回退回 temp 名
            try:
                if os.path.exists(tmp_img):
                    # temp 还在 -> 正式未替换 -> noop
                    pass
                elif os.path.exists(snap.output_path):
                    # 正式已被替换但 label rename 失败 -> 把 image 回到 temp
                    os.replace(snap.output_path, tmp_img)
            except OSError:
                pass
            # 恢复备份
            try:
                if had_img and os.path.exists(backup_img):
                    os.replace(backup_img, snap.output_path)
                if had_label and os.path.exists(backup_label):
                    os.replace(backup_label, snap.label_path)
            except OSError:
                pass
            # 清理残留 temp / 备份
            self._cleanup_tmp(tmp_img, tmp_label, backup_img, backup_label)
            _abort("Q: Failed to rename output files. Old files restored.")
            return

        # ---- 4. apply 预验证过的 Reference State ----
        # apply_prepared_reference 仅做赋值不会失败；prepared_ref 已在 _on_commit 中预验证
        if not self._reference_alignment_service.apply_prepared_reference(snap.prepared_ref):
            # 文件已写成功但 ref 应用失败（理论不应发生）-> 留当前图让用户决定
            _abort("Q: Files saved but reference apply failed. Staying on current image.")
            return

        # ---- 5. P1-4: 成功清理 + busy = false ----
        self._reference_index = snap.current_index
        self._current_prediction = None     # 清空旧 prediction
        self._busy = False
        self._page.set_busy(False)
        self._commit_snapshot = None
        self._refresh_status()
        self._page.set_status_text("Q: Commit success!")
        self._log(f"output image path: {snap.output_path}")
        self._log(f"label path: {snap.label_path}")
        self._log(f"rolling ref updated -> index={snap.current_index} | {snap.current_path}")
        self._log("Q success")

        # ---- 6. show_next_image()（Q 成功以前绝不切下一张）----
        self._on_next()

    @staticmethod
    def _cleanup_tmp(*paths: str) -> None:
        """清理临时文件。"""
        for p in paths:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass

    def _on_help(self) -> None:
        """显示帮助说明。"""
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self._page, "LaMa Erasure 使用说明",
            "工作流：\n"
            "1. Open 打开图片\n"
            f"2. 先选 Keypoints 数量，再左键画 mask（{self._current_expected_count()} 个）/ 右键擦\n"
            "3. SetRef 设为基准\n"
            "4. TestOne 测试追踪\n"
            "5. 下一张 -> A 预测 mask -> 人工修正 -> Q 提交\n"
            "6. Q 自动下一张 -> A -> 修正 -> Q 循环\n\n"
            "快捷键：A = AssistMask，Q = Commit"
        )

    # ================================================================ 内部：QImage <-> numpy 转换
    @staticmethod
    def _qimage_to_rgb_np(qimg: QImage) -> np.ndarray:
        """QImage -> numpy HxWx3 uint8 RGB。

        处理 RGB888 格式 + bytesPerLine padding。
        返回 .copy() 以脱离 QImage 内部 buffer。
        """
        if qimg.isNull():
            return np.zeros((0, 0, 3), dtype=np.uint8)
        img = qimg.convertToFormat(QImage.Format.Format_RGB888)
        w, h = img.width(), img.height()
        stride = img.bytesPerLine()
        buf = bytes(img.constBits())
        arr = np.frombuffer(buf, dtype=np.uint8, count=h * stride)
        arr = arr.reshape(h, stride)
        # 处理 stride padding
        if stride != w * 3:
            arr = arr[:, :w * 3]
        arr = arr.reshape(h, w, 3)
        return arr.copy()

    @staticmethod
    def _mask_qimage_to_np(qimg: QImage) -> np.ndarray:
        """QImage -> numpy HxW uint8 0/255（Grayscale8）。

        处理 bytesPerLine padding。
        """
        if qimg.isNull():
            return np.zeros((0, 0), dtype=np.uint8)
        img = qimg.convertToFormat(QImage.Format.Format_Grayscale8)
        w, h = img.width(), img.height()
        stride = img.bytesPerLine()
        buf = bytes(img.constBits())
        arr = np.frombuffer(buf, dtype=np.uint8, count=h * stride)
        arr = arr.reshape(h, stride)
        if stride != w:
            arr = arr[:, :w]
        return arr.copy()

    @staticmethod
    def _np_mask_to_qimage(np_mask: np.ndarray) -> QImage:
        """numpy HxW uint8 0/255 -> QImage Grayscale8。

        返回 .copy() 以脱离 numpy buffer。
        """
        if np_mask is None or np_mask.size == 0:
            return QImage()
        if np_mask.ndim == 3:
            np_mask = cv2.cvtColor(np_mask, cv2.COLOR_RGB2GRAY)
        h, w = np_mask.shape[:2]
        np_mask = np.ascontiguousarray(np_mask)
        img = QImage(np_mask.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        return img.copy()

    @staticmethod
    def _np_rgb_to_qimage(np_rgb: np.ndarray) -> QImage:
        """numpy HxWx3 uint8 RGB -> QImage RGB888。

        返回 .copy() 以脱离 numpy buffer。
        """
        if np_rgb is None or np_rgb.size == 0:
            return QImage()
        h, w = np_rgb.shape[:2]
        np_rgb = np.ascontiguousarray(np_rgb)
        img = QImage(np_rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        return img.copy()

    # ================================================================ 内部：current_prediction 访问
    def _get_predicted_centers(self) -> dict[int, tuple[float, float]]:
        """从 current_prediction 提取 {track_id: (cx, cy)}（P0-2 修订）。

        规则：
        - 只返回 ok=True 的 track 的 current_center
        - failed track 不允许用未位移的 ref_center 作为正常预测（这是危险兜底）
        - 返回长度 < N 时上层 _on_commit 会 BLOCK，绝不偷偷用 y/x 重排建 ID
        """
        if self._current_prediction is None:
            return {}
        centers = {}
        for t in self._current_prediction.tracks:
            if t.ok:
                centers[t.id] = t.current_center
        return centers

    # ================================================================ 状态显示
    def _refresh_status(self) -> None:
        """刷新 Page 的永久状态栏。"""
        n = len(self._image_files)
        if n <= 0 or self._current_index < 0:
            self._page.set_current_info(-1, 0, "-")
        else:
            filename = Path(self._image_files[self._current_index]).name
            self._page.set_current_info(self._current_index, n, filename)

        # ---- 关键点数量 N（SetRef 后锁定，之前取 UI 当前值）----
        kp = self._current_expected_count()

        if self._reference_index >= 0:
            self._page.set_reference_info(self._reference_index, None)
            self._page.set_reference_ready(
                self._reference_alignment_service.is_ready(), kp
            )
        else:
            self._page.set_reference_info(None, None)
            self._page.set_reference_ready(False)

        # ---- prediction 状态：从 current_prediction 取 ----
        if self._current_prediction is not None:
            self._page.set_prediction_info(
                self._current_prediction.success,
                self._current_prediction.total
            )
        else:
            self._page.set_prediction_info(0, kp)

        self._page.set_busy(self._busy)
