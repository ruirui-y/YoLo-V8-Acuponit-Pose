"""LaMa Erasure 工作区页面（PySide6 QWidget）。

只负责：
1. 创建控件（Toolbar / Buttons / Slider / Canvas / Status）
2. 布局
3. 显示状态（Current / Reference / Prediction / Busy）
4. 发 Signal

明确禁止在本文件实现：
- OpenCV 算法
- ONNX 推理
- Reference Tracking
- YOLO label 生成
- 文件保存工作流

由 LamaController 连接本页 signals，调用 services 处理业务。

参考 C++ LamaErasure/MainWindow.cpp::buildUi() 的工具栏布局，
但裁剪掉 Batch / AutoMask / TrackROI / SaveAs 等历史功能。
"""
from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider, QSpinBox,
    QToolBar, QVBoxLayout, QWidget,
)

from .widgets.inpaint_canvas import InpaintCanvas
from .lama_controller import LamaController


class LamaPage(QWidget):
    """LaMa 擦除工作区：纯 UI + 信号发射。"""

    # ---- 关键点数量选择范围 ----
    # 默认 7：与历史 Hand 工作流完全一致，不碰新控件的用户行为不变。
    MIN_KEYPOINT_COUNT = 1
    MAX_KEYPOINT_COUNT = 32
    DEFAULT_KEYPOINT_COUNT = 7

    # ---- 对外信号（Controller 连接）----
    openRequested = Signal()
    clearMaskRequested = Signal()
    setRefRequested = Signal()
    testOneRequested = Signal()
    assistMaskRequested = Signal()        # A 键
    commitRequested = Signal()            # Q 键
    helpRequested = Signal()
    prevRequested = Signal()
    nextRequested = Signal()
    brushRadiusChanged = Signal(int)       # 画笔半径

    def __init__(self, parent: QWidget | None = None, log_sink: Callable[[str], None] | None = None) -> None:
        super().__init__(parent)
        # ---- 应用级共享日志 sink（MainWindow 注入 SharedLogPanel.append_log）----
        self._log_sink: Callable[[str], None] = (
            log_sink if callable(log_sink) else (lambda _text: None)
        )
        # 关键点数量是否被锁定（SetRef 成功 -> True；Reset / 重新 Open -> False）
        self._keypoint_count_locked: bool = False
        self.controller: LamaController
        self._build_ui()
        # ---- 创建 Controller（连接本页 signals + 初始化）----
        # 延迟 import 避免 widgets 模块在 LamaPage import 阶段被循环依赖
        self.controller = LamaController(self, log_sink=self._log_sink)

    # ================================================================ UI 构建
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- Toolbar ----
        self._toolbar = QToolBar("LaMa Tools", self)
        self._toolbar.setMovable(False)
        root.addWidget(self._toolbar)

        self._add_action("Open", "打开单张或文件夹中的图片", self.openRequested)
        self._add_action("ClearMask", "清除当前涂抹", self.clearMaskRequested)
        self._add_action("SetRef", "将当前 mask 设为基准", self.setRefRequested)
        self._add_action("TestOne", "测试基准追踪效果", self.testOneRequested)
        self._add_action("AssistMask", "A 键：基于基准预测 mask 草稿", self.assistMaskRequested)
        self._add_action("Commit", "Q 键：擦除+保存+标签+下一张", self.commitRequested)
        self._add_action("Help", "查看操作说明", self.helpRequested)

        self._toolbar.addSeparator()
        self._toolbar.addWidget(QLabel("Brush:"))
        self._brush_slider = QSlider(Qt.Orientation.Horizontal)
        self._brush_slider.setRange(1, 80)
        self._brush_slider.setValue(9)
        self._brush_slider.setFixedWidth(160)
        self._toolbar.addWidget(self._brush_slider)

        # ---- 关键点数量 Keypoints：未建立 Reference 时可改，SetRef 后由 Controller 锁定 ----
        self._toolbar.addWidget(QLabel("Keypoints:"))
        self._keypoint_count_spin = QSpinBox()
        self._keypoint_count_spin.setRange(
            self.MIN_KEYPOINT_COUNT, self.MAX_KEYPOINT_COUNT)
        self._keypoint_count_spin.setValue(self.DEFAULT_KEYPOINT_COUNT)
        self._keypoint_count_spin.setToolTip(
            "本次标注的关键点数量；SetRef 成功后锁定，重新 Open / Reset 后解锁"
        )
        self._toolbar.addWidget(self._keypoint_count_spin)

        # ---- 中央：左右翻页 + 画布 ----
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        self._btn_prev = QPushButton("<")
        self._btn_prev.setFixedWidth(40)
        self._btn_prev.setSizePolicy(QSizePolicy.Policy.Fixed,
                                     QSizePolicy.Policy.Expanding)

        self.canvas = InpaintCanvas(self)

        self._btn_next = QPushButton(">")
        self._btn_next.setFixedWidth(40)
        self._btn_next.setSizePolicy(QSizePolicy.Policy.Fixed,
                                     QSizePolicy.Policy.Expanding)

        h.addWidget(self._btn_prev)
        h.addWidget(self.canvas, 1)
        h.addWidget(self._btn_next)
        root.addLayout(h)

        # ---- 状态栏（永久信息）----
        self._status_label = QLabel("Ready")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._status_label.setStyleSheet("padding: 4px;")
        root.addWidget(self._status_label)

        # ---- 信号连接 ----
        self._btn_prev.clicked.connect(self.prevRequested)
        self._btn_next.clicked.connect(self.nextRequested)
        self._brush_slider.valueChanged.connect(self.brushRadiusChanged)
        self.brushRadiusChanged.connect(self.canvas.set_brush_radius)

        # ---- 快捷键 ----
        from PySide6.QtGui import QKeySequence, QShortcut
        QShortcut(QKeySequence("A"), self, activated=self.assistMaskRequested.emit)
        QShortcut(QKeySequence("Q"), self, activated=self.commitRequested.emit)

    def _add_action(self, text: str, tooltip: str, signal: Signal):
        """工具栏按钮 + 信号绑定。"""
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.clicked.connect(signal.emit)
        self._toolbar.addWidget(btn)

    # ================================================================ 对外接口（Controller 调用）
    def set_status_text(self, text: str) -> None:
        """显示状态栏文本（临时消息）。"""
        self._status_label.setText(text)

    def set_current_info(self, current_idx: int, total: int, filename: str) -> None:
        """显示 Current: x/total | filename。current_idx 为 -1 表示无图。"""
        if total <= 0 or current_idx < 0:
            cur_text = "Current: -/-"
            name_text = "-"
        else:
            cur_text = f"Current: {current_idx + 1}/{total}"
            name_text = filename
        self._update_permanent_label(current=cur_text, name=name_text)

    def set_reference_info(self, ref_idx: int | None, ref_filename: str) -> None:
        """显示 Reference 信息。ref_idx 为 None 表示未设基准。"""
        if ref_idx is None:
            ref_text = "Ref: none"
        else:
            ref_text = f"Ref: {ref_idx + 1}"
        self._update_permanent_label(reference=ref_text)

    def set_reference_ready(self, ready: bool, keypoint_count: int | None = None) -> None:
        """显示 Reference 状态；ready 且给出数量时显示 “Reference Ready | N keypoints”。"""
        if ready:
            text = "Reference Ready"
            if keypoint_count is not None and keypoint_count > 0:
                text = f"{text} | {keypoint_count} keypoints"
        else:
            text = "Reference Not Ready"
        self._update_permanent_label(reference_state=text)

    def set_prediction_info(self, success: int, total: int) -> None:
        """显示 Prediction: success/total（total = 当前 session 关键点数量）。"""
        self._update_permanent_label(prediction=f"Prediction: {success}/{total}")

    # ---- 关键点数量（Controller 读取 / 锁定）----
    def KeypointCount(self) -> int:
        """当前 UI 选择的关键点数量 N（Controller 在 SetRef 时读取）。"""
        return int(self._keypoint_count_spin.value())

    def SetKeypointCountEditable(self, editable: bool) -> None:
        """是否允许修改关键点数量。

        SetRef 成功后由 Controller 传 editable=False 锁定；
        重新 Open / Reset session 后传 editable=True 解锁。
        """
        self._keypoint_count_locked = not bool(editable)
        self._keypoint_count_spin.setEnabled(bool(editable))

    def set_busy(self, busy: bool) -> None:
        """busy 时禁用 Q / A / Prev / Next / Open / SetRef / TestOne。"""
        for btn in self.findChildren(QPushButton):
            btn.setEnabled(not busy)
        # canvas 不禁用：用户在 busy 期间仍可看图（实际 Q 成功前 Controller 不会让用户操作）
        # 但为了避免破坏状态，busy 时整体禁用更安全
        self.canvas.setEnabled(not busy)
        # 关键点数量：busy 期间一律禁用，恢复时按当前锁定状态还原
        self._keypoint_count_spin.setEnabled((not busy) and (not self._keypoint_count_locked))

    # ---- busy 状态下仍允许显示瞬时状态 ----
    def set_busy_text(self, text: str) -> None:
        self._status_label.setText(text)

    # ================================================================ 内部
    def _update_permanent_label(self,
                               current: str | None = None,
                               name: str | None = None,
                               reference: str | None = None,
                               reference_state: str | None = None,
                               prediction: str | None = None,
                               busy_state: str | None = None) -> None:
        """合并显示一行永久状态信息。

        格式：Current: x/total | Ref: idx | filename | Reference Ready | Prediction: x/N | Ready
        """
        if not hasattr(self, "_perm_state"):
            self._perm_state: dict[str, str] = {
                "current": "Current: -/-",
                "reference": "Ref: none",
                "name": "-",
                "reference_state": "Reference Not Ready",
                "prediction": "Prediction: -/-",
                "busy_state": "Ready",
            }
        if current is not None:
            self._perm_state["current"] = current
        if reference is not None:
            self._perm_state["reference"] = reference
        if name is not None:
            self._perm_state["name"] = name
        if reference_state is not None:
            self._perm_state["reference_state"] = reference_state
        if prediction is not None:
            self._perm_state["prediction"] = prediction
        if busy_state is not None:
            self._perm_state["busy_state"] = busy_state
        s = self._perm_state
        self._status_label.setText(
            f"{s['current']} | {s['reference']} | {s['name']} | "
            f"{s['reference_state']} | {s['prediction']} | {s['busy_state']}"
        )
