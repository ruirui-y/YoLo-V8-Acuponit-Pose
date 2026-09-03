"""Pose 状态 + 停止面板（信号驱动，与 MainWindow / Controller 完全解耦）。

运行日志已上移为应用级组件（widgets/shared_log_panel.SharedLogPanel），
本 Panel 不再拥有 QPlainTextEdit，也不再负责日志收起/展开。

Panel 只暴露一个视觉区域：
- status_widget：状态区（状态标签 + 停止当前任务按钮），由 Page 放到左侧工作区

停止按钮发出 stopRequested 信号（Controller 处理进程终止）。
（类名暂不改动，保持最小改动。）
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QPushButton, QWidget,
)


class StatusLogPanel(QWidget):
    # ---- 对外信号 ----
    stopRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        # ---- 状态区（放到左侧工作区）----
        self.status_widget = QGroupBox("状态", self)
        h5 = QHBoxLayout(self.status_widget)
        self.lbl_status = QLabel("未开始")
        h5.addWidget(QLabel("状态:"))
        h5.addWidget(self.lbl_status)
        self.btn_stop = QPushButton("停止当前任务")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stopRequested.emit)
        h5.addWidget(self.btn_stop)

    # ---- 公开：展示方法 ----
    def set_status(self, text):
        self.lbl_status.setText(text)

    def set_stop_enabled(self, enabled):
        """供 Controller 在任务运行期间控制停止按钮启停。"""
        self.btn_stop.setEnabled(enabled)
