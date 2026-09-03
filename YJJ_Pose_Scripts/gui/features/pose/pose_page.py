"""RGBD Pose 实验工作区页面（Feature Page）。

负责组装现有 DatasetPanel / TrainPanel / EvalPanel / StatusLogPanel：
- 单列可滚动工作区：Dataset / Train / Eval / Pose 状态区

运行日志已上移为应用级组件（widgets/shared_log_panel.SharedLogPanel），
由 MainWindow 创建并放在顶层 Horizontal QSplitter 右侧，
本页不再创建 QSplitter，也不再管理日志控件 / 收起-展开逻辑；
只通过构造注入的 log_sink 写日志。
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from .panels.dataset_panel import DatasetPanel
from .panels.eval_panel import EvalPanel
from .panels.status_log_panel import StatusLogPanel
from .panels.train_panel import TrainPanel
from .pose_controller import PoseController


class PosePage(QWidget):
    """RGBD Pose 实验工作区：组装 Panel + 创建 Controller（单列布局）。"""

    def __init__(
        self,
        parent: QWidget | None = None,
        log_sink: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent)

        # ---- 应用级共享日志 sink（MainWindow 注入 SharedLogPanel.append_log）----
        self._log_sink = log_sink if callable(log_sink) else (lambda _text: None)

        # ---- 创建各 Panel（Panel 构造不连接 Controller，只创建控件）----
        self.dataset_panel = DatasetPanel()
        self.train_panel = TrainPanel()
        self.eval_panel = EvalPanel()
        self.status_log_panel = StatusLogPanel()

        # ---- 创建 Controller（连接 Panel signals + 初始化）----
        self.controller = PoseController(
            self.dataset_panel, self.train_panel,
            self.eval_panel, self.status_log_panel,
            log_sink=self._log_sink,
        )

        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # ---- 单列可滚动工作区：Dataset / Train / Eval / Pose 状态 ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        scroll.setWidget(body)
        v = QVBoxLayout(body)
        v.addWidget(self.dataset_panel)
        v.addWidget(self.train_panel)
        v.addWidget(self.eval_panel)
        v.addWidget(self.status_log_panel.status_widget)
        root.addWidget(scroll)

    def save_settings(self) -> None:
        """供 MainWindow closeEvent 调用，持久化 Pose 配置。"""
        self.controller.save_settings()
