"""RGBD Pose 实验工作区页面（Feature Page）。

负责组装现有 DatasetPanel / TrainPanel / EvalPanel / StatusLogPanel，
以及当前左右 QSplitter 布局：
- 左侧（可滚动）：Dataset / Train / Eval / Status
- 右侧：运行日志

从 MainWindow._build_ui() 迁移的布局逻辑，MainWindow 不再直接知道这些 Panel。
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QScrollArea, QSplitter, QVBoxLayout, QWidget,
)

from .panels.dataset_panel import DatasetPanel
from .panels.eval_panel import EvalPanel
from .panels.status_log_panel import StatusLogPanel
from .panels.train_panel import TrainPanel
from .pose_controller import PoseController


class PosePage(QWidget):
    """RGBD Pose 实验工作区：组装 Panel + 创建 Controller + 左右 Splitter 布局。"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # ---- 创建各 Panel（Panel 构造不连接 Controller，只创建控件）----
        self.dataset_panel = DatasetPanel()
        self.train_panel = TrainPanel()
        self.eval_panel = EvalPanel()
        self.status_log_panel = StatusLogPanel()

        # ---- 创建 Controller（连接 Panel signals + 初始化）----
        self.controller = PoseController(
            self.dataset_panel, self.train_panel,
            self.eval_panel, self.status_log_panel,
        )

        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # 左右结构：左侧可滚动（Dataset/Train/Eval/状态），右侧运行日志
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ---- 左侧：可滚动区 ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        scroll.setWidget(body)
        v = QVBoxLayout(body)
        v.addWidget(self.dataset_panel)
        v.addWidget(self.train_panel)
        v.addWidget(self.eval_panel)
        v.addWidget(self.status_log_panel.status_widget)
        splitter.addWidget(scroll)

        # ---- 右侧：运行日志 ----
        splitter.addWidget(self.status_log_panel.log_widget)

        # 默认比例 65% / 35%；右侧日志最小宽度约 350px
        splitter.setStretchFactor(0, 65)
        splitter.setStretchFactor(1, 35)
        self.status_log_panel.log_widget.setMinimumWidth(350)
        self._splitter = splitter
        self._log_collapsed = False
        self._log_prev_sizes = None

        # 收起 / 展开日志：由 Page 处理（Splitter 布局是 Page 的职责）
        self.status_log_panel.toggleLogRequested.connect(self._on_toggle_log)

    def _on_toggle_log(self):
        """收起 / 展开右侧运行日志：用 QSplitter.setSizes 控制右侧宽度。"""
        sp = self._splitter
        if not self._log_collapsed:
            # 收起：记住收起前宽度，临时取消右侧最小宽度限制，右侧设 0
            self._log_prev_sizes = sp.sizes()
            self.status_log_panel.set_log_min_width(0)
            sp.setSizes([sum(self._log_prev_sizes), 0])
            self._log_collapsed = True
            self.status_log_panel.set_toggle_log_text("展开日志")
        else:
            # 展开：恢复收起前右侧宽度，恢复右侧最小宽度
            if self._log_prev_sizes and len(self._log_prev_sizes) == 2:
                sp.setSizes(self._log_prev_sizes)
            else:
                total = sum(sp.sizes()) or 1000
                sp.setSizes([int(total * 0.65), int(total * 0.35)])
            self.status_log_panel.set_log_min_width(350)
            self._log_collapsed = False
            self.status_log_panel.set_toggle_log_text("收起日志")

    def save_settings(self):
        """供 MainWindow closeEvent 调用，持久化 Pose 配置。"""
        self.controller.save_settings()
