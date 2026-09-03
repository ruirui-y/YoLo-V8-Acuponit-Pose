"""YJJ RGBD Pose 实验桌面 GUI — Application Shell (PySide6)

MainWindow 只负责：
1. 创建顶层 QMainWindow
2. 创建导航栏（RGBD Pose / LaMa Erasure）
3. 创建唯一的 SharedLogPanel（应用级运行日志）
4. 创建 QStackedWidget + 各 Feature Page，并注入同一个日志 sink
5. 页面切换（只切左侧 stack，右侧日志始终不变）
6. 必要的顶层窗口尺寸和标题

布局：
    MainWindow
    └─ root (QVBoxLayout)
       ├─ nav
       └─ Horizontal QSplitter
          ├─ QStackedWidget
          │  ├─ PosePage
          │  └─ LamaPage
          └─ SharedLogPanel   ← 唯一日志实例，切页不变

所有业务逻辑（rgb_yaml / rgbd_yaml / dataset / train / eval / ablate / QProcess /
QSettings / stdout 解析 / LaMa 工作流）均在 features/ 和 core/。
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QMainWindow, QPushButton, QSplitter, QStackedWidget,
    QVBoxLayout, QWidget,
)

from features.pose.pose_page import PosePage
from features.lama.lama_page import LamaPage
from widgets.shared_log_panel import SharedLogPanel


# 右侧日志最小宽度
_LOG_MIN_WIDTH = 350
# 默认左右比例
_SPLIT_LEFT_RATIO = 0.65


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YJJ RGBD Pose 训练 GUI")
        self.resize(1400, 820)

        # ---- 1. 先创建唯一的运行日志（Page 的 log_sink 来源）----
        self.shared_log = SharedLogPanel()
        self.shared_log.setMinimumWidth(_LOG_MIN_WIDTH)
        log_sink = self.shared_log.append_log      # 同一个 callable

        # ---- 2. 创建各 Feature Page（注入同一个 log_sink）----
        self.pose_page = PosePage(log_sink=log_sink)
        self.lama_page = LamaPage(log_sink=log_sink)

        # ---- 3. 导航栏 ----
        self.nav_pose = QPushButton("RGBD Pose")
        self.nav_pose.clicked.connect(self._show_pose)
        self.nav_lama = QPushButton("LaMa Erasure")
        self.nav_lama.clicked.connect(self._show_lama)

        # ---- 4. 左侧 QStackedWidget ----
        self.stack = QStackedWidget()
        self.stack.addWidget(self.pose_page)
        self.stack.addWidget(self.lama_page)

        # ---- 5. Horizontal QSplitter：左 stack / 右 shared log ----
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.stack)
        self.splitter.addWidget(self.shared_log)
        self.splitter.setStretchFactor(0, 65)
        self.splitter.setStretchFactor(1, 35)
        total = self.width() or 1400
        self.splitter.setSizes([
            int(total * _SPLIT_LEFT_RATIO),
            int(total * (1.0 - _SPLIT_LEFT_RATIO)),
        ])

        # ---- 6. 布局 ----
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        nav_bar = QHBoxLayout()
        nav_bar.addWidget(self.nav_pose)
        nav_bar.addWidget(self.nav_lama)
        nav_bar.addStretch()
        root.addLayout(nav_bar)
        root.addWidget(self.splitter)
        self.setCentralWidget(central)

        # 默认显示 RGBD Pose
        self._show_pose()

    # ================================================================ 页面切换
    def _show_pose(self):
        """只切换左侧 stack，右侧 SharedLogPanel 保持不变。"""
        self.stack.setCurrentWidget(self.pose_page)

    def _show_lama(self):
        """只切换左侧 stack，右侧 SharedLogPanel 保持不变。"""
        self.stack.setCurrentWidget(self.lama_page)

    def closeEvent(self, event):
        self.pose_page.save_settings()
        super().closeEvent(event)
