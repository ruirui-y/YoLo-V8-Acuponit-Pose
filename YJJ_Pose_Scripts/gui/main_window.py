"""YJJ RGBD Pose 实验桌面 GUI — Application Shell (PySide6)

重构后 MainWindow 只负责：
1. 创建顶层 QMainWindow
2. 创建导航栏（RGBD Pose / LaMa Erasure）
3. 创建 QStackedWidget
4. 创建 / 注册各 Feature Page
5. 页面切换
6. 必要的顶层窗口尺寸和标题

所有业务逻辑（rgb_yaml / rgbd_yaml / dataset / train / eval / ablate / QProcess /
QSettings / stdout 解析）均已移入 features/pose/ 和 core/。
"""
from PySide6.QtWidgets import (
    QHBoxLayout, QMainWindow, QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

from features.pose.pose_page import PosePage
from features.lama.lama_page import LamaPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YJJ RGBD Pose 训练 GUI")
        self.resize(1200, 820)

        # ---- 创建各 Feature Page ----
        self.pose_page = PosePage()
        self.lama_page = LamaPage()

        # ---- 导航栏 ----
        self.nav_pose = QPushButton("RGBD Pose")
        self.nav_pose.clicked.connect(self._show_pose)
        self.nav_lama = QPushButton("LaMa Erasure")
        self.nav_lama.clicked.connect(self._show_lama)

        # ---- QStackedWidget ----
        self.stack = QStackedWidget()
        self.stack.addWidget(self.pose_page)
        self.stack.addWidget(self.lama_page)

        # ---- 布局 ----
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        nav_bar = QHBoxLayout()
        nav_bar.addWidget(self.nav_pose)
        nav_bar.addWidget(self.nav_lama)
        nav_bar.addStretch()
        root.addLayout(nav_bar)
        root.addWidget(self.stack)
        self.setCentralWidget(central)

        # 默认显示 RGBD Pose
        self._show_pose()

    def _show_pose(self):
        self.stack.setCurrentWidget(self.pose_page)

    def _show_lama(self):
        self.stack.setCurrentWidget(self.lama_page)

    def closeEvent(self, event):
        self.pose_page.save_settings()
        super().closeEvent(event)
