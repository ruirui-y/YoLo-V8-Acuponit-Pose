"""
YJJ RGBD Pose 实验桌面 GUI 启动入口 (PySide6)

仅负责创建 QApplication + MainWindow 并进入事件循环；
所有 UI 装配 / 子进程协调 / 业务逻辑在 main_window.py 与各 panels/ 模块。

启动（保持不变）:
    python YJJ_Pose_Scripts/gui/pose_training_gui.py
"""
import sys

from PySide6.QtWidgets import QApplication

from main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
