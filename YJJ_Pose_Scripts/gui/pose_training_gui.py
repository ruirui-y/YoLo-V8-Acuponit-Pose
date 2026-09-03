"""
YJJ RGBD Pose 实验桌面 GUI 启动入口 (PySide6)

仅负责创建 QApplication + MainWindow 并进入事件循环；
所有 UI 装配 / 子进程协调 / 业务逻辑在 main_window.py 与各 panels/ 模块。

启动（保持不变）:
    python YJJ_Pose_Scripts/gui/pose_training_gui.py
"""

from __future__ import annotations

import sys

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from main_window import MainWindow


def apply_dark_theme(app: QApplication) -> None:
    """应用全局深色主题。"""
    app.setStyle("Fusion")

    palette = QPalette()

    palette.setColor(QPalette.Window, QColor(32, 33, 36))
    palette.setColor(QPalette.WindowText, QColor(230, 230, 230))

    palette.setColor(QPalette.Base, QColor(24, 25, 28))
    palette.setColor(QPalette.AlternateBase, QColor(40, 41, 45))

    palette.setColor(QPalette.Text, QColor(230, 230, 230))
    palette.setColor(QPalette.PlaceholderText, QColor(140, 140, 140))

    palette.setColor(QPalette.Button, QColor(45, 46, 50))
    palette.setColor(QPalette.ButtonText, QColor(230, 230, 230))

    palette.setColor(QPalette.ToolTipBase, QColor(45, 46, 50))
    palette.setColor(QPalette.ToolTipText, QColor(230, 230, 230))

    palette.setColor(QPalette.Highlight, QColor(60, 100, 160))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

    palette.setColor(QPalette.Link, QColor(100, 160, 220))
    palette.setColor(QPalette.BrightText, QColor(255, 90, 90))

    app.setPalette(palette)


def main() -> None:
    app = QApplication(sys.argv)

    apply_dark_theme(app)

    w = MainWindow()
    w.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()