"""LaMa Erasure 扩展入口页面（占位）。

本次只建立简单占位页面，验证 MainWindow 的模块切换架构已成立。
不要迁移 LamaErasure C++ 项目、不要添加 ONNX / OpenCV / LaMa 推理代码。

未来结构：
features/lama/
├─ lama_page.py       ← 本文件
├─ lama_controller.py
├─ widgets/
└─ services/
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class LamaPage(QWidget):
    """LaMa Erasure 占位页面。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("LaMa Erasure\n\n待接入 LamaErasure 模块")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
