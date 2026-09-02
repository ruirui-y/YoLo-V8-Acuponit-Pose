"""状态 + 停止 + 运行日志面板（信号驱动，与 MainWindow / Controller 完全解耦）。

Panel 暴露两个独立视觉区域：
- status_widget：状态区（含状态标签 / 停止按钮 / 收起-展开日志按钮），由 Page 放到左侧滚动区
- log_widget：运行日志区（QPlainTextEdit），由 Page 放到右侧 QSplitter

停止按钮发出 stopRequested 信号（Controller 处理进程终止）；
收起/展开按钮发出 toggleLogRequested 信号（Page 处理 Splitter 布局）。
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout,
    QWidget,
)


class StatusLogPanel(QWidget):
    # ---- 对外信号 ----
    stopRequested = Signal()
    toggleLogRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        # ---- 状态区（放到左侧滚动区）----
        self.status_widget = QGroupBox("状态", self)
        h5 = QHBoxLayout(self.status_widget)
        self.lbl_status = QLabel("未开始")
        h5.addWidget(QLabel("状态:"))
        h5.addWidget(self.lbl_status)
        self.btn_stop = QPushButton("停止当前任务")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stopRequested.emit)
        h5.addWidget(self.btn_stop)
        self.btn_toggle_log = QPushButton("收起日志")
        self.btn_toggle_log.clicked.connect(self.toggleLogRequested.emit)
        h5.addWidget(self.btn_toggle_log)

        # ---- 日志区（放到右侧 splitter）----
        self.log_widget = QGroupBox("运行日志", self)
        lv = QVBoxLayout(self.log_widget)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        lv.addWidget(self.log)

    # ---- 公开：展示方法 ----
    def set_status(self, text):
        self.lbl_status.setText(text)

    def append_log(self, text):
        self.log.appendPlainText(text)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def set_stop_enabled(self, enabled):
        """供 Controller 在任务运行期间控制停止按钮启停。"""
        self.btn_stop.setEnabled(enabled)

    def set_toggle_log_text(self, text):
        """供 Page 在收起/展开后更新按钮文字。"""
        self.btn_toggle_log.setText(text)

    def set_log_min_width(self, width):
        """供 Page 在收起/展开时调整日志区最小宽度。"""
        self.log_widget.setMinimumWidth(width)
