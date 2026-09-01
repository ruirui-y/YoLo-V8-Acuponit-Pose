"""状态 + 停止 + 运行日志面板（纯 UI，不启动子进程）。

布局调整（日志移到右侧）后，本面板不再自己排版，而是暴露两个独立视觉区域：
- status_widget：状态区（含状态标签 / 停止按钮 / 收起-展开日志按钮），由 MainWindow 放到左侧滚动区
- log_widget：运行日志区（QPlainTextEdit），由 MainWindow 放到右侧 QSplitter

MainWindow 通过 set_status / append_log 写回；停止按钮、收起/展开按钮的 clicked 信号
连接到 MainWindow 的对应方法（协调逻辑留在 MainWindow）。
"""
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout,
    QWidget,
)


class StatusLogPanel(QWidget):
    def __init__(self, main):
        super().__init__()
        # ---- 状态区（放到左侧滚动区）----
        self.status_widget = QGroupBox("状态", self)
        h5 = QHBoxLayout(self.status_widget)
        self.lbl_status = QLabel("未开始")
        h5.addWidget(QLabel("状态:"))
        h5.addWidget(self.lbl_status)
        # 停止当前任务（初始禁用，仅子进程运行期间可点）
        self.btn_stop = QPushButton("停止当前任务")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(main._on_stop)
        h5.addWidget(self.btn_stop)
        # 收起 / 展开日志（控制右侧 splitter 宽度）；初始文字“收起日志”
        self.btn_toggle_log = QPushButton("收起日志")
        self.btn_toggle_log.clicked.connect(main._on_toggle_log)
        h5.addWidget(self.btn_toggle_log)

        # ---- 日志区（放到右侧 splitter）----
        self.log_widget = QGroupBox("运行日志", self)
        lv = QVBoxLayout(self.log_widget)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        lv.addWidget(self.log)

        # 面板自身不排版：两个子区域由 MainWindow 分别放入左 / 右

    # ---- 纯展示接口（供 MainWindow 调用）----
    def set_status(self, text: str):
        self.lbl_status.setText(text)

    def append_log(self, text: str):
        self.log.appendPlainText(text)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())
