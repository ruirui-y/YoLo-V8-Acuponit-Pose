"""应用级共享运行日志面板（Application Shell 组件）。

整个 GUI 只存在这一个 QPlainTextEdit 日志实例：
由 MainWindow 创建，放在 Horizontal QSplitter 右侧；
左侧 QStackedWidget 在 PosePage / LamaPage 之间切换时，本面板始终保持不变。

只负责显示，不含任何 Pose / LaMa 业务逻辑：
- QGroupBox("运行日志")
- QPlainTextEdit（readonly + NoWrap）
- append_log(text)：追加并自动滚到底部

各 feature 通过构造注入的 log_sink（即本类的 append_log）写日志，
不持有本类实例，也不知道日志控件的存在。
"""
from PySide6.QtWidgets import QGroupBox, QPlainTextEdit, QVBoxLayout


class SharedLogPanel(QGroupBox):
    """共享运行日志面板：全应用唯一的日志 QPlainTextEdit 宿主。"""

    def __init__(self, parent=None):
        super().__init__("运行日志", parent)
        lay = QVBoxLayout(self)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        lay.addWidget(self.log)

    # ---- 公开：日志 sink（注入给各 feature Controller）----
    def append_log(self, text):
        """追加日志文本并自动滚动到底部。"""
        self.log.appendPlainText(text)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())
