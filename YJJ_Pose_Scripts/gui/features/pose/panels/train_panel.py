"""运行环境 + 训练配置 + 权重面板（信号驱动，与 MainWindow / Controller 完全解耦）。

Panel 只负责控件创建 / 展示 / 发信号；按钮 clicked 连接到自己的 signal，
Controller 连接这些 signal 并处理业务逻辑。
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QWidget,
)

from gui_config import _REPO_ROOT


class TrainPanel(QWidget):
    # ---- 对外信号 ----
    trainRgbRequested = Signal()
    build4chRequested = Signal()
    trainRgbdRequested = Signal()
    baseWeightChanged = Signal(str)         # 基础权重文本变更（editingFinished）
    settingsDirty = Signal()                # 路径有变动，请求保存 QSettings

    def __init__(
        self,
        parent: QWidget | None = None,
        start_dir_provider: Callable[[QLineEdit], str] | None = None,
    ) -> None:
        super().__init__(parent)
        # browse 起始目录 QSettings fallback 提供者（由 Controller 注入，可空）
        self._start_dir_provider: Callable[[QLineEdit], str] | None = start_dir_provider
        self._build_ui()

    def set_start_dir_provider(self, provider: Callable[[QLineEdit], str] | None) -> None:
        """供 Controller 在构造完成后注入 QSettings fallback 路径查询回调。"""
        self._start_dir_provider = provider

    def _build_ui(self) -> None:
        # ---- 运行环境（Python 训练环境）----
        ge = QGroupBox("运行环境")
        fe = QFormLayout(ge)
        self.le_py = QLineEdit()
        self.row_py = self._make_browse_row(self.le_py, is_dir=False, filt="Python 可执行文件 (*.exe)")
        self.le_py.editingFinished.connect(self.settingsDirty.emit)
        fe.addRow("Python训练环境:", self.row_py)

        # ---- 训练配置 ----
        g2 = QGroupBox("训练配置")
        f2 = QFormLayout(g2)
        self.le_fliplr = QLineEdit("0.0")
        self.le_patience = QLineEdit("20")
        f2.addRow("fliplr:", self.le_fliplr)
        f2.addRow("patience:", self.le_patience)

        # ---- 权重（傻瓜化：基础母版 + 自动派生 4ch）----
        gw = QGroupBox("权重")
        fw = QFormLayout(gw)
        self.le_base = QLineEdit()
        self.row_base = self._make_browse_row(self.le_base, is_dir=False, filt="Weights (*.pt)")
        self.le_base.editingFinished.connect(
            lambda: self.baseWeightChanged.emit(self.le_base.text().strip()))
        fw.addRow("基础 Pose 权重:", self.row_base)

        # RGB：直接用基础权重训练
        self.btn_train_rgb = QPushButton("开始训练 RGB")
        self.btn_train_rgb.clicked.connect(self.trainRgbRequested.emit)
        fw.addRow("RGB:", self.btn_train_rgb)

        # RGBD：4ch 权重由基础权重自动派生，先生成再训练
        self.lbl_4ch = QLabel("-")
        fw.addRow("4ch权重:", self.lbl_4ch)
        self.btn_build = QPushButton("生成4ch权重")
        self.btn_build.clicked.connect(self.build4chRequested.emit)
        self.btn_train_rgbd = QPushButton("开始训练 RGBD")
        self.btn_train_rgbd.clicked.connect(self.trainRgbdRequested.emit)
        h4 = QHBoxLayout()
        h4.addWidget(self.btn_build)
        h4.addWidget(self.btn_train_rgbd)
        fw.addRow(h4)

        v = QVBoxLayout(self)
        v.addWidget(ge)
        v.addWidget(g2)
        v.addWidget(gw)

    # ---- 内部：浏览对话框 ----
    def _make_browse_row(self, le: QLineEdit, is_dir: bool = False, filt: str = "All (*)") -> QWidget:
        """创建 QLineEdit + "选择..." 按钮的组合控件。"""
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(le)
        b = QPushButton("选择...")
        b.clicked.connect(lambda: self._browse(le, is_dir, filt))
        h.addWidget(b)
        return w

    def _resolve_start(self, le: QLineEdit) -> str:
        """文件/目录对话框起始目录优先级：当前输入框 > QSettings 上次 > 项目根。"""
        cur = le.text().strip()
        if cur:
            return cur
        if self._start_dir_provider is not None:
            v = self._start_dir_provider(le)
            if v:
                return v
        return str(_REPO_ROOT)

    def _browse(self, le: QLineEdit, is_dir: bool, filt: str) -> None:
        """打开文件/目录对话框并设置文本。"""
        start = self._resolve_start(le)
        if is_dir:
            p = QFileDialog.getExistingDirectory(self, "选择目录", start)
        else:
            p, _ = QFileDialog.getOpenFileName(self, "选择文件", start, filt)
        if p:
            le.setText(p)
            self.settingsDirty.emit()
            # 浏览选择基础权重后立即刷新 4ch 显示（与原行为一致）
            if le is self.le_base:
                self.baseWeightChanged.emit(self.le_base.text().strip())

    # ---- 公开：展示方法 ----
    def set_4ch_text(self, text: str) -> None:
        self.lbl_4ch.setText(text)

    def set_operation_buttons_enabled(self, enabled: bool) -> None:
        """供 Controller 在任务运行期间控制训练/构建按钮启停。"""
        self.btn_train_rgb.setEnabled(enabled)
        self.btn_build.setEnabled(enabled)
        self.btn_train_rgbd.setEnabled(enabled)
