"""运行环境 + 训练配置 + 权重面板（纯 UI，不启动子进程）。

包含：Python 训练环境、fliplr/patience、基础 Pose 权重、4ch 权重显示、
RGB 训练 / 生成4ch / RGBD 训练 按钮。
暂时保持当前 4ch 权重“自动派生”行为（命名规则不变，本面板不做自定义文件名）。

按钮的 clicked 信号连接到 MainWindow 的对应动作；基础权重编辑后刷新 4ch 显示
也连接到 MainWindow._refresh_4ch_label（派生路径逻辑留在 MainWindow / gui_config）。
"""
from PySide6.QtWidgets import (
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QWidget,
)


class TrainPanel(QWidget):
    def __init__(self, main):
        super().__init__()
        # ---- 运行环境（Python 训练环境）----
        ge = QGroupBox("运行环境")
        fe = QFormLayout(ge)
        self.le_py = QLineEdit()
        self.row_py = main._with_browse(self.le_py, dir=False, filt="Python 可执行文件 (*.exe)")
        self.le_py.editingFinished.connect(main._save_settings)
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
        # 基础 Pose 权重：yolov8n-pose.pt 母版，可无限次读取，不改坏
        self.le_base = QLineEdit()
        self.row_base = main._with_browse(self.le_base, dir=False, filt="Weights (*.pt)")
        self.le_base.editingFinished.connect(main._refresh_4ch_label)
        fw.addRow("基础 Pose 权重:", self.row_base)

        # RGB：直接用基础权重训练
        self.btn_train_rgb = QPushButton("开始训练 RGB")
        self.btn_train_rgb.clicked.connect(main._on_train_rgb)
        fw.addRow("RGB:", self.btn_train_rgb)

        # RGBD：4ch 权重由基础权重自动派生，先生成再训练
        self.lbl_4ch = QLabel("-")
        fw.addRow("4ch权重:", self.lbl_4ch)
        self.btn_build = QPushButton("生成4ch权重")
        self.btn_build.clicked.connect(main._on_build)
        self.btn_train_rgbd = QPushButton("开始训练 RGBD")
        self.btn_train_rgbd.clicked.connect(main._on_train_rgbd)
        h4 = QHBoxLayout()
        h4.addWidget(self.btn_build)
        h4.addWidget(self.btn_train_rgbd)
        fw.addRow(h4)

        v = QVBoxLayout(self)
        v.addWidget(ge)
        v.addWidget(g2)
        v.addWidget(gw)

    # ---- 纯展示接口 ----
    def set_4ch_text(self, text):
        self.lbl_4ch.setText(text)
