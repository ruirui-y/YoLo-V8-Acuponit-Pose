"""数据集准备面板（纯 UI：控件创建 + 与数据集展示强相关的纯展示方法）。

真正启动 prepare 的 QProcess 行为由 MainWindow 协调：
- 浏览 / 模式切换 / 已有数据集检查等按钮、信号均连接到 MainWindow 的方法。
- 本面板只暴露控件（供 MainWindow 读取取值）与纯展示方法（reset_stats / set_stats / set_ready）。
"""
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QRadioButton, QVBoxLayout, QWidget,
)


class DatasetPanel(QWidget):
    def __init__(self, main):
        super().__init__()
        # ---- 数据集准备 ----
        g = QGroupBox("数据集准备")
        form = QFormLayout(g)

        # 数据集来源模式（默认：使用已有数据集）
        self.rb_new = QRadioButton("准备新数据集")
        self.rb_existing = QRadioButton("使用已有数据集")
        self.rb_existing.setChecked(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.rb_new, 1)
        self.mode_group.addButton(self.rb_existing, 2)
        self.mode_group.buttonClicked.connect(main._on_mode_changed)
        hmode = QHBoxLayout()
        hmode.addWidget(self.rb_new)
        hmode.addWidget(self.rb_existing)
        form.addRow("数据集来源:", hmode)

        # 已有数据集目录（仅“使用已有数据集”模式启用）
        self.le_existing = QLineEdit()
        we = QWidget()
        he = QHBoxLayout(we)
        he.setContentsMargins(0, 0, 0, 0)
        he.addWidget(self.le_existing)
        self.btn_browse_existing = QPushButton("选择...")
        self.btn_browse_existing.clicked.connect(main._on_browse_existing)
        he.addWidget(self.btn_browse_existing)
        self.row_existing = we
        self.le_existing.editingFinished.connect(main._check_existing_dataset)
        form.addRow("已有数据集目录:", self.row_existing)

        # —— 使用已有数据集：明确显示 RGB / RGBD 来源（透明、不猜测） ——
        # RGB 固定识别 <root>/rgb/data_rgb.yaml，仅展示完整路径（只读）
        self.le_rgb_yaml = QLineEdit()
        self.le_rgb_yaml.setReadOnly(True)
        self.le_rgb_yaml.setText("-")
        form.addRow("RGB数据集:", self.le_rgb_yaml)

        # RGBD 自动扫描所有满足 rgbd:true & channels:4 的 4ch 变体，下拉选择
        self.cb_rgbd_variant = QComboBox()
        self.cb_rgbd_variant.setEnabled(False)  # 仅“使用已有数据集”模式启用
        self.cb_rgbd_variant.setToolTip("自动扫描目录下所有 4 通道 RGBD 数据集（按 yaml 内容判断）")
        self.cb_rgbd_variant.currentTextChanged.connect(main._on_rgbd_variant_changed)
        form.addRow("RGBD数据集:", self.cb_rgbd_variant)

        # 当前选中的 RGBD yaml 完整路径（只读，确保不静默选择）
        self.le_rgbd_yaml = QLineEdit()
        self.le_rgbd_yaml.setReadOnly(True)
        self.le_rgbd_yaml.setText("-")
        form.addRow("当前 RGBD YAML:", self.le_rgbd_yaml)

        # 准备新数据集相关控件
        self.le_rgb = QLineEdit()
        self.le_depth_npy = QLineEdit()
        self.le_label = QLineEdit()
        self.le_out = QLineEdit()
        self.le_class = QLineEdit("hand")
        self.le_low = QLineEdit("1100")
        self.le_high = QLineEdit("1850")
        self.row_rgb = main._with_browse(self.le_rgb, dir=True)
        form.addRow("RGB目录:", self.row_rgb)
        self.row_depth_npy = main._with_browse(self.le_depth_npy, dir=True)
        form.addRow("Raw Depth NPY目录:", self.row_depth_npy)
        self.row_label = main._with_browse(self.le_label, dir=True)
        form.addRow("Labels目录:", self.row_label)
        self.row_out = main._with_browse(self.le_out, dir=True)
        form.addRow("输出目录:", self.row_out)
        form.addRow("类别名:", self.le_class)
        form.addRow("Depth low(mm):", self.le_low)
        form.addRow("Depth high(mm):", self.le_high)
        self.btn_prepare = QPushButton("准备数据集")
        self.btn_prepare.clicked.connect(main._on_prepare)
        form.addRow(self.btn_prepare)

        # 数据集统计显示
        st = QFormLayout()
        self.lbl_total = QLabel("-")
        self.lbl_train = QLabel("-")
        self.lbl_val = QLabel("-")
        self.lbl_test = QLabel("-")
        self.lbl_kpt = QLabel("-")
        self.lbl_rgbch = QLabel("-")
        self.lbl_rgbdch = QLabel("-")
        self.lbl_rgb_ds = QLabel("-")
        self.lbl_rgbd_ds = QLabel("-")
        self.lbl_ready = QLabel("-")
        st.addRow("总样本数:", self.lbl_total)
        st.addRow("train数量:", self.lbl_train)
        st.addRow("val数量:", self.lbl_val)
        st.addRow("test数量:", self.lbl_test)
        st.addRow("关键点数量:", self.lbl_kpt)
        st.addRow("RGB通道数:", self.lbl_rgbch)
        st.addRow("RGBD通道数:", self.lbl_rgbdch)
        st.addRow("RGB数据集:", self.lbl_rgb_ds)
        st.addRow("RGBD数据集:", self.lbl_rgbd_ds)
        st.addRow("状态:", self.lbl_ready)
        form.addRow(st)

        v = QVBoxLayout(self)
        v.addWidget(g)

    # ---- 与数据集展示强相关的纯展示方法 ----
    def reset_stats(self):
        """清空统计显示，避免上一套数据集数据残留造成误解（不含状态行）。"""
        for lbl in (self.lbl_total, self.lbl_train, self.lbl_val,
                    self.lbl_test, self.lbl_kpt, self.lbl_rgbch, self.lbl_rgbdch,
                    self.lbl_rgb_ds, self.lbl_rgbd_ds):
            lbl.setText("-")

    def set_stats(self, merged, ready):
        """写回现场统计/报告合并后的统计 + 状态行（None 值显示“-”）。"""
        self.lbl_total.setText(str(merged["total"]))
        self.lbl_train.setText(str(merged["train"]))
        self.lbl_val.setText(str(merged["val"]))
        self.lbl_test.setText(str(merged["test"]))
        self.lbl_kpt.setText(
            str(merged["keypoint_count"]) if merged["keypoint_count"] is not None else "-")
        self.lbl_rgbch.setText(
            str(merged["rgb_channels"]) if merged["rgb_channels"] is not None else "-")
        self.lbl_rgbdch.setText(
            str(merged["rgbd_channels"]) if merged["rgbd_channels"] is not None else "-")
        self.lbl_rgb_ds.setText(
            str(merged["rgb_dataset"]) if merged.get("rgb_dataset") is not None else "-")
        self.lbl_rgbd_ds.setText(
            str(merged["rgbd_dataset"]) if merged.get("rgbd_dataset") is not None else "-")
        self.lbl_ready.setText(ready)

    def set_ready(self, text):
        self.lbl_ready.setText(text)
