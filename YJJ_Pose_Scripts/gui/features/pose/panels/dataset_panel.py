"""数据集准备面板（信号驱动，与 MainWindow / Controller 完全解耦）。

Panel 只负责：
- 创建自己的控件
- 控件启停（模式切换时内部处理 enable/disable）
- 展示数据（统计 / yaml 路径 / 变体下拉）
- 提供输入值（公开控件引用供 Controller 读取）
- 发出自己的 signal

Controller 连接这些 signal 并处理业务逻辑；Panel 不调用任何 Controller 方法。

browse 起始目录优先级由 Panel 内部 + 注入的 start_dir_provider 共同实现：
    当前输入框 → provider（Controller 用它读 QSettings 上次路径）→ _REPO_ROOT
provider 可在 Controller 构造完成后通过 set_start_dir_provider 注入，避免 Panel→Controller 反向依赖。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QRadioButton, QVBoxLayout, QWidget,
)

from gui_config import _REPO_ROOT


class DatasetPanel(QWidget):
    # ---- 对外信号 ----
    modeChanged = Signal()                  # 数据集来源模式切换
    existingDirChanged = Signal()           # 已有数据集目录变更（browse / 编辑完成）
    prepareRequested = Signal()             # 点击"准备数据集"
    rgbdVariantChanged = Signal(str)        # RGBD 变体下拉切换
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
        g = QGroupBox("数据集准备")
        form = QFormLayout(g)

        # ---- 数据集来源模式（默认：使用已有数据集）----
        self.rb_new = QRadioButton("准备新数据集")
        self.rb_existing = QRadioButton("使用已有数据集")
        self.rb_existing.setChecked(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.rb_new, 1)
        self.mode_group.addButton(self.rb_existing, 2)
        self.mode_group.buttonClicked.connect(self._on_mode_clicked)
        hmode = QHBoxLayout()
        hmode.addWidget(self.rb_new)
        hmode.addWidget(self.rb_existing)
        form.addRow("数据集来源:", hmode)

        # ---- 已有数据集目录（仅"使用已有数据集"模式启用）----
        self.le_existing = QLineEdit()
        we = QWidget()
        he = QHBoxLayout(we)
        he.setContentsMargins(0, 0, 0, 0)
        he.addWidget(self.le_existing)
        self.btn_browse_existing = QPushButton("选择...")
        self.btn_browse_existing.clicked.connect(self._on_browse_existing)
        he.addWidget(self.btn_browse_existing)
        self.row_existing = we
        self.le_existing.editingFinished.connect(self.existingDirChanged.emit)
        form.addRow("已有数据集目录:", self.row_existing)

        # —— 使用已有数据集：明确显示 RGB / RGBD 来源（透明、不猜测） ——
        # RGB 固定识别 <root>/rgb/data_rgb.yaml，仅展示完整路径（只读）
        self.le_rgb_yaml = QLineEdit()
        self.le_rgb_yaml.setReadOnly(True)
        self.le_rgb_yaml.setText("-")
        form.addRow("RGB数据集:", self.le_rgb_yaml)

        # RGBD 自动扫描所有满足 rgbd:true & channels:4 的 4ch 变体，下拉选择
        self.cb_rgbd_variant = QComboBox()
        self.cb_rgbd_variant.setEnabled(False)
        self.cb_rgbd_variant.setToolTip("自动扫描目录下所有 4 通道 RGBD 数据集（按 yaml 内容判断）")
        self.cb_rgbd_variant.currentTextChanged.connect(self._on_variant_text_changed)
        form.addRow("RGBD数据集:", self.cb_rgbd_variant)

        # 当前选中的 RGBD yaml 完整路径（只读，确保不静默选择）
        self.le_rgbd_yaml = QLineEdit()
        self.le_rgbd_yaml.setReadOnly(True)
        self.le_rgbd_yaml.setText("-")
        form.addRow("当前 RGBD YAML:", self.le_rgbd_yaml)

        # ---- 准备新数据集相关控件 ----
        self.le_rgb = QLineEdit()
        self.le_depth_npy = QLineEdit()
        self.le_label = QLineEdit()
        self.le_out = QLineEdit()
        self.le_class = QLineEdit("hand")
        self.le_low = QLineEdit("1100")
        self.le_high = QLineEdit("1850")
        self.row_rgb = self._make_browse_row(self.le_rgb, is_dir=True)
        form.addRow("RGB目录:", self.row_rgb)
        self.row_depth_npy = self._make_browse_row(self.le_depth_npy, is_dir=True)
        form.addRow("Raw Depth NPY目录:", self.row_depth_npy)
        self.row_label = self._make_browse_row(self.le_label, is_dir=True)
        form.addRow("Labels目录:", self.row_label)
        self.row_out = self._make_browse_row(self.le_out, is_dir=True)
        form.addRow("输出目录:", self.row_out)
        form.addRow("类别名:", self.le_class)
        form.addRow("Depth low(mm):", self.le_low)
        form.addRow("Depth high(mm):", self.le_high)
        self.btn_prepare = QPushButton("准备数据集")
        self.btn_prepare.clicked.connect(self.prepareRequested.emit)
        form.addRow(self.btn_prepare)

        # ---- 数据集统计显示 ----
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

        # 初始应用模式状态
        self._apply_mode_state()

    # ---- 内部：模式切换 ----
    def _on_mode_clicked(self) -> None:
        """模式切换时内部处理 enable/disable，再发出信号供 Controller 反应。"""
        self._apply_mode_state()
        self.modeChanged.emit()

    def _apply_mode_state(self) -> None:
        """根据当前模式启停控件（纯 UI 状态）。"""
        existing = self.rb_existing.isChecked()
        for row in (self.row_rgb, self.row_depth_npy, self.row_label, self.row_out):
            row.setEnabled(not existing)
        self.le_class.setEnabled(not existing)
        self.le_low.setEnabled(not existing)
        self.le_high.setEnabled(not existing)
        self.btn_prepare.setEnabled(not existing)
        for w in (self.row_existing, self.le_rgb_yaml, self.cb_rgbd_variant, self.le_rgbd_yaml):
            w.setEnabled(existing)

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

    def _on_browse_existing(self) -> None:
        """浏览已有数据集根目录。"""
        start = self._resolve_start(self.le_existing)
        p = QFileDialog.getExistingDirectory(self, "选择已有数据集根目录", start)
        if p:
            self.le_existing.setText(p)
            self.settingsDirty.emit()
            self.existingDirChanged.emit()

    # ---- 内部：变体下拉 ----
    def _on_variant_text_changed(self, dir_name: str) -> None:
        """用户从下拉选择变体时发出信号。"""
        if dir_name:
            self.rgbdVariantChanged.emit(dir_name)

    # ---- 公开：变体下拉操作（供 Controller 调用）----
    def populate_variants(self, variants: list[tuple[str, Path]], saved: str | None = None) -> None:
        """填充 RGBD 变体下拉并选择默认项（blockSignals 避免填充期间误触发）。

        variants: list[(dir_name, yaml_path)]
        saved: QSettings 保存的变体名（或 None）
        选择优先级：已保存 > 默认 rgbd_rawdepth > 第一个
        填充完成后手动发出 rgbdVariantChanged 信号。
        """
        cb = self.cb_rgbd_variant
        order = [v[0] for v in variants]
        cb.blockSignals(True)
        cb.clear()
        for dir_name, y in variants:
            cb.addItem(dir_name, str(y))
        preferred = "rgbd_rawdepth"
        if saved in order:
            default_name = saved
        elif preferred in order:
            default_name = preferred
        elif order:
            default_name = order[0]
        else:
            default_name = None
        if default_name:
            cb.setCurrentIndex(order.index(default_name))
        cb.blockSignals(False)
        if default_name:
            self.rgbdVariantChanged.emit(default_name)

    def clear_variants(self) -> None:
        """清空变体下拉（blockSignals 避免误触发）。"""
        self.cb_rgbd_variant.blockSignals(True)
        self.cb_rgbd_variant.clear()
        self.cb_rgbd_variant.blockSignals(False)

    # ---- 公开：展示方法 ----
    def reset_stats(self) -> None:
        """清空统计显示，避免上一套数据集数据残留造成误解。"""
        for lbl in (self.lbl_total, self.lbl_train, self.lbl_val,
                    self.lbl_test, self.lbl_kpt, self.lbl_rgbch, self.lbl_rgbdch,
                    self.lbl_rgb_ds, self.lbl_rgbd_ds):
            lbl.setText("-")

    def set_stats(self, merged: dict[str, Any], ready: str) -> None:
        """写回现场统计/报告合并后的统计 + 状态行（None 值显示"-"）。"""
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

    def set_ready(self, text: str) -> None:
        self.lbl_ready.setText(text)

    def set_rgb_yaml_text(self, text: str) -> None:
        self.le_rgb_yaml.setText(text or "-")

    def set_rgbd_yaml_text(self, text: str) -> None:
        self.le_rgbd_yaml.setText(text or "-")

    def set_rgbd_ds_text(self, text: str) -> None:
        self.lbl_rgbd_ds.setText(text or "-")

    def set_prepare_enabled(self, enabled: bool) -> None:
        """供 Controller 在任务运行期间控制 prepare 按钮启停。"""
        self.btn_prepare.setEnabled(enabled)

    def is_existing_mode(self) -> bool:
        return self.rb_existing.isChecked()

    def is_new_mode(self) -> bool:
        return self.rb_new.isChecked()
