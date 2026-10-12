"""公共测试集筛选页面。

用途：把「所有参与对比的权重都能命中」的测试图片筛出来并导出到新目录，
作为后续统一口径（公共集全量）的评估测试集。

本 Panel 只提供 UI 与信号；路径校验、命令构造、子进程执行由 Controller 负责。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton,
    QVBoxLayout, QWidget,
)

from gui_config import _REPO_ROOT


class FilterPanel(QWidget):
    """基于权重命中的公共测试集筛选：选权重 → 求交集 → 导出。"""

    filterRequested = Signal()
    settingsDirty = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        start_dir_provider: Callable[[QLineEdit], str] | None = None,
    ) -> None:
        super().__init__(parent)
        self._start_dir_provider = start_dir_provider
        self._result_dir: Path | None = None
        self._build_ui()

    def set_start_dir_provider(self, provider: Callable[[QLineEdit], str] | None) -> None:
        self._start_dir_provider = provider

    # ================================================================ UI
    @staticmethod
    def _hint(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        return label

    @staticmethod
    def _wrap(layout: QHBoxLayout) -> QWidget:
        box = QWidget()
        box.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        return box

    def _build_ui(self) -> None:
        group = QGroupBox("公共测试集筛选（全部权重共同命中）")
        form = QFormLayout(group)

        # ---- 源测试集 ----
        self.le_filter_dataset = QLineEdit()
        self.le_filter_dataset.setPlaceholderText(
            "源数据集根目录（含 rgb/ 与 rgbd/），如 hand_cross_subject")
        self.row_filter_dataset = self._make_browse_row(self.le_filter_dataset, is_dir=True)
        form.addRow("源测试集目录:", self.row_filter_dataset)
        self.lbl_filter_dataset_info = self._hint("-")
        self.lbl_filter_dataset_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("源数据状态:", self.lbl_filter_dataset_info)

        # ---- 权重列表（可增删）----
        self.lst_weights = QListWidget()
        self.lst_weights.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_weights.setMinimumHeight(112)
        self.lst_weights.setAlternatingRowColors(True)
        form.addRow("参与筛选的权重:", self.lst_weights)

        weight_buttons = QHBoxLayout()
        self.btn_add_weight = QPushButton("+ 添加权重")
        self.btn_add_weight.clicked.connect(self._add_weights)
        self.btn_remove_weight = QPushButton("删除选中")
        self.btn_remove_weight.clicked.connect(self._remove_selected)
        self.btn_clear_weights = QPushButton("清空")
        self.btn_clear_weights.clicked.connect(self._clear_weights)
        for btn in (self.btn_add_weight, self.btn_remove_weight, self.btn_clear_weights):
            weight_buttons.addWidget(btn)
        form.addRow("", self._wrap(weight_buttons))
        form.addRow(self._hint(
            "3 通道权重（RGB）读 rgb/，4 通道权重（RGBD）读 rgbd/；"
            "按文件 stem 对齐后取「全部权重都命中」的图片。"))

        # ---- 输出目录 ----
        self.le_filter_out = QLineEdit()
        self.le_filter_out.setPlaceholderText("公共测试集输出目录（将新建 rgb/ 与 rgbd/）")
        self.row_filter_out = self._make_browse_row(self.le_filter_out, is_dir=True)
        form.addRow("输出目录:", self.row_filter_out)

        self.le_filter_min = QLineEdit()
        self.le_filter_min.setPlaceholderText("0 = 不告警；例如 80")
        form.addRow("公共集下限告警:", self.le_filter_min)

        form.addRow(self._hint(
            "判定口径复用 eval_rgb_vs_rgbd_nme.py（IoU≥0.50 / conf=0.001 / imgsz=640）。"
            "指定的 Python 解释器需能加载全部所选权重（含 YOLO26 需 ultralytics ≥ 8.4）。"))

        # ---- 执行 ----
        self.btn_filter = QPushButton("开始筛选")
        self.btn_filter.clicked.connect(self.filterRequested.emit)
        form.addRow(self.btn_filter)

        # ---- 结果 ----
        self.lbl_filter_weights = QLabel("-")
        self.lbl_filter_common = QLabel("-")
        self.lbl_filter_result_dir = self._hint("-")
        for label in (self.lbl_filter_weights, self.lbl_filter_common, self.lbl_filter_result_dir):
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.setWordWrap(True)
        form.addRow("各权重命中:", self.lbl_filter_weights)
        form.addRow("公共集:", self.lbl_filter_common)
        form.addRow("结果目录:", self.lbl_filter_result_dir)
        self.btn_filter_open = QPushButton("打开结果目录")
        self.btn_filter_open.setEnabled(False)
        self.btn_filter_open.clicked.connect(self._open_result)
        form.addRow(self.btn_filter_open)

        layout = QVBoxLayout(self)
        layout.addWidget(group)

        for le in (self.le_filter_dataset, self.le_filter_out):
            le.editingFinished.connect(self.settingsDirty.emit)
        self.clear_filter_results()

    # ================================================================ 浏览
    def _make_browse_row(self, le: QLineEdit, is_dir: bool = False,
                         filt: str = "All (*)") -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(le)
        btn = QPushButton("选择...")
        btn.clicked.connect(lambda: self._browse(le, is_dir, filt))
        h.addWidget(btn)
        return row

    def _resolve_start(self, le: QLineEdit) -> str:
        cur = le.text().strip()
        if cur and Path(cur).exists():
            path = Path(cur)
            return str(path if path.is_dir() else path.parent)
        if self._start_dir_provider is not None:
            result = self._start_dir_provider(le)
            if result:
                return result
        return str(_REPO_ROOT)

    def _browse(self, le: QLineEdit, is_dir: bool, filt: str) -> None:
        start = self._resolve_start(le)
        if is_dir:
            chosen = QFileDialog.getExistingDirectory(self, "选择目录", start)
        else:
            chosen, _ = QFileDialog.getOpenFileName(self, "选择文件", start, filt)
        if chosen:
            le.setText(chosen)
            self.settingsDirty.emit()

    # ================================================================ 权重列表
    def _add_weights(self) -> None:
        start = self._resolve_start(self.le_filter_dataset)
        files, _ = QFileDialog.getOpenFileNames(self, "选择权重（可多选）", start,
                                                "Weights (*.pt)")
        if not files:
            return
        existing = set(self.weights())
        for path in files:
            if path not in existing:
                self._append_weight(path)
                existing.add(path)
        self.settingsDirty.emit()

    def _append_weight(self, path: str) -> None:
        item = QListWidgetItem(path)
        item.setToolTip(path)
        self.lst_weights.addItem(item)

    def _remove_selected(self) -> None:
        for item in self.lst_weights.selectedItems():
            self.lst_weights.takeItem(self.lst_weights.row(item))
        self.settingsDirty.emit()

    def _clear_weights(self) -> None:
        self.lst_weights.clear()
        self.settingsDirty.emit()

    def weights(self) -> list[str]:
        return [self.lst_weights.item(i).text() for i in range(self.lst_weights.count())]

    def set_weights(self, paths: list[str]) -> None:
        self.lst_weights.clear()
        for path in paths:
            self._append_weight(path)

    # ================================================================ 读写字段
    def dataset_root(self) -> str:
        return self.le_filter_dataset.text().strip()

    def set_dataset_root(self, text: str) -> None:
        self.le_filter_dataset.setText(text)

    def output_dir(self) -> str:
        return self.le_filter_out.text().strip()

    def set_output_dir(self, text: str) -> None:
        self.le_filter_out.setText(text)

    def min_common(self) -> int:
        text = self.le_filter_min.text().strip()
        try:
            return int(text) if text else 0
        except ValueError:
            return 0

    def set_min_common(self, value: int) -> None:
        self.le_filter_min.setText(str(value) if value else "")

    def set_dataset_info(self, text: str) -> None:
        self.lbl_filter_dataset_info.setText(text or "-")

    # ================================================================ 结果
    def display_result(self, summary: dict[str, Any], result_dir: Path) -> None:
        weights = summary.get("weights", [])
        total = summary.get("total_stems", 0)
        parts = [
            f"{w.get('name', '?')}={w.get('matched', 0)}/{total}"
            for w in weights
        ]
        self.lbl_filter_weights.setText("，".join(parts) if parts else "-")
        self.lbl_filter_common.setText(
            f"{summary.get('common_count', 0)} / {total} 张"
            f"（剔除 {summary.get('dropped_count', 0)} 张）")
        self._result_dir = Path(result_dir)
        self.lbl_filter_result_dir.setText(str(result_dir))
        self.lbl_filter_result_dir.setToolTip(
            "包含 rgb/、rgbd/、filter_report.csv、filter_summary.json、data_*.yaml")
        self.btn_filter_open.setEnabled(True)

    def clear_filter_results(self, *_args: object) -> None:
        self.lbl_filter_weights.setText("-")
        self.lbl_filter_common.setText("-")
        self.lbl_filter_result_dir.setText("-")
        self._result_dir = None
        self.btn_filter_open.setEnabled(False)

    def _open_result(self) -> None:
        if self._result_dir is not None and self._result_dir.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._result_dir)))

    # ================================================================ 忙碌态
    def set_operation_buttons_enabled(self, enabled: bool) -> None:
        for btn in (self.btn_filter, self.btn_add_weight,
                    self.btn_remove_weight, self.btn_clear_weights):
            btn.setEnabled(enabled)
        for widget in (self.row_filter_dataset, self.row_filter_out,
                       self.lst_weights, self.le_filter_min):
            widget.setEnabled(enabled)
