"""统一测试页面：顶部一次指定模型/数据集，下方按 mAP、NME、Depth 消融切换。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from gui_config import _REPO_ROOT


class EvalPanel(QWidget):
    # 仅提供 UI 和信号；Controller 决定实际使用的路径及业务逻辑。
    evalRgbRequested = Signal()
    evalRgbdRequested = Signal()
    compareRequested = Signal()
    depthAblationRequested = Signal()
    nmeRequested = Signal()
    evalDatasetChanged = Signal()         # 统一测试数据集发生变更
    rgbdWeightChanged = Signal()
    settingsDirty = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        start_dir_provider: Callable[[QLineEdit], str] | None = None,
    ) -> None:
        super().__init__(parent)
        self._start_dir_provider = start_dir_provider
        self._nme_result_dir: Path | None = None
        self._build_ui()

    def set_start_dir_provider(self, provider: Callable[[QLineEdit], str] | None) -> None:
        self._start_dir_provider = provider

    @staticmethod
    def _hint(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        return label

    def _build_ui(self) -> None:
        # ---- 统一评估配置：所有测试都使用这里的根目录和权重 ----
        config_group = QGroupBox("1. 测试配置（mAP / NME / Depth 消融共用）")
        config_form = QFormLayout(config_group)

        self.le_eval_dataset = QLineEdit()
        self.le_eval_dataset.setPlaceholderText("选择包含 rgb/ 和 rgbd/ 的数据集，如 hand_cross_subject")
        self.row_eval_dataset = self._make_browse_row(self.le_eval_dataset, is_dir=True)
        config_form.addRow("测试数据集目录:", self.row_eval_dataset)

        self.le_eval_rgb = QLineEdit()
        self.row_eval_rgb = self._make_browse_row(self.le_eval_rgb, filt="Weights (*.pt)")
        config_form.addRow("RGB 模型 (best.pt):", self.row_eval_rgb)

        self.le_eval_rgbd = QLineEdit()
        self.row_eval_rgbd = self._make_browse_row(self.le_eval_rgbd, filt="Weights (*.pt)")
        config_form.addRow("RGBD 模型 (best.pt):", self.row_eval_rgbd)

        config_form.addRow(self._hint(
            "只测试所选数据集的 test。这里选择的数据集独立于上方“数据集准备”页面，"
            "不会因切换准备/已有数据集而偷偷改变。"))

        self.lbl_test_summary = self._hint("请选择测试数据集目录")
        self.lbl_test_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        config_form.addRow("当前测试数据:", self.lbl_test_summary)
        self.lbl_split = QLabel("test（固定）")

        self.btn_show_eval_paths = QPushButton("查看实际 YAML / test 图片路径 ▸")
        self.btn_show_eval_paths.setCheckable(True)
        self.btn_show_eval_paths.toggled.connect(self._toggle_eval_paths)
        config_form.addRow(self.btn_show_eval_paths)

        self.eval_paths_box = QWidget()
        path_form = QFormLayout(self.eval_paths_box)
        path_form.setContentsMargins(0, 0, 0, 0)
        self.lbl_rgb_test_yaml = self._hint("-")
        self.lbl_rgb_test_img = self._hint("-")
        self.lbl_rgb_test_cnt = QLabel("-")
        self.lbl_rgbd_test_yaml = self._hint("-")
        self.lbl_rgbd_test_img = self._hint("-")
        self.lbl_rgbd_test_cnt = QLabel("-")
        self.lbl_id_consistency = QLabel("-")
        for label, value in (
            ("RGB YAML:", self.lbl_rgb_test_yaml),
            ("RGB test 图片:", self.lbl_rgb_test_img),
            ("RGB 数量:", self.lbl_rgb_test_cnt),
            ("RGBD YAML:", self.lbl_rgbd_test_yaml),
            ("RGBD test 图片:", self.lbl_rgbd_test_img),
            ("RGBD 数量:", self.lbl_rgbd_test_cnt),
            ("ID 一致性:", self.lbl_id_consistency),
        ):
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            path_form.addRow(label, value)
        self.eval_paths_box.setVisible(False)
        config_form.addRow(self.eval_paths_box)

        # ---- 三种测试放在 Tab，避免上下堆叠而找不到设置 ----
        self.tabs = QTabWidget()
        self._build_map_tab()
        self._build_nme_tab()
        self._build_ablate_tab()

        layout = QVBoxLayout(self)
        layout.addWidget(config_group)
        layout.addWidget(self.tabs)

        # 兼容已有 Controller 属性；本质上是同一个 QLineEdit，绝不再出现第二套数据集。
        self.le_nme_dataset = self.le_eval_dataset
        self.row_nme_dataset = self.row_eval_dataset

        self.le_eval_dataset.textChanged.connect(self._on_shared_config_changed)
        self.le_eval_dataset.textChanged.connect(lambda _text: self.evalDatasetChanged.emit())
        self.le_eval_dataset.editingFinished.connect(self.settingsDirty.emit)
        for le in (self.le_eval_rgb, self.le_eval_rgbd):
            le.textChanged.connect(self._on_shared_config_changed)
            le.editingFinished.connect(self.settingsDirty.emit)
        self.le_eval_rgbd.textChanged.connect(lambda _text: self.rgbdWeightChanged.emit())
        self.clear_results()
        self.clear_nme_results()

    def _toggle_eval_paths(self, shown: bool) -> None:
        self.eval_paths_box.setVisible(shown)
        self.btn_show_eval_paths.setText(
            "收起实际路径 ▾" if shown else "查看实际 YAML / test 图片路径 ▸")

    def _build_map_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(self._hint(
            "官方 Box / Pose 指标：P、R、mAP50、mAP50-95。"
            "两个模型均使用上方所选数据集的 test。"))

        buttons = QHBoxLayout()
        self.btn_eval_rgb = QPushButton("只测 RGB")
        self.btn_eval_rgb.clicked.connect(self.evalRgbRequested.emit)
        self.btn_eval_rgbd = QPushButton("只测 RGBD")
        self.btn_eval_rgbd.clicked.connect(self.evalRgbdRequested.emit)
        self.btn_eval_cmp = QPushButton("RGB / RGBD 对比测试")
        self.btn_eval_cmp.clicked.connect(self.compareRequested.emit)
        for b in (self.btn_eval_rgb, self.btn_eval_rgbd, self.btn_eval_cmp):
            buttons.addWidget(b)
        layout.addLayout(buttons)

        metric_grid = QGridLayout()
        metric_grid.setHorizontalSpacing(15)
        for c, name in enumerate(("指标", "P", "R", "mAP50", "mAP50-95")):
            header = QLabel(name)
            header.setAlignment(Qt.AlignCenter)
            metric_grid.addWidget(header, 0, c)
        self.eval_lbl: dict[str, dict[str, dict[str, QLabel]]] = {"rgb": {}, "rgbd": {}}
        for r, (leg, kind, name) in enumerate((
            ("rgb", "box", "RGB Box"),
            ("rgb", "pose", "RGB Pose"),
            ("rgbd", "box", "RGBD Box"),
            ("rgbd", "pose", "RGBD Pose"),
        ), start=1):
            title = QLabel(name)
            metric_grid.addWidget(title, r, 0)
            self.eval_lbl[leg][kind] = {}
            for c, key in enumerate(("p", "r", "map50", "map"), start=1):
                label = QLabel("-")
                label.setAlignment(Qt.AlignCenter)
                metric_grid.addWidget(label, r, c)
                self.eval_lbl[leg][kind][key] = label
        for c in range(5):
            metric_grid.setColumnStretch(c, 1)
        layout.addLayout(metric_grid)
        self.lbl_cmp_box = self._hint("-")
        self.lbl_cmp_pose = self._hint("-")
        layout.addWidget(self.lbl_cmp_box)
        layout.addWidget(self.lbl_cmp_pose)
        layout.addStretch(1)
        self.tabs.addTab(tab, "mAP 官方指标")

    def _build_nme_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)
        form.addRow(self._hint(
            "模型和测试数据集使用上面的统一配置。"
            "IoU≥0.50，conf=0.001；NME 越低越好（按 GT 框对角线归一化）。"))

        self.le_nme_depth = QLineEdit()
        self.le_nme_depth.setPlaceholderText("可留空；仅用于按距离统计，不影响模型的四通道输入")
        self.row_nme_depth = self._make_browse_row(self.le_nme_depth, is_dir=True)
        form.addRow("真实 Depth NPY (可选):", self.row_nme_depth)
        self.le_nme_depth.textChanged.connect(self.clear_nme_results)
        self.le_nme_depth.editingFinished.connect(self.settingsDirty.emit)
        form.addRow(self._hint("RGBD 的第四通道直接从所选数据集 rgbd/images/test 中读取。"
                                "这里的 NPY 路径只用于深度距离分组，不会替换图片输入。"))

        self.btn_nme = QPushButton("开始 RGB vs RGBD NME 对比")
        self.btn_nme.clicked.connect(self.nmeRequested.emit)
        form.addRow(self.btn_nme)
        self.lbl_nme_coverage = QLabel("-")
        self.lbl_nme_nmes = QLabel("-")
        self.lbl_nme_wins = QLabel("-")
        for label in (self.lbl_nme_coverage, self.lbl_nme_nmes, self.lbl_nme_wins):
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.setWordWrap(True)
        form.addRow("匹配数量:", self.lbl_nme_coverage)
        form.addRow("平均 NME:", self.lbl_nme_nmes)
        form.addRow("胜负统计:", self.lbl_nme_wins)

        self.nme_table = QTableWidget(7, 6)
        self.nme_table.setHorizontalHeaderLabels(
            ["关键点", "RGB px", "RGBD px", "差值 px", "RGBD 胜", "RGB 胜"])
        self.nme_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.nme_table.verticalHeader().hide()
        self.nme_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.nme_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.nme_table.setMinimumHeight(245)
        self.nme_table.setAlternatingRowColors(True)
        form.addRow("K0～K6 误差:", self.nme_table)

        self.lbl_nme_files = self._hint("-")
        self.lbl_nme_files.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("结果目录:", self.lbl_nme_files)
        self.btn_nme_open = QPushButton("打开结果目录")
        self.btn_nme_open.setEnabled(False)
        self.btn_nme_open.clicked.connect(self._open_nme_results)
        form.addRow(self.btn_nme_open)
        self.tabs.addTab(tab, "NME 定位误差")

    def _build_ablate_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(self._hint("Depth 消融：True Depth 与 Zero Depth。"
                                     "使用上方选中的 RGBD best.pt 及同一测试数据集的 test。"))
        self.lbl_ablate_title = QLabel("True Depth vs Zero Depth")
        self.lbl_ablate_split = QLabel("test（固定）")
        self.lbl_ablate_yaml = QLabel("-")
        self.lbl_ablate_pt = QLabel("-")
        self.lbl_ablate_info = self._hint("-")
        layout.addWidget(self.lbl_ablate_info)
        self.btn_ablate = QPushButton("开始 Depth 消融测试")
        self.btn_ablate.clicked.connect(self.depthAblationRequested.emit)
        layout.addWidget(self.btn_ablate)
        layout.addStretch(1)
        self.tabs.addTab(tab, "Depth 消融")

    def _on_shared_config_changed(self, *_args: object) -> None:
        """模型或数据集改动立即清空旧结果，防止误把前一次统计当成当前测试。"""
        self.clear_results()
        self.clear_nme_results()

    def _make_browse_row(self, le: QLineEdit, is_dir: bool = False, filt: str = "All (*)") -> QWidget:
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
            p = Path(cur)
            return str(p if p.is_dir() else p.parent)
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
            chosen, _ = QFileDialog.getOpenFileName(self, "选择权重", start, filt)
        if chosen:
            le.setText(chosen)
            self.settingsDirty.emit()

    def display_result(self, leg: str, res: dict[str, Any]) -> None:
        blk = self.eval_lbl[leg]
        blk["box"]["p"].setText(f"{res['box_p']:.4f}")
        blk["box"]["r"].setText(f"{res['box_r']:.4f}")
        blk["box"]["map50"].setText(f"{res['box_map50']:.4f}")
        blk["box"]["map"].setText(f"{res['box_map']:.4f}")
        blk["pose"]["p"].setText(f"{res['pose_p']:.4f}")
        blk["pose"]["r"].setText(f"{res['pose_r']:.4f}")
        blk["pose"]["map50"].setText(f"{res['pose_map50']:.4f}")
        blk["pose"]["map"].setText(f"{res['pose_map']:.4f}")

    def display_compare(self, r: dict[str, Any], d: dict[str, Any], db_box: float, db_pose: float) -> None:
        self.lbl_cmp_box.setText(
            f"Box mAP50-95：RGB={r['box_map']:.4f}，RGBD={d['box_map']:.4f}，"
            f"RGBD-RGB={db_box:+.4f}")
        self.lbl_cmp_pose.setText(
            f"Pose mAP50-95：RGB={r['pose_map']:.4f}，RGBD={d['pose_map']:.4f}，"
            f"RGBD-RGB={db_pose:+.4f}")

    def clear_results(self) -> None:
        for leg in ("rgb", "rgbd"):
            for kind in ("box", "pose"):
                for label in self.eval_lbl[leg][kind].values():
                    label.setText("-")
        self.lbl_cmp_box.setText("-")
        self.lbl_cmp_pose.setText("-")

    def set_test_info(self, info: dict[str, Any]) -> None:
        self.lbl_rgb_test_yaml.setText(info.get("rgb_yaml") or "-")
        self.lbl_rgb_test_img.setText(info.get("rgb_img") or "-")
        self.lbl_rgb_test_cnt.setText(
            str(info["rgb_cnt"]) if info.get("rgb_cnt") is not None else "-")
        self.lbl_rgbd_test_yaml.setText(info.get("rgbd_yaml") or "-")
        self.lbl_rgbd_test_img.setText(info.get("rgbd_img") or "-")
        self.lbl_rgbd_test_cnt.setText(
            str(info["rgbd_cnt"]) if info.get("rgbd_cnt") is not None else "-")
        self.lbl_split.setText(info.get("split") or "test（固定）")
        self.lbl_id_consistency.setText(info.get("id_text") or "-")
        self.lbl_test_summary.setText(info.get("summary") or "请选择测试数据集目录")

    def set_ablate_info(self, info: dict[str, Any]) -> None:
        yaml = info.get("rgbd_yaml") or ""
        pt = info.get("rgbd_pt") or ""
        self.lbl_ablate_yaml.setText(str(yaml) or "-")
        self.lbl_ablate_pt.setText(str(pt) or "-")
        self.lbl_ablate_info.setText(
            f"使用 RGBD 模型：{Path(pt).name if pt else '未选择'}；"
            f"数据集：{Path(yaml).parent.parent.name if yaml else '未选择'} / test")
        self.lbl_ablate_info.setToolTip(f"模型: {pt or '-'}\nYAML: {yaml or '-'}")

    @staticmethod
    def _fmt_nme(value: float | None, digits: int = 6) -> str:
        return "-" if value is None else f"{value:.{digits}f}"

    def clear_nme_results(self, *_args: object) -> None:
        self.lbl_nme_coverage.setText("-")
        self.lbl_nme_nmes.setText("-")
        self.lbl_nme_wins.setText("-")
        self.lbl_nme_files.setText("-")
        self.btn_nme_open.setEnabled(False)
        self._nme_result_dir = None
        self.nme_table.setRowCount(7)
        for row in range(7):
            for col, value in enumerate((f"K{row}", "-", "-", "-", "-", "-")):
                self.nme_table.setItem(row, col, QTableWidgetItem(value))

    def display_nme_result(self, summary: dict[str, Any], result_dir: Path) -> None:
        self.lbl_nme_coverage.setText(
            f"总数={summary['total']}，RGB={summary['rgb_matched']}，"
            f"RGBD={summary['rgbd_matched']}，共同={summary['common']}，"
            f"RGB-only={summary['rgb_only']}，RGBD-only={summary['rgbd_only']}，"
            f"Neither={summary['neither']}")
        self.lbl_nme_nmes.setText(
            f"RGB={self._fmt_nme(summary['rgb_nme'])}，"
            f"RGBD={self._fmt_nme(summary['rgbd_nme'])}，"
            f"差值(RGBD-RGB)={self._fmt_nme(summary['delta_nme'])}")
        self.lbl_nme_wins.setText(
            f"RGBD={summary['rgbd_wins']}，RGB={summary['rgb_wins']}，"
            f"平局={summary['ties']}（只统计共同匹配）")
        kpoints = summary["keypoints"]
        self.nme_table.setRowCount(len(kpoints))
        for row, kp in enumerate(kpoints):
            r, d = kp["rgb_mean"], kp["rgbd_mean"]
            delta = d - r if r is not None and d is not None else None
            values = (
                f"K{kp['keypoint']}", self._fmt_nme(r, 2),
                self._fmt_nme(d, 2), self._fmt_nme(delta, 2),
                f"{kp['rgbd_wins']}/{kp['count']}", f"{kp['rgb_wins']}/{kp['count']}",
            )
            for col, text in enumerate(values):
                self.nme_table.setItem(row, col, QTableWidgetItem(text))
        self._nme_result_dir = Path(result_dir)
        self.lbl_nme_files.setText(str(result_dir))
        self.lbl_nme_files.setToolTip("包含 per_image.csv、rgbd_better.csv、visualizations/、run_config.json")
        self.btn_nme_open.setEnabled(True)

    def _open_nme_results(self) -> None:
        if self._nme_result_dir is not None and self._nme_result_dir.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._nme_result_dir)))

    def set_operation_buttons_enabled(self, enabled: bool) -> None:
        for b in (self.btn_eval_rgb, self.btn_eval_rgbd, self.btn_eval_cmp,
                  self.btn_ablate, self.btn_nme):
            b.setEnabled(enabled)
        for w in (self.row_eval_dataset, self.row_eval_rgb, self.row_eval_rgbd,
                  self.row_nme_depth):
            w.setEnabled(enabled)
