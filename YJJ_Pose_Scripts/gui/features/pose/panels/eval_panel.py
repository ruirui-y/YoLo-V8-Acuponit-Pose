"""测试结果面板（信号驱动，与 MainWindow / Controller 完全解耦）。

Panel 只负责控件创建 / 展示 / 发信号；按钮 clicked 连接到自己的 signal，
Controller 连接这些 signal 并处理业务逻辑。
"""
from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QWidget,
)

from gui_config import _REPO_ROOT


class EvalPanel(QWidget):
    # ---- 对外信号 ----
    evalRgbRequested = Signal()
    evalRgbdRequested = Signal()
    compareRequested = Signal()
    depthAblationRequested = Signal()
    rgbdWeightChanged = Signal()            # RGBD best.pt 路径变更
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
        gt = QGroupBox("测试结果")
        ft = QFormLayout(gt)

        # ---- 权重选择 ----
        self.le_eval_rgb = QLineEdit()
        self.row_eval_rgb = self._make_browse_row(self.le_eval_rgb, is_dir=False, filt="Weights (*.pt)")
        self.le_eval_rgb.editingFinished.connect(self.settingsDirty.emit)
        ft.addRow("RGB best.pt:", self.row_eval_rgb)
        self.le_eval_rgbd = QLineEdit()
        self.row_eval_rgbd = self._make_browse_row(self.le_eval_rgbd, is_dir=False, filt="Weights (*.pt)")
        self.le_eval_rgbd.editingFinished.connect(self.settingsDirty.emit)
        self.le_eval_rgbd.editingFinished.connect(self.rgbdWeightChanged.emit)
        ft.addRow("RGBD best.pt:", self.row_eval_rgbd)

        # ---- 当前测试数据（只读，split 固定为 test）----
        self.lbl_rgb_test_yaml = QLabel("-")
        ft.addRow("RGB Test YAML:", self.lbl_rgb_test_yaml)
        self.lbl_rgb_test_img = QLabel("-")
        ft.addRow("RGB Test images:", self.lbl_rgb_test_img)
        self.lbl_rgb_test_cnt = QLabel("-")
        ft.addRow("RGB Test count:", self.lbl_rgb_test_cnt)
        self.lbl_rgbd_test_yaml = QLabel("-")
        ft.addRow("RGBD Test YAML:", self.lbl_rgbd_test_yaml)
        self.lbl_rgbd_test_img = QLabel("-")
        ft.addRow("RGBD Test images:", self.lbl_rgbd_test_img)
        self.lbl_rgbd_test_cnt = QLabel("-")
        ft.addRow("RGBD Test count:", self.lbl_rgbd_test_cnt)
        self.lbl_split = QLabel("test（固定）")
        ft.addRow("Split:", self.lbl_split)
        self.lbl_id_consistency = QLabel("-")
        ft.addRow("Test ID一致性:", self.lbl_id_consistency)

        # ---- Depth 消融测试 ----
        self.lbl_ablate_title = QLabel("Depth Ablation: True Depth vs Zero Depth")
        ft.addRow(self.lbl_ablate_title)
        self.lbl_ablate_split = QLabel("test（固定）")
        ft.addRow("Split:", self.lbl_ablate_split)
        self.lbl_ablate_yaml = QLabel("-")
        ft.addRow("当前 RGBD YAML:", self.lbl_ablate_yaml)
        self.lbl_ablate_pt = QLabel("-")
        ft.addRow("当前 RGBD best.pt:", self.lbl_ablate_pt)
        self.btn_ablate = QPushButton("Depth 消融测试")
        self.btn_ablate.clicked.connect(self.depthAblationRequested.emit)
        ft.addRow(self.btn_ablate)

        # ---- 动作按钮 ----
        self.btn_eval_rgb = QPushButton("测试 RGB")
        self.btn_eval_rgb.clicked.connect(self.evalRgbRequested.emit)
        self.btn_eval_rgbd = QPushButton("测试 RGBD")
        self.btn_eval_rgbd.clicked.connect(self.evalRgbdRequested.emit)
        self.btn_eval_cmp = QPushButton("对比测试")
        self.btn_eval_cmp.clicked.connect(self.compareRequested.emit)
        heval = QHBoxLayout()
        heval.addWidget(self.btn_eval_rgb)
        heval.addWidget(self.btn_eval_rgbd)
        heval.addWidget(self.btn_eval_cmp)
        ft.addRow(heval)

        # ---- 指标展示块（split=test）----
        self.eval_lbl: dict[str, dict[str, dict[str, QLabel]]] = {"rgb": {}, "rgbd": {}}
        self.eval_lbl["rgb"]["box"] = self._add_metric_block(ft, "RGB Box (split=test)")
        self.eval_lbl["rgb"]["pose"] = self._add_metric_block(ft, "RGB Pose (split=test)")
        self.eval_lbl["rgbd"]["box"] = self._add_metric_block(ft, "RGBD Box (split=test)")
        self.eval_lbl["rgbd"]["pose"] = self._add_metric_block(ft, "RGBD Pose (split=test)")

        # ---- 对比（差值 = RGBD - RGB）----
        self.lbl_cmp_box = QLabel("-")
        self.lbl_cmp_pose = QLabel("-")
        ft.addRow("RGB vs RGBD Box mAP50-95:", self.lbl_cmp_box)
        ft.addRow("RGB vs RGBD Pose mAP50-95:", self.lbl_cmp_pose)

        v = QVBoxLayout(self)
        v.addWidget(gt)

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

    def _add_metric_block(self, parent: QFormLayout, title: str) -> dict[str, QLabel]:
        """在 parent(QFormLayout) 下新增一个指标分组，返回 {"p","r","map50","map"} 四个 QLabel。"""
        gb = QGroupBox(title)
        f = QFormLayout(gb)
        lbl: dict[str, QLabel] = {}
        for key, disp in (("p", "P"), ("r", "R"), ("map50", "mAP50"), ("map", "mAP50-95")):
            l = QLabel("-")
            lbl[key] = l
            f.addRow(disp + ":", l)
        parent.addRow(gb)
        return lbl

    # ---- 公开：展示方法 ----
    def display_result(self, leg: str, res: dict[str, Any]) -> None:
        """写回单个 leg（rgb/rgbd）的 Box/Pose 8 项指标。"""
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
        """写回对比结果：差值 = RGBD - RGB。"""
        self.lbl_cmp_box.setText(
            f"RGB={r['box_map']:.4f}  RGBD={d['box_map']:.4f}  差值(RGBD-RGB)={db_box:+.4f}")
        self.lbl_cmp_pose.setText(
            f"RGB={r['pose_map']:.4f}  RGBD={d['pose_map']:.4f}  差值(RGBD-RGB)={db_pose:+.4f}")

    def clear_results(self) -> None:
        """清空所有指标展示。"""
        for leg in ("rgb", "rgbd"):
            for kind in ("box", "pose"):
                for l in self.eval_lbl[leg][kind].values():
                    l.setText("-")
        self.lbl_cmp_box.setText("-")
        self.lbl_cmp_pose.setText("-")

    def set_test_info(self, info: dict[str, Any]) -> None:
        """写回"当前测试数据"只读区（split 固定 test）。"""
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

    def set_ablate_info(self, info: dict[str, Any]) -> None:
        """写回"Depth 消融测试"只读信息。"""
        self.lbl_ablate_yaml.setText(info.get("rgbd_yaml") or "-")
        self.lbl_ablate_pt.setText(info.get("rgbd_pt") or "-")

    def set_operation_buttons_enabled(self, enabled: bool) -> None:
        """供 Controller 在任务运行期间控制评估/消融按钮启停。"""
        self.btn_eval_rgb.setEnabled(enabled)
        self.btn_eval_rgbd.setEnabled(enabled)
        self.btn_eval_cmp.setEnabled(enabled)
        self.btn_ablate.setEnabled(enabled)
