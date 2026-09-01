"""测试结果面板（test 集评估：RGB vs RGBD 论文对比，纯 UI，不启动子进程）。

包含：RGB best.pt / RGBD best.pt 选择、测试 RGB / 测试 RGBD / 对比测试 按钮、
RGB 与 RGBD 的 Box/Pose 指标展示块、对比结果展示。

按钮的 clicked 信号连接到 MainWindow 的对应动作；本面板只负责控件与展示：
- display_result(leg, res)：写回单条 leg 的 Box/Pose 8 项指标
- display_compare(r, d, db_box, db_pose)：写回对比差值
- clear_results()：清空所有指标显示
"""
from PySide6.QtWidgets import (
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QWidget,
)


class EvalPanel(QWidget):
    def __init__(self, main):
        super().__init__()
        # ---- 测试结果（test 集评估：RGB vs RGBD 论文对比）----
        gt = QGroupBox("测试结果")
        ft = QFormLayout(gt)
        # 权重选择（默认尝试自动定位本次训练产物，可手动改）
        self.le_eval_rgb = QLineEdit()
        self.row_eval_rgb = main._with_browse(self.le_eval_rgb, dir=False, filt="Weights (*.pt)")
        self.le_eval_rgb.editingFinished.connect(main._save_settings)
        ft.addRow("RGB best.pt:", self.row_eval_rgb)
        self.le_eval_rgbd = QLineEdit()
        self.row_eval_rgbd = main._with_browse(self.le_eval_rgbd, dir=False, filt="Weights (*.pt)")
        self.le_eval_rgbd.editingFinished.connect(main._save_settings)
        ft.addRow("RGBD best.pt:", self.row_eval_rgbd)
        # 动作按钮
        self.btn_eval_rgb = QPushButton("测试 RGB")
        self.btn_eval_rgb.clicked.connect(main._on_eval_rgb)
        self.btn_eval_rgbd = QPushButton("测试 RGBD")
        self.btn_eval_rgbd.clicked.connect(main._on_eval_rgbd)
        self.btn_eval_cmp = QPushButton("对比测试")
        self.btn_eval_cmp.clicked.connect(main._on_eval_cmp)
        heval = QHBoxLayout()
        heval.addWidget(self.btn_eval_rgb)
        heval.addWidget(self.btn_eval_rgbd)
        heval.addWidget(self.btn_eval_cmp)
        ft.addRow(heval)
        # 指标展示块（split=test）
        self.eval_lbl = {"rgb": {}, "rgbd": {}}
        self.eval_lbl["rgb"]["box"] = self._add_metric_block(ft, "RGB Box (split=test)")
        self.eval_lbl["rgb"]["pose"] = self._add_metric_block(ft, "RGB Pose (split=test)")
        self.eval_lbl["rgbd"]["box"] = self._add_metric_block(ft, "RGBD Box (split=test)")
        self.eval_lbl["rgbd"]["pose"] = self._add_metric_block(ft, "RGBD Pose (split=test)")
        # 对比（差值 = RGBD - RGB）
        self.lbl_cmp_box = QLabel("-")
        self.lbl_cmp_pose = QLabel("-")
        ft.addRow("RGB vs RGBD Box mAP50-95:", self.lbl_cmp_box)
        ft.addRow("RGB vs RGBD Pose mAP50-95:", self.lbl_cmp_pose)

        v = QVBoxLayout(self)
        v.addWidget(gt)

    # ---- 纯展示方法 ----
    def _add_metric_block(self, parent, title):
        """在 parent(QFormLayout) 下新增一个指标分组，返回 {"p","r","map50","map"} 四个 QLabel。"""
        gb = QGroupBox(title)
        f = QFormLayout(gb)
        lbl = {}
        for key, disp in (("p", "P"), ("r", "R"), ("map50", "mAP50"), ("map", "mAP50-95")):
            l = QLabel("-")
            lbl[key] = l
            f.addRow(disp + ":", l)
        parent.addRow(gb)
        return lbl

    def display_result(self, leg, res):
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

    def display_compare(self, r, d, db_box, db_pose):
        """写回对比结果：差值 = RGBD - RGB。"""
        self.lbl_cmp_box.setText(
            f"RGB={r['box_map']:.4f}  RGBD={d['box_map']:.4f}  差值(RGBD-RGB)={db_box:+.4f}")
        self.lbl_cmp_pose.setText(
            f"RGB={r['pose_map']:.4f}  RGBD={d['pose_map']:.4f}  差值(RGBD-RGB)={db_pose:+.4f}")

    def clear_results(self):
        """清空所有指标展示（保留上次结果直到下一次评估覆盖；此方法供需要时调用）。"""
        for leg in ("rgb", "rgbd"):
            for kind in ("box", "pose"):
                for l in self.eval_lbl[leg][kind].values():
                    l.setText("-")
        self.lbl_cmp_box.setText("-")
        self.lbl_cmp_pose.setText("-")
