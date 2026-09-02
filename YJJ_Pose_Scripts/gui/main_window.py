"""
YJJ RGBD Pose 实验桌面 GUI 主控窗口 (PySide6)

设计要点：
- 所有外部操作（prepare / train / build weights / eval）均通过 QProcess 异步执行，
  不阻塞 GUI 主线程；stdout / stderr 实时回显到底部日志框。
- 运行期间禁用所有操作按钮，防止重复启动。
- 不修改现有训练 / 数据处理核心脚本，只调用它们：
    YJJ_Pose_Scripts/01_data/prepare_pose_rgbd_dataset.py
    YJJ_Pose_Scripts/02_model/build_rgbd_pose_weights.py
    YJJ_Pose_Scripts/03_train/train_rgb_pose.py
    YJJ_Pose_Scripts/03_train/train_rgbd_pose.py
    YJJ_Pose_Scripts/04_eval/test_pose_rgbd.py
- 子进程使用“Python训练环境”选择框指定的解释器（默认回退到启动 GUI 的 Python），
  因此该训练环境需具备 ultralytics + torch + opencv；GUI 自身只需 PySide6。
- 启动子进程时：working directory 设为项目根，并在 PYTHONPATH 前置项目根，
  使项目根下的本地 ultralytics 可被 from ultralytics import YOLO 找到。

本模块只负责“协调”：创建各 Panel、装配布局、QSettings 持久化、rgb_yaml/rgbd_yaml、
QProcess、prepare/train/build/eval 各动作、stop/terminate/kill、eval chain、
stdout/stderr 解析、控件启停协调。各 Panel 仅持有控件与纯展示方法。
"""
import json
import os
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QSettings, Qt, QTimer
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLineEdit, QMainWindow, QPushButton, QScrollArea,
    QSplitter, QVBoxLayout, QWidget,
)

from gui_config import (_DEFAULT_BASE_WEIGHT, _FOURCH_DIR, _REPO_ROOT, _SCRIPTS)
from panels.dataset_panel import DatasetPanel
from panels.eval_panel import EvalPanel
from panels.status_log_panel import StatusLogPanel
from panels.train_panel import TrainPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YJJ RGBD Pose 训练 GUI")
        self.resize(1200, 820)

        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.readyReadStandardError.connect(self._on_stderr)
        self.process.started.connect(self._on_started)
        self.process.finished.connect(self._on_finished)
        self.current_op = None  # prepare / train_rgb / build / train_rgbd / eval_*
        self.user_stopped = False  # 用户主动点击“停止当前任务”标记
        # 评估（test 集）状态
        self.eval_results = {}        # leg("rgb"/"rgbd") -> 解析到的 metrics dict
        self.eval_chain = []          # 对比测试剩余 leg 队列
        self._compare_pending = False
        self._last_eval_result = None
        self._last_ablate_result = None

        # 统一的数据集 yaml 路径：训练按钮始终从这里取
        self.rgb_yaml = None
        self.rgbd_yaml = None
        # 4ch 变体扫描结果缓存（dir_name -> yaml_path）与填充守卫
        self._rgbd_variants = []
        self._populating = False

        # 上一次路径持久化（QSettings，下次启动自动恢复）
        self._settings = QSettings("YJJ", "RGBDPoseTrainGUI")
        self._settings_keys = {}  # le(widget) -> settings key

        # 创建各面板（面板构造期间会把按钮/编辑信号连接到本类方法）
        self.dataset_panel = DatasetPanel(self)
        self.train_panel = TrainPanel(self)
        self.eval_panel = EvalPanel(self)
        self.status_log_panel = StatusLogPanel(self)

        # 路径 <-> QSettings key 映射（用于持久化与对话框起始目录）
        self._settings_keys = {
            self.dataset_panel.le_existing: "existing_dir",
            self.dataset_panel.le_rgb: "rgb_dir",
            self.dataset_panel.le_depth_npy: "depth_npy_dir",
            self.dataset_panel.le_label: "label_dir",
            self.dataset_panel.le_out: "out_dir",
            self.dataset_panel.le_low: "depth_low",
            self.dataset_panel.le_high: "depth_high",
            self.train_panel.le_base: "base_path",
            self.train_panel.le_py: "py_path",
            self.eval_panel.le_eval_rgb: "eval_rgb_pt",
            self.eval_panel.le_eval_rgbd: "eval_rgbd_pt",
        }

        self._build_ui()

    # ----------------------------------------------------------------- UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # 左右结构：左侧可滚动（Dataset/Train/Eval/状态），右侧运行日志
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ---- 左侧：可滚动区 ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        scroll.setWidget(body)
        v = QVBoxLayout(body)
        v.addWidget(self.dataset_panel)
        v.addWidget(self.train_panel)
        v.addWidget(self.eval_panel)
        v.addWidget(self.status_log_panel.status_widget)  # 状态区在左侧
        splitter.addWidget(scroll)

        # ---- 右侧：运行日志 ----
        splitter.addWidget(self.status_log_panel.log_widget)

        # 默认比例 65% / 35%；右侧日志最小宽度约 350px
        splitter.setStretchFactor(0, 65)
        splitter.setStretchFactor(1, 35)
        self.status_log_panel.log_widget.setMinimumWidth(350)
        self._splitter = splitter
        self._log_collapsed = False
        self._log_prev_sizes = None
        self.status_log_panel.btn_toggle_log.clicked.connect(self._on_toggle_log)

        # 启动恢复上一次路径，再按当前模式初始化控件与数据集统计
        self._restore_settings()
        # Python 训练环境：未从 QSettings 恢复时回退到启动 GUI 的 Python，允许手动改
        if not self.train_panel.le_py.text().strip():
            self.train_panel.le_py.setText(sys.executable)
        # 基础权重：未从 QSettings 恢复且默认母版存在时自动填入
        if not self.train_panel.le_base.text().strip() and _DEFAULT_BASE_WEIGHT.exists():
            self.train_panel.le_base.setText(str(_DEFAULT_BASE_WEIGHT))
        # 评估权重：未保存时尝试自动定位本次训练产物（允许手动改）
        _rgb_def = _REPO_ROOT / "runs" / "pose" / "train_pose_rgb_3ch2" / "weights" / "best.pt"
        _rgbd_def = _REPO_ROOT / "runs" / "pose" / "train_pose_rgbd_4ch" / "weights" / "best.pt"
        if not self.eval_panel.le_eval_rgb.text().strip() and _rgb_def.exists():
            self.eval_panel.le_eval_rgb.setText(str(_rgb_def))
        if not self.eval_panel.le_eval_rgbd.text().strip() and _rgbd_def.exists():
            self.eval_panel.le_eval_rgbd.setText(str(_rgbd_def))
        self._refresh_4ch_label()
        self._on_mode_changed(self.dataset_panel.rb_existing)

    def _with_browse(self, le: QLineEdit, dir: bool, filt: str = "All (*)", save: bool = False):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(le)
        b = QPushButton("选择...")
        b.clicked.connect(lambda: self._browse(le, dir, filt, save))
        h.addWidget(b)
        return w

    def _start_dir(self, le: QLineEdit) -> str:
        """文件/目录对话框起始目录优先级：当前输入框 > QSettings 上一次 > 项目根。"""
        cur = le.text().strip()
        if cur:
            return cur
        key = self._settings_keys.get(le)
        if key:
            v = self._settings.value(key)
            if v:
                return str(v)
        return str(_REPO_ROOT)

    def _browse(self, le: QLineEdit, dir: bool, filt: str, save: bool = False):
        start = self._start_dir(le)
        if dir:
            p = QFileDialog.getExistingDirectory(self, "选择目录", start)
        elif save:
            p, _ = QFileDialog.getSaveFileName(self, "选择保存位置", start, filt)
        else:
            p, _ = QFileDialog.getOpenFileName(self, "选择文件", start, filt)
        if p:
            le.setText(p)
            self._save_settings()
            if le is self.train_panel.le_base:
                self._refresh_4ch_label()

    # ----------------------------------------------------- 数据集来源模式
    def _on_mode_changed(self, _btn):
        """切换“准备新数据集 / 使用已有数据集”，同步启停控件与 yaml 路径。"""
        existing = self.dataset_panel.rb_existing.isChecked()
        # 准备新数据集整行（含输入框与“选择...”按钮）：仅“准备新数据集”模式启用
        for row in (self.dataset_panel.row_rgb, self.dataset_panel.row_depth_npy,
                    self.dataset_panel.row_label, self.dataset_panel.row_out):
            row.setEnabled(not existing)
        self.dataset_panel.le_class.setEnabled(not existing)
        self.dataset_panel.le_low.setEnabled(not existing)
        self.dataset_panel.le_high.setEnabled(not existing)
        self.dataset_panel.btn_prepare.setEnabled(not existing)
        # 已有数据集目录整行 + RGB/RGBD 透明显示控件：仅“使用已有数据集”模式启用
        for w in (self.dataset_panel.row_existing,
                  self.dataset_panel.le_rgb_yaml,
                  self.dataset_panel.cb_rgbd_variant,
                  self.dataset_panel.le_rgbd_yaml):
            w.setEnabled(existing)
        self._update_yaml_paths()
        if existing:
            self._check_existing_dataset()
        # 模式切换后刷新“测试结果”区上方测试数据只读显示（含 RGB/RGBD 路径与 ID 一致性）
        self._refresh_eval_test_info()
        self._refresh_ablate_info()

    def _on_browse_existing(self):
        start = self._start_dir(self.dataset_panel.le_existing)
        p = QFileDialog.getExistingDirectory(self, "选择已有数据集根目录", start)
        if p:
            self.dataset_panel.le_existing.setText(p)
            self._save_settings()
            self._check_existing_dataset()

    def _update_yaml_paths(self):
        """根据当前模式统一计算 self.rgb_yaml / self.rgbd_yaml。

        - 使用已有数据集模式：rgb_yaml 固定为 <root>/rgb/data_rgb.yaml；
          rgbd_yaml 由 _check_existing_dataset 扫描下拉决定，这里不写死。
        - 准备新数据集模式：prepare 产物固定为 <out>/dataset/<cls>/{rgb,rgbd}/...，
          rgbd_yaml 仍指 rgbd 子目录（与 prepare 输出一致）。
        """
        if self.dataset_panel.rb_existing.isChecked():
            root = self.dataset_panel.le_existing.text().strip()
            base = Path(root) if root else None
            self.rgb_yaml = base / "rgb" / "data_rgb.yaml" if base else None
            # rgbd_yaml 保持由 _check_existing_dataset 扫描设定，不在此写死
        else:
            out = self.dataset_panel.le_out.text().strip()
            cls = self.dataset_panel.le_class.text().strip() or "hand"
            base = Path(out) / "dataset" / cls if out else None
            if base:
                self.rgb_yaml = base / "rgb" / "data_rgb.yaml"
                self.rgbd_yaml = base / "rgbd" / "data_rgbd.yaml"
            else:
                self.rgb_yaml = None
                self.rgbd_yaml = None

    # ----------------------------------------------------- 4ch 变体扫描与选择
    def _parse_variant_yaml(self, yaml_path: Path):
        """极简解析单个 data_*.yaml，返回 {rgbd:bool, channels:int|None, kpt:int|None}。
        支持内联列表 kpt_shape:[7,3] 与多行块列表；不依赖 PyYAML。"""
        try:
            text = yaml_path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            return {"rgbd": False, "channels": None, "kpt": None}
        rgbd = False
        channels = None
        kpt = None
        in_kpt = False
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("rgbd:"):
                v = s.split(":", 1)[1].strip().lower().split("#", 1)[0].strip()
                rgbd = (v == "true")
                continue
            if s.startswith("channels:"):
                try:
                    channels = int(s.split(":", 1)[1].strip())
                except ValueError:
                    channels = None
                continue
            if s.startswith("kpt_shape:"):
                rest = s[len("kpt_shape:"):].split("#", 1)[0].strip()
                if rest.startswith("[") and rest.endswith("]"):
                    nums = [x.strip() for x in rest[1:-1].split(",") if x.strip()]
                    if nums:
                        try:
                            kpt = int(nums[0])
                        except ValueError:
                            kpt = None
                else:
                    in_kpt = True
                continue
            if in_kpt:
                if s.startswith("-"):
                    try:
                        kpt = int(s[1:].strip())
                    except ValueError:
                        pass
                    in_kpt = False
                else:
                    in_kpt = False
        return {"rgbd": rgbd, "channels": channels, "kpt": kpt}

    def _scan_rgbd_variants(self, base: Path):
        """扫描 base 下所有子目录，找出 yaml 同时满足 rgbd:true 且 channels:4 的 4ch 数据集。

        通过读取各目录内 data_*.yaml 的内容判断，不依赖目录名猜测。
        返回 list[(dir_name, yaml_path)]，按 dir_name 排序。
        """
        found = []
        if not base or not base.exists():
            return found
        for sub in sorted(base.iterdir()):
            if not sub.is_dir():
                continue
            for y in sub.glob("data_*.yaml"):
                meta = self._parse_variant_yaml(y)
                if meta["rgbd"] is True and meta["channels"] == 4:
                    found.append((sub.name, y))
                    break  # 同一变体目录只取首个匹配的 4ch yaml
        return found

    def _refresh_rgbd_variants(self, base: Path, variants):
        """刷新 RGBD 变体下拉框并应用当前选择（已保存 > 默认 rgbd_rawdepth > 首个）。

        内部通过 blockSignals 避免填充期间误触发；结束后显式 _apply_rgbd_variant 同步
        self.rgbd_yaml / “当前 RGBD YAML”显示 / QSettings / 统计区。
        """
        self._rgbd_variants = variants
        cb = self.dataset_panel.cb_rgbd_variant
        order = [v[0] for v in variants]
        cb.blockSignals(True)
        cb.clear()
        for dir_name, y in variants:
            cb.addItem(dir_name, str(y))  # item text = 真实目录名（透明），userData = yaml 路径
        # 选择优先级：已保存 > 默认建议 rgbd_rawdepth > 第一个
        saved = self._settings.value("rgbd_variant")
        preferred = "rgbd_rawdepth"
        if saved in order:
            default_name = saved
        elif preferred in order:
            default_name = preferred
        else:
            default_name = order[0]
        cb.setCurrentIndex(order.index(default_name))
        cb.blockSignals(False)
        self._apply_rgbd_variant(default_name)

    def _on_rgbd_variant_changed(self, dir_name: str):
        """下拉框切换回调（填充期间 _populating 守卫为 True，忽略中间信号）。"""
        if getattr(self, "_populating", False):
            return
        self._apply_rgbd_variant(dir_name)

    def _apply_rgbd_variant(self, dir_name: str):
        """应用选中的 RGBD 变体：同步 self.rgbd_yaml、当前 RGBD YAML 显示、统计区、
        QSettings 持久化，并留下日志（确保选择对用户可见，不静默）。"""
        variant = next((v for v in self._rgbd_variants if v[0] == dir_name), None)
        if variant is None:
            self.dataset_panel.le_rgbd_yaml.setText("-")
            self.dataset_panel.lbl_rgbd_ds.setText("-")
            self.rgbd_yaml = None
            return
        dname, y = variant
        self.rgbd_yaml = Path(y)
        self.dataset_panel.le_rgbd_yaml.setText(str(y))
        self.dataset_panel.lbl_rgbd_ds.setText(dname)
        self._settings.setValue("rgbd_variant", dname)
        self._settings.sync()
        self._log(f"[状态] 当前 RGBD 变体: {dname}  ->  {y}")
        # 变体切换立即同步“测试结果”区上方测试数据路径显示
        self._refresh_eval_test_info()
        self._refresh_ablate_info()

    def _resolve_test_set(self, yaml_path):
        """解析单个 data_*.yaml 的 path + test 字段，返回 (test_images_dir, count, id_set)。

        - test_images_dir: 真实测试图像目录（path 为相对时按 yaml 父目录解析）
        - count: 实际图片数量（目录缺失/无图则为 0）
        - id_set: 测试图像 stem 集合（用于跨 RGB/RGBD 一致性校验；目录缺失则为 None）

        任何无法解析的情况返回 (None, 0, None)。
        """
        if not yaml_path or not Path(yaml_path).exists():
            return None, 0, None
        try:
            text = Path(yaml_path).read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            return None, 0, None
        path_field = None
        test_field = "images/test"
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("path:"):
                path_field = s.split(":", 1)[1].strip().split("#", 1)[0].strip()
            elif s.startswith("test:"):
                test_field = s.split(":", 1)[1].strip().split("#", 1)[0].strip()
        if not path_field:
            return None, 0, None
        root = Path(path_field)
        if not root.is_absolute():
            root = Path(yaml_path).parent / path_field
        test_dir = root / test_field
        if not test_dir.exists():
            return test_dir, 0, None
        img_ext = (".png", ".jpg", ".jpeg")
        ids = sorted(p.stem for p in test_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in img_ext)
        return test_dir, len(ids), set(ids)

    def _refresh_eval_test_info(self):
        """刷新“测试结果”区上方“当前测试数据”只读显示：

        - RGB / RGBD 各取当前 self.rgb_yaml / self.rgbd_yaml 的 path+test 解析真实目录与数量
        - Split 固定 test
        - Test ID 一致性：RGB test ID 集合 vs RGBD test ID 集合

        不修改任何评估逻辑；仅写回 GUI 只读标签。
        """
        ep = self.eval_panel
        rgb_dir, rgb_cnt, rgb_ids = self._resolve_test_set(self.rgb_yaml)
        rgbd_dir, rgbd_cnt, rgbd_ids = self._resolve_test_set(self.rgbd_yaml)
        if rgb_ids is not None and rgbd_ids is not None:
            if rgb_ids == rgbd_ids:
                id_text = f"{rgb_cnt}/{rgbd_cnt} 完全一致"
            else:
                only_rgb = len(rgb_ids - rgbd_ids)
                only_rgbd = len(rgbd_ids - rgb_ids)
                parts = []
                if only_rgb:
                    parts.append(f"RGB 独有 {only_rgb}")
                if only_rgbd:
                    parts.append(f"RGBD 独有 {only_rgbd}")
                id_text = "不一致（" + "，".join(parts) + "）"
        else:
            id_text = "无法校验（目录缺失）"
        ep.set_test_info({
            "rgb_yaml": str(self.rgb_yaml) if self.rgb_yaml else None,
            "rgb_img": str(rgb_dir) if rgb_dir else None,
            "rgb_cnt": rgb_cnt,
            "rgbd_yaml": str(self.rgbd_yaml) if self.rgbd_yaml else None,
            "rgbd_img": str(rgbd_dir) if rgbd_dir else None,
            "rgbd_cnt": rgbd_cnt,
            "split": "test（固定）",
            "id_text": id_text,
        })

    def _refresh_ablate_info(self):
        """刷新“Depth 消融测试”只读信息：使用当前 RGBD 权重 + 当前 RGBD variant YAML。

        - 当前 RGBD YAML：self.rgbd_yaml（随变体切换 / 模式切换同步）
        - 当前 RGBD best.pt：eval_panel.le_eval_rgbd 输入框实时文本（用户输入即更新）
        - Split 固定 test（与评估一致，不允许手工改）
        """
        ep = self.eval_panel
        ep.set_ablate_info({
            "rgbd_yaml": str(self.rgbd_yaml) if self.rgbd_yaml else None,
            "rgbd_pt": self.eval_panel.le_eval_rgbd.text().strip() or None,
        })

    def _restore_settings(self):
        """启动时恢复上一次保存的路径到对应输入框。"""
        for le, key in self._settings_keys.items():
            v = self._settings.value(key)
            if v:
                le.setText(str(v))

    def _save_settings(self):
        """把当前所有路径写入 QSettings（对话框选择后 / 关闭时调用）。"""
        for le, key in self._settings_keys.items():
            t = le.text().strip()
            if t:
                self._settings.setValue(key, t)
        self._settings.sync()

    def _parse_yaml_meta(self, rgb_yaml: Path, rgbd_yaml: Path):
        """极简解析本项目 data_rgb.yaml / data_rgbd.yaml：
        返回 (kpt_shape_first, rgb_channels, rgbd_channels)。
        支持标量字段与块列表 kpt_shape:（不依赖 PyYAML）。
        """

        def parse(text: str):
            channels = None
            kpt = None
            in_kpt = False
            for line in text.splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if s.startswith("kpt_shape:"):
                    rest = s[len("kpt_shape:"):].split("#", 1)[0].strip()
                    if rest.startswith("[") and rest.endswith("]"):
                        # 内联列表格式：kpt_shape: [7, 3]
                        inside = rest[1:-1]
                        nums = [x.strip() for x in inside.split(",") if x.strip()]
                        if nums:
                            try:
                                kpt = int(nums[0])  # 取第一个元素 K
                            except ValueError:
                                kpt = None
                    else:
                        # 多行块列表格式：kpt_shape: 后跟 - 7 / - 3
                        in_kpt = True
                    continue
                if in_kpt:
                    if s.startswith("-"):
                        try:
                            val = int(s[1:].strip())
                        except ValueError:
                            continue
                        kpt = val if kpt is None else kpt  # 取第一个元素 K
                        in_kpt = False
                    else:
                        in_kpt = False
                if s.startswith("channels:"):
                    try:
                        channels = int(s.split(":", 1)[1].strip())
                    except ValueError:
                        channels = None
            return kpt, channels

        rk, rc = parse(rgb_yaml.read_text(encoding="utf-8"))
        dk, dc = parse(rgbd_yaml.read_text(encoding="utf-8"))
        return (rk if rk is not None else dk), rc, dc

    def _stat_existing_dataset(self, base: Path, rgbd_sub: str):
        """现场统计已有数据集：数 images 各 split，校验 RGB 与所选 RGBD 变体一致，
        读各自 yaml 的 kpt_shape/channels。返回 dict（含 rgb_dataset/rgbd_dataset 来源名），
        images 目录缺失时返回 None（上层按不完整处理）。只读不写回，不直接设置界面标签。"""
        splits = ("train", "val", "test")
        img_ext = (".png", ".jpg", ".jpeg")

        def count_imgs(sub: str):
            d = base / sub
            if not d.exists():
                return None
            return sum(1 for p in d.iterdir() if p.suffix.lower() in img_ext)

        rgb_counts = {s: count_imgs(f"rgb/images/{s}") for s in splits}
        rgbd_counts = {s: count_imgs(f"{rgbd_sub}/images/{s}") for s in splits}
        if any(c is None for c in rgb_counts.values()) or any(c is None for c in rgbd_counts.values()):
            return None
        mismatch = [s for s in splits if rgb_counts[s] != rgbd_counts[s]]
        if mismatch:
            self._log("[警告] RGB 与所选 RGBD 变体各 split 数量不一致: " + ", ".join(
                f"{s}: rgb={rgb_counts[s]} {rgbd_sub}={rgbd_counts[s]}" for s in mismatch))
        total = sum(rgb_counts.values())
        try:
            kpt, rc, dc = self._parse_yaml_meta(
                base / "rgb" / "data_rgb.yaml",
                Path(self.rgbd_yaml) if self.rgbd_yaml else base / rgbd_sub)
        except Exception as e:  # noqa: BLE001
            self._log(f"[警告] 读取 yaml 元数据失败: {e}")
            kpt, rc, dc = None, None, None
        return {
            "total": total,
            "train": rgb_counts["train"],
            "val": rgb_counts["val"],
            "test": rgb_counts["test"],
            "keypoint_count": kpt,
            "rgb_channels": rc,
            "rgbd_channels": dc,
            "rgb_dataset": "rgb",
            "rgbd_dataset": rgbd_sub,
        }

    def _check_existing_dataset(self):
        """检查已选数据集根目录：

        - RGB 固定识别 <root>/rgb/data_rgb.yaml（必须存在）。
        - RGBD 自动扫描 root 下所有满足 yaml rgbd:true & channels:4 的 4ch 变体，
          下拉供用户选择；不再写死为 rgbd/data_rgbd.yaml。
        - 选中变体后做现场统计；若 dataset_report.json 存在，用其有效字段覆盖补齐。
        """
        root = self.dataset_panel.le_existing.text().strip()
        if not root:
            return
        base = Path(root)
        ry = base / "rgb" / "data_rgb.yaml"
        if not ry.exists():
            self._log(f"[错误] 已有数据集缺少 RGB yaml: rgb/data_rgb.yaml ({ry})")
            self.dataset_panel.reset_stats()
            self.dataset_panel.set_ready("已有数据集不完整")
            self.rgb_yaml = None
            self.rgbd_yaml = None
            self.dataset_panel.le_rgb_yaml.setText("-")
            self.dataset_panel.cb_rgbd_variant.clear()
            self.dataset_panel.le_rgbd_yaml.setText("-")
            return
        # RGB 固定路径，只读展示
        self.rgb_yaml = ry
        self.dataset_panel.le_rgb_yaml.setText(str(ry))

        # 扫描所有 4ch 变体
        variants = self._scan_rgbd_variants(base)
        if not variants:
            self._log("[错误] 未找到任何 4 通道 RGBD 数据集（需 yaml 声明 rgbd:true 且 channels:4）")
            self.dataset_panel.reset_stats()
            self.dataset_panel.set_ready("已有数据集不完整")
            self.rgbd_yaml = None
            self.dataset_panel.cb_rgbd_variant.clear()
            self.dataset_panel.le_rgbd_yaml.setText("-")
            return

        # 刷新下拉框并应用当前选择（内部设定 self.rgbd_yaml + 显示 + 持久化）
        self._refresh_rgbd_variants(base, variants)

        # 1) 现场统计（用选中的 RGBD 变体目录）：无论 report 是否存在都先做
        rgbd_sub = self.rgbd_yaml.parent.name
        stat = self._stat_existing_dataset(base, rgbd_sub)
        if stat is None:
            self._log("[警告] 已有数据集 images 目录缺失，无法现场统计")
            self.dataset_panel.reset_stats()
            self.dataset_panel.set_ready("已有数据集不完整")
            return

        # 2) report 有效字段优先，缺失/None 用现场统计补齐
        merged = dict(stat)
        rp = base / "dataset_report.json"
        if rp.exists():
            try:
                d = json.loads(rp.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                self._log(f"[警告] dataset_report.json 解析失败，回退现场统计: {e}")
                d = {}
            if isinstance(d, dict):
                report_map = {
                    "total_samples": "total",
                    "train_samples": "train",
                    "val_samples": "val",
                    "test_samples": "test",
                    "keypoint_count": "keypoint_count",
                    "rgb_channels": "rgb_channels",
                    "rgbd_channels": "rgbd_channels",
                }
                for rk, mk in report_map.items():
                    v = d.get(rk)
                    if v is not None:
                        merged[mk] = v
                if d.get("validation_passed"):
                    ready = "已有数据集已加载（report + 现场补齐）"
                else:
                    ready = "已有数据集校验未通过（report + 现场补齐）"
            else:
                ready = "已有数据集已加载（现场统计）"
        else:
            ready = "已有数据集已加载（现场统计）"

        # 3) 回写显示（None 值显示 -；含 RGB/RGBD 数据集来源）
        self.dataset_panel.set_stats(merged, ready)
        self._log("[状态] 已有数据集已加载")

    # --------------------------------------------------------------- 状态/日志
    def _set_status(self, text: str):
        self.status_log_panel.set_status(text)

    def _log(self, text: str):
        self.status_log_panel.append_log(text)

    def _on_toggle_log(self):
        """收起 / 展开右侧运行日志：用 QSplitter.setSizes 控制右侧宽度。"""
        sp = self._splitter
        if not self._log_collapsed:
            # 收起：记住收起前宽度，临时取消右侧最小宽度限制，右侧设 0
            self._log_prev_sizes = sp.sizes()
            self.status_log_panel.log_widget.setMinimumWidth(0)
            sp.setSizes([sum(self._log_prev_sizes), 0])
            self._log_collapsed = True
            self.status_log_panel.btn_toggle_log.setText("展开日志")
        else:
            # 展开：恢复收起前右侧宽度，恢复右侧最小宽度
            if self._log_prev_sizes and len(self._log_prev_sizes) == 2:
                sp.setSizes(self._log_prev_sizes)
            else:
                total = sum(sp.sizes()) or 1000
                sp.setSizes([int(total * 0.65), int(total * 0.35)])
            self.status_log_panel.log_widget.setMinimumWidth(350)
            self._log_collapsed = False
            self.status_log_panel.btn_toggle_log.setText("收起日志")

    def _disable_all(self, disable: bool):
        for b in (self.dataset_panel.btn_prepare, self.train_panel.btn_train_rgb,
                  self.train_panel.btn_build, self.train_panel.btn_train_rgbd,
                  self.eval_panel.btn_eval_rgb, self.eval_panel.btn_eval_rgbd,
                  self.eval_panel.btn_eval_cmp, self.eval_panel.btn_ablate):
            b.setEnabled(not disable)

    def _restore_controls(self):
        """任务结束后恢复“当前模式应有的控件状态”，而非无条件全部启用。

        - 三个训练/构建动作按钮始终恢复可用
        - 准备数据集按钮仅“准备新数据集”模式可用（使用已有数据集模式保持禁用）
        - 三个评估按钮始终恢复可用
        """
        self.train_panel.btn_train_rgb.setEnabled(True)
        self.train_panel.btn_build.setEnabled(True)
        self.train_panel.btn_train_rgbd.setEnabled(True)
        self.dataset_panel.btn_prepare.setEnabled(self.dataset_panel.rb_new.isChecked())
        self.eval_panel.btn_eval_rgb.setEnabled(True)
        self.eval_panel.btn_eval_rgbd.setEnabled(True)
        self.eval_panel.btn_eval_cmp.setEnabled(True)
        self.eval_panel.btn_ablate.setEnabled(True)

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)

    # -------------------------------------------------------------- 启动子进程
    def _run(self, op: str, script: Path, args: list, status: str):
        if self.process.state() == QProcess.ProcessState.Running:
            self._log("[警告] 已有任务在运行，忽略本次请求")
            return
        # 子进程解释器：使用“Python训练环境”选择框指定的 exe，而非 sys.executable
        py = self.train_panel.le_py.text().strip()
        if not py:
            self._log("[错误] 未设置 Python 训练环境路径，请先选择训练用 Python 解释器")
            return
        if not Path(py).exists():
            self._log(f"[错误] Python 训练环境不存在: {py}")
            return
        self.current_op = op
        self.user_stopped = False  # 新任务开始，重置用户停止标记
        self._disable_all(True)
        self.status_log_panel.btn_stop.setEnabled(True)  # 运行期间允许停止
        self._set_status(status)
        # 工作目录设为项目根；PYTHONPATH 前置项目根，确保本地 ultralytics 可被找到
        env = QProcessEnvironment.systemEnvironment()
        existing = env.value("PYTHONPATH", "")
        new_pp = str(_REPO_ROOT) if not existing else f"{_REPO_ROOT}{os.pathsep}{existing}"
        env.insert("PYTHONPATH", new_pp)
        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(str(_REPO_ROOT))
        cmd = [str(script)] + args
        self._log(">>> " + " ".join([py, *cmd]))
        self.process.start(py, cmd)

    def _on_prepare(self):
        rgb = self.dataset_panel.le_rgb.text().strip()
        depth_npy = self.dataset_panel.le_depth_npy.text().strip()
        label = self.dataset_panel.le_label.text().strip()
        out = self.dataset_panel.le_out.text().strip()
        cls = self.dataset_panel.le_class.text().strip() or "hand"
        if not (rgb and depth_npy and label and out):
            self._log("[错误] RGB / Raw Depth NPY / Labels / 输出目录 必须全部填写")
            return
        # Depth low/high：允许用户编辑，缺省回退 1100/1850
        try:
            low = float(self.dataset_panel.le_low.text().strip() or "1100")
            high = float(self.dataset_panel.le_high.text().strip() or "1850")
        except ValueError:
            self._log("[错误] Depth low/high(mm) 必须为数字")
            return
        if not (high > low):
            self._log("[错误] Depth low(mm) 必须严格小于 high(mm)")
            return
        self._run("prepare", _SCRIPTS["prepare"],
                  ["--rgb_dir", rgb, "--depth_npy_dir", depth_npy,
                   "--label_dir", label, "--output_dir", out, "--class_name", cls,
                   "--depth_low", str(low), "--depth_high", str(high)],
                  "处理中")

    def _derive_4ch_path(self, base_text: str) -> Path:
        """由基础权重派生 4ch 输出路径：固定放到 weights/4ch/<stem>_4ch.pt。"""
        base = Path(base_text)
        return _FOURCH_DIR / f"{base.stem}_4ch.pt"

    def _refresh_4ch_label(self):
        """根据当前基础权重刷新 4ch 权重显示路径。"""
        base = self.train_panel.le_base.text().strip()
        if base:
            self.train_panel.set_4ch_text(str(self._derive_4ch_path(base)))
        else:
            self.train_panel.set_4ch_text("-")

    def _on_train_rgb(self):
        base = self.train_panel.le_base.text().strip()
        if not base:
            self._log("[错误] 请先选择基础 Pose 权重（yolov8n-pose.pt）")
            return
        if not Path(base).exists():
            self._log(f"[错误] 基础权重不存在: {base}")
            return
        yaml = self.rgb_yaml
        if not yaml or not Path(yaml).exists():
            self._log("[错误] 未确定 RGB data yaml：请先在“使用已有数据集”选择目录，"
                      "或在“准备新数据集”执行准备数据集")
            return
        self._run("train_rgb", _SCRIPTS["train_rgb"],
                  ["--data", str(yaml),
                   "--fliplr", self.train_panel.le_fliplr.text().strip(),
                   "--patience", self.train_panel.le_patience.text().strip(),
                   "--weights", base],
                  "训练中")

    def _on_build(self):
        base = self.train_panel.le_base.text().strip()
        if not base:
            self._log("[错误] 请先选择基础 Pose 权重（yolov8n-pose.pt）")
            return
        if not Path(base).exists():
            self._log(f"[错误] 基础权重不存在: {base}")
            return
        out = self._derive_4ch_path(base)
        out.parent.mkdir(parents=True, exist_ok=True)  # 确保 4ch/ 目录存在
        self._run("build", _SCRIPTS["build"],
                  ["--input", base, "--output", str(out)],
                  "处理中")

    def _on_train_rgbd(self):
        base = self.train_panel.le_base.text().strip()
        if not base:
            self._log("[错误] 请先选择基础 Pose 权重（yolov8n-pose.pt）")
            return
        w4 = self._derive_4ch_path(base)
        if not w4.exists():
            self._log(f"[错误] 4通道权重不存在: {w4}，请先点击“生成4ch权重”")
            return
        yaml = self.rgbd_yaml
        if not yaml or not Path(yaml).exists():
            self._log("[错误] 未确定 RGBD data yaml：请先在“使用已有数据集”选择目录，"
                      "或在“准备新数据集”执行准备数据集")
            return
        self._run("train_rgbd", _SCRIPTS["train_rgbd"],
                  ["--data", str(yaml),
                   "--fliplr", self.train_panel.le_fliplr.text().strip(),
                   "--patience", self.train_panel.le_patience.text().strip(),
                   "--weights", str(w4)],
                  "训练中")

    # --------------------------------------------------------- 测试集评估
    def _on_eval_rgb(self):
        self._start_eval_leg("rgb")

    def _on_eval_rgbd(self):
        self._start_eval_leg("rgbd")

    def _on_eval_cmp(self):
        # 对比测试启动前再次校验 RGB 与 RGBD test ID 一致性，不一致则拒绝并打印错误
        rgb_dir, rgb_cnt, rgb_ids = self._resolve_test_set(self.rgb_yaml)
        rgbd_dir, rgbd_cnt, rgbd_ids = self._resolve_test_set(self.rgbd_yaml)
        if rgb_ids is None or rgbd_ids is None:
            self._log("[错误] 对比测试被拒绝：RGB 或 RGBD test 目录缺失/无法解析，"
                      "请先确认数据集已正确加载")
            self._set_status("对比测试被拒绝")
            return
        if rgb_ids != rgbd_ids:
            only_rgb = sorted(rgb_ids - rgbd_ids)
            only_rgbd = sorted(rgbd_ids - rgb_ids)
            msg = (f"[错误] 对比测试被拒绝：RGB 与 RGBD test ID 不一致 "
                   f"(RGB={rgb_cnt}, RGBD={rgbd_cnt})")
            if only_rgb:
                msg += f" | RGB 独有: {only_rgb[:10]}{'...' if len(only_rgb) > 10 else ''}"
            if only_rgbd:
                msg += f" | RGBD 独有: {only_rgbd[:10]}{'...' if len(only_rgbd) > 10 else ''}"
            self._log(msg)
            self._set_status("对比测试被拒绝")
            return
        # 依次测试 RGB 与 RGBD，最后对比差值（不重复链式启动：队首 pop 后即交给 _on_finished 续跑）
        self.eval_results.clear()
        self.eval_chain = ["rgb", "rgbd"]
        self._compare_pending = True

        first = self.eval_chain.pop(0)
        if not self._start_eval_leg(first):
            self.eval_chain.clear()
            self._compare_pending = False
            self._set_status("评估中断")

    def _start_eval_leg(self, leg):
        """启动单个 leg（rgb/rgbd）的 test 集评估；返回是否成功启动。"""
        name = "RGB" if leg == "rgb" else "RGBD"
        tag = "rgb" if leg == "rgb" else "rgbd"
        le = self.eval_panel.le_eval_rgb if leg == "rgb" else self.eval_panel.le_eval_rgbd
        weights = le.text().strip()
        yaml = self.rgb_yaml if leg == "rgb" else self.rgbd_yaml
        if not weights or not Path(weights).exists():
            self._log(f"[错误] {name} 权重不存在: {weights}")
            return False
        if not yaml or not Path(yaml).exists():
            self._log(f"[错误] 未确定 {name} data yaml：请先在“使用已有数据集”选择数据集目录"
                      f"（需含 dataset/hand/{tag}/data_{tag}.yaml）")
            return False
        # 共用各自 YAML 中同一批 test ID（两 yaml 由同一次 prepare 生成，test 划分一致）
        self._last_eval_result = None
        self.current_op = f"eval_{leg}"
        self._run(f"eval_{leg}", _SCRIPTS["eval"],
                  ["--weights", weights, "--data", str(yaml),
                   "--split", "test", "--imgsz", "640", "--batch", "4"],
                  f"评估{name}中")
        return True

    def _build_ablate_args(self):
        """构造 Depth 消融子进程参数（不直接启动）：使用当前 RGBD 权重 + 当前 RGBD YAML，
        仅 true_depth / zero_depth 两变体，split 固定 test，输出到 runs/ablation/depth_ablation_gui。

        权重路径与 YAML 均来自 GUI 当前状态（不写死），便于 smoke test 直接校验参数。
        """
        weights = self.eval_panel.le_eval_rgbd.text().strip()
        yaml = self.rgbd_yaml
        out = _REPO_ROOT / "runs" / "ablation" / "depth_ablation_gui"
        args = ["--weights", weights,
                "--rgbd-yaml", str(yaml) if yaml else "",
                "--variants", "true_depth", "zero_depth",
                "--split", "test",
                "--out", str(out)]
        return {"weights": weights, "yaml": str(yaml) if yaml else None, "args": args}

    def _on_ablate_depth(self):
        info = self._build_ablate_args()
        weights, ypath = info["weights"], info["yaml"]
        if not weights or not Path(weights).exists():
            self._log(f"[错误] RGBD 权重不存在: {weights}")
            return
        if not ypath or not Path(ypath).exists():
            self._log("[错误] 未确定 RGBD data yaml：请先在“使用已有数据集”选择数据集目录")
            return
        test_dir, test_cnt, _ = self._resolve_test_set(self.rgbd_yaml)
        # 启动前明确打印本次消融使用的模型 / 数据 / split / 变体 / test 信息
        self._log("[Depth Ablation]")
        self._log(f"Model: {weights}")
        self._log(f"Data: {ypath}")
        self._log("Split: test")
        self._log("Variants:\n- true_depth\n- zero_depth")
        self._log(f"Test images: {test_dir}")
        self._log(f"Test count: {test_cnt}")
        self._run("ablate", _SCRIPTS["ablate"], info["args"], "Depth 消融中")

    def _finish_compare(self):
        r = self.eval_results.get("rgb")
        d = self.eval_results.get("rgbd")
        if not r or not d:
            self._log("[警告] 对比测试缺少 RGB 或 RGBD 结果，无法计算差值")
            self._set_status("对比不完整")
            return
        db_box = d["box_map"] - r["box_map"]
        db_pose = d["pose_map"] - r["pose_map"]
        self.eval_panel.display_compare(r, d, db_box, db_pose)
        self._log("[对比] RGB vs RGBD 已完成，差值 = RGBD - RGB")
        self._set_status("对比完成")

    # --------------------------------------------------------- QProcess 回调
    def _on_stdout(self):
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace").rstrip("\n")
        if data:
            self._log(data)
            # 捕获评估结果 JSON（单行 POSE_EVAL_JSON ...）
            for line in data.splitlines():
                if "POSE_EVAL_JSON" in line:
                    try:
                        self._last_eval_result = json.loads(line.split("POSE_EVAL_JSON", 1)[1].strip())
                    except Exception as e:  # noqa: BLE001
                        self._log(f"[警告] 评估结果 JSON 解析失败: {e}")
                elif "DEPTH_ABLATION_JSON" in line:
                    try:
                        self._last_ablate_result = json.loads(
                            line.split("DEPTH_ABLATION_JSON", 1)[1].strip())
                    except Exception as e:  # noqa: BLE001
                        self._log(f"[警告] 消融结果 JSON 解析失败: {e}")

    def _on_stderr(self):
        data = bytes(self.process.readAllStandardError()).decode("utf-8", "replace").rstrip("\n")
        if data:
            self._log("[err] " + data)

    def _on_started(self):
        self._log(f"[启动] {self.current_op}")

    def _on_stop(self):
        """请求停止当前运行中的子进程：先 terminate，3s 后若仍存活则 kill。不阻塞 GUI。"""
        if self.process.state() == QProcess.ProcessState.Running:
            self._log("[停止] 正在请求停止当前任务...")
            self.user_stopped = True
            self.process.terminate()
            QTimer.singleShot(3000, self._check_stop)
        else:
            self._log("[停止] 当前没有运行中的任务")

    def _check_stop(self):
        if (
                self.user_stopped
                and self.process.state() != QProcess.ProcessState.NotRunning
        ):
            self._log("[停止] 任务未正常退出，强制结束")
            self.process.kill()

    def _on_finished(self, code: int, _status):
        # ---- 评估（eval_*）分支：优先处理，避免与训练/准备逻辑混淆 ----
        if self.current_op and self.current_op.startswith("eval_"):
            leg = self.current_op.split("_", 1)[1]  # "rgb" / "rgbd"
            if self.user_stopped:
                self._log("[结束] 用户已停止当前任务")
                self._set_status("已停止")
                self.eval_chain.clear()
                self._compare_pending = False
            elif code == 0 and self._last_eval_result:
                self.eval_results[leg] = self._last_eval_result
                self.eval_panel.display_result(leg, self._last_eval_result)
                self._log(f"[评估] {leg} 测试集结果已显示")
            else:
                self._log(f"[警告] {leg} 评估失败 exit_code={code}，未能获取结果")
                self._set_status("评估失败")
            # 对比测试链式：继续下一 leg 或收尾
            if self.eval_chain:
                nxt = self.eval_chain.pop(0)
                if self._start_eval_leg(nxt):
                    return  # 下一 leg 已启动，等待其 _on_finished
                self.eval_chain.clear()
                self._compare_pending = False
                self._set_status("评估中断")
            elif self._compare_pending:
                self._compare_pending = False
                self._finish_compare()
            # 收尾：停止按钮禁用 + 恢复其它按钮
            self.status_log_panel.btn_stop.setEnabled(False)
            self._restore_controls()
            self.current_op = None
            return

        # ---- Depth 消融 (ablate) 分支 ----
        if self.current_op == "ablate":
            if self.user_stopped:
                self._log("[结束] 用户已停止当前任务")
                self._set_status("已停止")
            elif code == 0 and self._last_ablate_result:
                r = self._last_ablate_result
                self._log("[Depth Ablation] 完成")
                self._log(f"  结论: {r.get('conclusion', '')}")
                self._set_status("Depth 消融完成")
            else:
                self._log(f"[警告] Depth 消融失败 exit_code={code}，未能获取结果")
                self._set_status("Depth 消融失败")
            self.status_log_panel.btn_stop.setEnabled(False)
            self._restore_controls()
            self.current_op = None
            return

        if self.user_stopped:
            # 用户主动停止：状态显示“已停止”，不归为失败
            self._log("[结束] 用户已停止当前任务")
            self._set_status("已停止")
        else:
            ok = (code == 0)
            self._log(f"[结束] exit_code={code} -> {'完成' if ok else '失败'}")
            self._set_status("完成" if ok else "失败")
        # 停止按钮必禁用；其它按钮恢复当前模式应有的状态
        self.status_log_panel.btn_stop.setEnabled(False)
        self._restore_controls()
        if self.current_op == "prepare" and code == 0 and not self.user_stopped:
            self._load_prep_status()
        self.current_op = None

    def _load_prep_status(self):
        out = Path(self.dataset_panel.le_out.text().strip())
        cls = self.dataset_panel.le_class.text().strip() or "hand"
        base = out / "dataset" / cls
        # 准备模式完成后统一设置训练用的 yaml 路径
        self.rgb_yaml = base / "rgb" / "data_rgb.yaml"
        self.rgbd_yaml = base / "rgbd" / "data_rgbd.yaml"
        rp = base / "dataset_report.json"
        if not rp.exists():
            self._log("[警告] 未找到 dataset_report.json，无法刷新数据集状态")
            return
        d = json.loads(rp.read_text(encoding="utf-8"))
        dp = self.dataset_panel
        dp.lbl_total.setText(str(d.get("total_samples", "-")))
        dp.lbl_train.setText(str(d.get("train_samples", "-")))
        dp.lbl_val.setText(str(d.get("val_samples", "-")))
        dp.lbl_test.setText(str(d.get("test_samples", "-")))
        dp.lbl_kpt.setText(str(d.get("keypoint_count", "-")))
        dp.lbl_rgbch.setText(str(d.get("rgb_channels", "-")))
        dp.lbl_rgbdch.setText(str(d.get("rgbd_channels", "-")))
        if d.get("validation_passed"):
            dp.lbl_ready.setText("DATASET READY")
            self._log("[状态] 数据集已就绪")
        else:
            dp.lbl_ready.setText("校验未通过")
            self._log("[状态] 数据集校验未通过，请检查")
        # 准备模式完成后同步测试数据只读显示
        self._refresh_eval_test_info()
        self._refresh_ablate_info()
