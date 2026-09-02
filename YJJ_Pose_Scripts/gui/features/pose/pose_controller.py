"""Pose 工作区协调控制器。

从 MainWindow 移出的全部协调逻辑：
- 连接 Panel signals
- 维护 rgb_yaml / rgbd_yaml / RGBD variant
- prepare / train / build / eval / compare / ablate 工作流
- eval chain
- 当前 operation 状态
- 更新 Panel 展示
- 调用 ProcessRunner / Services

依赖方向：
    PosePage → PoseController → Services / ProcessRunner / SettingsStore
    PoseController 连接 Panel signals，Panel 不调用 Controller。
"""
import sys
from pathlib import Path

from gui_config import _DEFAULT_BASE_WEIGHT, _REPO_ROOT
from core.process_runner import ProcessRunner
from core.settings_store import SettingsStore, make_start_dir_provider
from .services.dataset_service import DatasetService
from .services.command_service import CommandService
from .services.result_parser import ResultParser


class PoseController:
    """Pose 工作区协调器：连接 Panel 信号 → 调用 Service / ProcessRunner。"""

    def __init__(self, dataset_panel, train_panel, eval_panel, status_log_panel):
        self.dataset_panel = dataset_panel
        self.train_panel = train_panel
        self.eval_panel = eval_panel
        self.status_log_panel = status_log_panel

        # ---- 基础设施 ----
        self._runner = ProcessRunner()
        self._settings_store = SettingsStore()

        # ---- 状态 ----
        self.rgb_yaml = None
        self.rgbd_yaml = None
        self._rgbd_variants = []
        self.eval_results = {}
        self.eval_chain = []
        self._compare_pending = False
        self._last_eval_result = None
        self._last_ablate_result = None

        # 路径 <-> QSettings key 映射
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

        self._connect_signals()
        self._init()

    # ================================================================ 信号连接
    def _connect_signals(self):
        # ---- Dataset Panel ----
        dp = self.dataset_panel
        dp.modeChanged.connect(self._on_mode_changed)
        dp.existingDirChanged.connect(self._check_existing_dataset)
        dp.prepareRequested.connect(self._on_prepare)
        dp.rgbdVariantChanged.connect(self._apply_rgbd_variant)
        dp.settingsDirty.connect(self._save_settings)

        # ---- Train Panel ----
        tp = self.train_panel
        tp.trainRgbRequested.connect(self._on_train_rgb)
        tp.build4chRequested.connect(self._on_build)
        tp.trainRgbdRequested.connect(self._on_train_rgbd)
        tp.baseWeightChanged.connect(self._on_base_weight_changed)
        tp.settingsDirty.connect(self._save_settings)

        # ---- Eval Panel ----
        ep = self.eval_panel
        ep.evalRgbRequested.connect(self._on_eval_rgb)
        ep.evalRgbdRequested.connect(self._on_eval_rgbd)
        ep.compareRequested.connect(self._on_eval_cmp)
        ep.depthAblationRequested.connect(self._on_ablate_depth)
        ep.rgbdWeightChanged.connect(self._refresh_ablate_info)
        ep.settingsDirty.connect(self._save_settings)

        # ---- Status Log Panel ----
        self.status_log_panel.stopRequested.connect(self._on_stop)

        # ---- ProcessRunner ----
        self._runner.started.connect(self._on_started)
        self._runner.stdoutReceived.connect(self._on_stdout)
        self._runner.stderrReceived.connect(self._on_stderr)
        self._runner.finished.connect(self._on_finished)

        # ---- 注入 browse 起始目录 QSettings fallback ----
        # provider 是 core/settings_store.make_start_dir_provider 返回的纯闭包，
        # 仅关闭 settings_store + le→key 映射；Panel 不持有 Controller，也不调用其方法。
        start_dir_provider = make_start_dir_provider(
            self._settings_store, self._settings_keys)
        self.dataset_panel.set_start_dir_provider(start_dir_provider)
        self.train_panel.set_start_dir_provider(start_dir_provider)
        self.eval_panel.set_start_dir_provider(start_dir_provider)

    # ================================================================ 初始化
    def _init(self):
        """启动恢复 + 设置默认值 + 应用当前模式。"""
        self._restore_settings()
        # Python 训练环境：未从 QSettings 恢复时回退到启动 GUI 的 Python
        if not self.train_panel.le_py.text().strip():
            self.train_panel.le_py.setText(sys.executable)
        # 基础权重：未从 QSettings 恢复且默认母版存在时自动填入
        if not self.train_panel.le_base.text().strip() and _DEFAULT_BASE_WEIGHT.exists():
            self.train_panel.le_base.setText(str(_DEFAULT_BASE_WEIGHT))
        # 评估权重：未保存时尝试自动定位本次训练产物
        _rgb_def = _REPO_ROOT / "runs" / "pose" / "train_pose_rgb_3ch2" / "weights" / "best.pt"
        _rgbd_def = _REPO_ROOT / "runs" / "pose" / "train_pose_rgbd_4ch" / "weights" / "best.pt"
        if not self.eval_panel.le_eval_rgb.text().strip() and _rgb_def.exists():
            self.eval_panel.le_eval_rgb.setText(str(_rgb_def))
        if not self.eval_panel.le_eval_rgbd.text().strip() and _rgbd_def.exists():
            self.eval_panel.le_eval_rgbd.setText(str(_rgbd_def))
        self._refresh_4ch_label()
        self._on_mode_changed()

    # ================================================================ 模式切换
    def _on_mode_changed(self):
        """切换"准备新数据集 / 使用已有数据集"后同步 yaml 路径与数据集统计。

        Panel 内部已处理控件启停，Controller 只负责业务逻辑。
        """
        self._update_yaml_paths()
        if self.dataset_panel.is_existing_mode():
            self._check_existing_dataset()
        self._refresh_eval_test_info()
        self._refresh_ablate_info()

    def _update_yaml_paths(self):
        """根据当前模式统一计算 self.rgb_yaml / self.rgbd_yaml。"""
        if self.dataset_panel.is_existing_mode():
            root = self.dataset_panel.le_existing.text().strip()
            base = Path(root) if root else None
            self.rgb_yaml = base / "rgb" / "data_rgb.yaml" if base else None
            # rgbd_yaml 保持由 _check_existing_dataset 扫描设定
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

    # ================================================================ 已有数据集检查
    def _check_existing_dataset(self):
        """检查已选数据集根目录：扫描 RGBD 变体 + 现场统计 + report 合并。"""
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
            self.dataset_panel.set_rgb_yaml_text("-")
            self.dataset_panel.clear_variants()
            self.dataset_panel.set_rgbd_yaml_text("-")
            return
        # RGB 固定路径，只读展示
        self.rgb_yaml = ry
        self.dataset_panel.set_rgb_yaml_text(str(ry))

        # 扫描所有 4ch 变体
        variants = DatasetService.scan_rgbd_variants(base)
        if not variants:
            self._log("[错误] 未找到任何 4 通道 RGBD 数据集（需 yaml 声明 rgbd:true 且 channels:4）")
            self.dataset_panel.reset_stats()
            self.dataset_panel.set_ready("已有数据集不完整")
            self.rgbd_yaml = None
            self.dataset_panel.clear_variants()
            self.dataset_panel.set_rgbd_yaml_text("-")
            return

        # 刷新下拉框并应用当前选择（内部设定 self.rgbd_yaml + 显示 + 持久化）
        saved = self._settings_store.get("rgbd_variant")
        self._rgbd_variants = variants
        self.dataset_panel.populate_variants(variants, saved)
        # populate_variants 发出 rgbdVariantChanged → _apply_rgbd_variant（同步执行）
        # 此时 self.rgbd_yaml 已被 _apply_rgbd_variant 设定

        # 1) 现场统计
        rgbd_sub = self.rgbd_yaml.parent.name if self.rgbd_yaml else None
        stat = DatasetService.stat_existing_dataset(base, rgbd_sub, self.rgbd_yaml)
        if stat is None:
            self._log("[警告] 已有数据集 images 目录缺失，无法现场统计")
            self.dataset_panel.reset_stats()
            self.dataset_panel.set_ready("已有数据集不完整")
            return

        # mismatch 警告
        if "mismatch_warning" in stat:
            self._log(f"[警告] {stat['mismatch_warning']}")

        # 2) report 有效字段优先，缺失/None 用现场统计补齐
        merged, ready = DatasetService.merge_report(stat, base)

        # 3) 回写显示
        self.dataset_panel.set_stats(merged, ready)
        self._log("[状态] 已有数据集已加载")

    # ================================================================ RGBD 变体
    def _apply_rgbd_variant(self, dir_name):
        """应用选中的 RGBD 变体：同步 rgbd_yaml、显示、QSettings、刷新只读信息。"""
        variant = next((v for v in self._rgbd_variants if v[0] == dir_name), None)
        if variant is None:
            self.dataset_panel.set_rgbd_yaml_text("-")
            self.dataset_panel.set_rgbd_ds_text("-")
            self.rgbd_yaml = None
            return
        dname, y = variant
        self.rgbd_yaml = Path(y)
        self.dataset_panel.set_rgbd_yaml_text(str(y))
        self.dataset_panel.set_rgbd_ds_text(dname)
        self._settings_store.set("rgbd_variant", dname)
        self._settings_store.sync()
        self._log(f"[状态] 当前 RGBD 变体: {dname}  ->  {y}")
        self._refresh_eval_test_info()
        self._refresh_ablate_info()

    # ================================================================ 测试数据只读信息
    def _refresh_eval_test_info(self):
        """刷新"测试结果"区上方"当前测试数据"只读显示。"""
        ep = self.eval_panel
        rgb_dir, rgb_cnt, rgb_ids = DatasetService.resolve_test_set(self.rgb_yaml)
        rgbd_dir, rgbd_cnt, rgbd_ids = DatasetService.resolve_test_set(self.rgbd_yaml)
        id_text = DatasetService.format_id_consistency(
            rgb_cnt, rgbd_cnt, rgb_ids, rgbd_ids)
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
        """刷新"Depth 消融测试"只读信息。"""
        ep = self.eval_panel
        ep.set_ablate_info({
            "rgbd_yaml": str(self.rgbd_yaml) if self.rgbd_yaml else None,
            "rgbd_pt": self.eval_panel.le_eval_rgbd.text().strip() or None,
        })

    # ================================================================ QSettings
    def _restore_settings(self):
        """启动时恢复上一次保存的路径到对应输入框。"""
        for le, key in self._settings_keys.items():
            v = self._settings_store.get(key)
            if v:
                le.setText(str(v))

    def _save_settings(self):
        """把当前所有路径写入 QSettings。"""
        for le, key in self._settings_keys.items():
            t = le.text().strip()
            if t:
                self._settings_store.set(key, t)
        self._settings_store.sync()

    def save_settings(self):
        """供 PosePage / MainWindow closeEvent 调用。"""
        self._save_settings()

    # ================================================================ 状态/日志
    def _set_status(self, text):
        self.status_log_panel.set_status(text)

    def _log(self, text):
        self.status_log_panel.append_log(text)

    # ================================================================ 控件启停
    def _disable_all(self, disable):
        """运行期间禁用所有操作按钮。"""
        self.dataset_panel.set_prepare_enabled(not disable)
        self.train_panel.set_operation_buttons_enabled(not disable)
        self.eval_panel.set_operation_buttons_enabled(not disable)

    def _restore_controls(self):
        """任务结束后恢复"当前模式应有的控件状态"。"""
        self.train_panel.set_operation_buttons_enabled(True)
        self.dataset_panel.set_prepare_enabled(self.dataset_panel.is_new_mode())
        self.eval_panel.set_operation_buttons_enabled(True)

    # ================================================================ 启动子进程
    def _run(self, op, cmd, status):
        """启动子进程执行指定命令。

        op: 操作标识（prepare / train_rgb / build / train_rgbd / eval_* / ablate）
        cmd: 完整命令列表（含 script 路径 + args），由 CommandService 构造
        status: 状态栏显示文本
        """
        if self._runner.is_running:
            self._log("[警告] 已有任务在运行，忽略本次请求")
            return
        py = self.train_panel.le_py.text().strip()
        if not py:
            self._log("[错误] 未设置 Python 训练环境路径，请先选择训练用 Python 解释器")
            return
        if not Path(py).exists():
            self._log(f"[错误] Python 训练环境不存在: {py}")
            return
        self._disable_all(True)
        self.status_log_panel.set_stop_enabled(True)
        self._set_status(status)
        self._log(">>> " + " ".join([py, *cmd]))
        self._runner.start(op, py, cmd, str(_REPO_ROOT), _REPO_ROOT)

    # ================================================================ prepare
    def _on_prepare(self):
        rgb = self.dataset_panel.le_rgb.text().strip()
        depth_npy = self.dataset_panel.le_depth_npy.text().strip()
        label = self.dataset_panel.le_label.text().strip()
        out = self.dataset_panel.le_out.text().strip()
        cls = self.dataset_panel.le_class.text().strip() or "hand"
        if not (rgb and depth_npy and label and out):
            self._log("[错误] RGB / Raw Depth NPY / Labels / 输出目录 必须全部填写")
            return
        try:
            low = float(self.dataset_panel.le_low.text().strip() or "1100")
            high = float(self.dataset_panel.le_high.text().strip() or "1850")
        except ValueError:
            self._log("[错误] Depth low/high(mm) 必须为数字")
            return
        if not (high > low):
            self._log("[错误] Depth low(mm) 必须严格小于 high(mm)")
            return
        cmd = CommandService.build_prepare_command(rgb, depth_npy, label, out, cls, low, high)
        self._run("prepare", cmd, "处理中")

    # ================================================================ 4ch 权重
    def _on_base_weight_changed(self, _base_text):
        """基础权重文本变更后刷新 4ch 显示。"""
        self._refresh_4ch_label()

    def _refresh_4ch_label(self):
        """根据当前基础权重刷新 4ch 权重显示路径。"""
        base = self.train_panel.le_base.text().strip()
        if base:
            path = CommandService.derive_4ch_path(base)
            self.train_panel.set_4ch_text(str(path))
        else:
            self.train_panel.set_4ch_text("-")

    # ================================================================ 训练
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
            self._log('[错误] 未确定 RGB data yaml：请先在"使用已有数据集"选择目录，'
                      '或在"准备新数据集"执行准备数据集')
            return
        cmd = CommandService.build_train_rgb_command(
            yaml, self.train_panel.le_fliplr.text().strip(),
            self.train_panel.le_patience.text().strip(), base)
        self._run("train_rgb", cmd, "训练中")

    def _on_build(self):
        base = self.train_panel.le_base.text().strip()
        if not base:
            self._log("[错误] 请先选择基础 Pose 权重（yolov8n-pose.pt）")
            return
        if not Path(base).exists():
            self._log(f"[错误] 基础权重不存在: {base}")
            return
        cmd = CommandService.build_4ch_command(base)
        self._run("build", cmd, "处理中")

    def _on_train_rgbd(self):
        base = self.train_panel.le_base.text().strip()
        if not base:
            self._log("[错误] 请先选择基础 Pose 权重（yolov8n-pose.pt）")
            return
        w4 = CommandService.derive_4ch_path(base)
        if not w4.exists():
            self._log(f'[错误] 4通道权重不存在: {w4}，请先点击"生成4ch权重"')
            return
        yaml = self.rgbd_yaml
        if not yaml or not Path(yaml).exists():
            self._log('[错误] 未确定 RGBD data yaml：请先在"使用已有数据集"选择目录，'
                      '或在"准备新数据集"执行准备数据集')
            return
        cmd = CommandService.build_train_rgbd_command(
            yaml, self.train_panel.le_fliplr.text().strip(),
            self.train_panel.le_patience.text().strip(), str(w4))
        self._run("train_rgbd", cmd, "训练中")

    # ================================================================ 测试集评估
    def _on_eval_rgb(self):
        self._start_eval_leg("rgb")

    def _on_eval_rgbd(self):
        self._start_eval_leg("rgbd")

    def _on_eval_cmp(self):
        # 对比测试启动前再次校验 RGB 与 RGBD test ID 一致性
        _, rgb_cnt, rgb_ids = DatasetService.resolve_test_set(self.rgb_yaml)
        _, rgbd_cnt, rgbd_ids = DatasetService.resolve_test_set(self.rgbd_yaml)
        if rgb_ids is None or rgbd_ids is None:
            self._log("[错误] 对比测试被拒绝：RGB 或 RGBD test 目录缺失/无法解析，"
                      "请先确认数据集已正确加载")
            self._set_status("对比测试被拒绝")
            return
        if rgb_ids != rgbd_ids:
            self._log(DatasetService.format_id_mismatch_error(
                rgb_cnt, rgbd_cnt, rgb_ids, rgbd_ids))
            self._set_status("对比测试被拒绝")
            return
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
            self._log(f'[错误] 未确定 {name} data yaml：请先在"使用已有数据集"选择数据集目录'
                      f'（需含 dataset/hand/{tag}/data_{tag}.yaml）')
            return False
        self._last_eval_result = None
        cmd = CommandService.build_eval_command(weights, yaml)
        self._run(f"eval_{leg}", cmd, f"评估{name}中")
        return True

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

    # ================================================================ Depth 消融
    def _on_ablate_depth(self):
        weights = self.eval_panel.le_eval_rgbd.text().strip()
        yaml = self.rgbd_yaml
        ypath = str(yaml) if yaml else None
        if not weights or not Path(weights).exists():
            self._log(f"[错误] RGBD 权重不存在: {weights}")
            return
        if not ypath or not Path(ypath).exists():
            self._log('[错误] 未确定 RGBD data yaml：请先在"使用已有数据集"选择数据集目录')
            return
        test_dir, test_cnt, _ = DatasetService.resolve_test_set(self.rgbd_yaml)
        self._log("[Depth Ablation]")
        self._log(f"Model: {weights}")
        self._log(f"Data: {ypath}")
        self._log("Split: test")
        self._log("Variants:\n- true_depth\n- zero_depth")
        self._log(f"Test images: {test_dir}")
        self._log(f"Test count: {test_cnt}")
        out = _REPO_ROOT / "runs" / "ablation" / "depth_ablation_gui"
        cmd = CommandService.build_depth_ablation_command(weights, yaml, out)
        self._run("ablate", cmd, "Depth 消融中")

    # ================================================================ ProcessRunner 回调
    def _on_started(self, op_name):
        self._log(f"[启动] {op_name}")

    def _on_stdout(self, data):
        self._log(data)
        for line in data.splitlines():
            if "POSE_EVAL_JSON" in line:
                result = ResultParser.parse_pose_eval_json(line)
                if result is not None:
                    self._last_eval_result = result
                else:
                    self._log("[警告] 评估结果 JSON 解析失败")
            elif "DEPTH_ABLATION_JSON" in line:
                result = ResultParser.parse_depth_ablation_json(line)
                if result is not None:
                    self._last_ablate_result = result
                else:
                    self._log("[警告] 消融结果 JSON 解析失败")

    def _on_stderr(self, data):
        self._log("[err] " + data)

    def _on_stop(self):
        """请求停止当前运行中的子进程。"""
        if self._runner.is_running:
            self._log("[停止] 正在请求停止当前任务...")
            self._runner.stop()
        else:
            self._log("[停止] 当前没有运行中的任务")

    def _on_finished(self, op, code):
        # ---- 评估（eval_*）分支 ----
        if op and op.startswith("eval_"):
            leg = op.split("_", 1)[1]
            if self._runner.user_stopped:
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
            # 收尾
            self.status_log_panel.set_stop_enabled(False)
            self._restore_controls()
            return

        # ---- Depth 消融 (ablate) 分支 ----
        if op == "ablate":
            if self._runner.user_stopped:
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
            self.status_log_panel.set_stop_enabled(False)
            self._restore_controls()
            return

        # ---- prepare / train / build 分支 ----
        if self._runner.user_stopped:
            self._log("[结束] 用户已停止当前任务")
            self._set_status("已停止")
        else:
            ok = (code == 0)
            self._log(f"[结束] exit_code={code} -> {'完成' if ok else '失败'}")
            self._set_status("完成" if ok else "失败")
        self.status_log_panel.set_stop_enabled(False)
        self._restore_controls()
        if op == "prepare" and code == 0 and not self._runner.user_stopped:
            self._load_prep_status()

    # ================================================================ prepare 完成后
    def _load_prep_status(self):
        """prepare 成功后从产物路径读取 dataset_report.json 并刷新数据集状态。"""
        out = self.dataset_panel.le_out.text().strip()
        cls = self.dataset_panel.le_class.text().strip() or "hand"
        base = Path(out) / "dataset" / cls
        # 准备模式完成后统一设置训练用的 yaml 路径
        self.rgb_yaml = base / "rgb" / "data_rgb.yaml"
        self.rgbd_yaml = base / "rgbd" / "data_rgbd.yaml"
        d = DatasetService.load_report_json(out, cls)
        if d is None:
            self._log("[警告] 未找到 dataset_report.json，无法刷新数据集状态")
            return
        merged, ready = DatasetService.report_to_stats(d)
        self.dataset_panel.set_stats(merged, ready)
        if d.get("validation_passed"):
            self._log("[状态] 数据集已就绪")
        else:
            self._log("[状态] 数据集校验未通过，请检查")
        # 准备模式完成后同步测试数据只读显示
        self._refresh_eval_test_info()
        self._refresh_ablate_info()
