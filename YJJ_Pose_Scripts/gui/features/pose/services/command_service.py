"""命令构造服务（不负责 QProcess，只负责"应该执行哪个 script + 哪些 args"）。

把 PoseController 中大量 ["--data", ..., "--weights", ...] 拼接代码集中到此，
Controller 只调用 build_xxx_command(...) 获取 args list。
"""
from pathlib import Path

from gui_config import _FOURCH_DIR, _SCRIPTS


class CommandService:

    @staticmethod
    def derive_4ch_path(base_text):
        """由基础权重派生 4ch 输出路径：固定放到 weights/4ch/<stem>_4ch.pt。"""
        base = Path(base_text)
        return _FOURCH_DIR / f"{base.stem}_4ch.pt"

    @staticmethod
    def build_prepare_command(rgb, depth_npy, label, out, cls, low, high):
        """构造 prepare 子进程参数。"""
        return [
            str(_SCRIPTS["prepare"]),
            "--rgb_dir", rgb, "--depth_npy_dir", depth_npy,
            "--label_dir", label, "--output_dir", out, "--class_name", cls,
            "--depth_low", str(low), "--depth_high", str(high),
        ]

    @staticmethod
    def build_train_rgb_command(yaml, fliplr, patience, weights):
        """构造 RGB 训练子进程参数。"""
        return [
            str(_SCRIPTS["train_rgb"]),
            "--data", str(yaml),
            "--fliplr", str(fliplr),
            "--patience", str(patience),
            "--weights", str(weights),
        ]

    @staticmethod
    def build_train_rgbd_command(yaml, fliplr, patience, weights):
        """构造 RGBD 训练子进程参数。"""
        return [
            str(_SCRIPTS["train_rgbd"]),
            "--data", str(yaml),
            "--fliplr", str(fliplr),
            "--patience", str(patience),
            "--weights", str(weights),
        ]

    @staticmethod
    def build_4ch_command(base):
        """构造 4ch 权重生成子进程参数。"""
        out = CommandService.derive_4ch_path(base)
        out.parent.mkdir(parents=True, exist_ok=True)
        return [str(_SCRIPTS["build"]), "--input", str(base), "--output", str(out)]

    @staticmethod
    def build_eval_command(weights, yaml):
        """构造 test 集评估子进程参数（split 固定 test）。"""
        return [
            str(_SCRIPTS["eval"]),
            "--weights", str(weights), "--data", str(yaml),
            "--split", "test", "--imgsz", "640", "--batch", "4",
        ]

    @staticmethod
    def build_depth_ablation_command(weights, yaml, out):
        """构造 Depth 消融子进程参数（仅 true_depth / zero_depth，split 固定 test）。"""
        return [
            str(_SCRIPTS["ablate"]),
            "--weights", str(weights),
            "--rgbd-yaml", str(yaml) if yaml else "",
            "--variants", "true_depth", "zero_depth",
            "--split", "test",
            "--out", str(out),
        ]
