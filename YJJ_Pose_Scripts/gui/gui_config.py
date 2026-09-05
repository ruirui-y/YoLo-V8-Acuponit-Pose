"""YJJ RGBD Pose GUI 路径配置（与业务脚本解耦的纯常量）。

仅存放“脚本/目录/权重”相关的绝对路径推导，不依赖 PySide6，
也不包含任何 UI 或子进程逻辑，方便被 main_window 与 panels 复用。
"""
from pathlib import Path

# 脚本路径（相对本文件定位，不写死绝对路径）
# gui_config.py 位于 YJJ_Pose_Scripts/gui/，故 parents[2] 即项目根
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = {
    "prepare": _REPO_ROOT / "YJJ_Pose_Scripts" / "01_data" / "prepare_pose_rgbd_dataset.py",
    "cross_rebuild": _REPO_ROOT / "YJJ_Pose_Scripts" / "01_data" / "rebuild_cross_subject_testset.py",
    "train_rgb": _REPO_ROOT / "YJJ_Pose_Scripts" / "03_train" / "train_rgb_pose.py",
    "train_rgbd": _REPO_ROOT / "YJJ_Pose_Scripts" / "03_train" / "train_rgbd_pose.py",
    "build": _REPO_ROOT / "YJJ_Pose_Scripts" / "02_model" / "build_rgbd_pose_weights.py",
    "eval": _REPO_ROOT / "YJJ_Pose_Scripts" / "04_eval" / "test_pose_rgbd.py",
    "ablate": _REPO_ROOT / "YJJ_Pose_Scripts" / "04_eval" / "ablate_rgbd_depth.py",
}
# 权重目录：基础母版固定放此处，4ch 权重统一生成到 4ch/ 子目录
_WEIGHTS_DIR = _REPO_ROOT / "YJJ_Pose_Scripts" / "weights"
_FOURCH_DIR = _WEIGHTS_DIR / "4ch"
_DEFAULT_BASE_WEIGHT = _WEIGHTS_DIR / "yolov8n-pose.pt"
