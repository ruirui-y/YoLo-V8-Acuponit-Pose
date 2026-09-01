"""YJJ RGBD Pose GUI 各 UI 面板集合。

每个 Panel 只负责自身控件的创建与“纯展示”方法，不启动子进程；
子进程协调 / QSettings / QProcess / 业务逻辑统一由 main_window.MainWindow 处理。
"""
from .dataset_panel import DatasetPanel
from .eval_panel import EvalPanel
from .status_log_panel import StatusLogPanel
from .train_panel import TrainPanel

__all__ = ["DatasetPanel", "EvalPanel", "StatusLogPanel", "TrainPanel"]
