"""QSettings 薄包装（通用 GUI 基础设施）。

各 Feature Controller 通过本类读写自己命名空间下的配置，
避免所有设置堆到 MainWindow。

本模块同时提供独立 helper `make_start_dir_provider`，供 Controller 把
"le → QSettings 上次保存路径" 的查询能力以纯 callable 形式注入 Panel，
Panel 因此无需持有或反向调用 Controller。
"""
from PySide6.QtCore import QSettings


class SettingsStore:
    def __init__(self, org="YJJ", app="RGBDPoseTrainGUI"):
        self._settings = QSettings(org, app)

    def get(self, key, default=None):
        v = self._settings.value(key)
        return v if v is not None else default

    def set(self, key, value):
        self._settings.setValue(key, value)

    def sync(self):
        self._settings.sync()


def make_start_dir_provider(settings_store, le_to_key):
    """构造 Panel browse 起始目录 QSettings fallback 的纯 callable。

    返回值：callable(le) -> str
        - 命中 QSettings 上次保存的同 key 路径 -> 返回 str(path)
        - 未命中或 le 不在映射里 -> 返回 "" （Panel 自己补 _REPO_ROOT 兜底）

    设计目标：
    - 把"le → settings key → QSettings 上次路径"的查询逻辑下沉到基础设施层，
      Controller 只负责把 (settings_store, le_to_key) 这两份数据传进来；
    - 返回的是闭包，不是 Controller 实例方法，Panel 不持有 Controller。
    """
    def _provider(le):
        key = le_to_key.get(le)
        if key:
            v = settings_store.get(key)
            if v:
                return str(v)
        return ""
    return _provider

