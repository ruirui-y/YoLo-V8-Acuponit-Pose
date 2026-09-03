"""通用 QProcess 管理器（GUI 基础设施，不含任何业务逻辑）。

封装 QProcess 的启动 / 终止 / 强杀 / stdout / stderr，通过 Qt Signal 对外暴露。
任何 Feature（Pose / Lama 等）均可复用本层；不包含 YOLO / RGBD / Pose 特有逻辑。
"""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QObject, QTimer, Signal


class ProcessRunner(QObject):
    # ---- 对外信号 ----
    started = Signal(str)            # op_name
    stdoutReceived = Signal(str)
    stderrReceived = Signal(str)
    finished = Signal(str, int)      # op_name, exit_code
    runningChanged = Signal(bool)    # is_running

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process: QProcess = QProcess(self)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.readyReadStandardError.connect(self._on_stderr)
        self._process.started.connect(self._on_started)
        self._process.finished.connect(self._on_finished)
        self._current_op: str | None = None
        self._user_stopped: bool = False

    @property
    def is_running(self) -> bool:
        return self._process.state() == QProcess.ProcessState.Running

    @property
    def current_op(self) -> str | None:
        return self._current_op

    @property
    def user_stopped(self) -> bool:
        return self._user_stopped

    def start(
        self,
        op_name: str,
        executable: str,
        args: list[str],
        working_dir: str,
        repo_root: str | Path,
    ) -> bool:
        if self.is_running:
            return False
        self._current_op = op_name
        self._user_stopped = False
        # 工作目录设为项目根；PYTHONPATH 前置项目根，确保本地 ultralytics 可被找到
        env = QProcessEnvironment.systemEnvironment()
        existing = env.value("PYTHONPATH", "")
        new_pp = str(repo_root) if not existing else f"{repo_root}{os.pathsep}{existing}"
        env.insert("PYTHONPATH", new_pp)
        self._process.setProcessEnvironment(env)
        self._process.setWorkingDirectory(str(working_dir))
        cmd = [str(a) for a in args]
        self._process.start(executable, cmd)
        return True

    def stop(self) -> bool:
        """请求停止当前运行中的子进程：先 terminate，3s 后若仍存活则 kill。"""
        if not self.is_running:
            return False
        self._user_stopped = True
        self._process.terminate()
        QTimer.singleShot(3000, self._check_stop)
        return True

    def _check_stop(self) -> None:
        if self._user_stopped and self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()

    def _on_stdout(self) -> None:
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", "replace").rstrip("\n")
        if data:
            self.stdoutReceived.emit(data)

    def _on_stderr(self) -> None:
        data = bytes(self._process.readAllStandardError()).decode("utf-8", "replace").rstrip("\n")
        if data:
            self.stderrReceived.emit(data)

    def _on_started(self) -> None:
        self.started.emit(self._current_op)
        self.runningChanged.emit(True)

    def _on_finished(self, code: int, _status: QProcess.ExitStatus) -> None:
        op = self._current_op
        self._current_op = None
        self.finished.emit(op, code)
        self.runningChanged.emit(False)
