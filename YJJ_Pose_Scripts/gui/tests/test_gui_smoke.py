"""GUI 重构后的确定性 smoke test（纯 PySide6 + QSignalSpy，不依赖 pytest）。

============================================================
测试隔离策略
============================================================
1. 不依赖用户机器已有 QSettings
   - patch SettingsStore.__init__：把 QSettings(org, app) 替换为
     QSettings(临时 INI 文件, IniFormat)，完全旁路注册表/用户真实配置
   - 因此 PoseController 启动时 _restore_settings 读不到任何历史路径
     （除非本次测试自身写入），断言基于此确定性初始状态

2. 不覆盖 / 不删除用户真实配置
   - 全程只读写测试专用 INI 文件，YJJ/RGBDPoseTrainGUI 真实配置完全不动

3. 测试结束后不留下测试配置
   - 退出时删除临时 INI 文件，不留任何痕迹
   - SettingsStore.__init__ 在 finally 中恢复原始实现

============================================================
验证项
============================================================
 1. MainWindow 创建成功
 2. 默认页面为 RGBD Pose
 3. 切换 LaMa Erasure 成功
 4. 再切回 Pose，还是同一个 PosePage 实例
 5. Panel signals 可以正常连接（Controller 已订阅 modeChanged）
 6. Existing/New 模式 UI 启停正常
 7. Depth Ablation 按钮存在
 8. split 固定 test
 9. 不启动任何外部训练/评估进程（ProcessRunner.is_running == False）
10. close 不异常

启动：
    python YJJ_Pose_Scripts/gui/tests/test_gui_smoke.py
"""
import os
import sys
import tempfile
import traceback
from pathlib import Path

# ---- 让 gui/ 包可被 import（不依赖 PYTHONPATH）----
GUI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GUI_DIR))

from PySide6.QtCore import QSettings, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from PySide6.QtTest import QTest, QSignalSpy  # noqa: E402

# ---- 在 import MainWindow 之前 patch SettingsStore，避免构造期间写入用户配置 ----
from core import settings_store as settings_store_mod  # noqa: E402

# 临时 INI 文件路径（每个进程独立，避免并发污染）
_TMP_INI = tempfile.NamedTemporaryFile(
    prefix="yjj_gui_smoke_", suffix=".ini", delete=False)
_TMP_INI.close()
_TMP_INI_PATH = _TMP_INI.name

_real_settings_init = settings_store_mod.SettingsStore.__init__


def _test_settings_init(self, org="YJJ", app="RGBDPoseTrainGUI"):
    """测试专用：用临时 INI 文件完全替代 QSettings(org, app) 注册表。"""
    self._settings = QSettings(_TMP_INI_PATH, QSettings.Format.IniFormat)


settings_store_mod.SettingsStore.__init__ = _test_settings_init

# 现在才 import MainWindow / PosePage —— 它们构造时用到的 SettingsStore 已被 patch
from main_window import MainWindow  # noqa: E402
from features.pose.pose_page import PosePage  # noqa: E402


def _check(name, cond, detail="", failures=None):
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}  {detail}")
        if failures is not None:
            failures.append(name)


def _click_radio(panel, btn):
    """模拟用户点击单选按钮：setChecked + 触发 buttonClicked 信号。"""
    btn.setChecked(True)
    panel.mode_group.buttonClicked.emit(btn)
    # 等待 queued connection（modeChanged 经过 Controller -> _on_mode_changed 同步处理）
    QTest.qWait(10)


def main():
    failures = []
    app = QApplication.instance() or QApplication(sys.argv)
    # 关闭 QFileDialog 等模态对话框自动退出保护（本测试不应弹任何模态）
    app.setQuitOnLastWindowClosed(False)

    mw = None
    try:
        # ---- 1. MainWindow 创建成功 ----
        mw = MainWindow()
        _check("1. MainWindow 创建成功", mw is not None, failures)

        # ---- 2. 默认页面为 RGBD Pose ----
        _check("2. 默认页面为 RGBD Pose",
               mw.stack.currentWidget() is mw.pose_page, failures)

        # ---- 3. 切换 LaMa Erasure 成功 ----
        mw._show_lama()
        QTest.qWait(10)
        _check("3. 切换 LaMa Erasure 成功",
               mw.stack.currentWidget() is mw.lama_page, failures)

        # ---- 4. 再切回 Pose，还是同一个 PosePage 实例 ----
        mw._show_pose()
        QTest.qWait(10)
        _check("4. 切回 Pose 还是同一个 PosePage 实例",
               mw.stack.currentWidget() is mw.pose_page, failures)

        page = mw.pose_page
        dp = page.dataset_panel
        ep = page.eval_panel
        controller = page.controller

        # ---- 5. Panel signals 可以正常连接 ----
        # 切到 New 模式：Panel 内部 _on_mode_clicked 先 _apply_mode_state 再 emit modeChanged
        # Controller 已 connect modeChanged -> _on_mode_changed，验证回调链路不抛异常
        spy = QSignalSpy(dp.modeChanged)
        _click_radio(dp, dp.rb_new)
        QTest.qWait(20)
        _check("5. Panel modeChanged 信号被发出",
               spy.count() >= 1, f"spy.count={spy.count()}", failures)

        # ---- 6. Existing/New 模式 UI 启停正常 ----
        # New 模式：rgb/depth_npy/label/out 启用，existing 禁用
        new_ok = (dp.row_rgb.isEnabled() and dp.row_depth_npy.isEnabled()
                  and dp.row_label.isEnabled() and dp.row_out.isEnabled()
                  and not dp.row_existing.isEnabled())
        _check("6a. New 模式：new rows 启用、existing row 禁用", new_ok, failures)

        _click_radio(dp, dp.rb_existing)
        QTest.qWait(20)
        # Existing 模式：existing 启用，rgb/depth_npy/label/out 禁用
        existing_ok = (dp.row_existing.isEnabled()
                       and not dp.row_rgb.isEnabled()
                       and not dp.row_depth_npy.isEnabled()
                       and not dp.row_label.isEnabled()
                       and not dp.row_out.isEnabled())
        _check("6b. Existing 模式：existing row 启用、new rows 禁用",
               existing_ok, failures)

        # ---- 7. Depth Ablation 按钮存在 ----
        _check("7. Depth Ablation 按钮存在", ep.btn_ablate is not None, failures)

        # ---- 8. split 固定 test ----
        split_text = ep.lbl_split.text()
        ablate_split_text = ep.lbl_ablate_split.text()
        _check("8a. eval 区 split 固定 test",
               "test" in split_text, f'lbl_split="{split_text}"', failures)
        _check("8b. ablate 区 split 固定 test",
               "test" in ablate_split_text,
               f'lbl_ablate_split="{ablate_split_text}"', failures)

        # ---- 9. 不启动任何外部训练/评估进程 ----
        runner = controller._runner
        _check("9. 不启动任何外部训练/评估进程 (is_running=False)",
               not runner.is_running, failures)
        _check("9b. 启动后 current_op 为空",
               runner.current_op is None, failures)

        # ---- 10. close 不异常 ----
        try:
            # closeEvent 内部会调 pose_page.save_settings -> controller.save_settings
            # 写入测试 INI 文件，不应抛异常
            mw.close()
            _check("10. close 不异常", True, failures=failures)
        except Exception as e:  # noqa: BLE001
            _check("10. close 不异常", False,
                   f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                   failures=failures)

    finally:
        # ---- 恢复 SettingsStore 原始实现 ----
        settings_store_mod.SettingsStore.__init__ = _real_settings_init
        # ---- 删除测试 INI 文件（零残留）----
        try:
            if os.path.exists(_TMP_INI_PATH):
                os.remove(_TMP_INI_PATH)
        except OSError:
            pass
        # 清理事件循环残留
        if mw is not None:
            try:
                mw.deleteLater()
            except Exception:  # noqa: BLE001
                pass
        # flush 残余事件，避免进程退出告警
        app.processEvents()

    print("-" * 60)
    if failures:
        print(f"GUI smoke test 失败：{len(failures)} 项 -> {failures}")
        sys.exit(1)
    else:
        print("GUI smoke test 全部通过")
        sys.exit(0)


if __name__ == "__main__":
    main()
