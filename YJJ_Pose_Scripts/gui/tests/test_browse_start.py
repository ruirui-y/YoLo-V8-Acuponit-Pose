"""DatasetPanel browse 起始目录优先级（纯逻辑）测试。

覆盖需求：
9.  当前输入框有路径（目录 -> 该目录；文件 -> parent）-> 从当前路径打开
10. 无当前路径时：字段级 last dir > 全局 last dir > 项目根
11. Canonical Template（文件）Browse 从上次文件 parent 打开

启动：
    python YJJ_Pose_Scripts/gui/tests/test_browse_start.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.pose.panels.dataset_panel import resolve_browse_start  # noqa: E402


def _check(name: str, cond: bool, failures: list[str], detail: str = "") -> None:
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}  {detail}")
        failures.append(name)


def main() -> None:
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="browse_start_") as td:
        root = Path(td)
        d1 = root / "alpha"
        d2 = root / "beta"
        d1.mkdir()
        d2.mkdir()
        f1 = d1 / "canonical_template.txt"
        f1.write_text("x", encoding="utf-8")
        fallback = root / "project_root"

        # 9. 当前输入框有目录 -> 用当前目录
        _check("9a. 当前目录优先于 last_dir",
               resolve_browse_start(str(d2), str(d1), str(root), str(fallback))
               == str(d2), failures)
        _check("9b. 当前文件 -> 用其 parent",
               resolve_browse_start(str(f1), str(d2), str(root), str(fallback))
               == str(d1), failures)
        _check("9c. 当前路径不存在 -> 忽略并降级",
               resolve_browse_start(str(d1 / "nope"), str(d2), None, str(fallback))
               == str(d2), failures)

        # 10. 无当前路径：字段 last > 全局 last > fallback
        _check("10a. 字段级 last_dir 优先于全局",
               resolve_browse_start("", str(d2), str(d1), str(fallback)) == str(d2),
               failures)
        _check("10b. 无字段 last 时用全局 last_dir",
               resolve_browse_start("", None, str(d1), str(fallback)) == str(d1),
               failures)
        _check("10c. 全空 -> fallback（项目根）",
               resolve_browse_start("", None, None, str(fallback)) == str(fallback),
               failures)

        # 11. 文件选择：上次目录指向文件 parent
        _check("11a. last_dir 指向模板文件 parent 时定位到该目录",
               resolve_browse_start("", str(d1), str(d2), str(fallback)) == str(d1),
               failures)

    print("-" * 60)
    if failures:
        print(f"RESULT: {len(failures)} FAILED -> {failures}")
        sys.exit(1)
    print("RESULT: ALL PASSED")


if __name__ == "__main__":
    main()
