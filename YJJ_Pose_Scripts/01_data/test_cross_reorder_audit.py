"""Cross-subject canonical reorder 全量 audit 单元测试（仅本任务功能）。

覆盖（临时目录 + 模拟 fixture，不碰真实 dataset）：
1. 第一张 residual>阈值时不得提前停止，必须扫完全部
2. 101 张模拟：total/blocked 统计正确
3. <=0.15 / <=0.16 / <=0.18 分桶正确
4. min / median / p95 / max 计算正确
5. best/second mapping 相同 -> 正确计入 same
6. best/second mapping 不同 -> 正确计入 different
7. 存在 production BLOCK -> validate 最终 BLOCK（不返回 PASS 数据）
8. 全部 production OK -> validate PASS（reorder_ok=True, blocked=0）
9. AMBIGUOUS 单张仍继续扫描其它，最终整体 BLOCK
10. 异常 ERROR 单张仍完成其余扫描，report 记录 message，最终整体 BLOCK

启动：
    python YJJ_Pose_Scripts/01_data/test_cross_reorder_audit.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pose_label_reorder as plr          # noqa: E402
import rebuild_cross_subject_testset as xsub  # noqa: E402


def _check(name: str, cond: bool, failures: list[str], detail: str = "") -> None:
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}  {detail}")
        failures.append(name)


CANON = [
    (0.3125, 0.8333), (0.3281, 0.6250), (0.2969, 0.4167), (0.3594, 0.3958),
    (0.2656, 0.2292), (0.3906, 0.2500), (0.3438, 0.1250),
]


def _text(pts: list[tuple[float, float]]) -> str:
    tok = ["0", "0.500000", "0.500000", "0.900000", "0.900000"]
    for x, y in pts:
        tok += [f"{x:.6f}", f"{y:.6f}", "2"]
    return " ".join(tok) + "\n"


def _write_label(path: Path, pts: list[tuple[float, float]]) -> None:
    path.write_text(_text(pts), encoding="utf-8")


def _dup_last_two(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = list(pts)
    out[4] = out[5]  # 让两个点完全重合 -> candidate 歧义
    return out


def _make_audit(file: str, status: str, best_mapping: str, best_res: float,
                second_mapping: str | None = None, second_res: float | None = None,
                message: str = "") -> plr.LabelAudit:
    best_ref = "t_train_x.txt" if best_mapping else ""
    sec_ref = "t_train_y.txt" if second_mapping else ""
    gap = (second_res - best_res) if (second_res is not None and best_res is not None) else None
    return plr.LabelAudit(file, status, best_ref, best_mapping, best_res,
                          sec_ref, second_mapping, second_res, gap, message)


# ============================================================ 迷你 base/external fixture
def _make_base(base: Path) -> None:
    """最小 base：rgb/rgbd train+val 各 2 张（canonical P7）+ yaml。"""
    for side in ("rgb", "rgbd"):
        for split in ("train", "val"):
            for sub in ("images", "labels"):
                (base / side / sub / split).mkdir(parents=True, exist_ok=True)
            for sid in ("a", "b"):
                (base / side / "images" / split / f"color_{sid}.jpg").write_bytes(b"stub")
                _write_label(base / side / "labels" / split / f"color_{sid}.txt", CANON)
    base.joinpath("rgb", "data_rgb.yaml").write_text(
        "path: .\nnc: 1\nkpt_shape: [7, 3]\nchannels: 3\n", encoding="utf-8")
    base.joinpath("rgbd", "data_rgbd.yaml").write_text(
        "path: .\nnc: 1\nkpt_shape: [7, 3]\nchannels: 4\nrgbd: true\n", encoding="utf-8")


# 与 canonical 距离结构明显不同的另一“手型”（保证与 bank 匹配 residual 高且唯一）
_FIST = [
    (0.10, 0.62), (0.26, 0.55), (0.17, 0.40), (0.31, 0.33),
    (0.12, 0.20), (0.29, 0.14), (0.44, 0.48),
]


def _make_external_dir(root: Path, items: list[tuple[str, str]]) -> tuple[Path, Path, Path]:
    """items: [(sid, kind)] kind in ok/bad/amb/err；生成 img/lab/npy 目录。"""
    img_d = root / "ext_img"
    lab_d = root / "ext_lab"
    npy_d = root / "ext_npy"
    for d in (img_d, lab_d, npy_d):
        d.mkdir(parents=True, exist_ok=True)
    for sid, kind in items:
        (img_d / f"color_{sid}.jpg").write_bytes(b"stub")
        (npy_d / f"depth_{sid}.npy").write_bytes(b"stub")
        lp = lab_d / f"color_{sid}.txt"
        if kind == "bad":
            _write_label(lp, _FIST)              # 明显不同手型 -> residual 超限
        elif kind == "amb":
            _write_label(lp, _dup_last_two(CANON))
        elif kind == "err":
            lp.write_text("not a yolo pose label line\n", encoding="utf-8")
        else:
            _write_label(lp, CANON)
    return img_d, lab_d, npy_d


def main() -> None:
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="cross_audit_") as td:
        root = Path(td)

        print("[1] 第一张 blocked（residual/ambiguous）：必须扫完全部，不得提前退出")
        refs = [plr.Reference("r_canon.txt", CANON)]
        d = root / "case1"
        d.mkdir()
        _write_label(d / "z_bad.txt", _FIST)
        _write_label(d / "a_ok.txt", CANON)
        audits = plr.scan_bank_audit(refs, [("z_bad.txt", d / "z_bad.txt"),
                                            ("a_ok.txt", d / "a_ok.txt")])
        _check("1a. 两条都被扫描（未提前退出）", len(audits) == 2, failures)
        _check("1b. 第一条 blocked、第二条 OK（继续扫描）",
               audits[0].status != "OK" and audits[1].status == "OK",
               failures, [a.status for a in audits])
        _check("1c. 合成 RESIDUAL_BLOCK audit 带 best/second 诊断字段",
               _make_audit("x.txt", "RESIDUAL_BLOCK", "K0<-K1;K1<-K0;K2<-K2;K3<-K3;K4<-K4;K5<-K5;K6<-K6",
                           0.155, None, None, "best residual > 0.15").best_reference != ""
               and _make_audit("x.txt", "RESIDUAL_BLOCK", "K0<-K1;K1<-K0;K2<-K2;K3<-K3;K4<-K4;K5<-K5;K6<-K6",
                               0.155, None, None, "best residual > 0.15").message != "",
               failures)

        print("[2/3/4] 101 张合成 audit 统计（total/blocked/分桶/分位）")
        sim: list[plr.LabelAudit] = []
        residuals = []
        for i in range(95):                                   # OK, <=0.15
            r = 0.02 + 0.001 * (i % 10)
            sim.append(_make_audit(f"ok_{i}.txt", "OK", "K0<-K0;K1<-K1;K2<-K2;K3<-K3;K4<-K4;K5<-K5;K6<-K6", r,
                                   "K0<-K0;K1<-K1;K2<-K2;K3<-K3;K4<-K4;K5<-K5;K6<-K6", r + 0.01))
            residuals.append(r)
        for i in range(3):                                    # (0.15, 0.16]
            r = 0.155 + 0.001 * i
            sim.append(_make_audit(f"mid_{i}.txt", "RESIDUAL_BLOCK",
                                   "K0<-K1;K1<-K0;K2<-K2;K3<-K3;K4<-K4;K5<-K5;K6<-K6",
                                   r, None, None, "best residual > 0.15"))
            residuals.append(r)
        for i in range(3):                                    # (0.16, 0.18]
            r = 0.170 + 0.002 * i
            sim.append(_make_audit(f"hi_{i}.txt", "RESIDUAL_BLOCK",
                                   "K0<-K0;K1<-K1;K2<-K2;K3<-K3;K4<-K4;K5<-K5;K6<-K6",
                                   r, None, None, "best residual > 0.15"))
            residuals.append(r)
        stats = plr.audit_statistics(sim)
        _check("2a. total=101", int(stats["total"]) == 101, failures, str(stats["total"]))
        _check("2b. production_ok=95 / blocked=6",
               int(stats["production_ok"]) == 95 and int(stats["production_blocked"]) == 6,
               failures, f"{stats['production_ok']}/{stats['production_blocked']}")
        _check("3a. <=0.15=95 / <=0.16=98 / <=0.18=101",
               int(stats["le_015"]) == 95 and int(stats["le_016"]) == 98
               and int(stats["le_018"]) == 101,
               failures, f"{stats['le_015']}/{stats['le_016']}/{stats['le_018']}")
        rv = np.array(sorted(residuals), dtype=float)
        _check("4a. min 正确", abs(float(stats["min_residual"]) - float(rv.min())) < 1e-12,
               failures, str(stats["min_residual"]))
        _check("4b. median 正确", abs(float(stats["median_residual"])
                                      - float(np.percentile(rv, 50))) < 1e-9,
               failures, str(stats["median_residual"]))
        _check("4c. p95 正确", abs(float(stats["p95_residual"])
                                   - float(np.percentile(rv, 95))) < 1e-9,
               failures, str(stats["p95_residual"]))
        _check("4d. max 正确", abs(float(stats["max_residual"]) - float(rv.max())) < 1e-12,
               failures, str(stats["max_residual"]))

        print("[5] best/second mapping 相同 -> same 计数")
        same_map = "K0<-K1;K1<-K0;K2<-K2;K3<-K3;K4<-K4;K5<-K5;K6<-K6"
        s5 = [
            _make_audit("s0.txt", "OK", same_map, 0.155, same_map, 0.1557),
            _make_audit("s1.txt", "OK", same_map, 0.160, same_map, 0.1604),
            _make_audit("s2.txt", "RESIDUAL_BLOCK", "K0<-K0;K1<-K1;K2<-K2;K3<-K3;K4<-K4;K5<-K5;K6<-K6",
                        0.15, None, None, ""),
        ]
        st5 = plr.audit_statistics(s5)
        _check("5a. same=2 / different=0（无 second 的不计）",
               int(st5["best_second_same"]) == 2
               and int(st5["best_second_different"]) == 0,
               failures, f"{st5['best_second_same']}/{st5['best_second_different']}")

        print("[6] best/second mapping 不同 -> different 计数")
        diff_map = "K0<-K5;K1<-K1;K2<-K2;K3<-K3;K4<-K4;K5<-K0;K6<-K6"
        s6 = [
            _make_audit("d0.txt", "OK", same_map, 0.155, diff_map, 0.1557),
            _make_audit("d1.txt", "OK", same_map, 0.160, diff_map, 0.1604),
        ]
        st6 = plr.audit_statistics(s6)
        _check("6a. different=2", int(st6["best_second_different"]) == 2,
               failures, str(st6["best_second_different"]))
        _check("6b. most common mapping 统计正确",
               st6["most_common_mapping"] == same_map
               and int(st6["most_common_count"]) == 2,
               failures, f"{st6['most_common_mapping']} x {st6['most_common_count']}")

        print("[7/8] validate 最终 PASS / BLOCK（production 规则不变）")
        base = root / "dataset" / "hand"
        _make_base(base)
        tmpl = root / "tmpl7.txt"
        _write_label(tmpl, CANON)
        # PASS 场景：全部 ok
        ed_ok, el_ok, en_ok = _make_external_dir(root / "ok_ext",
                                                 [("e1", "ok"), ("e2", "ok")])
        r8 = xsub.validate(base, ed_ok, el_ok, en_ok, template=tmpl)
        _check("8a. 全 OK -> reorder_ok=True blocked=0 total=2",
               bool(r8.get("reorder_ok")) is True
               and int(r8.get("reorder_total")) == 2
               and int(r8.get("reorder_blocked")) == 0
               and int(r8.get("reorder_ok_count")) == 2,
               failures, str({k: r8.get(k) for k in ("reorder_ok", "reorder_total",
                                                      "reorder_ok_count",
                                                      "reorder_blocked")}))
        # BLOCK 场景：bad 几何（residual 超限）
        ed_b, el_b, en_b = _make_external_dir(root / "bad_ext",
                                              [("b1", "bad"), ("b2", "ok")])
        try:
            xsub.validate(base, ed_b, el_b, en_b, template=tmpl)
            _check("7a. 存在 production BLOCK -> validate BLOCK", False, failures,
                   "未 BLOCK")
        except xsub.CrossSubjectError as e:
            _check("7a. 存在 production BLOCK -> validate BLOCK",
                   "canonical reorder BLOCK" in str(e) and "@" in str(e),
                   failures, str(e))

        print("[9] AMBIGUOUS 单张：仍扫完其余，最终整体 BLOCK")
        ed_a, el_a, en_a = _make_external_dir(root / "amb_ext",
                                              [("m1", "amb"), ("m2", "ok")])
        try:
            xsub.validate(base, ed_a, el_a, en_a, template=tmpl)
            _check("9a. AMBIGUOUS 存在 -> validate BLOCK", False, failures, "未 BLOCK")
        except xsub.CrossSubjectError as e:
            _check("9a. AMBIGUOUS 存在 -> validate BLOCK",
                   "canonical reorder BLOCK" in str(e), failures, str(e))

        print("[10] ERROR 单张：完成其余扫描并记录 message（scan 层单元）")
        # 注：validate 的基础格式检查（validate_labels）会在 audit 前拦住坏行；
        # ERROR 语义在 scan_bank_audit 层（任意 label 文件）保证“不中断其余”。
        audit_dir = root / "audit_out"
        ed_r = root / "err_scan"
        ed_r.mkdir()
        _write_label(ed_r / "x2.txt", CANON)
        (ed_r / "x1.txt").write_text("not a yolo pose label line\n", encoding="utf-8")
        audits10 = plr.scan_bank_audit(
            refs,
            [("x1.txt", ed_r / "x1.txt"), ("x2.txt", ed_r / "x2.txt")])
        _check("10a. ERROR 后仍扫完其余（len=2, [ERROR, OK]）",
               len(audits10) == 2 and audits10[0].status == "ERROR"
               and audits10[1].status == "OK",
               failures, [a.status for a in audits10])
        _check("10b. ERROR 行 message 记录原因",
               audits10[0].message != "", failures, audits10[0].message)
        csv_path = audit_dir / "canonical_reorder_audit.csv"
        plr.write_audit_report(audits10, csv_path)
        csv_text = csv_path.read_text(encoding="utf-8-sig")
        _check("10c. audit CSV 含 ERROR 与 OK 行",
               "ERROR" in csv_text and ",OK," in csv_text
               and "x1.txt" in csv_text and "x2.txt" in csv_text,
               failures)
        # 不污染正式 dataset：hand 目录里不得出现 audit/report 文件
        _check("10d. 正式 dataset/hand 未被写入 audit/report",
               not any(p.name.startswith("canonical_") or p.name.startswith(".prepare")
                       for p in base.iterdir()), failures)

    print("-" * 60)
    if failures:
        print(f"RESULT: {len(failures)} FAILED -> {failures}")
        sys.exit(1)
    print("RESULT: ALL PASSED")


if __name__ == "__main__":
    main()
