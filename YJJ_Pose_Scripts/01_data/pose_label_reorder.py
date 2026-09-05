"""Canonical template 关键点重排核心（纯逻辑，无 Qt / 无 subprocess）。

来源：从 ``06_debug/reorder_pose_labels_by_template.py`` 抽取，供
Cross-subject Test Set 重建等流程以 import 方式复用（GUI 不再单独起 reorder 子进程）。

能力：
- 读取 canonical template label（定义 K0..K(N-1) 语义顺序）
- 单标签：忽略目标 label 自带编号，按整体几何匹配 template（支持平移/旋转/scale），
  输出 mapping / normalized RMS 残差 / BLOCK 原因（含歧义）
- 整目录重排（供旧 CLI wrapper 复用）

算法（与旧脚本一致，另补歧义护栏）：
1. 每点到其余点的距离排序签名（平移/旋转/等比缩放不变）
2. 匈牙利做初始一对一匹配
3. iterative similarity 拟合 + 匈牙利重分配收敛
4. normalized RMS 阈值检查（默认 0.17）
5. 歧义护栏：best 与“任一两两 swap 后重拟合”的最小残差过于接近 -> AMBIGUOUS BLOCK
   （两个点完全重合/近似对称时绝不静默猜 ID）

注意：输入坐标为 YOLO 归一化坐标（图像内 0..1），算法在归一化平面上做相似变换，
残差按目标点集 pairwise scale 归一化（与旧脚本口径一致）。
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

# 与旧脚本一致：归一化残差阈值
# 2026-09-05 依据真实 canonical_reorder_audit.csv（101 samples / max residual
# 0.164199 / best mapping 101/101 一致）：0.15 -> 0.17，validate 与 rebuild 共用此默认。
DEFAULT_MAX_RESIDUAL = 0.17
# 歧义护栏：second-best 与 best 残差 gap 小于 max(1e-3, best*gap) 视为歧义
DEFAULT_AMBIGUITY_GAP_RATIO = 0.20
# similarity ICP 最大轮数
_REFINE_ROUNDS = 8


class ReorderError(ValueError):
    """重排失败（K 不一致 / 格式非法 / residual 超限 / 歧义），message 即原因。"""


class ReorderAmbiguousError(ReorderError):
    """几何歧义：best 与次优过于接近，不猜 ID。"""


@dataclass
class ReorderRow:
    """单个文件的 report 行。"""
    file: str
    status: str          # OK / BLOCK / AMBIGUOUS
    mapping: str         # "K0<-K1;K1<-K0;..."（dest K_i <- source K_{assignment[i]}）
    normalized_rms: str
    message: str
    reference: str = ""  # reference bank 命中的 train label 文件名（单模板场景为空）


# ============================================================ 读标签
def parse_label_line(line: str) -> tuple[list[str], list[list[str]], list[tuple[float, float]]]:
    """解析单行 YOLO-Pose 标签。

    返回 (head[5], triples[N][3], pts[(x,y), ...])。
    任意点 visibility<=0 视为不可见 -> 抛错（本工具要求全部可见）。
    """
    parts = line.split()
    rem = len(parts) - 5
    if rem <= 0 or rem % 3 != 0:
        raise ReorderError(f"invalid YOLO Pose token count {len(parts)}")
    n = rem // 3
    head = parts[:5]
    triples = [parts[5 + 3 * i:5 + 3 * (i + 1)] for i in range(n)]
    pts: list[tuple[float, float]] = []
    for i, tri in enumerate(triples):
        x, y, v = (float(t) for t in tri)
        if v <= 0:
            raise ReorderError(f"K{i} visibility={v}; expected visible points")
        pts.append((x, y))
    return head, triples, pts


def read_label(path: str | Path) -> tuple[list[str], list[list[str]], list[tuple[float, float]]]:
    """读单个 label 文件（要求恰 1 个对象行）。"""
    p = Path(path)
    lines = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(lines) != 1:
        raise ReorderError(f"expected exactly 1 object line, got {len(lines)}")
    return parse_label_line(lines[0])


def template_keypoint_count(template_path: str | Path) -> int:
    """读取 canonical template 并返回其关键点数量 N（自动决定模板维度）。"""
    _h, _t, pts = read_label(template_path)
    return len(pts)


# ============================================================ 几何匹配（原算法移植）
def _pairwise_scale(points: list[tuple[float, float]]) -> float:
    ds = []
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            ds.append(math.hypot(points[i][0] - points[j][0],
                                 points[i][1] - points[j][1]))
    ds = sorted(d for d in ds if d > 1e-12)
    if not ds:
        raise ReorderError("degenerate point set")
    return ds[len(ds) // 2]


def _signatures(points: list[tuple[float, float]]) -> list[list[float]]:
    s = _pairwise_scale(points)
    out = []
    for i, p in enumerate(points):
        vals = [math.hypot(p[0] - q[0], p[1] - q[1]) / s
                for j, q in enumerate(points) if i != j]
        out.append(sorted(vals))
    return out


def _hungarian(cost: list[list[float]]) -> list[int]:
    """方阵匈牙利（e-maxx 版），返回 assignment[i]=列。"""
    n = len(cost)
    m = len(cost[0]) if n else 0
    if n == 0 or m == 0 or n > m:
        raise ReorderError("Hungarian expects 0 < rows <= cols")

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    return assignment


def _initial_assignment(template_pts: list[tuple[float, float]],
                        target_pts: list[tuple[float, float]]) -> list[int]:
    a = _signatures(template_pts)
    b = _signatures(target_pts)
    n = len(a)
    cost = []
    for i in range(n):
        row = [sum(abs(x - y) for x, y in zip(a[i], b[j])) / max(1, n - 1)
               for j in range(n)]
        cost.append(row)
    return _hungarian(cost)


def _fit_similarity(src: list[tuple[float, float]],
                    dst: list[tuple[float, float]]) -> tuple[float, float, float, float, float]:
    """dst ~= scale*R*src + t，禁反射；返回 (scale, c, s, tx, ty)。"""
    n = len(src)
    sx = sum(p[0] for p in src) / n
    sy = sum(p[1] for p in src) / n
    dx = sum(p[0] for p in dst) / n
    dy = sum(p[1] for p in dst) / n

    a = b = denom = 0.0
    for (x, y), (X, Y) in zip(src, dst):
        x -= sx
        y -= sy
        X -= dx
        Y -= dy
        a += x * X + y * Y
        b += x * Y - y * X
        denom += x * x + y * y

    if denom <= 1e-15:
        raise ReorderError("degenerate template geometry")
    rnorm = math.hypot(a, b)
    if rnorm <= 1e-15:
        raise ReorderError("cannot estimate rotation")

    c = a / rnorm
    s = b / rnorm
    scale = rnorm / denom
    tx = dx - scale * (c * sx - s * sy)
    ty = dy - scale * (s * sx + c * sy)
    return scale, c, s, tx, ty


def _apply_similarity(points: list[tuple[float, float]],
                      tfm: tuple[float, float, float, float, float]
                      ) -> list[tuple[float, float]]:
    scale, c, s, tx, ty = tfm
    return [(scale * (c * x - s * y) + tx, scale * (s * x + c * y) + ty)
            for x, y in points]


def _normalized_rms(template_pts: list[tuple[float, float]],
                    target_pts: list[tuple[float, float]],
                    assignment: list[int],
                    tfm: tuple[float, float, float, float, float]) -> float:
    pred = _apply_similarity(template_pts, tfm)
    err2 = 0.0
    for i, j in enumerate(assignment):
        err2 += (pred[i][0] - target_pts[j][0]) ** 2 + (pred[i][1] - target_pts[j][1]) ** 2
    rms = math.sqrt(err2 / len(template_pts))
    return rms / _pairwise_scale(target_pts)


def _refine_assignment(template_pts: list[tuple[float, float]],
                       target_pts: list[tuple[float, float]],
                       assignment: list[int],
                       rounds: int = _REFINE_ROUNDS
                       ) -> tuple[list[int], tuple[float, float, float, float, float]]:
    current = list(assignment)
    for _ in range(rounds):
        matched = [target_pts[current[i]] for i in range(len(template_pts))]
        tfm = _fit_similarity(template_pts, matched)
        pred = _apply_similarity(template_pts, tfm)
        cost = [[math.hypot(pred[i][0] - target_pts[j][0],
                            pred[i][1] - target_pts[j][1])
                 for j in range(len(target_pts))]
                for i in range(len(template_pts))]
        nxt = _hungarian(cost)
        if nxt == current:
            return current, tfm
        current = nxt
    matched = [target_pts[current[i]] for i in range(len(template_pts))]
    return current, _fit_similarity(template_pts, matched)


def _swap_residual(template_pts: list[tuple[float, float]],
                   target_pts: list[tuple[float, float]],
                   assignment: list[int], i: int, j: int) -> float:
    """交换 assignment[i]/[j] 后重拟合 similarity 的 normalized RMS（次优估计）。"""
    swapped = list(assignment)
    swapped[i], swapped[j] = swapped[j], swapped[i]
    matched = [target_pts[swapped[k]] for k in range(len(template_pts))]
    tfm = _fit_similarity(template_pts, matched)
    return _normalized_rms(template_pts, target_pts, swapped, tfm)


# ============================================================ 公开 API
def _match_once(template_pts: list[tuple[float, float]],
                target_pts: list[tuple[float, float]]
                ) -> tuple[list[int], float]:
    """签名初值 + ICP 收敛后的 (assignment, normalized_rms)（不做阈值/歧义判断）。"""
    if len(template_pts) != len(target_pts):
        raise ReorderError(
            f"template has {len(template_pts)} points, target has {len(target_pts)}")
    n = len(template_pts)
    if n == 0:
        raise ReorderError("empty point set")
    assignment = _initial_assignment(template_pts, target_pts)
    assignment, tfm = _refine_assignment(template_pts, target_pts, assignment)
    best = _normalized_rms(template_pts, target_pts, assignment, tfm)
    return assignment, best


def _is_ambiguous(template_pts: list[tuple[float, float]],
                  target_pts: list[tuple[float, float]],
                  assignment: list[int], best: float,
                  ambiguity_gap_ratio: float) -> bool:
    """歧义护栏：best 与“任一两两 swap 重拟合”的最小残差过近 -> True（n<3 恒 False）。"""
    n = len(template_pts)
    if n < 3:
        return False
    second_best = min(_swap_residual(template_pts, target_pts, assignment, i, j)
                      for i in range(n) for j in range(i + 1, n))
    return (second_best - best) <= max(1e-3, best * ambiguity_gap_ratio)


def compute_assignment(
    template_pts: list[tuple[float, float]],
    target_pts: list[tuple[float, float]],
    max_residual: float = DEFAULT_MAX_RESIDUAL,
    ambiguity_gap_ratio: float = DEFAULT_AMBIGUITY_GAP_RATIO,
) -> tuple[list[int], float]:
    """按 template 整体几何匹配 target，返回 (assignment, normalized_rms)。

    assignment[i] = target 点集中被放入 K_i 的下标。
    - K 不一致 / 几何退化 -> ReorderError
    - residual > max_residual -> ReorderError
    - best 与次优（两两 swap）过于接近 -> ReorderAmbiguousError（不静默猜）
    """
    assignment, best = _match_once(template_pts, target_pts)
    if best > max_residual:
        raise ReorderError(
            f"geometry residual {best:.6f} > {max_residual:.6f}")
    if _is_ambiguous(template_pts, target_pts, assignment, best,
                     ambiguity_gap_ratio):
        second_best = min(_swap_residual(template_pts, target_pts, assignment, i, j)
                          for i in range(len(template_pts))
                          for j in range(i + 1, len(template_pts)))
        raise ReorderAmbiguousError(
            f"ambiguous mapping (best={best:.6f}, second={second_best:.6f}); "
            "do NOT guess IDs")
    return assignment, best


# ============================================================ Train Reference Bank
@dataclass
class Reference:
    """Base train label 之一：其 K0..K(N-1) 即训练体系 canonical 语义。"""
    name: str            # 文件名（report 用，如 color_xxx.txt）
    pts: list[tuple[float, float]]


@dataclass
class BankMatch:
    """某张 external label 与 Reference Bank 匹配的结果。"""
    reference: Reference          # 命中的最佳 train reference
    assignment: list[int]         # dest K_i <- source 第 assignment[i] 个三元组
    normalized_rms: float         # 最佳 residual（normalized）
    second_name: str              # 次佳 reference 名（诊断）
    second_rms: float             # 次佳 residual
    candidates: int               # 参与比较的 reference 数（排除歧义/退化候选）
    compared: int                 # 参与比较（含被排除）的 reference 总数


def load_reference_bank(labels_train_dir: str | Path,
                        expected_k: int | None = None) -> list[Reference]:
    """读取 base 训练集 labels/train/*.txt 作为 canonical reference bank。

    前提：这些 label 已属同一训练体系（K0..K(N-1) 同语义）。
    expected_k: 非 None 时逐文件校验 K 并拒绝数量不一致的文件。
    目录为空或全部解析失败 -> ReorderError。
    """
    d = Path(labels_train_dir)
    if not d.is_dir():
        raise ReorderError(f"reference bank 目录不存在: {d}")
    files = sorted(p for p in d.glob("*.txt") if p.is_file())
    if not files:
        raise ReorderError(f"reference bank 为空（无 train labels）: {d}")
    refs: list[Reference] = []
    errors: list[str] = []
    for p in files:
        try:
            _h, _t, pts = read_label(p)
        except ReorderError as e:
            errors.append(f"{p.name}: {e}")
            continue
        if expected_k is not None and len(pts) != expected_k:
            errors.append(f"{p.name}: K={len(pts)} != 期望 {expected_k}")
            continue
        refs.append(Reference(name=p.name, pts=pts))
    if not refs:
        raise ReorderError(
            f"reference bank 没有可用 label（errors: {errors[:3]}）")
    return refs


def _scan_usable(references: list[Reference],
                 target_pts: list[tuple[float, float]],
                 ambiguity_gap_ratio: float) -> list[BankMatch]:
    """与全部 reference 逐一比较（signature + Hungarian + ICP + 歧义护栏），
    返回按 normalized_rms 升序的“可用”候选（排除歧义/退化/K 不一致）。"""
    usable: list[BankMatch] = []
    for ref in references:
        if len(ref.pts) != len(target_pts):
            continue
        try:
            asg, res = _match_once(ref.pts, target_pts)
        except ReorderError:
            continue  # 退化候选，不可用
        if _is_ambiguous(ref.pts, target_pts, asg, res, ambiguity_gap_ratio):
            continue  # 该 reference 下 mapping 自歧义 -> 不采用
        usable.append(BankMatch(reference=ref, assignment=asg, normalized_rms=res,
                                second_name="", second_rms=float("inf"),
                                candidates=0, compared=0))
    usable.sort(key=lambda m: m.normalized_rms)
    # 填充 second 诊断（按 residual 的第二名）
    if len(usable) >= 2:
        usable[0].second_name = usable[1].reference.name
        usable[0].second_rms = usable[1].normalized_rms
    return usable


def bank_best_match(references: list[Reference],
                    target_pts: list[tuple[float, float]],
                    max_residual: float = DEFAULT_MAX_RESIDUAL,
                    ambiguity_gap_ratio: float = DEFAULT_AMBIGUITY_GAP_RATIO,
                    ) -> BankMatch:
    """external 点集与整个 Reference Bank 比较，选 normalized residual 最小者。

    规则：
    - 对每个 reference 执行 shape signature + Hungarian + ICP；
    - 该 reference 若 self-symmetric 导致 swap-ambiguity -> 该候选不可信，跳过；
    - 全候选不可信/退化 -> ReorderAmbiguousError（不猜 ID）；
    - best residual > max_residual -> ReorderError（消息含 best/second 诊断）。
    """
    if not references:
        raise ReorderError("reference bank 为空")
    usable = _scan_usable(references, target_pts, ambiguity_gap_ratio)
    if not usable:
        raise ReorderAmbiguousError(
            f"no unambiguous reference matched (compared={len(references)}); "
            "do NOT guess IDs")
    best = usable[0]
    if best.normalized_rms > max_residual:
        gap = (f"{(best.second_rms - best.normalized_rms):.6f}"
               if best.second_name else "n/a")
        raise ReorderError(
            f"best reference {best.reference.name} residual "
            f"{best.normalized_rms:.6f} > {max_residual:.6f} "
            f"(second {best.second_name or '-'}: "
            f"{best.second_rms:.6f}, gap {gap})")
    return best


def bank_dry_run(references: list[Reference],
                 label_files: list[tuple[str, Path]],
                 max_residual: float = DEFAULT_MAX_RESIDUAL,
                 ambiguity_gap_ratio: float = DEFAULT_AMBIGUITY_GAP_RATIO,
                 ) -> tuple[list[ReorderRow], float]:
    """对一组 external label 做只读 reference-bank 重排试运行（严格：遇错即抛）。

    label_files: list[(display_name, path)]（matched 样本的 label 文件）。
    全部成功返回 (rows 全 OK, max_residual_over_all)；任何一张失败抛 ReorderError/
    ReorderAmbiguousError，message 注明文件名（调用方转成 BLOCK）。
    """
    rows: list[ReorderRow] = []
    max_rms = 0.0
    for name, path in label_files:
        try:
            head, triples, pts = read_label(path)
            m = bank_best_match(references, pts, max_residual, ambiguity_gap_ratio)
        except ReorderAmbiguousError as e:
            raise ReorderAmbiguousError(f"{name}: {e}") from e
        except ReorderError as e:
            raise ReorderError(f"{name}: {e}") from e
        rows.append(ReorderRow(name, "OK", mapping_text(m.assignment),
                               f"{m.normalized_rms:.6f}", "",
                               reference=m.reference.name))
        max_rms = max(max_rms, m.normalized_rms)
    return rows, max_rms


# ============================================================ 全量 audit（Validate 用）
@dataclass
class LabelAudit:
    """单张 external label 的 audit 结果（含 best/second 全诊断，即使失败）。"""
    file: str
    status: str                 # OK / RESIDUAL_BLOCK / AMBIGUOUS / ERROR
    best_reference: str
    best_mapping: str
    best_residual: float | None
    second_reference: str
    second_mapping: str | None
    second_residual: float | None
    reference_gap: float | None   # second_residual - best_residual（无 second 时 None）
    message: str


def _audit_from_usable(name: str, usable: list[BankMatch],
                       max_residual: float) -> LabelAudit:
    """由按 residual 升序的可用候选生成单张 audit（OK/RESIDUAL_BLOCK）。"""
    best = usable[0]
    second = usable[1] if len(usable) >= 2 else None
    mapping = mapping_text(best.assignment)
    status = "OK" if best.normalized_rms <= max_residual else "RESIDUAL_BLOCK"
    second_mapping = mapping_text(second.assignment) if second else None
    message = ""
    if status == "RESIDUAL_BLOCK":
        message = (f"best residual {best.normalized_rms:.6f} > {max_residual:.6f}"
                   + (f" (second {second.reference.name}: {second.normalized_rms:.6f}, "
                      f"gap {second.normalized_rms - best.normalized_rms:.6f})"
                      if second else ""))
    return LabelAudit(
        file=name,
        status=status,
        best_reference=best.reference.name,
        best_mapping=mapping,
        best_residual=best.normalized_rms,
        second_reference=second.reference.name if second else "",
        second_mapping=second_mapping,
        second_residual=second.normalized_rms if second else None,
        reference_gap=(second.normalized_rms - best.normalized_rms) if second else None,
        message=message,
    )


def scan_bank_audit(references: list[Reference],
                    label_files: list[tuple[str, Path]],
                    max_residual: float = DEFAULT_MAX_RESIDUAL,
                    ambiguity_gap_ratio: float = DEFAULT_AMBIGUITY_GAP_RATIO,
                    ) -> list[LabelAudit]:
    """全量扫描 audit：对全部 external label 逐张判定，绝不因首张失败提前退出。

    每张 status：
      OK             best residual <= max_residual
      RESIDUAL_BLOCK best residual >  max_residual
      AMBIGUOUS      无任何可信 reference（全部歧义/退化）
      ERROR          读 label / 匹配过程异常（record message）
    返回 list[LabelAudit]（长度 == label_files）。
    """
    audits: list[LabelAudit] = []
    for name, path in label_files:
        try:
            _head, _tri, pts = read_label(path)
            usable = _scan_usable(references, pts, ambiguity_gap_ratio)
        except ReorderError as e:
            audits.append(LabelAudit(name, "ERROR", "", "", None, "", None, None, None,
                                     str(e)))
            continue
        except Exception as e:  # noqa: BLE001 —— 单张异常不中断整体扫描
            audits.append(LabelAudit(name, "ERROR", "", "", None, "", None, None, None,
                                     f"{type(e).__name__}: {e}"))
            continue
        if not usable:
            audits.append(LabelAudit(
                name, "AMBIGUOUS", "", "", None, "", None, None, None,
                f"no unambiguous reference matched (compared={len(references)}); "
                "do NOT guess IDs"))
            continue
        audits.append(_audit_from_usable(name, usable, max_residual))
    return audits


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * p
    lo = int(idx)
    frac = idx - lo
    if lo + 1 >= len(sorted_vals):
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[lo + 1] - sorted_vals[lo]) * frac


def audit_statistics(audits: list[LabelAudit]) -> dict[str, object]:
    """汇总 audit 的 production 判定 + 诊断统计。

    - production OK = status==OK；任何其它 status 都算 production BLOCK
    - residual 分桶/分位数只统计有 best_residual 的样本（AMBIGUOUS/ERROR 无残差）
    - best/second mapping same/different 只在两者都存在时统计
    """
    total = len(audits)
    production_ok = sum(1 for a in audits if a.status == "OK")
    production_blocked = total - production_ok

    residuals = sorted(a.best_residual for a in audits
                       if a.best_residual is not None)
    le_015 = sum(1 for r in residuals if r <= 0.15)
    le_016 = sum(1 for r in residuals if r <= 0.16)
    le_018 = sum(1 for r in residuals if r <= 0.18)

    same = sum(1 for a in audits
               if a.best_mapping is not None and a.second_mapping is not None
               and a.best_mapping == a.second_mapping)
    different = sum(1 for a in audits
                    if a.best_mapping is not None and a.second_mapping is not None
                    and a.best_mapping != a.second_mapping)

    most_common = ""
    most_common_count = 0
    mapped = [a.best_mapping for a in audits if a.best_mapping]
    if mapped:
        from collections import Counter
        top = Counter(mapped).most_common(1)[0]
        most_common, most_common_count = top

    return {
        "total": total,
        "production_ok": production_ok,
        "production_blocked": production_blocked,
        "residual_count": len(residuals),
        "le_015": le_015,
        "le_016": le_016,
        "le_018": le_018,
        "min_residual": (residuals[0] if residuals else None),
        "median_residual": (_percentile(residuals, 0.5) if residuals else None),
        "p95_residual": (_percentile(residuals, 0.95) if residuals else None),
        "max_residual": (residuals[-1] if residuals else None),
        "best_second_same": same,
        "best_second_different": different,
        "most_common_mapping": most_common,
        "most_common_count": most_common_count,
        "most_common_ratio": (most_common_count / len(mapped) if mapped else None),
    }


def write_audit_report(audits: list[LabelAudit], report_path: str | Path) -> None:
    """写 audit CSV（file,status,best_reference,best_mapping,best_residual,
    second_reference,second_mapping,second_residual,reference_gap,message）。"""
    p = Path(report_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "status", "best_reference", "best_mapping",
                    "best_residual", "second_reference", "second_mapping",
                    "second_residual", "reference_gap", "message"])
        for a in audits:
            w.writerow([
                a.file, a.status, a.best_reference, a.best_mapping,
                (f"{a.best_residual:.6f}" if a.best_residual is not None else ""),
                a.second_reference,
                (a.second_mapping or ""),
                (f"{a.second_residual:.6f}" if a.second_residual is not None else ""),
                (f"{a.reference_gap:.6f}" if a.reference_gap is not None else ""),
                a.message,
            ])


def mapping_text(assignment: list[int]) -> str:
    """把 assignment 转成 report 用文本：K0<-K1;K1<-K0;..."""
    return ";".join(f"K{i}<-K{j}" for i, j in enumerate(assignment))


def build_reordered_text(head: list[str], triples: list[list[str]],
                         assignment: list[int]) -> str:
    """按 assignment 重排三元组，输出单行标签文本（class/bbox 保持原样）。"""
    out_parts = list(head)
    for src_idx in assignment:
        out_parts.extend(triples[src_idx])
    return " ".join(out_parts)


def reorder_file(template_path: str | Path, src_path: str | Path,
                 dst_path: str | Path | None = None,
                 max_residual: float = DEFAULT_MAX_RESIDUAL,
                 ambiguity_gap_ratio: float = DEFAULT_AMBIGUITY_GAP_RATIO,
                 ) -> tuple[list[int], float]:
    """读 src label，按 template 重排，写到 dst_path（None 则不写）。

    返回 (assignment, normalized_rms)。任何失败抛 ReorderError 子类。
    """
    _th, _tt, template_pts = read_label(template_path)
    head, triples, target_pts = read_label(src_path)
    assignment, residual = compute_assignment(
        template_pts, target_pts, max_residual, ambiguity_gap_ratio)
    if dst_path is not None:
        p = Path(dst_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(build_reordered_text(head, triples, assignment) + "\n",
                     encoding="utf-8")
    return assignment, residual


def reorder_directory(src: str | Path, dst: str | Path,
                      template_path: str | Path,
                      max_residual: float = DEFAULT_MAX_RESIDUAL,
                      dry_run: bool = False) -> tuple[list[ReorderRow], int, int]:
    """整目录重排（旧 CLI wrapper 复用）：读 src 下全部 .txt，写重排副本到 dst。

    返回 (rows, ok_count, fail_count)。单个文件失败不中断，计入 report。
    """
    src_p = Path(src)
    dst_p = Path(dst)
    _th, _tt, template_pts = read_label(template_path)
    files = sorted(src_p.rglob("*.txt"))
    rows: list[ReorderRow] = []
    ok = 0
    fail = 0
    for path in files:
        rel = str(path.relative_to(src_p))
        try:
            head, triples, pts = read_label(path)
            assignment, residual = compute_assignment(template_pts, pts, max_residual)
            if not dry_run:
                out = dst_p / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(build_reordered_text(head, triples, assignment) + "\n",
                               encoding="utf-8")
            rows.append(ReorderRow(rel, "OK", mapping_text(assignment),
                                   f"{residual:.6f}", ""))
            ok += 1
        except ReorderAmbiguousError as e:
            rows.append(ReorderRow(rel, "AMBIGUOUS", "", "", str(e)))
            fail += 1
        except ReorderError as e:
            rows.append(ReorderRow(rel, "BLOCK", "", "", str(e)))
            fail += 1
    return rows, ok, fail


def write_report(rows: list[ReorderRow], report_path: str | Path) -> None:
    """写 CSV report（utf-8-sig，含 BOM 便于 Excel）。旧 5 列格式。"""
    p = Path(report_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "status", "mapping", "normalized_rms", "message"])
        for r in rows:
            w.writerow([r.file, r.status, r.mapping, r.normalized_rms, r.message])


def write_report_with_reference(rows: list[ReorderRow], report_path: str | Path) -> None:
    """写 canonical reorder report（含 reference_file 列），供 Cross-subject 数据集追溯。"""
    p = Path(report_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "status", "reference_file", "mapping",
                    "normalized_rms", "message"])
        for r in rows:
            w.writerow([r.file, r.status, r.reference, r.mapping,
                        r.normalized_rms, r.message])
