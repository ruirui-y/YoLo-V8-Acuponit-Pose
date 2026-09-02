"""数据集纯逻辑服务（不依赖 QWidget / PySide6 UI）。

从 MainWindow 移出的所有数据集解析逻辑：
- parse_variant_yaml: 极简解析 data_*.yaml 的 rgbd/channels/kpt_shape
- scan_rgbd_variants: 扫描目录下所有 4ch RGBD 变体
- resolve_test_set: 解析 yaml 的 path+test 字段获取测试集信息
- parse_yaml_meta: 解析 rgb/rgbd yaml 的 kpt_shape/channels
- stat_existing_dataset: 现场统计已有数据集
- merge_report: 合并 dataset_report.json 与现场统计

输入：Path / str / 参数
输出：dict / list / Path / tuple
"""
import json
from pathlib import Path


class DatasetService:

    @staticmethod
    def parse_variant_yaml(yaml_path):
        """极简解析单个 data_*.yaml，返回 {rgbd:bool, channels:int|None, kpt:int|None}。

        支持内联列表 kpt_shape:[7,3] 与多行块列表；不依赖 PyYAML。
        """
        try:
            text = Path(yaml_path).read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            return {"rgbd": False, "channels": None, "kpt": None}
        rgbd = False
        channels = None
        kpt = None
        in_kpt = False
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("rgbd:"):
                v = s.split(":", 1)[1].strip().lower().split("#", 1)[0].strip()
                rgbd = (v == "true")
                continue
            if s.startswith("channels:"):
                try:
                    channels = int(s.split(":", 1)[1].strip())
                except ValueError:
                    channels = None
                continue
            if s.startswith("kpt_shape:"):
                rest = s[len("kpt_shape:"):].split("#", 1)[0].strip()
                if rest.startswith("[") and rest.endswith("]"):
                    nums = [x.strip() for x in rest[1:-1].split(",") if x.strip()]
                    if nums:
                        try:
                            kpt = int(nums[0])
                        except ValueError:
                            kpt = None
                else:
                    in_kpt = True
                continue
            if in_kpt:
                if s.startswith("-"):
                    try:
                        kpt = int(s[1:].strip())
                    except ValueError:
                        pass
                    in_kpt = False
                else:
                    in_kpt = False
        return {"rgbd": rgbd, "channels": channels, "kpt": kpt}

    @staticmethod
    def scan_rgbd_variants(base):
        """扫描 base 下所有子目录，找出 yaml 同时满足 rgbd:true 且 channels:4 的 4ch 数据集。

        返回 list[(dir_name, yaml_path)]，按 dir_name 排序。
        """
        found = []
        if not base or not Path(base).exists():
            return found
        for sub in sorted(Path(base).iterdir()):
            if not sub.is_dir():
                continue
            for y in sub.glob("data_*.yaml"):
                meta = DatasetService.parse_variant_yaml(y)
                if meta["rgbd"] is True and meta["channels"] == 4:
                    found.append((sub.name, y))
                    break
        return found

    @staticmethod
    def resolve_test_set(yaml_path):
        """解析单个 data_*.yaml 的 path + test 字段，返回 (test_images_dir, count, id_set)。

        - test_images_dir: 真实测试图像目录（path 为相对时按 yaml 父目录解析）
        - count: 实际图片数量（目录缺失/无图则为 0）
        - id_set: 测试图像 stem 集合（用于跨 RGB/RGBD 一致性校验；目录缺失则为 None）

        任何无法解析的情况返回 (None, 0, None)。
        """
        if not yaml_path or not Path(yaml_path).exists():
            return None, 0, None
        try:
            text = Path(yaml_path).read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            return None, 0, None
        path_field = None
        test_field = "images/test"
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("path:"):
                path_field = s.split(":", 1)[1].strip().split("#", 1)[0].strip()
            elif s.startswith("test:"):
                test_field = s.split(":", 1)[1].strip().split("#", 1)[0].strip()
        if not path_field:
            return None, 0, None
        root = Path(path_field)
        if not root.is_absolute():
            root = Path(yaml_path).parent / path_field
        test_dir = root / test_field
        if not test_dir.exists():
            return test_dir, 0, None
        img_ext = (".png", ".jpg", ".jpeg")
        ids = sorted(p.stem for p in test_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in img_ext)
        return test_dir, len(ids), set(ids)

    @staticmethod
    def parse_yaml_meta(rgb_yaml, rgbd_yaml):
        """极简解析本项目 data_rgb.yaml / data_rgbd.yaml：
        返回 (kpt_shape_first, rgb_channels, rgbd_channels)。
        """
        def parse(text):
            channels = None
            kpt = None
            in_kpt = False
            for line in text.splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if s.startswith("kpt_shape:"):
                    rest = s[len("kpt_shape:"):].split("#", 1)[0].strip()
                    if rest.startswith("[") and rest.endswith("]"):
                        inside = rest[1:-1]
                        nums = [x.strip() for x in inside.split(",") if x.strip()]
                        if nums:
                            try:
                                kpt = int(nums[0])
                            except ValueError:
                                kpt = None
                    else:
                        in_kpt = True
                    continue
                if in_kpt:
                    if s.startswith("-"):
                        try:
                            val = int(s[1:].strip())
                        except ValueError:
                            continue
                        kpt = val if kpt is None else kpt
                        in_kpt = False
                    else:
                        in_kpt = False
                if s.startswith("channels:"):
                    try:
                        channels = int(s.split(":", 1)[1].strip())
                    except ValueError:
                        channels = None
            return kpt, channels

        rk, rc = parse(Path(rgb_yaml).read_text(encoding="utf-8"))
        dk, dc = parse(Path(rgbd_yaml).read_text(encoding="utf-8"))
        return (rk if rk is not None else dk), rc, dc

    @staticmethod
    def stat_existing_dataset(base, rgbd_sub, rgbd_yaml):
        """现场统计已有数据集：数 images 各 split，校验 RGB 与所选 RGBD 变体一致，
        读各自 yaml 的 kpt_shape/channels。返回 dict（含 rgb_dataset/rgbd_dataset 来源名），
        images 目录缺失时返回 None。只读不写回，不直接设置界面标签。"""
        splits = ("train", "val", "test")
        img_ext = (".png", ".jpg", ".jpeg")

        def count_imgs(sub):
            d = Path(base) / sub
            if not d.exists():
                return None
            return sum(1 for p in d.iterdir() if p.suffix.lower() in img_ext)

        rgb_counts = {s: count_imgs(f"rgb/images/{s}") for s in splits}
        rgbd_counts = {s: count_imgs(f"{rgbd_sub}/images/{s}") for s in splits}
        if any(c is None for c in rgb_counts.values()) or any(c is None for c in rgbd_counts.values()):
            return None
        mismatch = [s for s in splits if rgb_counts[s] != rgbd_counts[s]]
        mismatch_warning = None
        if mismatch:
            mismatch_warning = "RGB 与所选 RGBD 变体各 split 数量不一致: " + ", ".join(
                f"{s}: rgb={rgb_counts[s]} {rgbd_sub}={rgbd_counts[s]}" for s in mismatch)
        total = sum(rgb_counts.values())
        try:
            kpt, rc, dc = DatasetService.parse_yaml_meta(
                Path(base) / "rgb" / "data_rgb.yaml",
                Path(rgbd_yaml) if rgbd_yaml else Path(base) / rgbd_sub)
        except Exception:  # noqa: BLE001
            kpt, rc, dc = None, None, None
        result = {
            "total": total,
            "train": rgb_counts["train"],
            "val": rgb_counts["val"],
            "test": rgb_counts["test"],
            "keypoint_count": kpt,
            "rgb_channels": rc,
            "rgbd_channels": dc,
            "rgb_dataset": "rgb",
            "rgbd_dataset": rgbd_sub,
        }
        if mismatch_warning:
            result["mismatch_warning"] = mismatch_warning
        return result

    @staticmethod
    def merge_report(stat, base):
        """合并 dataset_report.json 有效字段与现场统计结果。

        返回 (merged_dict, ready_text)。
        report 有效字段优先，缺失/None 用现场统计补齐。
        """
        merged = dict(stat)
        rp = Path(base) / "dataset_report.json"
        if rp.exists():
            try:
                d = json.loads(rp.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                d = {}
            if isinstance(d, dict):
                report_map = {
                    "total_samples": "total",
                    "train_samples": "train",
                    "val_samples": "val",
                    "test_samples": "test",
                    "keypoint_count": "keypoint_count",
                    "rgb_channels": "rgb_channels",
                    "rgbd_channels": "rgbd_channels",
                }
                for rk, mk in report_map.items():
                    v = d.get(rk)
                    if v is not None:
                        merged[mk] = v
                if d.get("validation_passed"):
                    ready = "已有数据集已加载（report + 现场补齐）"
                else:
                    ready = "已有数据集校验未通过（report + 现场补齐）"
            else:
                ready = "已有数据集已加载（现场统计）"
        else:
            ready = "已有数据集已加载（现场统计）"
        return merged, ready

    @staticmethod
    def load_report_json(out_dir, cls):
        """从 prepare 产物路径读取 dataset_report.json，返回 dict 或 None。"""
        rp = Path(out_dir) / "dataset" / cls / "dataset_report.json"
        if not rp.exists():
            return None
        try:
            return json.loads(rp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def report_to_stats(d):
        """把 dataset_report.json 的字段名映射为 stats dict（供 DatasetPanel.set_stats）。

        缺失字段统一显示"-"；validation_passed 决定 ready 文本。
        """
        def _g(key):
            v = d.get(key)
            return v if v is not None else "-"
        merged = {
            "total": _g("total_samples"),
            "train": _g("train_samples"),
            "val": _g("val_samples"),
            "test": _g("test_samples"),
            "keypoint_count": d.get("keypoint_count"),
            "rgb_channels": d.get("rgb_channels"),
            "rgbd_channels": d.get("rgbd_channels"),
        }
        ready = "DATASET READY" if d.get("validation_passed") else "校验未通过"
        return merged, ready

    @staticmethod
    def format_id_consistency(rgb_cnt, rgbd_cnt, rgb_ids, rgbd_ids):
        """格式化 RGB vs RGBD test ID 一致性只读文本（None 表示目录缺失）。"""
        if rgb_ids is None or rgbd_ids is None:
            return "无法校验（目录缺失）"
        if rgb_ids == rgbd_ids:
            return f"{rgb_cnt}/{rgbd_cnt} 完全一致"
        only_rgb = len(rgb_ids - rgbd_ids)
        only_rgbd = len(rgbd_ids - rgb_ids)
        parts = []
        if only_rgb:
            parts.append(f"RGB 独有 {only_rgb}")
        if only_rgbd:
            parts.append(f"RGBD 独有 {only_rgbd}")
        return "不一致（" + "，".join(parts) + "）"

    @staticmethod
    def format_id_mismatch_error(rgb_cnt, rgbd_cnt, rgb_ids, rgbd_ids):
        """构造 test ID 不一致时对比测试被拒绝的错误消息（含前 10 个独有 ID）。"""
        only_rgb = sorted(rgb_ids - rgbd_ids)
        only_rgbd = sorted(rgbd_ids - rgb_ids)
        msg = (f"[错误] 对比测试被拒绝：RGB 与 RGBD test ID 不一致 "
               f"(RGB={rgb_cnt}, RGBD={rgbd_cnt})")
        if only_rgb:
            msg += f" | RGB 独有: {only_rgb[:10]}{'...' if len(only_rgb) > 10 else ''}"
        if only_rgbd:
            msg += f" | RGBD 独有: {only_rgbd[:10]}{'...' if len(only_rgbd) > 10 else ''}"
        return msg
