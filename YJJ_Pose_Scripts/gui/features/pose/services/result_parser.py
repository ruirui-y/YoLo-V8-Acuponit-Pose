"""stdout 结果解析服务（纯逻辑，不依赖 QWidget）。

从 stdout 中提取 POSE_EVAL_JSON / DEPTH_ABLATION_JSON 等结果 JSON，
Controller 只接收解析后的 dict。
"""
import json


class ResultParser:

    @staticmethod
    def parse_pose_eval_json(line):
        """从单行 stdout 中提取 POSE_EVAL_JSON 后面的 JSON dict。

        不含 POSE_EVAL_JSON 标记时返回 None；解析失败时返回 None。
        """
        if "POSE_EVAL_JSON" not in line:
            return None
        try:
            return json.loads(line.split("POSE_EVAL_JSON", 1)[1].strip())
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def parse_depth_ablation_json(line):
        """从单行 stdout 中提取 DEPTH_ABLATION_JSON 后面的 JSON dict。

        不含 DEPTH_ABLATION_JSON 标记时返回 None；解析失败时返回 None。
        """
        if "DEPTH_ABLATION_JSON" not in line:
            return None
        try:
            return json.loads(line.split("DEPTH_ABLATION_JSON", 1)[1].strip())
        except Exception:  # noqa: BLE001
            return None
