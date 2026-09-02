"""LaMa Feature services 包。

- mask_label_service: Final Mask -> YOLO Pose Label
- reference_alignment_service: 多点局部模板追踪（rolling reference + predict）
- lama_inference_service: LaMa ONNX Runtime 推理
"""
from .mask_label_service import MaskLabelService, MaskComponent, LabelResult
from .reference_alignment_service import (
    ReferenceAlignmentService, LocalTrack, TrackPrediction, PredictionResult,
)
from .lama_inference_service import LamaInferenceService

__all__ = [
    "MaskLabelService", "MaskComponent", "LabelResult",
    "ReferenceAlignmentService", "LocalTrack", "TrackPrediction", "PredictionResult",
    "LamaInferenceService",
]
