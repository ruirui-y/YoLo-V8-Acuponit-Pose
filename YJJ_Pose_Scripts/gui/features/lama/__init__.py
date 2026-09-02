"""LaMa Erasure Feature 扩展入口。

Phase 1（已完成）：LamaPage + InpaintCanvas + LamaController 骨架
- LamaPage: 工具栏 / 画布 / 状态栏 / 信号
- InpaintCanvas: 左键画 mask、右键擦、Ctrl+滚轮缩放
- LamaController: Open / Prev / Next / ClearMask 已接入

Phase 2（已完成）：MaskLabelService
- 7 component + bbox + 7 keypoints + YOLO label 格式

Phase 3（已完成）：ReferenceAlignmentService
- SetRef / predict / stable ID（按 (y,x) 升序固定 ID 0..6）

Phase 4（当前）：A + TestOne 走同一 predict() API
- LamaController 接入 ReferenceAlignmentService
- SetRef: 从 finalMask 提取 7 centers + (y,x) 排序 -> set_reference
- A 键: predict(mode="assist") -> 写回 canvas + 保存 current_prediction
- TestOne: predict(mode="strict") -> overlay 预览窗（不含 LaMa inference）
- QImage <-> numpy 转换辅助方法

Phase 5（待）：LamaInferenceService（ONNX 推理）
Phase 6（待）：Q 原子 Commit 状态机（Image+Label+Rolling Ref+Next）
"""
