"""LaMa 擦除画布控件（PySide6 QGraphicsView）。

参考 C++ LamaErasure/InpaintCanvas.cpp /.h，仅迁移本工作流真正需要的部分：
- 图片显示（QGraphicsPixmapItem）
- Mask 半透明红色 overlay 显示
- 左键连续绘制白色 mask
- 右键连续擦除 mask
- 画笔半径可调
- Ctrl+滚轮缩放

明确不实现（与 C++ 历史功能切割）：
- TrackingROI（C++ 黄色矩形，本工作流不用）
- 手动画 BoundingBox（C++ Shift+左键，本工作流不用）
- Canvas labelPoints（C++ 记录穴位点数组，本工作流标签从 finalMask 重提）
- 任何 LaMa / Reference / YOLO / Pose 业务逻辑

外部 API（与 C++ InpaintCanvas 对齐）：
    load_image(path) -> bool
    set_image(qimage)            # 同时清空 mask
    set_mask(qimage_gray8)
    clear_mask()
    source_image() -> QImage     # RGB888
    mask_image()  -> QImage      # Grayscale8，0=保留 / 255=洞
    set_brush_radius(int)
    brush_radius() -> int

信号：
    maskChanged()                # mask 被改动（画/擦/clear/set）
    imageChanged()               # 主图被换（load/set）

坐标约定（与 C++ 一致）：
    scene == image，即 scene 坐标即原图像素坐标。
    鼠标 widget 坐标 -> mapToScene -> 原图像素坐标。
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QPoint
from PySide6.QtGui import (
    QImage, QPainter, QPen, QColor, QBrush, QPixmap,
    QMouseEvent, QWheelEvent, QResizeEvent,
)
from PySide6.QtWidgets import (
    QWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
)


# ============================================================ 工具：mask -> 红色半透明 overlay
def _make_mask_overlay_rgba(mask_gray: QImage, alpha: int = 110) -> QImage:
    """把 0/255 灰度 mask 转成半透明红色 RGBA overlay（与 C++ makeMaskOverlayRGBA 一致）。

    P1-3: 禁止 Python H*W 双层像素循环；改用 vectorized numpy + QImage buffer
    一次性构造 RGBA 数组，再 wrap 成 QImage。每次 mouseMove 只走一次 numpy 操作。
    """
    if mask_gray.isNull():
        return QImage()
    # 统一格式：Grayscale8
    m = mask_gray.convertToFormat(QImage.Format.Format_Grayscale8)
    w, h = m.width(), m.height()
    stride = m.bytesPerLine()
    # 一次性取出 mask bytes -> numpy (h, stride)，再切到 (h, w)
    buf = bytes(m.constBits())
    arr = np.frombuffer(buf, dtype=np.uint8, count=h * stride).reshape(h, stride)
    if stride != w:
        arr = arr[:, :w]
    mask_np = arr.copy()      # 脱离 QImage 内部 buffer

    # 构造 RGBA uint8 数组（HxWx4），非零位置写红色，零位置写透明
    # Format_ARGB32_Premultiplied 内存字节序为 BGRA（小端），且 R/G/B 已预乘 A/255
    # 红色 unpremultiplied = (R=255, G=0, B=0, A=alpha)
    # 预乘后 R = 255 * alpha / 255 = alpha，所以 premultiplied = (B=0, G=0, R=alpha, A=alpha)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    nonzero = mask_np > 0
    alpha_u8 = int(alpha)
    rgba[..., 2][nonzero] = alpha_u8               # R (premultiplied)
    rgba[..., 3][nonzero] = alpha_u8               # A

    # numpy 数组 -> QImage（用 bytesPerLine = w*4，连续无 padding）
    rgba = np.ascontiguousarray(rgba)
    out = QImage(rgba.tobytes(), w, h, w * 4, QImage.Format.Format_ARGB32_Premultiplied)
    # 必须 copy 脱离 numpy buffer ownership，避免外部修改
    return out.copy()


# ============================================================ InpaintCanvas
class InpaintCanvas(QGraphicsView):
    """LaMa 工作流专用画布：左键画 mask、右键擦 mask，仅此而已。

    不含 TrackROI / BBox / labelPoints / Pose / Reference 任何业务。
    """

    # ---- 对外信号 ----
    maskChanged = Signal()
    imageChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # ---- 渲染参数（与 C++ 一致）----
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

        # ---- 内部状态 ----
        self._src: QImage = QImage()                            # RGB888
        self._mask: QImage = QImage()                           # Grayscale8 (0=keep, 255=hole)
        self._brush_radius: int = 9
        self._drawing: bool = False
        self._last_paint_pos: QPointF = QPointF()
        self._zoom: float = 1.0

        # ---- 场景与图层 ----
        self._scene: QGraphicsScene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item_img: QGraphicsPixmapItem = self._scene.addPixmap(QPixmap())  # 底层：原图
        self._item_mask: QGraphicsPixmapItem = self._scene.addPixmap(QPixmap()) # 上层：mask overlay
        self._item_mask.setZValue(10)

    # ============================================================ 公开 API
    def load_image(self, path: str) -> bool:
        """从文件加载图片，自动清空 mask。"""
        img = QImage(path)
        if img.isNull():
            return False
        self.set_image(img)
        return True

    def set_image(self, img: QImage) -> None:
        """设置当前显示图（同时重置 mask）。"""
        if img.isNull():
            return
        self._src = img.convertToFormat(QImage.Format.Format_RGB888)
        # 重置 mask：与原图同尺寸，纯黑（0）
        self._mask = QImage(self._src.size(), QImage.Format.Format_Grayscale8)
        self._mask.fill(0)

        self._update_pixmap()
        self._update_mask_pixmap()

        self._scene.setSceneRect(QRectF(0, 0, self._src.width(), self._src.height()))
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        self.imageChanged.emit()

    def set_mask(self, mask_gray: QImage) -> None:
        """外部设置 mask（Grayscale8，0/255）。尺寸不符会快速缩放。"""
        if self._src.isNull():
            return
        m = mask_gray.convertToFormat(QImage.Format.Format_Grayscale8)
        if m.size() != self._src.size():
            m = m.scaled(self._src.size(),
                         Qt.AspectRatioMode.IgnoreAspectRatio,
                         Qt.TransformationMode.FastTransformation)
        self._mask = m
        self._update_mask_pixmap()
        self.maskChanged.emit()

    def clear_mask(self):
        """清空 mask。"""
        if self._mask.isNull():
            return
        self._mask.fill(0)
        self._update_mask_pixmap()
        self.maskChanged.emit()

    def source_image(self) -> QImage:
        """当前原图（RGB888）。"""
        return self._src

    def mask_image(self) -> QImage:
        """当前 mask（Grayscale8，0=保留 / 255=洞）。"""
        return self._mask

    def set_brush_radius(self, r: int) -> None:
        self._brush_radius = max(1, r)

    def brush_radius(self) -> int:
        return self._brush_radius

    # ============================================================ 内部更新
    def _update_pixmap(self) -> None:
        if self._src.isNull():
            return
        self._item_img.setPixmap(QPixmap.fromImage(self._src))

    def _update_mask_pixmap(self) -> None:
        if self._mask.isNull():
            return
        overlay = _make_mask_overlay_rgba(self._mask, 110)
        self._item_mask.setPixmap(QPixmap.fromImage(overlay))

    def _view_to_scene_pos(self, vp: QPoint) -> QPointF:
        return self.mapToScene(vp)

    # ============================================================ 画笔
    def _apply_brush(self, a: QPointF, b: QPointF, erase: bool) -> None:
        """在 mask 上画一笔（a->b 连线 + b 处圆）。"""
        if self._src.isNull() or self._mask.isNull():
            return
        p = QPainter(self._mask)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        c = Qt.GlobalColor.black if erase else Qt.GlobalColor.white
        pen = QPen(c, self._brush_radius * 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(QBrush(c))
        # 连续笔触
        p.drawLine(a, b)
        # 单击或慢移也形成圆形笔触
        p.drawEllipse(b, float(self._brush_radius), float(self._brush_radius))
        p.end()

    # ============================================================ 鼠标事件
    def mousePressEvent(self, e: QMouseEvent) -> None:
        if self._src.isNull():
            super().mousePressEvent(e)
            return
        # 只处理左/右键；中键等交给默认
        if e.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            super().mousePressEvent(e)
            return
        self._drawing = True
        sp = self._view_to_scene_pos(e.position().toPoint())
        # 左键画白，右键擦黑（与 C++ 一致）
        erase = (e.button() == Qt.MouseButton.RightButton)
        self._apply_brush(sp, sp, erase)
        self._last_paint_pos = sp
        self._update_mask_pixmap()
        self.maskChanged.emit()
        e.accept()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        # 连续画笔：左键拖动涂白、右键拖动擦黑
        buttons = e.buttons()
        if buttons & (Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton):
            sp = self._view_to_scene_pos(e.position().toPoint())
            erase = bool(buttons & Qt.MouseButton.RightButton)
            self._apply_brush(self._last_paint_pos, sp, erase)
            self._last_paint_pos = sp
            self._update_mask_pixmap()
            self.maskChanged.emit()
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        self._drawing = False
        super().mouseReleaseEvent(e)

    def wheelEvent(self, e: QWheelEvent) -> None:
        # Ctrl+滚轮缩放（与 C++ 一致）
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.1 if e.angleDelta().y() > 0 else 1.0 / 1.1
            self._zoom *= factor
            self.scale(factor, factor)
            e.accept()
            return
        super().wheelEvent(e)

    def resizeEvent(self, e: QResizeEvent) -> None:
        super().resizeEvent(e)
        # 不强行 fit，避免用户缩放被重置（与 C++ 一致）
