"""CameraWidget — displays the live webcam frame with a skeleton overlay.

Public contract (do not change):
    - Class: CameraWidget(QWidget)
    - Slot: set_frame(qimage: QImage, landmarks_xy: list[tuple[float, float]])
      where landmarks_xy contains (x_norm, y_norm) in [0, 1] for each of the 33
      BlazePose joints. Empty list = no person detected this frame.
"""
from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt, pyqtSlot
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import QWidget


# Hard-coded BlazePose 33-joint connections (no mediapipe import).
BLAZEPOSE_EDGES: list[tuple[int, int]] = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
]

# Palette (kept local so this widget is self-contained).
_BG_COLOR = QColor("#0e1117")
_CARD_COLOR = QColor("#161b22")
_BORDER_COLOR = QColor("#262c36")
_ACCENT = QColor("#78b4ff")
_MUTED_TEXT = QColor("#9aa1b3")

_CORNER_RADIUS = 12.0
_JOINT_RADIUS = 5.0
_BONE_WIDTH = 2.0


class CameraWidget(QWidget):
    """Polished video-player-style camera surface with BlazePose overlay."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._qimage: QImage | None = None
        self._landmarks: list[tuple[float, float]] = []
        self.setMinimumSize(320, 240)
        # Opaque background; we paint everything ourselves.
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    # ------------------------------------------------------------------
    # Public slot — frozen signature.
    # ------------------------------------------------------------------
    @pyqtSlot(QImage, object)
    def set_frame(self, qimage: QImage, landmarks_xy: list) -> None:
        # Copy guards against the producer recycling the underlying buffer.
        self._qimage = qimage.copy() if not qimage.isNull() else None
        self._landmarks = list(landmarks_xy) if landmarks_xy else []
        self.update()

    # ------------------------------------------------------------------
    # Qt overrides.
    # ------------------------------------------------------------------
    def resizeEvent(self, ev) -> None:  # noqa: N802 (Qt naming)
        super().resizeEvent(ev)
        self.update()

    def paintEvent(self, ev) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        widget_rect = QRectF(self.rect())

        # Outer charcoal background (fills corners outside the rounded card).
        painter.fillRect(widget_rect, _BG_COLOR)

        # Rounded card path; everything inside is clipped to it.
        card_path = QPainterPath()
        card_path.addRoundedRect(widget_rect, _CORNER_RADIUS, _CORNER_RADIUS)
        painter.save()
        painter.setClipPath(card_path)

        # Card fill (shows through letterbox bars if the frame doesn't fill).
        painter.fillRect(widget_rect, _CARD_COLOR)

        # Draw the scaled, letterboxed frame.
        frame_rect = self._fitted_rect(widget_rect)
        if self._qimage is not None and frame_rect.width() > 0 and frame_rect.height() > 0:
            painter.drawImage(frame_rect, self._qimage)
            self._draw_skeleton(painter, frame_rect)
        else:
            self._draw_no_person(painter, widget_rect)

        # If we have a frame but no landmarks, overlay the hint inside the frame.
        if self._qimage is not None and not self._landmarks:
            self._draw_no_person(painter, frame_rect)

        painter.restore()

        # Subtle rounded border on top of the clipped contents.
        pen = QPen(_BORDER_COLOR)
        pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        # Inset by 0.5px so the 1px stroke sits inside the widget.
        painter.drawPath(card_path)

        painter.end()

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------
    def _fitted_rect(self, bounds: QRectF) -> QRectF:
        """Aspect-ratio-preserving rect for the current frame, centred in bounds."""
        if self._qimage is None or self._qimage.isNull():
            return QRectF()
        img_w = self._qimage.width()
        img_h = self._qimage.height()
        if img_w <= 0 or img_h <= 0:
            return QRectF()
        scale = min(bounds.width() / img_w, bounds.height() / img_h)
        w = img_w * scale
        h = img_h * scale
        x = bounds.x() + (bounds.width() - w) / 2.0
        y = bounds.y() + (bounds.height() - h) / 2.0
        return QRectF(x, y, w, h)

    def _draw_skeleton(self, painter: QPainter, rect: QRectF) -> None:
        if not self._landmarks:
            return
        pts = [
            (rect.x() + x * rect.width(), rect.y() + y * rect.height())
            for (x, y) in self._landmarks
        ]

        # Bones first so joints sit on top.
        bone_color = QColor(_ACCENT)
        bone_color.setAlphaF(0.85)
        pen = QPen(bone_color)
        pen.setWidthF(_BONE_WIDTH)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        for a, b in BLAZEPOSE_EDGES:
            if a < len(pts) and b < len(pts):
                ax, ay = pts[a]
                bx, by = pts[b]
                painter.drawLine(int(round(ax)), int(round(ay)), int(round(bx)), int(round(by)))

        # Joints.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(_ACCENT))
        for x, y in pts:
            painter.drawEllipse(QRectF(x - _JOINT_RADIUS, y - _JOINT_RADIUS,
                                       2 * _JOINT_RADIUS, 2 * _JOINT_RADIUS))

    def _draw_no_person(self, painter: QPainter, rect: QRectF) -> None:
        font = QFont(painter.font())
        font.setPointSize(16)
        painter.setFont(font)
        painter.setPen(_MUTED_TEXT)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No person detected")
