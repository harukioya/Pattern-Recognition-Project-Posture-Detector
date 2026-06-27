"""BarsPanel — grouped probability bars by exercise.

Public contract (do not change):
    - Class: BarsPanel(QWidget)
    - Slot: set_prediction(pred: Prediction)
    - objectName: "BarsPanel"
"""
from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QRectF, QSize, Qt, pyqtSlot
from PyQt6.QtGui import QColor, QPainter, QPainterPath
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.state import DISPLAY_CLASSES, EXERCISES, Prediction, labels_for_exercise


# Palette
COLOR_CARD = "#161b22"
COLOR_BORDER = "#1f2632"
COLOR_TEXT_PRIMARY = "#f0f2f5"
COLOR_TEXT_SECOND = "#9aa1b3"
COLOR_TEXT_DIM = "#5d6478"
COLOR_SUCCESS = "#69dc82"
COLOR_WARNING = "#ffa55f"
COLOR_BAR_TRACK = "#1f2632"


# -------------------------------------------------------------------- bar widget
class ProbBar(QWidget):
    """Custom horizontal bar painted with QPainter."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._value: float = 0.0
        self._is_top: bool = False
        self._is_correct_top: bool = False
        self._is_active_section: bool = True
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(14)

    # ---------- accessors
    @property
    def value(self) -> float:
        return self._value

    @property
    def is_top(self) -> bool:
        return self._is_top

    @property
    def is_correct_top(self) -> bool:
        return self._is_correct_top

    def set_state(
        self,
        value: float,
        is_top: bool,
        is_correct_top: bool,
        is_active_section: bool = True,
    ) -> None:
        v = float(value)
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        self._value = v
        self._is_top = bool(is_top)
        self._is_correct_top = bool(is_correct_top)
        self._is_active_section = bool(is_active_section)
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt API)
        return QSize(140, 14)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt API)
        return QSize(80, 14)

    # ---------- paint
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = self.width()
        h = self.height()
        bar_h = 8.0
        y = (h - bar_h) / 2.0
        radius = 3.0

        # Track
        track_rect = QRectF(0.0, y, float(w), bar_h)
        track_path = QPainterPath()
        track_path.addRoundedRect(track_rect, radius, radius)
        painter.fillPath(track_path, QColor(COLOR_BAR_TRACK))

        # Fill
        fill_w = max(0.0, min(1.0, self._value)) * float(w)
        if fill_w > 0.0:
            fill_rect = QRectF(0.0, y, fill_w, bar_h)
            fill_path = QPainterPath()
            fill_path.addRoundedRect(fill_rect, radius, radius)

            if not self._is_active_section:
                color = QColor(COLOR_TEXT_DIM)
                color.setAlphaF(0.35)
            elif self._is_top and self._is_correct_top:
                color = QColor(COLOR_SUCCESS)
            elif self._is_top:
                color = QColor(COLOR_WARNING)
            else:
                color = QColor(COLOR_TEXT_DIM)
                color.setAlphaF(0.7)

            painter.fillPath(fill_path, color)

        painter.end()


# -------------------------------------------------------------------- panel
@dataclass
class _Row:
    label_widget: QLabel
    bar: ProbBar
    value_widget: QLabel
    index: int
    exercise: str


def _humanise(name: str) -> str:
    return name.replace("_", " ")


class BarsPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("BarsPanel")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.setStyleSheet(
            f"""
            QWidget#BarsPanel {{
                background-color: {COLOR_CARD};
                border: 1px solid {COLOR_BORDER};
                border-radius: 14px;
            }}
            """
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)

        self._rows: dict[str, _Row] = {}
        self._section_headers: dict[str, QLabel] = {}

        for i, exercise in enumerate(EXERCISES):
            if i > 0:
                divider = QFrame()
                divider.setFrameShape(QFrame.Shape.NoFrame)
                divider.setFixedHeight(1)
                divider.setStyleSheet(
                    f"background-color: {COLOR_BORDER}; border: none;"
                )
                divider.setSizePolicy(
                    QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
                )
                outer.addWidget(divider)

            section_header = QLabel(exercise.upper())
            section_header.setStyleSheet(
                f"color: {COLOR_TEXT_SECOND};"
                "font-size: 10pt;"
                "font-weight: 600;"
                "background: transparent;"
                "border: none;"
            )
            outer.addWidget(section_header)
            self._section_headers[exercise] = section_header

            for class_name in labels_for_exercise(exercise):
                idx = DISPLAY_CLASSES.index(class_name)
                error_name = class_name.split("/", 1)[1]

                row_layout = QHBoxLayout()
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(10)

                label = QLabel(_humanise(error_name))
                label.setFixedWidth(130)
                label.setStyleSheet(
                    f"color: {COLOR_TEXT_SECOND};"
                    "font-size: 12pt;"
                    "background: transparent;"
                    "border: none;"
                )
                label.setAlignment(
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                )

                bar = ProbBar()

                value = QLabel("0.00")
                value.setFixedWidth(40)
                value.setStyleSheet(
                    f"color: {COLOR_TEXT_DIM};"
                    "font-size: 11pt;"
                    "background: transparent;"
                    "border: none;"
                )
                value.setAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )

                row_layout.addWidget(label, 0)
                row_layout.addWidget(bar, 1)
                row_layout.addWidget(value, 0)

                outer.addLayout(row_layout)

                self._rows[class_name] = _Row(
                    label_widget=label,
                    bar=bar,
                    value_widget=value,
                    index=idx,
                    exercise=exercise,
                )

        outer.addStretch(1)

    # ------------------------------------------------------------------ slots
    @pyqtSlot(object)
    def set_prediction(self, pred: Prediction) -> None:
        probs = pred.probs
        top_label = pred.label
        is_correct = pred.is_correct
        gated = pred.gated_exercise
        gate_probs = pred.gate_probs
        # When the gate is uncertain, treat *no* section as active so nothing
        # is highlighted — bars just show raw model probabilities, dim.
        if getattr(pred, "is_uncertain", False):
            gated = "__none__"

        # Section headers: append gate confidence and highlight the active one.
        for j, ex in enumerate(EXERCISES):
            header = self._section_headers.get(ex)
            if header is None:
                continue
            gate_pct = int(round(float(gate_probs[j]) * 100))
            header.setText(f"{ex.upper()}    {gate_pct}%")
            if ex == gated:
                header.setStyleSheet(
                    f"color: {COLOR_TEXT_PRIMARY};"
                    "font-size: 10pt;"
                    "font-weight: 600;"
                    "background: transparent;"
                    "border: none;"
                )
            else:
                header.setStyleSheet(
                    f"color: {COLOR_TEXT_DIM};"
                    "font-size: 10pt;"
                    "font-weight: 600;"
                    "background: transparent;"
                    "border: none;"
                )

        for class_name, row in self._rows.items():
            p = float(probs[row.index])
            is_top = class_name == top_label
            is_active = row.exercise == gated
            row.bar.set_state(p, is_top, is_correct, is_active_section=is_active)
            row.value_widget.setText(f"{p:.2f}")

            if not is_active:
                value_color = COLOR_TEXT_DIM
                label_color = COLOR_TEXT_DIM
            elif is_top and is_correct:
                value_color = COLOR_SUCCESS
                label_color = COLOR_TEXT_PRIMARY
            elif is_top:
                value_color = COLOR_WARNING
                label_color = COLOR_TEXT_PRIMARY
            else:
                value_color = COLOR_TEXT_DIM
                label_color = COLOR_TEXT_SECOND

            row.value_widget.setStyleSheet(
                f"color: {value_color};"
                "font-size: 11pt;"
                "background: transparent;"
                "border: none;"
            )
            row.label_widget.setStyleSheet(
                f"color: {label_color};"
                "font-size: 12pt;"
                "background: transparent;"
                "border: none;"
            )
