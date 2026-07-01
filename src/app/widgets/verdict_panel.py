"""VerdictPanel — large form-verdict tile at the top of the right column.

Public contract (do not change):
    - Class: VerdictPanel(QWidget)
    - Slot: set_prediction(pred: Prediction)
    - objectName: "VerdictPanel"
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.state import Prediction


# Palette
COLOR_CARD = "#161b22"
COLOR_BORDER = "#1f2632"
COLOR_TEXT_PRIMARY = "#f0f2f5"
COLOR_TEXT_SECOND = "#9aa1b3"
COLOR_TEXT_DIM = "#5d6478"
COLOR_SUCCESS = "#69dc82"
COLOR_WARNING = "#ffa55f"
COLOR_ACCENT = "#78b4ff"


class VerdictPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("VerdictPanel")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        # Card styling on the widget itself (objectName-scoped to avoid leaking).
        self.setStyleSheet(
            f"""
            QWidget#VerdictPanel {{
                background-color: {COLOR_CARD};
                border: 1px solid {COLOR_BORDER};
                border-radius: 14px;
            }}
            """
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 16, 20, 16)
        outer.setSpacing(10)

        # Exercise tag (small, uppercase, letter-spaced).
        self._exercise = QLabel("READY")
        self._exercise.setStyleSheet(
            f"color: {COLOR_TEXT_SECOND};"
            "font-size: 11pt;"
            "font-weight: 600;"
            "background: transparent;"
            "border: none;"
        )

        # Verdict + confidence row.
        row = QHBoxLayout()
        row.setSpacing(16)
        row.setContentsMargins(0, 0, 0, 0)

        self._verdict = QLabel("Stand back so your full body fits in view")
        self._verdict.setWordWrap(True)
        self._verdict.setStyleSheet(
            f"color: {COLOR_TEXT_PRIMARY};"
            "font-size: 28pt;"
            "font-weight: 700;"
            "background: transparent;"
            "border: none;"
        )
        self._verdict.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._verdict.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self._confidence = QLabel("")
        self._confidence.setStyleSheet(
            f"color: {COLOR_TEXT_PRIMARY};"
            "font-size: 24pt;"
            "font-weight: 500;"
            "background: transparent;"
            "border: none;"
        )
        self._confidence.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._confidence.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        row.addWidget(self._verdict, 1)
        row.addWidget(self._confidence, 0)

        # 3-px coloured underline whose colour tracks the verdict.
        self._underline = QFrame()
        self._underline.setFixedHeight(3)
        self._underline.setFrameShape(QFrame.Shape.NoFrame)
        self._underline.setStyleSheet(
            f"background-color: {COLOR_BORDER};"
            "border: none;"
            "border-radius: 1px;"
        )
        self._underline.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        outer.addWidget(self._exercise)
        outer.addLayout(row)
        outer.addStretch(1)
        outer.addWidget(self._underline)

    # ------------------------------------------------------------------ mode
    def reset_for_mode(self, exercise: str | None) -> None:
        """Called by MainWindow on welcome/mode-select transitions."""
        if exercise is None:
            self._exercise.setText("READY")
            self._verdict.setText("Pick an exercise to start")
        else:
            display = exercise if exercise == "Lunges" else exercise.capitalize()
            self._exercise.setText(display.upper())
            self._verdict.setText("Stand back so your full body fits in view")
        self._verdict.setStyleSheet(
            f"color: {COLOR_TEXT_SECOND};"
            "font-size: 24pt;"
            "font-weight: 500;"
            "background: transparent;"
            "border: none;"
        )
        self._confidence.setText("")
        self._underline.setStyleSheet(
            f"background-color: {COLOR_BORDER};"
            "border: none;"
            "border-radius: 1px;"
        )

    # ------------------------------------------------------------------ slots
    @pyqtSlot(object)
    def set_prediction(self, pred: Prediction) -> None:
        if getattr(pred, "is_uncertain", False):
            self._exercise.setText("STATUS")
            self._verdict.setText("Hold a clear pose or start moving")
            color = COLOR_TEXT_SECOND
            self._verdict.setStyleSheet(
                f"color: {color};"
                "font-size: 24pt;"
                "font-weight: 500;"
                "background: transparent;"
                "border: none;"
            )
            self._confidence.setText("")
            self._underline.setStyleSheet(
                f"background-color: {COLOR_BORDER};"
                "border: none;"
                "border-radius: 1px;"
            )
            return

        self._exercise.setText("FORM")

        if pred.is_correct:
            verdict_text = "Form looks good"
            color = COLOR_SUCCESS
        else:
            verdict_text = pred.error_name.replace("_", " ").title()
            color = COLOR_WARNING

        self._verdict.setText(verdict_text)
        self._verdict.setStyleSheet(
            f"color: {color};"
            "font-size: 28pt;"
            "font-weight: 700;"
            "background: transparent;"
            "border: none;"
        )
        self._confidence.setText(f"{pred.confidence:.0%}")
        self._underline.setStyleSheet(
            f"background-color: {color};"
            "border: none;"
            "border-radius: 1px;"
        )
