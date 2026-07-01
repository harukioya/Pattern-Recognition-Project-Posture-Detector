"""CoachPanel — Ollama-backed coaching text with a status dot.

Public contract (do not change):
    - Class: CoachPanel(QWidget)
    - Slot: set_prediction(pred: Prediction)
        track the currently predicted label so we can decide whether a cue is
        stale (cue was for a different label than the current prediction).
    - Slot: set_cue(label_for: str, cue_text: str)
        called by CoachThread.cue_ready when Ollama returns text. label_for is
        the label the cue was generated for; only show it if it matches the
        most recent prediction.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QColor, QPainter, QPaintEvent
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.state import Prediction


# Palette ---------------------------------------------------------------------
_BG = "#0e1117"
_CARD = "#161b22"
_BORDER = "#1f2632"
_TEXT_PRIMARY = "#f0f2f5"
_TEXT_SECOND = "#9aa1b3"
_TEXT_DIM = "#5d6478"
_SUCCESS = "#69dc82"
_WARNING = "#ffa55f"

# Body messages ---------------------------------------------------------------
_MSG_NO_PRED = "Stand back so your full body fits in view"
_MSG_CORRECT = "Form looks clean. Keep your tempo steady."
_MSG_ANALYZING = "Analyzing your form…"


class _StatusDot(QWidget):
    """A small antialiased filled circle, ~10px diameter inside a 12x12 box."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self._color = QColor(_TEXT_DIM)

    def set_color(self, color: QColor) -> None:
        if color == self._color:
            return
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 (Qt)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._color)
        # Inset by 1px so the 10px circle sits centred in the 12x12 box.
        painter.drawEllipse(1, 1, 10, 10)
        painter.end()


class CoachPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CoachPanel")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        # Internal state -----------------------------------------------------
        self._current_label: str = ""
        self._current_is_correct: bool = False
        self._has_prediction: bool = False
        self._latest_cue_for: str = ""
        self._latest_cue_text: str = ""

        # Layout -------------------------------------------------------------
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        self._dot = _StatusDot(self)
        self._title = QLabel("COACH")
        self._title.setObjectName("CoachPanelTitle")
        header.addWidget(self._dot, 0, Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self._title, 0, Qt.AlignmentFlag.AlignVCenter)
        header.addStretch(1)
        root.addLayout(header)

        self._body = QLabel(_MSG_NO_PRED)
        self._body.setObjectName("CoachPanelBody")
        self._body.setWordWrap(True)
        self._body.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._body.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        # Reserve room for ~3 lines at 16pt with comfortable leading so the
        # panel does not reflow when the text length changes.
        self._body.setMinimumHeight(84)
        self._body.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        root.addWidget(self._body, 1)

        # Styling ------------------------------------------------------------
        self.setStyleSheet(
            f"""
            QWidget#CoachPanel {{
                background-color: {_CARD};
                border: 1px solid {_BORDER};
                border-radius: 14px;
            }}
            QLabel#CoachPanelTitle {{
                color: {_TEXT_SECOND};
                font-size: 12pt;
                font-weight: 600;
                background: transparent;
                border: none;
            }}
            QLabel#CoachPanelBody {{
                color: {_TEXT_PRIMARY};
                font-size: 16pt;
                background: transparent;
                border: none;
            }}
            """
        )

        self._refresh()

    # ------------------------------------------------------------------ slots
    @pyqtSlot(object)
    def set_prediction(self, pred: Prediction) -> None:
        if getattr(pred, "is_uncertain", False):
            self._has_prediction = True
            self._current_label = ""
            self._current_is_correct = False
            self._set_dot(_TEXT_DIM)
            self._set_body("Waiting for a clear pose…", _TEXT_SECOND)
            return
        self._has_prediction = True
        self._current_label = pred.label
        self._current_is_correct = bool(pred.is_correct)
        self._refresh()

    @pyqtSlot(str, str)
    def set_cue(self, label_for: str, cue_text: str) -> None:
        self._latest_cue_for = label_for
        self._latest_cue_text = cue_text
        self._refresh()

    def reset(self) -> None:
        """Called by MainWindow on welcome/mode-select transitions."""
        self._current_label = ""
        self._current_is_correct = False
        self._has_prediction = False
        self._latest_cue_for = ""
        self._latest_cue_text = ""
        self._refresh()

    # -------------------------------------------------------------- internal
    def _refresh(self) -> None:
        """Recompute dot colour and body text from internal state."""
        if not self._has_prediction:
            self._set_dot(_TEXT_DIM)
            self._set_body(_MSG_NO_PRED, _TEXT_PRIMARY)
            return

        if self._current_is_correct:
            self._set_dot(_SUCCESS)
            self._set_body(_MSG_CORRECT, _TEXT_PRIMARY)
            return

        # Incorrect form ----------------------------------------------------
        self._set_dot(_WARNING)
        has_fresh_cue = (
            self._latest_cue_for == self._current_label
            and bool(self._latest_cue_text)
        )
        if has_fresh_cue:
            self._set_body(self._latest_cue_text, _TEXT_PRIMARY)
        else:
            self._set_body(_MSG_ANALYZING, _TEXT_SECOND)

    def _set_dot(self, hex_color: str) -> None:
        self._dot.set_color(QColor(hex_color))

    def _set_body(self, text: str, hex_color: str) -> None:
        if self._body.text() != text:
            self._body.setText(text)
        # Per-state colour applied inline so transitions are crisp; the rest
        # of the body styling stays in the widget stylesheet.
        self._body.setStyleSheet(
            f"color: {hex_color};"
            " font-size: 16pt;"
            " background: transparent;"
            " border: none;"
        )
