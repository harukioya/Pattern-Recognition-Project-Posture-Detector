"""HeaderBar — prominent top strip showing the current exercise.

Sits above the main camera+panels row. Mirrors AppStatusBar's design at the
top of the window.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget

from app.state import Prediction


_BG = "#0e1117"
_BORDER = "#1f2632"
_TEXT_PRIMARY = "#f0f2f5"
_TEXT_SECOND = "#9aa1b3"


class HeaderBar(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("HeaderBar")
        self.setFixedHeight(64)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(12)

        # Exercise (large) - the headline
        self._exercise = QLabel("Posture Coach")
        self._exercise.setObjectName("HeaderExercise")
        self._exercise.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )

        # Subtitle on the right — small app branding
        self._right = QLabel("Posture Coach")
        self._right.setObjectName("HeaderRight")
        self._right.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        layout.addWidget(self._exercise, 1)
        layout.addWidget(self._right, 0)

        # Track the active per-exercise mode. When set, prediction updates
        # keep the header pinned to this label instead of following the
        # (legacy) gate's argmax.
        self._mode: str | None = None

        self.setStyleSheet(
            f"""
            QWidget#HeaderBar {{
                background-color: {_BG};
                border: none;
                border-bottom: 1px solid {_BORDER};
            }}
            QLabel#HeaderExercise {{
                color: {_TEXT_PRIMARY};
                font-size: 26pt;
                font-weight: 700;
                background: transparent;
                border: none;
            }}
            QLabel#HeaderRight {{
                color: {_TEXT_SECOND};
                font-size: 11pt;
                font-weight: 500;
                background: transparent;
                border: none;
            }}
            """
        )

    def set_mode(self, exercise: str | None) -> None:
        """Called by MainWindow on welcome/mode-select transitions.

        `None` -> welcome-screen title; otherwise pin the header to the
        selected exercise so the user always sees which specialist is
        active, independent of prediction confidence.
        """
        self._mode = exercise
        if exercise is None:
            self._exercise.setText("Choose your exercise")
            self._exercise.setStyleSheet(
                f"color: {_TEXT_SECOND};"
                "font-size: 26pt;"
                "font-weight: 600;"
                "background: transparent;"
                "border: none;"
            )
            self._right.setText("Posture Coach")
            return
        display = exercise if exercise == "Lunges" else exercise.capitalize()
        self._exercise.setText(display)
        self._exercise.setStyleSheet(
            f"color: {_TEXT_PRIMARY};"
            "font-size: 26pt;"
            "font-weight: 700;"
            "background: transparent;"
            "border: none;"
        )
        self._right.setText("H → home    Q → quit")

    @pyqtSlot(object)
    def set_prediction(self, pred: Prediction) -> None:
        # In per-exercise mode the header title stays pinned to the mode.
        # We only use predictions to shift between "waiting" grey and the
        # normal white treatment of the mode name.
        if self._mode is None:
            return
        display = self._mode if self._mode == "Lunges" else self._mode.capitalize()
        self._exercise.setText(display)
        if getattr(pred, "is_uncertain", False):
            self._exercise.setStyleSheet(
                f"color: {_TEXT_SECOND};"
                "font-size: 26pt;"
                "font-weight: 600;"
                "background: transparent;"
                "border: none;"
            )
            return
        self._exercise.setStyleSheet(
            f"color: {_TEXT_PRIMARY};"
            "font-size: 26pt;"
            "font-weight: 700;"
            "background: transparent;"
            "border: none;"
        )
