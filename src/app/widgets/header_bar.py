"""HeaderBar - prominent top strip showing the current exercise.

Sits above the main camera+panels row. Mirrors AppStatusBar's design at the
top of the window.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

from app.state import Prediction


_BG = "#0e1117"
_BORDER = "#1f2632"
_TEXT_PRIMARY = "#f0f2f5"
_TEXT_SECOND = "#9aa1b3"
_RECORD = "#ff5f57"
_RECORD_DARK = "#2b1111"


class HeaderBar(QWidget):
    record_toggled = pyqtSignal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        recording_enabled: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("HeaderBar")
        self.setFixedHeight(64)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._recording_enabled = recording_enabled

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(12)

        # Exercise (large) - the headline
        self._exercise = QLabel("Posture Coach")
        self._exercise.setObjectName("HeaderExercise")
        self._exercise.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )

        # Subtitle on the right - small app branding / shortcuts
        self._right = QLabel("Posture Coach")
        self._right.setObjectName("HeaderRight")
        self._right.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        self._record_button: QPushButton | None = None
        if recording_enabled:
            self._record_button = QPushButton("Record")
            self._record_button.setObjectName("RecordButton")
            self._record_button.setCursor(Qt.CursorShape.PointingHandCursor)
            self._record_button.setProperty("recording", False)
            self._record_button.clicked.connect(self.record_toggled.emit)

        layout.addWidget(self._exercise, 1)
        layout.addWidget(self._right, 0)
        if self._record_button is not None:
            layout.addWidget(self._record_button, 0)

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
            QPushButton#RecordButton {{
                background-color: transparent;
                color: {_TEXT_PRIMARY};
                border: 1px solid {_BORDER};
                border-radius: 6px;
                padding: 7px 14px;
                font-size: 11pt;
                font-weight: 700;
            }}
            QPushButton#RecordButton:hover {{
                border-color: {_RECORD};
            }}
            QPushButton#RecordButton[recording="true"] {{
                background-color: {_RECORD_DARK};
                color: {_RECORD};
                border-color: {_RECORD};
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
        if self._recording_enabled:
            self._right.setText("R record    H home    Q quit")
        else:
            self._right.setText("H home    Q quit")

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

    @pyqtSlot(bool, str)
    def set_recording(self, active: bool, path: str) -> None:
        if self._record_button is None:
            return
        self._record_button.setText("Stop" if active else "Record")
        self._record_button.setProperty("recording", bool(active))
        self._record_button.style().unpolish(self._record_button)
        self._record_button.style().polish(self._record_button)
        self._record_button.update()

        if active:
            self._right.setText("Recording")
        elif path:
            self._right.setText(f"Saved {Path(path).name}")
        elif self._mode is not None:
            self._right.setText("R record    H home    Q quit")
        else:
            self._right.setText("Posture Coach")