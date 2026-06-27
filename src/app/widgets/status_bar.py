"""StatusBar — bottom strip with FPS, buffer fill, and a quit hint.

Public contract (do not change):
    - Class: AppStatusBar(QWidget)
    - Slot: set_fps(fps: float)
    - Slot: set_buffer(n: int)
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget


# Palette ---------------------------------------------------------------------
_BG = "#0e1117"
_BORDER = "#1f2632"
_TEXT_DIM = "#5d6478"

# Buffer capacity matches the pipeline's window length.
_BUFFER_CAPACITY = 64

# Monospace font candidates, in order of preference.
_MONO_FAMILIES = ["SF Mono", "Menlo", "Monaco", "Consolas", "monospace"]


def _mono_font() -> QFont:
    """Pick the first available monospace family from a curated list."""
    font = QFont()
    font.setFamilies(_MONO_FAMILIES)
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPointSize(10)
    return font


class AppStatusBar(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("AppStatusBar")
        self.setFixedHeight(28)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Cached primitives -------------------------------------------------
        self._fps: float = 0.0
        self._buf: int = 0

        # Layout ------------------------------------------------------------
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(8)

        mono = _mono_font()

        self._left = QLabel(self._format_left())
        self._left.setObjectName("AppStatusBarLeft")
        self._left.setFont(mono)
        self._left.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )

        self._right = QLabel("Q to quit")
        self._right.setObjectName("AppStatusBarRight")
        self._right.setFont(mono)
        self._right.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        layout.addWidget(self._left, 0)
        layout.addStretch(1)
        layout.addWidget(self._right, 0)

        # Styling -----------------------------------------------------------
        # Solid bg, 1px top border in border colour, no other chrome.
        self.setStyleSheet(
            f"""
            QWidget#AppStatusBar {{
                background-color: {_BG};
                border: none;
                border-top: 1px solid {_BORDER};
            }}
            QLabel#AppStatusBarLeft,
            QLabel#AppStatusBarRight {{
                color: {_TEXT_DIM};
                background: transparent;
                border: none;
                padding: 0;
            }}
            """
        )

    # ------------------------------------------------------------------ slots
    @pyqtSlot(float)
    def set_fps(self, fps: float) -> None:
        self._fps = float(fps)
        self._left.setText(self._format_left())

    @pyqtSlot(int)
    def set_buffer(self, n: int) -> None:
        self._buf = int(n)
        self._left.setText(self._format_left())

    # -------------------------------------------------------------- internal
    def _format_left(self) -> str:
        return f"{self._fps:5.1f} fps   buffer {self._buf:>2}/{_BUFFER_CAPACITY}"
