"""WelcomeScreen — pick an exercise before the workout starts.

Public contract:
    - Class: WelcomeScreen(QWidget)
    - Signal: mode_selected(str) with values in EXERCISES

Shown as the first page of the MainWindow's QStackedWidget on the
experiment/per-exercise-modes branch. Once the user picks a card the
pipeline loads that exercise's specialist ensemble and swaps to the
workout view.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.state import EXERCISES

_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "assets"

_CARD_IMAGES: dict[str, str] = {
    "SQUAT": "squat.jpg",
    "Lunges": "lunges.jpg",
    "Plank": "plank.png",
}


_BG = "#0e1117"
_CARD = "#161b22"
_CARD_HOVER = "#1c2230"
_BORDER = "#1f2632"
_ACCENT = "#78b4ff"
_TEXT_PRIMARY = "#f0f2f5"
_TEXT_SECOND = "#9aa1b3"
_TEXT_DIM = "#5d6478"


_CARD_SUBTITLES: dict[str, str] = {
    "SQUAT": "5 form cues — depth, stance, knee track, torso lean",
    "Lunges": "3 form cues — depth and knee-over-toe",
    "Plank": "3 form cues — back arch and hunch",
}

_HOTKEYS: dict[str, str] = {
    "SQUAT": "S",
    "Lunges": "L",
    "Plank": "P",
}


_CARD_RADIUS = 16.0


class ModeCard(QFrame):
    """One clickable card in the welcome screen. Full-bleed exercise photo
    with a bottom gradient scrim for legibility.
    """

    clicked = pyqtSignal(str)

    def __init__(self, exercise: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.exercise = exercise
        self.setObjectName("ModeCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(260, 320)
        # Prevent QSS on the frame from painting an opaque rectangle over the
        # custom paintEvent. All the visuals come from paintEvent below.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)

        self._hover: bool = False
        self._pixmap: QPixmap | None = self._load_pixmap(exercise)

        display_name = exercise if exercise == "Lunges" else exercise.capitalize()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 22, 24, 22)
        outer.setSpacing(8)

        hotkey = QLabel(_HOTKEYS[exercise])
        hotkey.setObjectName("ModeCardHotkey")
        hotkey.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        hotkey.setFixedWidth(36)

        title = QLabel(display_name)
        title.setObjectName("ModeCardTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        subtitle = QLabel(_CARD_SUBTITLES[exercise])
        subtitle.setObjectName("ModeCardSubtitle")
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        outer.addWidget(hotkey, 0, Qt.AlignmentFlag.AlignLeft)
        outer.addStretch(1)
        outer.addWidget(title, 0)
        outer.addWidget(subtitle, 0)

        self._apply_label_style()

    @staticmethod
    def _load_pixmap(exercise: str) -> QPixmap | None:
        name = _CARD_IMAGES.get(exercise)
        if name is None:
            return None
        path = _ASSETS_DIR / name
        if not path.exists():
            return None
        pix = QPixmap(str(path))
        return pix if not pix.isNull() else None

    def _apply_label_style(self) -> None:
        # Labels are always painted over the photo/scrim — no reliance on
        # QSS-driven card background. Hover only shifts the hotkey chip.
        hotkey_border = _ACCENT if self._hover else "rgba(240, 242, 245, 0.55)"
        hotkey_color = _ACCENT if self._hover else _TEXT_PRIMARY
        self.setStyleSheet(
            f"""
            QLabel#ModeCardTitle {{
                color: {_TEXT_PRIMARY};
                font-size: 34pt;
                font-weight: 700;
                background: transparent;
                border: none;
            }}
            QLabel#ModeCardSubtitle {{
                color: rgba(240, 242, 245, 0.78);
                font-size: 12pt;
                background: transparent;
                border: none;
            }}
            QLabel#ModeCardHotkey {{
                color: {hotkey_color};
                font-size: 12pt;
                font-weight: 700;
                background: rgba(14, 17, 23, 0.55);
                border: 1px solid {hotkey_border};
                border-radius: 6px;
                padding: 2px 8px;
            }}
            """
        )

    # ---------- paint
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        rect = QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0)
        clip = QPainterPath()
        clip.addRoundedRect(rect, _CARD_RADIUS, _CARD_RADIUS)
        painter.setClipPath(clip)

        # 1) Base fill (fallback if the image doesn't cover perfectly).
        painter.fillRect(rect, QColor(_CARD_HOVER if self._hover else _CARD))

        # 2) Photo — scaled to cover the whole card, center-cropped.
        if self._pixmap is not None:
            pw, ph = self._pixmap.width(), self._pixmap.height()
            cw, ch = float(self.width()), float(self.height())
            if pw > 0 and ph > 0:
                scale = max(cw / pw, ch / ph)
                tw, th = pw * scale, ph * scale
                tx = (cw - tw) / 2.0
                ty = (ch - th) / 2.0
                painter.drawPixmap(
                    QRectF(tx, ty, tw, th), self._pixmap,
                    QRectF(0, 0, pw, ph),
                )

        # 3) Bottom-to-top dark gradient so title/subtitle stay readable.
        gradient = QLinearGradient(0.0, 0.0, 0.0, float(self.height()))
        gradient.setColorAt(0.0, QColor(14, 17, 23, 60))     # top: light haze
        gradient.setColorAt(0.55, QColor(14, 17, 23, 150))   # mid
        gradient.setColorAt(1.0, QColor(14, 17, 23, 235))    # bottom: near opaque
        painter.fillRect(rect, gradient)

        # 4) Hover overlay + border.
        if self._hover:
            painter.fillRect(rect, QColor(120, 180, 255, 30))
            border_color = QColor(_ACCENT)
        else:
            border_color = QColor(_BORDER)
        pen = painter.pen()
        pen.setColor(border_color)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(clip)

        painter.end()

    def enterEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self._hover = True
        self._apply_label_style()
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self._hover = False
        self._apply_label_style()
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt API)
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.exercise)
        super().mousePressEvent(event)


class WelcomeScreen(QWidget):
    """Landing page with three exercise-mode cards."""

    mode_selected = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("WelcomeScreen")
        self.setStyleSheet(
            f"QWidget#WelcomeScreen {{ background-color: {_BG}; }}"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(48, 48, 48, 48)
        outer.setSpacing(24)

        title = QLabel("Choose your exercise")
        title.setStyleSheet(
            f"color: {_TEXT_PRIMARY};"
            "font-size: 40pt;"
            "font-weight: 700;"
            "background: transparent;"
            "border: none;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        subtitle = QLabel(
            "Per-exercise specialist models — press the hotkey or click a card. "
            "Press H at any time to return here."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(
            f"color: {_TEXT_SECOND};"
            "font-size: 13pt;"
            "background: transparent;"
            "border: none;"
        )

        cards_row = QHBoxLayout()
        cards_row.setSpacing(24)
        cards_row.setContentsMargins(0, 0, 0, 0)

        self._cards: dict[str, ModeCard] = {}
        for ex in EXERCISES:
            card = ModeCard(ex, self)
            card.clicked.connect(self._on_card_clicked)
            self._cards[ex] = card
            cards_row.addWidget(card, 1)

        outer.addWidget(title, 0)
        outer.addWidget(subtitle, 0)
        outer.addStretch(1)
        outer.addLayout(cards_row, 0)
        outer.addStretch(1)

    def _on_card_clicked(self, exercise: str) -> None:
        self.mode_selected.emit(exercise)

    def trigger_hotkey(self, key: str) -> None:
        """MainWindow calls this when S / L / P is pressed on the welcome page."""
        for ex, hk in _HOTKEYS.items():
            if hk == key:
                self.mode_selected.emit(ex)
                return
