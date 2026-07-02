"""Entry point for the squat video review and SAM3D frame generator."""
from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.review_window import SquatReviewWindow  # noqa: E402


def _place_on_primary_screen(w: SquatReviewWindow, app: QApplication) -> None:
    screen = app.primaryScreen()
    if screen is None:
        return
    available = screen.availableGeometry()
    width = min(1400, int(available.width() * 0.95))
    height = min(900, int(available.height() * 0.95))
    left = available.x() + (available.width() - width) // 2
    top = available.y() + (available.height() - height) // 2
    w.setGeometry(left, top, width, height)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Squat Review")
    w = SquatReviewWindow()
    _place_on_primary_screen(w, app)
    w.show()
    w.raise_()
    w.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
