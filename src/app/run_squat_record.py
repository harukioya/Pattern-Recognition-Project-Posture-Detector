"""Squat-only app with manual video recording."""
from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

# Allow `python src/app/run_squat_record.py` and `python -m app.run_squat_record`.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.main_window import MainWindow  # noqa: E402


def _place_on_primary_screen(w: MainWindow, app: QApplication) -> None:
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
    app.setApplicationName("Squat Recorder")
    w = MainWindow(
        window_title="Squat Recorder",
        forced_exercise="SQUAT",
        recording_enabled=True,
    )
    _place_on_primary_screen(w, app)
    w.show()
    w.raise_()
    w.activateWindow()
    QTimer.singleShot(250, lambda: w.start_mode("SQUAT"))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
