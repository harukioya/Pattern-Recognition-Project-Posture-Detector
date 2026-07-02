"""Entry point - `python -m app.run` from src/."""
from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

# Allow `python src/app/run.py` and `python -m app.run` both
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.main_window import MainWindow  # noqa: E402


def _prewarm_camera(camera_index: int = 0) -> bool:
    """Open the camera once on the main thread so macOS AVFoundation can
    display its permission prompt. VideoCapture from a QThread silently
    fails when permission hasn't been granted yet.
    """
    import cv2
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        return False
    # Read one frame so the OS actually completes the permission grant.
    ok, _ = cap.read()
    cap.release()
    return bool(ok)


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
    app.setApplicationName("Posture Coach")

    if not _prewarm_camera():
        print(
            "\n[run] Camera not accessible. On macOS this usually means the "
            "terminal running this script has not been granted camera "
            "permission.\n"
            "  1. Open System Settings -> Privacy & Security -> Camera.\n"
            "  2. Enable your terminal (Terminal / iTerm2 / VS Code / ...).\n"
            "  3. Fully quit and re-open the terminal, then re-run.\n"
            "Continuing to launch the UI anyway - mode selection will work "
            "but the camera feed will stay blank.\n",
            file=sys.stderr,
        )

    w = MainWindow()
    _place_on_primary_screen(w, app)
    w.show()
    w.raise_()
    w.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()