"""Mode-picker app with manual video recording enabled."""
from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

# Allow `python src/app/run_record.py` and `python -m app.run_record`.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.main_window import MainWindow  # noqa: E402
from app.run import _place_on_primary_screen, _prewarm_camera  # noqa: E402


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Posture Recorder")

    if not _prewarm_camera():
        print(
            "\n[run_record] Camera not accessible. The UI will still open, but "
            "recording requires a working camera.\n",
            file=sys.stderr,
        )

    w = MainWindow(window_title="Posture Recorder", recording_enabled=True)
    _place_on_primary_screen(w, app)
    w.show()
    w.raise_()
    w.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()