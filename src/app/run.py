"""Entry point — `python -m app.run` from src/."""
from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

# Allow `python src/app/run.py` and `python -m app.run` both
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.main_window import MainWindow  # noqa: E402


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Posture Coach")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
