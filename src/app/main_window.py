"""MainWindow — the Posture Coach desktop app.

Lays out the camera on the left and a stacked column of panels on the right.
Owns the pipeline + coach threads.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from app.coach_thread import CoachThread
from app.pipeline_thread import PipelineThread
from app.widgets.bars_panel import BarsPanel
from app.widgets.camera_widget import CameraWidget
from app.widgets.coach_panel import CoachPanel
from app.widgets.header_bar import HeaderBar
from app.widgets.status_bar import AppStatusBar
from app.widgets.verdict_panel import VerdictPanel

APP_DIR = Path(__file__).resolve().parent
STYLE_PATH = APP_DIR / "style.qss"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("MainWindow")
        self.setWindowTitle("Posture Coach")
        self.resize(QSize(1400, 900))

        # Central layout: camera | panels
        central = QWidget(self)
        central.setObjectName("MainWindow")
        h = QHBoxLayout(central)
        h.setContentsMargins(16, 16, 16, 12)
        h.setSpacing(16)

        self.camera = CameraWidget(self)
        self.camera.setMinimumSize(720, 540)
        h.addWidget(self.camera, stretch=7)

        right_col = QVBoxLayout()
        right_col.setSpacing(12)
        self.verdict = VerdictPanel(self)
        self.bars = BarsPanel(self)
        self.coach = CoachPanel(self)
        right_col.addWidget(self.verdict, stretch=2)
        right_col.addWidget(self.bars, stretch=5)
        right_col.addWidget(self.coach, stretch=2)
        h.addLayout(right_col, stretch=3)

        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.header = HeaderBar(self)
        outer.addWidget(self.header, stretch=0)
        outer.addWidget(central, stretch=1)
        self.status = AppStatusBar(self)
        outer.addWidget(self.status, stretch=0)

        wrapper = QWidget(self)
        wrapper.setObjectName("MainWindow")
        wrapper.setLayout(outer)
        self.setCentralWidget(wrapper)

        # Threads
        self.pipeline = PipelineThread(parent=self)
        self.coach_thread = CoachThread(parent=self)

        # Wire signals
        self.pipeline.frame_ready.connect(self.camera.set_frame)
        self.pipeline.prediction_updated.connect(self.header.set_prediction)
        self.pipeline.prediction_updated.connect(self.verdict.set_prediction)
        self.pipeline.prediction_updated.connect(self.bars.set_prediction)
        self.pipeline.prediction_updated.connect(self.coach.set_prediction)
        self.pipeline.fps_updated.connect(self.status.set_fps)
        self.pipeline.buffer_updated.connect(self.status.set_buffer)
        self.pipeline.coach_should_request.connect(self.coach_thread.request_cue)
        self.coach_thread.cue_ready.connect(self.coach.set_cue)

        # Stylesheet
        if STYLE_PATH.exists():
            self.setStyleSheet(STYLE_PATH.read_text())

        # Start workers
        self.coach_thread.start()
        self.pipeline.start()

    def keyPressEvent(self, ev) -> None:  # noqa: N802
        if ev.key() == Qt.Key.Key_Q:
            self.close()
        super().keyPressEvent(ev)

    def closeEvent(self, ev) -> None:  # noqa: N802
        self.pipeline.stop()
        self.coach_thread.stop()
        self.pipeline.wait(2000)
        self.coach_thread.wait(2000)
        super().closeEvent(ev)
