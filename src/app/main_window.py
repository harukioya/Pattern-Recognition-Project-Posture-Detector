"""MainWindow - the Posture Coach desktop app.

On the experiment/per-exercise-modes branch the main window swaps between
a WelcomeScreen (mode picker) and the workout view via QStackedWidget.
The PipelineThread is created lazily on mode selection so we only pay
model-loading cost when the user actually starts a session.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.coach_thread import CoachThread
from app.pipeline_thread import PipelineThread
from app.state import EXERCISES
from app.widgets.bars_panel import BarsPanel
from app.widgets.camera_widget import CameraWidget
from app.widgets.coach_panel import CoachPanel
from app.widgets.header_bar import HeaderBar
from app.widgets.status_bar import AppStatusBar
from app.widgets.verdict_panel import VerdictPanel
from app.widgets.welcome_screen import WelcomeScreen

APP_DIR = Path(__file__).resolve().parent
STYLE_PATH = APP_DIR / "style.qss"


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        window_title: str = "Posture Coach",
        forced_exercise: str | None = None,
        recording_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.setObjectName("MainWindow")
        self.setWindowTitle(window_title)
        self.recording_enabled = recording_enabled
        self.forced_exercise = forced_exercise
        self.resize(QSize(1400, 900))

        self.header = HeaderBar(self, recording_enabled=recording_enabled)
        self.status = AppStatusBar(self)
        if recording_enabled:
            self.header.record_toggled.connect(self._toggle_recording)

        # Workout page - camera on the left, panels on the right.
        workout = QWidget(self)
        workout.setObjectName("WorkoutPage")
        wh = QHBoxLayout(workout)
        wh.setContentsMargins(16, 16, 16, 12)
        wh.setSpacing(16)

        self.camera = CameraWidget(self)
        self.camera.setMinimumSize(720, 540)
        wh.addWidget(self.camera, stretch=7)

        right_col = QVBoxLayout()
        right_col.setSpacing(12)
        self.verdict = VerdictPanel(self)
        self.bars = BarsPanel(self)
        self.coach = CoachPanel(self)
        right_col.addWidget(self.verdict, stretch=2)
        right_col.addWidget(self.bars, stretch=5)
        right_col.addWidget(self.coach, stretch=2)
        wh.addLayout(right_col, stretch=3)

        # Landing page - three exercise cards.
        self.welcome = WelcomeScreen(self)
        self.welcome.mode_selected.connect(self._on_mode_selected)

        self.stack = QStackedWidget(self)
        self.stack.addWidget(self.welcome)   # index 0
        self.stack.addWidget(workout)        # index 1

        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self.header, stretch=0)
        outer.addWidget(self.stack, stretch=1)
        outer.addWidget(self.status, stretch=0)

        wrapper = QWidget(self)
        wrapper.setObjectName("MainWindow")
        wrapper.setLayout(outer)
        self.setCentralWidget(wrapper)

        # Coach thread runs the whole time; pipeline is created per mode.
        self.coach_thread = CoachThread(parent=self)
        self.coach_thread.cue_ready.connect(self.coach.set_cue)
        self.coach_thread.start()

        self.pipeline: PipelineThread | None = None

        # Stylesheet
        if STYLE_PATH.exists():
            self.setStyleSheet(STYLE_PATH.read_text(encoding="utf-8"))

        self._go_home()

    # ------------------------------------------------------------------ modes
    def start_mode(self, exercise: str) -> None:
        self._on_mode_selected(exercise)

    def _on_mode_selected(self, exercise: str) -> None:
        if exercise not in EXERCISES:
            return
        self._teardown_pipeline()
        self.header.set_mode(exercise)
        self.bars.set_mode(exercise)
        self.verdict.reset_for_mode(exercise)
        self.coach.reset()
        self.status.set_buffer(0)

        self.pipeline = PipelineThread(mode=exercise, parent=self)
        self.pipeline.frame_ready.connect(self.camera.set_frame)
        self.pipeline.prediction_updated.connect(self.header.set_prediction)
        self.pipeline.prediction_updated.connect(self.verdict.set_prediction)
        self.pipeline.prediction_updated.connect(self.bars.set_prediction)
        self.pipeline.prediction_updated.connect(self.coach.set_prediction)
        self.pipeline.fps_updated.connect(self.status.set_fps)
        self.pipeline.buffer_updated.connect(self.status.set_buffer)
        self.pipeline.coach_should_request.connect(self.coach_thread.request_cue)
        if self.recording_enabled:
            self.pipeline.recording_state_changed.connect(self.header.set_recording)
        self.pipeline.start()

        self.stack.setCurrentIndex(1)

    def _teardown_pipeline(self) -> None:
        if self.pipeline is None:
            return
        self.pipeline.stop()
        self.pipeline.wait(2000)
        self.pipeline = None
        if self.recording_enabled:
            self.header.set_recording(False, "")

    def _go_home(self) -> None:
        self._teardown_pipeline()
        self.header.set_mode(None)
        self.bars.set_mode(None)
        self.verdict.reset_for_mode(None)
        self.coach.reset()
        self.stack.setCurrentIndex(0)

    def _toggle_recording(self) -> None:
        if self.pipeline is not None:
            self.pipeline.toggle_recording()

    # ------------------------------------------------------------------ keys
    def keyPressEvent(self, ev) -> None:  # noqa: N802
        key = ev.key()
        if key == Qt.Key.Key_Q:
            self.close()
            return
        if self.stack.currentIndex() == 0:
            # Welcome hotkeys - S / L / P jump straight to a mode.
            if key == Qt.Key.Key_S:
                self._on_mode_selected("SQUAT")
                return
            if key == Qt.Key.Key_L:
                self._on_mode_selected("Lunges")
                return
            if key == Qt.Key.Key_P:
                self._on_mode_selected("Plank")
                return
        else:
            if self.recording_enabled and key == Qt.Key.Key_R:
                self._toggle_recording()
                return
            # Workout hotkeys - H / Home / Escape returns to welcome.
            if key in (Qt.Key.Key_H, Qt.Key.Key_Home, Qt.Key.Key_Escape):
                self._go_home()
                return
        super().keyPressEvent(ev)

    def closeEvent(self, ev) -> None:  # noqa: N802
        self._teardown_pipeline()
        self.coach_thread.stop()
        self.coach_thread.wait(2000)
        super().closeEvent(ev)