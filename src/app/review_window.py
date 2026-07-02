"""Video review window that can send the paused frame to SAM3D."""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RECORDINGS_DIR = ROOT / "recordings"
SAM3D_HELPER_SCRIPT = ROOT / "tools" / "sam3d" / "demo_export_mesh.py"
DEFAULT_SAM3D_HOST_ROOT = Path(os.environ.get("SAM3D_HOST_ROOT", r"E:\DockerE\SAM3D"))
SAM3D_CONTAINER = os.environ.get("SAM3D_CONTAINER", "sam3d_body")
SAM3D_CONTAINER_ROOT = os.environ.get("SAM3D_CONTAINER_ROOT", "/workspace/SAM3D")

VIDEO_FILTER = "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*.*)"


@dataclass(frozen=True)
class Sam3DResult:
    job_id: str
    request_dir: Path
    output_dir: Path
    frame_path: Path
    render_path: Path | None
    mesh_path: Path | None
    stdout: str
    stderr: str


class ImageView(QLabel):
    def __init__(self, placeholder: str, parent: QWidget | None = None) -> None:
        super().__init__(placeholder, parent)
        self._image: QImage | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "background-color: #0e1117; color: #9aa1b3; "
            "border: 1px solid #1f2632; border-radius: 8px;"
        )

    def set_image(self, image: QImage | None) -> None:
        self._image = image.copy() if image is not None and not image.isNull() else None
        self._refresh_pixmap()

    def set_image_path(self, path: Path | None) -> None:
        if path is None or not path.exists():
            self.set_image(None)
            return
        self.set_image(QImage(str(path)))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._image is None or self._image.isNull():
            self.setPixmap(QPixmap())
            if not self.text():
                self.setText("No frame")
            return
        self.setText("")
        pixmap = QPixmap.fromImage(self._image)
        self.setPixmap(
            pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class Sam3DWorker(QThread):
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        frame_bgr,
        *,
        frame_index: int,
        timestamp_s: float,
        host_root: Path = DEFAULT_SAM3D_HOST_ROOT,
        container_name: str = SAM3D_CONTAINER,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.frame_bgr = frame_bgr.copy()
        self.frame_index = int(frame_index)
        self.timestamp_s = float(timestamp_s)
        self.host_root = host_root
        self.container_name = container_name

    def run(self) -> None:
        try:
            result = self._run_job()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(result)

    def _run_job(self) -> Sam3DResult:
        docker = shutil.which("docker")
        if docker is None:
            raise RuntimeError("Docker was not found on PATH.")

        repo_dir = self.host_root / "sam-3d-body"
        checkpoint = repo_dir / "checkpoints" / "sam-3d-body-dinov3" / "model.ckpt"
        mhr = repo_dir / "checkpoints" / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt"
        script = repo_dir / "demo_export_mesh.py"
        for path in (self.host_root, repo_dir, checkpoint, mhr, SAM3D_HELPER_SCRIPT):
            if not path.exists():
                raise RuntimeError(f"Missing SAM3D path: {path}")
        shutil.copy2(SAM3D_HELPER_SCRIPT, script)

        job_id = time.strftime("job_%Y%m%d_%H%M%S") + f"_f{self.frame_index}"
        request_dir = self.host_root / "requests" / job_id
        output_dir = self.host_root / "outputs" / job_id
        request_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        frame_path = request_dir / "frame.png"
        if not cv2.imwrite(str(frame_path), self.frame_bgr):
            raise RuntimeError(f"Could not write frame image: {frame_path}")

        container_request = f"{SAM3D_CONTAINER_ROOT}/requests/{job_id}"
        container_output = f"{SAM3D_CONTAINER_ROOT}/outputs/{job_id}"
        command = (
            "cd /workspace/SAM3D/sam-3d-body && "
            "python demo_export_mesh.py "
            f"--image_folder {container_request} "
            f"--output_folder {container_output} "
            "--checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt "
            "--mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt "
            "--export_viewer_obj"
        )
        proc = subprocess.run(
            [docker, "exec", self.container_name, "bash", "-lc", command],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=900,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "SAM3D Docker command failed.\n"
                f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
            )

        renders = sorted(output_dir.glob("*.png"))
        viewer_meshes = sorted(output_dir.glob("*_viewer.obj"))
        raw_meshes = sorted(
            path for path in output_dir.glob("*.obj") if not path.name.endswith("_viewer.obj")
        )
        meshes = viewer_meshes or raw_meshes
        return Sam3DResult(
            job_id=job_id,
            request_dir=request_dir,
            output_dir=output_dir,
            frame_path=frame_path,
            render_path=renders[0] if renders else None,
            mesh_path=meshes[0] if meshes else None,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )


class SquatReviewWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Squat Review")
        self.resize(1400, 900)

        self.cap: cv2.VideoCapture | None = None
        self.video_path: Path | None = None
        self.frame_count = 0
        self.fps = 30.0
        self.current_index = 0
        self.current_frame_bgr = None
        self.sam3d_worker: Sam3DWorker | None = None
        self.last_result: Sam3DResult | None = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance_frame)

        self._build_ui()
        self._apply_style()
        self._set_video_loaded(False)

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 16, 16, 12)
        outer.setSpacing(12)

        header = QHBoxLayout()
        self.title = QLabel("Squat Review")
        self.title.setObjectName("Title")
        self.open_button = QPushButton("Open Video")
        self.open_button.clicked.connect(self.open_video)
        header.addWidget(self.title, 1)
        header.addWidget(self.open_button, 0)
        outer.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(16)
        outer.addLayout(content, 1)

        left = QVBoxLayout()
        left.setSpacing(10)
        self.video_view = ImageView("Open a squat video")
        left.addWidget(self.video_view, 1)

        controls = QGridLayout()
        controls.setHorizontalSpacing(10)
        controls.setVerticalSpacing(8)
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        self.generate_button = QPushButton("Generate 3D")
        self.generate_button.clicked.connect(self.generate_3d)
        self.time_label = QLabel("00:00.00 / 00:00.00")
        self.frame_label = QLabel("frame 0 / 0")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(self.seek_frame)

        controls.addWidget(self.play_button, 0, 0)
        controls.addWidget(self.generate_button, 0, 1)
        controls.addWidget(self.time_label, 0, 2)
        controls.addWidget(self.frame_label, 0, 3)
        controls.addWidget(self.slider, 1, 0, 1, 4)
        left.addLayout(controls)
        content.addLayout(left, 7)

        right = QVBoxLayout()
        right.setSpacing(10)
        self.result_title = QLabel("SAM3D Result")
        self.result_title.setObjectName("PanelTitle")
        self.result_view = ImageView("No 3D result yet")
        self.result_view.setMinimumSize(360, 300)
        self.status = QLabel("Open a video, pause at a frame, then generate 3D.")
        self.status.setWordWrap(True)
        self.open_folder_button = QPushButton("Open Output Folder")
        self.open_folder_button.clicked.connect(self.open_output_folder)
        self.open_mesh_button = QPushButton("Open OBJ")
        self.open_mesh_button.clicked.connect(self.open_mesh)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(180)

        right.addWidget(self.result_title)
        right.addWidget(self.result_view, 1)
        right.addWidget(self.status)
        right.addWidget(self.open_folder_button)
        right.addWidget(self.open_mesh_button)
        right.addWidget(self.log)
        content.addLayout(right, 3)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background-color: #0e1117; color: #f0f2f5; }
            QLabel#Title { font-size: 26pt; font-weight: 700; }
            QLabel#PanelTitle { color: #9aa1b3; font-size: 12pt; font-weight: 700; }
            QPushButton {
                background-color: #161b22;
                color: #f0f2f5;
                border: 1px solid #1f2632;
                border-radius: 6px;
                padding: 8px 14px;
                font-weight: 700;
            }
            QPushButton:hover { border-color: #78b4ff; }
            QPushButton:disabled { color: #5d6478; border-color: #1f2632; }
            QSlider::groove:horizontal { height: 6px; background: #1f2632; border-radius: 3px; }
            QSlider::handle:horizontal { width: 16px; background: #78b4ff; margin: -5px 0; border-radius: 8px; }
            QTextEdit { background-color: #0b0e13; border: 1px solid #1f2632; border-radius: 6px; }
            """
        )

    def _set_video_loaded(self, loaded: bool) -> None:
        self.play_button.setEnabled(loaded)
        self.generate_button.setEnabled(loaded)
        self.slider.setEnabled(loaded)
        self.open_folder_button.setEnabled(False)
        self.open_mesh_button.setEnabled(False)

    def open_video(self) -> None:
        start_dir = DEFAULT_RECORDINGS_DIR if DEFAULT_RECORDINGS_DIR.exists() else ROOT
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open squat video",
            str(start_dir),
            VIDEO_FILTER,
        )
        if filename:
            self.load_video(Path(filename))

    def load_video(self, path: Path) -> None:
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self.status.setText(f"Could not open video: {path}")
            self._set_video_loaded(False)
            return

        self.cap = cap
        self.video_path = path
        self.frame_count = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if self.fps <= 1.0 or self.fps > 240.0:
            self.fps = 30.0
        self.slider.setRange(0, self.frame_count - 1)
        self.current_index = 0
        self.title.setText(path.name)
        self.result_view.set_image(None)
        self.result_view.setText("No 3D result yet")
        self.last_result = None
        self.log.clear()
        self._set_video_loaded(True)
        self.show_frame(0)
        self.status.setText("Pause or scrub to the frame you want, then generate 3D.")

    def toggle_playback(self) -> None:
        if self.cap is None:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            interval_ms = max(1, int(round(1000.0 / self.fps)))
            self.timer.start(interval_ms)
            self.play_button.setText("Pause")

    def _advance_frame(self) -> None:
        if self.current_index >= self.frame_count - 1:
            self.timer.stop()
            self.play_button.setText("Play")
            return
        self.show_frame(self.current_index + 1)

    def seek_frame(self, index: int) -> None:
        self.timer.stop()
        self.play_button.setText("Play")
        self.show_frame(index)

    def show_frame(self, index: int) -> None:
        if self.cap is None:
            return
        index = max(0, min(int(index), self.frame_count - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame_bgr = self.cap.read()
        if not ok:
            self.status.setText(f"Could not read frame {index}.")
            return
        self.current_index = index
        self.current_frame_bgr = frame_bgr
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.video_view.set_image(qimg)
        self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        self._update_time_labels()

    def _update_time_labels(self) -> None:
        current = self.current_index / self.fps
        total = max(0, self.frame_count - 1) / self.fps
        self.time_label.setText(f"{self._format_time(current)} / {self._format_time(total)}")
        self.frame_label.setText(f"frame {self.current_index + 1} / {self.frame_count}")

    @staticmethod
    def _format_time(seconds: float) -> str:
        minutes = int(seconds // 60)
        rest = seconds - minutes * 60
        return f"{minutes:02d}:{rest:05.2f}"

    def generate_3d(self) -> None:
        if self.current_frame_bgr is None:
            self.status.setText("No frame is loaded.")
            return
        if self.sam3d_worker is not None and self.sam3d_worker.isRunning():
            self.status.setText("SAM3D is already running.")
            return
        self.timer.stop()
        self.play_button.setText("Play")
        self.generate_button.setEnabled(False)
        self.open_folder_button.setEnabled(False)
        self.open_mesh_button.setEnabled(False)
        timestamp_s = self.current_index / self.fps
        self.status.setText(
            f"Generating 3D from frame {self.current_index + 1} at {self._format_time(timestamp_s)}..."
        )
        self.log.append("Starting SAM3D Docker job. This can take a while.")
        self.sam3d_worker = Sam3DWorker(
            self.current_frame_bgr,
            frame_index=self.current_index,
            timestamp_s=timestamp_s,
            parent=self,
        )
        self.sam3d_worker.finished_ok.connect(self._on_sam3d_done)
        self.sam3d_worker.failed.connect(self._on_sam3d_failed)
        self.sam3d_worker.start()

    def _on_sam3d_done(self, result: Sam3DResult) -> None:
        self.last_result = result
        self.generate_button.setEnabled(True)
        self.open_folder_button.setEnabled(True)
        self.open_mesh_button.setEnabled(result.mesh_path is not None)
        self.result_view.set_image_path(result.render_path)
        mesh_text = result.mesh_path.name if result.mesh_path else "no OBJ found"
        self.status.setText(f"SAM3D done: {result.job_id} ({mesh_text})")
        self.log.append(result.stdout.strip() or "SAM3D completed with no stdout.")
        if result.stderr.strip():
            self.log.append("stderr:\n" + result.stderr.strip())

    def _on_sam3d_failed(self, message: str) -> None:
        self.generate_button.setEnabled(True)
        self.status.setText("SAM3D failed. Check the log panel.")
        self.log.append(message)

    def open_output_folder(self) -> None:
        if self.last_result is None:
            return
        os.startfile(self.last_result.output_dir)  # noqa: S606 - local user action

    def open_mesh(self) -> None:
        if self.last_result is None or self.last_result.mesh_path is None:
            return
        os.startfile(self.last_result.mesh_path)  # noqa: S606 - local user action

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.sam3d_worker is not None and self.sam3d_worker.isRunning():
            self.status.setText("SAM3D is still running; wait for it to finish before closing.")
            event.ignore()
            return
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        super().closeEvent(event)
