"""QThread that owns the webcam, MediaPipe, and a per-exercise ensemble.

On the experiment/per-exercise-modes branch the exercise-gate is skipped
entirely: the thread is constructed with a fixed `mode` (SQUAT / Lunges /
Plank) and only loads that exercise's specialist ensemble.

Emits Qt signals so widgets in the GUI thread can render without ever
blocking on inference or video I/O.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

# Reach back into the existing project code
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from blazepose_to_body25 import blazepose_to_body25, normalise_like_ec3d
from ec3d_dataset import (
    extract_features,
    feature_dim,
    local_labels_for_exercise,
)
from ensemble_eval import build_model

from app.state import (
    CORRECT_CLASSES,
    DISPLAY_CLASSES,
    EXERCISES,
    Prediction,
)

ROOT = Path(__file__).resolve().parent.parent.parent
# Heavy MediaPipe model: more accurate joints (esp. feet) -> better stance
# discrimination at inference. Falls back to lite if heavy isn't present.
_HEAVY = ROOT / "models" / "pose_landmarker_heavy.task"
_LITE = ROOT / "models" / "pose_landmarker_lite.task"
MODEL_FILE = _HEAVY if _HEAVY.exists() else _LITE
CKPT_DIR = ROOT / "checkpoints"

WINDOW = 64
PRED_EVERY = 4

# Per-exercise ensembles. 4 seeds each, trained via train.py with
# --exercise-filter <ex> --ckpt-tag pex_<ex>_s<seed>.
PEX_CKPTS: dict[str, list[str]] = {
    "SQUAT": [f"bilstm_ec3d_best_pex_squat_s{s}.pt" for s in range(4)],
    "Lunges": [f"bilstm_ec3d_best_pex_lunges_s{s}.pt" for s in range(4)],
    "Plank": [f"bilstm_ec3d_best_pex_plank_s{s}.pt" for s in range(4)],
}


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _local_probs_to_display(
    local_probs: np.ndarray, exercise: str
) -> np.ndarray:
    """Scatter a (k,) local-probability vector into a (11,) DISPLAY_CLASSES
    vector, leaving other-exercise slots at zero."""
    out = np.zeros(len(DISPLAY_CLASSES), dtype=np.float32)
    for local_id, label in enumerate(local_labels_for_exercise(exercise)):
        if label in DISPLAY_CLASSES:
            out[DISPLAY_CLASSES.index(label)] = float(local_probs[local_id])
    return out


class PipelineThread(QThread):
    """Owns camera + pose model + per-exercise ensemble.

    Signals
    -------
    frame_ready(QImage, object)
        Raw RGB camera frame as a QImage, and a list of (x_norm, y_norm)
        landmark tuples (one per BlazePose joint) for the widget to draw
        the skeleton. `object` is a Python list, not a Qt type.

    prediction_updated(Prediction)
        Latest model prediction, scattered into the 11-class display space.

    fps_updated(float)
        FPS measured over the last ~1 second.

    buffer_updated(int)
        Current pose-buffer fill 0..WINDOW.

    coach_should_request(str, float)
        Fired when we think the coach should ask Ollama (incorrect class,
        confident enough, throttled).
    """

    frame_ready = pyqtSignal(QImage, object)
    prediction_updated = pyqtSignal(object)        # Prediction
    fps_updated = pyqtSignal(float)
    buffer_updated = pyqtSignal(int)
    coach_should_request = pyqtSignal(str, float)

    def __init__(
        self,
        mode: str,
        camera_index: int = 0,
        coach_interval_s: float = 4.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if mode not in EXERCISES:
            raise ValueError(f"unknown mode: {mode}; expected one of {EXERCISES}")
        self.mode = mode
        self.camera_index = camera_index
        self.coach_interval_s = coach_interval_s
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        device = _pick_device()
        local_labels = local_labels_for_exercise(self.mode)
        n_local = len(local_labels)

        models, args_list = [], []
        for name in PEX_CKPTS[self.mode]:
            path = CKPT_DIR / name
            if not path.exists():
                print(f"[pipeline] WARN: missing checkpoint {name}, skipping")
                continue
            ckpt = torch.load(path, map_location=device, weights_only=True)
            m = build_model(ckpt["args"], n_local, device)
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            models.append(m)
            args_list.append(ckpt["args"])

        if not models:
            print(
                f"[pipeline] ERROR: no {self.mode} checkpoints found in {CKPT_DIR}. "
                "Train them via `python src/train.py --exercise-filter "
                f"{self.mode} --ckpt-tag pex_{self.mode.lower()}_sN ...`."
            )
            return
        feature_mode = args_list[0]["feature_mode"]
        _ = feature_dim(feature_mode)  # validate
        print(f"[pipeline] mode={self.mode}  ensemble={len(models)} models  classes={local_labels}")

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"[pipeline] could not open camera {self.camera_index}")
            return

        base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_FILE))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        landmarker = vision.PoseLandmarker.create_from_options(options)

        pose_buffer: deque = deque(maxlen=WINDOW)
        start_time = time.time()
        fps_t0 = time.time()
        fps_frames = 0
        last_probs: np.ndarray | None = None  # smoothed local probs
        last_label = ""
        last_coach_t = 0.0
        frame_count = 0

        while self._running:
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            frame_bgr = cv2.flip(frame_bgr, 1)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            landmarks_xy: list[tuple[float, float]] = []
            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                world = result.pose_world_landmarks[0]
                landmarks_xy = [(lm.x, lm.y) for lm in lms]

                world_arr = np.array(
                    [[lm.x, -lm.y, lm.z] for lm in world], dtype=np.float32
                )
                pose_buffer.append(blazepose_to_body25(world_arr))

                if (len(pose_buffer) == WINDOW
                        and frame_count % PRED_EVERY == 0):
                    seq = np.stack(pose_buffer, axis=0)
                    seq = normalise_like_ec3d(seq)
                    feats = extract_features(seq, feature_mode)
                    x = torch.from_numpy(feats).unsqueeze(0).to(device)
                    with torch.no_grad():
                        probs_sum = None
                        for m in models:
                            p = torch.softmax(m(x), dim=-1).cpu().numpy()[0]
                            probs_sum = p if probs_sum is None else probs_sum + p
                        local_probs = probs_sum / len(models)

                    if last_probs is None:
                        last_probs = local_probs
                    else:
                        last_probs = 0.6 * last_probs + 0.4 * local_probs

                    top_idx = int(np.argmax(last_probs))
                    top_label = local_labels[top_idx]

                    # Person-not-moving heuristic — same as gate-mode pipeline.
                    seq_var = float(np.mean(np.var(seq, axis=0)))
                    is_uncertain = seq_var < 0.0008

                    display_probs = _local_probs_to_display(last_probs, self.mode)
                    last_label = top_label
                    pred = Prediction(
                        label=last_label,
                        confidence=float(last_probs[top_idx]),
                        probs=display_probs,
                        is_correct=last_label in CORRECT_CLASSES,
                        gated_exercise=self.mode,
                        gate_probs=np.zeros(len(EXERCISES), dtype=np.float32),
                        is_uncertain=is_uncertain,
                        selected_mode=self.mode,
                    )
                    self.prediction_updated.emit(pred)

                    # Throttled coaching request (skip when uncertain)
                    now = time.time()
                    if (not is_uncertain
                            and last_label not in CORRECT_CLASSES
                            and pred.confidence > 0.35
                            and (now - last_coach_t) > self.coach_interval_s):
                        self.coach_should_request.emit(last_label, pred.confidence)
                        last_coach_t = now

            # Emit raw frame as QImage + landmarks for widget rendering.
            qimg = QImage(
                rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888
            ).copy()
            self.frame_ready.emit(qimg, landmarks_xy)
            self.buffer_updated.emit(len(pose_buffer))

            frame_count += 1
            fps_frames += 1
            if fps_frames >= 30:
                now = time.time()
                self.fps_updated.emit(fps_frames / (now - fps_t0))
                fps_t0 = now
                fps_frames = 0

        cap.release()
        landmarker.close()
