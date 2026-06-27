"""QThread that owns the webcam, MediaPipe, and the ensemble model.

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
from ec3d_dataset import MISTAKE_LABELS, extract_features, feature_dim
from ensemble_eval import build_model

from app.state import (
    CORRECT_CLASSES,
    DISPLAY_CLASSES,
    EXERCISES,
    Prediction,
    indices_for_exercise,
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

DEFAULT_CKPTS = [
    # Fine-tuned Hybrid ensemble — trained on EC3D trainval + ~44 self-recorded
    # MediaPipe clips so the model has seen real MediaPipe noise. Per-seed
    # EC3D-test F1: 0.814 / 0.736 / 0.816 / 0.776.
    "bilstm_ec3d_best_ft_s0.pt",
    "bilstm_ec3d_best_ft_s1.pt",
    "bilstm_ec3d_best_ft_s2.pt",
    "bilstm_ec3d_best_ft_s3.pt",
]

# 3-class exercise-gate ensemble (also fine-tuned with self-data).
DEFAULT_GATE_CKPTS = [
    "gate_hybrid_tv_ft_s0.pt",
    "gate_hybrid_tv_ft_s1.pt",
    "gate_hybrid_tv_ft_s2.pt",
    "gate_hybrid_tv_ft_s3.pt",
]


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _probs_from_full(p_full: np.ndarray) -> np.ndarray:
    """Project 12-class probs down to the 11 we display (drop squat_extra),
    re-normalise so the displayed values still sum to 1.
    """
    keep_idx = [MISTAKE_LABELS.index(c) for c in DISPLAY_CLASSES]
    sub = p_full[keep_idx]
    s = sub.sum()
    return sub / (s if s > 0 else 1.0)


class PipelineThread(QThread):
    """Owns camera + pose model + ensemble.

    Signals
    -------
    frame_ready(QImage, object)
        Raw RGB camera frame as a QImage, and a list of
        (x_norm, y_norm) landmark tuples (one per BlazePose joint) for the
        widget to draw the skeleton. `object` is a Python list, not a Qt type.

    prediction_updated(Prediction)
        Latest model prediction (with display-class probs).

    fps_updated(float)
        FPS measured over the last ~1 second.

    buffer_updated(int)
        Current pose-buffer fill 0..WINDOW.

    coach_should_request(str, float)
        Convenience: fired when we think the coach should ask Ollama
        (incorrect class, confident enough, throttled).
    """

    frame_ready = pyqtSignal(QImage, object)
    prediction_updated = pyqtSignal(object)        # Prediction
    fps_updated = pyqtSignal(float)
    buffer_updated = pyqtSignal(int)
    coach_should_request = pyqtSignal(str, float)

    def __init__(
        self,
        ckpts: list[str] | None = None,
        gate_ckpts: list[str] | None = None,
        camera_index: int = 0,
        coach_interval_s: float = 4.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.ckpts = ckpts or DEFAULT_CKPTS
        self.gate_ckpts = gate_ckpts if gate_ckpts is not None else DEFAULT_GATE_CKPTS
        self.camera_index = camera_index
        self.coach_interval_s = coach_interval_s
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        device = _pick_device()
        n_classes = len(MISTAKE_LABELS)
        models, args_list = [], []
        for name in self.ckpts:
            ckpt = torch.load(CKPT_DIR / name, map_location=device, weights_only=True)
            m = build_model(ckpt["args"], n_classes, device)
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            models.append(m)
            args_list.append(ckpt["args"])
        feature_mode = args_list[0]["feature_mode"]
        _ = feature_dim(feature_mode)  # validate

        # Exercise gate (3-class). Optional — falls back if checkpoints missing.
        gate_models = []
        for name in self.gate_ckpts:
            path = CKPT_DIR / name
            if not path.exists():
                continue
            ckpt = torch.load(path, map_location=device, weights_only=True)
            m = build_model(ckpt["args"], len(EXERCISES), device)
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            gate_models.append(m)
        print(f"[pipeline] gate ensemble: {len(gate_models)} models")

        # Precompute per-exercise index slices for fast routing
        ex_index_slices = {ex: indices_for_exercise(ex) for ex in EXERCISES}

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
        last_probs: np.ndarray | None = None
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
                        full_probs = probs_sum / len(models)

                        # ---- Exercise gate ----
                        if gate_models:
                            gate_sum = None
                            for gm in gate_models:
                                gp = torch.softmax(gm(x), dim=-1).cpu().numpy()[0]
                                gate_sum = gp if gate_sum is None else gate_sum + gp
                            gate_probs = gate_sum / len(gate_models)
                        else:
                            # Fall back: aggregate per-error probs into 3 buckets
                            tmp = _probs_from_full(full_probs)
                            gate_probs = np.array([
                                tmp[ex_index_slices[ex]].sum() for ex in EXERCISES
                            ], dtype=np.float32)

                    display_probs = _probs_from_full(full_probs)
                    if last_probs is None:
                        last_probs = display_probs
                    else:
                        last_probs = 0.6 * last_probs + 0.4 * display_probs

                    # Determine gated exercise + top class within it
                    gated_ex_idx = int(np.argmax(gate_probs))
                    gated_ex = EXERCISES[gated_ex_idx]
                    in_ex = ex_index_slices[gated_ex]
                    sub_probs = last_probs[in_ex]
                    sub_top = int(np.argmax(sub_probs))
                    top_label = DISPLAY_CLASSES[in_ex[sub_top]]

                    # Uncertainty: gate not confident, OR person not moving
                    gate_max = float(gate_probs.max())
                    # Pose movement = variance of joint positions across the buffer
                    # summed over joints + axes. Static pose -> very low value.
                    seq_var = float(np.mean(np.var(seq, axis=0)))
                    is_uncertain = (gate_max < 0.55) or (seq_var < 0.0008)

                    last_label = top_label
                    pred = Prediction(
                        label=last_label,
                        confidence=float(sub_probs[sub_top]),
                        probs=last_probs.copy(),
                        is_correct=last_label in CORRECT_CLASSES,
                        gated_exercise=gated_ex,
                        gate_probs=gate_probs.astype(np.float32),
                        is_uncertain=is_uncertain,
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
