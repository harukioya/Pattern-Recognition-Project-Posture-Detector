"""Headless MediaPipe Tasks API check: runs PoseLandmarker on a synthetic frame.

Verifies install + model file before any webcam access is needed.
For the real FPS measurement run smoke_mediapipe.py manually.
"""
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"


def main() -> None:
    print(f"mediapipe : {mp.__version__}")
    print(f"opencv    : {cv2.__version__}")
    print(f"numpy     : {np.__version__}")
    print(f"model     : {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")

    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    n_warm, n_bench = 3, 30
    for _ in range(n_warm):
        landmarker.detect(mp_image)

    t0 = time.time()
    for _ in range(n_bench):
        result = landmarker.detect(mp_image)
    dt = time.time() - t0

    n_landmarks = len(result.pose_landmarks[0]) if result.pose_landmarks else 0
    print(f"forward x{n_bench}: {dt:.2f} s  ({n_bench / dt:.1f} synthetic-fps)")
    print(f"landmarks : {n_landmarks} (0 expected on random noise)")
    print("install OK")
    landmarker.close()


if __name__ == "__main__":
    main()
