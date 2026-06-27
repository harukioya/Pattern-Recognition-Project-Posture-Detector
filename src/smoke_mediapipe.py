"""Webcam smoke test using MediaPipe Tasks PoseLandmarker.

Press 'q' to quit. Confirms (a) webcam access, (b) real-time FPS on this
laptop, (c) 33 landmarks per detected pose.
"""
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"


def main() -> None:
    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    frame_count = 0
    t0 = time.time()
    last_report = t0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - t0) * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        n_landmarks = 0
        if result.pose_landmarks:
            lms = result.pose_landmarks[0]
            n_landmarks = len(lms)
            h, w = frame.shape[:2]
            for lm in lms:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)

        frame_count += 1
        now = time.time()
        if now - last_report >= 1.0:
            fps = frame_count / (now - t0)
            print(f"fps={fps:5.1f}  landmarks={n_landmarks}")
            last_report = now

        cv2.imshow("smoke_mediapipe (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
