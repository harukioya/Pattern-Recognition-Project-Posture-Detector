"""Convert a labeled workout video into self-recorded training data.

Usage:
    python src/import_video_data.py path/to/video.mp4 --label SQUAT/correct
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from blazepose_to_body25 import blazepose_to_body25
from ec3d_dataset import MISTAKE_LABELS


ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = (
    ROOT / "models" / "pose_landmarker_heavy.task"
    if (ROOT / "models" / "pose_landmarker_heavy.task").exists()
    else ROOT / "models" / "pose_landmarker_lite.task"
)
SAVE_DIR = ROOT / "data" / "self_recorded"


def _default_output(video: Path, label: str) -> Path:
    safe_label = label.replace("/", "_")
    return SAVE_DIR / f"{video.stem}__{safe_label}.pkl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--label", required=True, choices=MISTAKE_LABELS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--every", type=int, default=1, help="process every Nth video frame")
    args = parser.parse_args()

    if args.every < 1:
        raise SystemExit("--every must be >= 1")
    if not MODEL_FILE.exists():
        raise SystemExit(f"missing MediaPipe pose model: {MODEL_FILE}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {args.video}")

    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_FILE))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    frames_body25: list[np.ndarray] = []
    seen = 0
    processed = 0
    last_ts = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        seen += 1
        if (seen - 1) % args.every:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if ts_ms <= last_ts:
            ts_ms = last_ts + 1
        last_ts = ts_ms

        result = landmarker.detect_for_video(mp_image, ts_ms)
        processed += 1
        if not result.pose_world_landmarks:
            continue

        world = result.pose_world_landmarks[0]
        world_arr = np.array([[lm.x, -lm.y, lm.z] for lm in world], dtype=np.float32)
        frames_body25.append(blazepose_to_body25(world_arr))

    cap.release()
    landmarker.close()

    if not frames_body25:
        raise SystemExit("no pose landmarks detected; nothing saved")

    out = args.out or _default_output(args.video, args.label)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    arr = np.stack(frames_body25, axis=0)
    with out.open("wb") as f:
        pickle.dump(
            {
                "label": args.label,
                "frames_body25": arr,
                "source_video": str(args.video),
                "processed_frames": processed,
            },
            f,
        )

    print(f"saved {arr.shape[0]} detected pose frames -> {out}")


if __name__ == "__main__":
    main()
