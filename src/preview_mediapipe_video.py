"""Preview MediaPipe pose detection on one video or a folder of videos."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp


VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
ROOT = Path(__file__).resolve().parent.parent
HEAVY_MODEL = ROOT / "models" / "pose_landmarker_heavy.task"
LITE_MODEL = ROOT / "models" / "pose_landmarker_lite.task"
MODEL_FILE = HEAVY_MODEL if HEAVY_MODEL.exists() else LITE_MODEL

BLAZEPOSE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
]


def _videos(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.suffix.lower() in VIDEO_EXTS)


def _draw_points(frame, landmarks) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in BLAZEPOSE_EDGES:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (80, 220, 120), 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 3, (80, 180, 255), -1, cv2.LINE_AA)


def _put_status(frame, text: str, has_pose: bool) -> None:
    cv2.putText(
        frame,
        text,
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (40, 220, 80) if has_pose else (40, 40, 255),
        2,
        cv2.LINE_AA,
    )


def _preview_with_solutions(videos: list[Path], speed: float) -> None:
    mp_pose = mp.solutions.pose
    drawer = mp.solutions.drawing_utils
    style = mp.solutions.drawing_styles

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        _preview(videos, speed, lambda rgb, _ts: pose.process(rgb), drawer, style, mp_pose)


def _preview_with_tasks(videos: list[Path], speed: float) -> None:
    if not MODEL_FILE.exists():
        raise SystemExit(
            f"missing MediaPipe task model: {MODEL_FILE}\n"
            "Download one first, or run with Python 3.12 / mediapipe 0.10.15."
        )
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_FILE)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    try:
        _preview(videos, speed, landmarker.detect_for_video)
    finally:
        landmarker.close()


def _preview(videos: list[Path], speed: float, detect, drawer=None, style=None, mp_pose=None) -> None:
    next_ts_ms = 0
    for i, video in enumerate(videos, 1):
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"skip: could not open {video}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        delay_ms = max(1, int(1000 / (fps * speed)))
        frame_step_ms = max(1, int(1000 / fps))
        total = detected = frame_i = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            total += 1
            frame_i += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts_ms = next_ts_ms
            next_ts_ms += frame_step_ms

            if drawer is None:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detect(mp_image, ts_ms)
                landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                if landmarks:
                    _draw_points(frame, landmarks)
            else:
                result = detect(rgb, ts_ms)
                landmarks = result.pose_landmarks
                if landmarks:
                    drawer.draw_landmarks(
                        frame,
                        landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=style.get_default_pose_landmarks_style(),
                    )

            has_pose = landmarks is not None
            detected += int(has_pose)
            _put_status(
                frame,
                f"{i}/{len(videos)} {video.name}  frame {frame_i}  {'POSE' if has_pose else 'NO POSE'}",
                has_pose,
            )

            cv2.imshow("MediaPipe video preview  q=quit n=next p=pause", frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return
            if key == ord("n"):
                break
            if key == ord("p"):
                while (cv2.waitKey(50) & 0xFF) not in {ord("p"), ord("q")}:
                    time.sleep(0.01)

        cap.release()
        pct = detected / total * 100 if total else 0.0
        print(f"{video.name}: pose detected in {detected}/{total} frames ({pct:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?", default=ROOT / "data" / "correct-youtube")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    args = parser.parse_args()

    videos = _videos(args.path)
    if not videos:
        raise SystemExit(f"no videos found in {args.path}")

    if hasattr(mp, "solutions"):
        _preview_with_solutions(videos, args.speed)
    else:
        _preview_with_tasks(videos, args.speed)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
