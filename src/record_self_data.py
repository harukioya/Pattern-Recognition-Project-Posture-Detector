"""Self-recording tool for fine-tuning data.

Walks you through 11 short sessions (one per form class), captures MediaPipe
pose_world_landmarks per frame, and saves to data/self_recorded/.

Usage:
    python src/record_self_data.py [--duration 30] [--countdown 5]

Controls (during countdown):
    SPACE  start now (skip the countdown)
    S      skip this session
    Q      quit early (everything captured so far is kept)

Controls (during recording):
    P      pause / resume — timer freezes, no frames captured while paused
    R      restart this session from the countdown
    S      skip and move to the next session
    Q      quit early (everything captured so far is kept)

If a label was already saved in a previous run, the tool auto-skips it.
Pass --redo-all to re-record all labels, or --start-from N to begin
at session N (1-indexed).
"""
from __future__ import annotations

import argparse
import pickle
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from blazepose_to_body25 import blazepose_to_body25

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = (ROOT / "models" / "pose_landmarker_heavy.task"
              if (ROOT / "models" / "pose_landmarker_heavy.task").exists()
              else ROOT / "models" / "pose_landmarker_lite.task")
SAVE_DIR = ROOT / "data" / "self_recorded"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# Exact label space we want to fine-tune on. Order is the recording order.
SESSIONS: list[tuple[str, str, str]] = [
    # (exercise, error_name, human_description)
    ("SQUAT", "correct",          "Standard bodyweight squat with good form"),
    ("SQUAT", "feet_too_wide",    "Squat with feet much wider than shoulder-width"),
    ("SQUAT", "knees_inward",     "Squat letting knees collapse inward toward each other"),
    ("SQUAT", "not_low_enough",   "Half-squats; never go below parallel"),
    ("SQUAT", "front_bent",       "Squat with an exaggerated forward torso lean"),
    ("Lunges", "correct",         "Forward lunges with good knee tracking"),
    ("Lunges", "not_low_enough",  "Shallow lunges; never drop the back knee"),
    ("Lunges", "knee_passes_toe", "Lunges where the front knee tracks well past the toes"),
    ("Plank",  "correct",         "Standard plank, body straight from shoulders to ankles"),
    ("Plank",  "arched_back",     "Plank with hips sagging toward the floor"),
    ("Plank",  "hunch_back",      "Plank with hips raised high (pike / mountain shape)"),
]


def _draw_centered(frame: np.ndarray, lines: list[str], colors: list[tuple] | None = None) -> None:
    h, w = frame.shape[:2]
    if colors is None:
        colors = [(240, 240, 245)] * len(lines)
    # Translucent panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h // 3), (w, 2 * h // 3), (15, 20, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, dst=frame)
    y = h // 3 + 70
    for i, ln in enumerate(lines):
        size = 1.4 if i == 0 else 0.9
        thick = 3 if i == 0 else 2
        (tw, th), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, size, thick)
        cv2.putText(
            frame, ln, ((w - tw) // 2, y),
            cv2.FONT_HERSHEY_SIMPLEX, size, colors[i], thick, cv2.LINE_AA,
        )
        y += int(th * 1.8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--countdown", type=int, default=5)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--redo-all", action="store_true",
                        help="re-record every session even if a .pkl already exists")
    parser.add_argument("--start-from", type=int, default=1,
                        help="1-indexed session to begin at (skips earlier ones)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {args.camera}")

    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_FILE))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    pipeline_start = time.time()

    cv2.namedWindow("self-record", cv2.WINDOW_NORMAL)
    cv2.moveWindow("self-record", 60, 60)

    captured: list[tuple[str, str, np.ndarray]] = []

    # Discover what's already been recorded so we can auto-skip
    already_done: set[str] = set()
    if not args.redo_all:
        for exercise, error, _desc in SESSIONS:
            if (SAVE_DIR / f"{exercise}_{error}.pkl").exists():
                already_done.add(f"{exercise}/{error}")

    print(f"\nReady. {len(SESSIONS)} sessions, {args.duration:.0f}s each.")
    if already_done:
        print(f"Auto-skipping {len(already_done)} already-recorded session(s): "
              f"{sorted(already_done)}")
    if args.start_from > 1:
        print(f"Starting from session #{args.start_from}")
    print("Stand back so your full body fits in view.\n")

    skip = False
    quit_all = False

    for sess_i, (exercise, error, desc) in enumerate(SESSIONS):
        if quit_all:
            break
        if sess_i + 1 < args.start_from:
            continue
        label = f"{exercise}/{error}"
        if label in already_done:
            print(f"[{sess_i+1}/{len(SESSIONS)}] {label}: already saved, skipping")
            continue
        print(f"[{sess_i+1}/{len(SESSIONS)}] {label}: {desc}")

        # ---- Countdown phase ----
        skip = False
        countdown_start = time.time()
        while True:
            elapsed = time.time() - countdown_start
            remaining = args.countdown - int(elapsed)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            # Still run MediaPipe to warm up
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - pipeline_start) * 1000)
            landmarker.detect_for_video(mp_image, ts_ms)

            _draw_centered(
                frame,
                [
                    f"GET READY  [{sess_i+1}/{len(SESSIONS)}]",
                    f"{label}",
                    desc,
                    f"Starting in {max(0, remaining)}…  (SPACE = start now, S = skip, Q = quit)",
                ],
                colors=[(120, 200, 255), (240, 240, 245), (155, 162, 178), (155, 162, 178)],
            )
            cv2.imshow("self-record", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                quit_all = True
                break
            if k == ord("s"):
                skip = True
                break
            if k == ord(" ") or remaining <= 0:
                break

        if quit_all or skip:
            if skip:
                print(f"  ... skipped {label}")
            continue

        # ---- Recording phase ----
        # The whole session can be replayed (R) if you want a do-over; pause (P)
        # freezes the timer and capture without losing the frames already collected.
        restart_session = True
        while restart_session:
            restart_session = False
            record_start = time.time()
            paused = False
            pause_started_at: float | None = None
            total_paused = 0.0
            frames_world: list[np.ndarray] = []
            skip_session = False
            while True:
                now = time.time()
                elapsed = (now - record_start) - total_paused
                remaining = args.duration - elapsed
                if not paused and remaining <= 0:
                    break

                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int((time.time() - pipeline_start) * 1000)
                result = landmarker.detect_for_video(mp_image, ts_ms)

                if not paused and result.pose_landmarks and result.pose_world_landmarks:
                    lms = result.pose_landmarks[0]
                    world = result.pose_world_landmarks[0]
                    world_arr = np.array(
                        [[lm.x, -lm.y, lm.z] for lm in world], dtype=np.float32
                    )
                    body25 = blazepose_to_body25(world_arr)
                    frames_world.append(body25)
                    h, w = frame.shape[:2]
                    for lm in lms:
                        cv2.circle(
                            frame, (int(lm.x * w), int(lm.y * h)),
                            3, (105, 220, 130), -1,
                        )

                if paused:
                    # Dim the frame so it's visually obvious we're paused
                    frame = (frame * 0.55).astype(np.uint8)
                    _draw_centered(
                        frame,
                        [
                            "PAUSED",
                            f"{label}",
                            f"{remaining:0.1f}s remaining  (paused)",
                            "P = resume   R = restart   S = skip   Q = quit",
                        ],
                        colors=[(120, 200, 255), (240, 240, 245),
                                (155, 162, 178), (155, 162, 178)],
                    )
                else:
                    _draw_centered(
                        frame,
                        [
                            "RECORDING",
                            f"{label}",
                            f"{remaining:0.1f}s remaining",
                            f"frames: {len(frames_world)}   "
                            "P=pause  R=restart  S=skip  Q=quit",
                        ],
                        colors=[(105, 220, 130), (240, 240, 245),
                                (155, 162, 178), (155, 162, 178)],
                    )
                cv2.imshow("self-record", frame)

                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    quit_all = True
                    break
                if k == ord("s"):
                    skip_session = True
                    break
                if k == ord("r"):
                    print(f"  ... restarting session {label}")
                    restart_session = True
                    break
                if k == ord("p"):
                    if paused:
                        # Resume: bank the paused interval so the timer ignores it
                        if pause_started_at is not None:
                            total_paused += time.time() - pause_started_at
                        pause_started_at = None
                        paused = False
                    else:
                        pause_started_at = time.time()
                        paused = True

            if quit_all or skip_session:
                if skip_session:
                    print(f"  ... skipped {label}")
                break

        if frames_world:
            arr = np.stack(frames_world, axis=0)
            out = SAVE_DIR / f"{exercise}_{error}.pkl"
            with out.open("wb") as f:
                pickle.dump({"label": label, "frames_body25": arr}, f)
            captured.append((label, str(out.name), arr))
            print(f"  saved {arr.shape[0]} frames -> {out.name}")
        else:
            print(f"  WARNING: no landmarks captured for {label}; skipping save")

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

    if captured:
        print(f"\nDone. {len(captured)} sessions saved to {SAVE_DIR}/")
        for lab, name, arr in captured:
            print(f"  {lab:30s}  {arr.shape[0]:4d} frames  {name}")
    else:
        print("\nNothing captured.")


if __name__ == "__main__":
    main()
