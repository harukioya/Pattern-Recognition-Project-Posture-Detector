"""Real-time webcam demo: MediaPipe pose -> trained Hybrid model -> Ollama coach.

Pipeline (per frame):
  1. Read frame from webcam.
  2. Run MediaPipe PoseLandmarker -> 33-landmark BlazePose (image + world).
  3. Convert world coords to BODY_25 and renormalise to EC3D-like space.
  4. Buffer the latest WINDOW frames.
  5. Once buffer is full, run top-K Hybrid ensemble; predict among 12 classes.
  6. Draw skeleton + form verdict + small confidence bars + coaching text.
  7. Periodically ask Ollama for a coaching cue when form is wrong; the LLM
     runs on a background thread so the camera loop never stalls.

Press 'q' to quit.
"""
from __future__ import annotations

import argparse
import queue
import textwrap
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import ollama
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont

from ec3d_dataset import MISTAKE_LABELS, extract_features, feature_dim
from ensemble_eval import build_model

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "pose_landmarker_lite.task"
CKPT_DIR = ROOT / "checkpoints"

WINDOW = 64               # frames of pose history -> one prediction
PRED_EVERY = 4            # run model every N frames (smooths display)
COACH_INTERVAL_S = 4.0    # min seconds between two Ollama coaching requests
OLLAMA_MODEL = "llama3.2:3b"

CORRECT_CLASSES: set[str] = {"SQUAT/correct", "Lunges/correct", "Plank/correct"}

OLLAMA_SYSTEM = (
    "You are a concise strength-training form coach. "
    "Given a single detected form error from a workout pose classifier, "
    "respond with ONE short actionable cue, under 18 words. "
    "Do not greet, do not list multiple cues, do not hedge, do not add quotes."
)


# ---- Modern palette (RGBA) -----------------------------------------------
class C:
    panel = (18, 22, 30, 200)        # dark translucent strip
    panel_solid = (24, 28, 38, 235)  # right-side bar panel
    text_primary = (240, 242, 245)
    text_secondary = (155, 162, 178)
    text_dim = (105, 110, 125)
    success = (105, 220, 130)
    warning = (255, 165, 95)
    accent = (120, 180, 255)
    bar_bg = (45, 50, 64)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Find a sensible system font that's actually pretty."""
    candidates = [
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# Fonts loaded once at import time
FONT_EXERCISE = _load_font(16)   # "SQUAT" tag above verdict
FONT_VERDICT = _load_font(36)    # the verdict itself
FONT_BODY = _load_font(20)       # coaching text
FONT_LABEL = _load_font(13)      # bar labels
FONT_VALUE = _load_font(12)      # bar values
FONT_TINY = _load_font(11)       # FPS / buffer HUD


def _round_text(s: str) -> str:
    return s.replace("_", " ").title()


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False) or [text]

from blazepose_to_body25 import blazepose_to_body25, normalise_like_ec3d


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CoachThread(threading.Thread):
    """Background thread that pulls (label, conf) from a queue and asks Ollama.

    The latest result is stored in `self.latest_cue` for the UI thread to read.
    """

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._req_q: queue.Queue[tuple[str, float]] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self.latest_cue: str = ""
        self.latest_for_label: str = ""

    def request(self, label: str, confidence: float) -> None:
        """Submit a coaching request; replaces any pending one."""
        try:
            self._req_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._req_q.put_nowait((label, confidence))
        except queue.Full:
            pass

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                label, conf = self._req_q.get(timeout=0.5)
            except queue.Empty:
                continue
            user_msg = f"Detected form error: {label} (confidence {conf:.2f})."
            try:
                resp = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=[
                        {"role": "system", "content": OLLAMA_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    options={"temperature": 0.2, "num_predict": 48},
                )
                self.latest_cue = resp["message"]["content"].strip()
                self.latest_for_label = label
            except Exception as e:
                self.latest_cue = f"(coach offline: {type(e).__name__})"
                self.latest_for_label = label


@torch.no_grad()
def ensemble_predict(models: list, args_list: list, x: torch.Tensor) -> np.ndarray:
    """Average softmax probs across the ensemble."""
    probs_sum = None
    for m, margs in zip(models, args_list):
        # All models in our ensemble share the same feature mode (pose_extras).
        logits = m(x)
        p = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        probs_sum = p if probs_sum is None else probs_sum + p
    return probs_sum / len(models)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpts", nargs="+",
        default=[
            "bilstm_ec3d_best_hybrid_tv_s3.pt",
            "bilstm_ec3d_best_hybrid_tv_s8.pt",
            "bilstm_ec3d_best_hybrid_tv_s0.pt",
            "bilstm_ec3d_best_hybrid_tv_s4.pt",
        ],
        help="One or more 11-class checkpoint filenames to ensemble.",
    )
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    device = pick_device()
    print(f"device: {device}")

    # Load ensemble models
    models, args_list = [], []
    n_classes = len(MISTAKE_LABELS)
    for name in args.ckpts:
        ckpt = torch.load(CKPT_DIR / name, map_location=device, weights_only=True)
        margs = ckpt["args"]
        m = build_model(margs, n_classes, device)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        models.append(m)
        args_list.append(margs)
        print(f"  loaded {name}  arch={margs.get('arch','?')}  feat={margs['feature_mode']}")
    feature_mode = args_list[0]["feature_mode"]
    fdim = feature_dim(feature_mode)
    print(f"feature_mode={feature_mode}  fdim={fdim}")

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"could not open webcam (index {args.camera})")

    # MediaPipe Pose
    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_FILE))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    pose_buffer: deque = deque(maxlen=WINDOW)
    frame_count = 0
    last_probs: np.ndarray | None = None
    last_pred_label = ""
    start_time = time.time()
    fps_t0 = time.time()
    fps_frames = 0
    fps_display = 0.0
    last_coach_request_t = 0.0

    coach = CoachThread()
    coach.start()

    cv2.namedWindow("posture coach (q to quit)", cv2.WINDOW_NORMAL)
    cv2.moveWindow("posture coach (q to quit)", 60, 60)

    print("\nReady. Press 'q' to quit.")
    print("Stand back so your full body is visible.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Mirror for natural-feeling display
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        if result.pose_landmarks:
            lms = result.pose_landmarks[0]              # image-normalised (for drawing)
            world = result.pose_world_landmarks[0]      # metric 3D, hip-centered (for the model)
            # World coords: x = lateral, y = down (toward feet), z = forward.
            # EC3D convention: y = up (spine vertical). Flip y to match.
            world_arr = np.array(
                [[lm.x, -lm.y, lm.z] for lm in world], dtype=np.float32
            )
            body25 = blazepose_to_body25(world_arr)
            pose_buffer.append(body25)
            # Draw the skeleton using the image-normalised landmarks.
            for lm in lms:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)

            # Run model every PRED_EVERY frames once buffer is full
            if len(pose_buffer) == WINDOW and frame_count % PRED_EVERY == 0:
                seq = np.stack(pose_buffer, axis=0)                  # (T, 25, 3)
                seq = normalise_like_ec3d(seq)
                feats = extract_features(seq, feature_mode)          # (T, F)
                x = torch.from_numpy(feats).unsqueeze(0).to(device)  # (1, T, F)
                new_probs = ensemble_predict(models, args_list, x)
                # Light temporal smoothing so the bars don't flicker every 4 frames.
                if last_probs is None:
                    last_probs = new_probs
                else:
                    last_probs = 0.6 * last_probs + 0.4 * new_probs
                top_idx = int(np.argmax(last_probs))
                last_pred_label = MISTAKE_LABELS[top_idx]
        else:
            cv2.putText(frame, "no person detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ----- HUD rendering (PIL for nicer fonts) -----
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        # Tiny FPS HUD (always visible)
        draw.text((20, h - 22),
                  f"{fps_display:4.1f} fps   buffer {len(pose_buffer)}/{WINDOW}",
                  fill=C.text_dim, font=FONT_TINY)

        if last_probs is not None:
            top_idx = int(np.argmax(last_probs))
            label = MISTAKE_LABELS[top_idx]
            confidence = float(last_probs[top_idx])
            exercise, status = label.split("/")
            is_correct = label in CORRECT_CLASSES
            verdict_color = C.success if is_correct else C.warning

            # ---- Top strip: verdict ----
            top_strip_h = 96
            draw.rectangle((0, 0, w, top_strip_h), fill=C.panel)
            # Soft accent line below the strip
            draw.rectangle((0, top_strip_h, w, top_strip_h + 2),
                           fill=(verdict_color[0], verdict_color[1], verdict_color[2], 180))

            # Exercise tag (small, secondary)
            draw.text((24, 16), exercise.upper(),
                      fill=C.text_secondary, font=FONT_EXERCISE)
            # Verdict (large, primary)
            verdict_text = "Form looks good" if is_correct else _round_text(status)
            draw.text((24, 38), verdict_text, fill=verdict_color, font=FONT_VERDICT)
            # Confidence (right-aligned-ish)
            conf_text = f"{confidence:.0%}"
            cw = draw.textlength(conf_text, font=FONT_VERDICT)
            draw.text((w - cw - 28, 38), conf_text,
                      fill=C.text_primary, font=FONT_VERDICT)
            draw.text((w - cw - 110, 22), "confidence",
                      fill=C.text_secondary, font=FONT_EXERCISE)

            # ---- Right side: probability bars, grouped by exercise ----
            # Group labels by exercise prefix, in display order. Skip
            # SQUAT/squat_extra (training artifact, not a useful display class).
            groups: list[tuple[str, list[int]]] = []
            for ex_name in ("SQUAT", "Lunges", "Plank"):
                ids = [
                    i for i, lab in enumerate(MISTAKE_LABELS)
                    if lab.startswith(ex_name + "/") and lab != "SQUAT/squat_extra"
                ]
                groups.append((ex_name, ids))

            panel_w = 260
            panel_x = w - panel_w - 12
            panel_y = top_strip_h + 14
            row_h = 22
            header_h = 22
            gap_h = 10
            n_rows = sum(len(ids) for _, ids in groups)
            panel_h = (n_rows * row_h
                       + len(groups) * header_h
                       + (len(groups) - 1) * gap_h + 18)
            draw.rounded_rectangle(
                (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h),
                radius=10, fill=C.panel_solid,
            )

            cur_y = panel_y + 10
            for gi, (ex_name, ids) in enumerate(groups):
                # Section header
                draw.text((panel_x + 12, cur_y),
                          ex_name.upper(),
                          fill=C.text_secondary, font=FONT_EXERCISE)
                cur_y += header_h

                for i in ids:
                    lab = MISTAKE_LABELS[i]
                    p = float(last_probs[i])
                    err_name = lab.split("/", 1)[1].replace("_", " ")
                    if err_name == "correct":
                        err_name = "correct"  # keep lowercase to match other entries
                    is_top = i == top_idx
                    # Bar background
                    bar_x = panel_x + 12
                    bar_track_w = panel_w - 70
                    draw.rounded_rectangle(
                        (bar_x, cur_y + 14, bar_x + bar_track_w, cur_y + 19),
                        radius=2, fill=C.bar_bg,
                    )
                    # Bar fill
                    if p > 0.01:
                        fill_w = max(2, int(p * bar_track_w))
                        if is_top:
                            col = C.success if is_correct else C.warning
                        else:
                            col = C.text_secondary
                        draw.rounded_rectangle(
                            (bar_x, cur_y + 14, bar_x + fill_w, cur_y + 19),
                            radius=2, fill=col,
                        )
                    # Label (indented)
                    draw.text((bar_x + 4, cur_y - 1), err_name,
                              fill=C.text_primary if is_top else C.text_secondary,
                              font=FONT_LABEL)
                    # Value
                    val_text = f"{p:.2f}"
                    vw = draw.textlength(val_text, font=FONT_VALUE)
                    draw.text((panel_x + panel_w - vw - 12, cur_y - 1), val_text,
                              fill=C.text_primary if is_top else C.text_dim,
                              font=FONT_VALUE)
                    cur_y += row_h

                if gi != len(groups) - 1:
                    cur_y += gap_h - 4
                    # thin divider line
                    draw.rectangle(
                        (panel_x + 16, cur_y, panel_x + panel_w - 16, cur_y + 1),
                        fill=C.bar_bg,
                    )
                    cur_y += 6

            # ---- Bottom strip: coaching ----
            cue_h = 92
            draw.rectangle((0, h - cue_h, w, h), fill=C.panel)
            # Status dot + label
            dot_r = 6
            draw.ellipse(
                (24, h - cue_h + 20 - dot_r, 24 + dot_r * 2, h - cue_h + 20 + dot_r),
                fill=verdict_color,
            )
            label_text = "Coach" if not is_correct else "Coach"
            draw.text((44, h - cue_h + 14), label_text,
                      fill=C.text_secondary, font=FONT_EXERCISE)

            if is_correct:
                cue_text = "Form looks clean. Keep your tempo steady."
                cue_color = C.text_primary
            elif coach.latest_cue and coach.latest_for_label == label:
                cue_text = coach.latest_cue
                cue_color = C.text_primary
            else:
                cue_text = "Analyzing your form..."
                cue_color = C.text_secondary

            # Wrap to two lines max
            max_chars = max(30, (w - 60) // 11)
            lines = _wrap(cue_text, max_chars)[:2]
            for i, ln in enumerate(lines):
                draw.text((24, h - cue_h + 38 + i * 26),
                          ln, fill=cue_color, font=FONT_BODY)

            # ---- Throttled coaching request ----
            now = time.time()
            if (not is_correct
                    and confidence > 0.35
                    and (now - last_coach_request_t) > COACH_INTERVAL_S):
                coach.request(label, confidence)
                last_coach_request_t = now
        else:
            # Pre-warmup state
            draw.rectangle((0, 0, w, 60), fill=C.panel)
            draw.text((24, 20), "Warming up — stand back so your full body fits in view",
                      fill=C.text_primary, font=FONT_BODY)

        composed = Image.alpha_composite(pil, overlay).convert("RGB")
        frame = cv2.cvtColor(np.array(composed), cv2.COLOR_RGB2BGR)
        cv2.imshow("posture coach (q to quit)", frame)

        frame_count += 1
        fps_frames += 1
        if fps_frames >= 30:
            now = time.time()
            fps_display = fps_frames / (now - fps_t0)
            fps_t0 = now
            fps_frames = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    coach.stop()


if __name__ == "__main__":
    main()
