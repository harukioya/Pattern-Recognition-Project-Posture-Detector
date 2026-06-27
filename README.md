# Posture Coach — Live Workout Form Classifier

Final project for the **Pattern Recognition** graduate class (Waseda Spring 2026).

Trains a deep model on the EC3D 3D-pose dataset, then deploys it as a live PyQt
desktop app driven by MediaPipe pose estimation, with optional Ollama-backed
natural-language coaching.

**Benchmark result**: 0.838 macro-F1 / 82.6% accuracy on the EC3D held-out
test subject (Isinsu) using a 4-seed top-K-by-val HybridSTGCN ensemble.

For the full session log, architecture history, and lessons learned, see
[`SESSION_HANDOFF.md`](./SESSION_HANDOFF.md).

---

## What's in here

- **Per-error model** — 4-seed Hybrid (ST-GCN over BODY_25 joints + MLP over
  engineered features) ensemble that picks the top form-error class within an
  exercise.
- **Exercise gate** — separate 4-seed 3-class Hybrid ensemble that chooses
  between Squat / Lunges / Plank before the per-error head runs.
- **Live desktop app** — PyQt6 with a custom-painted camera widget showing the
  MediaPipe skeleton overlay, a verdict tile, grouped probability bars per
  exercise, an Ollama coaching panel, and a status bar.
- **Self-recording tool** — captures MediaPipe `pose_world_landmarks` while
  walking you through the 11 form classes, so you can fine-tune the model
  to your own body and camera setup.

---

## Requirements

- macOS (Apple Silicon recommended) — paths and Ollama install assume Mac.
- Python 3.12 (the venv was built with `python3` from `/Library/Frameworks/Python.framework`).
- A working webcam.
- Homebrew (for Ollama, optional).

---

## Fine-tune the model for yourself

### 1. Clone + Python environment

```bash
git clone https://github.com/harukioya/Pattern-Recognition-Project-Posture-Detector.git
cd Pattern-Recognition-Project-Posture-Detector

# Make sure you are running native arm64 (run `arch`; if it prints `i386`,
# do `arch -arm64 zsh` first).

python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2. Install Ollama (optional — for coaching cues)

```bash
brew install ollama
brew services start ollama
ollama pull llama3.2:3b
```

If Ollama isn't installed, the coach panel will just print
`(coach offline: <ConnectionError>)` and the rest of the app still works.

### 3. Download MediaPipe pose-landmarker models

```bash
mkdir -p models
curl -sL -o models/pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
curl -sL -o models/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

### 4. Download the EC3D dataset

- Go to [Jacoo-Zhao/3D-Pose-Based-Feedback-For-Physical-Exercises](https://github.com/Jacoo-Zhao/3D-Pose-Based-Feedback-For-Physical-Exercises).
- Follow the Google Drive link in their README.
- Drop `data_3D.pickle` (21 MB) into `data/`.

### 5. Train the base models on EC3D (about 5 minutes)

```bash
# Four Hybrid (per-error) seeds
for seed in 0 1 2 3; do
  python src/train.py --arch hybrid --feature-mode pose_extras \
    --train-split trainval --epochs 60 --seed $seed \
    --ckpt-tag hybrid_tv_s$seed
done

# Four Gate (3-class) seeds
for seed in 0 1 2 3; do
  python src/train_gate.py --epochs 60 --seed $seed --ckpt-tag s$seed
done
```

You can sanity-check the per-error ensemble at this point:

```bash
python src/ensemble_eval.py --split test --tta-crops 1 \
  --ckpts bilstm_ec3d_best_hybrid_tv_s0.pt \
          bilstm_ec3d_best_hybrid_tv_s1.pt \
          bilstm_ec3d_best_hybrid_tv_s2.pt \
          bilstm_ec3d_best_hybrid_tv_s3.pt
```

Expect roughly macro-F1 = 0.82 on EC3D test.

### 6. Record your own form data (about 10 minutes wall clock)

```bash
python src/record_self_data.py
```

The tool walks through 11 sessions (Squat correct / wide / inward / shallow /
front-bent, Lunge correct / shallow / knee-past-toe, Plank correct / arched /
hunched), 30 seconds each.

| Key | While countdown | While recording |
|---|---|---|
| `SPACE` | start now (skip countdown) | — |
| `P` | — | pause / resume; timer freezes |
| `R` | — | restart this session from the countdown |
| `S` | skip the session | skip and move on |
| `Q` | quit; everything already saved is kept | same |

Each session writes its own `.pkl` to `data/self_recorded/`. Re-running the
script will auto-skip labels you've already recorded; pass `--redo-all` to
re-record everything.

#### Recording tips
- Stand back so your **full body** fits in the frame.
- Use a **side view** for lunges and planks (the knee-past-toe and arched-back
  features need to be visible in profile).
- A 45° angle works well for squats.
- **Exaggerate the errors** — subtle imperfections won't survive MediaPipe noise.

### 7. Fine-tune the models on EC3D + your data (about 5 minutes)

```bash
for seed in 0 1 2 3; do
  python src/train.py --arch hybrid --feature-mode pose_extras \
    --train-split trainval --epochs 60 --seed $seed \
    --ckpt-tag ft_s$seed --include-self-data
done

for seed in 0 1 2 3; do
  python src/train_gate.py --epochs 60 --seed $seed \
    --ckpt-tag ft_s$seed --include-self-data
done
```

The app's defaults in `src/app/pipeline_thread.py` already point at
`bilstm_ec3d_best_ft_s0..3.pt` and `gate_hybrid_tv_ft_s0..3.pt`, so the next
launch will pick up the fine-tuned ensemble automatically.

### 8. Launch the desktop app

```bash
python src/app/run.py
```

On first launch macOS will ask for camera permission for whichever terminal
app started Python. Grant it, fully quit and reopen Terminal, then re-run.

---

## What to expect when you run the app

- A 1400×900 window with the camera on the left, three stacked cards on the
  right (verdict, probability bars, coaching), and a thin status bar at the
  bottom.
- The header shows the **gate's predicted exercise** (Squat / Lunges / Plank)
  or `Waiting for a clear pose` when confidence is low or you're standing
  still.
- The verdict card shows `Form looks good` or a humanised error name
  (`Knees Inward`, etc.) with the model's confidence.
- The right panel shows per-class probability bars grouped by exercise; the
  gated exercise is at full brightness and others are dimmed.
- The coach panel asks Ollama for a one-line cue every ~4 seconds when the
  form is incorrect.

Press `Q` to quit.

---

## Honest caveats

- The EC3D training data is small (287 trainval clips, 4 subjects) and clean
  (4-camera triangulated 3D), while live MediaPipe input is single-camera
  noisy 2.5D. There is a real **domain gap** that fine-tuning with self-data
  helps with but does not eliminate.
- The hardest class for the deep model is `Lunges/knee_passes_toe`, especially
  in live view; the gate sometimes mistakes a standing pose for a lunge
  because of MediaPipe's noisy z-axis.
- For the most reliable demo, consider augmenting the deep model with a
  rule-based joint-angle analyzer at inference time. See `SESSION_HANDOFF.md`
  for the discussion.

---

## Repository layout

```
src/
├── ec3d_dataset.py         # EC3D loader + feature extraction
├── model.py                # BiLSTM / Transformer / ST-GCN / Hybrid heads
├── train.py                # generic trainer (any arch)
├── train_gate.py           # 3-class exercise-gate trainer
├── ensemble_eval.py        # average N checkpoints, evaluate
├── rank_by_val.py          # honest top-K seed selection via Vidit val
├── blazepose_to_body25.py  # MediaPipe 33 -> BODY_25 25-joint mapping
├── self_data.py            # SelfRecordedDataset for fine-tuning
├── record_self_data.py     # interactive 11-session recording tool
├── synth_mediapipe.py      # noise + yaw augmentation (deprecated)
└── app/                    # PyQt6 desktop app
    ├── run.py              # entry point
    ├── main_window.py
    ├── pipeline_thread.py  # webcam + MediaPipe + model inference
    ├── coach_thread.py     # Ollama coaching
    ├── state.py            # Prediction dataclass + class constants
    ├── style.qss           # dark theme stylesheet
    └── widgets/
        ├── header_bar.py
        ├── camera_widget.py
        ├── verdict_panel.py
        ├── bars_panel.py
        ├── coach_panel.py
        └── status_bar.py
```

---

## Acknowledgments

- **EC3D dataset**: Zhao et al., "3D Pose Based Feedback for Physical
  Exercises," ACCV 2022.
- **MediaPipe Pose**: Google Research.
- **Ollama / Llama 3.2**: Meta + the Ollama community.
- Built for the Pattern Recognition class (Spring 2026) under instructors
  Ogawa, Kobayashi, Hayashi, and Hayamizu.
