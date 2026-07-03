# Posture Coach

Live workout form classifier — final project for the Pattern Recognition graduate class (Waseda, Spring 2026).

**Team**: Haruki Oyama, Shuma Kise, Jina Lee.

Three per-exercise Hybrid-STGCN specialists trained on EC3D 3D-pose data, deployed as a PyQt6 desktop app driven by MediaPipe pose estimation. The user picks Squat, Lunges, or Plank on a welcome screen; the selected exercise's 4-seed ensemble classifies form errors in real time and (optionally) an Ollama-backed coach offers a one-line cue when form is off.

## Requirements

- macOS with a webcam.
- Python 3.12.
- Homebrew + Ollama (optional — for the coaching panel).

## Setup

### 1. Clone and install

```bash
git clone https://github.com/harukioya/Pattern-Recognition-Project-Posture-Detector.git
cd Pattern-Recognition-Project-Posture-Detector
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2. Download the MediaPipe pose-landmarker models

```bash
mkdir -p models
curl -sL -o models/pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
curl -sL -o models/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

### 3. Download the EC3D dataset

Follow the Google Drive link in [Jacoo-Zhao/3D-Pose-Based-Feedback-For-Physical-Exercises](https://github.com/Jacoo-Zhao/3D-Pose-Based-Feedback-For-Physical-Exercises) and place `data_3D.pickle` (21 MB) into `data/`.

### 4. Train the per-exercise specialists (~25 min on Apple Silicon)

```bash
for ex in SQUAT Lunges Plank; do
  lower=$(echo "$ex" | tr '[:upper:]' '[:lower:]')
  for seed in 0 1 2 3; do
    python src/train.py --arch hybrid --feature-mode pose_extras \
      --train-split trainval --epochs 60 --seed $seed \
      --ckpt-tag pex_${lower}_s${seed} \
      --exercise-filter $ex --include-self-data --quiet
  done
done
```

Produces 12 checkpoints under `checkpoints/` (4 seeds × 3 exercises) matching the filenames the app loads.

### 5. Launch the app

```bash
python src/app/run.py
```

macOS will prompt for camera permission on first launch.

## Using the app

Pick an exercise on the welcome screen — `S` squat, `L` lunges, `P` plank, or click a card. The workout view shows the camera with a MediaPipe skeleton overlay on the left and, on the right, the top form verdict, per-class probability bars for that exercise, and the coaching panel.

- `H` / `Home` / `Esc` — return to the welcome screen.
- `Q` — quit.

## Optional: coaching panel

```bash
brew install ollama
brew services start ollama
ollama pull llama3.2:3b
```

Without Ollama the coach panel is idle; everything else still works.

## Optional: fine-tune on your own body

```bash
python src/record_self_data.py
```

Records short MediaPipe clips of the 11 form classes into `data/self_recorded/`. Re-run the step 4 training command afterwards — `--include-self-data` folds them in alongside EC3D.

## Repository layout

```
src/
├── ec3d_dataset.py         EC3D loader + feature extraction
├── model.py                Model heads (BiLSTM / Transformer / ST-GCN / Hybrid)
├── train.py                Trainer; --exercise-filter selects the specialist
├── ensemble_eval.py        Evaluate an ensemble on EC3D test
├── self_data.py            SelfRecordedDataset for fine-tuning
├── record_self_data.py     Interactive self-recording tool
├── blazepose_to_body25.py  MediaPipe 33-joint → BODY_25 mapping
└── app/                    PyQt6 desktop app
```

## Acknowledgments

- EC3D dataset: Zhao et al., "3D Pose Based Feedback for Physical Exercises," ACCV 2022.
- MediaPipe Pose: Google Research.
- Built for the Pattern Recognition class (Spring 2026) under instructors Ogawa, Kobayashi, Hayashi, and Hayamizu.
