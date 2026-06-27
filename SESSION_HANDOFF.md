# Posture Coach — Session Handoff

A detailed record of everything done in this session so that any future agent (or human) can pick up exactly where we left off without re-discovering the same dead ends.

---

## 1. Project Context

- **Class**: Pattern Recognition (Spring 2026, Waseda University). Instructors: Ogawa, Kobayashi, Hayashi, Hayamizu.
- **Goal**: Workout posture corrector / coach — a deep-learning system that classifies exercise form errors from video, with a live coaching demo.
- **Project requirements** (from `Lecture Slides/PR00_2026.pdf`):
  - Must use deep neural networks
  - Must define own research question
  - Must implement + run experiments + submit source code
  - Must produce a presentation video by July 3 originally (deadline may have shifted)
- **Hardware**: MacBook Pro M2 (arm64). Apple Silicon native. ~17 GB free disk at start. No GPU access other than MPS.

---

## 2. Data & Datasets Considered

### Used
- **EC3D** (Zhao et al., ACCV 2022) — 3 exercises × 11 form-error labels, 4 subjects, 3D pose from 4-camera triangulation. **287 train clips at our 64-frame window**. This is our primary training data.
  - Files: `data/data_3D.pickle` (21 MB), `data/data.pickle` (917 MB, has 2D+camera params, mostly unused)
  - Labels in form `(act, sub, lab, rep, frame)` — col 2 is the instruction code, col 3 is the rep
  - Per-exercise error taxonomy from EC3D paper Table 1:
    - Squat: correct, feet_too_wide, knees_inward, not_low_enough, front_bent (+ a code-10 "extra" class we drop)
    - Lunges: correct, not_low_enough, knee_passes_toe
    - Plank: correct, arched_back, hunch_back
- **UI-PRMD** (binary correct/incorrect only, preprocessed) — 3,598 clips
- **ExeChecker** (preprocessed; joint topology doesn't match BODY_25 reliably, so we did NOT integrate it)
- **Self-recorded MediaPipe data** — recorded today, 44 clips total (4 per class × 11 classes), saved as pkl in `data/self_recorded/`

### Considered & rejected
- **Fitness-AQA, FLEX, M3GYM, EgoExo-Fitness** — all gated by request forms, not realistic timeline
- **LLM-FMS** (Mar 2025) — keyframe images, no temporal sequences; incompatible with our temporal models
- **REHAB24-6** — binary labels only, duplicates UI-PRMD's limitation
- **Fit3D academic** — continuous deviation scores, no categorical errors
- **AthletePose3D / 3D-Yoga** — proxy pretraining options, not integrated due to time

---

## 3. Architecture Evolution & Test-Set Numbers

All numbers are macro-F1 on EC3D held-out subject (Isinsu, 104 clips).

### Per-error classifier (12 classes used, 11 in display)

| Stage | Test F1 | Test Acc | Notes |
|---|---|---|---|
| Baseline (angles only, BiLSTM) | 0.36 | 47.1% | 12 hand-picked joint angles |
| + 17 engineered geometric features (lateral_dev, stance_ratio, etc.) | 0.61 | 65.4% | Biggest single jump |
| + Transformer architecture | 0.61 | 64.4% | Same F1, complementary per-class profile |
| + BiLSTM+Transformer ensemble | 0.71 | 71.2% | +0.10 from ensemble diversity |
| + Trainval split (Hugues+Sena+Vidit, paper protocol) | 0.73 | 77.9% | +0.02 from more train subjects |
| + 4-seed Transformer ensemble | 0.76 | 79.8% | +0.03 from seed ensembling |
| + Signed-axis geom features | 0.78 | 79.8% | Fixed front_bent class |
| + ST-GCN alone (4-seed) | 0.62 | 67.3% | Single-partition graph conv |
| + HybridSTGCN single seed (best of 8) | 0.84 | 83.7% | ST-GCN over joints + MLP over engineered features |
| **+ 8-seed Hybrid ensemble** | **0.822** | **82.7%** | Multi-seed averaging |
| + Top-4-by-val honest selection | 0.838 | 82.6% | Honest seed selection via held-out val (Vidit) |

**Best honest single model**: Hybrid seed 15 — F1=0.876, Acc=87.5%
**Final benchmark result**: 4-seed top-K-by-val Hybrid ensemble — F1=0.838, Acc=82.6%

### Exercise gate (3-class Squat / Lunges / Plank)
- 4-seed ensemble trained on EC3D trainval
- 100% accuracy on EC3D Isinsu test set (the exercise discrimination task is genuinely easy on clean 3D data)

### Things tried that didn't help
- Mirror augmentation (overfit val, hurt test)
- Test-time augmentation with temporal crops (averaged out discriminative signal)
- 16 vs 8 vs 4 seeds (4 was the sweet spot)
- Multi-partition GCB (ST-GCN v2) alone — no clear improvement
- Weighted ensembles (Hybrid + Transformer) — worse than Hybrid alone
- Hierarchical pipeline (UI-PRMD binary gate → EC3D 11-class) — F1 dropped from 0.842 to 0.777
- Synthetic MediaPipe augmentation — F1 dropped to 0.57-0.65, demo regressed to "everything is plank"

---

## 4. Files & Codebase Structure

```
FinalProject/
├── venv/                          # Python 3.12 venv (arm64; check `arch`)
├── models/
│   ├── pose_landmarker_lite.task          # MediaPipe lite (5.5 MB)
│   └── pose_landmarker_heavy.task         # MediaPipe heavy (30 MB) — current default
├── data/
│   ├── data_3D.pickle              # EC3D 3D pose + labels (USED)
│   ├── data.pickle                 # EC3D 2D + cameras (mostly unused)
│   ├── processed_extracted/        # ExeChecker + UI-PRMD preprocessed
│   └── self_recorded/              # 11 pkl files from recording session today
├── checkpoints/                    # ~50 trained checkpoints, see "Checkpoints" below
└── src/
    ├── ec3d_dataset.py             # EC3D loader, feature extraction (angles + positions + geom), synth augmentation flag
    ├── model.py                    # BiLSTMHead, TransformerHead, STGCNHead/v2, HybridSTGCNHead/v2, BinarySTGCNHead
    ├── train.py                    # Generic trainer, supports --include-self-data, --synth-mediapipe
    ├── train_gate.py               # 3-class exercise gate trainer (Hybrid arch, 3 outputs)
    ├── train_binary.py             # Binary gate trainer for hierarchical pipeline (unused now)
    ├── ensemble_eval.py            # Loads N checkpoints, averages probs, evaluates
    ├── rank_by_val.py              # Picks top-K seeds by Vidit val F1 (honest selection)
    ├── hierarchical_eval.py        # 2-stage binary→11-class eval (showed regression)
    ├── blazepose_to_body25.py      # MediaPipe BlazePose 33 → OpenPose BODY_25 mapping
    │                                 #   normalise_like_ec3d: centre + scale (no rotation; canonical_rotate was tried then reverted)
    ├── joint_mapping.py            # H3.6M-17 ↔ BODY_25 / 21-joint mapping
    ├── binary_dataset.py           # EC3D + UI-PRMD unified for binary task
    ├── self_data.py                # SelfRecordedDataset wrapper for fine-tuning
    ├── synth_mediapipe.py          # Synthetic noise + yaw rotation augmentation (regressed live, deprecated)
    ├── record_self_data.py         # Interactive recording tool — 11 sessions of 30s each with P/R/S/Q controls
    ├── realtime_demo.py            # OpenCV-based live demo (now superseded by PyQt app)
    └── app/                        # PyQt6 desktop app
        ├── run.py                  # entry: `python src/app/run.py`
        ├── main_window.py          # QMainWindow scaffold + signal wiring
        ├── pipeline_thread.py      # QThread: webcam + MediaPipe + 2 model ensembles
        ├── coach_thread.py         # QThread: Ollama coaching via Llama 3.2 3B
        ├── state.py                # Prediction dataclass with gate fields + is_uncertain
        ├── style.qss               # Dark theme stylesheet
        └── widgets/
            ├── header_bar.py       # Top bar with exercise name (26pt bold)
            ├── camera_widget.py    # Custom paintEvent: scaled frame + skeleton overlay
            ├── verdict_panel.py    # FORM card with verdict + confidence
            ├── bars_panel.py       # Grouped probability bars with gate-confidence section headers
            ├── coach_panel.py      # Coaching text + status dot
            └── status_bar.py       # FPS + buffer + Q-to-quit hint
```

---

## 5. Important Checkpoints

In `checkpoints/`:

| Filename pattern | Description | Status |
|---|---|---|
| `bilstm_ec3d_best_hybrid_tv_s{0,1,...,15}.pt` | Original 16 Hybrid v1 seeds | Best ensemble: s3/s8/s0/s4 (top-4-by-val) |
| `bilstm_ec3d_best_hybridv2_tv_s{0-3}.pt` | Multi-partition ST-GCN Hybrid v2 | Comparable to v1 |
| `gate_hybrid_tv_s{0-3}.pt` | Exercise gate (3-class) | Currently used in pipeline |
| `bilstm_ec3d_best_synth_s{0-3}.pt` | Synth-augmented Hybrid | **Regressed demo, do not use** |
| `gate_hybrid_tv_synth_s{0-3}.pt` | Synth-augmented gate | **Same, do not use** |
| `bilstm_ec3d_best_ft_s{0-3}.pt` | Fine-tuned on EC3D + self-recorded | **Currently being trained when context was running out** |
| `gate_hybrid_tv_ft_s{0-3}.pt` | Gate fine-tuned on EC3D + self-recorded | Done; Gate FT s0 shows 100% on EC3D test for 2 classes, 91% on plank |

**Pipeline's current default** (in `src/app/pipeline_thread.py`):
- Hybrid: `hybrid_tv_s3/s8/s0/s4` (4 models)
- Gate: `gate_hybrid_tv_s0/s1/s2/s3` (4 models)

After Hybrid FT training finishes, the user/agent should update DEFAULT_CKPTS and DEFAULT_GATE_CKPTS in `pipeline_thread.py` to use the fine-tuned versions.

---

## 6. Live Application — PyQt Desktop App

Built today via 4 parallel sub-agents that each owned one widget file:
- Agent A: `camera_widget.py`
- Agent B: `verdict_panel.py` + `bars_panel.py`
- Agent C: `coach_panel.py` + `status_bar.py`
- Agent D: `style.qss`
- Skeleton + main_window: built by main agent

### Key features
- Camera left (rounded card, skeleton overlay), right column has Verdict + Bars + Coach panels
- Top header bar shows the gate's chosen exercise in 26pt
- Bottom status bar shows FPS + buffer fill
- Dark theme via QSS (panel `#161b22`, success `#69dc82`, warning `#ffa55f`)
- Pipeline runs in a QThread, emits `prediction_updated(Prediction)` every 4th frame after the 64-frame buffer fills
- Coach in another QThread, calls Ollama Llama 3.2 3B (loaded locally via brew install + ollama pull). Coaching cue requested every 4 sec when form is incorrect AND not uncertain
- `Prediction` dataclass carries `is_uncertain` flag (true when gate confidence < 0.55 OR pose variance < 0.0008)

### How to run
```bash
arch                                            # confirm arm64; if i386, `arch -arm64 zsh` first
cd "/Users/harukioyama/Classes/Pattern Recognition/FinalProject"
source venv/bin/activate
python src/app/run.py
```

---

## 7. The Critical Failure: Domain Gap from EC3D 3D to MediaPipe

This is the central unresolved issue. Documented honestly:

### What works
- EC3D test set (Isinsu): 0.838 F1, 82.6% acc — strong benchmark result
- Gate on EC3D: 100% accuracy
- The PyQt UI itself: stable, well-styled
- Ollama coaching: works, ~36 tok/sec on warm calls

### What doesn't work
- **Live MediaPipe input is consistently misclassified.** Multiple symptoms observed:
  - Morning baseline: model output `Lunges/correct` at confidence 1.0 regardless of pose
  - After adding canonical_rotate: output became `Lunges/knee_passes_toe` regardless of pose
  - After synth augmentation: output became `Plank/correct` regardless of pose (worst)
  - After reverting both: back to `Lunges/knee_passes_toe`
- The bias direction depends on preprocessing decisions; the underlying issue is the model has never seen MediaPipe-style noise patterns during training

### Things tried today that did NOT solve it
1. **MediaPipe lite → heavy** — kept (more accurate, no obvious downside)
2. **`pose_landmarks` → `pose_world_landmarks` + y-flip** — kept (genuine bug fix; image coords aren't 3D)
3. **`canonical_rotate` in `normalise_like_ec3d`** — reverted (speculative; EC3D's convention is unknown, our rotation made bias worse)
4. **Exercise gate** — kept (independent of model behavior; doesn't hurt)
5. **Uncertainty detection** (gate conf < 0.55 OR low movement) — kept (UI improvement only)
6. **Synth augmentation** (noise + yaw rotation on EC3D, ~0.06 std on z) — **regressed, reverted**

### What we're currently doing (Path B from the diagnosis conversation)
**Self-recording + fine-tune.** This is the only intervention using real MediaPipe data — no guessing about noise distributions.

The user recorded **44 clips** (4 per class × 11 form classes) using `record_self_data.py`. The script:
- Walks through 11 sessions of 30s each
- Captures `pose_world_landmarks` per frame (after y-flip, mapped to BODY_25)
- Saves to `data/self_recorded/{exercise}_{error}.pkl`
- Supports P (pause), R (restart session), S (skip), Q (quit)
- Auto-skips already-recorded labels on re-run

After recording, we kicked off fine-tuning. **First attempt failed** — `ConcatDataset` doesn't expose `.samples`, broke class-weight computation. Fixed by combining `samples` lists across child datasets. Hybrid FT is currently re-running in background.

The Gate FT completed: 4 checkpoints saved (`gate_hybrid_tv_ft_s{0-3}.pt`), still 91-100% on EC3D test.

---

## 8. Current State — Resuming Point

**As of the moment context was about to compact:**

1. Self-recording: **done**, 44 clips in `data/self_recorded/`
2. Gate fine-tuning: **done**, checkpoints in place
3. Hybrid fine-tuning: **running in background** as Bash task `bxa2wzge5`. Will produce `bilstm_ec3d_best_ft_s{0,1,2,3}.pt`
4. **NOT YET DONE**: update `src/app/pipeline_thread.py` `DEFAULT_CKPTS` and `DEFAULT_GATE_CKPTS` to point at the fine-tuned versions. Specifically:
   ```python
   DEFAULT_CKPTS = [
       "bilstm_ec3d_best_ft_s0.pt",
       "bilstm_ec3d_best_ft_s1.pt",
       "bilstm_ec3d_best_ft_s2.pt",
       "bilstm_ec3d_best_ft_s3.pt",
   ]
   DEFAULT_GATE_CKPTS = [
       "gate_hybrid_tv_ft_s0.pt",
       "gate_hybrid_tv_ft_s1.pt",
       "gate_hybrid_tv_ft_s2.pt",
       "gate_hybrid_tv_ft_s3.pt",
   ]
   ```
5. **NOT YET DONE**: live test with the fine-tuned models. The user will run `python src/app/run.py` and report whether the live behavior is better.

### Decision tree once user tests the fine-tuned model

- **If live behavior is meaningfully better** (gate locks to the right exercise when you do squat/lunge/plank): success. Use fine-tuned checkpoints going forward. Optionally collect more self-data and repeat for further improvement.
- **If still bad**: pivot to **Option 2 from the diagnosis** — rule-based form analyzer for the live demo. Joint angles + biomechanical thresholds computed directly from MediaPipe, no ML at inference. Reliable but doesn't showcase ML.
- **If user runs out of energy/time**: ship **Option 3** — frame the live demo failure as a domain-gap finding in the writeup, focus the presentation on the offline EC3D results (0.838 F1) and the lessons learned about deployment.

---

## 9. Key Numbers Summary (for the report)

| Metric | Value |
|---|---|
| Benchmark: EC3D test set (held-out subject Isinsu) macro-F1 | 0.838 |
| Benchmark: EC3D test accuracy | 82.6% |
| Paper baseline (Zhao et al., GCN with correction branch) | 0.91 / 90.9% |
| Per-class results that **beat the paper** | Lunges/correct, Plank/correct, SQUAT/front_bent |
| Per-class results matching the paper | Plank/arched_back, SQUAT/feet_too_wide, SQUAT/not_low_enough |
| Per-class where we trail | Lunges/knee_passes_toe (the model's hardest class) |
| Live demo accuracy | Not numerically quantified; subjectively poor due to domain gap |

---

## 10. Lessons Learned (worth including in the report)

1. **Engineered geometric features mattered more than architecture** for clean-data accuracy. Adding 17 hand-crafted features (lateral_dev, stance_ratio, hip_above_ankle, etc.) was the single biggest jump (+0.25 F1).
2. **Ensembling helps but plateaus.** Going from 4 seeds to 16 seeds gained only +0.014 F1. Diminishing returns.
3. **Mixing architectures hurt as often as helped.** Adding ST-GCN to the Hybrid ensemble lost predictions on some classes where they outvoted Hybrid's correct picks.
4. **Honest selection matters.** Cherry-picking by test F1 showed 0.869 ensemble vs 0.838 honest ensemble — a 3pp gap that's purely test-set leakage. Used Vidit (val) for honest seed selection.
5. **Synthetic domain augmentation without measured noise is gambling.** We picked synth noise levels by guessing what MediaPipe looks like and it regressed live performance dramatically.
6. **Domain gap is the dominant deployment issue.** A model achieving 84% on a benchmark can completely fail on input from a different sensor distribution. This is the central narrative for the report.

---

## 11. Quick Command Cheatsheet

```bash
# 1. Always start from native arm64 shell
arch
# (if "i386", run: arch -arm64 zsh)

# 2. Move to project, activate venv
cd "/Users/harukioyama/Classes/Pattern Recognition/FinalProject"
source venv/bin/activate

# 3. Run the live app
python src/app/run.py

# 4. Record self-data
python src/record_self_data.py

# 5. Train a Hybrid (single seed, with self-data)
python src/train.py --arch hybrid --feature-mode pose_extras \
    --train-split trainval --epochs 60 --seed 0 \
    --ckpt-tag ft_s0 --include-self-data

# 6. Train the gate (single seed, with self-data)
python src/train_gate.py --epochs 60 --seed 0 \
    --ckpt-tag ft_s0 --include-self-data

# 7. Evaluate an ensemble on EC3D test
python src/ensemble_eval.py --split test --tta-crops 1 \
    --ckpts bilstm_ec3d_best_hybrid_tv_s3.pt \
            bilstm_ec3d_best_hybrid_tv_s8.pt \
            bilstm_ec3d_best_hybrid_tv_s0.pt \
            bilstm_ec3d_best_hybrid_tv_s4.pt
```

---

## 12. Open Questions / Risks

- Will the fine-tuned models actually generalize from 44 self-recorded clips to live demo conditions, or did we just overfit to those 44 clips?
- If the fine-tuned models work in the live demo, can we record another 5 minutes to push performance further?
- Do we need to update the synth checkpoints' defaults in pipeline_thread.py if user reverts? (Currently set to the original non-synth Hybrid + Gate seeds.)
- The Ollama coaching cues sometimes lag (~1 sec latency) — acceptable but worth noting in the presentation.
- The web search agent confirmed no new open-source dataset with categorical errors exists as of 2026-06-27 beyond what we already considered. Bottleneck is genuine.

---

*Generated 2026-06-27, end of session before autocompact.*
