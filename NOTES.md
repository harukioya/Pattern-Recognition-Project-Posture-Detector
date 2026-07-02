# Experiment — per-exercise modes

Branch: `experiment/per-exercise-modes`.
Motivation: on live MediaPipe input the 3-class gate ensemble misclassifies the
exercise (usually predicting Lunges regardless of what the user is doing),
which makes every downstream prediction wrong. The domain gap between EC3D's
4-camera 3D and MediaPipe's monocular pseudo-3D is the underlying reason. See
`NEXT_SESSION_PROMPT.md` for what was already tried and rejected.

This experiment replaces the auto-detect gate with an **explicit user pick** and
trains **specialist models** so each exercise only competes against its own
error classes.

## What changed

### Training

- `src/ec3d_dataset.py`: `EC3DSequenceDataset` now accepts
  `exercise_filter: str | None`. When set, sequences are filtered to that
  exercise and the returned `mistake_id` is a **local** id inside a
  per-exercise label space. `SQUAT/squat_extra` is dropped so the SQUAT model
  has exactly 5 outputs; Lunges and Plank keep their 3 each.
  Helpers: `local_labels_for_exercise`, `global_to_local_id`,
  `local_to_global_id`.
- `src/self_data.py`: `SelfRecordedDataset` gains the same `exercise_filter`
  argument, using the same local-id convention so both sources are
  concat-compatible during per-exercise training.
- `src/train.py`: new `--exercise-filter {SQUAT,Lunges,Plank}` flag that plumbs
  through both datasets, resizes `n_classes` (5 or 3), and re-reports metrics
  in the local label space.

### Training run

Reproduced with:

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

Outputs (single-model, no ensembling) on Isinsu (EC3D test subject):

| Exercise | s0 | s1 | s2 | s3 | mean |
|----------|----|----|----|----|------|
| SQUAT  (5 classes) | 0.667 | 0.667 | 0.791 | 0.687 | ~0.70 |
| Lunges (3 classes) | 0.682 | 0.695 | 0.623 | 0.697 | ~0.67 |
| Plank  (3 classes) | 0.943 | 1.000 | 0.963 | 1.000 | ~0.98 |

Plank essentially saturates. SQUAT and Lunges are in the same range as the
prior 12-class specialists — the class space is smaller but so is the
per-class training set. The app-level effect (below) is the real gain.

### App

- `src/app/state.py`: `Prediction` gains `selected_mode: str | None`.
  `Prediction.exercise` returns the selected mode when set, else the gate pick.
- `src/app/widgets/welcome_screen.py` (new): landing page with three
  `ModeCard`s. Click or press `S` / `L` / `P` to pick.
- `src/app/pipeline_thread.py`: rewritten around a required `mode` argument.
  Loads only that exercise's 4-model ensemble, skips the gate entirely, and
  scatters local probabilities back into the 11-class DISPLAY_CLASSES vector
  the widgets expect. Missing-checkpoint warnings surface at startup.
- `src/app/main_window.py`: `QStackedWidget` with welcome (index 0) and
  workout (index 1). PipelineThread is created lazily on mode selection and
  torn down cleanly on `H` / `Home` / `Esc` (return to welcome) or window
  close. `Q` still quits.
- `src/app/widgets/header_bar.py`: `set_mode` API. Header pins to the
  selected exercise; predictions only toggle white/grey to show
  confidence/uncertainty, they no longer overwrite the label.
- `src/app/widgets/bars_panel.py`: `set_mode` hides non-selected sections'
  rows, headers, and dividers so only the active exercise's bars are
  visible. Legacy gate-mode behaviour (all sections + gate percentages)
  is preserved when `set_mode(None)` is used.
- `src/app/widgets/verdict_panel.py`: `reset_for_mode` neutral state.
- `src/app/widgets/coach_panel.py`: `reset` clears predictions/cues on
  mode switch.

## Mode-selection UX

1. App launches on the welcome screen. Header reads "Choose your exercise";
   three cards below.
2. Click a card, or press its hotkey (`S`, `L`, `P`). MainWindow tears down
   any prior PipelineThread, creates a new one wired to the selected
   exercise, and swaps to the workout view.
3. In-workout: the header shows the mode name. Only that exercise's
   probability bars are rendered. Predictions run against the 4-seed
   specialist ensemble.
4. Press `H` (or `Home` / `Esc`) to stop the pipeline and return to the
   welcome screen. Selecting a different mode from there loads a fresh
   ensemble.
5. `Q` quits.

## What to verify manually

The smoke test (`QT_QPA_PLATFORM=offscreen`) confirmed:

- MainWindow constructs; welcome screen shows 3 cards.
- Clicking a card loads all 4 ensemble models for that exercise and swaps
  to the workout view.
- Going home tears down the pipeline cleanly.

What still needs a live camera:

- Verify probability bars only show the selected exercise's classes.
- Compare qualitative live behaviour to the `main` branch: does the
  "everything is Lunges" failure mode go away? Are within-mode errors
  (e.g. `SQUAT/knees_inward` when knees collapse in) directionally
  correct?
- Verify hotkey navigation feels right during actual use.

## Deliverables status

- [x] Branch `experiment/per-exercise-modes` created.
- [x] Training-side changes (dataset + train.py).
- [x] 12 checkpoints under `checkpoints/` (gitignored).
- [x] App-side changes (welcome, pipeline, main window, widgets).
- [x] This NOTES.md.
- [ ] Live-camera comparison vs. `main` — pending user verification.
- [ ] **Not merged to `main`.** This is an experiment branch by design.

---

# Experiment — enrich all three exercises with YouTube correct-form videos

Goal: broaden what "correct" looks like for each exercise beyond EC3D's
captured subjects, using short correct-form clips sourced from YouTube —
additive only, EC3D data and loading logic untouched.

Why this should help: the gate-misclassification issue at the top of this
file was traced to the domain gap between EC3D's 4-camera 3D and MediaPipe's
monocular pseudo-3D. Videos run through `import_video_data.py` go through
the same MediaPipe monocular pipeline the live app uses at inference — so
this data sits in the *same domain* the model is actually evaluated in,
unlike EC3D itself.

Scope note: correct-only (no incorrect-form videos — too diverse to source
reliably) and mixed side/oblique camera angles. Fine for this purpose since
the goal is broadening what "correct" looks like, not teaching a new
decision boundary; `train.py`'s inverse-frequency `class_weights()`
(beta=0.5) keeps it from dominating the loss as it grows.

## Data

- 45 videos, 15 each for SQUAT / Lunges / Plank, under
  `data/correct-youtube/youtube-{squat,lunge,plank}/` (gitignored, not in
  git — mostly 5-60s clips, side/oblique view).
- Imported via `import_video_data.py --label <Exercise>/correct` →
  `data/self_recorded/*.pkl` (also gitignored). `import_video_data.py`
  drops any frame with no detected `pose_world_landmarks`, so bad frames
  are skipped, not corrupted in.
- Detection quality: 41/45 clips at 100% frame detection; worst case 90.3%
  (`squat14`). Nothing discarded.
- After 64-frame windowing (`SelfRecordedDataset`, non-overlapping):
  **89 windows** added to `SQUAT/correct`, **133** to `Lunges/correct`,
  **94** to `Plank/correct`.

## Integration

`--include-self-data` concatenates the imported `.pkl` data onto the
*EC3D train split only* (`train.py:207-216`) — val/test stay pure EC3D
(Vidit/Isinsu), so the results below are a clean before/after on data the
new videos never touched. Trained into a separately-tagged checkpoint set
(`pex_<ex>_enriched_s0..3`) so the original baseline (`pex_<ex>_s0..3`) is
untouched on disk.

Reproduced with:

```bash
for ex in SQUAT Lunges Plank; do
  lower=$(echo "$ex" | tr '[:upper:]' '[:lower:]')
  for seed in 0 1 2 3; do
    python src/train.py --arch hybrid --feature-mode pose_extras \
      --train-split trainval --epochs 60 --seed $seed \
      --ckpt-tag pex_${lower}_enriched_s${seed} \
      --exercise-filter $ex --include-self-data --quiet
  done
done
```

## Results (4-seed ensemble, Isinsu test split, macro-F1)

| Exercise | baseline | enriched | delta |
|---|---|---|---|
| SQUAT  | 0.759 | **0.817** | +0.058 |
| Lunges | **0.829** | 0.799 | −0.030 |
| Plank  | 0.963 | 1.000 | +0.037 |

- **SQUAT** — clean improvement. `SQUAT/correct` recall itself is
  unchanged (10/11 both); the gain is in `not_low_enough` (3/7 → 5/7), so
  the extra correct-only data isn't just inflating the class it enriched.
- **Lunges** — small regression: `not_low_enough` recall dropped
  (7/10 → 6/10), everything else unchanged. One flipped example, but a
  real (if small) dip — the risk flagged when this was planned.
- **Plank** — baseline was already near-ceiling on a tiny test set (23
  clips); the "gain" is one flipped example (`hunch_back` 9/10 → 10/10).
  Likely noise rather than a real signal either way.

## How to test both versions

The app picks which checkpoint set to load via the `PEX_VARIANT` env var
(`src/app/pipeline_thread.py`) — no file renaming needed, baseline and
enriched checkpoints can sit side by side in `checkpoints/`.

1. Drop the 12 `bilstm_ec3d_best_pex_{squat,lunges,plank}_enriched_s0..3.pt`
   files into `checkpoints/`, alongside the existing baseline ones.
2. Run the app:
   ```bash
   python src/app/run.py                        # baseline (default)
   PEX_VARIANT=enriched python src/app/run.py    # EC3D + YouTube
   ```
3. `PEX_VARIANT` is unset by default, so anyone who doesn't have the
   enriched checkpoints yet sees no behaviour change at all.


