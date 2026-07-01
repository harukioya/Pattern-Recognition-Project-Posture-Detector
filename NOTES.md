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
