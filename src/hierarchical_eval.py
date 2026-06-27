"""Two-stage hierarchical evaluator.

Stage 1: binary correct/incorrect classifier (BinarySTGCNHead trained on
         EC3D+UI-PRMD with H3.6M-17 joints).
Stage 2: existing 11-class Hybrid ensemble (trained on EC3D BODY_25 features).
         Runs only when Stage 1 predicts "incorrect".

The hierarchical output is the 11-class prediction matching EC3D's MISTAKE_LABELS.
When Stage 1 says "correct", we predict the matching "{exercise}/correct" class.

We need the predicted exercise to map "correct" -> the right one of three correct
labels. Use a simple per-window pose-based heuristic: the test clip already has
an exercise label, so for evaluation we just use the ground-truth exercise when
constructing the correct-class. (Reasonable since at inference time the user
selects which exercise they are doing.)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from binary_dataset import build_unified_splits, UnifiedBinaryDataset
from ec3d_dataset import (
    EC3DSequenceDataset,
    EXERCISES,
    MISTAKE_LABELS,
    feature_dim,
    load_sequences,
)
from ensemble_eval import build_model
from model import BinarySTGCNHead

CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
EXERCISE_TO_CORRECT_ID = {
    "SQUAT": MISTAKE_LABELS.index("SQUAT/correct"),
    "Lunges": MISTAKE_LABELS.index("Lunges/correct"),
    "Plank": MISTAKE_LABELS.index("Plank/correct"),
}


def pick_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


@torch.no_grad()
def stage1_probs(model: torch.nn.Module, samples, device, batch=64) -> np.ndarray:
    """Returns (N, 2) softmax over [incorrect, correct] in order of samples list."""
    model.eval()
    loader = DataLoader(UnifiedBinaryDataset(samples), batch_size=batch, shuffle=False)
    out = []
    for x, _y, _src in loader:
        x = x.to(device)
        probs = torch.softmax(model(x), dim=-1).cpu().numpy()
        out.append(probs)
    return np.concatenate(out, axis=0)


@torch.no_grad()
def stage2_probs(ckpts: list[str], device, batch=32) -> tuple[np.ndarray, np.ndarray]:
    """Returns (avg_probs, targets) for the EC3D test set across the ensemble."""
    seqs = load_sequences()
    n_classes = len(MISTAKE_LABELS)
    probs_sum = None
    targets_ref = None
    loaders_by_feat: dict[tuple[str, int], DataLoader] = {}
    for name in ckpts:
        ckpt = torch.load(CKPT_DIR / name, map_location=device, weights_only=True)
        margs = ckpt["args"]
        key = (margs["feature_mode"], margs.get("window", 64))
        if key not in loaders_by_feat:
            ds = EC3DSequenceDataset(seqs, mode="test", window=key[1], feature_mode=key[0])
            loaders_by_feat[key] = DataLoader(ds, batch_size=batch, shuffle=False)
        model = build_model(margs, n_classes, device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        all_probs, all_t = [], []
        for x, _ex, mis in loaders_by_feat[key]:
            x = x.to(device)
            probs = torch.softmax(model(x), dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_t.append(mis.numpy())
        probs = np.concatenate(all_probs, axis=0)
        targets = np.concatenate(all_t, axis=0)
        if probs_sum is None:
            probs_sum = np.zeros_like(probs)
            targets_ref = targets
        elif not np.array_equal(targets, targets_ref):
            raise RuntimeError("target order mismatch in stage 2 ensemble")
        probs_sum += probs
    return probs_sum / len(ckpts), targets_ref


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary-ckpt", type=str, required=True,
                        help="filename of the binary classifier checkpoint")
    parser.add_argument("--stage2-ckpts", nargs="+", required=True,
                        help="11-class ensemble checkpoints (Hybrid models)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="P(incorrect) threshold to route to stage 2")
    args = parser.parse_args()

    device = pick_device()
    print(f"device: {device}")

    # ----- Stage 2: build per-clip 11-class predictions from existing ensemble
    s2_probs_clip, s2_targets_clip = stage2_probs(args.stage2_ckpts, device)
    print(f"stage 2 ensemble: {len(args.stage2_ckpts)} models, "
          f"per-clip preds: {s2_probs_clip.shape}, targets: {s2_targets_clip.shape}")

    # ----- Stage 1: predict per EC3D test sequence (one prediction per rep)
    binary_ckpt = torch.load(CKPT_DIR / args.binary_ckpt, map_location=device, weights_only=True)
    binary_model = BinarySTGCNHead(
        hidden=binary_ckpt["args"].get("hidden", 64),
        dropout=binary_ckpt["args"].get("dropout", 0.4),
    ).to(device)
    binary_model.load_state_dict(binary_ckpt["state_dict"])

    _train, _val, test_samples = build_unified_splits(window=binary_ckpt["args"].get("window", 64))
    s1_probs = stage1_probs(binary_model, test_samples, device)
    print(f"stage 1 model: per-sequence preds: {s1_probs.shape}, threshold P(incorrect)>{args.threshold}")

    # ----- Reconcile per-sequence (Stage 1) with per-clip (Stage 2)
    # build_unified_splits returns sequences (one BinarySample per EC3D rep).
    # Stage 2 produces predictions per clip. We need to evaluate at the per-clip
    # granularity matching the Hybrid ensemble. Map each Stage-2 clip back to
    # its source sequence by scanning sequence order — both pipelines walk EC3D
    # in the same load_sequences() order, so indices align by sequence.
    #
    # Easier: rebuild a per-clip stage 1 by repeating the per-sequence stage 1
    # probability across each clip belonging to that sequence.
    seqs = load_sequences()
    test_seqs = [s for s in seqs if s.subject in ("Isinsu",)]
    # Confirm count matches
    assert len(test_seqs) == len(test_samples), (
        f"sequence-count mismatch: {len(test_seqs)} vs {len(test_samples)}"
    )

    # Match stage 2 per-clip index back to its source sequence and exercise.
    # EC3DSequenceDataset(window=64) builds clips in the same order as load_sequences().
    seq_to_n_clips: list[int] = []
    for s in test_seqs:
        T = s.frames.shape[0]
        if T < 64:
            seq_to_n_clips.append(1)
        else:
            seq_to_n_clips.append(1 + (T - 64) // 64)
    assert sum(seq_to_n_clips) == s2_probs_clip.shape[0], (
        f"clip count mismatch: sum={sum(seq_to_n_clips)} vs {s2_probs_clip.shape[0]}"
    )

    # Expand stage 1 per-sequence -> per-clip
    s1_probs_clip = np.repeat(s1_probs, seq_to_n_clips, axis=0)
    # Also repeat the sequence's exercise for routing the "correct" branch.
    exercise_per_clip = np.repeat([s.exercise for s in test_seqs], seq_to_n_clips)

    # ----- Hierarchical prediction (hard routing)
    is_incorrect = s1_probs_clip[:, 0] > args.threshold  # column 0 = incorrect
    hier_preds = np.empty(s2_probs_clip.shape[0], dtype=np.int64)
    s2_argmax = s2_probs_clip.argmax(axis=-1)
    for i in range(s2_probs_clip.shape[0]):
        if is_incorrect[i]:
            hier_preds[i] = s2_argmax[i]
        else:
            hier_preds[i] = EXERCISE_TO_CORRECT_ID[exercise_per_clip[i]]

    # ----- Soft fusion (multiplicative re-weighting)
    # Each class c is either a "correct" class or an "incorrect" class.
    # Multiply P_stage2(c) by Stage 1's matching probability, then argmax.
    correct_class_mask = np.zeros(len(MISTAKE_LABELS), dtype=bool)
    for cid in EXERCISE_TO_CORRECT_ID.values():
        correct_class_mask[cid] = True
    p_correct = s1_probs_clip[:, 1:2]    # (N, 1)
    p_incorrect = s1_probs_clip[:, 0:1]  # (N, 1)
    fusion_weights = np.where(correct_class_mask[None, :], p_correct, p_incorrect)
    soft_probs = s2_probs_clip * fusion_weights
    soft_probs /= soft_probs.sum(axis=-1, keepdims=True).clip(min=1e-9)
    soft_preds = soft_probs.argmax(axis=-1)

    # ----- Evaluate
    def _report(preds: np.ndarray, header: str) -> None:
        correct = np.zeros(len(MISTAKE_LABELS), dtype=np.int64)
        total = np.zeros(len(MISTAKE_LABELS), dtype=np.int64)
        for p, t in zip(preds, s2_targets_clip):
            total[t] += 1
            if p == t:
                correct[t] += 1
        print(f"\n--- {header} ---")
        for c in range(len(MISTAKE_LABELS)):
            if total[c] == 0:
                continue
            print(f"  {MISTAKE_LABELS[c]:30s} {correct[c]:3d}/{total[c]:3d}  ({correct[c] / total[c]:.2%})")
        acc = sum(correct) / max(sum(total), 1)
        f1 = f1_score(s2_targets_clip, preds, average="macro", zero_division=0)
        print(f"  ---> overall acc: {acc:.2%}, macro-F1: {f1:.3f}")

    _report(s2_argmax, "Stage 2 only (baseline = current single-stage ensemble)")
    _report(hier_preds, f"Hierarchical hard-routing (P(incorrect)>{args.threshold})")
    _report(soft_preds, "Hierarchical soft fusion (multiplicative re-weighting)")

    # Diagnostic
    routed_to_s2 = int(is_incorrect.sum())
    print(f"\nstage-1 routing: {routed_to_s2}/{len(is_incorrect)} clips -> stage 2 (incorrect)")
    print(f"                 {len(is_incorrect) - routed_to_s2}/{len(is_incorrect)} clips -> directly predicted as correct")


if __name__ == "__main__":
    main()
