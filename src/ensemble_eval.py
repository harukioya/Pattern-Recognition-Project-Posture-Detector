"""Ensemble several trained checkpoints with optional weights and TTA.

Loads checkpoints by name from FinalProject/checkpoints/, infers each model's
architecture and feature_mode from the saved args, and evaluates on the split.

--weights: per-model scalar weights for averaging softmax probabilities.
           Defaults to uniform. Must match the number of --ckpts.
--tta-crops: number of evenly-spaced temporal crops per sequence to average
             over. 1 = no TTA (default).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from ec3d_dataset import (
    EC3DSequenceDataset,
    MISTAKE_LABELS,
    extract_features,
    feature_dim,
    load_sequences,
)
from model import (
    BiLSTMHead,
    HybridSTGCNHead,
    HybridSTGCNv2Head,
    STGCNHead,
    STGCNv2Head,
    TransformerHead,
)

CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
SUBJECT_BY_SPLIT = {"val": ("Vidit",), "test": ("Isinsu",)}


def pick_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def build_model(args: dict, n_classes: int, device: torch.device):
    fm = args["feature_mode"]
    arch = args.get("arch", "bilstm")
    if arch == "bilstm":
        m = BiLSTMHead(feature_dim(fm), n_classes, hidden=args.get("hidden", 64), dropout=args.get("dropout", 0.3))
    elif arch == "transformer":
        m = TransformerHead(
            feature_dim(fm), n_classes,
            d_model=args.get("d_model", 128),
            n_heads=args.get("n_heads", 4),
            num_layers=args.get("n_layers", 2),
            dropout=args.get("dropout", 0.3),
            max_len=args.get("window", 64),
        )
    elif arch == "stgcn":
        m = STGCNHead(
            feature_dim(fm), n_classes,
            hidden=args.get("stgcn_hidden", 64),
            num_layers=args.get("stgcn_layers", 3),
            dropout=args.get("dropout", 0.4),
            temporal_kernel=args.get("stgcn_temporal_kernel", 9),
        )
    elif arch == "hybrid":
        m = HybridSTGCNHead(
            n_pose_features=75,
            n_extra_features=feature_dim(fm) - 75,
            n_classes=n_classes,
            stgcn_hidden=args.get("stgcn_hidden", 64),
            stgcn_layers=args.get("stgcn_layers", 3),
            dropout=args.get("dropout", 0.4),
            temporal_kernel=args.get("stgcn_temporal_kernel", 9),
        )
    elif arch == "stgcn_v2":
        m = STGCNv2Head(
            feature_dim(fm), n_classes,
            hidden=args.get("stgcn_hidden", 64),
            num_layers=args.get("stgcn_layers", 3),
            dropout=args.get("dropout", 0.4),
            temporal_kernel=args.get("stgcn_temporal_kernel", 9),
        )
    elif arch == "hybrid_v2":
        m = HybridSTGCNv2Head(
            n_pose_features=75,
            n_extra_features=feature_dim(fm) - 75,
            n_classes=n_classes,
            stgcn_hidden=args.get("stgcn_hidden", 64),
            stgcn_layers=args.get("stgcn_layers", 3),
            dropout=args.get("dropout", 0.4),
            temporal_kernel=args.get("stgcn_temporal_kernel", 9),
        )
    else:
        raise ValueError(f"unknown arch: {arch}")
    return m.to(device)


def build_tta_samples(sequences, subject_filter: tuple[str, ...], feature_mode: str,
                      window: int, n_crops: int):
    """Yields (clip_list_per_sequence, exercise_id, mistake_id) using n_crops temporal offsets."""
    out = []
    for s in sequences:
        if s.subject not in subject_filter:
            continue
        feats = extract_features(s.frames, feature_mode)
        T = feats.shape[0]
        if T <= window:
            pad = np.zeros((window - T, feats.shape[1]), dtype=np.float32)
            clips = [np.concatenate([feats, pad], axis=0)]
        else:
            if n_crops == 1:
                starts = [0]
            else:
                starts = np.linspace(0, T - window, n_crops, dtype=int).tolist()
            clips = [feats[start : start + window] for start in starts]
        out.append((clips, s.exercise_id, s.mistake_id))
    return out


@torch.no_grad()
def predict_tta(model, samples_for_feat, device, batch=32):
    """For each sample, average softmax probs across its temporal crops.

    Returns (N, C) probs in the order matching samples_for_feat.
    """
    model.eval()
    out = np.zeros((len(samples_for_feat), len(MISTAKE_LABELS)), dtype=np.float32)
    flat_clips: list[np.ndarray] = []
    flat_sample_idx: list[int] = []
    for i, (clips, _ex, _mis) in enumerate(samples_for_feat):
        for clip in clips:
            flat_clips.append(clip)
            flat_sample_idx.append(i)

    counts = np.zeros(len(samples_for_feat), dtype=np.int64)
    for start in range(0, len(flat_clips), batch):
        chunk = flat_clips[start : start + batch]
        idxs = flat_sample_idx[start : start + batch]
        x = torch.from_numpy(np.stack(chunk)).to(device)
        probs = torch.softmax(model(x), dim=-1).cpu().numpy()
        for k, i in enumerate(idxs):
            out[i] += probs[k]
            counts[i] += 1
    out = out / counts[:, None].clip(min=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="per-model weight; defaults to uniform")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--tta-crops", type=int, default=1,
                        help="number of temporal crops to average per sequence")
    args = parser.parse_args()

    device = pick_device()
    print(f"device: {device}  tta_crops: {args.tta_crops}")
    seqs = load_sequences()
    n_classes = len(MISTAKE_LABELS)

    if args.weights is None:
        weights = np.ones(len(args.ckpts), dtype=np.float32)
    else:
        if len(args.weights) != len(args.ckpts):
            raise SystemExit("--weights count must match --ckpts count")
        weights = np.array(args.weights, dtype=np.float32)
    weights = weights / weights.sum()
    print(f"weights: {weights.tolist()}")

    targets_ref = None
    probs_sum = None
    samples_cache: dict[tuple[str, int], list] = {}

    for ckpt_name, w in zip(args.ckpts, weights):
        path = CKPT_DIR / ckpt_name
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model_args = ckpt["args"]
        fm = model_args["feature_mode"]
        win = model_args.get("window", 64)
        key = (fm, win)
        if key not in samples_cache:
            samples_cache[key] = build_tta_samples(
                seqs, SUBJECT_BY_SPLIT[args.split], fm, win, args.tta_crops
            )
        samples = samples_cache[key]

        model = build_model(model_args, n_classes, device)
        model.load_state_dict(ckpt["state_dict"])
        probs = predict_tta(model, samples, device)
        targets = np.array([s[2] for s in samples])

        if targets_ref is None:
            targets_ref = targets
            probs_sum = np.zeros_like(probs)
        elif not np.array_equal(targets, targets_ref):
            raise RuntimeError("target order mismatch")
        probs_sum += w * probs
        print(f"  loaded {ckpt_name:50s}  w={w:.3f}  arch={model_args.get('arch','bilstm'):11s}  feat={fm}")

    preds = probs_sum.argmax(axis=-1)

    correct = np.zeros(n_classes, dtype=np.int64)
    total = np.zeros(n_classes, dtype=np.int64)
    for p, t in zip(preds, targets_ref):
        total[t] += 1
        if p == t:
            correct[t] += 1
    print(f"\n--- ensemble on {args.split} ({len(args.ckpts)} models, tta={args.tta_crops}) ---")
    for c in range(n_classes):
        if total[c] == 0:
            continue
        print(f"  {MISTAKE_LABELS[c]:30s} {correct[c]:3d}/{total[c]:3d}  ({correct[c] / total[c]:.2%})")
    acc = sum(correct) / max(sum(total), 1)
    f1 = f1_score(targets_ref, preds, average="macro", zero_division=0)
    print(f"  ---> overall acc: {acc:.2%}, macro-F1: {f1:.3f}")


if __name__ == "__main__":
    main()
