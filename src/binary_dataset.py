"""Unified binary correct/incorrect dataset combining EC3D and UI-PRMD.

Both are mapped to a common 17-joint Human3.6M layout, temporally resampled
to a fixed window, and per-sequence scale-normalised so the model sees
comparable poses across datasets.

EC3D labels: instruction code "1" -> correct, all other codes -> incorrect.
UI-PRMD labels: position 2 of label tuple (1 = correct, 0 = incorrect).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ec3d_dataset import TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS, load_sequences
from joint_mapping import N_H36M, body25_to_h36m

PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed_extracted"


def _resample(seq: np.ndarray, target_T: int) -> np.ndarray:
    """Linear-interpolate (T, J, C) along time to target_T frames."""
    T = seq.shape[0]
    if T == target_T:
        return seq
    idx = np.linspace(0, T - 1, target_T)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    frac = (idx - lo)[:, None, None]
    return ((1.0 - frac) * seq[lo] + frac * seq[hi]).astype(np.float32)


def _scale_normalise(seq: np.ndarray) -> np.ndarray:
    """Centre on joint 0 (Pelvis) and divide by per-sequence max magnitude.

    Robust to inter-dataset scale differences without depending on derived
    joints (Spine, Head) that we interpolated for BODY_25 -> H3.6M.
    """
    centred = seq - seq[:, 0:1, :]
    scale = np.max(np.linalg.norm(centred, axis=-1)) + 1e-6
    return (centred / scale).astype(np.float32)


@dataclass
class BinarySample:
    pose: np.ndarray  # (T, 17, 3) float32, normalised
    is_correct: int   # 1 if correct form, 0 if incorrect
    source: str       # "EC3D" or "UI-PRMD"
    subject: str
    exercise: str


def load_ec3d_binary(window: int) -> tuple[list[BinarySample], list[BinarySample], list[BinarySample]]:
    """Returns (train, val, test) lists of BinarySamples from EC3D."""
    raw = load_sequences()
    splits = {"train": [], "val": [], "test": []}
    for s in raw:
        if s.subject in TRAIN_SUBJECTS:
            split = "train"
        elif s.subject in VAL_SUBJECTS:
            split = "val"
        elif s.subject in TEST_SUBJECTS:
            split = "test"
        else:
            continue
        # EC3D instruction code "1" = correct, anything else = incorrect.
        is_correct = 1 if s.instruction_code == "1" else 0
        pose17 = body25_to_h36m(s.frames)                 # (T, 17, 3)
        pose17 = _resample(pose17, window)                # (window, 17, 3)
        pose17 = _scale_normalise(pose17)
        splits[split].append(
            BinarySample(pose17, is_correct, "EC3D", s.subject, s.exercise)
        )
    return splits["train"], splits["val"], splits["test"]


def load_uiprmd_binary(window: int) -> tuple[list[BinarySample], list[BinarySample]]:
    """Returns (train, val) lists from UI-PRMD preprocessed bundle.

    Source tensor layout: (N, C=3, T=104, V=17, M=1). We transpose to
    (T, V, C) per sample and resample to `window`.
    """
    out: dict[str, list[BinarySample]] = {"train": [], "val": []}
    for split in ("train", "val"):
        arr = np.load(PROC_DIR / "xsub" / f"seg_data_joint_{split}.npy")
        with (PROC_DIR / "xsub" / f"seg_label_{split}.pkl").open("rb") as f:
            names_list, label_list = pickle.load(f)
        for i, (name, lab) in enumerate(zip(names_list, label_list)):
            exercise_id, correct_flag = lab  # tuple
            pose = arr[i, ..., 0].transpose(1, 2, 0)        # (T, 17, 3)
            pose = _resample(pose, window)
            pose = _scale_normalise(pose)
            # name format: "001-DeepSquat-correct-R01" or "m001-DeepSquat-incorrect-R01"
            parts = name.split("-")
            subject = parts[0]
            exercise = parts[1]
            out[split].append(
                BinarySample(pose, int(correct_flag), "UI-PRMD", subject, exercise)
            )
    return out["train"], out["val"]


class UnifiedBinaryDataset(Dataset):
    """Yields (pose [T,17,3], is_correct, source_id) for a chosen split."""

    SOURCE_TO_ID = {"EC3D": 0, "UI-PRMD": 1}

    def __init__(self, samples: list[BinarySample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        pose = torch.from_numpy(s.pose)                          # (T, 17, 3)
        # Flatten (17, 3) to 51 features for our existing flat-feature trainer.
        flat = pose.reshape(pose.shape[0], -1)                   # (T, 51)
        return flat, torch.tensor(s.is_correct, dtype=torch.long), torch.tensor(self.SOURCE_TO_ID[s.source])


def build_unified_splits(window: int = 64):
    """Build the three combined splits.

    train  = EC3D train (Hugues+Sena) + UI-PRMD train (subjects 1-9 + mirrors)
    val    = EC3D val (Vidit) + UI-PRMD val (subject 10)
    test   = EC3D test (Isinsu)   -- pure EC3D test, no leakage from UI-PRMD
    """
    ec3d_train, ec3d_val, ec3d_test = load_ec3d_binary(window)
    ui_train, ui_val = load_uiprmd_binary(window)

    train = ec3d_train + ui_train
    val = ec3d_val + ui_val
    test = ec3d_test  # only EC3D, this is what we ultimately care about
    return train, val, test


if __name__ == "__main__":
    train, val, test = build_unified_splits(window=64)

    def _summary(name: str, samples: list[BinarySample]) -> None:
        from collections import Counter
        srcs = Counter(s.source for s in samples)
        ex = Counter(s.exercise for s in samples)
        balance = Counter(s.is_correct for s in samples)
        print(f"\n{name}: N={len(samples)}")
        print(f"  source : {dict(srcs)}")
        print(f"  exercises (top 10): {dict(list(ex.most_common(10)))}")
        print(f"  binary : {dict(balance)} (1=correct, 0=incorrect)")

    _summary("train", train)
    _summary("val", val)
    _summary("test", test)
    pose, lab, src = UnifiedBinaryDataset(train)[0]
    print(f"\nsample shape : {tuple(pose.shape)}  label={int(lab)}  source={int(src)}")
