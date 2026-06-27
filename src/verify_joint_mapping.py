"""Sanity-check joint mapping by comparing bone lengths across the 3 datasets.

If our mapping is correct, the relative bone-length proportions (after
normalization) should be similar across datasets — e.g. hip-to-knee should
always be a sensible fraction of pelvis-to-thorax.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from joint_mapping import (
    H36M_JOINT_NAMES,
    body25_to_h36m,
    execheck_21_to_h36m,
    normalize_pose,
)
from ec3d_dataset import load_sequences

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed_extracted"

BONE_PAIRS = [
    ("Pelvis", "Spine"),
    ("Spine", "Thorax"),
    ("Thorax", "Nose"),
    ("RHip", "RKnee"),
    ("RKnee", "RAnkle"),
    ("LHip", "LKnee"),
    ("LKnee", "LAnkle"),
    ("RShoulder", "RElbow"),
    ("RElbow", "RWrist"),
    ("LShoulder", "LElbow"),
    ("LElbow", "LWrist"),
]


def bone_lengths(pose_seq: np.ndarray) -> dict[str, float]:
    """Mean bone length across time for each named bone pair."""
    out: dict[str, float] = {}
    name_to_idx = {n: i for i, n in enumerate(H36M_JOINT_NAMES)}
    for a, b in BONE_PAIRS:
        ia, ib = name_to_idx[a], name_to_idx[b]
        d = np.linalg.norm(pose_seq[..., ia, :] - pose_seq[..., ib, :], axis=-1)
        out[f"{a}-{b}"] = float(d.mean())
    return out


def main() -> None:
    # EC3D sample
    ec3d_seqs = load_sequences()
    sample = next(s for s in ec3d_seqs if s.exercise == "SQUAT")
    ec3d_h36m = body25_to_h36m(sample.frames)         # (T, 17, 3)
    ec3d_norm = normalize_pose(ec3d_h36m)
    print(f"EC3D sample      : exercise={sample.exercise} subject={sample.subject} T={ec3d_h36m.shape[0]}")
    print(f"  raw bone lengths (m):")
    for k, v in bone_lengths(ec3d_h36m).items():
        print(f"    {k:25s} {v:.3f}")

    # UI-PRMD sample
    ui = np.load(DATA_DIR / "xsub" / "seg_data_joint_train.npy")[0]  # (3, 104, 17, 1)
    ui = ui[..., 0].transpose(1, 2, 0)                                # (T, 17, 3)
    print(f"\nUI-PRMD sample   : T={ui.shape[0]}  joint count={ui.shape[1]}")
    print(f"  raw bone lengths (unitless):")
    for k, v in bone_lengths(ui).items():
        print(f"    {k:25s} {v:.3f}")

    # ExeChecker sample (use 17 of 21)
    ex = np.load(DATA_DIR / "xsub_v6" / "seg_data_joint_train.npy")[0]  # (3, 160, 21, 1)
    ex = ex[..., 0].transpose(1, 2, 0)                                  # (T, 21, 3)
    ex17 = execheck_21_to_h36m(ex)                                      # (T, 17, 3)
    print(f"\nExeChecker sample: T={ex17.shape[0]}  joints {ex.shape[1]} -> {ex17.shape[1]}")
    print(f"  raw bone lengths (unitless):")
    for k, v in bone_lengths(ex17).items():
        print(f"    {k:25s} {v:.3f}")

    # After normalization, ratios should look similar
    print(f"\n--- after normalize_pose (unit pelvis->thorax) ---")
    for name, p in [("EC3D", normalize_pose(ec3d_h36m)),
                    ("UI-PRMD", normalize_pose(ui)),
                    ("ExeChecker", normalize_pose(ex17))]:
        print(f"\n{name} normalised bone lengths:")
        for k, v in bone_lengths(p).items():
            print(f"  {k:25s} {v:.3f}")


if __name__ == "__main__":
    main()
