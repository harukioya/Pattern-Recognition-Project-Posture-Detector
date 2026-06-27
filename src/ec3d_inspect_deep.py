"""Deeper EC3D inspection: dump label vocabulary, sample rows, and one full nested branch.

Goal: figure out (a) what the 5 label columns mean, (b) how sequences are structured
inside data.pickle, (c) the joint topology in the 25-node skeleton.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from collections import Counter

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def inspect_data_3d() -> None:
    with (DATA_DIR / "data_3D.pickle").open("rb") as f:
        d = pickle.load(f)

    labels = d["labels"]
    poses = d["poses"]
    print(f"labels: shape={labels.shape}, dtype={labels.dtype}")
    print(f"poses : shape={poses.shape}, dtype={poses.dtype}")

    print("\nfirst 10 label rows:")
    for row in labels[:10]:
        print(f"  {list(row)}")

    print("\nlast 10 label rows:")
    for row in labels[-10:]:
        print(f"  {list(row)}")

    for col in range(labels.shape[1]):
        vals = Counter(labels[:, col].tolist())
        print(f"\ncol {col}: {len(vals)} unique  -> {dict(list(vals.most_common())[:20])}")

    print(f"\npose value range: min={poses.min():.3f} max={poses.max():.3f}")
    print(f"pose per-axis ranges:")
    for ax, name in enumerate("xyz"):
        print(f"  {name}: min={poses[:, ax, :].min():.3f} max={poses[:, ax, :].max():.3f}")


def inspect_data_full() -> None:
    with (DATA_DIR / "data.pickle").open("rb") as f:
        d = pickle.load(f)

    frames = d["frames"]
    params = d["params"]

    print("\n=== params (cameras) ===")
    cam_key = list(params.keys())[0]
    cam = params[cam_key]
    print(f"cam {cam_key} intrinsics: {type(cam['intrinsics'])}")
    if hasattr(cam["intrinsics"], "shape"):
        print(f"  shape={cam['intrinsics'].shape}")
        print(f"  {cam['intrinsics']}")
    else:
        print(f"  {cam['intrinsics']}")
    print(f"cam {cam_key} extrinsics: {type(cam['extrinsics'])}")
    if hasattr(cam["extrinsics"], "shape"):
        print(f"  shape={cam['extrinsics'].shape}")
        print(f"  {cam['extrinsics']}")
    else:
        print(f"  {cam['extrinsics']}")

    print("\n=== frames[SQUAT][Hugues] structure ===")
    branch = frames["SQUAT"]["Hugues"]
    print(f"type={type(branch).__name__}")
    if isinstance(branch, dict):
        print(f"keys: {list(branch.keys())[:20]}")
        first_key = list(branch.keys())[0]
        sub = branch[first_key]
        print(f"\n[{first_key!r}] -> {type(sub).__name__}")
        if isinstance(sub, dict):
            print(f"  keys: {list(sub.keys())}")
            for k, v in sub.items():
                if hasattr(v, "shape"):
                    print(f"  [{k!r}] ndarray shape={v.shape} dtype={v.dtype}")
                elif isinstance(v, dict):
                    print(f"  [{k!r}] dict({len(v)} keys): {list(v.keys())[:10]}")
                    for kk, vv in list(v.items())[:3]:
                        if hasattr(vv, "shape"):
                            print(f"      [{kk!r}] ndarray shape={vv.shape} dtype={vv.dtype}")
                        else:
                            print(f"      [{kk!r}] {type(vv).__name__}: {str(vv)[:80]}")
                else:
                    print(f"  [{k!r}] {type(v).__name__}: {str(v)[:120]}")
        elif hasattr(sub, "shape"):
            print(f"  ndarray shape={sub.shape} dtype={sub.dtype}")

    counts = {ex: {subj: len(frames[ex][subj]) if isinstance(frames[ex][subj], (dict, list)) else "n/a"
                   for subj in frames[ex]} for ex in frames}
    print("\n=== sequences per (exercise, subject) ===")
    for ex, by_subj in counts.items():
        print(f"  {ex:10s}: {by_subj}")


if __name__ == "__main__":
    print("=" * 60)
    print("data_3D.pickle")
    print("=" * 60)
    inspect_data_3d()
    print("\n" + "=" * 60)
    print("data.pickle")
    print("=" * 60)
    inspect_data_full()
