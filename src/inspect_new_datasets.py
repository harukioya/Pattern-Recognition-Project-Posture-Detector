"""Inspect the unified ExeChecker + UI-PRMD preprocessed bundle.

Concise summary: tensor shape, label class distribution, sample names per
class, joint count. We never print the full filename list.
"""
from __future__ import annotations

import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed_extracted"


def _parse_meta(name: str) -> tuple[str, str, str, str]:
    """Filename pattern: 'SUBJECT-exercise-correct/incorrect-Rn' e.g. '0-arm_circle-correct-R1'."""
    parts = name.split("-")
    if len(parts) >= 4:
        return parts[0], parts[1], parts[2], parts[3]
    return ("?", name, "?", "?")


def inspect(folder: Path, name: str) -> None:
    print(f"\n{'=' * 60}\n{name}  ({folder.name})\n{'=' * 60}")
    for split in ("train", "val"):
        npy_path = folder / f"seg_data_joint_{split}.npy"
        pkl_path = folder / f"seg_label_{split}.pkl"
        if not (npy_path.exists() and pkl_path.exists()):
            continue
        arr = np.load(npy_path)
        with pkl_path.open("rb") as f:
            labels = pickle.load(f)

        print(f"\n  --- {split} ---")
        print(f"  data shape : {arr.shape}  dtype={arr.dtype}")
        print(f"  data range : [{arr.min():.3f}, {arr.max():.3f}]  mean={arr.mean():.3f}")

        names_list, label_list = labels  # tuple(list_of_names, list_of_ints)
        assert len(names_list) == arr.shape[0]
        print(f"  N={len(names_list)}, label vocab: {sorted(set(label_list))}")

        # Class distribution
        ctr = Counter(label_list)
        print(f"  class counts: {dict(sorted(ctr.items()))}")

        # Subject distribution (first column of name)
        subjects = Counter(_parse_meta(n)[0] for n in names_list)
        print(f"  subjects   : {dict(sorted(subjects.items()))}")

        # Exercise distribution
        exercises = Counter(_parse_meta(n)[1] for n in names_list)
        print(f"  exercises  : {dict(sorted(exercises.items()))}")

        # Correct/incorrect split
        cor = Counter(_parse_meta(n)[2] for n in names_list)
        print(f"  cor/incor  : {dict(sorted(cor.items()))}")

        # Show one sample name per unique label tuple
        seen: dict[tuple, str] = {}
        for nm, lab in zip(names_list, label_list):
            key = tuple(lab) if isinstance(lab, (list, tuple)) else (lab,)
            seen.setdefault(key, nm)
        print("  label -> sample name (first per unique label):")
        for c in sorted(seen.keys()):
            print(f"    {c}: {seen[c]!r}")


def main() -> None:
    inspect(DATA_DIR / "xsub_v6", "ExeChecker (xsub_v6)")
    inspect(DATA_DIR / "xsub", "UI-PRMD (xsub)")


if __name__ == "__main__":
    main()
