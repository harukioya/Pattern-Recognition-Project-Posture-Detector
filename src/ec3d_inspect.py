"""Inspect EC3D pose-data pickles to understand the schema before writing a loader.

Run after downloading data_3D.pickle and data.pickle into ../data/.
Prints top-level keys, array shapes, dtypes, label vocabulary, and a sample.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def summarize(obj, depth: int = 0, max_depth: int = 4) -> None:
    pad = "  " * depth
    if depth > max_depth:
        print(f"{pad}<truncated>")
        return
    if isinstance(obj, dict):
        print(f"{pad}dict({len(obj)} keys): {list(obj.keys())[:10]}")
        for k, v in list(obj.items())[:5]:
            print(f"{pad}  [{k!r}] ->")
            summarize(v, depth + 2, max_depth)
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{type(obj).__name__}(len={len(obj)})")
        if obj:
            summarize(obj[0], depth + 1, max_depth)
    elif isinstance(obj, np.ndarray):
        print(f"{pad}ndarray shape={obj.shape} dtype={obj.dtype}")
    else:
        rep = repr(obj)
        if len(rep) > 100:
            rep = rep[:100] + "..."
        print(f"{pad}{type(obj).__name__}: {rep}")


def main() -> None:
    candidates = sorted(DATA_DIR.glob("*.pickle")) + sorted(DATA_DIR.glob("*.pkl"))
    if not candidates:
        print(f"No .pickle files in {DATA_DIR}. Download EC3D first.", file=sys.stderr)
        sys.exit(1)
    for p in candidates:
        print(f"\n=== {p.name} ({p.stat().st_size / 1e6:.1f} MB) ===")
        with p.open("rb") as f:
            obj = pickle.load(f)
        summarize(obj)


if __name__ == "__main__":
    main()
