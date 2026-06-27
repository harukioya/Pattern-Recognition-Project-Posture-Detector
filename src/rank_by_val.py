"""Rank a list of checkpoints by their macro-F1 on the val subject (Vidit).

Used to honestly pick top-K seeds for ensembling without touching the test
subject (Isinsu).
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
    feature_dim,
    load_sequences,
)
from ensemble_eval import build_model, pick_device

CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


@torch.no_grad()
def eval_on(model, loader, device) -> tuple[float, float]:
    model.eval()
    all_p, all_t = [], []
    for x, _ex, mis in loader:
        x = x.to(device)
        preds = model(x).argmax(dim=-1).cpu().numpy()
        all_p.extend(preds.tolist())
        all_t.extend(mis.tolist())
    acc = float(np.mean(np.array(all_p) == np.array(all_t)))
    f1 = float(f1_score(all_t, all_p, average="macro", zero_division=0))
    return acc, f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", nargs="+", required=True)
    args = parser.parse_args()

    device = pick_device()
    seqs = load_sequences()
    n_classes = len(MISTAKE_LABELS)

    loaders: dict[tuple[str, int], DataLoader] = {}
    rows: list[tuple[str, float, float]] = []
    for name in args.ckpts:
        ckpt = torch.load(CKPT_DIR / name, map_location=device, weights_only=True)
        margs = ckpt["args"]
        key = (margs["feature_mode"], margs.get("window", 64))
        if key not in loaders:
            ds = EC3DSequenceDataset(seqs, mode="val", window=key[1], feature_mode=key[0])
            loaders[key] = DataLoader(ds, batch_size=32, shuffle=False)
        model = build_model(margs, n_classes, device)
        model.load_state_dict(ckpt["state_dict"])
        acc, f1 = eval_on(model, loaders[key], device)
        rows.append((name, acc, f1))

    rows.sort(key=lambda r: r[2], reverse=True)
    print(f"{'rank':>4}  {'val_F1':>6}  {'val_acc':>7}  ckpt")
    for i, (name, acc, f1) in enumerate(rows):
        print(f"{i:>4}  {f1:>6.3f}  {acc:>7.2%}  {name}")


if __name__ == "__main__":
    main()
