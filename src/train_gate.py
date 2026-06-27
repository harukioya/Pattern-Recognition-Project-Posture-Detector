"""Train a 3-class exercise-gate classifier on EC3D.

Output classes: 0=SQUAT, 1=Lunges, 2=Plank (matching EXERCISE_TO_ID in
ec3d_dataset.py). Reuses the existing HybridSTGCN architecture and
feature pipeline (pose_extras), just swaps the head to 3 outputs.

We always train on trainval (Hugues+Sena+Vidit) so the gate sees as many
EC3D subjects as possible before facing live MediaPipe data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ec3d_dataset import (
    EC3DSequenceDataset,
    EXERCISES,
    feature_dim,
    load_sequences,
)
from model import HybridSTGCNHead

CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


class _ExerciseTargetDataset(Dataset):
    """Wraps an EC3DSequenceDataset to yield (clip, exercise_id)."""

    def __init__(self, base: EC3DSequenceDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        clip, ex_id, _mis_id = self.base[i]
        return clip, ex_id


def _device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _epoch(model, loader, criterion, optimizer, device):
    train = optimizer is not None
    model.train(train)
    total_loss, n = 0.0, 0
    all_p, all_t = [], []
    for x, y in loader:
        x = x.to(device)
        y_d = y.to(device)
        logits = model(x)
        loss = criterion(logits, y_d)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        all_p.extend(logits.argmax(dim=-1).cpu().tolist())
        all_t.extend(y.tolist())
    acc = float(np.mean(np.array(all_p) == np.array(all_t)))
    f1 = float(f1_score(all_t, all_p, average="macro", zero_division=0))
    return total_loss / max(n, 1), acc, f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--feature-mode", default="pose_extras")
    parser.add_argument("--stgcn-hidden", type=int, default=64)
    parser.add_argument("--stgcn-layers", type=int, default=3)
    parser.add_argument("--stgcn-temporal-kernel", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt-tag", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include-self-data", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _device()
    print(f"device: {device}")

    seqs = load_sequences()
    train_base = EC3DSequenceDataset(
        seqs, mode="trainval", window=args.window, feature_mode=args.feature_mode,
    )
    test_base = EC3DSequenceDataset(
        seqs, mode="test", window=args.window, feature_mode=args.feature_mode
    )

    train_for_loader = train_base
    if args.include_self_data:
        from torch.utils.data import ConcatDataset
        from self_data import SelfRecordedDataset
        self_ds = SelfRecordedDataset(window=args.window, feature_mode=args.feature_mode)
        print(self_ds.summary())
        if len(self_ds) > 0:
            class _GateWrap(Dataset):
                def __init__(self, base): self.base = base
                def __len__(self): return len(self.base)
                def __getitem__(self, i):
                    clip, ex_id, _mis_id = self.base[i]
                    return clip, ex_id
            train_for_loader = ConcatDataset([
                _ExerciseTargetDataset(train_base), _GateWrap(self_ds),
            ])

    # train_for_loader is a ConcatDataset when --include-self-data added self
    # samples, otherwise we wrap the base trainval dataset for gate targets.
    from torch.utils.data import ConcatDataset as _ConcatDataset
    if isinstance(train_for_loader, _ConcatDataset):
        train_loader = DataLoader(train_for_loader, batch_size=args.batch, shuffle=True)
    else:
        train_loader = DataLoader(_ExerciseTargetDataset(train_base),
                                  batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(_ExerciseTargetDataset(test_base),
                             batch_size=args.batch, shuffle=False)

    n_pose = 75
    fdim = feature_dim(args.feature_mode)
    model = HybridSTGCNHead(
        n_pose_features=n_pose,
        n_extra_features=fdim - n_pose,
        n_classes=len(EXERCISES),
        stgcn_hidden=args.stgcn_hidden,
        stgcn_layers=args.stgcn_layers,
        dropout=args.dropout,
        temporal_kernel=args.stgcn_temporal_kernel,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}  feature_mode={args.feature_mode} fdim={fdim}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1 = _epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        if not args.quiet:
            print(f"ep {ep:3d}  train loss {tr_loss:.4f} acc {tr_acc:.3f} f1 {tr_f1:.3f}")

    # Save with the same "args" key shape the ensemble_eval loader expects.
    tag = f"_{args.ckpt_tag}" if args.ckpt_tag else ""
    out_path = CKPT_DIR / f"gate_hybrid_tv{tag}.pt"
    ckpt_args = vars(args).copy()
    ckpt_args["arch"] = "hybrid"  # so ensemble_eval.build_model picks the right class
    torch.save({"state_dict": model.state_dict(), "args": ckpt_args, "epoch": args.epochs},
               out_path)
    print(f"saved {out_path.name}")

    # Final test evaluation
    _, te_acc, te_f1 = _epoch(model, test_loader, criterion, None, device)
    print(f"TEST (Isinsu): acc={te_acc:.3f}  macro-F1={te_f1:.3f}")

    # Per-exercise accuracy
    model.eval()
    counts = {ex: [0, 0] for ex in EXERCISES}  # [correct, total]
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x.to(device)).argmax(dim=-1).cpu().numpy()
            ys = y.numpy()
            for p, t in zip(preds, ys):
                counts[EXERCISES[int(t)]][1] += 1
                if p == t:
                    counts[EXERCISES[int(t)]][0] += 1
    for ex, (c, t) in counts.items():
        if t:
            print(f"  {ex:8s} {c:3d}/{t:3d}  {c/t:.2%}")


if __name__ == "__main__":
    main()
