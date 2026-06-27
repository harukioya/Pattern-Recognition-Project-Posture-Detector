"""Train the Stage-1 binary correct/incorrect classifier.

Trains on the unified EC3D + UI-PRMD pool, selects the best epoch by val F1
on the held-out subjects, and evaluates once on EC3D test (Isinsu).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from binary_dataset import UnifiedBinaryDataset, build_unified_splits
from model import BinarySTGCNHead

CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


def pick_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def epoch_pass(model, loader, criterion, optimizer, device):
    train = optimizer is not None
    model.train(train)
    total_loss, total_n = 0.0, 0
    all_p, all_t = [], []
    for x, y, _src in loader:
        x = x.to(device)
        y_dev = y.to(device)
        logits = model(x)
        loss = criterion(logits, y_dev)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
        all_p.extend(logits.argmax(dim=-1).cpu().tolist())
        all_t.extend(y.tolist())
    acc = float(np.mean(np.array(all_p) == np.array(all_t)))
    f1 = float(f1_score(all_t, all_p, average="macro", zero_division=0))
    return total_loss / max(total_n, 1), acc, f1


@torch.no_grad()
def per_source_report(model, samples, device, header: str):
    model.eval()
    by_src: dict[str, list] = {}
    for s in samples:
        by_src.setdefault(s.source, []).append(s)
    print(f"\n--- {header} ---")
    for src, samps in by_src.items():
        ds = UnifiedBinaryDataset(samps)
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        all_p, all_t = [], []
        for x, y, _src in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=-1).cpu().numpy()
            all_p.extend(preds.tolist())
            all_t.extend(y.tolist())
        acc = float(np.mean(np.array(all_p) == np.array(all_t)))
        f1 = float(f1_score(all_t, all_p, average="macro", zero_division=0))
        print(f"  {src:10s}  N={len(samps):4d}  acc={acc:.3f}  macro-F1={f1:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt-tag", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device()
    print(f"device: {device}")

    train_samples, val_samples, test_samples = build_unified_splits(window=args.window)
    print(f"clips: train={len(train_samples)}  val={len(val_samples)}  test={len(test_samples)}")

    train_loader = DataLoader(UnifiedBinaryDataset(train_samples), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(UnifiedBinaryDataset(val_samples), batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(UnifiedBinaryDataset(test_samples), batch_size=args.batch, shuffle=False)

    model = BinarySTGCNHead(hidden=args.hidden, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    tag = f"_{args.ckpt_tag}" if args.ckpt_tag else ""
    best_path = CKPT_DIR / f"binary_stgcn_best{tag}.pt"

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1 = epoch_pass(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_f1 = epoch_pass(model, val_loader, criterion, None, device)
        scheduler.step()
        marker = ""
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save({"state_dict": model.state_dict(), "args": vars(args), "epoch": ep}, best_path)
            marker = "  <- best"
        if not args.quiet:
            print(
                f"ep {ep:3d}  train loss {tr_loss:.4f} acc {tr_acc:.3f} f1 {tr_f1:.3f}   "
                f"val loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.3f}{marker}"
            )

    print(f"\nbest val macro-F1: {best_val_f1:.3f}  ({best_path.name})")

    ckpt = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    per_source_report(model, val_samples, device, "val by source")
    per_source_report(model, test_samples, device, "TEST by source (EC3D Isinsu)")


if __name__ == "__main__":
    main()
