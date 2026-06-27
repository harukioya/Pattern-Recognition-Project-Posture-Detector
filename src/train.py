"""Train the BiLSTM mistake classifier on EC3D with proper val-based selection.

Splits: Hugues+Sena = train, Vidit = val (for epoch selection),
Isinsu = test (touched ONCE at the end).

Reports macro-F1 in addition to accuracy. Auto-uses MPS on Apple Silicon.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from ec3d_dataset import (
    EC3DSequenceDataset,
    FEATURE_MODES,
    MISTAKE_LABELS,
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
CKPT_DIR.mkdir(exist_ok=True)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_weights(samples: list[tuple], n_classes: int, exclude: set[int], beta: float) -> torch.Tensor:
    """Effective-number-style smoothing: w_c = 1 / (count_c ** beta).
    beta=0 -> uniform, beta=1 -> full inverse-frequency.
    """
    counts = Counter(int(s[2]) for s in samples)
    w = np.ones(n_classes, dtype=np.float32)
    for c in range(n_classes):
        if c in exclude:
            w[c] = 0.0
        elif counts.get(c, 0) > 0:
            w[c] = 1.0 / (counts[c] ** beta)
    nz = w[w > 0]
    if nz.size:
        w[w > 0] = w[w > 0] * (nz.size / nz.sum())
    return torch.from_numpy(w)


def epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float, float]:
    """Returns (mean_loss, accuracy, macro_f1)."""
    train = optimizer is not None
    model.train(train)
    total_loss, total_n = 0.0, 0
    all_preds: list[int] = []
    all_targets: list[int] = []
    for x, _ex_id, mis_id in loader:
        x = x.to(device)
        mis_id_dev = mis_id.to(device)
        logits = model(x)
        loss = criterion(logits, mis_id_dev)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_targets.extend(mis_id.tolist())
    acc = float(np.mean(np.array(all_preds) == np.array(all_targets))) if all_preds else 0.0
    macro_f1 = float(
        f1_score(all_targets, all_preds, average="macro", zero_division=0)
    ) if all_preds else 0.0
    return total_loss / max(total_n, 1), acc, macro_f1


@torch.no_grad()
def per_class_report(model: nn.Module, loader: DataLoader, device: torch.device, header: str) -> None:
    model.eval()
    n_classes = len(MISTAKE_LABELS)
    correct = np.zeros(n_classes, dtype=np.int64)
    total = np.zeros(n_classes, dtype=np.int64)
    all_preds: list[int] = []
    all_targets: list[int] = []
    for x, _ex_id, mis_id in loader:
        x = x.to(device)
        preds = model(x).argmax(dim=-1).cpu().numpy()
        mis = mis_id.numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(mis.tolist())
        for p, t in zip(preds, mis):
            total[t] += 1
            if p == t:
                correct[t] += 1
    print(f"\n--- {header} ---")
    for c in range(n_classes):
        if total[c] == 0:
            continue
        print(
            f"  {MISTAKE_LABELS[c]:30s} {correct[c]:3d}/{total[c]:3d}  ({correct[c] / total[c]:.2%})"
        )
    print(
        f"  ---> overall acc: {sum(correct) / max(sum(total), 1):.2%}, "
        f"macro-F1: {f1_score(all_targets, all_preds, average='macro', zero_division=0):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default="angles_positions")
    parser.add_argument(
        "--class-weight-beta", type=float, default=0.5,
        help="0=uniform, 1=full inverse frequency.",
    )
    parser.add_argument(
        "--ckpt-tag", type=str, default="",
        help="Suffix appended to checkpoint filename.",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-epoch lines")
    parser.add_argument("--mirror", action="store_true", help="add mirrored copies to train set")
    parser.add_argument("--include-self-data", action="store_true",
                        help="concat self-recorded MediaPipe data into the training set")
    parser.add_argument("--train-split", choices=["train", "trainval"], default="train",
                        help="'train' = Hugues+Sena (Vidit held out as val); "
                             "'trainval' = paper-style Hugues+Sena+Vidit, save final epoch")
    parser.add_argument(
        "--arch",
        choices=["bilstm", "transformer", "stgcn", "stgcn_v2", "hybrid", "hybrid_v2"],
        default="bilstm",
    )
    parser.add_argument("--d-model", type=int, default=128, help="transformer hidden size")
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--stgcn-hidden", type=int, default=64)
    parser.add_argument("--stgcn-layers", type=int, default=3)
    parser.add_argument("--stgcn-temporal-kernel", type=int, default=9)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device()
    print(f"device: {device}")

    seqs = load_sequences()
    train_ds = EC3DSequenceDataset(
        seqs, mode=args.train_split, window=args.window,
        feature_mode=args.feature_mode, mirror=args.mirror,
    )
    val_ds = EC3DSequenceDataset(seqs, mode="val", window=args.window, feature_mode=args.feature_mode)
    test_ds = EC3DSequenceDataset(seqs, mode="test", window=args.window, feature_mode=args.feature_mode)

    if args.include_self_data:
        from torch.utils.data import ConcatDataset
        from self_data import SelfRecordedDataset
        self_ds = SelfRecordedDataset(window=args.window, feature_mode=args.feature_mode)
        print(self_ds.summary())
        if len(self_ds) > 0:
            train_ds = ConcatDataset([train_ds, self_ds])
    print(f"feature mode: {args.feature_mode} (dim={feature_dim(args.feature_mode)})  mirror={args.mirror}")
    print(f"train split  : {args.train_split}")
    print(f"clips        : train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    squat_extra_id = MISTAKE_LABELS.index("SQUAT/squat_extra")
    n_classes = len(MISTAKE_LABELS)
    # Combine .samples across ConcatDataset children for class-weight counting.
    if hasattr(train_ds, "samples"):
        all_samples = train_ds.samples
    else:
        all_samples = []
        for sub in getattr(train_ds, "datasets", [train_ds]):
            all_samples.extend(getattr(sub, "samples", []))
    weights = class_weights(
        all_samples, n_classes, exclude={squat_extra_id}, beta=args.class_weight_beta
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    if args.arch == "bilstm":
        model = BiLSTMHead(
            feature_dim(args.feature_mode), n_classes, hidden=args.hidden, dropout=args.dropout
        ).to(device)
    elif args.arch == "transformer":
        model = TransformerHead(
            feature_dim(args.feature_mode), n_classes,
            d_model=args.d_model, n_heads=args.n_heads,
            num_layers=args.n_layers, dropout=args.dropout,
            max_len=args.window,
        ).to(device)
    elif args.arch == "stgcn":
        model = STGCNHead(
            feature_dim(args.feature_mode), n_classes,
            hidden=args.stgcn_hidden, num_layers=args.stgcn_layers,
            dropout=args.dropout, temporal_kernel=args.stgcn_temporal_kernel,
        ).to(device)
    elif args.arch == "stgcn_v2":
        model = STGCNv2Head(
            feature_dim(args.feature_mode), n_classes,
            hidden=args.stgcn_hidden, num_layers=args.stgcn_layers,
            dropout=args.dropout, temporal_kernel=args.stgcn_temporal_kernel,
        ).to(device)
    elif args.arch == "hybrid":
        if args.feature_mode != "pose_extras":
            raise SystemExit("--arch hybrid requires --feature-mode pose_extras")
        model = HybridSTGCNHead(
            n_pose_features=75,
            n_extra_features=feature_dim(args.feature_mode) - 75,
            n_classes=n_classes,
            stgcn_hidden=args.stgcn_hidden,
            stgcn_layers=args.stgcn_layers,
            dropout=args.dropout,
            temporal_kernel=args.stgcn_temporal_kernel,
        ).to(device)
    else:  # hybrid_v2
        if args.feature_mode != "pose_extras":
            raise SystemExit("--arch hybrid_v2 requires --feature-mode pose_extras")
        model = HybridSTGCNv2Head(
            n_pose_features=75,
            n_extra_features=feature_dim(args.feature_mode) - 75,
            n_classes=n_classes,
            stgcn_hidden=args.stgcn_hidden,
            stgcn_layers=args.stgcn_layers,
            dropout=args.dropout,
            temporal_kernel=args.stgcn_temporal_kernel,
        ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_f1 = -1.0
    tag = f"_{args.ckpt_tag}" if args.ckpt_tag else ""
    best_path = CKPT_DIR / f"bilstm_ec3d_best{tag}.pt"

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1 = epoch_pass(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        if args.train_split == "train":
            va_loss, va_acc, va_f1 = epoch_pass(model, val_loader, criterion, None, device)
            marker = ""
            if va_f1 > best_val_f1:
                best_val_f1 = va_f1
                torch.save(
                    {"state_dict": model.state_dict(), "args": vars(args), "epoch": ep},
                    best_path,
                )
                marker = "  <- best"
            if not args.quiet:
                print(
                    f"ep {ep:3d}  train loss {tr_loss:.4f} acc {tr_acc:.3f} f1 {tr_f1:.3f}   "
                    f"val loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.3f}{marker}"
                )
        else:
            # trainval mode: Vidit is part of training, so val metrics are training-set
            # metrics and saving best-by-val is meaningless. We save the final checkpoint.
            if not args.quiet:
                print(f"ep {ep:3d}  train loss {tr_loss:.4f} acc {tr_acc:.3f} f1 {tr_f1:.3f}")

    if args.train_split == "trainval":
        torch.save(
            {"state_dict": model.state_dict(), "args": vars(args), "epoch": args.epochs},
            best_path,
        )
        print(f"\nsaved final-epoch checkpoint ({best_path.name})")
    else:
        print(f"\nbest val macro-F1: {best_val_f1:.3f}  ({best_path.name})")
        ckpt = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        per_class_report(model, val_loader, device, header="val (Vidit) at best epoch")

    per_class_report(model, test_loader, device, header="TEST (Isinsu)")


if __name__ == "__main__":
    main()
