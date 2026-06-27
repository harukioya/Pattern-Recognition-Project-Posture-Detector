"""Quick verification of EC3DSequenceDataset: shapes, splits, class counts."""
from __future__ import annotations

from ec3d_dataset import (
    EC3DSequenceDataset,
    MISTAKE_LABELS,
    N_FEATURES,
    class_counts,
    load_sequences,
)


def main() -> None:
    seqs = load_sequences()
    print(f"raw sequences: {len(seqs)}")
    print(f"feature dim  : {N_FEATURES}")
    print(f"mistake set  : {len(MISTAKE_LABELS)}  -> {MISTAKE_LABELS}")

    train_ds = EC3DSequenceDataset(seqs, mode="train", window=64)
    test_ds = EC3DSequenceDataset(seqs, mode="test", window=64)
    print(f"train clips  : {len(train_ds)}")
    print(f"test  clips  : {len(test_ds)}")

    print("\n--- train class counts ---")
    for lab, n in sorted(class_counts(train_ds.samples).items()):
        print(f"  {lab:30s} {n}")

    print("\n--- test class counts ---")
    for lab, n in sorted(class_counts(test_ds.samples).items()):
        print(f"  {lab:30s} {n}")

    x, ex_id, mis_id = train_ds[0]
    print(f"\nsample shape : x={tuple(x.shape)} ex_id={int(ex_id)} mis_id={int(mis_id)}")
    print(f"sample range : min={x.min():.3f} max={x.max():.3f}")
    print(f"sample label : {MISTAKE_LABELS[int(mis_id)]}")


if __name__ == "__main__":
    main()
