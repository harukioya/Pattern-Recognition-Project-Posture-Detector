"""Self-recorded MediaPipe data wrapped as a torch Dataset.

Each `.pkl` in `data/self_recorded/` is a single 25-joint BODY_25 sequence
captured live from MediaPipe. We slice it into non-overlapping windows,
canonicalise via `normalise_like_ec3d`, and extract the same feature space
the model was trained on so the two data sources are interchangeable.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from blazepose_to_body25 import normalise_like_ec3d
from ec3d_dataset import (
    EXERCISE_TO_ID,
    MISTAKE_LABELS,
    extract_features,
    global_to_local_id,
    local_labels_for_exercise,
)

SAVE_DIR = Path(__file__).resolve().parent.parent / "data" / "self_recorded"


def load_self_recorded_entries() -> list[dict]:
    out = []
    for p in sorted(SAVE_DIR.glob("*.pkl")):
        with p.open("rb") as f:
            out.append(pickle.load(f))
    return out


class SelfRecordedDataset(Dataset):
    def __init__(
        self,
        window: int = 64,
        stride: int | None = None,
        feature_mode: str = "pose_extras",
        exercise_filter: str | None = None,
    ) -> None:
        self.window = window
        self.stride = stride or window
        self.feature_mode = feature_mode
        self.exercise_filter = exercise_filter
        self.local_labels: list[str] | None = (
            local_labels_for_exercise(exercise_filter) if exercise_filter else None
        )
        self.samples: list[tuple[np.ndarray, int, int]] = []
        self._counts: dict[str, int] = {}

        for entry in load_self_recorded_entries():
            label: str = entry["label"]
            if label not in MISTAKE_LABELS:
                print(f"[self_data] WARN: unknown label {label!r}, skipping")
                continue
            exercise = label.split("/", 1)[0]
            if exercise_filter is not None and exercise != exercise_filter:
                continue
            global_mis_id = MISTAKE_LABELS.index(label)
            if exercise_filter is not None:
                local_id = global_to_local_id(exercise_filter, global_mis_id)
                if local_id is None:
                    continue
                mis_id = local_id
            else:
                mis_id = global_mis_id
            ex_id = EXERCISE_TO_ID[exercise]
            frames: np.ndarray = entry["frames_body25"].astype(np.float32)

            T = frames.shape[0]
            if T < window:
                pad = np.zeros((window - T, 25, 3), dtype=np.float32)
                clip = np.concatenate([frames, pad], axis=0)
                normed = normalise_like_ec3d(clip)
                self.samples.append(
                    (extract_features(normed, feature_mode), ex_id, mis_id)
                )
                self._counts[label] = self._counts.get(label, 0) + 1
                continue

            for start in range(0, T - window + 1, self.stride):
                clip = frames[start : start + window]
                normed = normalise_like_ec3d(clip)
                feats = extract_features(normed, feature_mode)
                self.samples.append((feats, ex_id, mis_id))
                self._counts[label] = self._counts.get(label, 0) + 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        feats, ex_id, mis_id = self.samples[i]
        return (
            torch.from_numpy(feats),
            torch.tensor(ex_id, dtype=torch.long),
            torch.tensor(mis_id, dtype=torch.long),
        )

    def summary(self) -> str:
        if not self.samples:
            return "self-recorded: 0 clips (no pkl files found)"
        lines = [f"self-recorded: {len(self.samples)} clips"]
        for lab, n in sorted(self._counts.items()):
            lines.append(f"  {lab:30s} {n} clips")
        return "\n".join(lines)


if __name__ == "__main__":
    ds = SelfRecordedDataset()
    print(ds.summary())
