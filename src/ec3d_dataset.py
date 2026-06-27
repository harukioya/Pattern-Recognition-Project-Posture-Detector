"""EC3D sequence dataset for form-error classification.

Loads data_3D.pickle, groups consecutive frames into (act, sub, lab, rep)
sequences, computes joint angles, exposes a PyTorch Dataset.

Subject split follows the EC3D paper: subjects {1,2,3} (Hugues, Sena, Vidit)
train, subject 4 (Isinsu) test.
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Subject-independent split per EC3D paper (Table 1 caption):
# "subjects 1, 2, and 3 for training and 4 for testing"
# We further hold out Vidit as a validation subject for epoch selection so we
# never tune against Isinsu (the held-out test subject).
TRAIN_SUBJECTS = ("Hugues", "Sena")
VAL_SUBJECTS = ("Vidit",)
TEST_SUBJECTS = ("Isinsu",)
ALL_TRAIN_SUBJECTS = TRAIN_SUBJECTS + VAL_SUBJECTS  # for refit-on-all-train option

# Instruction-code -> (exercise, mistake_name) per Table 1 of the EC3D paper.
# Code 10 for SQUAT is present in the data but not in Table 1; we keep it as
# a 6th squat sub-class labeled "squat_extra" and can merge or drop later.
LABEL_MAP: dict[tuple[str, str], tuple[str, str]] = {
    ("SQUAT", "1"): ("SQUAT", "correct"),
    ("SQUAT", "2"): ("SQUAT", "feet_too_wide"),
    ("SQUAT", "3"): ("SQUAT", "knees_inward"),
    ("SQUAT", "4"): ("SQUAT", "not_low_enough"),
    ("SQUAT", "5"): ("SQUAT", "front_bent"),
    ("SQUAT", "10"): ("SQUAT", "squat_extra"),
    ("Lunges", "1"): ("Lunges", "correct"),
    ("Lunges", "4"): ("Lunges", "not_low_enough"),
    ("Lunges", "6"): ("Lunges", "knee_passes_toe"),
    ("Plank", "1"): ("Plank", "correct"),
    ("Plank", "7"): ("Plank", "arched_back"),
    ("Plank", "8"): ("Plank", "hunch_back"),
}

EXERCISES = ("SQUAT", "Lunges", "Plank")
EXERCISE_TO_ID = {e: i for i, e in enumerate(EXERCISES)}

# Flat 12-class label space (matches utils.py in the EC3D repo)
MISTAKE_LABELS: list[str] = [f"{ex}/{m}" for (ex, _code), (_e, m) in LABEL_MAP.items()]
MISTAKE_TO_ID = {lab: i for i, lab in enumerate(MISTAKE_LABELS)}


@dataclass
class Sequence:
    exercise: str
    subject: str
    instruction_code: str
    rep: str
    mistake_name: str
    frames: np.ndarray  # (T, 25, 3) float32, root-relative

    @property
    def exercise_id(self) -> int:
        return EXERCISE_TO_ID[self.exercise]

    @property
    def mistake_id(self) -> int:
        return MISTAKE_TO_ID[f"{self.exercise}/{self.mistake_name}"]


def load_sequences(path: Path | None = None) -> list[Sequence]:
    """Load data_3D.pickle and group consecutive frames into reps."""
    path = path or (DATA_DIR / "data_3D.pickle")
    with path.open("rb") as f:
        d = pickle.load(f)
    labels = d["labels"]              # (N, 5) <U6
    poses = d["poses"].astype(np.float32)  # (N, 3, 25)
    # Transpose to (N, 25, 3) so each frame is (joint, axis).
    poses = np.transpose(poses, (0, 2, 1))

    # Group consecutive frames sharing (act, sub, lab, rep).
    groups: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(labels):
        key = (str(row[0]), str(row[1]), str(row[2]), str(row[3]))
        groups[key].append(i)

    seqs: list[Sequence] = []
    skipped = 0
    for (act, sub, lab, rep), idxs in groups.items():
        idxs.sort()
        mapping = LABEL_MAP.get((act, lab))
        if mapping is None:
            skipped += 1
            continue
        exercise, mistake_name = mapping
        seqs.append(
            Sequence(
                exercise=exercise,
                subject=sub,
                instruction_code=lab,
                rep=rep,
                mistake_name=mistake_name,
                frames=poses[idxs],
            )
        )
    if skipped:
        print(f"[ec3d_dataset] skipped {skipped} groups with unmapped (act, lab)")
    return seqs


def joint_angle_features(frames: np.ndarray) -> np.ndarray:
    """Compute per-frame angle features from a (T, 25, 3) sequence.

    Joint layout assumed (OpenPose BODY_25, used by EC3D):
      0 nose, 1 neck, 2 R_shoulder, 3 R_elbow, 4 R_wrist,
      5 L_shoulder, 6 L_elbow, 7 L_wrist,
      8 mid_hip, 9 R_hip, 10 R_knee, 11 R_ankle,
      12 L_hip, 13 L_knee, 14 L_ankle,
      15 R_eye, 16 L_eye, 17 R_ear, 18 L_ear,
      19 L_bigtoe, 20 L_smalltoe, 21 L_heel,
      22 R_bigtoe, 23 R_smalltoe, 24 R_heel.
    """
    eps = 1e-6

    def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        v1 = a - b
        v2 = c - b
        cos = np.einsum("ti,ti->t", v1, v2) / (
            np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + eps
        )
        return np.arccos(np.clip(cos, -1.0, 1.0))

    f = frames  # (T, 25, 3)
    feats = np.stack(
        [
            angle(f[:, 9], f[:, 10], f[:, 11]),   # right knee
            angle(f[:, 12], f[:, 13], f[:, 14]),  # left knee
            angle(f[:, 1], f[:, 9], f[:, 10]),    # right hip
            angle(f[:, 1], f[:, 12], f[:, 13]),   # left hip
            angle(f[:, 10], f[:, 11], f[:, 11] + np.array([0.0, -1.0, 0.0])),  # right ankle vs gravity
            angle(f[:, 13], f[:, 14], f[:, 14] + np.array([0.0, -1.0, 0.0])),  # left ankle vs gravity
            angle(f[:, 2], f[:, 3], f[:, 4]),     # right elbow
            angle(f[:, 5], f[:, 6], f[:, 7]),     # left elbow
            angle(f[:, 1], f[:, 2], f[:, 3]),     # right shoulder
            angle(f[:, 1], f[:, 5], f[:, 6]),     # left shoulder
            angle(f[:, 0], f[:, 1], f[:, 8]),     # neck-spine (head-neck-hip)
            angle(f[:, 1], f[:, 8], f[:, 8] + np.array([0.0, -1.0, 0.0])),  # torso lean vs gravity
        ],
        axis=-1,
    ).astype(np.float32)
    return feats  # (T, 12)


def geometric_error_features(frames: np.ndarray) -> np.ndarray:
    """Hand-engineered features targeted at the EC3D mistake taxonomy.

    Each feature is signed/relative so the model doesn't have to discover the
    geometry from raw coordinates with only ~200 training clips.

    Returns (T, 14) float32.
    """
    f = frames  # (T, 25, 3) root-centred, spine vertical (y), facing constant.

    # Hip width and shoulder width as scale references (avoid div by zero).
    eps = 1e-6
    hip_width = np.linalg.norm(f[:, 9] - f[:, 12], axis=-1, keepdims=True) + eps  # (T,1)

    # 1-2. Knee valgus / varus: signed lateral deviation of each knee from the
    # hip-ankle line, normalised by hip width. Negative = inward.
    def lateral_dev(hip_i: int, knee_i: int, ankle_i: int) -> np.ndarray:
        hip, knee, ankle = f[:, hip_i], f[:, knee_i], f[:, ankle_i]
        # parameterise the hip-ankle line and project knee onto it
        line = ankle - hip
        line_len_sq = np.einsum("ti,ti->t", line, line) + eps
        t = np.einsum("ti,ti->t", knee - hip, line) / line_len_sq
        proj = hip + t[:, None] * line
        # Signed x-component of (knee - projection) — sign distinguishes inward/outward
        return ((knee - proj)[:, 0] / hip_width[:, 0]).astype(np.float32)

    knee_dev_R = lateral_dev(9, 10, 11)
    knee_dev_L = lateral_dev(12, 13, 14)

    # 3. Stance width: ankle-to-ankle / hip-to-hip ratio (>1 = wide stance)
    ankle_width = np.linalg.norm(f[:, 11] - f[:, 14], axis=-1) + eps
    stance_ratio = (ankle_width / hip_width[:, 0]).astype(np.float32)

    # 4. Squat depth: vertical hip drop relative to ankle height.
    hip_y = f[:, 8, 1]
    ankle_y = 0.5 * (f[:, 11, 1] + f[:, 14, 1])
    hip_above_ankle = ((hip_y - ankle_y) / hip_width[:, 0]).astype(np.float32)

    # 5. Knee-over-toe: signed offset on each horizontal axis (x, z). Letting both
    # signed values into the feature space lets the model discover which axis is the
    # subject's forward direction (EC3D normalises orientation but doesn't tell us which).
    def knee_minus_toe_xz(knee_i: int, toe_i: int) -> tuple[np.ndarray, np.ndarray]:
        d = (f[:, knee_i] - f[:, toe_i]) / hip_width
        return d[:, 0].astype(np.float32), d[:, 2].astype(np.float32)

    knee_minus_toe_x_R, knee_minus_toe_z_R = knee_minus_toe_xz(10, 22)
    knee_minus_toe_x_L, knee_minus_toe_z_L = knee_minus_toe_xz(13, 19)

    # 6. Forward lean (front_bent): signed neck-hip offset on each horizontal axis.
    lean_x = ((f[:, 1, 0] - f[:, 8, 0]) / hip_width[:, 0]).astype(np.float32)
    lean_z = ((f[:, 1, 2] - f[:, 8, 2]) / hip_width[:, 0]).astype(np.float32)

    # 7. Spine curvature proxy: angle (neck, mid_hip, foot_midpoint) - signed via y.
    mid_foot = 0.5 * (f[:, 11] + f[:, 14])
    v_top = f[:, 1] - f[:, 8]
    v_bot = mid_foot - f[:, 8]
    cos = np.einsum("ti,ti->t", v_top, v_bot) / (
        np.linalg.norm(v_top, axis=-1) * np.linalg.norm(v_bot, axis=-1) + eps
    )
    spine_axis_angle = np.arccos(np.clip(cos, -1.0, 1.0)).astype(np.float32)

    # 8. Plank arch/hunch indicator: vertical offset of hip from shoulder-ankle midline.
    sh_mid = 0.5 * (f[:, 2] + f[:, 5])
    line = mid_foot - sh_mid
    line_len_sq = np.einsum("ti,ti->t", line, line) + eps
    t = np.einsum("ti,ti->t", f[:, 8] - sh_mid, line) / line_len_sq
    proj = sh_mid + t[:, None] * line
    hip_offset_from_plankline = ((f[:, 8] - proj)[:, 1] / hip_width[:, 0]).astype(np.float32)

    # 9. Symmetric knee/foot mean & asymmetry
    knee_mean = 0.5 * (knee_dev_R + knee_dev_L)
    knee_asym = (knee_dev_R - knee_dev_L)

    return np.stack(
        [
            knee_dev_R,
            knee_dev_L,
            knee_mean,
            knee_asym,
            stance_ratio,
            hip_above_ankle,
            knee_minus_toe_x_R,
            knee_minus_toe_z_R,
            knee_minus_toe_x_L,
            knee_minus_toe_z_L,
            lean_x,
            lean_z,
            spine_axis_angle,
            hip_offset_from_plankline,
            np.linalg.norm(f[:, 11] - f[:, 14], axis=-1).astype(np.float32) / hip_width[:, 0],
            (f[:, 0, 1] - f[:, 1, 1]).astype(np.float32) / hip_width[:, 0],  # head-above-neck (chin tuck)
            (f[:, 1, 1] - f[:, 8, 1]).astype(np.float32) / hip_width[:, 0],  # torso vertical length
        ],
        axis=-1,
    ).astype(np.float32)  # (T, 17)


# Mirror augmentation: swap left/right joints and negate x-coordinate.
# Pairs follow OpenPose BODY_25 layout described above.
_MIRROR_PAIRS: tuple[tuple[int, int], ...] = (
    (2, 5), (3, 6), (4, 7),
    (9, 12), (10, 13), (11, 14),
    (15, 16), (17, 18),
    (19, 22), (20, 23), (21, 24),
)


def mirror_frames(frames: np.ndarray) -> np.ndarray:
    """Reflect a (T, 25, 3) sequence across the sagittal (yz) plane.

    Most EC3D errors (knees_inward, feet_too_wide, knee_passes_toe, etc.) are
    bilaterally symmetric, so the mistake label is invariant under mirroring.
    """
    out = frames.copy()
    out[:, :, 0] *= -1.0
    for a, b in _MIRROR_PAIRS:
        out[:, [a, b]] = out[:, [b, a]]
    return out


FEATURE_MODES = (
    "angles",
    "positions",
    "angles_positions",
    "all",
    "geom",                    # 17 hand-engineered error-targeted features
    "angles_geom",             # 12 angles + 17 geom = 29
    "angles_positions_geom",   # 12 + 75 + 17 = 104
    "pose_extras",             # 75 + 12 + 17 = 104, positions FIRST (for HybridSTGCN)
)


def _joint_positions(frames: np.ndarray) -> np.ndarray:
    """(T, 25, 3) -> (T, 75) flattened raw joint coordinates (already root-relative)."""
    return frames.reshape(frames.shape[0], -1).astype(np.float32)


def _joint_velocities(frames: np.ndarray) -> np.ndarray:
    """(T, 25, 3) -> (T, 75) first-difference velocities; first frame zero-padded."""
    pos = _joint_positions(frames)
    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    return vel


def extract_features(frames: np.ndarray, mode: str) -> np.ndarray:
    if mode not in FEATURE_MODES:
        raise ValueError(f"unknown feature mode: {mode}; expected one of {FEATURE_MODES}")
    if mode == "pose_extras":
        # Positions first so a HybridSTGCN can slice [: 75] and [75 :] cleanly.
        return np.concatenate(
            [
                _joint_positions(frames),
                joint_angle_features(frames),
                geometric_error_features(frames),
            ],
            axis=-1,
        ).astype(np.float32)
    parts: list[np.ndarray] = []
    if mode in ("angles", "angles_positions", "all", "angles_geom", "angles_positions_geom"):
        parts.append(joint_angle_features(frames))
    if mode in ("positions", "angles_positions", "all", "angles_positions_geom"):
        parts.append(_joint_positions(frames))
    if mode == "all":
        parts.append(_joint_velocities(frames))
    if mode in ("geom", "angles_geom", "angles_positions_geom"):
        parts.append(geometric_error_features(frames))
    return np.concatenate(parts, axis=-1).astype(np.float32)


def feature_dim(mode: str) -> int:
    dim = 0
    if mode in ("angles", "angles_positions", "all", "angles_geom", "angles_positions_geom"):
        dim += 12
    if mode in ("positions", "angles_positions", "all", "angles_positions_geom"):
        dim += 75
    if mode == "all":
        dim += 75
    if mode in ("geom", "angles_geom", "angles_positions_geom"):
        dim += 17
    if mode == "pose_extras":
        dim = 75 + 12 + 17
    return dim


# Back-compat for callers that import N_FEATURES (angles-only default).
N_FEATURES = 12


class EC3DSequenceDataset(Dataset):
    """Yields (features [T,F], exercise_id, mistake_id).

    Set `mode` to 'train' or 'test' for the subject-independent split.
    `feature_mode` selects the input representation; see FEATURE_MODES.
    `window` clips each sequence to a fixed-length window (or pads).
    """

    def __init__(
        self,
        sequences: list[Sequence],
        mode: str,
        window: int = 64,
        stride: int | None = None,
        feature_mode: str = "angles",
        mirror: bool = False,
        synth_mediapipe: bool = False,
        synth_views_per_clip: int = 1,
        synth_seed: int | None = None,
    ) -> None:
        assert mode in {"train", "val", "test", "trainval"}
        keep = {
            "train": TRAIN_SUBJECTS,
            "val": VAL_SUBJECTS,
            "test": TEST_SUBJECTS,
            "trainval": ALL_TRAIN_SUBJECTS,
        }[mode]
        self.window = window
        self.stride = stride or window  # non-overlapping by default
        self.feature_mode = feature_mode
        self.n_features = feature_dim(feature_mode)
        self.mirror = mirror
        self.synth_mediapipe = synth_mediapipe
        self.synth_views_per_clip = max(1, int(synth_views_per_clip))
        self._synth_rng = (
            np.random.default_rng(synth_seed) if synth_mediapipe else None
        )

        self.samples: list[tuple[np.ndarray, int, int]] = []
        # Cache raw (T, 25, 3) frame buffers per sample so synth augmentation
        # can fire on every __getitem__ call rather than once at construction.
        self._raw_frames: list[np.ndarray] = []
        for s in sequences:
            if s.subject not in keep:
                continue
            self._add_sequence(s.frames, s.exercise_id, s.mistake_id)
            if mirror:
                self._add_sequence(mirror_frames(s.frames), s.exercise_id, s.mistake_id)

    def _add_sequence(self, raw_frames: np.ndarray, ex_id: int, mis_id: int) -> None:
        # Pre-compute clip windows on the raw (T, 25, 3) frames so we can
        # augment on the fly if synth_mediapipe is enabled.
        T = raw_frames.shape[0]
        clip_starts: list[int] = []
        if T < self.window:
            # Pad: store as one window pre-padded
            pad = np.zeros((self.window - T, 25, 3), dtype=np.float32)
            raw_clip = np.concatenate([raw_frames, pad], axis=0).astype(np.float32)
            self._add_clip(raw_clip, ex_id, mis_id)
            return
        for start in range(0, T - self.window + 1, self.stride):
            raw_clip = raw_frames[start : start + self.window].astype(np.float32)
            self._add_clip(raw_clip, ex_id, mis_id)

    def _add_clip(self, raw_clip: np.ndarray, ex_id: int, mis_id: int) -> None:
        if self.synth_mediapipe:
            # Store raw frames; features are computed live in __getitem__
            # with fresh noise each call. Add `synth_views_per_clip` entries
            # so we effectively K× the dataset.
            for _ in range(self.synth_views_per_clip):
                self._raw_frames.append(raw_clip)
                self.samples.append((None, ex_id, mis_id))  # placeholder
        else:
            feats = extract_features(raw_clip, self.feature_mode)
            self.samples.append((feats, ex_id, mis_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        feats_cached, ex_id, mis_id = self.samples[i]
        if self.synth_mediapipe:
            from synth_mediapipe import synth_augment
            raw = self._raw_frames[i]
            augmented = synth_augment(raw, rng=self._synth_rng)
            feats = extract_features(augmented, self.feature_mode)
        else:
            feats = feats_cached
        return (
            torch.from_numpy(feats),
            torch.tensor(ex_id, dtype=torch.long),
            torch.tensor(mis_id, dtype=torch.long),
        )


def class_counts(samples: Iterable[tuple]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for _, _, mis_id in samples:
        out[MISTAKE_LABELS[mis_id]] += 1
    return dict(out)
