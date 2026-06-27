"""Synthetic-MediaPipe augmentation: make EC3D's clean 3D poses look like
what we get from MediaPipe + canonical rotation at inference.

Two effects applied:
    1. Random yaw rotation around the vertical (y) axis. After our canonical
       rotation MediaPipe still has residual orientation drift; simulating it
       teaches the model robustness to small mis-orientations.
    2. Anisotropic Gaussian noise: small on x/y (~1-2 cm in real space),
       much larger on z. MediaPipe's pseudo-depth z is its noisiest channel.

The noise stddevs are in EC3D's normalised units (~0.4 = shoulder-to-hip).
"""
from __future__ import annotations

import numpy as np


# Defaults calibrated by hand to roughly match MediaPipe pose_world_landmarks
# jitter when expressed in EC3D's hip-normalised coordinate scale.
DEFAULT_STD_XY = 0.018
DEFAULT_STD_Z = 0.060
DEFAULT_MAX_YAW_DEG = 25.0


def random_yaw_rotation(
    pose: np.ndarray, max_yaw_deg: float = DEFAULT_MAX_YAW_DEG, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Rotate the whole sequence by the same random yaw around the vertical axis.

    Same rotation applied to every frame so within-sequence motion is preserved.
    """
    if rng is None:
        rng = np.random.default_rng()
    yaw = rng.uniform(-max_yaw_deg, max_yaw_deg) * np.pi / 180.0
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    R = np.array(
        [[c, 0.0, s],
         [0.0, 1.0, 0.0],
         [-s, 0.0, c]],
        dtype=pose.dtype,
    )
    return pose @ R.T


def add_mediapipe_noise(
    pose: np.ndarray,
    std_xy: float = DEFAULT_STD_XY,
    std_z: float = DEFAULT_STD_Z,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Per-joint per-frame Gaussian noise; z stddev is larger than x/y stddev."""
    if rng is None:
        rng = np.random.default_rng()
    noise = np.empty_like(pose)
    noise[..., 0] = rng.normal(0.0, std_xy, size=pose.shape[:-1])
    noise[..., 1] = rng.normal(0.0, std_xy, size=pose.shape[:-1])
    noise[..., 2] = rng.normal(0.0, std_z,  size=pose.shape[:-1])
    return pose + noise


def synth_augment(
    pose: np.ndarray,
    std_xy: float = DEFAULT_STD_XY,
    std_z: float = DEFAULT_STD_Z,
    max_yaw_deg: float = DEFAULT_MAX_YAW_DEG,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Full augmentation: random yaw rotation + anisotropic Gaussian noise.

    Apply to a (T, 25, 3) BODY_25 pose tensor that's already been normalised
    (hip-centred, spine vertical) so the augmentation lives in the same
    coordinate system the model sees.
    """
    if pose.ndim != 3 or pose.shape[-2:] != (25, 3):
        raise ValueError(f"expected (T, 25, 3); got {pose.shape}")
    if rng is None:
        rng = np.random.default_rng()
    out = random_yaw_rotation(pose, max_yaw_deg=max_yaw_deg, rng=rng)
    out = add_mediapipe_noise(out, std_xy=std_xy, std_z=std_z, rng=rng)
    return out
