"""Convert MediaPipe BlazePose's 33-landmark output to OpenPose BODY_25 layout.

BlazePose 33 landmark order (from MediaPipe docs):
   0 nose
   1 left_eye_inner  2 left_eye        3 left_eye_outer
   4 right_eye_inner 5 right_eye       6 right_eye_outer
   7 left_ear        8 right_ear
   9 mouth_left     10 mouth_right
  11 left_shoulder  12 right_shoulder
  13 left_elbow     14 right_elbow
  15 left_wrist     16 right_wrist
  17 left_pinky     18 right_pinky
  19 left_index     20 right_index
  21 left_thumb     22 right_thumb
  23 left_hip       24 right_hip
  25 left_knee      26 right_knee
  27 left_ankle     28 right_ankle
  29 left_heel      30 right_heel
  31 left_foot_index 32 right_foot_index

BODY_25 layout we trained on (EC3D):
   0 nose, 1 neck (derived), 2 R_shoulder, 3 R_elbow, 4 R_wrist,
   5 L_shoulder, 6 L_elbow, 7 L_wrist, 8 mid_hip (derived),
   9 R_hip, 10 R_knee, 11 R_ankle, 12 L_hip, 13 L_knee, 14 L_ankle,
  15 R_eye, 16 L_eye, 17 R_ear, 18 L_ear,
  19 L_bigtoe, 20 L_smalltoe, 21 L_heel,
  22 R_bigtoe, 23 R_smalltoe, 24 R_heel
"""
from __future__ import annotations

import numpy as np

# Direct (no derivation) mapping: BODY_25 index -> BlazePose index
_DIRECT_MAP: dict[int, int] = {
    0: 0,    # nose
    2: 12,   # R_shoulder
    3: 14,   # R_elbow
    4: 16,   # R_wrist
    5: 11,   # L_shoulder
    6: 13,   # L_elbow
    7: 15,   # L_wrist
    9: 24,   # R_hip
    10: 26,  # R_knee
    11: 28,  # R_ankle
    12: 23,  # L_hip
    13: 25,  # L_knee
    14: 27,  # L_ankle
    15: 5,   # R_eye
    16: 2,   # L_eye
    17: 8,   # R_ear
    18: 7,   # L_ear
    21: 29,  # L_heel
    24: 30,  # R_heel
    19: 31,  # L_bigtoe  (BlazePose only has foot_index — use it for both toes)
    20: 31,  # L_smalltoe (approx — duplicate foot_index)
    22: 32,  # R_bigtoe
    23: 32,  # R_smalltoe (approx)
}


def blazepose_to_body25(landmarks: np.ndarray) -> np.ndarray:
    """Map a (33, 3) BlazePose landmark array to (25, 3) BODY_25.

    Joints 1 (neck) and 8 (mid_hip) are derived as midpoints of L/R counterparts.
    The pseudo-3D z channel from BlazePose is preserved as-is; downstream
    preprocessing handles re-normalisation.
    """
    if landmarks.shape != (33, 3):
        raise ValueError(f"expected (33, 3) BlazePose array, got {landmarks.shape}")
    out = np.zeros((25, 3), dtype=landmarks.dtype)
    for body25_idx, bp_idx in _DIRECT_MAP.items():
        out[body25_idx] = landmarks[bp_idx]
    # Neck = midpoint of left and right shoulder
    out[1] = 0.5 * (landmarks[11] + landmarks[12])
    # Mid_hip = midpoint of left and right hip
    out[8] = 0.5 * (landmarks[23] + landmarks[24])
    return out


def canonical_rotate(pose: np.ndarray) -> np.ndarray:
    """Rotate a (T, 25, 3) BODY_25 sequence into a canonical body frame.

    Aligns the body so the hip line (R_hip -> L_hip) points along +x and the
    spine (mid_hip -> neck) points along +y. The forward axis (chest normal)
    falls naturally on +z via the right-hand cross product.

    Uses the time-mean directions so the rotation is stable across the
    buffer (the user shouldn't be turning during a rep).
    """
    if pose.ndim != 3 or pose.shape[-2:] != (25, 3):
        raise ValueError(f"expected (T, 25, 3); got {pose.shape}")
    # Indices: 1=neck, 8=mid_hip, 9=R_hip, 12=L_hip
    hip_dir = (pose[:, 12] - pose[:, 9]).mean(axis=0)         # R_hip -> L_hip
    spine_dir = (pose[:, 1] - pose[:, 8]).mean(axis=0)        # mid_hip -> neck

    # Gram-Schmidt: orthonormal body basis (e1, e2, e3)
    e1 = hip_dir / (np.linalg.norm(hip_dir) + 1e-8)
    e2_raw = spine_dir - np.dot(spine_dir, e1) * e1
    e2 = e2_raw / (np.linalg.norm(e2_raw) + 1e-8)
    e3 = np.cross(e1, e2)
    R = np.stack([e1, e2, e3], axis=0).astype(np.float32)     # (3, 3) rows are basis vectors

    # Project every joint onto the body basis: out[..., k] = R[k, :] . pose[..., :]
    return pose @ R.T


def normalise_like_ec3d(pose: np.ndarray) -> np.ndarray:
    """Re-normalise a (T, 25, 3) BODY_25 pose to roughly match EC3D's space.

    Centre on mid_hip, scale by shoulder-to-hip distance. NO rotation —
    `canonical_rotate` was tried earlier today but appeared to put inputs in
    a coordinate system EC3D's models were never trained on (specifically the
    z-axis interpretation), which biased predictions toward Lunges/knee_passes_toe.
    Keep this function aligned with how training data was preprocessed.
    """
    if pose.ndim != 3 or pose.shape[-2:] != (25, 3):
        raise ValueError(f"expected (T, 25, 3); got {pose.shape}")
    out = pose.astype(np.float32)
    out = out - out[:, 8:9, :]
    shoulder_mid = 0.5 * (out[:, 2] + out[:, 5])
    sh_to_hip = np.linalg.norm(shoulder_mid - out[:, 8], axis=-1)
    scale = float(sh_to_hip.mean()) + 1e-6
    out = out / scale * 0.40
    return out
