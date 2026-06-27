"""Joint-format mapping between BODY_25 (EC3D), 21-joint ExeChecker, and H3.6M-17.

H3.6M 17-joint convention used by UI-PRMD (the canonical naming):
    0  Pelvis
    1  RHip       2  RKnee      3  RAnkle
    4  LHip       5  LKnee      6  LAnkle
    7  Spine
    8  Thorax
    9  Nose
   10  Head
   11  LShoulder 12  LElbow   13  LWrist
   14  RShoulder 15  RElbow   16  RWrist

We pick H3.6M-17 as the common ground.
"""
from __future__ import annotations

import numpy as np

H36M_JOINT_NAMES = (
    "Pelvis", "RHip", "RKnee", "RAnkle",
    "LHip", "LKnee", "LAnkle",
    "Spine", "Thorax", "Nose", "Head",
    "LShoulder", "LElbow", "LWrist",
    "RShoulder", "RElbow", "RWrist",
)
N_H36M = 17

# BODY_25 (OpenPose) joint indices used by EC3D.
_B25 = {
    "Nose": 0, "Neck": 1,
    "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7,
    "MidHip": 8,
    "RHip": 9, "RKnee": 10, "RAnkle": 11,
    "LHip": 12, "LKnee": 13, "LAnkle": 14,
}


def body25_to_h36m(pose: np.ndarray) -> np.ndarray:
    """Map a (..., 25, 3) BODY_25 pose tensor to (..., 17, 3) H3.6M layout.

    Spine and Head are derived (BODY_25 has no spine; head approximated as nose
    extended away from neck).
    """
    if pose.shape[-2] != 25 or pose.shape[-1] != 3:
        raise ValueError(f"expected (..., 25, 3); got {pose.shape}")
    out = np.empty(pose.shape[:-2] + (N_H36M, 3), dtype=pose.dtype)

    out[..., 0, :]  = pose[..., _B25["MidHip"], :]                     # Pelvis
    out[..., 1, :]  = pose[..., _B25["RHip"], :]                       # RHip
    out[..., 2, :]  = pose[..., _B25["RKnee"], :]                      # RKnee
    out[..., 3, :]  = pose[..., _B25["RAnkle"], :]                     # RAnkle
    out[..., 4, :]  = pose[..., _B25["LHip"], :]                       # LHip
    out[..., 5, :]  = pose[..., _B25["LKnee"], :]                      # LKnee
    out[..., 6, :]  = pose[..., _B25["LAnkle"], :]                     # LAnkle
    # Spine: midpoint between Neck and MidHip
    out[..., 7, :]  = 0.5 * (pose[..., _B25["Neck"], :] + pose[..., _B25["MidHip"], :])
    out[..., 8, :]  = pose[..., _B25["Neck"], :]                       # Thorax
    out[..., 9, :]  = pose[..., _B25["Nose"], :]                       # Nose
    # Head: nose extrapolated slightly above (along Nose - Neck direction)
    head_dir = pose[..., _B25["Nose"], :] - pose[..., _B25["Neck"], :]
    out[..., 10, :] = pose[..., _B25["Nose"], :] + 0.5 * head_dir      # Head (approx)
    out[..., 11, :] = pose[..., _B25["LShoulder"], :]
    out[..., 12, :] = pose[..., _B25["LElbow"], :]
    out[..., 13, :] = pose[..., _B25["LWrist"], :]
    out[..., 14, :] = pose[..., _B25["RShoulder"], :]
    out[..., 15, :] = pose[..., _B25["RElbow"], :]
    out[..., 16, :] = pose[..., _B25["RWrist"], :]
    return out


def execheck_21_to_h36m(pose: np.ndarray) -> np.ndarray:
    """ExeChecker's 21-joint layout = H3.6M-17 + (LHand, RHand, LFoot, RFoot).
    Take the first 17.
    """
    if pose.shape[-2] != 21 or pose.shape[-1] != 3:
        raise ValueError(f"expected (..., 21, 3); got {pose.shape}")
    return pose[..., :N_H36M, :]


def normalize_pose(pose: np.ndarray) -> np.ndarray:
    """Center on Pelvis (joint 0) and scale by hip-spine distance.

    Produces approximately unit-bone-length, hip-centered poses comparable
    across datasets with different absolute scales.
    """
    if pose.shape[-2] != N_H36M:
        raise ValueError("normalize_pose expects H3.6M-17 layout")
    centered = pose - pose[..., 0:1, :]
    # Bone length: pelvis -> thorax (joint 0 -> 8)
    bone = np.linalg.norm(pose[..., 8, :] - pose[..., 0, :], axis=-1, keepdims=True)
    bone = np.where(bone > 1e-6, bone, 1.0)
    return centered / bone[..., None]
