"""Stored light/canonical rotations are xyzw; MuJoCo and Viser use wxyz.

These functions only permute components. They preserve dtype, magnitudes and
quaternion signs so converting existing data does not change its motion.
"""

from __future__ import annotations

import numpy as np


def wxyz_to_xyzw(quaternions: np.ndarray) -> np.ndarray:
    return np.asarray(quaternions)[..., [1, 2, 3, 0]]


def xyzw_to_wxyz(quaternions: np.ndarray) -> np.ndarray:
    return np.asarray(quaternions)[..., [3, 0, 1, 2]]


def qpos_wxyz_to_xyzw(qpos: np.ndarray) -> np.ndarray:
    """Convert the free robot base; the remaining columns are scalar joints."""
    converted = np.asarray(qpos).copy()
    converted[..., 3:7] = wxyz_to_xyzw(converted[..., 3:7])
    return converted


def qpos_xyzw_to_wxyz(qpos: np.ndarray) -> np.ndarray:
    converted = np.asarray(qpos).copy()
    converted[..., 3:7] = xyzw_to_wxyz(converted[..., 3:7])
    return converted


def light_payload_to_xyzw(payload: dict) -> None:
    """Finish packaging a wxyz UMR/GMR result as an xyzw light clip."""
    payload["qpos"] = qpos_wxyz_to_xyzw(payload["qpos"])
    if "light_object_rot" in payload:
        payload["light_object_rot"] = wxyz_to_xyzw(payload["light_object_rot"])
