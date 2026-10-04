"""Read an IGRIS-C canonical directory for the Viser folder browser."""

from __future__ import annotations

import pickle
from pathlib import Path

import mujoco
import numpy as np
from pxr import Gf, Usd, UsdGeom


def scan_canonical_motion_files(root: Path) -> list[Path]:
    """Keep scene PKLs out of the clip dropdown."""
    motion_dir = Path(root) / "motion"
    return sorted(motion_dir.glob("*.pkl")) if motion_dir.is_dir() else []


def load_motion(path: Path) -> tuple[dict, np.ndarray, mujoco.MjModel]:
    path = Path(path).expanduser().resolve()
    if path.parent.name != "motion":
        raise ValueError(f"Canonical clip must be in a motion/ directory: {path}")
    with path.open("rb") as handle:
        motion = pickle.load(handle)
    if not isinstance(motion, dict):
        raise ValueError(f"Canonical motion is not a dict: {path}")
    root = path.parent.parent
    model_rel = motion.get("retarget_meta", {}).get("robot_model", "robot/igris_c.mjb")
    model_path = (root / model_rel).resolve()
    if not model_path.is_relative_to(root) or not model_path.is_file():
        raise FileNotFoundError(f"Bundled robot model missing: {model_path}")
    model = mujoco.MjModel.from_binary_path(str(model_path))
    positions = np.asarray(motion["root_pos"], dtype=np.float64)
    rotations = np.asarray(motion["root_rot"], dtype=np.float64)
    joints = np.asarray(motion["dof_pos"], dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Invalid root_pos shape in {path}: {positions.shape}")
    count = len(positions)
    if rotations.shape != (count, 4) or joints.shape != (count, model.nq - 7):
        raise ValueError(f"Canonical pose arrays do not match the robot model: {path}")
    qpos = np.concatenate((positions, rotations, joints), axis=1)
    if count == 0 or not np.isfinite(qpos).all():
        raise ValueError(f"Canonical motion is empty or contains nonfinite poses: {path}")
    fps = float(motion["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid canonical fps: {fps}")
    return motion, qpos, model


def _triangulate_mesh(mesh: UsdGeom.Mesh) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    faces = []
    offset = 0
    for count in counts:
        polygon = indices[offset : offset + count]
        for index in range(1, count - 1):
            faces.append((polygon[0], polygon[index], polygon[index + 1]))
        offset += count
    if offset != len(indices) or not faces:
        raise ValueError(f"USD mesh has invalid or empty faces: {mesh.GetPath()}")
    return vertices, np.asarray(faces, dtype=np.uint32)


def _load_usd_meshes(path: Path, prim_path: str) -> list[tuple[np.ndarray, np.ndarray]]:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Could not open USD asset: {path}")
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim:
        raise ValueError(f"USD root prim {prim_path!r} is missing: {path}")
    xforms = UsdGeom.XformCache()
    world_to_root = xforms.GetLocalToWorldTransform(root_prim).GetInverse()
    meshes = []
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.guide:
            continue
        vertices, faces = _triangulate_mesh(UsdGeom.Mesh(prim))
        mesh_to_root = xforms.GetLocalToWorldTransform(prim) * world_to_root
        vertices = np.asarray(
            [mesh_to_root.Transform(Gf.Vec3d(*point)) for point in vertices],
            dtype=np.float32,
        )
        meshes.append((vertices, faces))
    if not meshes:
        raise ValueError(f"No USD meshes below {prim_path!r}: {path}")
    return meshes


def load_scene_entities(motion_path: Path, frame_count: int, fps: float) -> list[dict]:
    """Load the matching scene PKL and its USD geometry from one dataset root."""
    motion_path = Path(motion_path).expanduser().resolve()
    root = motion_path.parent.parent
    scene_path = root / "scene_tracks" / motion_path.name
    if not scene_path.is_file():
        return []
    with scene_path.open("rb") as handle:
        scene = pickle.load(handle)
    if not isinstance(scene, dict) or not np.isclose(float(scene["fps"]), fps):
        raise ValueError(f"Scene fps differs from motion: {scene_path}")
    entities = []
    for entry in scene.get("scene_entities", []):
        position = np.asarray(entry["pose_pos"], dtype=np.float32)
        rotation = np.asarray(entry["pose_rot"], dtype=np.float32)
        if position.shape != (frame_count, 3) or rotation.shape != (frame_count, 4):
            raise ValueError(f"Scene pose shape differs from motion: {scene_path}, {entry['entity_id']}")
        asset_path = (root / entry["usd_asset"]).resolve()
        if not asset_path.is_relative_to(root) or not asset_path.is_file():
            raise FileNotFoundError(f"Canonical USD asset missing: {asset_path}")
        entities.append({
            "entity_id": str(entry["entity_id"]),
            "entity_type": str(entry["entity_type"]),
            "pose_pos": position,
            "pose_rot": rotation,
            "meshes": _load_usd_meshes(asset_path, str(entry["usd_prim_path"])),
        })
    return entities
