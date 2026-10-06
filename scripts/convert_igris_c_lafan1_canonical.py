#!/usr/bin/env python3
"""Convert IGRIS-C LAFAN1 retarget NPZ files to canonical motion and scene tracks."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mujoco
import numpy as np

from convert_igris_c_omomo_canonical import as_wxyz_rotation, make_motion, scalar, write_usd_atomic


def floor_scene(model: mujoco.MjModel, frames: int, fps: float, root: Path) -> dict:
    entities = []
    for index in range(model.ngeom):
        if model.geom_type[index] != mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if model.geom_bodyid[index] != 0:
            raise ValueError("Moving floor planes are not supported")
        half_x, half_y = map(float, model.geom_size[index, :2])
        if half_x <= 0 or half_y <= 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or f"floor_{index}"
        usd_path = root / "usd_assets" / "terrain" / f"{name}_{half_x:g}_{half_y:g}.usd"
        if not usd_path.is_file():
            vertices = np.array([[-half_x, -half_y, 0], [half_x, -half_y, 0],
                                 [half_x, half_y, 0], [-half_x, half_y, 0]], dtype=np.float32)
            faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
            write_usd_atomic(usd_path, "Terrain", [(vertices, faces)])
        quaternion, _ = as_wxyz_rotation(np.repeat(model.geom_quat[index][None], frames, axis=0))
        entities.append({
            "entity_id": name,
            "entity_type": "terrain",
            "usd_asset": usd_path.relative_to(root).as_posix(),
            "usd_prim_path": "/Terrain",
            "pose_pos": np.repeat(model.geom_pos[index][None], frames, axis=0).astype(np.float32),
            "pose_rot": quaternion,
            "mobility": "static",
            "semantic_label": "floor",
        })
    return {"fps": fps, "scene_entities": entities}


def convert(path: Path, root: Path) -> None:
    with np.load(path, allow_pickle=True) as result:
        if scalar(result, "robot_name") != "igris_c" or scalar(result, "source_format") != "smplx_npz":
            raise ValueError(f"Not an IGRIS-C SMPL-X retarget result: {path}")
        qpos = np.asarray(result["qpos"], dtype=np.float64)
        frame_ids = np.asarray(result["frame_ids"])
        fps = float(scalar(result, "fps"))
        if (qpos.ndim != 2 or qpos.shape[1] != 38 or len(qpos) == 0
                or len(frame_ids) != len(qpos) or not np.isfinite(qpos).all()
                or not np.isfinite(fps) or fps <= 0):
            raise ValueError(f"Invalid retarget motion: {path}")
        model_path = Path(str(scalar(result, "robot_xml"))).expanduser().resolve()
        model = mujoco.MjModel.from_xml_path(str(model_path))
        joint_names = [str(name) for name in result["robot_joint_names"].tolist()]
        motion = make_motion(model, qpos, fps, joint_names)
        scene = floor_scene(model, len(qpos), fps, root)

    bundled_model = root / "robot" / "igris_c.mjb"
    if not bundled_model.is_file():
        bundled_model.parent.mkdir(parents=True, exist_ok=True)
        mujoco.mj_saveModel(model, str(bundled_model))
    motion["retarget_meta"] = {"robot_model": bundled_model.relative_to(root).as_posix()}
    for directory, payload in ((root / "motion", motion), (root / "scene_tracks", scene)):
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{path.stem}.pkl").open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{path.name}: {len(qpos)} frames, {len(scene['scene_entities'])} scene entities", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One retarget NPZ or a directory of them")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Convert the first N clips; 0 means all")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    source = args.input.expanduser().resolve()
    root = args.output_root.expanduser().resolve()
    files = [source] if source.is_file() and source.suffix.lower() == ".npz" else sorted(source.glob("*.npz"))
    if not files:
        parser.error(f"No retarget NPZ files found: {source}")
    for path in files[:args.limit or None]:
        convert(path, root)


if __name__ == "__main__":
    main()
