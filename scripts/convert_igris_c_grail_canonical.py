#!/usr/bin/env python3
"""Convert IGRIS-C GRAIL retarget results to canonical motion and scene tracks."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom
from scipy.spatial.transform import Rotation

from convert_igris_c_omomo_canonical import (
    as_wxyz_rotation, has_separate_convex_collision, make_motion, make_usd, scalar, write_usd_atomic,
)
from humanoid_retarget_pipeline_hsi_hoi import load_grail_pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One GRAIL retarget NPZ or its directory")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--motion-dir", type=Path, default=Path("motion"))
    parser.add_argument("--scene-dir", type=Path, default=Path("scene_tracks"))
    parser.add_argument("--usd-dir", type=Path, default=Path("usd_assets"))
    parser.add_argument("--object-size", choices=("scaled", "original"), default="scaled",
                        help="Match the retarget solver's object-size setting")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of NPZ files; 0 means all")
    options = parser.parse_args()
    if options.limit < 0:
        parser.error("--limit must be nonnegative")
    options.output_root = options.output_root.expanduser().resolve()
    for name in ("motion_dir", "scene_dir", "usd_dir"):
        value = getattr(options, name)
        path = (value if value.is_absolute() else options.output_root / value).resolve()
        if not path.is_relative_to(options.output_root):
            parser.error(f"--{name.replace('_', '-')} must be inside --output-root")
        setattr(options, name, path)
    if len({options.motion_dir, options.scene_dir, options.usd_dir}) != 3:
        parser.error("motion, scene, and USD directories must be different")
    return options


def usd_meshes(path: Path, scale: float) -> list[tuple[np.ndarray, np.ndarray]]:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Cannot open GRAIL USD: {path}")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError(f"GRAIL USD must be Z-up: {path}")
    meters = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not np.isfinite(meters) or meters <= 0:
        raise ValueError(f"Invalid GRAIL USD metersPerUnit: {path}")
    xforms = UsdGeom.XformCache()
    meshes = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64).reshape(-1, 3)
        transform = xforms.GetLocalToWorldTransform(prim)
        vertices = np.asarray([transform.Transform(Gf.Vec3d(*point)) for point in points], dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if int(counts.sum()) != len(indices):
            raise ValueError(f"Invalid USD face indices: {path}")
        faces = []
        cursor = 0
        for count in counts:
            polygon = indices[cursor:cursor + int(count)]
            for corner in range(1, int(count) - 1):
                faces.append((polygon[0], polygon[corner], polygon[corner + 1]))
            cursor += int(count)
        if faces:
            # The source USD already contains GRAIL obj_scale. Only convert
            # stage units and the retarget frame scale here.
            meshes.append(((vertices * meters * scale).astype(np.float32), np.asarray(faces, dtype=np.int32)))
    if not meshes:
        raise ValueError(f"No meshes in GRAIL USD: {path}")
    return meshes


def coacd_parts(recon_path: Path, result, usd_source: Path) -> list[Path]:
    directory = Path(str(scalar(result, "source_object_dir", ""))).expanduser().resolve()
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"GRAIL CoACD metadata missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    count = int(metadata.get("collision_parts", 0))
    if count <= 0 or metadata.get("sequence_key") != recon_path.stem:
        raise ValueError(f"GRAIL CoACD metadata does not match {recon_path}: {metadata_path}")
    if Path(str(metadata.get("source_usd", ""))).resolve() != usd_source.resolve():
        raise ValueError(f"GRAIL CoACD source USD does not match: {metadata_path}")
    parts = [directory / f"grail_object_collision_{index}.obj" for index in range(count)]
    for part in parts:
        if not part.is_file():
            raise FileNotFoundError(f"GRAIL CoACD collision part missing: {part}")
    return parts


def coacd_meshes(parts: list[Path], scale: float) -> list[tuple[np.ndarray, np.ndarray]]:
    meshes = []
    for part in parts:
        mesh = trimesh.load(part, force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
            raise ValueError(f"Invalid GRAIL CoACD mesh: {part}")
        meshes.append(((vertices * scale).astype(np.float32), faces))
    return meshes


def object_pose(recon_path: Path, frame_ids: np.ndarray, result) -> tuple[np.ndarray, np.ndarray]:
    obj_data = load_grail_pickle(recon_path).get("obj_data", {})
    positions = np.asarray(obj_data["obj_t"], dtype=np.float64).reshape(-1, 3)
    rotations = np.asarray(obj_data["obj_R"], dtype=np.float64).reshape(-1, 3, 3)
    if len(positions) != len(rotations) or frame_ids.min() < 0 or frame_ids.max() >= len(positions):
        raise ValueError(f"GRAIL object pose does not cover retarget frames: {recon_path}")
    if str(scalar(result, "noitom_output_up", "z")).lower() != "z" or bool(scalar(result, "noitom_convert_y_up", False)):
        raise ValueError(f"Expected GRAIL Z-up pose metadata: {recon_path}")
    positions = positions[frame_ids].copy()
    if bool(scalar(result, "noitom_ground_align", False)):
        positions[:, 2] -= float(scalar(result, "noitom_floor_y", 0))
    positions[:, 2] += float(scalar(result, "noitom_ground_offset", 0))
    positions *= float(scalar(result, "smpl_scale", 1))
    xyzw = Rotation.from_matrix(rotations[frame_ids]).as_quat()
    wxyz, _ = as_wxyz_rotation(xyzw[:, [3, 0, 1, 2]])
    return positions.astype(np.float32), wxyz


def make_scene(model: mujoco.MjModel, result, frame_ids: np.ndarray, fps: float,
               options: argparse.Namespace) -> dict:
    recon_path = Path(str(scalar(result, "source_data"))).resolve()
    if not recon_path.is_file() or recon_path.parent.name != "recon":
        raise FileNotFoundError(f"GRAIL recon pickle missing: {recon_path}")
    subset_root = recon_path.parent.parent
    usd_source = subset_root / "object_usd" / f"{recon_path.stem}.usd"
    if not usd_source.is_file():
        raise FileNotFoundError(usd_source)
    is_terrain = recon_path.stem.startswith("terrain_")
    entity_type = "terrain" if is_terrain else "object"
    prim_name = "Terrain" if is_terrain else "Object"
    mesh_scale = 1.0 if options.object_size == "original" else float(scalar(result, "smpl_scale", 1))
    asset_group = "terrain" if is_terrain else "objects"
    usd_path = options.usd_dir / asset_group / options.object_size / subset_root.name / f"{recon_path.stem}.usd"
    parts = coacd_parts(recon_path, result, usd_source)
    if not usd_path.exists() or not has_separate_convex_collision(usd_path, len(parts)):
        source_stage = Usd.Stage.Open(str(usd_source))
        if source_stage is None:
            raise ValueError(f"Cannot open GRAIL USD: {usd_source}")
        unit_scale = float(UsdGeom.GetStageMetersPerUnit(source_stage))
        collision_meshes = coacd_meshes(parts, unit_scale * mesh_scale)
        write_usd_atomic(usd_path, prim_name, usd_meshes(usd_source, mesh_scale), collision_meshes)
    pos, rot = object_pose(recon_path, frame_ids, result)
    static = bool(np.allclose(pos, pos[0], atol=1e-5) and np.allclose(rot, rot[0], atol=1e-5))
    if is_terrain and not static:
        raise ValueError(f"GRAIL terrain has a moving pose: {recon_path}")
    source_label = recon_path.stem.split("__", 2)[1] if "__" in recon_path.stem else recon_path.stem
    label_parts = source_label.rsplit("_", 1)
    semantic_label = subset_root.name if is_terrain else (label_parts[0] if len(label_parts) == 2 and label_parts[1].isdigit() else source_label)
    entities = [{
        "entity_id": recon_path.stem,
        "entity_type": entity_type,
        "usd_asset": usd_path.relative_to(options.output_root).as_posix(),
        "usd_prim_path": f"/{prim_name}",
        "pose_pos": pos,
        "pose_rot": rot,
        "mobility": "static" if static else "dynamic",
        "semantic_label": semantic_label,
    }]
    for index in range(model.ngeom):
        if model.geom_type[index] != mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if model.geom_bodyid[index] != 0:
            raise ValueError("Moving MuJoCo floor planes need an explicit pose track")
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or f"floor_{index}"
        half_x, half_y = map(float, model.geom_size[index, :2])
        if half_x <= 0 or half_y <= 0:
            continue
        floor_path = options.usd_dir / "terrain" / f"{name}_{half_x:g}_{half_y:g}.usd"
        if not floor_path.exists():
            points = np.asarray([[-half_x, -half_y, 0], [half_x, -half_y, 0],
                                 [half_x, half_y, 0], [-half_x, half_y, 0]], dtype=np.float32)
            make_usd(floor_path, "Terrain", [(points, np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32))])
        quat, _ = as_wxyz_rotation(np.repeat(model.geom_quat[index][None, :], len(frame_ids), axis=0))
        entities.append({
            "entity_id": name,
            "entity_type": "terrain",
            "usd_asset": floor_path.relative_to(options.output_root).as_posix(),
            "usd_prim_path": "/Terrain",
            "pose_pos": np.repeat(model.geom_pos[index][None, :], len(frame_ids), axis=0).astype(np.float32),
            "pose_rot": quat,
            "mobility": "static",
            "semantic_label": "floor",
        })
    return {"fps": float(fps), "scene_entities": entities}


def convert(path: Path, options: argparse.Namespace) -> None:
    with np.load(path, allow_pickle=True) as result:
        if not str(scalar(result, "source_format", "")).startswith("grail_"):
            raise ValueError(f"Not a GRAIL retarget result: {path}")
        qpos = np.asarray(result["qpos"], dtype=np.float64)
        frame_ids = np.asarray(result["frame_ids"], dtype=np.int32)
        fps = float(scalar(result, "fps"))
        if not np.isfinite(fps) or fps <= 0 or qpos.ndim != 2 or len(qpos) != len(frame_ids) or not len(qpos):
            raise ValueError(f"Invalid motion shape or fps: {path}")
        model = mujoco.MjModel.from_xml_path(str(scalar(result, "robot_xml")))
        joints = [str(value) for value in result["robot_joint_names"].tolist()]
        motion = make_motion(model, qpos, fps, joints)
        scene = make_scene(model, result, frame_ids, fps, options)
        bundled_model = options.output_root / "robot" / "igris_c.mjb"
        if not bundled_model.exists():
            bundled_model.parent.mkdir(parents=True, exist_ok=True)
            mujoco.mj_saveModel(model, str(bundled_model))
        motion["retarget_meta"] = {"robot_model": bundled_model.relative_to(options.output_root).as_posix()}
    for directory, payload in ((options.motion_dir, motion), (options.scene_dir, scene)):
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{path.stem}.pkl").open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{path.name}: {len(qpos)} frames, {len(scene['scene_entities'])} entities")


def main() -> None:
    options = parse_args()
    source = options.input.expanduser().resolve()
    files = [source] if source.is_file() else sorted(source.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ files found: {source}")
    if options.limit:
        files = files[:options.limit]
    for path in files:
        convert(path, options)


if __name__ == "__main__":
    main()
