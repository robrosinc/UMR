#!/usr/bin/env python3
"""Convert IGRIS-C OMOMO retarget NPZ files to separate canonical motion/scene PKLs."""

from __future__ import annotations

import argparse
import csv
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from pxr import Usd, UsdGeom, Vt
from scipy.spatial.transform import Rotation


KEYBODIES = (
    "Link_Waist_Yaw", "Link_Waist_Roll", "Link_Waist_Pitch",
    "Link_Hip_Pitch_Left", "Link_Hip_Roll_Left", "Link_Hip_Yaw_Left",
    "Link_Knee_Pitch_Left", "Link_Ankle_Pitch_Left", "Link_Ankle_Roll_Left",
    "Link_Hip_Pitch_Right", "Link_Hip_Roll_Right", "Link_Hip_Yaw_Right",
    "Link_Knee_Pitch_Right", "Link_Ankle_Pitch_Right", "Link_Ankle_Roll_Right",
    "Link_Shoulder_Pitch_Left", "Link_Shoulder_Roll_Left", "Link_Shoulder_Yaw_Left",
    "Link_Elbow_Pitch_Left", "Link_Wrist_Yaw_Left", "Link_Wrist_Roll_Left",
    "Link_Wrist_Pitch_Left", "Left_Hand", "Link_Shoulder_Pitch_Right",
    "Link_Shoulder_Roll_Right", "Link_Shoulder_Yaw_Right", "Link_Elbow_Pitch_Right",
    "Link_Wrist_Yaw_Right", "Link_Wrist_Roll_Right", "Link_Wrist_Pitch_Right",
    "Right_Hand", "Link_Neck_Yaw", "Link_Neck_Pitch",
)
CANONICAL_JOINTS = tuple(
    "Joint_" + name.removeprefix("Link_") for name in KEYBODIES if name.startswith("Link_")
)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One retarget NPZ or its directory")
    parser.add_argument("--output-root", type=Path, required=True, help="Dataset root for relative USD paths")
    parser.add_argument("--motion-dir", type=Path, default=Path("motion"))
    parser.add_argument("--scene-dir", type=Path, default=Path("scene_tracks"))
    parser.add_argument("--usd-dir", type=Path, default=Path("usd_assets"))
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of NPZ files; 0 means all")
    result = parser.parse_args()
    if result.limit < 0:
        parser.error("--limit must be nonnegative")
    result.output_root = result.output_root.expanduser().resolve()
    for key in ("motion_dir", "scene_dir", "usd_dir"):
        value = getattr(result, key)
        path = (value if value.is_absolute() else result.output_root / value).resolve()
        if not path.is_relative_to(result.output_root):
            parser.error(f"--{key.replace('_', '-')} must be inside --output-root")
        setattr(result, key, path)
    if len({result.motion_dir, result.scene_dir, result.usd_dir}) != 3:
        parser.error("motion, scene, and USD directories must be different")
    return result


def as_wxyz_rotation(quats: np.ndarray) -> tuple[np.ndarray, Rotation]:
    quats = np.asarray(quats, dtype=np.float64).copy()
    if quats.ndim != 2 or quats.shape[1] != 4 or not np.isfinite(quats).all():
        raise ValueError("Expected finite wxyz quaternions of shape (T, 4)")
    norms = np.linalg.norm(quats, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError("Zero quaternion in pose track")
    quats /= norms[:, None]
    for i in range(1, len(quats)):
        if np.dot(quats[i - 1], quats[i]) < 0:
            quats[i] *= -1
    return quats.astype(np.float32), Rotation.from_quat(quats[:, [1, 2, 3, 0]])


def derivative(values: np.ndarray, fps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(values)
    if len(values) > 1:
        out[0] = (values[1] - values[0]) * fps
        out[-1] = (values[-1] - values[-2]) * fps
        if len(values) > 2:
            out[1:-1] = (values[2:] - values[:-2]) * (fps / 2)
    return out.astype(np.float32)


def angular_velocity(rotations: Rotation, fps: float) -> np.ndarray:
    count = len(rotations)
    out = np.zeros((count, 3), dtype=np.float64)
    if count > 1:
        out[0] = (rotations[0].inv() * rotations[1]).as_rotvec() * fps
        out[-1] = (rotations[-2].inv() * rotations[-1]).as_rotvec() * fps
        if count > 2:
            out[1:-1] = (rotations[:-2].inv() * rotations[2:]).as_rotvec() * (fps / 2)
    return out.astype(np.float32)


def make_motion(model: mujoco.MjModel, qpos: np.ndarray, fps: float, joint_names: list[str]) -> dict:
    count = len(qpos)
    if model.nq != 38 or qpos.shape != (count, 38):
        raise ValueError(f"Expected IGRIS-C qpos (T, 38), got {qpos.shape}, model nq={model.nq}")
    expected_joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                       for i in range(1, model.njnt)]
    if set(joint_names) != set(CANONICAL_JOINTS) or set(expected_joints) != set(CANONICAL_JOINTS) or len(joint_names) != 31:
        raise ValueError("Robot joint names do not match the IGRIS-C model")
    # The qpos columns follow the XML model's joint order; emit canonical name order.
    dof_columns = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
                   for name in CANONICAL_JOINTS]
    root_rot, root_rotation = as_wxyz_rotation(qpos[:, 3:7])
    root_pos = qpos[:, :3].astype(np.float32)
    dof_pos = qpos[:, dof_columns].astype(np.float32)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in KEYBODIES]
    if min(ids) < 0:
        raise ValueError("IGRIS-C model is missing a canonical keybody")
    data = mujoco.MjData(model)
    world_pos = np.empty((count, len(ids), 3), dtype=np.float32)
    world_rot = np.empty((count, len(ids), 4), dtype=np.float32)
    for i, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        world_pos[i] = data.xpos[ids]
        world_rot[i] = data.xquat[ids]
    root_repeated = Rotation.from_quat(np.repeat(root_rot[:, [1, 2, 3, 0]], len(ids), axis=0))
    local_pos = root_repeated.inv().apply(
        (world_pos - root_pos[:, None, :]).reshape(-1, 3)
    ).reshape(count, len(ids), 3).astype(np.float32)
    body_quats = world_rot.reshape(-1, 4)
    body_rotation = Rotation.from_quat(body_quats[:, [1, 2, 3, 0]])
    relative = (root_repeated.inv() * body_rotation).as_quat().reshape(count, len(ids), 4)
    local_rot = relative[..., [3, 0, 1, 2]].astype(np.float32)
    return {
        "fps": float(fps),
        "root_pos": root_pos,
        "root_rot": root_rot,
        "root_vel": derivative(root_pos, fps),
        "root_angvel": angular_velocity(root_rotation, fps),
        "dof_pos": dof_pos,
        "dof_vel": derivative(dof_pos, fps),
        "keybody_pos_world": world_pos,
        "keybody_rot_world": world_rot,
        "keybody_pos_local": local_pos,
        "keybody_rot_local": local_rot,
        "keybody_pos": world_pos.copy(),
        "link_body_list": list(KEYBODIES),
        "local_body_link_body_list": None,
        "local_body_pos": None,
    }


def scalar(data, key: str, default=None):
    return np.asarray(data[key]).reshape(-1)[0].item() if key in data else default


def object_pose(csv_path: Path, frame_ids: np.ndarray, result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"px", "py", "pz", "qx", "qy", "qz", "qw"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid object pose CSV: {csv_path}")
    if frame_ids.min() < 0 or frame_ids.max() >= len(rows):
        raise ValueError(f"Object pose CSV does not cover all result frame_ids: {csv_path}")
    selected = [rows[i] for i in frame_ids]
    pos = np.asarray([[float(row[k]) for k in ("px", "py", "pz")] for row in selected], dtype=np.float64)
    xyzw = np.asarray([[float(row[k]) for k in ("qx", "qy", "qz", "qw")] for row in selected], dtype=np.float64)
    output_up = str(scalar(result, "noitom_output_up", "z")).lower()
    convert_y = bool(scalar(result, "noitom_convert_y_up", True))
    if output_up not in ("y", "z"):
        raise ValueError(f"Unsupported OMOMO up axis: {output_up}")
    basis = np.eye(3) if output_up == "z" or not convert_y else np.asarray(
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    pos = pos @ basis.T
    if bool(scalar(result, "noitom_ground_align", False)):
        pos[:, 2] -= float(scalar(result, "noitom_floor_y", 0))
    pos[:, 2] += float(scalar(result, "noitom_ground_offset", 0))
    pos *= float(scalar(result, "smpl_scale", 1))
    rot_mats = basis @ Rotation.from_quat(xyzw).as_matrix() @ basis.T
    xyzw = Rotation.from_matrix(rot_mats).as_quat()
    wxyz, _ = as_wxyz_rotation(xyzw[:, [3, 0, 1, 2]])
    return pos.astype(np.float32), wxyz, basis


def make_usd(path: Path, root_name: str, meshes: list[tuple[np.ndarray, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{root_name}")
    stage.SetDefaultPrim(root.GetPrim())
    for index, (vertices, triangles) in enumerate(meshes):
        mesh = UsdGeom.Mesh.Define(stage, f"/{root_name}/Mesh_{index}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices, dtype=np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(triangles), 3, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(triangles, dtype=np.int32).reshape(-1)))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
    stage.GetRootLayer().Save()


def object_meshes(xml_path: Path, basis: np.ndarray, scale: float) -> list[tuple[np.ndarray, np.ndarray]]:
    xml = ET.parse(xml_path).getroot()
    assets = {node.get("name"): node for node in xml.findall(".//asset/mesh")}
    out = []
    for geom in xml.findall(".//worldbody/body/geom"):
        if geom.get("type") != "mesh" or geom.get("group") != "2":
            continue
        asset = assets[geom.get("mesh")]
        source = (xml_path.parent / asset.get("file")).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        mesh = trimesh.load(source, force="mesh", process=False)
        mesh_scale = np.fromstring(asset.get("scale", "1 1 1"), sep=" ", dtype=np.float64)
        if mesh_scale.size != 3:
            raise ValueError(f"Invalid mesh scale in {xml_path}")
        geom_pos = np.fromstring(geom.get("pos", "0 0 0"), sep=" ", dtype=np.float64)
        geom_quat = np.fromstring(geom.get("quat", "1 0 0 0"), sep=" ", dtype=np.float64)
        vertices = Rotation.from_quat(geom_quat[[1, 2, 3, 0]]).apply(np.asarray(mesh.vertices) * mesh_scale)
        vertices = (vertices + geom_pos) @ basis.T * scale
        out.append((vertices, np.asarray(mesh.faces, dtype=np.int32)))
    if not out:
        raise ValueError(f"No visual mesh geoms in {xml_path}")
    return out


def make_scene(model: mujoco.MjModel, result, frame_ids: np.ndarray, fps: float,
               usd_dir: Path, root: Path) -> dict:
    source_dir = Path(str(scalar(result, "source_object_dir", scalar(result, "source_data"))))
    if not source_dir.is_dir():
        raise FileNotFoundError(f"OMOMO source object directory missing: {source_dir}")
    entities = []
    scale = float(scalar(result, "smpl_scale", 1))
    for csv_path in sorted(source_dir.glob("prop_*.csv")):
        name = csv_path.stem.removeprefix("prop_")
        xml_path = source_dir / f"{name}.xml"
        if not xml_path.is_file():
            raise FileNotFoundError(xml_path)
        pos, rot, basis = object_pose(csv_path, frame_ids, result)
        usd_path = usd_dir / "objects" / source_dir.name / f"{name}.usda"
        if not usd_path.exists():
            make_usd(usd_path, "Object", object_meshes(xml_path, basis, scale))
        entities.append({
            "entity_id": name, "entity_type": "object", "usd_asset": usd_path.relative_to(root).as_posix(),
            "usd_prim_path": "/Object", "pose_pos": pos, "pose_rot": rot,
            "mobility": "dynamic", "semantic_label": name.removesuffix("_bottom"),
        })
    # MuJoCo's floor plane has a finite visual extent given by geom_size[:2].
    for i in range(model.ngeom):
        if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if model.geom_bodyid[i] != 0:
            raise ValueError("Moving MuJoCo floor planes need an explicit terrain pose track")
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"floor_{i}"
        half_x, half_y = map(float, model.geom_size[i, :2])
        if half_x <= 0 or half_y <= 0:
            continue
        usd_path = usd_dir / "terrain" / f"{name}_{half_x:g}_{half_y:g}.usda"
        if not usd_path.exists():
            points = np.asarray([[-half_x, -half_y, 0], [half_x, -half_y, 0],
                                 [half_x, half_y, 0], [-half_x, half_y, 0]], dtype=np.float32)
            faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
            make_usd(usd_path, "Terrain", [(points, faces)])
        quat = np.repeat(model.geom_quat[i][None, :], len(frame_ids), axis=0)
        quat, _ = as_wxyz_rotation(quat)
        entities.append({
            "entity_id": name, "entity_type": "terrain", "usd_asset": usd_path.relative_to(root).as_posix(),
            "usd_prim_path": "/Terrain", "pose_pos": np.repeat(model.geom_pos[i][None, :], len(frame_ids), axis=0).astype(np.float32),
            "pose_rot": quat, "mobility": "static", "semantic_label": "floor",
        })
    return {"fps": float(fps), "scene_entities": entities}


def convert(path: Path, options: argparse.Namespace) -> None:
    with np.load(path, allow_pickle=True) as result:
        qpos = np.asarray(result["qpos"], dtype=np.float64)
        frame_ids = np.asarray(result["frame_ids"], dtype=np.int32)
        fps = float(scalar(result, "fps"))
        if not np.isfinite(fps) or fps <= 0 or qpos.ndim != 2 or len(qpos) != len(frame_ids) or len(qpos) == 0:
            raise ValueError(f"Invalid motion shape or fps: {path}")
        model_path = Path(str(scalar(result, "robot_xml")))
        model = mujoco.MjModel.from_xml_path(str(model_path))
        bundled_model = options.output_root / "robot" / "igris_c.mjb"
        if not bundled_model.exists():
            bundled_model.parent.mkdir(parents=True, exist_ok=True)
            mujoco.mj_saveModel(model, str(bundled_model))
        joint_names = [str(v) for v in result["robot_joint_names"].tolist()]
        motion = make_motion(model, qpos, fps, joint_names)
        motion["retarget_meta"] = {"robot_model": bundled_model.relative_to(options.output_root).as_posix()}
        scene = make_scene(model, result, frame_ids, fps, options.usd_dir, options.output_root)
    for directory, payload in ((options.motion_dir, motion), (options.scene_dir, scene)):
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{path.stem}.pkl").open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{path.name}: {len(qpos)} frames, {len(scene['scene_entities'])} entities")


def main() -> None:
    options = args()
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
