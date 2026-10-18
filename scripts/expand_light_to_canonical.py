#!/usr/bin/env python3
"""Expand a self-contained light-form directory to IGRIS-C canonical form.

Only light-form fields are read. Source dataset names and source paths have no
meaning here. Object geometry is shared; a small USD instance applies each
clip's final mesh scale to both visual and collision geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics, Vt
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Light-form root containing clips/")
    parser.add_argument("--output-dir", type=Path, required=True, help="Canonical dataset root")
    parser.add_argument("--limit", type=int, default=0, help="First N clips; 0 converts all")
    parser.add_argument("--clip", help="Expand one clip by NPZ stem (use a clean output directory)")
    parser.add_argument("--clip-path", type=Path, help="Expand one NPZ by its path relative to clips/")
    args = parser.parse_args()
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.clip and args.clip_path:
        parser.error("--clip and --clip-path cannot be combined")
    if args.input_dir == args.output_dir or args.output_dir.is_relative_to(args.input_dir):
        parser.error("--output-dir must be outside --input-dir")
    if not (args.input_dir / "clips").is_dir():
        parser.error(f"Light-form clips/ directory missing: {args.input_dir}")
    return args


def scalar(data, key: str):
    return np.asarray(data[key]).reshape(-1)[0].item()


def resolve_ref(clip: Path, root: Path, ref: str) -> Path:
    path = Path(ref)
    if path.is_absolute():
        raise ValueError(f"Absolute light-form reference in {clip}: {ref}")
    target = (clip.parent / path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise FileNotFoundError(f"Missing or external light-form reference in {clip}: {ref}")
    return target


def quaternion_track(quats: np.ndarray) -> tuple[np.ndarray, Rotation]:
    quats = np.asarray(quats, dtype=np.float64).copy()
    if quats.ndim != 2 or quats.shape[1] != 4 or not np.isfinite(quats).all():
        raise ValueError("Expected finite wxyz quaternions of shape (T, 4)")
    norms = np.linalg.norm(quats, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError("Zero quaternion in pose track")
    quats /= norms[:, None]
    for index in range(1, len(quats)):
        if np.dot(quats[index - 1], quats[index]) < 0:
            quats[index] *= -1
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
    model_joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                         for i in range(1, model.njnt)]
    if (len(joint_names) != 31 or set(joint_names) != set(CANONICAL_JOINTS)
            or set(model_joint_names) != set(CANONICAL_JOINTS)):
        raise ValueError("Robot joint names do not match the IGRIS-C model")
    columns = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
               for name in CANONICAL_JOINTS]
    root_rot, root_rotation = quaternion_track(qpos[:, 3:7])
    root_pos = qpos[:, :3].astype(np.float32)
    dof_pos = qpos[:, columns].astype(np.float32)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in KEYBODIES]
    if min(ids) < 0:
        raise ValueError("IGRIS-C model is missing a canonical keybody")
    data = mujoco.MjData(model)
    world_pos = np.empty((count, len(ids), 3), dtype=np.float32)
    world_rot = np.empty((count, len(ids), 4), dtype=np.float32)
    for index, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        world_pos[index] = data.xpos[ids]
        world_rot[index] = data.xquat[ids]
    root_repeated = Rotation.from_quat(np.repeat(root_rot[:, [1, 2, 3, 0]], len(ids), axis=0))
    local_pos = root_repeated.inv().apply((world_pos - root_pos[:, None, :]).reshape(-1, 3))
    local_pos = local_pos.reshape(count, len(ids), 3).astype(np.float32)
    body_rotation = Rotation.from_quat(world_rot.reshape(-1, 4)[:, [1, 2, 3, 0]])
    relative = (root_repeated.inv() * body_rotation).as_quat().reshape(count, len(ids), 4)
    local_rot = relative[..., [3, 0, 1, 2]].astype(np.float32)
    return {
        "fps": float(fps), "root_pos": root_pos, "root_rot": root_rot,
        "root_vel": derivative(root_pos, fps), "root_angvel": angular_velocity(root_rotation, fps),
        "dof_pos": dof_pos, "dof_vel": derivative(dof_pos, fps),
        "keybody_pos_world": world_pos, "keybody_rot_world": world_rot,
        "keybody_pos_local": local_pos, "keybody_rot_local": local_rot,
        "keybody_pos": world_pos.copy(), "link_body_list": list(KEYBODIES),
        "local_body_link_body_list": None, "local_body_pos": None,
    }


def new_stage(path: Path, root_name: str) -> tuple[Usd.Stage, UsdGeom.Xform]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{root_name}")
    stage.SetDefaultPrim(root.GetPrim())
    return stage, root


def add_mesh(stage: Usd.Stage, path: str, vertices: np.ndarray, faces: np.ndarray,
             collision: bool, convex: bool = False) -> None:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices, dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(faces, dtype=np.int32).reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    if collision:
        if convex:
            mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
            mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
            UsdPhysics.Tokens.convexHull if convex else UsdPhysics.Tokens.none)


def parse_object_meshes(xml_path: Path, light_root: Path, group: str) -> list[tuple[np.ndarray, np.ndarray]]:
    xml = ET.parse(xml_path).getroot()
    assets = {node.get("name"): node for node in xml.findall("./asset/mesh")}
    compiler = xml.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    if Path(meshdir).is_absolute() or ".." in Path(meshdir).parts:
        raise ValueError(f"Invalid object meshdir in {xml_path}: {meshdir}")
    out = []
    for geom in xml.findall(".//worldbody/body/geom"):
        if geom.get("type") != "mesh" or geom.get("group") != group:
            continue
        asset = assets.get(geom.get("mesh"))
        if asset is None or not asset.get("file"):
            raise ValueError(f"Missing mesh asset in {xml_path}: {geom.get('mesh')}")
        ref = Path(asset.get("file"))
        if ref.is_absolute():
            raise ValueError(f"Absolute mesh path in {xml_path}: {ref}")
        mesh_path = (xml_path.parent / meshdir / ref).resolve()
        if not mesh_path.is_relative_to(light_root) or not mesh_path.is_file():
            raise FileNotFoundError(f"Missing or external mesh in {xml_path}: {ref}")
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        mesh_scale = np.fromstring(asset.get("scale", "1 1 1"), sep=" ", dtype=np.float64)
        geom_pos = np.fromstring(geom.get("pos", "0 0 0"), sep=" ", dtype=np.float64)
        geom_quat = np.fromstring(geom.get("quat", "1 0 0 0"), sep=" ", dtype=np.float64)
        if mesh_scale.shape != (3,) or geom_pos.shape != (3,) or geom_quat.shape != (4,):
            raise ValueError(f"Invalid mesh transform in {xml_path}")
        vertices = Rotation.from_quat(geom_quat[[1, 2, 3, 0]]).apply(
            np.asarray(mesh.vertices, dtype=np.float64) * mesh_scale) + geom_pos
        out.append((vertices, np.asarray(mesh.faces, dtype=np.int32)))
    return out


def write_base_object(xml_path: Path, light_root: Path, output: Path) -> Path:
    digest = hashlib.sha256(
        xml_path.relative_to(light_root).as_posix().encode() + b"\0" + xml_path.read_bytes()
    ).hexdigest()[:24]
    path = output / "usd_assets/objects/base" / f"{digest}.usd"
    if path.is_file():
        return path
    visual = parse_object_meshes(xml_path, light_root, "2")
    collision = parse_object_meshes(xml_path, light_root, "3")
    if not visual or not collision:
        raise ValueError(f"Object must have visual and collision mesh groups: {xml_path}")
    stage, _ = new_stage(path, "Object")
    for index, (vertices, faces) in enumerate(visual):
        add_mesh(stage, f"/Object/Mesh_{index}", vertices, faces, False)
    UsdGeom.Scope.Define(stage, "/Object/Collision")
    for index, (vertices, faces) in enumerate(collision):
        add_mesh(stage, f"/Object/Collision/Part_{index}", vertices, faces, True, True)
    stage.GetRootLayer().Save()
    return path


def write_scaled_object(base: Path, scale: float, output: Path) -> Path:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid object scale: {scale}")
    # The stored light scale is float32; its exact bits define asset identity.
    key = hashlib.sha256(base.stem.encode() + np.float32(scale).tobytes()).hexdigest()[:24]
    path = output / "usd_assets/objects/instances" / f"{key}.usd"
    if path.is_file():
        return path
    stage, _ = new_stage(path, "Object")
    child = UsdGeom.Xform.Define(stage, "/Object/Geometry")
    child.AddScaleOp().Set((float(scale),) * 3)
    child.GetPrim().GetReferences().AddReference(os.path.relpath(base, path.parent).replace(os.sep, "/"), "/Object")
    stage.GetRootLayer().Save()
    return path


def write_plane(size: list[float], output: Path) -> Path:
    if len(size) < 2 or not np.isfinite(size).all() or size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"Invalid plane size: {size}")
    half_x, half_y = map(float, size[:2])
    key = hashlib.sha256(np.asarray([half_x, half_y], dtype=np.float64).tobytes()).hexdigest()[:24]
    path = output / "usd_assets/terrain" / f"plane_{key}.usd"
    if path.is_file():
        return path
    vertices = np.asarray([[-half_x, -half_y, 0], [half_x, -half_y, 0],
                           [half_x, half_y, 0], [-half_x, half_y, 0]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    stage, _ = new_stage(path, "Terrain")
    add_mesh(stage, "/Terrain/Mesh_0", vertices, faces, True)
    stage.GetRootLayer().Save()
    return path


def make_scene(clip: Path, result, light_root: Path, output: Path, frames: int, fps: float) -> dict:
    object_keys = ("light_object_names", "light_object_xml_refs", "light_object_scales",
                   "light_object_pos", "light_object_rot")
    present = [key in result for key in object_keys]
    if any(present) and not all(present):
        raise ValueError(f"Incomplete light object arrays: {clip}")
    names = np.asarray(result["light_object_names"], dtype=str) if all(present) else np.empty(0, dtype=str)
    refs = np.asarray(result["light_object_xml_refs"], dtype=str) if all(present) else np.empty(0, dtype=str)
    scales = np.asarray(result["light_object_scales"], dtype=np.float32) if all(present) else np.empty(0, dtype=np.float32)
    positions = np.asarray(result["light_object_pos"], dtype=np.float32) if all(present) else np.empty((frames, 0, 3), dtype=np.float32)
    rotations = np.asarray(result["light_object_rot"], dtype=np.float32) if all(present) else np.empty((frames, 0, 4), dtype=np.float32)
    count = len(names)
    if (refs.shape != (count,) or scales.shape != (count,)
            or positions.shape != (frames, count, 3)
            or rotations.shape != (frames, count, 4)):
        raise ValueError(f"Invalid light object arrays: {clip}")
    if not np.isfinite(positions).all():
        raise ValueError(f"Nonfinite light object positions: {clip}")
    entities = []
    used_ids = set()
    for index, name in enumerate(names):
        name = str(name)
        if not name or name in used_ids:
            raise ValueError(f"Duplicate or empty scene entity ID: {name!r} in {clip}")
        used_ids.add(name)
        xml_path = resolve_ref(clip, light_root, str(refs[index]))
        base = write_base_object(xml_path, light_root, output)
        asset = write_scaled_object(base, float(scales[index]), output)
        rotation, _ = quaternion_track(rotations[:, index])
        entities.append({
            "entity_id": name, "entity_type": "object",
            "usd_asset": asset.relative_to(output).as_posix(), "usd_prim_path": "/Object",
            "pose_pos": positions[:, index].copy(), "pose_rot": rotation,
            "mobility": "dynamic", "semantic_label": name.removesuffix("_bottom"),
        })
    if "light_scene_ref" in result:
        scene_path = resolve_ref(clip, light_root, str(scalar(result, "light_scene_ref")))
        scene = json.loads(scene_path.read_text())
        if scene.get("version") != 1:
            raise ValueError(f"Unsupported light scene version: {scene_path}")
    else:
        scene_path = clip
        scene = {"static_entities": []}
    for item in scene.get("static_entities", []):
        entity_id = str(item["entity_id"])
        if entity_id in used_ids:
            raise ValueError(f"Duplicate scene entity ID: {entity_id} in {clip}")
        used_ids.add(entity_id)
        if item.get("entity_type") != "terrain" or item.get("primitive") != "plane":
            raise ValueError(f"Unsupported light scene entity in {scene_path}: {item}")
        asset = write_plane(item["size"], output)
        pos = np.asarray(item.get("pos", [0, 0, 0]), dtype=np.float32)
        rot = np.asarray(item.get("rot_wxyz", [1, 0, 0, 0]), dtype=np.float32)
        if pos.shape != (3,) or not np.isfinite(pos).all():
            raise ValueError(f"Invalid static terrain position in {scene_path}")
        rot, _ = quaternion_track(rot[None, :])
        entities.append({
            "entity_id": entity_id, "entity_type": "terrain",
            "usd_asset": asset.relative_to(output).as_posix(), "usd_prim_path": "/Terrain",
            "pose_pos": np.repeat(pos[None, :], frames, axis=0),
            "pose_rot": np.repeat(rot, frames, axis=0),
            "mobility": "static", "semantic_label": "floor",
        })
    return {"fps": fps, "scene_entities": entities}


def save_pickle(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def write_manifest(output: Path, clips: list[str]) -> None:
    objects = {}
    for name in clips:
        scene_path = output / "scene_tracks" / f"{name}.pkl"
        with scene_path.open("rb") as handle:
            scene = pickle.load(handle)
        for entity in scene["scene_entities"]:
            if entity["entity_type"] != "object":
                continue
            asset = entity["usd_asset"]
            object_id = Path(asset).relative_to("usd_assets/objects").with_suffix("").as_posix()
            entry = objects.setdefault(object_id, {
                "usd_path": asset, "usd_prim_path": entity["usd_prim_path"], "clips": []})
            entry["clips"].append({
                "motion_path": f"motion/{name}.pkl",
                "scene_track_path": f"scene_tracks/{name}.pkl",
                "entity_id": entity["entity_id"],
            })
    payload = {"schema_version": 1, "pose_source": "scene_tracks",
               "objects": dict(sorted(objects.items())),
               "motions": [f"motion/{name}.pkl" for name in clips]}
    (output / "objects_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if args.clip_path is not None:
        if args.clip_path.is_absolute() or args.clip_path.suffix.lower() != ".npz":
            raise ValueError("--clip-path must be a relative NPZ path below clips/")
        selected = (args.input_dir / "clips" / args.clip_path).resolve()
        if not selected.is_relative_to(args.input_dir / "clips") or not selected.is_file():
            raise FileNotFoundError(f"Light-form clip is missing or outside clips/: {args.clip_path}")
        clips = [selected]
    else:
        clips = sorted((args.input_dir / "clips").rglob("*.npz"))
    if args.clip:
        clips = [clip for clip in clips if clip.stem == args.clip]
    if args.limit:
        clips = clips[:args.limit]
    if not clips:
        raise FileNotFoundError(f"No light-form clips in {args.input_dir / 'clips'}")
    names = [clip.stem for clip in clips]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate clip stems; canonical motion/ uses flat clip names")
    expected = {f"{name}.pkl" for name in names}
    for directory in (args.output_dir / "motion", args.output_dir / "scene_tracks"):
        stale = {path.name for path in directory.glob("*.pkl")} - expected
        if stale:
            raise ValueError(
                f"Existing {directory} contains {len(stale)} clips outside this run; "
                f"choose a clean output directory (example: {sorted(stale)[0]})")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = {}
    for index, clip in enumerate(clips, 1):
        with np.load(clip, allow_pickle=False) as result:
            if int(scalar(result, "light_form_version")) != 1:
                raise ValueError(f"Unsupported light form version: {clip}")
            qpos = np.asarray(result["qpos"], dtype=np.float64)
            fps = float(scalar(result, "fps"))
            if qpos.ndim != 2 or len(qpos) == 0 or not np.isfinite(qpos).all() or not np.isfinite(fps) or fps <= 0:
                raise ValueError(f"Invalid qpos or fps: {clip}")
            robot_xml = resolve_ref(clip, args.input_dir, str(scalar(result, "robot_xml")))
            if robot_xml not in models:
                model = mujoco.MjModel.from_xml_path(str(robot_xml))
                digest = hashlib.sha256(robot_xml.read_bytes()).hexdigest()[:16]
                model_path = args.output_dir / "robot" / f"igris_c_{digest}.mjb"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                if not model_path.is_file():
                    mujoco.mj_saveModel(model, str(model_path))
                models[robot_xml] = (model, model_path)
            model, model_path = models[robot_xml]
            joint_names = [str(name) for name in np.asarray(result["robot_joint_names"], dtype=str)]
            motion = make_motion(model, qpos, fps, joint_names)
            motion["retarget_meta"] = {"robot_model": model_path.relative_to(args.output_dir).as_posix()}
            scene = make_scene(clip, result, args.input_dir, args.output_dir, len(qpos), fps)
        save_pickle(args.output_dir / "motion" / f"{clip.stem}.pkl", motion)
        save_pickle(args.output_dir / "scene_tracks" / f"{clip.stem}.pkl", scene)
        if index % 100 == 0 or index == len(clips):
            print(f"[{index}/{len(clips)}] clips expanded", flush=True)
    write_manifest(args.output_dir, names)
    print(f"Canonical form ready: {args.output_dir}")


if __name__ == "__main__":
    main()
