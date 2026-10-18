#!/usr/bin/env python3
"""Package OMOMO UMR results as portable, self-contained light-form clips.

The robot and object meshes are copied once. Object MJCF templates are deduplicated
after removing the clip-specific uniform mesh scale; that scale and the selected
object pose samples live in each clip NPZ. No source_data path is needed afterward.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "output/igris_c_omomo_dummy")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "light_form/igris_c_omomo")
    parser.add_argument("--sample-root", type=Path, default=ROOT / "sample_data/omomo")
    parser.add_argument("--robot-xml", type=Path, default=ROOT / "assets/igris_c/igris_c_dummy_hand.xml")
    parser.add_argument("--limit", type=int, default=0, help="Convert only the first N clips (0: all).")
    return parser.parse_args()


def scalar(result, key: str, default=None):
    return np.asarray(result[key]).reshape(-1)[0].item() if key in result else default


def source_directory(result, sample_root: Path) -> Path:
    source = str(scalar(result, "source_object_dir", scalar(result, "source_data", "")))
    marker = "/sample_data/omomo/"
    if marker not in source:
        raise ValueError(f"Expected an OMOMO source directory, got {source!r}")
    suffix = Path(source.split(marker, 1)[1])
    if suffix.is_absolute() or ".." in suffix.parts:
        raise ValueError(f"Unsafe OMOMO source directory: {source}")
    directory = sample_root / suffix
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return directory


def relative_to_clip(clip: Path, target: Path) -> str:
    return Path(os.path.relpath(target, clip.parent)).as_posix()


def install_robot(robot_xml: Path, output: Path) -> Path:
    root = ET.parse(robot_xml).getroot()
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir") if compiler is not None else None
    if not meshdir or Path(meshdir).is_absolute() or ".." in Path(meshdir).parts:
        raise ValueError(f"Unsupported robot meshdir in {robot_xml}: {meshdir}")
    target = output / "assets/robot/model.xml"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(robot_xml, target)
    source_meshes = robot_xml.parent / meshdir
    if not source_meshes.is_dir():
        raise FileNotFoundError(source_meshes)
    shutil.copytree(source_meshes, target.parent / meshdir, dirs_exist_ok=True)
    return target


def install_scene(robot_xml: Path, output: Path) -> Path:
    root = ET.parse(robot_xml).getroot()
    entities = []
    for index, geom in enumerate(root.findall("./worldbody/geom")):
        if geom.get("type") != "plane":
            continue
        entities.append({
            "entity_id": geom.get("name", f"plane_{index}"),
            "entity_type": "terrain",
            "primitive": "plane",
            "size": [float(value) for value in geom.get("size", "1 1 0.05").split()],
            "pos": [float(value) for value in geom.get("pos", "0 0 0").split()],
            "rot_wxyz": [float(value) for value in geom.get("quat", "1 0 0 0").split()],
        })
    scene = output / "scenes/plane.json"
    scene.parent.mkdir(parents=True, exist_ok=True)
    scene.write_text(json.dumps({"version": 1, "static_entities": entities}, indent=2) + "\n")
    return scene


def install_object_template(xml_path: Path, sample_root: Path, output: Path,
                            copied_meshes: set[Path], templates: dict[str, Path]) -> tuple[Path, float]:
    root = ET.parse(xml_path).getroot()
    asset_dir = (sample_root / "object_mjcf/assets").resolve()
    target_mesh_dir = output / "assets/objects/meshes"
    scales = []
    for mesh in root.findall("./asset/mesh"):
        if not mesh.get("file"):
            continue
        source = (xml_path.parent / mesh.get("file")).resolve()
        if not source.is_relative_to(asset_dir) or not source.is_file():
            raise ValueError(f"OMOMO mesh is missing or outside object_mjcf/assets: {source}")
        values = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ", dtype=np.float64)
        if values.shape != (3,) or not np.allclose(values, values[0], atol=1e-7, rtol=0):
            raise ValueError(f"Nonuniform object mesh scale in {xml_path}")
        scales.append(float(values[0]))
        mesh.set("scale", "1 1 1")
        mesh.set("file", f"../meshes/{source.name}")
        target = target_mesh_dir / source.name
        if source not in copied_meshes:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size != source.stat().st_size:
                raise ValueError(f"Object mesh filename collision: {target}")
            if not target.exists():
                shutil.copy2(source, target)
            copied_meshes.add(source)
    if not scales or not np.allclose(scales, scales[0], atol=1e-7, rtol=0):
        raise ValueError(f"Object XML needs one uniform scale for all meshes: {xml_path}")
    payload = ET.tostring(root, encoding="utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if digest not in templates:
        path = output / "assets/objects/templates" / f"{xml_path.stem}_{digest[:16]}.xml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        templates[digest] = path
    return templates[digest], scales[0]


def object_pose(csv_path: Path, frame_ids: np.ndarray, result) -> tuple[np.ndarray, np.ndarray]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"px", "py", "pz", "qx", "qy", "qz", "qw"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid object trajectory: {csv_path}")
    if frame_ids.size == 0 or frame_ids.min() < 0 or frame_ids.max() >= len(rows):
        raise ValueError(f"Object trajectory does not cover frame_ids: {csv_path}")
    selected = [rows[int(index)] for index in frame_ids]
    position = np.asarray([[float(row[k]) for k in ("px", "py", "pz")] for row in selected], dtype=np.float64)
    xyzw = np.asarray([[float(row[k]) for k in ("qx", "qy", "qz", "qw")] for row in selected], dtype=np.float64)
    output_up = str(scalar(result, "noitom_output_up", "z")).lower()
    if output_up not in ("y", "z"):
        raise ValueError(f"Unsupported OMOMO up axis: {output_up}")
    convert_y = bool(scalar(result, "noitom_convert_y_up", True))
    basis = np.eye(3) if output_up == "z" or not convert_y else np.asarray(
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    position = position @ basis.T
    if bool(scalar(result, "noitom_ground_align", False)):
        position[:, 2] -= float(scalar(result, "noitom_floor_y", 0))
    position[:, 2] += float(scalar(result, "noitom_ground_offset", 0))
    position *= float(scalar(result, "smpl_scale", 1))
    matrices = basis @ Rotation.from_quat(xyzw).as_matrix() @ basis.T
    xyzw = Rotation.from_matrix(matrices).as_quat()
    wxyz = xyzw[:, [3, 0, 1, 2]]
    wxyz /= np.linalg.norm(wxyz, axis=1, keepdims=True)
    return position.astype(np.float32), wxyz.astype(np.float32)


def convert_clip(source: Path, input_dir: Path, output: Path, sample_root: Path,
                 robot: Path, scene: Path, copied_meshes: set[Path], templates: dict[str, Path]) -> int:
    clip = output / "clips" / source.relative_to(input_dir)
    with np.load(source, allow_pickle=True) as result:
        qpos = np.asarray(result["qpos"])
        frame_ids = np.asarray(result["frame_ids"], dtype=np.int32)
        if qpos.ndim != 2 or len(qpos) != len(frame_ids):
            raise ValueError(f"qpos/frame_ids mismatch: {source}")
        directory = source_directory(result, sample_root)
        names, xml_refs, scales, positions, rotations = [], [], [], [], []
        for csv_path in sorted(directory.glob("prop_*.csv")):
            name = csv_path.stem.removeprefix("prop_")
            xml_path = directory / f"{name}.xml"
            if not xml_path.is_file():
                raise FileNotFoundError(xml_path)
            template, scale = install_object_template(xml_path, sample_root, output, copied_meshes, templates)
            pos, rot = object_pose(csv_path, frame_ids, result)
            names.append(name)
            xml_refs.append(relative_to_clip(clip, template))
            scales.append(scale * float(scalar(result, "smpl_scale", 1)))
            positions.append(pos)
            rotations.append(rot)
        if not names:
            raise ValueError(f"OMOMO clip has no prop trajectory: {source}")
        # Store only portable source metadata. Absolute paths in the original NPZ
        # are replaced by clip-relative references below.
        omit = {"robot_xml", "source_data", "source_object_dir"}
        payload = {key: result[key] for key in result.files if key not in omit}
        if "robot_joint_names" in payload:
            payload["robot_joint_names"] = np.asarray(payload["robot_joint_names"], dtype=str)
        payload.update({
            "light_form_version": np.asarray(1, dtype=np.int32),
            "robot_xml": np.asarray(relative_to_clip(clip, robot)),
            "light_scene_ref": np.asarray(relative_to_clip(clip, scene)),
            "light_object_names": np.asarray(names, dtype=str),
            "light_object_xml_refs": np.asarray(xml_refs, dtype=str),
            "light_object_scales": np.asarray(scales, dtype=np.float32),
            "light_object_pos": np.stack(positions, axis=1),
            "light_object_rot": np.stack(rotations, axis=1),
        })
    clip.parent.mkdir(parents=True, exist_ok=True)
    temporary = clip.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(clip)
    return len(names)


def main() -> None:
    args = args_parser()
    input_dir = args.input_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    sample_root = args.sample_root.expanduser().resolve()
    robot_xml = args.robot_xml.expanduser().resolve()
    if output == input_dir or output.is_relative_to(input_dir):
        raise ValueError("Output must be separate from the UMR input directory")
    sources = sorted(input_dir.rglob("*.npz"))
    if args.limit > 0:
        sources = sources[:args.limit]
    if not sources:
        raise FileNotFoundError(f"No UMR NPZ clips in {input_dir}")
    expected_clips = {output / "clips" / source.relative_to(input_dir) for source in sources}
    stale_clips = set((output / "clips").rglob("*.npz")) - expected_clips
    if stale_clips:
        raise ValueError(
            f"Output contains {len(stale_clips)} clips outside this conversion. "
            f"Choose a clean output directory, for example: {next(iter(stale_clips))}"
        )
    output.mkdir(parents=True, exist_ok=True)
    robot = install_robot(robot_xml, output)
    scene = install_scene(robot_xml, output)
    copied_meshes: set[Path] = set()
    templates: dict[str, Path] = {}
    objects = 0
    for index, source in enumerate(sources, 1):
        objects += convert_clip(source, input_dir, output, sample_root, robot, scene, copied_meshes, templates)
        if index % 500 == 0 or index == len(sources):
            print(f"[{index}/{len(sources)}] clips, {objects} object instances, "
                  f"{len(templates)} templates, {len(copied_meshes)} shared meshes", flush=True)
    manifest = {
        "format": "umr_light", "version": 1, "source": "omomo",
        "clip_count": len(sources), "object_instances": objects,
        "object_templates": len(templates), "shared_meshes": len(copied_meshes),
        "clip_paths_are_relative_to_npz": True,
        "units": "meters", "up_axis": "Z", "quaternion_order": "wxyz",
    }
    (output / "format.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Light form ready: {output}")


if __name__ == "__main__":
    main()
