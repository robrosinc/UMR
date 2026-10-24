#!/usr/bin/env python3
"""Package GRAIL retarget NPZs and their prepared MJCF assets as light form v1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from convert_omomo_umr_to_light import install_robot, install_scene, relative_to_clip, scalar


ROOT = Path(__file__).resolve().parents[1]


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sequence"
    return name if len(name) <= 180 else f"{name[:160]}_{hashlib.sha256(name.encode()).hexdigest()[:12]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "output/igris_c_grail")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "light_form/igris_c_grail")
    parser.add_argument("--sample-root", type=Path, default=ROOT / "sample_data/grail")
    parser.add_argument("--robot-xml", type=Path, default=ROOT / "assets/igris_c/igris_c.xml")
    parser.add_argument("--limit", type=int, default=0, help="First N clips; 0 converts all")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    for field in ("input_dir", "output_dir", "sample_root", "robot_xml"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    if args.output_dir == args.input_dir or args.output_dir.is_relative_to(args.input_dir):
        parser.error("--output-dir must be outside --input-dir")
    return args


def source_assets(data, sample_root: Path) -> tuple[str, str, Path]:
    saved = Path(str(scalar(data, "source_data", "")))
    key = str(scalar(data, "source_sequence_key", saved.stem))
    if saved.suffix != ".pkl" or saved.parent.name != "recon" or saved.stem != key:
        raise ValueError(f"Expected GRAIL recon/<sequence>.pkl source, got {saved}")
    subset = saved.parent.parent.name
    if subset in {"", ".", ".."} or Path(subset).name != subset:
        raise ValueError(f"Invalid GRAIL subset: {subset}")
    # Do not follow the absolute source_data/source_object_dir saved in a retarget NPZ.
    # The caller can relocate the prepared dataset with --sample-root.
    directory = (sample_root / subset / "object_mjcf" / safe_name(key)).resolve()
    if not directory.is_relative_to(sample_root) or not directory.is_dir():
        raise FileNotFoundError(f"Prepared GRAIL object assets missing: {directory}")
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    count = int(metadata.get("collision_parts", 0))
    if metadata.get("sequence_key") != key or count < 1:
        raise ValueError(f"Invalid GRAIL object metadata: {metadata_path}")
    for index in range(count):
        path = directory / f"grail_object_collision_{index}.obj"
        if not path.is_file():
            raise FileNotFoundError(path)
    return subset, key, directory


def local_file(directory: Path, reference: str) -> Path:
    ref = Path(reference)
    if not reference or ref.is_absolute() or ".." in ref.parts:
        raise ValueError(f"External object asset reference: {reference!r}")
    source = (directory / ref).resolve()
    if not source.is_relative_to(directory.resolve()) or not source.is_file():
        raise FileNotFoundError(f"Missing object asset: {source}")
    return source


def copy_obj_dependencies(source: Path, target: Path, directory: Path, metadata: dict) -> None:
    """Keep OBJ/MTL/texture references local, including older absolute map_Kd paths."""
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines:
        if not line.startswith("mtllib "):
            continue
        mtl_ref = line.removeprefix("mtllib ").strip()
        mtl = local_file(directory, mtl_ref)
        mtl_target = target.parent / mtl_ref
        mtl_target.parent.mkdir(parents=True, exist_ok=True)
        mtl_lines = []
        for mtl_line in mtl.read_text(encoding="utf-8", errors="replace").splitlines():
            if mtl_line.startswith("map_Kd "):
                raw = mtl_line.removeprefix("map_Kd ").strip()
                texture_name = str(metadata.get("texture") or Path(raw).name)
                texture = local_file(directory, texture_name)
                texture_target = mtl_target.parent / texture.name
                shutil.copy2(texture, texture_target)
                mtl_line = f"map_Kd {texture.name}"
            mtl_lines.append(mtl_line)
        mtl_target.write_text("\n".join(mtl_lines) + "\n", encoding="utf-8")


def install_object(subset: str, key: str, directory: Path, output: Path,
                   cache: dict[Path, tuple[Path, float]]) -> tuple[Path, float]:
    xml_path = directory / "grail_object.xml"
    if xml_path in cache:
        return cache[xml_path]
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    root = ET.parse(xml_path).getroot()
    if root.find("include") is not None:
        raise ValueError(f"Unsupported object include: {xml_path}")
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    if Path(meshdir).is_absolute() or ".." in Path(meshdir).parts:
        raise ValueError(f"External object meshdir: {xml_path}")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    mesh_dir = output / "assets/objects/meshes" / subset / safe_name(key)
    scales = []
    for mesh in root.findall("./asset/mesh"):
        source = local_file(directory, str(Path(meshdir) / mesh.get("file", "")))
        scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ", dtype=np.float64)
        if scale.shape != (3,) or not np.isfinite(scale).all() or scale[0] <= 0 or not np.allclose(scale, scale[0]):
            raise ValueError(f"Nonuniform or invalid GRAIL mesh scale: {xml_path}")
        scales.append(float(scale[0]))
        target = mesh_dir / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if source.suffix.lower() == ".obj":
            copy_obj_dependencies(source, target, directory, metadata)
        mesh.set("file", f"../meshes/{subset}/{safe_name(key)}/{source.name}")
        mesh.set("scale", "1 1 1")
    if not scales or not np.allclose(scales, scales[0]):
        raise ValueError(f"GRAIL object meshes must share one scale: {xml_path}")
    visual = root.findall(".//worldbody/body/geom[@group='2']")
    collision = root.findall(".//worldbody/body/geom[@group='3']")
    if not visual or len(collision) != int(metadata["collision_parts"]):
        raise ValueError(f"GRAIL visual/collision geoms missing: {xml_path}")
    payload = ET.tostring(root, encoding="utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:24]
    template = output / "assets/objects/templates" / f"grail_{digest}.xml"
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_bytes(payload)
    cache[xml_path] = (template, scales[0])
    return cache[xml_path]


def object_pose(csv_path: Path, frame_ids: np.ndarray, data) -> tuple[np.ndarray, np.ndarray]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"px", "py", "pz", "qx", "qy", "qz", "qw"}.issubset(rows[0]):
        raise ValueError(f"Invalid GRAIL object trajectory: {csv_path}")
    if frame_ids.size == 0 or frame_ids.min() < 0 or frame_ids.max() >= len(rows):
        raise ValueError(f"GRAIL object trajectory does not cover frame_ids: {csv_path}")
    if str(scalar(data, "noitom_output_up", "z")).lower() != "z" or bool(scalar(data, "noitom_convert_y_up", False)):
        raise ValueError(f"Expected GRAIL Z-up pose metadata: {csv_path}")
    selected = [rows[int(index)] for index in frame_ids]
    pos = np.asarray([[float(row[k]) for k in ("px", "py", "pz")] for row in selected], dtype=np.float64)
    xyzw = np.asarray([[float(row[k]) for k in ("qx", "qy", "qz", "qw")] for row in selected], dtype=np.float64)
    if not np.isfinite(pos).all() or not np.isfinite(xyzw).all():
        raise ValueError(f"Nonfinite GRAIL object trajectory: {csv_path}")
    if bool(scalar(data, "noitom_ground_align", False)):
        pos[:, 2] -= float(scalar(data, "noitom_floor_y", 0))
    pos[:, 2] += float(scalar(data, "noitom_ground_offset", 0))
    pos *= float(scalar(data, "smpl_scale", 1))
    rot = Rotation.from_quat(xyzw).as_quat()[:, [3, 0, 1, 2]]
    return pos.astype(np.float32), rot.astype(np.float32)


def convert_clip(source: Path, args: argparse.Namespace, robot: Path, scene: Path,
                 cache: dict[Path, tuple[Path, float]]) -> str:
    clip = args.output_dir / "clips" / source.relative_to(args.input_dir)
    # Existing UMR results store robot_joint_names as an object string array.
    with np.load(source, allow_pickle=True) as data:
        if str(scalar(data, "source_format", "")) != "grail_hsi_hoi_smplx" or str(scalar(data, "robot_name", "")) != "igris_c":
            raise ValueError(f"Not an IGRIS-C GRAIL retarget result: {source}")
        qpos = np.asarray(data["qpos"])
        frames = np.asarray(data["frame_ids"], dtype=np.int32)
        fps = float(scalar(data, "fps"))
        scale = float(scalar(data, "smpl_scale", 1))
        if (qpos.ndim != 2 or qpos.shape[1] != 38 or not len(qpos) or frames.shape != (len(qpos),)
                or not np.isfinite(qpos).all() or not np.isfinite(fps) or fps <= 0 or not np.isfinite(scale) or scale <= 0):
            raise ValueError(f"Invalid GRAIL retarget motion: {source}")
        subset, key, directory = source_assets(data, args.sample_root)
        template, mesh_scale = install_object(subset, key, directory, args.output_dir, cache)
        pos, rot = object_pose(directory / "prop_grail_object.csv", frames, data)
        is_terrain = key.startswith("terrain_")
        aligned_rot = rot.copy()
        aligned_rot[np.sum(aligned_rot * rot[0], axis=1) < 0] *= -1
        is_static = bool(np.allclose(pos, pos[0], atol=1e-5)
                         and np.allclose(aligned_rot, aligned_rot[0], atol=1e-5))
        if is_terrain and not is_static:
            raise ValueError(f"GRAIL terrain has a moving pose: {source}")
        source_label = key.split("__", 2)[1] if "__" in key else key
        label_parts = source_label.rsplit("_", 1)
        label = subset if is_terrain else (label_parts[0] if len(label_parts) == 2 and label_parts[1].isdigit() else source_label)
        omit = {"robot_xml", "source_data", "source_object_dir"}
        payload = {field: data[field] for field in data.files if field not in omit}
        payload["robot_joint_names"] = np.asarray(data["robot_joint_names"], dtype=str)
        payload.update({
            "light_form_version": np.asarray(1, dtype=np.int32),
            "robot_xml": np.asarray(relative_to_clip(clip, robot)),
            "light_scene_ref": np.asarray(relative_to_clip(clip, scene)),
            "light_object_names": np.asarray([key], dtype=str),
            "light_object_xml_refs": np.asarray([relative_to_clip(clip, template)], dtype=str),
            "light_object_scales": np.asarray([mesh_scale * scale], dtype=np.float32),
            "light_object_pos": pos[:, None, :],
            "light_object_rot": rot[:, None, :],
            "light_object_entity_types": np.asarray(["terrain" if is_terrain else "object"], dtype=str),
            "light_object_semantic_labels": np.asarray([label], dtype=str),
            "light_object_mobilities": np.asarray(["static" if is_static else "dynamic"], dtype=str),
        })
    clip.parent.mkdir(parents=True, exist_ok=True)
    temporary = clip.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(clip)
    return "terrain" if is_terrain else "object"


def main() -> None:
    args = parse_args()
    sources = sorted(args.input_dir.rglob("*.npz"))
    if args.limit:
        sources = sources[:args.limit]
    if not sources:
        raise FileNotFoundError(f"No GRAIL UMR NPZ clips in {args.input_dir}")
    expected = {args.output_dir / "clips" / source.relative_to(args.input_dir) for source in sources}
    stale = set((args.output_dir / "clips").rglob("*.npz")) - expected
    if stale:
        raise ValueError(f"Output contains clips outside this conversion: {sorted(stale)[0]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    robot = install_robot(args.robot_xml, args.output_dir)
    scene = install_scene(args.robot_xml, args.output_dir)
    cache: dict[Path, tuple[Path, float]] = {}
    counts = {"object": 0, "terrain": 0}
    for index, source in enumerate(sources, 1):
        counts[convert_clip(source, args, robot, scene, cache)] += 1
        if index % 100 == 0 or index == len(sources):
            print(f"[{index}/{len(sources)}] GRAIL clips, {counts['terrain']} terrain, {counts['object']} objects", flush=True)
    manifest = {
        "format": "umr_light", "version": 1, "source": "grail",
        "clip_count": len(sources), "object_instances": len(sources),
        "terrain_instances": counts["terrain"], "object_templates": len(cache),
        "clip_paths_are_relative_to_npz": True,
        "units": "meters", "up_axis": "Z", "quaternion_order": "wxyz",
    }
    (args.output_dir / "format.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Light form ready: {args.output_dir}")


if __name__ == "__main__":
    main()
