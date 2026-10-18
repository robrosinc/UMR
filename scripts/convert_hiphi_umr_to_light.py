#!/usr/bin/env python3
"""Package HiPHI UMR motions, object tracks, and shared meshes as light form v1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from convert_omomo_umr_to_light import install_robot, install_scene, relative_to_clip, scalar
from hiphi_layout import resolve_hiphi_objects


ROOT = Path(__file__).resolve().parents[1]
Y_TO_Z_QUAT = "0.707106781 0.707106781 0 0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "output/igris_c_hiphi_contact_only/igris_c_dummy_hand")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "light_form/igris_c_hiphi")
    parser.add_argument("--sample-root", type=Path, default=ROOT / "sample_data/hiphi")
    parser.add_argument("--robot-xml", type=Path, default=ROOT / "assets/igris_c/igris_c_dummy_hand.xml")
    parser.add_argument("--limit", type=int, default=0, help="First N clips; 0 converts all")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    for key in ("input_dir", "output_dir", "sample_root", "robot_xml"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.output_dir == args.input_dir or args.output_dir.is_relative_to(args.input_dir):
        parser.error("--output-dir must be separate from --input-dir")
    return args


def source_directory(data, sample_root: Path) -> Path:
    saved = str(scalar(data, "source_object_dir", scalar(data, "source_data", "")))
    marker = "/sample_data/hiphi/"
    if marker not in saved:
        raise ValueError(f"Expected HiPHI source path, got {saved!r}")
    suffix = Path(saved.split(marker, 1)[1])
    if suffix.is_absolute() or ".." in suffix.parts:
        raise ValueError(f"Unsafe HiPHI source path: {saved}")
    path = (sample_root / suffix).resolve()
    if not path.is_relative_to(sample_root) or not path.is_dir():
        raise FileNotFoundError(path)
    return path


def fallback_xml(mesh_path: Path) -> ET.Element:
    """Use the original visual OBJ as a convex-hull collision source when no XML exists."""
    name = "hiphi_object"
    root = ET.Element("mujoco", {"model": name})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", {"name": "visual", "file": mesh_path.name, "scale": "0.01 0.01 0.01"})
    ET.SubElement(asset, "mesh", {"name": "collision", "file": mesh_path.name, "scale": "0.01 0.01 0.01"})
    body = ET.SubElement(ET.SubElement(root, "worldbody"), "body", {"name": name})
    ET.SubElement(body, "freejoint", {"name": "hiphi_object_freejoint"})
    ET.SubElement(body, "geom", {"name": "visual", "type": "mesh", "mesh": "visual", "group": "2",
                                      "contype": "0", "conaffinity": "0", "quat": Y_TO_Z_QUAT})
    ET.SubElement(body, "geom", {"name": "collision", "type": "mesh", "mesh": "collision", "group": "3",
                                      "contype": "1", "conaffinity": "1", "quat": Y_TO_Z_QUAT})
    return root


def install_object(entry: dict[str, str], sample_root: Path, output: Path,
                   mesh_cache: dict[Path, Path], template_cache: dict[Path, tuple[Path, float, bool]]) -> tuple[Path, float, bool]:
    visual = Path(entry["obj"]).resolve()
    xml_path = Path(entry["xml"]).resolve()
    if xml_path in template_cache:
        return template_cache[xml_path]
    mesh_root = (sample_root / "object_meshes").resolve()
    if not visual.is_relative_to(mesh_root) or not visual.is_file() or not xml_path.is_relative_to(mesh_root):
        raise FileNotFoundError(f"Missing or external HiPHI mesh: {visual}")
    fallback = not xml_path.is_file()
    root = fallback_xml(visual) if fallback else ET.parse(xml_path).getroot()
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    if Path(meshdir).is_absolute() or ".." in Path(meshdir).parts:
        raise ValueError(f"Unsafe meshdir in {xml_path}: {meshdir}")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    scales = []
    for mesh in root.findall("./asset/mesh"):
        ref = Path(mesh.get("file", ""))
        if not ref.parts or ref.is_absolute() or ".." in ref.parts:
            raise ValueError(f"Unsafe mesh reference in {xml_path}: {ref}")
        source = (xml_path.parent / meshdir / ref).resolve()
        if not source.is_relative_to(mesh_root) or not source.is_file():
            raise FileNotFoundError(f"Missing or external HiPHI mesh in {xml_path}: {source}")
        scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ", dtype=np.float64)
        if scale.shape != (3,) or not np.allclose(scale, scale[0], rtol=0, atol=1e-9) or scale[0] <= 0:
            raise ValueError(f"Nonuniform or invalid object scale in {xml_path}: {scale}")
        scales.append(float(scale[0]))
        if source not in mesh_cache:
            digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
            target = output / "assets/objects/meshes" / f"{digest}_{source.name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file():
                shutil.copy2(source, target)
            mesh_cache[source] = target
        mesh.set("file", f"../meshes/{mesh_cache[source].name}")
        mesh.set("scale", "1 1 1")
    if not scales or not np.allclose(scales, scales[0], rtol=0, atol=1e-9):
        raise ValueError(f"HiPHI object meshes must share one scale: {xml_path}")
    for element in root.iter():
        if element.tag == "include" or (element.tag != "mesh" and element.get("file")):
            raise ValueError(f"Unhandled external asset in {xml_path}: {element.tag}")
    payload = ET.tostring(root, encoding="utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:24]
    target = output / "assets/objects/templates" / f"object_{digest}.xml"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        target.write_bytes(payload)
    result = (target, scales[0], fallback)
    template_cache[xml_path] = result
    return result


def object_pose(csv_path: Path, frame_ids: np.ndarray, data) -> tuple[np.ndarray, np.ndarray, bool]:
    with csv_path.open(newline="", errors="replace") as handle:
        rows = list(csv.DictReader(handle))
    required = {"px", "py", "pz", "qx", "qy", "qz", "qw"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid HiPHI object trajectory: {csv_path}")
    if frame_ids.size == 0 or frame_ids.min() < 0:
        raise ValueError(f"Invalid frame IDs for {csv_path}")
    clamped = bool(frame_ids.max() >= len(rows))
    selected = [rows[int(i)] for i in np.clip(frame_ids, 0, len(rows) - 1)]
    pos = np.asarray([[float(row[k]) for k in ("px", "py", "pz")] for row in selected], dtype=np.float32)
    xyzw = np.asarray([[float(row[k]) for k in ("qx", "qy", "qz", "qw")] for row in selected], dtype=np.float64)
    output_up = str(scalar(data, "noitom_output_up", "y")).lower()
    convert_y = bool(scalar(data, "noitom_convert_y_up", True))
    ground_align = bool(scalar(data, "noitom_ground_align", True))
    if output_up not in ("y", "z"):
        raise ValueError(f"Unsupported HiPHI output up axis: {output_up}")
    basis = np.eye(3) if not (convert_y and output_up == "y") else np.asarray(
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    pos = pos @ basis.T
    if ground_align:
        pos[:, 2] -= float(scalar(data, "noitom_floor_y", 0))
    pos[:, 2] += float(scalar(data, "noitom_ground_offset", 0))
    pos *= float(scalar(data, "smpl_scale", 1))
    rot = Rotation.from_matrix(basis @ Rotation.from_quat(xyzw).as_matrix() @ basis.T).as_quat()
    wxyz = rot[:, [3, 0, 1, 2]].astype(np.float32)
    wxyz /= np.maximum(np.linalg.norm(wxyz, axis=1, keepdims=True), 1e-12)
    return pos.astype(np.float32), wxyz, clamped


def convert_clip(source: Path, args, robot: Path, scene: Path,
                 mesh_cache: dict[Path, Path], template_cache: dict[Path, tuple[Path, float, bool]]) -> tuple[int, int, int]:
    clip = args.output_dir / "clips" / source.relative_to(args.input_dir)
    with np.load(source, allow_pickle=True) as data:
        if str(scalar(data, "source_format")) != "hiphi_smplx" or str(scalar(data, "robot_name")) != "igris_c_dummy_hand":
            raise ValueError(f"Not an IGRIS-C HiPHI result: {source}")
        qpos = np.asarray(data["qpos"])
        frame_ids = np.asarray(data["frame_ids"], dtype=np.int32)
        fps = float(scalar(data, "fps"))
        if (qpos.ndim != 2 or qpos.shape[1] != 38 or not len(qpos)
                or frame_ids.shape != (len(qpos),) or not np.isfinite(qpos).all()
                or not np.isfinite(fps) or fps <= 0):
            raise ValueError(f"Invalid HiPHI motion: {source}")
        entries = resolve_hiphi_objects(source_directory(data, args.sample_root))
        names, xml_refs, scales, positions, rotations = [], [], [], [], []
        fallbacks = clamped_tracks = 0
        for entry in entries:
            template, mesh_scale, fallback = install_object(entry, args.sample_root, args.output_dir,
                                                             mesh_cache, template_cache)
            pos, rot, clamped = object_pose(Path(entry["prop"]), frame_ids, data)
            names.append(entry["name"])
            xml_refs.append(relative_to_clip(clip, template))
            scales.append(mesh_scale * float(scalar(data, "smpl_scale", 1)))
            positions.append(pos)
            rotations.append(rot)
            fallbacks += int(fallback)
            clamped_tracks += int(clamped)
        omit = {"robot_xml", "source_data", "source_object_dir"}
        payload = {key: data[key] for key in data.files if key not in omit}
        payload["robot_joint_names"] = np.asarray(data["robot_joint_names"], dtype=str)
        payload.update({
            "light_form_version": np.asarray(1, dtype=np.int32),
            "robot_xml": np.asarray(relative_to_clip(clip, robot)),
            "light_scene_ref": np.asarray(relative_to_clip(clip, scene)),
            "light_object_names": np.asarray(names, dtype=str),
            "light_object_xml_refs": np.asarray(xml_refs, dtype=str),
            "light_object_scales": np.asarray(scales, dtype=np.float32),
            "light_object_pos": np.stack(positions, axis=1) if positions else np.empty((len(qpos), 0, 3), dtype=np.float32),
            "light_object_rot": np.stack(rotations, axis=1) if rotations else np.empty((len(qpos), 0, 4), dtype=np.float32),
        })
    clip.parent.mkdir(parents=True, exist_ok=True)
    temporary = clip.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(clip)
    return len(entries), fallbacks, clamped_tracks


def main() -> None:
    args = parse_args()
    sources = sorted(args.input_dir.rglob("*.npz"))
    if args.limit:
        sources = sources[:args.limit]
    if not sources:
        raise FileNotFoundError(f"No UMR NPZ clips in {args.input_dir}")
    expected = {args.output_dir / "clips" / source.relative_to(args.input_dir) for source in sources}
    stale = set((args.output_dir / "clips").rglob("*.npz")) - expected
    if stale:
        raise ValueError(f"Output has {len(stale)} clips outside this conversion: {sorted(stale)[0]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    robot = install_robot(args.robot_xml, args.output_dir)
    scene = install_scene(args.robot_xml, args.output_dir)
    mesh_cache: dict[Path, Path] = {}
    template_cache: dict[Path, tuple[Path, float, bool]] = {}
    objects = fallbacks = clamped = 0
    for index, source in enumerate(sources, 1):
        n, f, c = convert_clip(source, args, robot, scene, mesh_cache, template_cache)
        objects += n
        fallbacks += f
        clamped += c
        if index % 1000 == 0 or index == len(sources):
            print(f"[{index}/{len(sources)}] clips, {objects} objects, {len(template_cache)} templates, "
                  f"{len(mesh_cache)} meshes, {fallbacks} fallback collision instances", flush=True)
    manifest = {
        "format": "umr_light", "version": 1, "source": "hiphi",
        "clip_count": len(sources), "object_instances": objects,
        "object_templates": len(template_cache), "shared_object_meshes": len(mesh_cache),
        "fallback_collision_instances": fallbacks, "clamped_object_tracks": clamped,
        "fallback_collision": "original visual OBJ used as convex hull source when prepared MJCF is absent",
        "fallback_freejoints_named": True,
        "clip_paths_are_relative_to_npz": True,
        "units": "meters", "up_axis": "Z", "quaternion_order": "wxyz",
    }
    (args.output_dir / "format.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Light form ready: {args.output_dir}")


if __name__ == "__main__":
    main()
