#!/usr/bin/env python3
"""Package IGRIS-C AMASS UMR clips as portable light-form motion clips.

AMASS has no separate scene objects. Generated per-clip robot XML files are
deduplicated, their mesh paths are made package-relative, and the static floor
is recorded in a shared light scene. Source AMASS files are not needed later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "output/igris_c_amass/igris_c")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "light_form/igris_c_amass")
    parser.add_argument("--limit", type=int, default=0, help="First N clips; 0 converts all")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir == args.input_dir or args.output_dir.is_relative_to(args.input_dir):
        parser.error("--output-dir must be outside --input-dir")
    return args


def scalar(data, key: str):
    return np.asarray(data[key]).reshape(-1)[0].item()


def relative_to_clip(clip: Path, target: Path) -> str:
    return Path(os.path.relpath(target, clip.parent)).as_posix()


def source_robot_xml(value: str, clip: Path) -> Path:
    saved = Path(value).expanduser()
    candidates = [saved] if saved.is_absolute() else [clip.parent / saved, ROOT / saved]
    # Retarget NPZs may have been moved with this checkout. Their generated
    # XML names are stable inside assets/igris_c/.
    candidates.append(ROOT / "assets/igris_c" / saved.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Robot XML for {clip} is missing: {value}")


def install_model(source_xml: Path, output: Path, mesh_cache: dict[Path, Path]) -> tuple[Path, Path]:
    root = ET.parse(source_xml).getroot()
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    if Path(meshdir).is_absolute():
        raise ValueError(f"Absolute compiler meshdir in {source_xml}: {meshdir}")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    for mesh in root.findall("./asset/mesh"):
        name = mesh.get("file")
        if not name:
            raise ValueError(f"Mesh asset has no file in {source_xml}")
        ref = Path(name)
        source = (ref if ref.is_absolute() else source_xml.parent / meshdir / ref).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Robot mesh for {source_xml} is missing: {source}")
        if source not in mesh_cache:
            digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
            target = output / "assets/robot/meshes" / f"{digest}_{source.name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file():
                shutil.copy2(source, target)
            mesh_cache[source] = target
        mesh.set("file", f"../meshes/{mesh_cache[source].name}")
    for element in root.iter():
        if element.get("file") and element.tag != "mesh":
            raise ValueError(f"Unhandled file reference in {source_xml}: {element.tag}")
        if element.tag == "include":
            raise ValueError(f"Robot XML include must be bundled explicitly: {source_xml}")
    payload = ET.tostring(root, encoding="utf-8")
    model_hash = hashlib.sha256(payload).hexdigest()[:24]
    model = output / "assets/robot/models" / f"model_{model_hash}.xml"
    model.parent.mkdir(parents=True, exist_ok=True)
    if model.is_file() and model.read_bytes() != payload:
        raise ValueError(f"Robot model hash collision: {model}")
    if not model.is_file():
        model.write_bytes(payload)

    entities = []
    for index, geom in enumerate(root.findall("./worldbody/geom")):
        if geom.get("type") != "plane":
            continue
        entities.append({
            "entity_id": geom.get("name", f"plane_{index}"),
            "entity_type": "terrain", "primitive": "plane",
            "size": [float(value) for value in geom.get("size", "1 1 0.05").split()],
            "pos": [float(value) for value in geom.get("pos", "0 0 0").split()],
            "rot_wxyz": [float(value) for value in geom.get("quat", "1 0 0 0").split()],
        })
    scene_data = {"version": 1, "static_entities": entities}
    scene_payload = json.dumps(scene_data, indent=2, ensure_ascii=False) + "\n"
    scene_hash = hashlib.sha256(scene_payload.encode()).hexdigest()[:24]
    scene = output / "scenes" / f"scene_{scene_hash}.json"
    scene.parent.mkdir(parents=True, exist_ok=True)
    if scene.is_file() and scene.read_text() != scene_payload:
        raise ValueError(f"Scene hash collision: {scene}")
    if not scene.is_file():
        scene.write_text(scene_payload)
    return model, scene


def convert_clip(source: Path, input_dir: Path, output: Path,
                 model_cache: dict[str, tuple[Path, Path]], mesh_cache: dict[Path, Path]) -> None:
    clip = output / "clips" / source.relative_to(input_dir)
    with np.load(source, allow_pickle=True) as data:
        if str(scalar(data, "robot_name")) != "igris_c" or str(scalar(data, "source_format")) != "smplx_npz":
            raise ValueError(f"Not an IGRIS-C SMPL-X retarget result: {source}")
        qpos = np.asarray(data["qpos"])
        frame_ids = np.asarray(data["frame_ids"])
        fps = float(scalar(data, "fps"))
        if (qpos.ndim != 2 or qpos.shape[1] != 38 or len(qpos) == 0
                or frame_ids.shape != (len(qpos),) or not np.isfinite(qpos).all()
                or not np.isfinite(fps) or fps <= 0):
            raise ValueError(f"Invalid IGRIS-C SMPL-X motion: {source}")
        xml_path = source_robot_xml(str(scalar(data, "robot_xml")), source)
        xml_bytes = xml_path.read_bytes()
        source_hash = hashlib.sha256(xml_bytes).hexdigest()
        if source_hash not in model_cache:
            model_cache[source_hash] = install_model(xml_path, output, mesh_cache)
        model, scene = model_cache[source_hash]
        omit = {"robot_xml", "source_data", "source_object_dir"}
        payload = {key: data[key] for key in data.files if key not in omit}
        payload["robot_joint_names"] = np.asarray(data["robot_joint_names"], dtype=str)
        payload.update({
            "light_form_version": np.asarray(1, dtype=np.int32),
            "robot_xml": np.asarray(relative_to_clip(clip, model)),
            "light_scene_ref": np.asarray(relative_to_clip(clip, scene)),
        })
    clip.parent.mkdir(parents=True, exist_ok=True)
    temporary = clip.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(clip)


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
        raise ValueError(
            f"Output contains {len(stale)} clips outside this conversion; "
            f"choose a clean output directory (example: {sorted(stale)[0]})")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_cache: dict[str, tuple[Path, Path]] = {}
    mesh_cache: dict[Path, Path] = {}
    for index, source in enumerate(sources, 1):
        convert_clip(source, args.input_dir, args.output_dir, model_cache, mesh_cache)
        if index % 500 == 0 or index == len(sources):
            print(f"[{index}/{len(sources)}] clips, {len(model_cache)} robot models, "
                  f"{len(mesh_cache)} shared meshes", flush=True)
    manifest = {
        "format": "umr_light", "version": 1, "source": "amass",
        "clip_count": len(sources), "robot_models": len(model_cache),
        "shared_robot_meshes": len(mesh_cache),
        "clip_paths_are_relative_to_npz": True,
        "units": "meters", "up_axis": "Z", "quaternion_order": "wxyz",
    }
    (args.output_dir / "format.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Light form ready: {args.output_dir}")


if __name__ == "__main__":
    main()
