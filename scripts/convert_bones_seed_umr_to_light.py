#!/usr/bin/env python3
"""Package IGRIS-C Bones Seed UMR results as portable light-form clips.

The retarget run used one IGRIS-C robot model. Many clip-specific generated XML
paths in the UMR NPZs point to another machine or no longer exist, so the
configured robot XML is bundled once after checking available generated XMLs
against it. Bones Seed clips here have no separate moving scene objects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from convert_amass_umr_to_light import install_model, relative_to_clip, scalar


ROOT = Path(__file__).resolve().parents[1]
MODEL_FIELDS = (
    "body_pos", "body_quat", "body_mass", "body_inertia",
    "geom_pos", "geom_quat", "geom_size", "geom_type", "geom_group",
    "mesh_vert", "mesh_face", "jnt_qposadr", "jnt_axis", "jnt_range",
    "actuator_gainprm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "output/igris_c_bones_seed/igris_c")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "light_form/igris_c_bones_seed")
    parser.add_argument("--robot-xml", type=Path, default=ROOT / "assets/igris_c/igris_c.xml")
    parser.add_argument("--limit", type=int, default=0, help="First N clips; 0 converts all")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.robot_xml = args.robot_xml.expanduser().resolve()
    if args.output_dir == args.input_dir or args.output_dir.is_relative_to(args.input_dir):
        parser.error("--output-dir must be outside --input-dir")
    if not args.robot_xml.is_file():
        parser.error(f"Robot XML missing: {args.robot_xml}")
    return args


def validate_available_xmls(sources: list[Path], robot_xml: Path) -> int:
    """Check that the surviving per-clip XMLs describe the configured robot."""
    xml_dir = robot_xml.parent
    hashes: dict[str, Path] = {}
    count = 0
    for source in sources:
        candidate = xml_dir / f"{source.stem}.floating_mjcf.xml"
        if not candidate.is_file():
            continue
        count += 1
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        hashes.setdefault(digest, candidate)
        if len(hashes) > 1:
            raise ValueError(f"Bones Seed generated robot XMLs differ: {list(hashes.values())}")
    base = mujoco.MjModel.from_xml_path(str(robot_xml))
    if hashes:
        generated = mujoco.MjModel.from_xml_path(str(next(iter(hashes.values()))))
        for name in ("nq", "nv", "nbody", "ngeom", "nmesh", "njnt", "nu"):
            if getattr(base, name) != getattr(generated, name):
                raise ValueError(f"Configured robot differs from generated XML: {name}")
        for name in MODEL_FIELDS:
            if not np.array_equal(getattr(base, name), getattr(generated, name)):
                raise ValueError(f"Configured robot differs from generated XML: {name}")
    if base.nq != 38:
        raise ValueError(f"Expected IGRIS-C nq=38, got {base.nq}")
    if count:
        print(f"Verified robot model against {count} available generated XMLs", flush=True)
    else:
        print("No generated XMLs found; using the configured robot XML", flush=True)
    return count


def convert_clip(source: Path, input_dir: Path, output: Path, robot: Path, scene: Path) -> None:
    clip = output / "clips" / source.relative_to(input_dir)
    with np.load(source, allow_pickle=True) as data:
        source_format = str(scalar(data, "source_format"))
        if str(scalar(data, "robot_name")) != "igris_c" or not source_format.startswith("soma_bvh_boneseed_"):
            raise ValueError(f"Not an IGRIS-C Bones Seed UMR result: {source}")
        qpos = np.asarray(data["qpos"])
        frame_ids = np.asarray(data["frame_ids"])
        fps = float(scalar(data, "fps"))
        if (qpos.ndim != 2 or qpos.shape[1] != 38 or len(qpos) == 0
                or frame_ids.shape != (len(qpos),) or not np.isfinite(qpos).all()
                or not np.isfinite(fps) or fps <= 0):
            raise ValueError(f"Invalid Bones Seed motion: {source}")
        omit = {"robot_xml", "source_data", "source_object_dir"}
        payload = {key: data[key] for key in data.files if key not in omit}
        payload["robot_joint_names"] = np.asarray(data["robot_joint_names"], dtype=str)
        payload.update({
            "light_form_version": np.asarray(1, dtype=np.int32),
            "robot_xml": np.asarray(relative_to_clip(clip, robot)),
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
    available_xmls = validate_available_xmls(sources, args.robot_xml)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh_cache: dict[Path, Path] = {}
    robot, scene = install_model(args.robot_xml, args.output_dir, mesh_cache)
    for index, source in enumerate(sources, 1):
        convert_clip(source, args.input_dir, args.output_dir, robot, scene)
        if index % 5000 == 0 or index == len(sources):
            print(f"[{index}/{len(sources)}] clips, 1 robot model, {len(mesh_cache)} shared meshes", flush=True)
    manifest = {
        "format": "umr_light", "version": 1, "source": "bones_seed",
        "clip_count": len(sources), "robot_models": 1,
        "shared_robot_meshes": len(mesh_cache),
        "verified_generated_xmls": available_xmls,
        "clip_paths_are_relative_to_npz": True,
        "units": "meters", "up_axis": "Z", "quaternion_order": "wxyz",
    }
    (args.output_dir / "format.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Light form ready: {args.output_dir}")


if __name__ == "__main__":
    main()
