#!/usr/bin/env python3
"""Package GMR PICO IGRIS-C PKLs as portable light-form v1 clips.

PICO's retargeted PKLs contain robot motion but no moving object tracks. The
GMR robot XML and its meshes are bundled once, along with its static floor.
Take names are included in clip filenames because canonical output is flat.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import mujoco
import numpy as np

from convert_amass_umr_to_light import install_model


ROOT = Path(__file__).resolve().parents[1]
GMR_ROOT = ROOT.parent / "GMR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=GMR_ROOT / "pico/retargeted")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "light_form/igris_c_pico")
    parser.add_argument("--robot-xml", type=Path,
                        default=GMR_ROOT / "assets/robros_igris_c_v2/igris_c_v2.xml")
    parser.add_argument("--limit", type=int, default=0, help="First N sorted PKLs; 0 converts all")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.robot_xml = args.robot_xml.expanduser().resolve()
    if not args.input_dir.is_dir():
        parser.error(f"Input directory missing: {args.input_dir}")
    if not args.robot_xml.is_file():
        parser.error(f"Robot XML missing: {args.robot_xml}")
    if args.output_dir == args.input_dir or args.output_dir.is_relative_to(args.input_dir):
        parser.error("--output-dir must be outside --input-dir")
    return args


def clip_name(source: Path, input_dir: Path) -> str:
    """Encode the relative source path into a unique flat canonical clip ID."""
    relative = source.relative_to(input_dir).with_suffix("")
    return "__".join(relative.parts)


def convert_clip(source: Path, input_dir: Path, output_dir: Path, robot: Path,
                 scene: Path, joint_names: list[str]) -> None:
    with source.open("rb") as handle:
        motion = pickle.load(handle)
    if not isinstance(motion, dict):
        raise ValueError(f"PICO motion is not a dictionary: {source}")
    metadata = [motion[key] for key in ("record_meta", "retarget_meta") if key in motion]
    if (not metadata or any(not isinstance(meta, dict) for meta in metadata)
            or not any(meta.get("robot") == "robros_igris_c_v2" for meta in metadata)
            or any(meta.get("robot") not in (None, "robros_igris_c_v2") for meta in metadata)):
        raise ValueError(f"PICO motion does not identify the expected GMR robot: {source}")

    root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float32)
    fps = float(np.asarray(motion["fps"]).item())
    frames = len(root_pos)
    if (root_pos.shape != (frames, 3) or root_rot.shape != (frames, 4)
            or dof_pos.shape != (frames, len(joint_names)) or frames == 0
            or not np.isfinite(root_pos).all() or not np.isfinite(root_rot).all()
            or not np.isfinite(dof_pos).all() or not np.isfinite(fps) or fps <= 0):
        raise ValueError(f"Invalid PICO pose arrays or FPS: {source}")
    norms = np.linalg.norm(root_rot.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0, atol=1e-3):
        raise ValueError(f"PICO root quaternions are not normalized: {source}")
    for meta in metadata:
        for key in ("num_frames", "num_output_frames"):
            if key in meta and int(meta[key]) != frames:
                raise ValueError(f"PICO metadata {key} differs from pose arrays: {source}")
        for key in ("saved_fps", "fps"):
            if key in meta and not np.isclose(float(meta[key]), fps):
                raise ValueError(f"PICO metadata {key} differs from motion FPS: {source}")

    clip = output_dir / "clips" / f"{clip_name(source, input_dir)}.npz"
    qpos = np.concatenate((root_pos, root_rot, dof_pos), axis=1)
    payload = {
        "light_form_version": np.asarray(1, dtype=np.int32),
        "source_format": np.asarray("gmr_pico_pkl"),
        "source_clip": np.asarray(source.relative_to(input_dir).as_posix()),
        "robot_name": np.asarray("igris_c"),
        "qpos": qpos,
        "fps": np.asarray(fps, dtype=np.float32),
        "frame_ids": np.arange(frames, dtype=np.int32),
        "robot_joint_names": np.asarray(joint_names, dtype=str),
        "robot_xml": np.asarray(Path(os.path.relpath(robot, clip.parent)).as_posix()),
        "light_scene_ref": np.asarray(Path(os.path.relpath(scene, clip.parent)).as_posix()),
    }
    clip.parent.mkdir(parents=True, exist_ok=True)
    temporary = clip.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(clip)


def main() -> None:
    args = parse_args()
    sources = sorted(args.input_dir.rglob("*.pkl"))
    if args.limit:
        sources = sources[:args.limit]
    if not sources:
        raise FileNotFoundError(f"No PICO PKLs in {args.input_dir}")
    names = [clip_name(source, args.input_dir) for source in sources]
    if len(set(names)) != len(names):
        raise ValueError("PICO source paths map to duplicate flat clip names")
    expected = {f"{name}.npz" for name in names}
    stale = {path.name for path in (args.output_dir / "clips").rglob("*.npz")} - expected
    if stale:
        raise ValueError(f"Output contains {len(stale)} clips outside this run; use a clean output directory")

    model = mujoco.MjModel.from_xml_path(str(args.robot_xml))
    if model.nq != 38 or model.njnt != 32:
        raise ValueError(f"Expected IGRIS-C with 38 qpos and 32 joints: {args.robot_xml}")
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                   for i in range(1, model.njnt)]
    if len(set(joint_names)) != 31 or any(name is None for name in joint_names):
        raise ValueError(f"Unexpected joint names in PICO robot model: {args.robot_xml}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh_cache: dict[Path, Path] = {}
    robot, scene = install_model(args.robot_xml, args.output_dir, mesh_cache)
    for index, source in enumerate(sources, 1):
        convert_clip(source, args.input_dir, args.output_dir, robot, scene, joint_names)
        if index % 50 == 0 or index == len(sources):
            print(f"[{index}/{len(sources)}] PICO clips, {len(mesh_cache)} shared robot meshes", flush=True)
    manifest = {
        "format": "umr_light", "version": 1, "source": "pico",
        "clip_count": len(sources), "robot_models": 1,
        "shared_robot_meshes": len(mesh_cache),
        "clip_paths_are_relative_to_npz": True,
        "units": "meters", "up_axis": "Z", "quaternion_order": "wxyz",
    }
    (args.output_dir / "format.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Light form ready: {args.output_dir}")


if __name__ == "__main__":
    main()
