#!/usr/bin/env python3
"""Convert OmniContact BVH captures into UMR's flat SMPL-X HSI/HOI layout.

The released BVH has the SMPL-X body/finger joint hierarchy. Rotations are
mapped directly; this does not reproduce the sample's additional upper-body IK.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "sample_data/omnicontact/soccer/case3_kick_right/20260330000586_1_1775269234"
VERSION = 1
BODY_JOINTS = [
    "Hips", "LeftUpLeg", "RightUpLeg", "Spine", "LeftLeg", "RightLeg",
    "Spine2", "LeftFoot", "RightFoot", "Spine4", "LeftToeBase",
    "RightToeBase", "Neck", "LeftShoulder", "RightShoulder", "Head",
    "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand",
    "RightHand",
]
FINGERS = ("Index", "Middle", "Pinky", "Ring", "Thumb")
JOINTS = BODY_JOINTS + [None, None, None] + [
    f"{side}Hand{finger}{part}"
    for side in ("Left", "Right")
    for finger in FINGERS
    for part in (1, 2, 3)
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/home/robros/workspace/motion_datas/OmniContact"))
    parser.add_argument("--output", type=Path, default=ROOT / "sample_data/omnicontact")
    parser.add_argument("--reference", type=Path, default=REFERENCE, help="Fitted sample supplying a neutral SMPL-X shape and pelvis offset")
    parser.add_argument("--seq-key", action="append", default=[], help="Capture ID, or category/case/capture path; repeatable")
    parser.add_argument("--limit", type=int, default=0, help="Maximum captures to process; 0 means all")
    parser.add_argument("--target-fps", type=int, default=0, help="Subsample 90 Hz source to this integer FPS; 0 preserves source FPS")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing flat motion files, including the fitted example")
    parser.add_argument("--plan", action="store_true", help="List selected captures without writing files")
    args = parser.parse_args()
    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.reference = args.reference.expanduser().resolve()
    if args.limit < 0 or args.target_fps < 0:
        parser.error("--limit and --target-fps must be nonnegative")
    if args.input == args.output or args.input in args.output.parents:
        parser.error("--output must be outside --input")
    return args


def parse_bvh(path: Path, frame_indices: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        names = []
        channels = []
        for line in handle:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] in {"ROOT", "JOINT"}:
                names.append(tokens[1])
            elif tokens[0] == "CHANNELS":
                channels.append((names[-1], tokens[2:]))
            elif tokens[:2] == ["Frame", "Time:"]:
                frame_time = float(tokens[2])
                break
        else:
            raise ValueError(f"No BVH frame time: {path}")
        values = np.fromstring(handle.read(), sep=" ", dtype=np.float32)
    width = sum(len(spec) for _, spec in channels)
    if width == 0 or values.size % width:
        raise ValueError(f"Invalid BVH channel count: {path}, values={values.size}, channels={width}")
    frames = values.reshape(-1, width)
    if frame_indices is not None:
        frames = frames[frame_indices]
    if not np.all(np.isfinite(frames)):
        raise ValueError(f"Nonfinite BVH values: {path}")
    poses = np.zeros((len(frames), 55, 3), dtype=np.float32)
    name_to_pose = {name: idx for idx, name in enumerate(JOINTS) if name}
    available = {name for name, _ in channels}
    # Ten early captures use a shorter Spine -> Spine1 -> Neck hierarchy.
    # Keep the top torso rotation at SMPL-X spine3 and leave spine2 neutral.
    if "Spine2" not in available and "Spine4" not in available and "Spine1" in available:
        del name_to_pose["Spine2"]
        del name_to_pose["Spine4"]
        name_to_pose["Spine1"] = 9
    found = set()
    root = None
    cursor = 0
    for name, spec in channels:
        data = frames[:, cursor:cursor + len(spec)]
        cursor += len(spec)
        if name == "Hips":
            translation_ids = [i for i, field in enumerate(spec) if field.endswith("position")]
            axes = "".join(spec[i][0] for i in translation_ids)
            if axes != "XYZ":
                raise ValueError(f"Expected XYZ root translation, got {axes}: {path}")
            root = data[:, translation_ids].astype(np.float32) * 0.01  # BVH centimeters -> meters
        if name not in name_to_pose:
            continue
        rotation_ids = [i for i, field in enumerate(spec) if field.endswith("rotation")]
        axes = "".join(spec[i][0] for i in rotation_ids)
        if axes != "ZXY":
            raise ValueError(f"Expected ZXY joint rotation, got {axes} for {name}: {path}")
        poses[:, name_to_pose[name]] = Rotation.from_euler(
            axes, data[:, rotation_ids], degrees=True
        ).as_rotvec().astype(np.float32)
        found.add(name)
    missing = set(name_to_pose) - found
    if root is None or missing:
        raise ValueError(f"Unsupported OmniContact BVH hierarchy in {path}; missing={sorted(missing)}")
    return poses.reshape(len(frames), 165), root, 1.0 / frame_time


def frame_count_and_fps(path: Path) -> tuple[int, float]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        count = None
        for line in handle:
            tokens = line.split()
            if tokens[:1] == ["Frames:"]:
                count = int(tokens[1])
            elif tokens[:2] == ["Frame", "Time:"]:
                if count is None:
                    raise ValueError(f"No frame count: {path}")
                return count, 1.0 / float(tokens[2])
    raise ValueError(f"No frame time: {path}")


def link_or_copy(source: Path, target: Path, overwrite: bool = False) -> None:
    if target.exists() and not overwrite:
        return
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def write_object_xml(path: Path, stem: str) -> None:
    root = ET.Element("mujoco", {"model": stem.lower()})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", {
        "name": "object_mesh", "file": f"{stem}.obj", "scale": "0.01 0.01 0.01",
    })
    body = ET.SubElement(ET.SubElement(root, "worldbody"), "body", {"name": stem.lower()})
    ET.SubElement(body, "geom", {
        "name": "object_geom", "type": "mesh", "mesh": "object_mesh",
        "quat": "0.707106781 0.707106781 0 0", "friction": "1 0.005 0.0001",
        "contype": "1", "conaffinity": "1",
    })
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_prop_csv(source: Path, target: Path, indices: np.ndarray, total_frames: int) -> None:
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if columns is None or not {"px", "py", "pz", "qx", "qy", "qz", "qw"}.issubset(columns):
            raise ValueError(f"Unsupported object CSV: {source}")
        rows = list(reader)
    if len(rows) != total_frames:
        raise ValueError(f"Object/BVH frame mismatch: {source}={len(rows)}, BVH={total_frames}")
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows[int(index)] for index in indices)


def process(meta_path: Path, args: argparse.Namespace, betas: np.ndarray, pelvis_offset: np.ndarray) -> str:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    source_dir = meta_path.parent
    relative = source_dir.relative_to(args.input / "raw_mocap")
    destination = args.output / relative
    bvh = source_dir / meta["human_motion_file"]
    total_frames, fps = frame_count_and_fps(bvh)
    if int(meta["frame_count"]) != total_frames:
        raise ValueError(f"Metadata/BVH frame mismatch: {bvh}")
    if args.target_fps:
        stride = round(fps / args.target_fps)
        if stride <= 0 or abs(fps / stride - args.target_fps) > 0.01:
            raise ValueError(f"Cannot sample {fps:.6f} FPS at {args.target_fps} FPS: {bvh}")
    else:
        stride = 1
    indices = np.arange(0, total_frames, stride, dtype=np.int32)
    objects = meta.get("objects", [])
    motion_files = [destination / name for name in ("poses.npy", "transl.npy", "betas.npy", "gender.npy", "model_type.npy", "mocap_framerate.npy", "output_up.npy")]
    existing = all(path.is_file() for path in motion_files)
    if existing and not args.overwrite:
        stored_fps = float(np.load(destination / "mocap_framerate.npy"))
        if abs(stored_fps - fps / stride) > 0.01:
            raise ValueError(
                f"Existing motion is {stored_fps:.3f} FPS, requested {fps / stride:.3f} FPS; "
                "use --overwrite or another output directory"
            )
    destination.mkdir(parents=True, exist_ok=True)
    if not existing or args.overwrite:
        poses, root_position, actual_fps = parse_bvh(bvh, indices)
        if len(poses) != len(indices) or abs(actual_fps - fps) > 1e-4:
            raise ValueError(f"BVH parse mismatch: {bvh}")
        # The sample's fitted SMPL-X origin is above its BVH Hips joint.
        transl = root_position + pelvis_offset[None, :]
        for name, value in {
            "poses": poses, "transl": transl, "betas": betas,
            "gender": np.asarray("neutral"), "model_type": np.asarray("smplx"),
            "mocap_framerate": np.asarray(fps / stride, dtype=np.float32),
            "output_up": np.asarray("y"),
        }.items():
            np.save(destination / f"{name}.npy", value)
        (destination / "conversion.json").write_text(json.dumps({
            "converter_version": VERSION, "source_bvh": str(bvh),
            "method": "direct_bvh_joint_rotations; sample_shape_and_pelvis_offset",
            "reference": str(args.reference), "source_fps": fps, "output_fps": fps / stride,
            "frames": len(indices), "pelvis_offset_m": pelvis_offset.tolist(),
            "note": "Additional sample upper-body IK fitting is not reproduced.",
        }, indent=2) + "\n", encoding="utf-8")
    for item in objects:
        stem = str(item["object_name"]).removeprefix("prop_")
        mesh_source = args.input / "assets" / "objects" / f"{stem}.obj"
        if not mesh_source.is_file():
            raise FileNotFoundError(f"Missing object mesh: {mesh_source}")
        link_or_copy(mesh_source, destination / f"{stem}.obj", args.overwrite)
        xml = destination / f"{stem}.xml"
        if not xml.exists() or args.overwrite:
            write_object_xml(xml, stem)
        prop = destination / f"prop_{stem}.csv"
        prop_source = source_dir / item["source_file"]
        if not prop.is_file() or args.overwrite:
            if stride == 1:
                # The source CSV already has the exact UMR columns.
                with prop_source.open(newline="", encoding="utf-8") as handle:
                    if sum(1 for _ in handle) - 1 != total_frames:
                        raise ValueError(f"Object/BVH frame mismatch: {prop_source}")
                link_or_copy(prop_source, prop, args.overwrite)
            else:
                write_prop_csv(prop_source, prop, indices, total_frames)
    return "reused" if existing and not args.overwrite else "converted"


def main() -> None:
    args = parse_args()
    if not (args.input / "raw_mocap").is_dir():
        raise FileNotFoundError(f"OmniContact raw_mocap missing: {args.input}")
    betas = np.asarray(np.load(args.reference / "betas.npy"), dtype=np.float32).reshape(-1)
    if betas.size < 10:
        raise ValueError(f"Need 10 reference SMPL-X betas: {args.reference}")
    # Mean offset against the released fitted example, retaining its world frame.
    reference_bvh = args.input / "raw_mocap/soccer/case3_kick_right/20260330000586_1_1775269234/motion_actor.bvh"
    reference_root = parse_bvh(reference_bvh)[1]
    reference_trans = np.asarray(np.load(args.reference / "transl.npy"), dtype=np.float32)
    if reference_trans.shape != reference_root.shape:
        raise ValueError("Reference sample and BVH frame counts differ")
    pelvis_offset = np.mean(reference_trans - reference_root, axis=0).astype(np.float32)
    selected = set(args.seq_key)
    paths = sorted((args.input / "raw_mocap").glob("*/*/*/capture_meta.json"))
    paths = [path for path in paths if not selected or path.parent.name in selected or str(path.parent.relative_to(args.input / "raw_mocap")) in selected]
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise FileNotFoundError(f"No matching OmniContact captures under {args.input / 'raw_mocap'}")
    print(f"[OmniContactConvert] captures={len(paths)} pelvis_offset_m={pelvis_offset.tolist()}", flush=True)
    if args.plan:
        for path in paths:
            print(path.parent.relative_to(args.input / "raw_mocap"))
        return
    counts = {"converted": 0, "reused": 0, "failed": 0}
    errors = []
    for index, path in enumerate(paths, 1):
        label = str(path.parent.relative_to(args.input / "raw_mocap"))
        try:
            outcome = process(path, args, betas[:10], pelvis_offset)
            counts[outcome] += 1
            print(f"[OmniContactConvert] {index}/{len(paths)} {label} {outcome}", flush=True)
        except Exception as error:
            counts["failed"] += 1
            errors.append(f"{label}\t{type(error).__name__}: {error}")
            print(f"[OmniContactConvert][FAIL] {errors[-1]}", flush=True)
    if errors:
        (args.output / "conversion_failures.txt").write_text("\n".join(errors) + "\n", encoding="utf-8")
    print(f"[OmniContactConvert] complete {counts} output={args.output}", flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
