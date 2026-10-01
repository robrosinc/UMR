#!/usr/bin/env python3
"""Convert OMOMO sequence joblibs to UMR's flat SMPL-X + object layout.

Only load OMOMO joblibs from a trusted source: joblib deserialization executes
Python objects. The source files are never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


VERSION = 2
SPLITS = ("train", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="OMOMO root containing data/*.p and data/captured_objects/*.obj")
    parser.add_argument("--output", type=Path, required=True, help="UMR output root")
    parser.add_argument("--fps", type=float, required=True, help="Frame rate of the released sequences; OMOMO joblibs do not store it")
    parser.add_argument("--seq-key", action="append", help="Convert only this sequence; repeatable")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of sequences, after filtering; 0 means all")
    parser.add_argument("--collision", choices=("hull", "coacd"), default="hull", help="Collision mesh preparation")
    parser.add_argument(
        "--coacd-threshold",
        type=float,
        default=0.03,
        help="CoACD concavity threshold; lower values produce finer decompositions (default: 0.03)",
    )
    parser.add_argument("--scale-warning", type=float, default=0.01, help="Warn above this relative per-frame scale range")
    parser.add_argument("--overwrite", action="store_true", help="Replace previously converted sequence files")
    args = parser.parse_args()
    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be positive and finite")
    if args.limit < 0 or args.scale_warning < 0:
        parser.error("--limit and --scale-warning must be nonnegative")
    if not np.isfinite(args.coacd_threshold) or not 0.01 <= args.coacd_threshold <= 1.0:
        parser.error("--coacd-threshold must be finite and in [0.01, 1.0]")
    if args.input == args.output or args.input in args.output.parents:
        parser.error("--output must be outside --input")
    return args


def source_mesh(data_dir: Path, object_name: str, part: str | None) -> Path:
    suffix = f"_{part}" if part else ""
    path = data_dir / "captured_objects" / f"{object_name}_cleaned_simplified{suffix}.obj"
    if not path.is_file():
        raise FileNotFoundError(f"OMOMO object mesh missing: {path}")
    return path


def obj_vertices_faces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                indices = []
                for field in line.split()[1:]:
                    index = int(field.split("/", 1)[0])
                    indices.append(index - 1 if index > 0 else len(vertices) + index)
                for index in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[index], indices[index + 1]))
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"Expected triangular OBJ geometry in {path}")
    return points, triangles


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for point in vertices:
            handle.write(f"v {point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n")
        for face in faces:
            handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def prepare_asset(
    source: Path,
    assets_dir: Path,
    collision: str,
    coacd_threshold: float,
) -> tuple[Path, list[Path]]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    visual = assets_dir / source.name
    if not visual.exists():
        shutil.copy2(source, visual)
    threshold_tag = f"_t{coacd_threshold:.6g}".replace(".", "p") if collision == "coacd" else ""
    asset_tag = f"{collision}{threshold_tag}"
    pattern = f"{source.stem}_{asset_tag}_collision_*.obj"
    cached = sorted(assets_dir.glob(pattern))
    if cached:
        return visual, cached

    vertices, faces = obj_vertices_faces(source)
    if collision == "hull":
        hull = ConvexHull(vertices)
        center = vertices[hull.vertices].mean(axis=0)
        hull_faces = hull.simplices.copy()
        for face in hull_faces:
            triangle = vertices[face]
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            if np.dot(normal, triangle.mean(axis=0) - center) < 0:
                face[1], face[2] = face[2], face[1]
        collision_paths = [assets_dir / f"{source.stem}_{collision}_collision_0.obj"]
        write_obj(collision_paths[0], vertices, hull_faces)
    else:
        try:
            import coacd
        except ImportError as error:
            raise RuntimeError("--collision coacd requires the coacd package from requirements-umr.txt") from error
        result = coacd.run_coacd(coacd.Mesh(vertices, faces), threshold=coacd_threshold)
        collision_paths = []
        for index, (part_vertices, part_faces) in enumerate(result):
            target = assets_dir / f"{source.stem}_{asset_tag}_collision_{index}.obj"
            write_obj(target, np.asarray(part_vertices), np.asarray(part_faces))
            collision_paths.append(target)
        if not collision_paths:
            raise RuntimeError(f"CoACD produced no collision meshes for {source}")
    return visual, collision_paths


def frame_array(record: dict, key: str, frames: int, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(record[key], dtype=np.float64)
    if value.shape != (frames, *shape) or not np.all(np.isfinite(value)):
        raise ValueError(f"{record['seq_name']}: invalid {key} shape/values: {value.shape}")
    return value


def write_mjcf(path: Path, stem: str, visual: Path, collisions: list[Path], scale: float) -> None:
    root = ET.Element("mujoco", {"model": stem})
    asset = ET.SubElement(root, "asset")
    scale_text = f"{scale:.10g} {scale:.10g} {scale:.10g}"
    for mesh_path, mesh_name in [(visual, f"{stem}_visual_mesh"), *[(part, f"{stem}_collision_{index}_mesh") for index, part in enumerate(collisions)]]:
        relative = os.path.relpath(mesh_path, path.parent)
        ET.SubElement(asset, "mesh", {"name": mesh_name, "file": relative, "scale": scale_text})
    world = ET.SubElement(root, "worldbody")
    body = ET.SubElement(world, "body", {"name": stem})
    ET.SubElement(body, "freejoint", {"name": f"{stem}_freejoint"})
    ET.SubElement(body, "geom", {
        "name": f"{stem}_visual", "type": "mesh", "mesh": f"{stem}_visual_mesh",
        "group": "2", "contype": "0", "conaffinity": "0", "rgba": "1 1 1 1",
    })
    for index, _ in enumerate(collisions):
        ET.SubElement(body, "geom", {
            "name": f"{stem}_collision_{index}", "type": "mesh",
            "mesh": f"{stem}_collision_{index}_mesh", "group": "3",
            "contype": "1", "conaffinity": "1", "rgba": "0.2 0.5 0.8 0.15",
        })
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_trajectory(path: Path, position: np.ndarray, matrices: np.ndarray) -> None:
    quaternions = Rotation.from_matrix(matrices).as_quat()  # xyzw
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("frame_id", "px", "py", "pz", "qx", "qy", "qz", "qw"))
        for index, (xyz, quat) in enumerate(zip(position, quaternions)):
            writer.writerow((index, *xyz.tolist(), *quat.tolist()))


def expected_files(seq_dir: Path, object_name: str) -> list[Path]:
    stems = (object_name, f"{object_name}_bottom") if object_name in {"mop", "vacuum"} else (object_name,)
    base = [seq_dir / name for name in ("poses.npy", "transl.npy", "betas.npy", "gender.npy", "model_type.npy", "mocap_framerate.npy", "output_up.npy", "metadata.json")]
    return base + [seq_dir / f"{stem}.xml" for stem in stems] + [seq_dir / f"prop_{stem}.csv" for stem in stems]


def output_current(seq_dir: Path, object_name: str, args: argparse.Namespace) -> bool:
    if not all(path.is_file() for path in expected_files(seq_dir, object_name)):
        return False
    try:
        metadata = json.loads((seq_dir / "metadata.json").read_text(encoding="utf-8"))
        return (
            metadata.get("converter_version") == VERSION
            and float(metadata.get("fps")) == args.fps
            and all(
                part.get("collision") == args.collision
                and (
                    args.collision != "coacd"
                    or float(part.get("coacd_threshold", -1.0)) == args.coacd_threshold
                )
                for part in metadata["object_parts"].values()
            )
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def convert_sequence(record: dict, split: str, data_dir: Path, output: Path, args: argparse.Namespace) -> list[tuple[str, float, int]]:
    name = str(record["seq_name"])
    if not name or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError(f"Unsafe OMOMO sequence name: {name!r}")
    components = name.split("_")
    if len(components) < 3:
        raise ValueError(f"Unexpected OMOMO sequence name: {name}")
    object_name = components[1]
    seq_dir = output / split / name
    existing = seq_dir.exists()
    if existing and not args.overwrite:
        if output_current(seq_dir, object_name, args):
            return []
        raise FileExistsError(f"Incomplete/incompatible output; use --overwrite to replace: {seq_dir}")

    frames = len(np.asarray(record["trans"]))
    root_orient = frame_array(record, "root_orient", frames, (3,))
    body_pose = frame_array(record, "pose_body", frames, (63,))
    translation = frame_array(record, "trans", frames, (3,))
    betas = np.asarray(record["betas"], dtype=np.float32).reshape(-1)
    if betas.size not in (10, 16) or not np.all(np.isfinite(betas)):
        raise ValueError(f"{name}: expected 10 or 16 finite betas, got {betas.shape}")
    gender = str(np.asarray(record["gender"]).item()).lower()
    if gender not in {"female", "male", "neutral"}:
        raise ValueError(f"{name}: unsupported gender {gender!r}")
    poses = np.zeros((frames, 165), dtype=np.float32)
    poses[:, :3] = root_orient
    poses[:, 3:66] = body_pose

    parts = [(object_name, None, "obj_trans", "obj_rot", "obj_scale")]
    if object_name in {"mop", "vacuum"}:
        parts[0] = (object_name, "top", "obj_trans", "obj_rot", "obj_scale")
        parts.append((f"{object_name}_bottom", "bottom", "obj_bottom_trans", "obj_bottom_rot", "obj_bottom_scale"))
    prepared = []
    warnings = []
    for stem, mesh_part, trans_key, rot_key, scale_key in parts:
        position = frame_array(record, trans_key, frames, (3, 1)).reshape(frames, 3)
        rotation = frame_array(record, rot_key, frames, (3, 3))
        scale = frame_array(record, scale_key, frames, ())
        valid_scale = scale[scale > 0]
        if not len(valid_scale):
            raise ValueError(f"{name}: no positive values in {scale_key}")
        invalid_frames = int(frames - len(valid_scale))
        median_scale = float(np.median(valid_scale))
        variation = float(np.ptp(valid_scale) / median_scale)
        if variation > args.scale_warning or invalid_frames:
            warnings.append((stem, variation, invalid_frames))
        source = source_mesh(data_dir, object_name, mesh_part)
        prepared.append((stem, position, rotation, median_scale, variation, invalid_frames, source))

    seq_dir.mkdir(parents=True, exist_ok=True)
    np.save(seq_dir / "poses.npy", poses)
    np.save(seq_dir / "transl.npy", translation.astype(np.float32))
    np.save(seq_dir / "betas.npy", betas)
    np.save(seq_dir / "gender.npy", np.asarray(gender))
    np.save(seq_dir / "model_type.npy", np.asarray("smplx"))
    np.save(seq_dir / "mocap_framerate.npy", np.asarray(args.fps, dtype=np.float32))
    np.save(seq_dir / "output_up.npy", np.asarray("z"))
    object_meta = {}
    assets_dir = output / "object_mjcf" / "assets"
    for stem, position, rotation, scale, variation, invalid_frames, source in prepared:
        visual, collisions = prepare_asset(source, assets_dir, args.collision, args.coacd_threshold)
        write_mjcf(seq_dir / f"{stem}.xml", stem, visual, collisions, scale)
        write_trajectory(seq_dir / f"prop_{stem}.csv", position, rotation)
        object_meta[stem] = {
            "source_mesh": str(source),
            "median_scale": scale,
            "relative_scale_range": variation,
            "nonpositive_scale_frames": invalid_frames,
            "collision": args.collision,
            "coacd_threshold": args.coacd_threshold if args.collision == "coacd" else None,
        }
    metadata = {"converter_version": VERSION, "source_sequence": name, "split": split, "frames": frames, "fps": args.fps, "object_parts": object_meta, "note": "SMPL-H body parameters placed in SMPL-X body slots; remaining pose slots are zero. Per-frame positive object scale is approximated by its median; nonpositive scale frames are not represented exactly."}
    (seq_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return warnings


def main() -> None:
    args = parse_args()
    data_dir = args.input / "data" if (args.input / "data").is_dir() else args.input
    paths = {split: data_dir / f"{split}_diffusion_manip_seq_joints24.p" for split in SPLITS}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"OMOMO sequence joblibs missing: {missing}")
    if args.collision == "coacd":
        try:
            import coacd  # noqa: F401
        except ImportError as error:
            raise RuntimeError("--collision coacd requires coacd from requirements-umr.txt") from error
    requested = set(args.seq_key or ())
    found = set()
    converted = skipped = 0
    two_part_seen = False
    warning_items = []
    for split, path in paths.items():
        records = joblib.load(path)
        for record in records.values():
            name = str(record["seq_name"])
            if requested and name not in requested:
                continue
            found.add(name)
            object_name = name.split("_")[1]
            two_part_seen |= object_name in {"mop", "vacuum"}
            seq_dir = args.output / split / name
            already = seq_dir.exists() and not args.overwrite and output_current(seq_dir, object_name, args)
            warnings = convert_sequence(record, split, data_dir, args.output, args)
            warning_items.extend((name, part, variation, invalid_frames) for part, variation, invalid_frames in warnings)
            skipped += int(already)
            converted += int(not already)
            if converted and converted % 100 == 0 and not already:
                print(f"Converted {converted} sequences", flush=True)
            if args.limit and converted + skipped >= args.limit:
                break
        del records
        if args.limit and converted + skipped >= args.limit:
            break
    unknown = requested - found if not args.limit else set()
    if unknown:
        raise ValueError(f"Requested sequence keys not found: {sorted(unknown)}")
    print(f"OMOMO conversion complete: converted={converted}, reused={skipped}, output={args.output}")
    if warning_items:
        varying = [(name, part, variation) for name, part, variation, _ in warning_items if variation > args.scale_warning]
        missing_scale = [(name, part, count) for name, part, _, count in warning_items if count]
        if varying:
            examples = ", ".join(f"{name}/{part}={variation:.1%}" for name, part, variation in varying[:10])
            print(f"WARNING: {len(varying)} object parts exceed {args.scale_warning:.1%} scale variation; fixed-size MJCF approximation. Examples: {examples}")
        if missing_scale:
            examples = ", ".join(f"{name}/{part}={count} frames" for name, part, count in missing_scale[:10])
            print(f"WARNING: {len(missing_scale)} object parts contain nonpositive scale frames; those frames cannot be represented exactly. Examples: {examples}")
    if two_part_seen:
        print("NOTE: mop/vacuum have independently moving parts; UMR's contact retargeter currently uses only the first discovered object.")


if __name__ == "__main__":
    main()
