#!/usr/bin/env python3
"""Prepare original HiPHI archives or extracted packages for UMR.

HiPHI's BVH joint offsets describe a capture skeleton, not SMPL-X parameters.
The official fitter supplies the body-pose/translation mapping; this script
handles dataset discovery, metadata, object assets, and repeatable batch runs.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MOTION_NAME = "motion_actor_smplx.npz"
DEFAULT_INPUT = ROOT.parent / "motion_datas/HiPHI_origin"
DEFAULT_OUTPUT = ROOT / "sample_data/hiphi"
LOCAL_MODEL_EXAMPLE = ROOT / "external_assets/hiphi/SMPLX_NEUTRAL.npz"
LOCAL_BETA_FIT_EXAMPLE = ROOT / "external_assets/hiphi/beta_fit"
DEFAULT_MODEL = LOCAL_MODEL_EXAMPLE if LOCAL_MODEL_EXAMPLE.is_file() else None
DEFAULT_BETA_FIT_DATA = (LOCAL_BETA_FIT_EXAMPLE if all(
    (LOCAL_BETA_FIT_EXAMPLE / name).is_file()
    for name in ("neutral_smpl_mean_params.h5", "gmm_08.pkl")
) else None)
SHARED_ASSET_LOCK = Lock()


@dataclass(frozen=True)
class Sequence:
    relative: Path
    archive: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Original HiPHI root with data/*.tar.zst, or an extracted dataset root")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="UMR dataset root (default: sample_data/hiphi)")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL,
                        help="Licensed SMPLX_NEUTRAL.npz (uses the local model when available)")
    shape = parser.add_mutually_exclusive_group()
    shape.add_argument("--betas", type=Path, help="NPY/NPZ with at least 10 beta coefficients")
    shape.add_argument("--fit-betas", action="store_true", help="Fit body shape from the BVH first frame")
    parser.add_argument("--beta-fit-data", type=Path, default=DEFAULT_BETA_FIT_DATA,
                        help="Directory with neutral_smpl_mean_params.h5 and gmm_08.pkl (uses local files when available)")
    parser.add_argument("--beta-device", default="cuda:0", help="Shape-fitting device (e.g. cpu)")
    parser.add_argument("--beta-iters", type=int, default=100,
                        help="Maximum iterations in each beta-fitting optimization step (default: 100)")
    parser.add_argument("--mink-iters", type=int, default=10,
                        help="Maximum body IK iterations per frame (default: 10)")
    parser.add_argument("--converter", default="hiphi2smplx", help="Official fitter executable")
    parser.add_argument("--seq-key", action="append", default=[],
                        help="Motion ID or frame/lu/motion_id; repeatable")
    parser.add_argument("--limit", type=int, default=0, help="Maximum selected sequences; 0 means all")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent clip conversions within each archive (default: 4)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing converted motion")
    parser.add_argument("--plan", action="store_true", help="Show selected packages without changing files")
    args = parser.parse_args()
    if not args.fit_betas and args.betas is None and args.beta_fit_data is not None:
        args.fit_betas = True
    if args.converter == "hiphi2smplx" and shutil.which(args.converter) is None:
        sibling_converter = Path(sys.executable).parent / args.converter
        if sibling_converter.is_file():
            args.converter = str(sibling_converter)
    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.beta_iters < 1:
        parser.error("--beta-iters must be at least 1")
    if args.mink_iters < 1:
        parser.error("--mink-iters must be at least 1")
    if not args.input.is_dir():
        parser.error(f"HiPHI input root does not exist: {args.input}")
    if args.input != args.output and (args.input in args.output.parents or args.output in args.input.parents):
        parser.error("--input and --output must be separate roots, or the same directory")
    return args


def check_motion(path: Path, expected_frames: int | None = None, expected_fps: float | None = None) -> None:
    with np.load(path, allow_pickle=False) as packed:
        pose_key = "poses" if "poses" in packed else "smpl_pose_axis_angle"
        trans_key = "transl" if "transl" in packed else "trans"
        poses = packed[pose_key]
        transl = packed[trans_key]
        betas = packed["betas"]
        fps_key = "mocap_framerate" if "mocap_framerate" in packed else "fps"
        fps = float(np.asarray(packed[fps_key]).item())
        model_type = str(np.asarray(packed["model_type"]).item()) if "model_type" in packed else "smplx"
        up = str(np.asarray(packed["output_up"]).item()) if "output_up" in packed else "y"
        if poses.ndim not in (2, 3) or poses.shape[1:] not in ((165,), (55, 3)):
            raise ValueError(f"Invalid SMPL-X poses shape {poses.shape}: {path}")
        if transl.shape != (len(poses), 3) or betas.size < 10:
            raise ValueError(f"Invalid translation or betas shape: {path}")
        if expected_frames is not None and len(poses) != expected_frames:
            raise ValueError(f"Frame count differs from metadata ({len(poses)} != {expected_frames}): {path}")
        if expected_fps is not None and not np.isclose(fps, expected_fps, rtol=1e-4, atol=1e-3):
            raise ValueError(f"Frame rate differs from metadata ({fps} != {expected_fps}): {path}")
        if not np.isfinite(fps) or fps <= 0 or model_type.lower() != "smplx" or up.lower() != "y":
            raise ValueError(f"Invalid frame rate, model type, or up axis: {path}")
        if not np.isfinite(poses).all() or not np.isfinite(transl).all() or not np.isfinite(betas).all():
            raise ValueError(f"Nonfinite SMPL-X values: {path}")


def relative_for_motion_id(motion_id: str) -> Path:
    match = re.fullmatch(r"([^/\\-]+)-([^/\\]+)_\d{4}(?:__mirror)?", motion_id)
    if match is None:
        raise ValueError(f"Invalid HiPHI motion ID in archive index: {motion_id!r}")
    return Path(match.group(1), match.group(2), motion_id)


def discover(root: Path, requested: list[str], limit: int) -> list[Sequence]:
    data_root = root / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"Expected HiPHI data directory: {data_root}")
    index_path = data_root / "motion_to_part.csv"
    found: list[Sequence] = []
    if index_path.is_file():
        with index_path.open(newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            if not {"motion_id", "mirrored_motion_id", "archive_name"}.issubset(rows.fieldnames or []):
                raise ValueError(f"Invalid HiPHI archive index: {index_path}")
            for row in rows:
                archive_name = row["archive_name"]
                if Path(archive_name).name != archive_name or not archive_name.endswith(".tar.zst"):
                    raise ValueError(f"Invalid archive name in {index_path}: {archive_name!r}")
                for motion_id in (row["motion_id"], row["mirrored_motion_id"]):
                    relative = relative_for_motion_id(motion_id)
                    if requested and motion_id not in requested and relative.as_posix() not in requested:
                        continue
                    found.append(Sequence(relative, data_root / archive_name))
    else:
        for metadata_path in sorted(data_root.rglob("metadata.json")):
            seq_dir = metadata_path.parent
            relative = seq_dir.relative_to(data_root)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if str(metadata.get("dataset", "")).lower() != "hiphi":
                continue
            if len(relative.parts) != 3 or relative.name != metadata.get("motion_id"):
                raise ValueError(f"Invalid HiPHI frame/lu/motion_id layout: {seq_dir}")
            if not (seq_dir / "motion_actor.bvh").is_file() and not (seq_dir / MOTION_NAME).is_file():
                raise FileNotFoundError(f"Neither BVH nor SMPL-X motion found: {seq_dir}")
            if requested and relative.name not in requested and relative.as_posix() not in requested:
                continue
            found.append(Sequence(relative))
    found.sort(key=lambda item: item.relative.as_posix())
    if requested:
        matched = {value for value in requested if any(
            value in (item.relative.name, item.relative.as_posix()) for item in found
        )}
        if missing := sorted(set(requested) - matched):
            raise FileNotFoundError(f"HiPHI sequences not found: {missing}")
    if not found:
        raise FileNotFoundError(f"No HiPHI sequences found under {data_root}")
    selected = found[:limit] if limit else found
    for item in selected:
        if item.archive is not None and not item.archive.is_file():
            raise FileNotFoundError(f"HiPHI archive not found: {item.archive}")
    return selected


def relative_asset(value: str, base: Path, root: Path) -> Path:
    """Resolve an official metadata path while keeping copies inside output."""
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe HiPHI asset path: {value}")
    resolved = (base / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"HiPHI asset escapes input root: {value}")
    return resolved


def copy_if_needed(source: Path, destination: Path, overwrite: bool) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == destination:
        return
    if destination.is_file() and not overwrite:
        if not filecmp.cmp(source, destination, shallow=False):
            raise FileExistsError(f"Existing asset differs from source: {destination}; use --overwrite")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def load_metadata(source: Path, relative: Path) -> dict:
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if str(metadata.get("dataset", "")).lower() != "hiphi":
        raise ValueError(f"Not a HiPHI package: {source}")
    if metadata.get("motion_id") != relative.name or metadata.get("frame") != relative.parts[0] or metadata.get("lu") != relative.parts[1]:
        raise ValueError(f"HiPHI metadata does not match package path: {source}")
    return metadata


def validate_object_track(path: Path, expected_frames: int, fps: float) -> None:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid HiPHI frame rate for object track: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"frame", "time_sec", "px", "py", "pz", "qx", "qy", "qz", "qw"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"HiPHI object track is missing columns: {path}")
        count = 0
        for count, row in enumerate(reader, 1):
            frame = int(row["frame"])
            timestamp = float(row["time_sec"])
            values = np.asarray([float(row[key]) for key in ("px", "py", "pz", "qx", "qy", "qz", "qw")])
            if (frame != count - 1 or not np.isclose(timestamp, (count - 1) / fps, atol=2e-3, rtol=0)
                    or not np.isfinite(values).all() or np.linalg.norm(values[3:]) < 1e-6):
                raise ValueError(f"Invalid or unsynchronized HiPHI object track at frame {count - 1}: {path}")
        if count != expected_frames:
            raise ValueError(f"HiPHI object track frame count differs from metadata ({count} != {expected_frames}): {path}")


def package_assets(source: Path, source_root: Path, metadata: dict, shared_root: Path) -> list[tuple[Path, Path]]:
    objects = metadata.get("objects", [])
    if not isinstance(objects, list) or (metadata.get("is_hoi") and not objects):
        raise ValueError(f"Invalid HiPHI object list: {source / 'metadata.json'}")
    assets: list[tuple[Path, Path]] = []
    for item in objects:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid HiPHI object entry: {source / 'metadata.json'}")
        object_id = item.get("object_id") or item.get("mesh_id")
        mesh_id = item.get("mesh_id") or object_id
        if not object_id or not mesh_id:
            raise ValueError(f"Missing HiPHI object_id/mesh_id: {source / 'metadata.json'}")
        track = relative_asset(item.get("trajectory_path") or f"object_tracks/{object_id}.csv", source, source_root)
        mesh = relative_asset(item.get("mesh_path") or f"object_meshes/{mesh_id}.obj", shared_root, shared_root)
        if not track.is_file() or not mesh.is_file():
            raise FileNotFoundError(f"HiPHI object track or mesh missing: {track}, {mesh}")
        validate_object_track(track, int(metadata["frame_count"]), float(metadata["fps"]))
        assets.extend(((track, track.relative_to(source_root)), (mesh, mesh.relative_to(shared_root))))
    return assets


def copy_package_assets(source: Path, source_root: Path, relative: Path, metadata: dict, args: argparse.Namespace) -> None:
    dest = args.output / "data" / relative
    assets = package_assets(source, source_root, metadata, args.input)
    copy_if_needed(source / "metadata.json", dest / "metadata.json", args.overwrite)
    for asset, target_relative in assets:
        if target_relative.parts[0] == "object_meshes":
            with SHARED_ASSET_LOCK:
                copy_if_needed(asset, args.output / target_relative, args.overwrite)
        else:
            copy_if_needed(asset, args.output / target_relative, args.overwrite)


def fit_bvh(source_bvh: Path, target: Path, args: argparse.Namespace, expected_frames: int, expected_fps: float) -> None:
    if args.model_path is None or not args.model_path.is_file():
        raise FileNotFoundError("BVH fitting requires --model-path pointing to SMPLX_NEUTRAL.npz")
    if not args.fit_betas and args.betas is None:
        raise ValueError("BVH fitting requires --betas or --fit-betas")
    command = [args.converter, "--input-bvh", str(source_bvh), "--model-path", str(args.model_path),
               "--mink-iters", str(args.mink_iters)]
    if args.fit_betas:
        if args.beta_fit_data is None or not args.beta_fit_data.is_dir():
            raise FileNotFoundError("--fit-betas requires --beta-fit-data directory")
        command.extend(["--fit-betas", "--beta-fit-data", str(args.beta_fit_data),
                        "--beta-device", args.beta_device, "--beta-iters", str(args.beta_iters)])
    else:
        if not args.betas.is_file():
            raise FileNotFoundError(args.betas)
        command.extend(["--betas", str(args.betas)])
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hiphi_fit_", dir=target.parent) as temp_dir:
        temp_path = Path(temp_dir) / MOTION_NAME
        command.extend(["--output", str(temp_path)])
        env = os.environ.copy()
        if args.workers > 1:
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[name] = "1"
        subprocess.run(command, check=True, env=env)
        check_motion(temp_path, expected_frames, expected_fps)
        temp_path.replace(target)


def output_ready(relative: Path, args: argparse.Namespace) -> bool:
    dest = args.output / "data" / relative
    target = dest / MOTION_NAME
    if args.overwrite or not target.is_file() or not (dest / "metadata.json").is_file():
        return False
    try:
        metadata = load_metadata(dest, relative)
        check_motion(target, int(metadata["frame_count"]), float(metadata["fps"]))
        package_assets(dest, args.output, metadata, args.output)
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
        return False
    return True


def extract_archive(archive: Path, selected: list[Sequence], destination: Path) -> None:
    if not archive.is_file():
        raise FileNotFoundError(f"HiPHI archive not found: {archive}")
    members = [f"HiPHI/data/{item.relative.as_posix()}/" for item in selected]
    command = ["tar", "--zstd", "-xf", str(archive), "-C", str(destination),
               "--strip-components=1", "--no-same-owner", "--no-same-permissions", *members]
    subprocess.run(command, check=True)


def process_sequence(item: Sequence, source_root: Path, args: argparse.Namespace) -> str:
    source = source_root / "data" / item.relative
    metadata = load_metadata(source, item.relative)
    expected_frames = int(metadata["frame_count"])
    expected_fps = float(metadata["fps"])
    dest = args.output / "data" / item.relative
    target = dest / MOTION_NAME
    source_npz = source / MOTION_NAME
    if output_ready(item.relative, args):
        return "reuse"
    if source_npz.is_file() and (source_npz != target or not (source / "motion_actor.bvh").is_file()):
        check_motion(source_npz, expected_frames, expected_fps)
        copy_package_assets(source, source_root, item.relative, metadata, args)
        copy_if_needed(source_npz, target, args.overwrite)
        return "copy" if source_npz != target else "reuse"
    if not (source / "motion_actor.bvh").is_file():
        raise FileNotFoundError(f"HiPHI BVH missing: {source}")
    copy_package_assets(source, source_root, item.relative, metadata, args)
    fit_bvh(source / "motion_actor.bvh", target, args, expected_frames, expected_fps)
    return "fit"


def needs_bvh_fit(item: Sequence, args: argparse.Namespace) -> bool:
    if item.archive is not None:
        return True
    source = args.input / "data" / item.relative
    source_npz = source / MOTION_NAME
    target = args.output / "data" / item.relative / MOTION_NAME
    return not (source_npz.is_file() and (source_npz != target or not (source / "motion_actor.bvh").is_file()))


def validate_fit_setup(args: argparse.Namespace) -> None:
    """Reject missing prerequisites before decompressing any HiPHI archive."""
    errors = []
    if args.model_path is None or not args.model_path.is_file():
        hint = f" (local model: {LOCAL_MODEL_EXAMPLE})" if LOCAL_MODEL_EXAMPLE.is_file() else ""
        errors.append(f"--model-path must point to SMPLX_NEUTRAL.npz{hint}")
    if args.fit_betas:
        if args.beta_fit_data is None or not args.beta_fit_data.is_dir():
            hint = f" (local resources: {LOCAL_BETA_FIT_EXAMPLE})" if LOCAL_BETA_FIT_EXAMPLE.is_dir() else ""
            errors.append(f"--fit-betas requires --beta-fit-data{hint}")
        else:
            missing = [name for name in ("neutral_smpl_mean_params.h5", "gmm_08.pkl")
                       if not (args.beta_fit_data / name).is_file()]
            if missing:
                errors.append(f"--beta-fit-data is missing {', '.join(missing)}: {args.beta_fit_data}")
    elif args.betas is None:
        errors.append("choose --fit-betas with --beta-fit-data, or --betas with an NPY/NPZ file")
    elif not args.betas.is_file():
        errors.append(f"--betas file does not exist: {args.betas}")
    if shutil.which(args.converter) is None:
        errors.append(f"converter executable not found: {args.converter}; install hiphi2smplx in this Python environment")
    if errors:
        raise ValueError("HiPHI BVH fitting prerequisites are missing:\n  - " + "\n  - ".join(errors))


def process_group(items: list[tuple[int, Sequence]], source_root: Path,
                  args: argparse.Namespace, total: int) -> Counter:
    """Process independent clips while keeping archive extraction alive."""
    counts: Counter = Counter()
    if args.workers == 1:
        for index, item in items:
            action = process_sequence(item, source_root, args)
            print(f"[HiPHI][{index}/{total}] {action}: {item.relative}", flush=True)
            counts[action] += 1
        return counts

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_sequence, item, source_root, args): (index, item)
            for index, item in items
        }
        try:
            for future in as_completed(futures):
                index, item = futures[future]
                action = future.result()
                print(f"[HiPHI][{index}/{total}] {action}: {item.relative}", flush=True)
                counts[action] += 1
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return counts


def main() -> int:
    args = parse_args()
    try:
        selected = discover(args.input, args.seq_key, args.limit)
        print(f"[HiPHI] input={args.input} output={args.output} sequences={len(selected)} workers={args.workers}")
        counts: Counter = Counter()
        if args.plan:
            for index, item in enumerate(selected, 1):
                action = "reuse" if output_ready(item.relative, args) else ("extract + fit" if item.archive else "fit/copy")
                print(f"[HiPHI][{index}/{len(selected)}] {action}: {item.relative}")
            print(f"[HiPHI] plan selected={len(selected)}")
            return 0

        ready = {item.relative for item in selected if output_ready(item.relative, args)}
        if any(needs_bvh_fit(item, args) for item in selected if item.relative not in ready):
            validate_fit_setup(args)

        archive_groups: dict[Path, list[tuple[int, Sequence]]] = defaultdict(list)
        extracted: list[tuple[int, Sequence]] = []
        for index, item in enumerate(selected, 1):
            if item.relative in ready:
                print(f"[HiPHI][{index}/{len(selected)}] reuse: {item.relative}", flush=True)
                counts["reuse"] += 1
            elif item.archive is None:
                extracted.append((index, item))
            else:
                archive_groups[item.archive].append((index, item))

        counts.update(process_group(extracted, args.input, args, len(selected)))

        for archive, items in archive_groups.items():
            print(f"[HiPHI] extract {len(items)} selected packages from {archive.name}", flush=True)
            with tempfile.TemporaryDirectory(prefix="hiphi_archive_") as temporary:
                source_root = Path(temporary)
                extract_archive(archive, [item for _, item in items], source_root)
                counts.update(process_group(items, source_root, args, len(selected)))
        print(f"[HiPHI] complete fitted={counts['fit']} copied={counts['copy']} reused={counts['reuse']}")
        return 0
    except (FileNotFoundError, FileExistsError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"[HiPHI] error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
