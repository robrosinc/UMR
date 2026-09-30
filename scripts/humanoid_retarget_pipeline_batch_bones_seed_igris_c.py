#!/usr/bin/env python3
"""Run UMR's BONES-SEED batch retargeter with the IGRIS C robot preset."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-folder", type=Path, default=ROOT / "sample_data/bones-seed/motions_uniform/bvh")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output/bones_seed_igris_c_retarget")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--correspondence-workers", type=int, default=1)
    args, extra = parser.parse_known_args()
    motion_folder = args.motion_folder.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not motion_folder.is_dir():
        parser.error(f"Motion folder missing: {motion_folder}; run scripts/prepare_bones_seed_data.py first")
    if not next(motion_folder.rglob("*.bvh"), None):
        parser.error(f"No BVH motions under {motion_folder}")
    if args.workers <= 0 or args.correspondence_workers <= 0:
        parser.error("Worker counts must be positive")
    cmd = [
        sys.executable,
        str(ROOT / "scripts/humanoid_retarget_pipeline_batch.py"),
        "--config", str(ROOT / "robot_configs/humanoid_retarget_igris_c_example.json"),
        "--batch-config", str(ROOT / "humanoid_retarget_defaults_batch_bones_seed.json"),
        "--motion-folder", str(motion_folder),
        "--output-root", str(output_root),
        "--workers", str(args.workers),
        "--correspondence-workers", str(args.correspondence_workers),
        *extra,
    ]
    raise SystemExit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
