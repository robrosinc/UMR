#!/usr/bin/env python3
"""Package IGRIS-C LAFAN1 UMR clips as portable light form v1.

LAFAN1 uses the same SMPL-X retarget result layout as AMASS: one robot model,
a static plane, and no separate moving object tracks. Reuse that packager and
record the correct source identity in the light-form manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from convert_amass_umr_to_light import convert_clip


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "output/igris_c_lafan1/igris_c")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "light_form/igris_c_lafan1")
    parser.add_argument("--limit", type=int, default=0, help="First N clips; 0 converts all")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir == args.input_dir or args.output_dir.is_relative_to(args.input_dir):
        parser.error("--output-dir must be outside --input-dir")
    return args


def main() -> None:
    args = parse_args()
    sources = sorted(args.input_dir.rglob("*.npz"))
    if args.limit:
        sources = sources[:args.limit]
    if not sources:
        raise FileNotFoundError(f"No LAFAN1 UMR NPZ clips in {args.input_dir}")
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
        if index % 50 == 0 or index == len(sources):
            print(f"[{index}/{len(sources)}] clips, {len(model_cache)} robot models, "
                  f"{len(mesh_cache)} shared meshes", flush=True)

    manifest = {
        "format": "umr_light", "version": 1, "source": "lafan1",
        "clip_count": len(sources), "robot_models": len(model_cache),
        "shared_robot_meshes": len(mesh_cache),
        "clip_paths_are_relative_to_npz": True,
        "units": "meters", "up_axis": "Z", "quaternion_order": "wxyz",
    }
    (args.output_dir / "format.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Light form ready: {args.output_dir}")


if __name__ == "__main__":
    main()
