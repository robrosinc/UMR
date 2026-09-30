#!/usr/bin/env python3
"""Convert a local BONES-SEED download into UMR's SOMA motion layout.

The BVH poses are already the representation consumed by UMR. Conversion
extracts the archive, places shape/rig files, and installs SOMA-X runtime
assets; it does not rewrite the BVH rotations.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_FILES = (
    "SOMA_neutral.npz",
    "SOMA_template_rig.usda",
    "SOMA_procedural_transforms.json",
    "MHR/SOMA_wrap_lod1.obj",
    "MHR/base_body_lod1.obj",
    "MHR/mhr_model_lod1.pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/home/robros/workspace/motion_datas/bones-seed_origin"))
    parser.add_argument("--output", type=Path, default=ROOT / "sample_data/bones-seed")
    parser.add_argument("--soma-assets", type=Path, default=ROOT.parent / "SOMA-X/assets")
    parser.add_argument("--force-extract", action="store_true", help="Extract the uniform archive again")
    parser.add_argument("--no-download-assets", action="store_true", help="Fail if required public SOMA-X assets are unavailable locally")
    args = parser.parse_args()
    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.soma_assets = args.soma_assets.expanduser().resolve()
    if args.input == args.output or args.input in args.output.parents:
        parser.error("--output must be outside --input")
    return args


def link_or_copy(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if target.stat().st_size == source.stat().st_size:
            return
        raise ValueError(f"Existing target has a different size: {target}")
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def install_shapes(args: argparse.Namespace) -> None:
    shapes = args.input / "soma_shapes"
    link_or_copy(
        shapes / "soma_base_fit_mhr_params.npz",
        args.output / "shapes/soma_uniform_fit_mhr_params/soma_base_fit_mhr_params.npz",
    )
    for name in ("soma_base_skel_minimal.bvh", "soma_base_skel_minimal.usd"):
        link_or_copy(shapes / "soma_base_rig" / name, args.output / "shapes/soma_base_rig" / name)
    proportion = shapes / "soma_proportion_fit_mhr_params"
    for source in sorted(proportion.glob("A*.npz")):
        link_or_copy(source, args.output / "shapes/soma_proportion_fit_mhr_params" / source.name)
    count = len(list((args.output / "shapes/soma_proportion_fit_mhr_params").glob("A*.npz")))
    print(f"[BoneSeedConvert] shape params: uniform=1 proportional={count}", flush=True)


def install_assets(args: argparse.Namespace) -> None:
    missing = []
    for name in ASSET_FILES:
        target = args.output / "soma_assets" / name
        if target.is_file():
            continue
        source = args.soma_assets / name
        if source.is_file():
            link_or_copy(source, target)
        else:
            missing.append(name)
    if missing:
        if args.no_download_assets:
            raise FileNotFoundError(f"Missing SOMA-X assets: {missing}; expected under {args.soma_assets}")
        hf = shutil.which("hf")
        if hf is None:
            raise FileNotFoundError(f"Missing SOMA-X assets: {missing}; install the hf CLI or supply --soma-assets")
        destination = args.output / "soma_assets"
        destination.mkdir(parents=True, exist_ok=True)
        command = [hf, "download", "nvidia/SOMA-X"]
        for name in missing:
            command.extend(("--include", name))
        command.extend(("--local-dir", str(destination)))
        print(f"[BoneSeedConvert] downloading public SOMA-X assets: {', '.join(missing)}", flush=True)
        subprocess.run(command, check=True)
        unresolved = [name for name in missing if not (destination / name).is_file()]
        if unresolved:
            raise FileNotFoundError(f"SOMA-X download did not provide: {unresolved}")
    print(f"[BoneSeedConvert] SOMA-X assets: {args.output / 'soma_assets'}", flush=True)


def extract_uniform(args: argparse.Namespace) -> None:
    archive = args.input / "soma_uniform.tar.gz"
    if not archive.is_file():
        raise FileNotFoundError(archive)
    target = args.output / "motions_uniform"
    target.mkdir(parents=True, exist_ok=True)
    marker = target / ".umr_extraction.json"
    source_info = {"archive": str(archive), "size": archive.stat().st_size, "mtime_ns": archive.stat().st_mtime_ns}
    if marker.is_file() and not args.force_extract:
        try:
            cached = json.loads(marker.read_text(encoding="utf-8"))
            if all(cached.get(key) == value for key, value in source_info.items()) and (target / "bvh").is_dir():
                print(f"[BoneSeedConvert] reuse extracted uniform motions: {cached['bvh_count']} BVHs", flush=True)
                return
        except (ValueError, KeyError):
            pass
    print(f"[BoneSeedConvert] extracting {archive} -> {target / 'bvh'}", flush=True)
    marker.unlink(missing_ok=True)
    # The upstream archive has one leading soma_uniform/ directory.
    extractor = ["tar", "--use-compress-program=pigz", "-xf"] if shutil.which("pigz") else ["tar", "-xzf"]
    subprocess.run(extractor + [str(archive), "--strip-components=1", "-C", str(target)], check=True)
    count = sum(1 for _ in (target / "bvh").rglob("*.bvh"))
    if count == 0:
        raise RuntimeError(f"Archive extracted no BVH files under {target / 'bvh'}")
    marker.write_text(json.dumps({**source_info, "bvh_count": count}, indent=2) + "\n", encoding="utf-8")
    print(f"[BoneSeedConvert] uniform motions: {count} BVHs", flush=True)


def main() -> None:
    args = parse_args()
    install_shapes(args)
    install_assets(args)
    extract_uniform(args)
    print(f"[BoneSeedConvert] complete output={args.output}", flush=True)


if __name__ == "__main__":
    main()
