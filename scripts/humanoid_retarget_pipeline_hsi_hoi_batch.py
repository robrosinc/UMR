#!/usr/bin/env python3
"""Batch OMOMO/GRAIL/OmniContact retargeting with one fit per SMPL-X template."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Sequence:
    key: str
    path: Path
    template: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "robot_configs/humanoid_retarget_igris_c_example.json")
    parser.add_argument("--defaults", type=Path, default=ROOT / "humanoid_retarget_defaults_hsi_hoi_standard.json")
    parser.add_argument("--data", type=Path, default=ROOT / "sample_data/omomo")
    parser.add_argument("--output", type=Path, default=None, help="Result .npz directory; default output/<robot>_retarget")
    parser.add_argument("--limit", type=int, default=0, help="First N sequences; 0 means all")
    parser.add_argument("--seq-key", action="append", default=[], help="Select a sequence stem; repeat to select more")
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--plan", action="store_true", help="List templates and sequence counts without loading the retargeter")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-retarget", action="store_true")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failed sequence")
    parser.add_argument("--max-frames", type=int, default=None, help="Retarget at most this many frames per sequence")
    args = parser.parse_args()
    args.config = args.config.expanduser().resolve()
    args.defaults = args.defaults.expanduser().resolve()
    args.data = args.data.expanduser().resolve()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.output is not None:
        args.output = args.output.expanduser().resolve()
    return args


def template_name(path: Path) -> str:
    gender = str(np.asarray(np.load(path / "gender.npy", allow_pickle=False)).item()).lower()
    model_type = str(np.asarray(np.load(path / "model_type.npy", allow_pickle=False)).item()).lower()
    if model_type != "smplx" or gender not in {"female", "male", "neutral"}:
        raise ValueError(f"Expected SMPL-X motion with known gender: {path}")
    betas = np.asarray(np.load(path / "betas.npy", allow_pickle=False), dtype=np.float32).reshape(-1)[:10]
    if betas.size < 10 or not np.all(np.isfinite(betas)):
        raise ValueError(f"Expected at least 10 finite SMPL-X betas: {path}")
    base = f"smplx_{gender}"
    if np.allclose(betas, 0.0):
        return base
    return f"{base}_betas_{hashlib.sha1(betas.tobytes()).hexdigest()[:10]}"


def discover(args: argparse.Namespace, base_config: dict) -> list[Sequence]:
    from humanoid_retarget_pipeline_hsi_hoi import is_grail_root, is_hsi_hoi_sequence_dir

    if is_grail_root(args.data):
        grail_roots = [args.data]
    else:
        grail_roots = [path for path in sorted(args.data.iterdir()) if is_grail_root(path)] if args.data.is_dir() else []
    selected = set(args.seq_key)
    if grail_roots:
        if args.split != "all":
            raise ValueError("--split applies only to OMOMO; pass a GRAIL subset as --data")
        template = str(base_config.get("smpl_template", {}).get("name", ""))
        if not template or template == "auto":
            raise ValueError("GRAIL batch requires a fixed smpl_template.name in the defaults")
        sequences = []
        seen = set()
        for root in grail_roots:
            for path in sorted((root / "recon").glob("*.pkl")):
                if selected and path.stem not in selected:
                    continue
                if not (root / "object_usd" / f"{path.stem}.usd").is_file():
                    raise FileNotFoundError(f"Missing GRAIL USD for {path}")
                if path.stem in seen:
                    raise ValueError(f"Duplicate GRAIL sequence stem across subsets: {path.stem}")
                seen.add(path.stem)
                sequences.append(Sequence(path.stem, path, template))
                if args.limit and len(sequences) >= args.limit:
                    return sequences
        if not sequences:
            raise FileNotFoundError(f"No matching GRAIL recon sequences under {args.data}")
        return sequences

    # OmniContact uses <category>/<case>/<capture>/poses.npy. The same flat
    # sequence layout also works when --data points to one case or capture.
    if is_hsi_hoi_sequence_dir(args.data):
        flat_paths = [args.data]
    elif args.data.is_dir():
        flat_paths = sorted({path.parent for path in args.data.rglob("poses.npy")})
    else:
        flat_paths = []
    is_omnicontact = any((path / "conversion.json").is_file() for path in flat_paths) or (
        args.data.name == "omnicontact" or "omnicontact" in args.data.parts
    )
    if is_omnicontact:
        if args.split != "all":
            raise ValueError("--split applies only to OMOMO; select an OmniContact category/case as --data")
        sequences = []
        selected = set(args.seq_key)
        for path in flat_paths:
            if not is_hsi_hoi_sequence_dir(path):
                continue
            relative = path.relative_to(args.data) if path != args.data else Path(path.name)
            key = "__".join(relative.parts)
            if selected and key not in selected and path.name not in selected:
                continue
            sequences.append(Sequence(key, path, template_name(path)))
            if args.limit and len(sequences) >= args.limit:
                break
        if not sequences:
            raise FileNotFoundError(f"No matching OmniContact SMPL-X sequences under {args.data}")
        return sequences

    splits = ("train", "test") if args.split == "all" else (args.split,)
    sequences = []
    seen = set()
    for split in splits:
        folder = args.data / split
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing OMOMO split: {folder}")
        for path in sorted(folder.iterdir()):
            if not path.is_dir() or not (path / "poses.npy").is_file() or (selected and path.name not in selected):
                continue
            if path.name in seen:
                raise ValueError(f"Duplicate sequence name across splits: {path.name}")
            seen.add(path.name)
            sequences.append(Sequence(path.name, path, template_name(path)))
            if args.limit and len(sequences) >= args.limit:
                return sequences
    if not sequences:
        raise FileNotFoundError(f"No matching OMOMO sequences under {args.data}")
    return sequences


def set_shared_cache(config: dict, template: str) -> None:
    from humanoid_retarget_pipeline_hsi_hoi import safe_name

    robot = str(config["robot"]["name"])
    suffix = f"{safe_name(robot)}_{safe_name(template)}_hsi_hoi"
    correspondence = config.setdefault("correspondence", {})
    correspondence.setdefault("dataset", {})["out"] = str(ROOT / "data" / f"correspondence_{suffix}.npz")
    correspondence.setdefault("train", {})["out_dir"] = str(ROOT / "output" / f"correspondence_{suffix}")


def sequence_config(base_config: dict, seq: Sequence, output: Path, work_dir: Path, args: argparse.Namespace, object_dir: Path | None = None):
    from humanoid_retarget_pipeline_hsi_hoi import (
        clean_config,
        export_standard_motion_npz,
        make_runtime_config,
        safe_name,
    )
    from humanoid_retarget_pipeline import smpl_template_config

    motion_path = export_standard_motion_npz(seq.key, seq.path, work_dir, dry_run=False, object_dir=object_dir)
    # make_runtime_config expects the single-sequence CLI's optional fields.
    options = argparse.Namespace(
        out=output / f"{safe_name(seq.key)}_hsi_hoi_{base_config['robot']['name']}.npz",
        start=None,
        end=None,
        stride=None,
        max_frames=args.max_frames,
        dry_run=False,
    )
    config, config_path = make_runtime_config(base_config, seq.key, seq.path, motion_path, work_dir, options, object_dir=object_dir)
    set_shared_cache(config, seq.template)
    actual_template = str(smpl_template_config(config)["name"])
    if actual_template != seq.template:
        raise ValueError(f"Template mismatch for {seq.key}: scanned={seq.template}, runtime={actual_template}")
    config_path.write_text(json.dumps(clean_config(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config, motion_path


def prepare_template(base_config: dict, representative: Sequence, output: Path, args: argparse.Namespace) -> Path:
    from humanoid_retarget_pipeline import (
        build_correspondence_dataset,
        correspondence_slots_compatible,
        slots_out,
        train_correspondence,
    )

    with tempfile.TemporaryDirectory(prefix="umr_hsi_hoi_template_") as temp:
        config, _motion = sequence_config(base_config, representative, output, Path(temp), args)
        slots = slots_out(config)
        if slots.is_file() and correspondence_slots_compatible(slots, config) and not (args.force_build or args.force_train):
            print(f"[HSIHOIBatch] reuse correspondence {representative.template}: {slots}", flush=True)
            return slots
        dataset = build_correspondence_dataset(config, force=args.force_build)
        slots = train_correspondence(config, dataset, force=args.force_train or args.force_build)
        if not correspondence_slots_compatible(slots, config):
            raise RuntimeError(f"Correspondence slots missing/incompatible for {representative.template}: {slots}")
        return slots


def retarget_sequence(base_config: dict, seq: Sequence, output: Path, slots: Path, args: argparse.Namespace) -> str:
    from humanoid_retarget_pipeline_hsi_hoi import (
        hsi_result_compatible,
        is_grail_sequence,
        patch_result_with_samp_metadata,
        prepare_grail_object_assets,
        retarget_hsi_motion,
    )
    from humanoid_retarget_pipeline import retarget_out

    with tempfile.TemporaryDirectory(prefix="umr_hsi_hoi_retarget_") as temp:
        work_dir = Path(temp)
        object_dir = (
            prepare_grail_object_assets(seq.key, seq.path, base_config, False, False, work_dir)
            if is_grail_sequence(seq.path)
            else None
        )
        config, motion_path = sequence_config(base_config, seq, output, work_dir, args, object_dir=object_dir)
        result = retarget_out(config)
        if result.is_file() and not args.force_retarget and hsi_result_compatible(result, config, motion_path, seq.key, seq.path):
            return "reused"
        result = retarget_hsi_motion(config, slots, motion_path, seq.key, seq.path, force=args.force_retarget)
        patch_result_with_samp_metadata(result, config, seq.key, seq.path, motion_path, dry_run=False, object_dir=object_dir)
        return "retargeted"


def main() -> None:
    args = parse_args()
    from humanoid_retarget_pipeline_hsi_hoi import load_hsi_hoi_config

    base_config = load_hsi_hoi_config(args.config, args.defaults)
    sequences = discover(args, base_config)
    groups: dict[str, list[Sequence]] = defaultdict(list)
    for seq in sequences:
        groups[seq.template].append(seq)
    print(f"[HSIHOIBatch] sequences={len(sequences)} unique_templates={len(groups)}")
    for name, members in sorted(groups.items()):
        print(f"[HSIHOIBatch] template={name} sequences={len(members)} representative={members[0].key}")
    if args.plan:
        return

    robot = str(base_config["robot"]["name"])
    output = args.output or ROOT / "output" / f"{robot}_retarget"
    output.mkdir(parents=True, exist_ok=True)

    slots_by_template = {}
    for name, members in sorted(groups.items()):
        print(f"[HSIHOIBatch] correspondence {name} ({len(slots_by_template) + 1}/{len(groups)})", flush=True)
        slots_by_template[name] = prepare_template(base_config, members[0], output, args)

    counts = {"retargeted": 0, "reused": 0, "failed": 0}
    failures = []
    for index, seq in enumerate(sequences, start=1):
        print(f"[HSIHOIBatch] motion {index}/{len(sequences)} {seq.key}", flush=True)
        try:
            status = retarget_sequence(base_config, seq, output, slots_by_template[seq.template], args)
            counts[status] += 1
        except Exception as error:
            counts["failed"] += 1
            failures.append(f"{seq.key}\t{type(error).__name__}: {error}")
            print(f"[HSIHOIBatch][FAIL] {failures[-1]}", flush=True)
            if args.fail_fast:
                break
    failure_file = output / "batch_failures.txt"
    failure_file.write_text("\n".join(failures) + ("\n" if failures else ""), encoding="utf-8")
    print(f"[HSIHOIBatch] complete {counts} output={output} failures={failure_file}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
