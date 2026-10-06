#!/usr/bin/env python3
"""Convert LAFAN1 BVH motions to UMR's SMPL-X NPZ layout.

The joint mapping, frame offsets, and neutral shape coefficients are adapted
from Joao Pedro Araujo's MIT-licensed lafan_to_smplx converter:
https://github.com/jaraujo98/lafan_to_smplx

MIT License
Copyright (c) 2025 Joao Pedro Araujo
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation
from smplx.joint_names import JOINT_NAMES


ROOT = Path(__file__).resolve().parents[1]
SHAPE_BETAS = np.array(
    [1.4775, 0.6674, -1.1742, 0.4731, 1.2984, -0.2159, 1.5276,
     -0.3152, -0.6441, -0.2986, 0.5089, -0.6354, 0.3321, -0.1099,
     -0.3060, -0.7330],
    dtype=np.float32,
)
SMPLX_TO_LAFAN = {
    "left_hip": "LeftUpLeg", "right_hip": "RightUpLeg", "spine1": "Spine",
    "left_knee": "LeftLeg", "right_knee": "RightLeg", "spine2": "Spine1",
    "left_ankle": "LeftFoot", "right_ankle": "RightFoot", "spine3": "Spine2",
    "left_foot": "LeftToe", "right_foot": "RightToe", "neck": "Neck",
    "left_collar": "LeftShoulder", "right_collar": "RightShoulder", "head": "Head",
    "left_shoulder": "LeftArm", "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm", "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand", "right_wrist": "RightHand",
}
AXIS_ROTATION = Rotation.from_matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])


def frame_offsets() -> dict[str, Rotation]:
    torso = Rotation.from_euler("z", -np.pi / 2) * Rotation.from_euler("y", -np.pi / 2)
    leg = Rotation.from_euler("z", np.pi / 2) * Rotation.from_euler("y", np.pi / 2)
    left_arm = Rotation.from_euler("x", np.pi / 2)
    right_arm = Rotation.from_euler("z", np.pi) * Rotation.from_euler("x", -np.pi / 2)
    result = {
        "Hips": torso, "Spine": torso, "Spine1": torso, "Spine2": torso,
        "Neck": torso, "Head": torso,
        "LeftFoot": Rotation.from_euler("z", 0.37117860986509) * Rotation.from_euler("y", np.pi / 2),
        "RightFoot": Rotation.from_euler("z", 0.37117860986509) * Rotation.from_euler("y", np.pi / 2),
        "LeftToe": Rotation.from_euler("y", np.pi / 2), "RightToe": Rotation.from_euler("y", np.pi / 2),
    }
    for name in ("LeftUpLeg", "LeftLeg", "RightUpLeg", "RightLeg"):
        result[name] = leg
    for name in ("LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"):
        result[name] = left_arm
    for name in ("RightShoulder", "RightArm", "RightForeArm", "RightHand"):
        result[name] = right_arm
    return result


def load_bvh(path: Path, max_frames: int) -> tuple[list[dict], np.ndarray, float]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    joints: list[dict] = []
    stack: list[int] = []
    end_site = False
    motion_line = -1
    for line_index, line in enumerate(lines):
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] in {"ROOT", "JOINT"}:
            joints.append({"name": tokens[1], "parent": stack[-1] if stack else -1,
                           "offset": np.zeros(3), "channels": []})
            stack.append(len(joints) - 1)
        elif tokens[:2] == ["End", "Site"]:
            end_site = True
        elif tokens[0] == "OFFSET" and not end_site:
            joints[stack[-1]]["offset"] = np.asarray(tokens[1:4], dtype=np.float64)
        elif tokens[0] == "CHANNELS" and not end_site:
            joints[stack[-1]]["channels"] = tokens[2:]
        elif tokens[0] == "}":
            if end_site:
                end_site = False
            else:
                stack.pop()
        elif tokens[0] == "MOTION":
            motion_line = line_index
            break
    if motion_line < 0 or not joints or stack:
        raise ValueError(f"Invalid BVH hierarchy: {path}")
    frame_count = None
    frame_time = None
    values_line = None
    for line_index in range(motion_line + 1, len(lines)):
        tokens = lines[line_index].split()
        if tokens[:1] == ["Frames:"]:
            frame_count = int(tokens[1])
        elif tokens[:2] == ["Frame", "Time:"]:
            frame_time = float(tokens[2])
            values_line = line_index + 1
            break
    if frame_count is None or frame_time is None or frame_time <= 0 or values_line is None:
        raise ValueError(f"Invalid BVH motion header: {path}")
    width = sum(len(joint["channels"]) for joint in joints)
    selected_count = min(frame_count, max_frames) if max_frames else frame_count
    values = np.fromstring("\n".join(lines[values_line:values_line + selected_count]), sep=" ")
    if values.size != selected_count * width or not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid BVH frame data: {path}; values={values.size}, expected={selected_count * width}")
    fps = 1.0 / frame_time
    if abs(fps - round(fps)) < 0.001:
        fps = float(round(fps))
    return joints, values.reshape(selected_count, width), fps


def lafan_pose(joints: list[dict], values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    names = [joint["name"] for joint in joints]
    name_to_index = {name: index for index, name in enumerate(names)}
    required = {"Hips", *SMPLX_TO_LAFAN.values()}
    missing = required - name_to_index.keys()
    if missing or len(name_to_index) != len(names) or names[0] != "Hips":
        raise ValueError(f"Unsupported LAFAN hierarchy; missing={sorted(missing)}")
    offsets = frame_offsets()
    world_rotations: list[Rotation] = []
    world_positions: list[np.ndarray] = []
    cursor = 0
    for joint in joints:
        channels = joint["channels"]
        data = values[:, cursor:cursor + len(channels)]
        cursor += len(channels)
        rotation_ids = [i for i, channel in enumerate(channels) if channel.endswith("rotation")]
        rotation_order = "".join(channels[i][0] for i in rotation_ids)
        if len(rotation_ids) != 3:
            raise ValueError(f"Expected three rotation channels for {joint['name']}")
        local_rotation = Rotation.from_euler(rotation_order.upper(), data[:, rotation_ids], degrees=True)
        parent = joint["parent"]
        position_ids = [i for i, channel in enumerate(channels) if channel.endswith("position")]
        if parent < 0:
            axes = [channels[i][0] for i in position_ids]
            if sorted(axes) != ["X", "Y", "Z"]:
                raise ValueError("Expected XYZ root position channels")
            root_position = np.stack([data[:, position_ids[axes.index(axis)]] for axis in "XYZ"], axis=1)
            world_rotations.append(local_rotation)
            world_positions.append(root_position)
        else:
            if position_ids:
                raise ValueError(f"Unexpected position channels for {joint['name']}")
            world_rotations.append(world_rotations[parent] * local_rotation)
            world_positions.append(world_positions[parent] + world_rotations[parent].apply(joint["offset"]))

    oriented = [AXIS_ROTATION * rotation for rotation in world_rotations]
    poses = np.zeros((len(values), 55, 3), dtype=np.float32)
    poses[:, 0] = (oriented[0] * offsets["Hips"]).as_rotvec().astype(np.float32)
    for pose_index, smplx_name in enumerate(JOINT_NAMES[1:22], start=1):
        lafan_name = SMPLX_TO_LAFAN.get(smplx_name)
        if lafan_name is None:
            continue
        joint_index = name_to_index[lafan_name]
        parent_index = joints[joint_index]["parent"]
        parent_name = names[parent_index]
        parent_rotation = oriented[parent_index] * offsets[parent_name]
        joint_rotation = oriented[joint_index] * offsets[lafan_name]
        poses[:, pose_index] = (parent_rotation.inv() * joint_rotation).as_rotvec().astype(np.float32)
    positions = np.stack(world_positions, axis=1)
    converted_positions = AXIS_ROTATION.apply(positions.reshape(-1, 3)).reshape(positions.shape) * 0.01
    return poses, converted_positions.mean(axis=1).astype(np.float32)


def convert(path: Path, model: smplx.SMPLX, device: torch.device, batch_size: int, max_frames: int) -> dict:
    joints, values, fps = load_bvh(path, max_frames)
    poses, source_centroid = lafan_pose(joints, values)
    translations = np.empty((len(poses), 3), dtype=np.float32)
    shape = torch.as_tensor(SHAPE_BETAS, device=device).reshape(1, -1)
    with torch.no_grad():
        for start in range(0, len(poses), batch_size):
            end = min(start + batch_size, len(poses))
            chunk = torch.as_tensor(poses[start:end], device=device)
            result = model(
                global_orient=chunk[:, 0],
                body_pose=chunk[:, 1:22].reshape(end - start, -1),
                betas=shape.expand(end - start, -1),
            )
            model_centroid = result.joints[:, :22].mean(dim=1).cpu().numpy()
            translations[start:end] = source_centroid[start:end] - model_centroid
    return {
        "poses": poses,
        "trans": translations,
        "betas": SHAPE_BETAS,
        "gender": "neutral",
        "mocap_frame_rate": np.float32(fps),
        "output_up": "z",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=Path("/home/robros/workspace/motion_datas/lafan1"))
    parser.add_argument("--output", type=Path, default=ROOT / "sample_data/lafan1_smplx")
    parser.add_argument("--model", type=Path, default=ROOT / "smpl/SMPLX_NEUTRAL.pkl")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="Maximum BVH files to convert; 0 means all")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum source frames per file; 0 means all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 0 or args.max_frames < 0:
        parser.error("--batch-size must be positive; --limit and --max-frames must be nonnegative")
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    if not source.exists() or not model_path.is_file():
        parser.error(f"Missing input or SMPL-X model: input={source} model={model_path}")
    if model_path.suffix.lower() not in {".pkl", ".npz"}:
        parser.error("--model must point to a .pkl or .npz SMPL-X model")
    files = [source] if source.is_file() else sorted(source.glob("*.bvh"))
    if not files or any(path.suffix.lower() != ".bvh" for path in files):
        parser.error(f"No BVH files found under {source}")
    if args.limit:
        files = files[:args.limit]
    pending = [path for path in files if args.overwrite or not (output / f"{path.stem}.npz").exists()]
    print(f"[LAFAN1Convert] selected={len(files)} pending={len(pending)} output={output}", flush=True)
    if args.plan or not pending:
        return
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    model = smplx.create(str(model_path), model_type="smplx", use_pca=False,
                         num_betas=len(SHAPE_BETAS), ext=model_path.suffix[1:].lower()).to(device).eval()
    output.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(pending, start=1):
        print(f"[LAFAN1Convert][{index}/{len(pending)}] {path.name}", flush=True)
        data = convert(path, model, device, args.batch_size, args.max_frames)
        with tempfile.NamedTemporaryFile(dir=output, suffix=".npz", delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            np.savez(temp_path, **data)
            os.replace(temp_path, output / f"{path.stem}.npz")
        finally:
            temp_path.unlink(missing_ok=True)
        print(f"[LAFAN1Convert] saved {path.stem}.npz frames={len(data['poses'])} fps={data['mocap_frame_rate']:.3f}", flush=True)


if __name__ == "__main__":
    main()
