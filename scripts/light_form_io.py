"""Shared output boundary for every dataset's light-form packager.

UMR/GMR motion and MuJoCo XML enter in native wxyz order. This module writes
xyzw light clips and scenes. New dataset adapters must use these writers;
they must not reorder rotations before calling them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from quaternion_layout import light_payload_to_xyzw, wxyz_to_xyzw


def write_light_clip_from_wxyz(clip: Path, payload: dict) -> None:
    """Convert root/object rotations and atomically save an xyzw light NPZ.

    The caller's payload is unchanged, so retrying a write cannot apply the
    permutation twice. No normalization, sign correction or dtype cast occurs.
    """
    stored = dict(payload)
    light_payload_to_xyzw(stored)
    clip.parent.mkdir(parents=True, exist_ok=True)
    temporary = clip.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **stored)
    temporary.replace(clip)


def write_light_scene_from_mujoco(root: ET.Element, directory: Path,
                                 filename: str | None = None) -> Path:
    """Extract static world planes, converting MuJoCo quat to rot_xyzw."""
    entities = []
    for index, geom in enumerate(root.findall("./worldbody/geom")):
        if geom.get("type") != "plane":
            continue
        entities.append({
            "entity_id": geom.get("name", f"plane_{index}"),
            "entity_type": "terrain", "primitive": "plane",
            "size": [float(value) for value in geom.get("size", "1 1 0.05").split()],
            "pos": [float(value) for value in geom.get("pos", "0 0 0").split()],
            "rot_xyzw": wxyz_to_xyzw(np.asarray(
                [float(value) for value in geom.get("quat", "1 0 0 0").split()]
            )).tolist(),
        })
    payload = json.dumps({"version": 1, "static_entities": entities},
                         indent=2, ensure_ascii=False) + "\n"
    if filename is None:
        digest = hashlib.sha256(payload.encode()).hexdigest()[:24]
        scene = directory / f"scene_{digest}.json"
        if scene.is_file() and scene.read_text() != payload:
            raise ValueError(f"Scene hash collision: {scene}")
    else:
        scene = directory / filename
    scene.parent.mkdir(parents=True, exist_ok=True)
    temporary = scene.with_suffix(".json.tmp")
    temporary.write_text(payload)
    temporary.replace(scene)
    return scene
