# OMOMO light form v1

`scripts/convert_omomo_umr_to_light.sh` packages the OMOMO UMR output under
`light_form/igris_c_omomo/`. This directory is self-contained for robot and object
playback. The original `output/`, `sample_data/`, and `assets/` directories are
not needed after conversion.

```text
igris_c_omomo/
  format.json
  clips/<clip_id>.npz
  scenes/plane.json
  assets/robot/model.xml
  assets/robot/meshes/igris_c/*
  assets/objects/templates/*.xml
  assets/objects/meshes/*.obj
```

Each clip keeps `qpos`, `fps`, `frame_ids`, and the non-path UMR metadata. The
source absolute paths are removed. `robot_xml`, `light_scene_ref`, and each
`light_object_xml_refs` entry are paths **relative to the clip NPZ**. The
object XML templates refer to shared OBJ files relative to the template.

| Clip NPZ field | Shape | Meaning |
| --- | --- | --- |
| `light_form_version` | scalar | Schema version, currently 1. |
| `light_object_names` | `(N,)` | Stable object names. |
| `light_object_xml_refs` | `(N,)` | Relative template references. |
| `light_object_scales` | `(N,) float32` | Uniform mesh scale, including UMR `smpl_scale`. |
| `light_object_pos` | `(T,N,3) float32` | Object world position, meters, Z up. |
| `light_object_rot` | `(T,N,4) float32` | Object world rotation, wxyz. |

The object tracks are already selected by `frame_ids` and transformed into the
retarget coordinate frame. Both visual and collision OBJ files remain in the
shared asset directory. XML templates contain the visual and collision geom
groups with unit mesh scale; each clip supplies its object scale. `plane.json`
preserves the static terrain parameters for later canonical expansion.

The viewer accepts either a single clip with `--result`, or the folder with
`--result-format umr --result-dir light_form/igris_c_omomo --viewer-backend viser`.
The NPZ's `light_form_version` field selects light-form loading automatically.

To expand this package to canonical form, set `LIGHT_INPUT_DIR` and
`CANONICAL_OUTPUT_DIR`, then run:

```bash
LIGHT_INPUT_DIR=/path/to/igris_c_omomo CANONICAL_OUTPUT_DIR=/path/to/canonical_omomo \
bash scripts/expand_light_to_canonical.sh
```

The expander uses `light_object_scales` exactly once. It writes shared USD
geometry plus small per-scale USD references, so object vertices and collision
meshes have the right effective size without copying the source OBJ per clip.
Object positions already include the retarget scale and are not scaled again.
The source-specific direct UMR-to-canonical scripts are deprecated.
