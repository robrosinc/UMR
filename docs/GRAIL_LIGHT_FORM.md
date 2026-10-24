# GRAIL light form v1

Run `bash scripts/humanoid_retarget_pipeline_hsi_hoi_batch_grail_igris_c.sh`
to retarget the ready dataset, then
`bash scripts/convert_grail_umr_to_light.sh` to package the results. The
default input is `output/igris_c_grail`, the prepared source root is
`sample_data/grail`, and the output is `light_form/igris_c_grail`.
`INPUT_DIR`, `OUTPUT_DIR`, `SAMPLE_ROOT`, `ROBOT_XML`, and `UMR_PYTHON` can be
set in the environment; Python options such as `--limit N` pass through the
shell script. The prepared `object_mjcf/<sequence>/` assets must match each
retargeted NPZ's source subset and sequence key.

```text
igris_c_grail/
  format.json
  clips/<clip_id>.npz
  scenes/plane.json
  assets/robot/model.xml
  assets/robot/meshes/igris_c/*
  assets/objects/templates/*.xml
  assets/objects/meshes/<subset>/<sequence>/*
```

Each clip contains `qpos`, `fps`, `frame_ids`, the five standard light object
arrays, and clip-relative robot, scene, and object template references. The
object positions and rotations are sampled from the prepared trajectory CSV at
`frame_ids`. Positions include ground alignment and `smpl_scale`. The mesh
scale in the NPZ includes `smpl_scale` once; the package's XML mesh scales are
unit scale. Visual OBJ, MTL, optional texture, and CoACD collision OBJ files
are all inside the package. Absolute source paths are removed from clip NPZs.

The optional `light_object_entity_types` field marks GRAIL slope and curb
meshes as static `terrain`; other GRAIL entries are `object`. The common
expander reads this field and places terrain meshes in `usd_assets/terrain`.
`light_object_semantic_labels` preserves a source label, and
`light_object_mobilities` marks static versus dynamic tracks. Older light-form
clips without these fields continue to expand as ordinary objects.

Expand with the common script:

```bash
LIGHT_INPUT_DIR=light_form/igris_c_grail \
CANONICAL_OUTPUT_DIR=canonical_form/igris_c_grail \
bash scripts/expand_light_to_canonical.sh
```
