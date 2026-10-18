# AMASS light form v1

Run `bash scripts/convert_amass_umr_to_light.sh`. Edit `INPUT_DIR` and
`OUTPUT_DIR` at the top of the shell script if needed; pass `--limit N` to
convert only the first N clips.

```text
igris_c_amass/
  format.json
  clips/<clip_id>.npz
  scenes/scene_<hash>.json
  assets/robot/models/model_<hash>.xml
  assets/robot/meshes/*
```

Each clip keeps its UMR `qpos`, `fps`, `frame_ids`, joint names, and portable
non-path metadata. Its `robot_xml` and `light_scene_ref` are paths relative to
the NPZ. Original absolute `robot_xml` and `source_data` paths are removed.
The robot XML references only meshes inside this package. Repeated per-clip
robot XML files and mesh files are stored once per unique content.

AMASS has no separate moving object tracks in this UMR output. The static
floor is recorded in the shared scene JSON; the bundled robot XML also keeps
its plane for MuJoCo playback. Viewer and common light→canonical expansion
work without the original UMR, AMASS source data, or repository assets.

To expand this dataset to canonical form, pass
`--input-dir light_form/igris_c_amass` and a separate `--output-dir` to
`scripts/expand_light_to_canonical.py`. See [the standalone expansion guide](LIGHT_TO_CANONICAL.md).
