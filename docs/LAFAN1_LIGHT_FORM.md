# LAFAN1 light form v1

Run `bash scripts/convert_lafan1_umr_to_light.sh`. Edit `INPUT_DIR` and
`OUTPUT_DIR` in the shell script if needed; `--limit N` converts the first N
clips. The default output is `light_form/igris_c_lafan1`.

```text
igris_c_lafan1/
  format.json
  clips/<clip_id>.npz
  scenes/scene_<hash>.json
  assets/robot/models/model_<hash>.xml
  assets/robot/meshes/*
```

The package preserves retargeted `qpos`, `fps`, `frame_ids`, joint names, and
portable metadata. It replaces the original robot XML path with a clip-relative
reference and removes the absolute `source_data` path. The robot XML and its
meshes are bundled once per unique content. LAFAN1 clips have no separate
moving object tracks; the scene JSON records the shared static floor.

Move the complete directory to another machine to play or expand it without
the original UMR output, LAFAN1 source files, or repository assets. Use the
common [light-to-canonical expander](LIGHT_TO_CANONICAL.md) with
`--input-dir light_form/igris_c_lafan1` and a separate output directory.
