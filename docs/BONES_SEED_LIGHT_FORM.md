# Bones Seed light form v1

Run `bash scripts/convert_bones_seed_umr_to_light.sh`. The shell script exposes
`INPUT_DIR`, `OUTPUT_DIR`, and `ROBOT_XML`; `--limit N` converts the first N
clips. The default output is `light_form/igris_c_bones_seed`.

```text
igris_c_bones_seed/
  format.json
  clips/<clip_id>.npz
  scenes/scene_<hash>.json
  assets/robot/models/model_<hash>.xml
  assets/robot/meshes/*
```

The NPZ keeps the retargeted qpos, fps, frame IDs, and portable metadata.
Original absolute `robot_xml` and `source_data` fields are removed. New
`robot_xml` and `light_scene_ref` paths are relative to the clip NPZ, and the
robot XML references only bundled meshes.

The UMR NPZs were produced with a single IGRIS-C robot. Many saved per-clip
XML paths point to a different checkout and most generated XMLs are absent.
The converter packages the configured `igris_c.xml` and checks the generated
XMLs that are available for equivalent compiled model data. Bones Seed clips
in this output have no separate moving object tracks; the scene JSON contains
only the static floor. The package can be viewed and expanded to canonical
form without the original BVH, source UMR output, or repository assets.
