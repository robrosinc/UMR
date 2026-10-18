# HiPHI light form v1

Run `bash scripts/convert_hiphi_umr_to_light.sh`. The shell script exposes
`INPUT_DIR`, `OUTPUT_DIR`, `SAMPLE_ROOT`, and `ROBOT_XML`; pass `--limit N` to
convert only the first N clips. The default output is
`light_form/igris_c_hiphi`.

```text
igris_c_hiphi/
  format.json
  clips/<clip_id>.npz
  scenes/plane.json
  assets/robot/model.xml
  assets/robot/meshes/igris_c/*
  assets/objects/templates/*.xml
  assets/objects/meshes/*.obj
```

Each clip keeps the retargeted qpos, fps, frame IDs, and portable metadata.
The original absolute `source_data` and `source_object_dir` fields are removed;
`robot_xml` is replaced with a clip-relative path. Object names, XML references,
scales, positions, and rotations use the same five light-form fields as OMOMO.
The tracks are sampled at `frame_ids` and transformed into meters, Z up,
including the retarget `smpl_scale`. The object mesh scale in the NPZ includes
the HiPHI centimeter-to-meter factor of 0.01; the bundled XML mesh scale is
unit scale. No-object clips have zero-length object arrays.

Prepared HiPHI object XMLs retain their visual and decomposed collision meshes.
For an object with no prepared XML, the converter packages the original OBJ as
both the visual mesh and the convex-hull collision source. This preserves the
visual geometry and motion but gives a coarser collision shape. The number of
such object instances is reported as `fallback_collision_instances` in
`format.json`. The original viewer also fell back to the OBJ for these objects.
Fallback XMLs contain a named freejoint so the object pose track can be bound
during playback.

The package can be moved to another machine without `output/`, `sample_data/`,
or repository assets. The generic `scripts/expand_light_to_canonical.sh` handles
this format without a HiPHI-specific canonical converter.
