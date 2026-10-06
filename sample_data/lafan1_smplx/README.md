# LAFAN1 / SMPL-X adapter

UMR consumes a surface-producing SMPL-X sequence rather than the original
LAFAN1 BVH directly. Download motion from the
[LAFAN1 repository](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
and convert it to SMPL-X locally with the included script. LAFAN1 is licensed under
[CC BY-NC-ND 4.0](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/license.txt).
This repository includes one converted sequence, `dance1_subject2.npz`. The
converter's joint mapping and shape coefficients follow the MIT-licensed
[`lafan_to_smplx`](https://github.com/jaraujo98/lafan_to_smplx) workflow.

## Convert BVH files

Place the neutral SMPL-X body model at `smpl/SMPLX_NEUTRAL.pkl`. With the
original BVH files under `/home/robros/workspace/motion_datas/lafan1/`, run:

```bash
bash scripts/convert_lafan1_to_umr.sh
```

The script converts each `*.bvh` in that directory into a same-named `.npz`
under `sample_data/lafan1_smplx/`. It keeps the source frame count and frame
rate and skips existing `.npz` files, including the bundled example. To
convert one file, choose a different output directory, or change the batch
size:

```bash
bash scripts/convert_lafan1_to_umr.sh /path/to/clip.bvh /path/to/output --batch-size 32
```

Use `--overwrite` to replace existing output, `--limit N` to convert the first
N source files, or `--plan` to list the selected and pending file counts.
The script selects CUDA when available; use `--device cpu` to force CPU.
Set `PYTHON_BIN` to use another Python environment with NumPy, SciPy, PyTorch,
and `smplx` installed. The Python entry point is
`scripts/convert_lafan1_to_umr.py` and accepts the same options with named
`--input` and `--output` paths.

## UMR layout

```text
sample_data/lafan1_smplx/
├── README.md
├── dance1_subject2.npz
└── <another-locally-converted-sequence>.npz
```

Each `.npz` represents one sequence, and its filename stem is the sequence key.
A compatible file contains:

- pose data in one of these forms: `poses` or `pose_aa`, shaped `[T, 165]` or
  `[T, 55, 3]`; alternatively `root_orient` plus `pose_body`;
- root translation as `trans` or `trans_orig`, shaped `[T, 3]`;
- `betas` or `beta`, normally 10 or 16 coefficients;
- optional scalar `gender`, defaulting to `neutral`;
- optional scalar `mocap_frame_rate`, `mocap_framerate`, or `fps`, defaulting to
  30; and
- optional scalar `output_up`, defaulting to `z`.

Pose rotations are SMPL-X axis-angle values in radians and translations should
be in meters. All arrays belonging to the motion must use the same frame count.
Place the corresponding licensed body model at `smpl/SMPLX_NEUTRAL.pkl`, or use
`SMPLX_MALE.pkl`/`SMPLX_FEMALE.pkl` when the sequence metadata selects that
gender.

## Run the included sequence

```bash
python scripts/humanoid_retarget_pipeline.py \
  --config robot_configs/humanoid_retarget_unitree_g1_example.json
```

The defaults select the included `dance1_subject2.npz`. For another file,
set `motion.data` to this directory and `motion.seq_key` to the new filename
stem in the selected defaults or robot configuration.

## Batch retargeting

The batch script scans this directory for converted `.npz` files and uses
`bidirectional` warm start with DP:

```bash
bash scripts/humanoid_retarget_pipeline_batch_lafan1_igris_c.sh
```

Set `WORKERS=8` to choose the retarget worker count. Extra command-line flags
are forwarded to the batch pipeline, for example `--limit 1 --dry-run` or
`--motion-folder /path/to/other/npz`. Results are stored under
`output/igris_c_lafan1/igris_c/`, where `batch_summary.json` records the status
and output path of every clip.

## Convert retarget results to IGRIS-C canonical

```bash
bash scripts/convert_igris_c_lafan1_canonical.sh
```

This reads `output/igris_c_lafan1/igris_c/*.npz` and writes canonical motion
PKLs, floor scene tracks and USD, and a bundled IGRIS-C robot model under
`output/igris_c_lafan1_canonical/`. Pass `--limit 1` to convert one clip first.
