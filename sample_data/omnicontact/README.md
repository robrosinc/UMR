# OmniContact adapter

The released OmniContact example uses the standard UMR HSI/HOI adapter. See the
[OmniContact project paper](https://huggingface.co/papers/2606.26201) for the
upstream data source. The local converter builds the flat SMPL-X sequence
layout below from the published BVH captures.

## UMR layout

```text
sample_data/omnicontact/
└── <category>/
    └── <motion>/
        └── <sequence-key>/
            ├── poses.npy
            ├── transl.npy
            ├── betas.npy
            ├── gender.npy                       # optional; neutral by default
            ├── model_type.npy                   # optional; must be smplx
            ├── mocap_framerate.npy              # optional; 30 by default
            ├── output_up.npy                    # optional; z by default
            ├── <object-name>.xml
            ├── <object-name>.obj
            └── prop_<object-name>.csv
```

The human-motion minimum is:

- `poses.npy` or `smpl_pose_axis_angle.npy`, shaped `[T, 165]` or `[T, 55, 3]`;
- `transl.npy` or `trans.npy`, shaped `[T, 3]`; and
- `betas.npy`, normally containing 10 or 16 SMPL-X shape coefficients.

All motion arrays must have the same frame count. Pose values are axis-angle
radians and translations are in meters. Place the licensed SMPL-X model for the
sequence gender in `smpl/`; the released neutral example uses
`smpl/SMPLX_NEUTRAL.pkl`.

An interaction object is discovered by a shared filename stem. Its
`prop_<object-name>.csv` is required and contains per-frame position and XYZW
quaternion columns (`px,py,pz,qx,qy,qz,qw`); `frame_id` and `timestamp` columns
may precede them. Provide an object MJCF XML when robot-object collision
constraints are required; an OBJ-only object can supply surface contact but
cannot be inserted into the MuJoCo collision model. Mesh paths inside the XML
must resolve relative to that XML. We recommend using
[CoACD](https://github.com/SarahWeiii/CoACD) to create separate convex collision
pieces for concave objects before retargeting.

## Convert the local raw dataset

```bash
scripts/convert_omnicontact_to_umr.sh \
  /home/robros/workspace/motion_datas/OmniContact \
  sample_data/omnicontact
```

Use `--seq-key <capture-id>` to select one capture or `--limit 10` for a small
subset. `--target-fps 30` reduces the 90 Hz source to 30 Hz; use a separate
output directory because the included fitted example is already at 90 Hz.
The converter preserves the existing fitted example unless `--overwrite` is
given. It uses the sample's SMPL-X betas and calibrated pelvis offset for the
other captures.
Most BVH joint rotations map directly to SMPL-X, but the fitted sample also has
an upper-body IK refinement that this converter does not reproduce. Actor body
sizes vary in the raw BVH; reusing one sample's betas is only an approximation.
Check the result in the viewer before using it for training or quality
comparisons.
Each generated `conversion.json` records this provenance. Object CSV frames
are subsampled together with the human motion, and MuJoCo XML plus the matching
mesh are placed in each sequence directory.

## Batch retarget to IGRIS C

```bash
scripts/humanoid_retarget_pipeline_hsi_hoi_batch_omnicontact_igris_c.sh \
  sample_data/omnicontact \
  output/igris_c_omnicontact_retarget \
  --limit 10
```

`--plan` lists sequences and correspondence templates without running the fit.
The batch driver fits one correspondence per unique gender/betas template and
reuses it for all matching motions. `--seq-key <capture-id>` selects one motion;
use the full `category__case__capture-id` key when duplicate capture IDs exist.
Pass `--force-retarget` to rerun motions already present in the output directory.
The IGRIS C config uses `assets/igris_c/igris_c.xml` and centers the robot
point cloud on `Link_Waist_Pitch`.

Files such as `motion_actor.npz`, IK reports, matched-name arrays, residuals,
`num_frames.npy`, and `frame_time.npy` may be retained as conversion provenance,
but the pipeline does not require them when the flat files above are present.

## Run the sample

```bash
python scripts/humanoid_retarget_pipeline_hsi_hoi.py \
  --config robot_configs/humanoid_retarget_unitree_g1_example.json \
  --defaults humanoid_retarget_defaults_hsi_hoi_standard.json
```

For another category or motion, pass the directory that contains the sequence
directories and the desired sequence key with `--data` and `--seq-key`.
