# HiPHI adapter

UMR accepts [HiPHI](https://noitom-robotics.github.io/hiphi) motion after it
has been converted from BVH to SMPL-X. This repository includes one converted
non-interaction example, `Getting_up-get_up_0004`. The full HiPHI dataset and
SMPL-X body models are not distributed with UMR.

## Convert the released BVH packages

The [official HiPHI release](https://github.com/noitom-robotics/hiphi/blob/main/docs/repository_layout.md)
stores motions in `data/*.tar.zst`, with `data/motion_to_part.csv` mapping each
motion and its mirrored counterpart to an archive. Keep the archives compressed.
The converter reads the index, extracts selected packages into temporary space,
fits the actor BVH to SMPL-X, and copies the metadata and referenced object
tracks and meshes to `sample_data/hiphi/`. It also accepts an already extracted
HiPHI root with `data/<frame>/<lu>/<motion_id>/` packages.

Install the [official `hiphi2smplx` converter](https://github.com/noitom-robotics/hiphi2smplx)
in the Python environment used by UMR. Its body fitting requires a licensed
`SMPLX_NEUTRAL.npz`. The converter checks these prerequisites before it extracts
an archive. The local model is `external_assets/hiphi/SMPLX_NEUTRAL.npz`, and
the beta-fitting resources are
`external_assets/hiphi/beta_fit/neutral_smpl_mean_params.h5` and
`external_assets/hiphi/beta_fit/gmm_08.pkl`. No sibling model repository is searched.
These licensed files are kept locally and ignored by Git; copy them into the
same paths on another machine. A one-motion conversion then needs only:

```bash
bash scripts/convert_hiphi_to_umr.sh --seq-key Self_motion-vault_0069
```

To convert the dataset with four clips fitting at once:

```bash
bash scripts/convert_hiphi_to_umr.sh --workers 4
```

`--workers N` sets the number of concurrent clip conversions within each
archive; the shell launcher defaults to 4, while `--workers 1`
restores sequential fitting. Completed packages are
reused when the command is restarted. Each worker starts its own `hiphi2smplx`
process and may use the selected GPU for body-shape fitting. If GPU memory or
RAM becomes tight, reduce the worker count. During parallel fitting the script
limits each fitter's OpenMP and BLAS threads to one to avoid CPU oversubscription.

Use `--skip-errors` to continue after a single clip fails. Each failed clip gets
a diagnostic log under `sample_data/hiphi/conversion_errors/<frame>/<lu>/<motion_id>.log`.
The final summary reports `failed=N` and returns a nonzero exit code if any
clips failed. Successful clips remain ready; rerunning the same command retries
only incomplete clips and removes a failure log after a successful retry.
Use `--exclude-seq-key Cause_to_move_in_place-twirl_0001` to omit a known bad
motion entirely on later runs. Add another `--exclude-seq-key` for its mirror if
needed.

The shell launcher also defaults to 1 beta iteration and 1 IK iteration per
frame. To try these settings on one unconverted motion, use:

```bash
bash scripts/convert_hiphi_to_umr.sh --seq-key MOTION_ID_HERE
```

The official defaults are 100 beta iterations and at most 10 IK iterations per
frame. Override the shell defaults with `--beta-iters` and `--mink-iters`.
Lower values trade fitting quality for speed; compare a sample clip before
applying them to the full dataset. Existing converted clips are reused unless
`--overwrite` is supplied.

For another machine, or to override the detected files, choose one of these
shape options:

```bash
# Preview one original motion without extraction. Add its __mirror ID to select both.
bash scripts/convert_hiphi_to_umr.sh --plan --seq-key Bringing-carry_0001

# Fit a body shape for each BVH (requires the two official beta-fit resources).
bash scripts/convert_hiphi_to_umr.sh \
  --model-path /path/to/SMPLX_NEUTRAL.npz \
  --fit-betas --beta-fit-data /path/to/beta_fit --beta-device cuda:0 \
  --seq-key Bringing-carry_0001

# Or reuse known beta coefficients from an NPY/NPZ file.
bash scripts/convert_hiphi_to_umr.sh \
  --model-path /path/to/SMPLX_NEUTRAL.npz --betas /path/to/betas.npy \
  --seq-key Bringing-carry_0001
```

The default input is the sibling `motion_datas/HiPHI_origin/` dataset and the
default output is `sample_data/hiphi/`. The first and second positional
arguments to the shell script override those roots. `--seq-key <motion_id>`
(repeatable), `--limit N`, and `--plan` select or preview a subset; existing
complete outputs are reused. A full archive is decompressed only once for all
selected motions within that archive, then its temporary files are removed.
The official fitter currently fits the 22 body joints; face and finger poses
remain neutral, so hand contact fidelity may differ from the included example.

The example already in `sample_data/hiphi/` is converted. The full dataset
needs to be obtained separately; this repository does not contain its BVH
archives. Running without `--seq-key` selects all indexed motions, so first
preview a limited subset when setting up the converter and model files.

## Dataset layout

Keep converted motions and the original HiPHI object meshes in this layout:

```text
sample_data/hiphi/
├── object_meshes/
│   └── <mesh_id>.obj
└── data/
    └── <frame>/<lu>/<motion_id>/
        ├── motion_actor_smplx.npz
        ├── metadata.json
        └── object_tracks/
            └── <object_id>.csv        # interaction motions only
```

Each sequence directory contains:

- `motion_actor_smplx.npz`: SMPL-X pose, translation, body shape, gender, and
  frame-rate data.
- `metadata.json`: a HiPHI record with `dataset`, `motion_id`, and `objects`.
- `object_tracks/<object_id>.csv`: the per-frame object trajectory for an
  interaction motion.

The `objects` list in `metadata.json` may specify `object_id`, `mesh_id`,
`mesh_path`, and `trajectory_path`. When paths are omitted, UMR resolves
`object_meshes/<mesh_id>.obj` from the dataset root and
`object_tracks/<object_id>.csv` from the sequence directory.

## Run the included example

Place an appropriate licensed SMPL-X body model under `smpl/`, then run:

```bash
python scripts/humanoid_retarget_pipeline_hiphi.py \
  --data sample_data/hiphi/data/Getting_up/get_up/Getting_up-get_up_0004
```

The same command accepts any converted motion directory that follows the
layout above. Plain motions use the non-interaction solver settings;
interaction motions automatically use the HiPHI HSI/HOI settings and their
object trajectories.

## Shared correspondence across body shapes

Different HiPHI actors and capture sessions may have different SMPL-X beta
parameters. UMR learns one beta-zero SMPL-X correspondence for each target
robot and transfers its fixed face and barycentric binding to the
sequence-specific body shape. The single-motion and batch adapters train this
shared correspondence when it is missing, then reuse it instead of retraining
correspondence for every motion.

## Interaction-object preprocessing

The downloaded HiPHI data provides object meshes as OBJ files. The standard
solver uses a hard robot-object penetration constraint. For that constraint,
UMR checks for a same-name MJCF beside each OBJ and runs CoACD if it is missing.
The generated assets are written under:

```text
sample_data/hiphi/object_meshes/object_collision/<mesh_id>/
```

The original OBJ remains the visual geometry. The generated convex pieces are
used for MuJoCo collision constraints and are reused on later runs. In
`--contact-only` mode, UMR samples contact points directly from the original
OBJ surface instead; the object is not replaced by a convex hull. That mode
does not prevent the robot from penetrating the object.

## Batch retargeting

The batch runner automatically skips clips without a completed
`motion_actor_smplx.npz`. The shell launcher defaults to `--stride 3`, which
retargets every third frame (about 30 Hz for 90 Hz HiPHI clips) and saves the
reduced FPS. Pass `--stride 1` to retain every frame. While conversion is still
running, run:

```bash
bash scripts/humanoid_retarget_pipeline_hiphi_batch.sh
```

Set `INPUT_DIR`, `OUTPUT_DIR`, and `CONTACT_ONLY` at the top of the shell script,
or override paths with `--data-root` and `--output-root`. Results are placed in an
`igris_c_dummy_hand/` subdirectory. Contact-only uses an `_contact_only`
output root unless `--output-root` is supplied.

To start retargeting immediately from completed SMPL-X clips and raw object
meshes, use the contact-only mode:

```bash
bash scripts/humanoid_retarget_pipeline_hiphi_batch.sh --contact-only
```

This mode keeps the object surface contact cost, skips CoACD, and disables the
robot-object penetration constraints. Robot motion can therefore intersect an
object. By default it writes to `output/batch_retarget_hiphi_contact_only/igris_c_dummy_hand/`
so results are not mixed with standard collision-constrained runs. Existing
processes keep their original mode; restart the batch runner to use this option.
The first run may still train the shared SMPL-X-to-robot correspondence before
the first motion starts; subsequent runs reuse it.

The batch runner takes a snapshot of converted clips, skips packages still
being fitted, and reuses compatible retarget results on later runs. It exits
after processing that snapshot; run it again to include newly converted clips.
Pass `--workers N` to run parallel retarget jobs. The shell launcher uses GPU 0
by default; pass `--retarget-gpus none` for CPU or another GPU ID to override it.
Object preprocessing and retargeting both default to one worker.

The default target is `robot_configs/humanoid_retarget_igris_c_hiphi.json`,
which uses the Igris C dummy-hand XML and a separate shared correspondence.
The dummy spheres follow wrist yaw; they simplify the target robot hand, not
the original HiPHI actor or object mesh. HiPHI BVH conversion uses the SMPL-X
body model to fit motion and copies the original object OBJ without fitting it.

The batch adapter discovers sequence directories from `metadata.json`, prepares
each unique missing object once, prepares or reuses the shared beta-zero
correspondence, and retargets the motions. Results are written under
`output/batch_retarget_hiphi/igris_c_dummy_hand/`.

Object preprocessing and retargeting default to one worker to limit peak
memory. Use `--object-workers N` and `--workers N` only when the machine has
enough memory. Use repeatable `--sequence <motion_id>` arguments to process a
subset of motions.
