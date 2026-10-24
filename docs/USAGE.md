
## Raw Dataset Preparation
* All raw data would be stored in NAS and our external SSD. This command is only just in case.

### OMOMO
```
hf download snorfyang/omomo \
  data.tar.gz \
  --repo-type dataset \
  --local-dir ./OMOMO
```

### GRAIL
```
hf download nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL \
  --repo-type dataset \
  --include "data/**" \
  --exclude "*.mp4" \
  --local-dir ./PhysicalAI-Robotics-Locomanipulation-GRAIL
```

### BONES-SEED
```
hf auth login

hf download bones-studio/seed soma_uniform.tar.gz \
  --repo-type dataset \
  --local-dir /home/robros/workspace/motion_datas/bones-seed

hf download bones-studio/seed \
  --repo-type dataset \
  --include "soma_shapes/**" \
  --local-dir /home/robros/workspace/motion_datas/bones-seed
```

### HiPHI
```
hf download noitomrobotics/HiPHI \
  --repo-type dataset \
  --include "data/*" \
  --include "object_meshes/*" \
  --local-dir /home/robros/workspace/motion/HiPHI
```

### OmniContact
```
hf download lightcone02/OmniContact-Dataset \
  --repo-type dataset \
  --include "raw_mocap/**" \
  --include "metadata/**" \
  --include "assets/objects/**" \
  --local-dir /home/robros/workspace/motion_datas/OmniContact
```

## Convert Raw Data to Retarget-Ready Data
### OMOMO
`bash scripts/convert_omomo_to_umr.sh`
### GRAIL
* don't need to convert
### BONES-SEED
`bash scripts/convert_bones_seed_to_umr.sh`
### HiPHI
`bash scripts/convert_hiphi_to_umr.sh --plan --limit 5`
`bash scripts/convert_hiphi_to_umr.sh --workers 4` converts four clips concurrently
(the default); use `--workers 1` for sequential fitting. Completed clips are
reused when restarted.

See [the HiPHI adapter guide](../sample_data/hiphi/README.md) for direct archive
conversion and the required SMPL-X model/shape options.
### OmniContact
`bash scripts/convert_omnicontact_to_umr.sh`
### 

## Retargeting Motion Datas
### OMOMO
`bash scripts/humanoid_retarget_pipeline_hsi_hoi_batch.sh`
* check omomo region
* Hand slots are force-bound to the configured palm geoms because point-cloud mapping can place them differently from the intended locations on long-armed robots. To disable this and use the default whole-visual-mesh binding, set `robot.slot_geom_names` to `{}` in `robot_configs/humanoid_retarget_igris_c_omomo.json`.
### GRAIL
`bash scripts/humanoid_retarget_pipeline_hsi_hoi_batch_grail_igris_c.sh`

The default input is `sample_data/grail` and the default retarget
output is `output/igris_c_grail`. Use `--plan` first to list sequences without
training. A subset can be passed as the first positional argument.

### AMASS (SMPL-X → IGRIS-C)
`bash scripts/humanoid_retarget_pipeline_batch_amass_igris_c.sh`

Set `--motion-folder` in the shell script to the AMASS directory to process.
The script uses `--force-retarget`, so rerunning it updates existing results.

This batch alone selects `retarget.source_ground_align: "adaptive_foot_joint"`
in `humanoid_retarget_defaults_batch_amass.json`. During retargeting, it
estimates floor-height drift from likely planted feet and subtracts the
smoothed height from the source root translation in memory. A planted foot
must move slower than 0.25 m/s, lie within 2 cm of the local low foot height,
and be within 6 cm of the mesh's lowest point. When neither foot qualifies,
the correction is interpolated between nearby planted frames; outside that
range, the nearest correction is held. This preserves the relative height of
actual jumps. If fewer than 5% of frames qualify, the clip uses the previous
single-height correction. The source `.npz` files are not changed, and robot
feet are not directly snapped to the ground.

## Create light form and canonical form

### AMASS
`bash scripts/convert_amass_umr_to_light.sh` (UMR → light form)

### Bones Seed
`bash scripts/convert_bones_seed_umr_to_light.sh` (UMR → light form)

### HiPHI
`bash scripts/convert_hiphi_umr_to_light.sh` (UMR → light form)

### GRAIL
`bash scripts/convert_grail_umr_to_light.sh` (UMR → light form)

The GRAIL converter reads prepared `object_mjcf/` assets under
`sample_data/grail` by default. Set `SAMPLE_ROOT` to the matching
GRAIL dataset root if the retargeted clips came from another location.

### LAFAN1
`bash scripts/convert_lafan1_umr_to_light.sh` (UMR → light form)

### OMOMO
`bash scripts/convert_omomo_umr_to_light.sh` (UMR → light form)

### PICO teleoperation (GMR PKL)
`bash scripts/convert_pico_pkl_to_light.sh` (GMR PICO PKL → light form)

The script reads `GMR/pico/retargeted/` and the matching GMR IGRIS-C v2
XML by default. Set `INPUT_DIR`, `OUTPUT_DIR`, and `ROBOT_XML` in the shell
script or environment to use other paths. It packages the robot and its static
floor and accepts both live `record_meta` and offline `retarget_meta` PKLs;
the retargeted PKLs do not contain moving object or raw device tracks.
Clip IDs include the take name (for example, `take_1__pico_0000`) so the
common canonical expander can write unique flat `motion/*.pkl` filenames.
The LAN viewer lists the generated `light_form/igris_c_pico` folder.

All sources: `bash scripts/expand_light_to_canonical.sh` (light form → canonical;
기존 직접 변환 스크립트는 deprecated)

NAS 배포에는 각 `light_form/<dataset>/` 디렉터리 전체를 사용한다. 사용자는
UMR 전체를 설치할 필요 없이 공통 expand 스크립트와 최소 의존성만으로
canonical form을 생성할 수 있다. 명령어와 배포 파일 목록은
[Light form → canonical](LIGHT_TO_CANONICAL.md)을 참조한다.


## Viewer Setting
* Selectable source
  * UMR retargeted
  * Canonical ROBROSLAB form converted

---
