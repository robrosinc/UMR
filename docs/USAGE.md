
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
### OmniContact
`bash scripts/convert_omnicontact_to_umr.sh`
### 

## Retargeting Motion Datas
### OMOMO
`bash scripts/humanoid_retarget_pipeline_hsi_hoi_batch.sh`
* check omomo region
* Hand slots are force-bound to the configured palm geoms because point-cloud mapping can place them differently from the intended locations on long-armed robots. To disable this and use the default whole-visual-mesh binding, set `robot.slot_geom_names` to `{}` in `robot_configs/humanoid_retarget_igris_c_omomo.json`.

## Convert to ROBROSLAB Canonical form
### OMOMO
`bash scripts/convert_igris_c_omomo_canonical.sh`
* differ each robot platform


## Viewer Setting
* Selectable source
  * UMR retargeted
  * Canonical ROBROSLAB form converted

---
