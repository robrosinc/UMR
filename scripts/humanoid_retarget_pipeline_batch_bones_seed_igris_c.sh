python scripts/humanoid_retarget_pipeline_batch.py \
    --config robot_configs/humanoid_retarget_igris_c_example.json \
    --batch-config humanoid_retarget_defaults_batch_bones_seed.json \
    --output-root output/igris_c_bones_seed \
    --workers 8 \
    --retarget-cpu-threads 1 \
    --stride 4 \
    # --force-retarget \
