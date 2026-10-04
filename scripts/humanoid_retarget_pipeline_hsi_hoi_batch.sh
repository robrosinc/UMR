# # omomo
# python /home/robros/workspace/UMR/scripts/humanoid_retarget_pipeline_hsi_hoi_batch.py \
#     --config /home/robros/workspace/UMR/robot_configs/humanoid_retarget_igris_c_omomo.json \
#     --defaults /home/robros/workspace/UMR/humanoid_retarget_defaults_hsi_hoi_omomo.json \
#     --data /home/robros/workspace/UMR/sample_data/omomo \
#     --output /home/robros/workspace/UMR/output/igris_c_omomo_dummy \
#     --workers 8 \
#     --retarget-cpu-threads 1 \
#     # --force-retarget

# GRAIL
python /home/robros/workspace/UMR/scripts/humanoid_retarget_pipeline_hsi_hoi_batch.py \
    --config /home/robros/workspace/UMR/robot_configs/humanoid_retarget_igris_c_grail.json \
    --defaults /home/robros/workspace/UMR/humanoid_retarget_defaults_hsi_hoi_grail.json \
    --data /home/robros/workspace/UMR/sample_data/grail/sitting \
    --output /home/robros/workspace/UMR/output/igris_c_grail \
    --workers 8 \
    --retarget-cpu-threads 1 \
    # --force-retarget

# # OmniContact
# python /home/robros/workspace/UMR/scripts/humanoid_retarget_pipeline_hsi_hoi_batch.py \
#     --config /home/robros/workspace/UMR/robot_configs/humanoid_retarget_igris_c_omomo.json \
#     --defaults /home/robros/workspace/UMR/humanoid_retarget_defaults_hsi_hoi_omnicontact.json \
#     --data /home/robros/workspace/UMR/sample_data/omnicontact/soccer \
#     --output /home/robros/workspace/UMR/output/igris_c_omnicontact \
#     --workers 8 \
#     --retarget-cpu-threads 1 \
#     --force-retarget \
#     # --force-build \
#     # --force-train
