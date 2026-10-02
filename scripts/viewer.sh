# # viser viewer - UMR format
# /home/robros/workspace/miniconda3/envs/UMR/bin/python /home/robros/workspace/UMR/scripts/visualize_robot_retarget_result.py \
#     --result-format umr \
#     --result-dir /home/robros/workspace/UMR/output/igris_c_omomo \
#     --viewer-backend viser \
#     --viser-host 0.0.0.0 \
#     --viser-port 8080 \
#     --play

# viser viewer - canonical format
/home/robros/workspace/miniconda3/envs/UMR/bin/python /home/robros/workspace/UMR/scripts/visualize_robot_retarget_result.py \
    --result-format canonical \
    --result-dir /home/robros/workspace/UMR/output/igris_c_omomo_canonical \
    --viewer-backend viser \
    --viser-host 0.0.0.0 \
    --viser-port 8080 \
    --play
