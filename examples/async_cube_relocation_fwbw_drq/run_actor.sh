export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python async_drq_randomized.py "$@" \
    --actor \
    --render \
    --env FrankaCubeRelocation-Vision-v0 \
    --exp_name=cubefranka \
    --seed 0 \
    --random_steps 200 \
    --encoder_type resnet-pretrained \
    --fw_ckpt_path /home/ccc/orca_ws/src/serl/examples/async_cube_relocation_fwbw_drq/cubefranka_fw \
    --bw_ckpt_path /home/ccc/orca_ws/src/serl/examples/async_cube_relocation_fwbw_drq/cubefranka_bw \
