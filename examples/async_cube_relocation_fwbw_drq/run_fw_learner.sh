export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python async_drq_randomized.py "$@" \
    --learner \
    --env FrankaCubeRelocation-Vision-v0 \
    --exp_name=cubefranka \
    --seed 0 \
    --random_steps 200 \
    --training_starts 200 \
    --critic_actor_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --fwbw fw \
    --demo_path /home/ccc/orca_ws/src/serl/examples/async_cube_relocation_fwbw_drq/fw_bin_demo_2025-06-21_14-23-51.pkl \
    --checkpoint_period 1000 \
    --checkpoint_path /home/ccc/orca_ws/src/serl/examples/async_cube_relocation_fwbw_drq/cubefranka_fw \
    --debug