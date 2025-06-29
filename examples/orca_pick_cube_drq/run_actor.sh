export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4 && \
python async_drq_randomized.py "$@" \
    --actor \
    --render \
    --env OrcaCubePickBinary-Vision-v0 \
    --exp_name=orca_pick_binary \
    --seed 0 \
    --random_steps 0 \
    --training_starts 0 \
    --encoder_type resnet-pretrained \
    --checkpoint_path /home/ccc/orca_ws/src/serl/examples/orca_pick_cube_drq/orca_pick_binary_1 \
    --demo_path /home/ccc/orca_ws/src/serl/examples/orca_pick_cube_drq/orca_cube_pick_21_demos_2025-06-27_11-17-48.pkl \
