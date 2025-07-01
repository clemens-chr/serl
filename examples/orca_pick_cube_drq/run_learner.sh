export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4 && \
python async_drq_randomized.py "$@" \
    --learner \
    --env OrcaCubePickBinary-Vision-v0 \
    --exp_name=orca_pick_binary_2 \
    --seed 0 \
    --random_steps 1000 \
    --training_starts 50 \
    --critic_actor_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --checkpoint_path /home/ccc/orca_ws/src/serl/examples/orca_pick_cube_drq/orca_pick_binary_2 \
    --demo_path /home/ccc/orca_ws/src/serl/examples/orca_pick_cube_drq/orca_cube_pick_21_demos_2025-06-29_14-24-02.pkl \
