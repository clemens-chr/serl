export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python async_drq_sim.py "$@" \
    --env OrcaRotate-v0 \
    --learner \
    --exp_name=orca_rotate_gp \
    --seed 0 \
    --max_steps 20001 \
    --training_starts 1000 \
    --checkpoint_period 2500 \
    --batch_size 256 \
    --critic_actor_ratio 4 \
    --encoder_type resnet-pretrained \
    --demo_path /home/ccc/orca_ws/src/serl/examples/orca_pick_cube_drq/orca_cube_pick_21_demos_2025-06-29_14-24-02.pkl \

# --demo_path /home/clemens/serl/serl/examples/async_drq_sim/demo_data/orca_static_grasp_3_demos_2025-05-29_16-41-43.pkl \
