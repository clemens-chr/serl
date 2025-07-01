export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python async_drq_sim.py "$@" \
    --env OrcaRotate-v0 \
    --actor \
    --exp_name=orca_rotate_gp \
    --seed 0 \
    --save_video \
    --video_period 10000 \
    --batch_size 256 \
    --random_steps 1000 \
    --encoder_type resnet-pretrained \
    --demo_path /home/ccc/orca_ws/src/serl/examples/orca_pick_cube_drq/orca_cube_pick_21_demos_2025-06-29_14-24-02.pkl \
