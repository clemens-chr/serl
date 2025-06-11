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
