export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python async_drq_sim.py "$@" \
    --actor \
    --exp_name=gripper_baseline_512 \
    --seed 0 \
    --save_video \
    --batch_size 512 \
    --video_period 10000 \
    --random_steps 1000 \
    --encoder_type resnet-pretrained \
