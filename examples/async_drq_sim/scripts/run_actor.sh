export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python async_drq_sim.py "$@" \
    --actor \
    --exp_name=gripper_demo_random_image \
    --seed 0 \
    --batch_size 256 \
    --save_video \
    --video_period 10000 \
    --random_steps 1000 \
    --encoder_type resnet-pretrained \
