export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python async_drq_sim.py "$@" \
    --env OrcaPickCubeVision-v0 \
    --learner \
    --exp_name=orca_new_site \
    --seed 0 \
    --max_steps 19500 \
    --training_starts 1000 \
    --checkpoint_period 5000 \
    --batch_size 256 \
    --critic_actor_ratio 4 \
    --encoder_type resnet-pretrained \
