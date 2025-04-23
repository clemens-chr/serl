export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python async_drq_sim.py "$@" \
    --env Orca1PickCubeVision-v0 \
    --actor \
    --exp_name=orca1_new_site \
    --eval_checkpoint_step 10000 \
    --render \
    --checkpoint_path /home/clemens/serl/serl/examples/async_drq_sim/orca1_new_site/20250422-221346/ \
