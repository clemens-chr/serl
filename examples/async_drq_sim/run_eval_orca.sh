export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python async_drq_sim.py "$@" \
    --env OrcaRotate-v0 \
    --actor \
    --render \
    --exp_name=orca_rotate_gp \
    --eval_checkpoint_step 15000 \
    --checkpoint_path /home/clemens/serl/serl/examples/async_drq_sim/exp/orca_rotate_gp/20250606-095302/checkpoints \
    --max_steps 15000 \