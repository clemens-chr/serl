export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python async_sac_state_orca.py "$@" \
    --env OrcaGraspStatic-v0 \
    --actor \
    --render \
    --exp_name=orca_grasp_static_sac \
    --eval_checkpoint_step 15000 \
    --debug