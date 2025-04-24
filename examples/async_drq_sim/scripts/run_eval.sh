export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python async_drq_sim.py "$@" \
    --actor \
    --exp_name=gripper_no_demo \
    --eval_checkpoint_step 10000 \
    --render \
    --checkpoint_path /home/clemens/serl/serl/examples/async_drq_sim/gripper_no_demo/20250421-183623/ \
