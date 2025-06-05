#!/bin/bash


# Run the actor
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python async_sac_state_orca.py "$@" \
    --env OrcaGraspStatic-v0 \
    --actor \
    --render \
    --exp_name=orca_grasp_static_sac \
    --seed 0 \
    --batch_size 256 \
    --random_steps 1000 \
    --debug 