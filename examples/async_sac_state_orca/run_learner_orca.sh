export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python async_sac_state_orca.py "$@" \
    --env OrcaGraspStatic-v0 \
    --learner \
    --exp_name=orca_grasp_static_sac \
    --seed 0 \
    --max_steps 20001 \
    --training_starts 1000 \
    --checkpoint_period 5000 \
    --batch_size 256 \
    --critic_actor_ratio 8 \

# --demo_path /path/to/your/demo.pkl \ 