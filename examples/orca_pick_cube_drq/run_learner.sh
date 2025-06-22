export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python async_drq_randomized.py "$@" \
    --learner \
    --env OrcaCubePick-Vision-v0 \
    --exp_name=serl_dev_drq_densereward_nodemo_orca_cube_pick_random_resnet \
    --seed 0 \
    --random_steps 1000 \
    --training_starts 50 \
    --critic_actor_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --checkpoint_period 1000 \
    --checkpoint_path /home/ccc/orca_ws/src/serl/examples/orca_pick_cube_drq/serl_dev_drq_densereward_nodemo_orca_cube_pick_random_resnet \
    --debug \
    --demo_path /home/ccc/orca_ws/src/serl/examples/orca_pick_cube_drq/orca_cube_pick_1000_demos_2025-06-21_14-30-00.pkl \
