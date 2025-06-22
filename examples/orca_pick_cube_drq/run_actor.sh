export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4 && \
python async_drq_randomized.py "$@" \
    --actor \
    --render \
    --env OrcaCubePick-Vision-v0 \
    --exp_name=serl_dev_drq_densereward_nodemo_orca_cube_pick_random_resnet \
    --seed 0 \
    --random_steps 0 \
    --training_starts 50 \
    --encoder_type resnet-pretrained \
    --demo_path /home/ccc/orca_ws/src/serl/examples/orca_pick_cube_drq/orca_cube_pick_1000_demos_2025-06-21_14-30-00.pkl \
    --debug \
