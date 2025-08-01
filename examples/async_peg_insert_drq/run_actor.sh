export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4 && \
python async_drq_randomized.py "$@" \
    --actor \
    --render \
    --env FrankaPegInsert-Vision-v0 \
    --exp_name=serl_dev_drq_densereward_nodemo_peg_insert_random_resnet \
    --seed 0 \
    --random_steps 0 \
    --training_starts 50 \
    --encoder_type resnet-pretrained \
    --demo_path /home/ccc/orca_ws/src/serl/examples/async_peg_insert_drq/peg_insert_20_demos_2025-06-15_17-22-04.pkl \
    --debug \
