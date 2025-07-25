export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python async_drq_randomized.py "$@" \
    --learner \
    --env FrankaPegInsert-Vision-v0 \
    --exp_name=serl_dev_drq_densereward_nodemo_peg_insert_random_resnet \
    --seed 0 \
    --random_steps 1000 \
    --training_starts 50 \
    --critic_actor_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --checkpoint_period 1000 \
    --checkpoint_path /home/ccc/orca_ws/src/serl/examples/async_peg_insert_drq/serl_dev_drq_densereward_nodemo_peg_insert_random_resnet \
    --debug \
    --demo_path /home/ccc/orca_ws/src/serl/examples/async_peg_insert_drq/peg_insert_20_demos_2025-06-15_17-22-04.pkl \
