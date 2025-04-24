export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python async_drq_sim.py "$@" \
    --learner \
    --exp_name=gripper_demo_random_image \
    --seed 0 \
    --training_starts 1000 \
    --save_video \
    --max_steps 15300 \
    --checkpoint_period 5000 \
    --batch_size 256 \
    --critic_actor_ratio 4 \
    --encoder_type resnet-pretrained \
    --demo_path /home/clemens/serl/serl/examples/async_drq_sim/franka_lift_cube_image_20_trajs_noise_images.pkl
    
