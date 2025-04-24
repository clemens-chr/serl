#!/bin/bash

# Use the default values if the env variables are not set
EXAMPLE_DIR=${EXAMPLE_DIR:-"/home/clemens/serl/serl/examples/async_drq_sim"}
CONDA_ENV=${CONDA_ENV:-"serl"}

cd $EXAMPLE_DIR
echo "Running from $(pwd)"

# Define the list of actor and learner commands
COMMAND_LIST=(
    "bash run_actor.sh --exp_name=gripper_demo_random_image && bash run_learner.sh --exp_name=gripper_demo_random_image --demo_path=/home/clemens/serl/serl/examples/async_drq_sim/franka_lift_cube_image_20_trajs_noise_images.pkl"
    "bash run_actor.sh --exp_name=gripper_demo_gray_image && bash run_learner.sh --exp_name=gripper_demo_gray_image --demo_path=/home/clemens/serl/serl/examples/async_drq_sim/franka_lift_cube_image_20_trajs_gray_images.pkl"
)

run_batch() {
    local actor_command="$1"
    local learner_command="$2"
    local session_name="serl_$(date +%s)"

    tmux new-session -d -s "$session_name"

    # Split the window
    tmux split-window -v -t "$session_name"

    # Run commands
    tmux send-keys -t "$session_name:0.0" "conda activate $CONDA_ENV && $actor_command" C-m
    tmux send-keys -t "$session_name:0.1" "conda activate $CONDA_ENV && $learner_command" C-m

    echo "Attach to tmux session '$session_name' to view logs. Detach with Ctrl+B then D."
    tmux attach -t "$session_name"

    # Once detached, wait, then kill
    echo "Sleeping 1.5 hours before killing session..."
    sleep 5400
    tmux kill-session -t "$session_name"
}

for commands in "${COMMAND_LIST[@]}"; do
    IFS="&&" read -r actor_command learner_command <<< "$commands"
    echo "Starting batch with actor: $actor_command and learner: $learner_command"
    run_batch "$actor_command" "$learner_command"
done

echo "All batches completed."
