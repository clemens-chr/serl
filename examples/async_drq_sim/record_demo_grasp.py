#!/usr/bin/env python3
import gym
import numpy as np
import pickle as pkl
import datetime
import os
from tqdm import tqdm
import copy
import time
import sys

import franka_sim
from franka_env.envs.wrappers import AVPInterventionPinch
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

def wait_for_enter():
    input("Press Enter to start recording the next demonstration...")

if __name__ == "__main__":
    # Create environment without image observations
    env = gym.make("OrcaGraspStatic-v0", image_obs=False, render_mode="human")
    env = AVPInterventionPinch(env)
    env = SERLObsWrapper(env)

    # Initialize variables
    success_needed = int(input("How many demonstrations do you want to record? "))
    success_count = 0
    total_count = 0
    transitions = []
    pbar = tqdm(total=success_needed)

    # Setup save path
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"orca_static_grasp_{success_needed}_demos_{uuid}.pkl"
    file_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(file_dir, file_name)

    print("Starting demonstration recording...")
    print("Each episode will run for 5 seconds.")
    print("The reward will be 1 if you keep the cube in the target region after 4 seconds.")

    while success_count < success_needed:
        wait_for_enter()
        obs, _ = env.reset()
        episode_transitions = []
        episode_start_time = time.time()
        
        while True:
            # Get action from AVP
            actions = np.zeros(17)  # Default zero action
            if "intervene_action" in info:
                actions = info["intervene_action"]

            # Take step in environment
            next_obs, rew, done, truncated, info = env.step(actions)

            # Store transition
            transition = {
                "observations": {
                    "state": {
                        "hand_pos": obs["state"]["hand_pos"].copy(),
                        "block_pos": obs["state"]["block_pos"].copy()
                    }
                },
                "actions": actions.copy(),
                "next_observations": {
                    "state": {
                        "hand_pos": next_obs["state"]["hand_pos"].copy(),
                        "block_pos": next_obs["state"]["block_pos"].copy()
                    }
                },
                "rewards": float(rew),
                "masks": 1.0 - done,
                "dones": done
            }
            episode_transitions.append(transition)
            
            obs = next_obs

            if done:
                if rew > 0:
                    success_count += 1
                    transitions.extend(episode_transitions)
                    pbar.update(1)
                    print(f"\nSuccess! Recorded demonstration {success_count}/{success_needed}")
                else:
                    print("\nFailed demonstration, try again")
                total_count += 1
                break

    # Save demonstrations
    with open(file_path, "wb") as f:
        pkl.dump(transitions, f)
        print(f"\nSaved {success_count} successful demonstrations to {file_path}")

    env.close()
    pbar.close() 