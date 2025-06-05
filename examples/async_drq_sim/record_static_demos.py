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

def generate_finger_action(t, noise_scale=1):
    """Generate finger actions based on time t (in seconds)"""
    base_action = np.zeros(17)
    
    # Skip these joints: index_abd(4), middle_abd(7), ring_abd(10), pinky_abd(13), wrist
    skip_indices = [4, 7, 10, 13]  # Indices for abduction joints
    thumb_abd_idx = 1  # Thumb abduction joint
    
    if t <= 2.0:  # Closing motion in first 2 seconds
        # Set thumb abduction to -30 degrees with some noise
        base_action[thumb_abd_idx] = np.deg2rad(0) + np.random.normal(0, noise_scale * 0.1)
        
        # Linearly close other joints from 0 to 0.8
        close_fraction = (t / 2.0) * 0.3
        
        # 10% chance to generate an opening movement during closing
        if np.random.random() < 0.1:
            # Reduce the closing fraction to create an opening movement
            close_fraction *= np.random.uniform(0.3, 0.7)
        
        for i in range(17):
            if i not in skip_indices and i != thumb_abd_idx:
                base_action[i] = close_fraction
                # Add significant noise during closing
                base_action[i] += np.random.normal(0, noise_scale * (1 - t/2.0))
                # 5% chance for each joint to make a larger opening movement
                if np.random.random() < 0.05:
                    base_action[i] *= np.random.uniform(0.2, 0.6)
    
    else:  # After 2 seconds, maintain position with larger noise
        # Keep thumb abduction at -30 degrees with noise
        base_action[thumb_abd_idx] = np.deg2rad(0) + np.random.normal(0, noise_scale * 0.15)
        
        # Maintain closed position for other joints with significant noise
        for i in range(17):
            if i not in skip_indices and i != thumb_abd_idx:
                # Base position of 0.8 with larger random variations
                base_action[i] = 0.8 + np.random.normal(0, noise_scale)
                # 15% chance for each joint to make a larger movement
                if np.random.random() < 0.15:
                    base_action[i] += np.random.uniform(-0.2, 0.2)
    
    # Clip to ensure actions stay in valid range
    return np.clip(base_action, -1.0, 1.0)

if __name__ == "__main__":
    # Create environment without image observations
    env = gym.make("OrcaGraspStatic-v0", render_mode="human")

    # Initialize variables
    success_needed = int(input("How many demonstrations do you want to record? "))
    success_count = 0
    transitions = []
    pbar = tqdm(total=success_needed)

    # Setup save path
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"demo_data/orca_static_grasp_{success_needed}_demos_{uuid}.pkl"
    file_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(file_dir, file_name)

    print("Starting demonstration recording...")
    print("Each episode will run for 5 seconds.")
    print("Fingers will close in first 2 seconds, then maintain position.")
    print("Reward will be 1 after 4 seconds.")

    first_block_pos = None

    while success_count < success_needed:
        # Reset environment and get initial observation
        obs, _ = env.reset()
        episode_transitions = []
        episode_start_time = time.time()
        
        while True:
            current_time = time.time() - episode_start_time
            
            # Generate action based on time
            actions = generate_finger_action(current_time)

            # Take step in environment
            next_obs, rew, done, truncated, info = env.step(actions)

            # Override reward: 1 after 4 seconds
            if current_time >= 4.0:
                rew = 1.0

            if first_block_pos is None:
                first_block_pos = obs["state"]["block_pos"].copy()
            
            # Add random noise to block position
            noise = np.random.normal(0, 0.01, size=3)  # Small Gaussian noise
            first_block_pos += noise

            print(first_block_pos)

            # Store transition
            transition = {
                "observations": {
                    "state": {
                        "hand_pos": obs["state"]["hand_pos"].copy(),
                        "block_pos": first_block_pos.copy()
                    },
                    "front": obs["images"]["front"].copy(),
                    "wrist": obs["images"]["wrist"].copy(),

                },
                "actions": actions.copy(),
                "next_observations": {
                    "state": {
                        "hand_pos": next_obs["state"]["hand_pos"].copy(),
                        "block_pos": first_block_pos.copy()
                    },  
                    "front": obs["images"]["front"].copy(),
                    "wrist": obs["images"]["wrist"].copy(),
                },
                "rewards": float(rew),
                "masks": 1.0 - done,
                "dones": done
            }
            episode_transitions.append(transition)
            
            obs = next_obs

            if done:
                success_count += 1
                transitions.extend(episode_transitions)
                pbar.update(1)
                print(f"\nRecorded demonstration {success_count}/{success_needed}")
                break

    # Save demonstrations
    with open(file_path, "wb") as f:
        pkl.dump(transitions, f)
        print(f"\nSaved {success_count} demonstrations to {file_path}")

    env.close()
    pbar.close() 