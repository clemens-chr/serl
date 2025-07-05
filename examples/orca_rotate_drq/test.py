import gym
from tqdm import tqdm
import numpy as np
import cv2  

import franka_env

from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    GripperCloseEnv,
    AVPIntervention,
    Quat2EulerWrapper,
)
import time
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

if __name__ == "__main__":
    env = gym.make("OrcaRotate-Vision-v0")
    # env = AVPIntervention(env, avp_ip = "192.168.1.10", debug=False, gripper_only=True)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    obs, _ = env.reset()

    success_count = 0
    total_count = 0
    num_episodes = 10  # Number of episodes to run

    print(f"Running {num_episodes} episodes with random actions...")


    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        obs, _ = env.reset()
        done = False
        step_count = 0
        start_time = time.time()

        
        # Frequency monitoring
        last_step_time = time.time()
        
        
        while not done:
            # Sample random action from action space
            actions = env.action_space.sample()
   
            next_obs, rew, done, truncated, info = env.step(action=actions)
            # Calculate and print frequency
            current_time = time.time()
            step_freq = 1.0 / (current_time - last_step_time)
            last_step_time = current_time
            
            step_count += 1
            
            # Print intervention info if it occurred
            if "intervene_action" in info:
                print(f"  Step {step_count}: Intervention occurred")
            
            obs = next_obs

        if rew == 1:
            success_count += 1
        total_count += 1
        
        
        
        print(f"  Episode {episode + 1} finished: reward={rew}, steps={step_count}")
        print(f"  Success rate: {success_count}/{total_count}")

    print(f"\nFinal results:")
    print(f"Total episodes: {total_count}")
    print(f"Successful episodes: {success_count}")
    print(f"Success rate: {success_count/total_count:.2%}")

    env.close()
