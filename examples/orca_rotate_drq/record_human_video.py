import gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
import os

import franka_env

from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    GripperCloseEnv,
    AVPIntervention,
    Quat2EulerWrapper,
    SGMWrapper,
)

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

from serl_launcher.wrappers.chunking import ChunkingWrapper

if __name__ == "__main__":
    env = gym.make("OrcaRotate-Vision-v0")
    env = AVPIntervention(env, avp_ip = "192.168.1.10", debug=False, gripper_only=True, record_raw_data=True)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    env = SGMWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    obs, _ = env.reset()

    transitions = []
    success_count = 0
    success_needed = 1
    total_count = 0
    pbar = tqdm(total=success_needed)

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"orca_cube_pick_{success_needed}_demos_{uuid}.pkl"
    file_dir = os.path.dirname(os.path.realpath(__file__))  # same dir as this script
    file_path = os.path.join(file_dir, file_name)

    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    if os.path.exists(file_path):
        raise FileExistsError(f"{file_name} already exists in {file_dir}")
    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"No permission to write to {file_dir}")

    while success_count < success_needed:
        actions = env.action_space.sample()
        next_obs, rew, done, truncated, info = env.step(action=actions)
        
        # Only record transition if there was an intervention
        if "intervene_action" in info:
            #print(f'intervene_action: {info["intervene_action"]}')
            actions = info["intervene_action"]
            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                    mask_image=info["mask_image"],
                    raw_data=info["raw_data"],
                )
            )
            transitions.append(transition)

        obs = next_obs

        if done:
            if rew == 1:
                success_count += 1
            total_count += 1
            print(
                f"{rew}\tGot {success_count} successes of {total_count} trials. {success_needed} successes needed."
            )
            pbar.update(rew)
            obs, _ = env.reset()

    with open(file_path, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_path}")

    env.close()
    pbar.close()
