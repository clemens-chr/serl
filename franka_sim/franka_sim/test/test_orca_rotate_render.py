import time

import gym
import mujoco
import mujoco.viewer
import numpy as np

import franka_sim

from franka_env.envs.wrappers import AVPIntervention

env = gym.make("OrcaRotate-v0", render_mode="human", image_obs=True)
env = AVPIntervention(env, model_path="/home/clemens/serl/serl/franka_sim/franka_sim/envs/models/orcahand_v1", avp_ip = '10.93.181.166')

time.sleep(4)
action_spec = env.action_space   

def sample():
    a = np.random.uniform(action_spec.low, action_spec.high, action_spec.shape)
    return a.astype(action_spec.dtype)


obs, info = env.reset()
frames = []

for i in range(1000):
    
    time_start = time.time()
    a = sample()
    obs, rew, done, truncated, info = env.step(a)

    images = obs["images"]
    frames.append(np.concatenate((images["front"], images["wrist"]), axis=0))

    if done:
        obs, info = env.reset()
    time_end = time.time()


import imageio

imageio.mimsave("orca_1_lift_cube_render_test.mp4", frames, fps=20)
