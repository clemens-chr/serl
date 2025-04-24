import os
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from gym.wrappers.record_video import RecordVideo

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    JoystickIntervention,
    AVPIntervention,
    AVPInterventionPinch
)

from experiments.pick_cube_sim.wrapper import GripperPenaltyWrapper

from franka_env.envs.relative_env import RelativeFrame
from serl_launcher.networks.reward_classifier import load_classifier_func

from franka_sim.envs.panda_pick_gym_env import PandaPickCubeGymEnv


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["front", "wrist"]
    classifier_keys = ["front", "wrist"]
    random_steps = 0
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"
    classifier = False

    def get_environment(self, classifier=False):
        env = PandaPickCubeGymEnv(render_mode="human", image_obs=True, reward_type="sparse", time_limit=100.0, control_dt=0.1)

        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        
        if save_video:
            env = GripperPenaltyWrapper(env, gripper_penalty=0.0)
        
        return env