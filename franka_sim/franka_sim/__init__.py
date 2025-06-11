from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
]

from gym.envs.registration import register

register(
    id="PandaPickCube-v0",
    entry_point="franka_sim.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
)
register(
    id="PandaPickCubeVision-v0",
    entry_point="franka_sim.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)

register(
    id="OrcaPickCubeVision-v0",
    entry_point="franka_sim.envs:OrcaPickCubeGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)

register(
    id="Orca1PickCubeVision-v0",
    entry_point="franka_sim.envs:Orca1PickCubeGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)

register(
    id="OrcaRotate-v0",
    entry_point="franka_sim.envs:OrcaRotateGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)

register(
    id="OrcaGraspStatic-v0",
    entry_point="franka_sim.envs:OrcaGraspStaticGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)

