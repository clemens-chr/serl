from gym.envs.registration import register
import numpy as np

register(
    id="FrankaEnv-Vision-v0",
    entry_point="franka_env.envs:FrankaEnv",
    max_episode_steps=100,
)

register(
    id="FrankaPegInsert-Vision-v0",
    entry_point="franka_env.envs.peg_env:FrankaPegInsert",
    max_episode_steps=100,
)

register(
    id="FrankaPCBInsert-Vision-v0",
    entry_point="franka_env.envs.pcb_env:FrankaPCBInsert",
    max_episode_steps=100,
)

register(
    id="FrankaCableRoute-Vision-v0",
    entry_point="franka_env.envs.cable_env:FrankaCableRoute",
    max_episode_steps=100,
)

register(
    id="FrankaBinRelocation-Vision-v0",
    entry_point="franka_env.envs.bin_relocation_env:FrankaBinRelocation",
    max_episode_steps=100,
)

register(
    id="FrankaCubeRelocation-Vision-v0",
    entry_point="franka_env.envs.cube_relocation_env:FrankaCubeRelocation",
    max_episode_steps=100,
)

register(
    id="OrcaCubePick-Vision-v0",
    entry_point="franka_env.envs.orca_cube_pick_env:OrcaCubePick",
    max_episode_steps=100,
)

register(
    id="OrcaCubePickBinary-Vision-v0",
    entry_point="franka_env.envs.orca_cube_pick_env:OrcaCubePickBinary",
    max_episode_steps=100,
)

register(
    id="OrcaRotate-Vision-v0",
    entry_point="franka_env.envs.orca_rotate.orca_rotate:OrcaRotate",
    max_episode_steps=100,
)