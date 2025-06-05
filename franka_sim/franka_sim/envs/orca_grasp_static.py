from pathlib import Path
from typing import Any, Literal, Tuple, Dict
from scipy.spatial.transform import Rotation as R

import gym
import mujoco
import numpy as np
from gym import spaces

from orca_core import OrcaHand

try:
    import mujoco_py
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
else:
    MUJOCO_PY_IMPORT_ERROR = None

from franka_sim.controllers import opspace
from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena_with_orca_static.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
_CARTESIAN_BOUNDS = np.asarray([[0.1, -0.3, 0], [1, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.03, -0.2, 0.18], [0.03, -0.2, 0.18]])  # Fixed point for cube spawn


class OrcaGraspStaticGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.01, 0.2]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        save_video=False,
        reward_type: str = "dense",
    ):
        self._action_scale = action_scale

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.camera_id = (0, 2)
        self.image_obs = image_obs
        self.reward_type = reward_type


        self._attachment_site_id = self._model.site("attachment_site").id
        self._block_z = self._model.geom("block").size[2]

        self.hand = OrcaHand('/home/clemens/serl_ws/src/dex-serl/franka_sim/franka_sim/envs/models/orcahand_v1')

        hand_dofs = len(self.hand.joint_ids)

        if image_obs:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(
                        {
                            # 17 dof hand
                            "hand_pos": spaces.Box(
                                -np.inf, np.inf, shape=(hand_dofs,), dtype=np.float32
                            ),
                            "block_pos": spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                        }
                    ),
                    "images": gym.spaces.Dict(
                        {
                            "front": gym.spaces.Box(
                                low=0,
                                high=255,
                                shape=(render_spec.height, render_spec.width, 3),
                                dtype=np.uint8,
                            ),
                            "wrist": gym.spaces.Box(
                                low=0,
                                high=255,
                                shape=(render_spec.height, render_spec.width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                }
            )
        else:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(
                        {
                            "hand_pos": spaces.Box(
                                -np.inf, np.inf, shape=(hand_dofs,), dtype=np.float32), 
                            "block_pos": spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32),
                        }
                    ),
                }
            )

        self.action_space = gym.spaces.Box(
            low=np.full((hand_dofs,), -1.0),
            high=np.full((hand_dofs,), 1.0),
            dtype=np.float32,
        )

        if save_video:
            print("Saving videos!")
        self.save_video = save_video
        self.recording_frames = []

        print(self.action_space)
        print(self.observation_space)

        # NOTE: gymnasium is used here since MujocoRenderer is not available in gym. It
        # is possible to add a similar viewer feature with gym, but that can be a future TODO
        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self._viewer = MujocoRenderer(
            self.model,
            self.data,
        )
        self._viewer.render(self.render_mode)

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        mujoco.mj_resetData(self._model, self._data)


        # Reset hand to home position.
        for joint in self.hand.joint_ids:
            self._data.ctrl[self._model.actuator(joint).id] = self.hand.joint_roms[joint][0]
            
        # Set block position from sampling bounds
        self.start_point = _SAMPLING_BOUNDS[0]  # Use the lower bound (same as upper bound in this case)
        self._data.jnt("block").qpos[:3] = self.start_point
        
        mujoco.mj_forward(self._model, self._data)

        obs = self._compute_observation()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:

        hand_joint_actions = action
            
        self._block_qposadr = self._model.jnt_qposadr[self._model.joint("block").id]

        # Apply random forces to the cube that increase with time
        current_time = self._data.time
        if current_time <= 5.0:

            # Scale force magnitude from 0 to 2 Newtons over 5 seconds
            force_magnitude = (current_time / 5.0) * 2.0 + 0.5
            # Generate random force direction
            random_force = np.random.uniform(-1, 1, size=3)
            random_force = random_force / np.linalg.norm(random_force) * force_magnitude
            # Apply the force to the cube
            self._data.xfrc_applied[self._model.body("block").id] = np.concatenate([random_force, np.zeros(3)])

        current_qpos_rot = self._data.qpos[self._block_qposadr + 3 : self._block_qposadr + 7].copy()
        r_current = R.from_quat([current_qpos_rot[1], current_qpos_rot[2], current_qpos_rot[3], current_qpos_rot[0]])

        delta_rot_degree_x = 0
        delta_rot_degree_y = 0
        delta_rot_degree_z = 0
        r_delta = R.from_euler('xyz', [delta_rot_degree_x, delta_rot_degree_y, delta_rot_degree_z], degrees=True)

        r_target = r_delta * r_current 

        target_quat_xyzw = r_target.as_quat()
        target_qpos_rot = np.array([target_quat_xyzw[3], target_quat_xyzw[0], target_quat_xyzw[1], target_quat_xyzw[2]])
        
        # self._data.qpos[self._block_qposadr + 3 : self._block_qposadr + 7] = target_qpos_rot

        for i, joint in enumerate(self.hand.joint_ids):
        
            current_value = self._data.ctrl[self._model.actuator(joint).id]
    
            # Get the relative change from the hand_pose action
            delta = hand_joint_actions[i] * self._action_scale[1]  # Scale the action appropriately
        
            # Calculate the target value as the current value plus the delta
            target_value = current_value + delta
        
            # Clip the target value to ensure it stays within the joint's range of motion
            target_value = np.clip(
                target_value,
                np.deg2rad(self.hand.joint_roms[joint][0]),  # Minimum range (in radians)
                np.deg2rad(self.hand.joint_roms[joint][1])   # Maximum range (in radians)
            )

            self._data.ctrl[self._model.actuator(joint).id] = target_value

        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)
                        

        obs = self._compute_observation()
        rew = self._compute_reward()
        if rew: 
            print("Reward: ", rew)
        terminated = self.time_limit_exceeded()
    
        return obs, rew, terminated, False, {}

    def render(self):
        rendered_frames = []
        if self._viewer is None:
            from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
            self._viewer = MujocoRenderer(self.model, self.data)
            print("Initialized during render OrcaPickCubeGymEnv with renderer:", self._viewer)

        for cam_id in self.camera_id:
            rendered_frames.append(
                self._viewer.render(render_mode="rgb_array", camera_id=cam_id)
            )
        return rendered_frames

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}
        
        # Get hand positions
        hand_pos = np.zeros(17)
        for i, joint in enumerate(self.hand.joint_ids):
            pos = self._data.ctrl[self._model.actuator(joint).id]
            min_rom = np.deg2rad(self.hand.joint_roms[joint][0])
            max_rom = np.deg2rad(self.hand.joint_roms[joint][1])
            # Scale the position to be between 0 and 1
            hand_pos[i] = (pos - min_rom) / (max_rom - min_rom)
            # Clip to ensure it's within the range
            hand_pos[i] = np.clip(hand_pos[i], 0, 1)

        obs["state"]["hand_pos"] = hand_pos.astype(np.float32)
        
        # Always get block position
        block_pos = self._data.sensor("block_pos").data.astype(np.float32)
        obs["state"]["block_pos"] = block_pos

        if self.image_obs:
            obs["images"] = {}
            obs["images"]["front"], obs["images"]["wrist"] = self.render()

        if self.render_mode == "human":
            self._viewer.render(self.render_mode)

        return obs

    def _compute_reward(self) -> float:
        if self.reward_type == "dense":
            # Get current block position
            block_pos = self._data.sensor("block_pos").data
            
            x_min, y_min, z_min = _SAMPLING_BOUNDS[0]
            x_max, y_max, z_max = _SAMPLING_BOUNDS[1]
            
            in_region = (
                x_min - 0.03 <= block_pos[0] <= x_max + 0.03 and
                y_min - 0.03 <= block_pos[1] <= y_max + 0.03 
            )
            
            if self._data.time >= 4.0:
                return float(in_region)
            return 0.0
            
        elif self.reward_type == "sparse":
            # Sparse reward for achieving a rotation threshold
            block_quat = self._data.sensor("block_quat").data
            target_quat = np.array([1, 0, 0, 0])
            dot_product = np.dot(block_quat, target_quat)
            angle_diff = 2 * np.arccos(np.clip(dot_product, -1.0, 1.0))
            return float(angle_diff < np.deg2rad(10))  # Reward if within 10 degrees of target


if __name__ == "__main__":
    env = OrcaGraspStaticGymEnv (render_mode="human")
    env.reset()
    import time
    for i in range(500):
        action = np.zeros(17)  
        action[0] = 0.09    
        action[1] = 0.0       
        action[2] = -0.1   

        env.step(action)  
        env.render()
    env.close()
