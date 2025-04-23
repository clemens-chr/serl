from pathlib import Path
from typing import Any, Literal, Tuple, Dict

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
_XML_PATH = _HERE / "xmls" / "arena_with_orca.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
_CARTESIAN_BOUNDS = np.asarray([[0.1, -0.3, 0], [1, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.5, -0.15], [0.75, 0.15]])


class Orca1PickCubeGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.1, 1]),
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

        # Caching.
        self._panda_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self._panda_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        # self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        # self._pinch_site_id = self._model.site("pinch").id

        self._attachment_site_id = self._model.site("attachment_site").id
        self._block_z = self._model.geom("block").size[2]

        self.hand = OrcaHand('/home/clemens/serl_ws/src/dex-serl/franka_sim/franka_sim/envs/models/orcahand_v1')

        hand_dofs = len(self.hand.joint_ids)

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "panda/tcp_pos": spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                        "panda/tcp_vel": spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                        # 17 dof hand
                        "hand_pos": spaces.Box(
                            -np.inf, np.inf, shape=(1,), dtype=np.float32
                        ),
                        "block_pos": spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                    }
                ),
            }
        )

        if self.image_obs:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(
                        {
                            "tcp_pos": spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "tcp_vel": spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "hand_pos": spaces.Box(
                                -np.inf, np.inf, shape=(1,), dtype=np.float32
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

        self.action_space = gym.spaces.Box(
            low=np.full((3 + 1,), -1.0),
            high=np.full((3 + 1,), 1.0),
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

        # Reset arm to home position.
        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        mujoco.mj_forward(self._model, self._data)

        # Reset hand to home position.
        for joint in self.hand.joint_ids:
            if joint == 'thumb_abd':
                self.data.ctrl[self._model.actuator(joint).id] = 30
            else:   
                self._data.ctrl[self._model.actuator(joint).id] = self.hand.joint_roms[joint][0]
                
        # Sample a new block position.
        block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        mujoco.mj_forward(self._model, self._data)

        # Cache the initial block height.
        self._z_init = self._data.sensor("block_pos").data[2]
        self._z_success = self._z_init + 0.2

        obs = self._compute_observation()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        take a step in the environment.
        Params:
            action: np.ndarray

        Returns:
            observation: dict[str, np.ndarray],
            reward: float,
            done: bool,
            truncated: bool,
            info: dict[str, Any]
        """
        x, y, z = action[:3]
        hand_pose = action[3:]

        # Set the mocap position.
        pos = self._data.mocap_pos[0].copy()
        dpos = np.asarray([x, y, z]) * self._action_scale[0]
        npos = np.clip(pos + dpos, *_CARTESIAN_BOUNDS)
        self._data.mocap_pos[0] = npos

        # Apply the relative hand_pose action (17 DOFs)
        for i, joint in enumerate(self.hand.joint_ids):

            if joint == 'index_abd' or joint == 'middle_abd' or joint == 'ring_abd' or joint == 'pinky_abd' or joint == 'wrist':
                # Skip the index finger abduction joint
                continue
        
            if joint == 'thumb_abd':
                # For the thumb abduction joint, we need to set a specific value
                self._data.ctrl[self._model.actuator(joint).id] = -30
                continue

            # Get the current control value for the actuator
            current_value = self._data.ctrl[self._model.actuator(joint).id]
        
            # Get the relative change from the hand_pose action
            delta = hand_pose * self._action_scale[1]  # Scale the action appropriately
        
            # Calculate the target value as the current value plus the delta
            target_value = current_value + delta
        
            # Clip the target value to ensure it stays within the joint's range of motion
            target_value = np.clip(
                target_value,
                np.deg2rad(self.hand.joint_roms[joint][0]),  # Minimum range (in radians)
                np.deg2rad(self.hand.joint_roms[joint][1])   # Maximum range (in radians)
            )
        
            # Set the new control value for the actuator
            self._data.ctrl[self._model.actuator(joint).id] = target_value

        for _ in range(self._n_substeps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._attachment_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                joint=_PANDA_HOME,
                gravity_comp=True,
            )
            self._data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)

        obs = self._compute_observation()
        rew = self._compute_reward()
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

    # Helper methods.

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        tcp_pos = self._data.sensor("attachment_pos").data
        tcp_quat = self._data.sensor("attachment_quat").data
        tcp_vel = self._data.sensor("attachment_linvel").data

        obs["state"]["tcp_pos"] = tcp_pos.astype(np.float32)
        obs["state"]["tcp_vel"] = tcp_vel.astype(np.float32)
        
        # hand_pos = np.zeros(17)
        # for i, joint in enumerate(self.hand.joint_ids):
        #     hand_pos[i] = self._data.ctrl[self._model.actuator(joint).id]

        hand_pos = np.zeros(1)
        ref_joint = "index_mcp"
        hand_pos = self._data.ctrl[self._model.actuator(ref_joint).id]

        # Normalize the hand_pos between the max ROM and min ROM of the reference joint
        min_rom = np.deg2rad(self.hand.joint_roms[ref_joint][0])
        max_rom = np.deg2rad(self.hand.joint_roms[ref_joint][1])
        hand_pos = (hand_pos - min_rom) / (max_rom - min_rom)
        hand_pos = np.clip(hand_pos, 0.0, 1.0)


        obs["state"]["hand_pos"] = hand_pos.astype(np.float32)

        if self.image_obs:
            obs["images"] = {}
            obs["images"]["front"], obs["images"]["wrist"] = self.render()
        else:
            block_pos = self._data.sensor("block_pos").data.astype(np.float32)
            obs["state"]["block_pos"] = block_pos

        if self.render_mode == "human":
            self._viewer.render(self.render_mode)

        return obs

    def _compute_reward(self) -> float:
        if self.reward_type == "dense":
            block_pos = self._data.sensor("block_pos").data
            center_pos = self._data.sensor("hand_center_pos").data
            dist = np.linalg.norm(block_pos - center_pos)
            r_close = np.exp(-20 * dist)
            r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
            r_lift = np.clip(r_lift, 0.0, 1.0)
            rew = 0.3 * r_close + 0.7 * r_lift
            # print(f"dist: {dist}, r_close: {r_close}, r_lift: {r_lift}, rew: {rew}")
            return rew
        elif self.reward_type == "sparse":
            block_pos = self._data.sensor("block_pos").data
            lift = block_pos[2] - self._z_init
            return float(lift > 0.2)


if __name__ == "__main__":
    env = Orca1PickCubeGymEnv(render_mode="human")
    env.reset()
    import time
    for i in range(100):
        action = np.random.uniform(-1, 1, size=(4,))  
        action[0] = 0.09    
        action[1] = 0.0       
        action[2] = -0.1   
        action[3] = 0.04

        env.step(action)  
        time.sleep(0.1)
        env.render()
    env.close()
