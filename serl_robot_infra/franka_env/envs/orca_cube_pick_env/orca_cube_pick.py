import numpy as np
import time
import requests
import copy
import cv2
import queue
import gym
import Rotation
from orca_core import OrcaHand

from franka_env.envs.franka_env import FrankaEnv
from franka_env.utils.rotations import euler_2_quat
from franka_env.envs.orca_cube_pick_env.config import OrcaCubePickEnvConfig
from serl_robot_infra.franka_env.envs.rewards.cube_reward import is_cube_lifted


class OrcaCubePick(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=OrcaCubePickEnvConfig)
        
        self.action_space = gym.spaces.Box(
            np.ones((23,), dtype=np.float32) * -1,
            np.ones((23,), dtype=np.float32),
        )
        
        self.observation_space["state"] = gym.spaces.Dict(
            {
                "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                "hand_pose": gym.spaces.Box(-1, 1, shape=(17,)),
                "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
            }
        )
        
        self.observation_space["images"] = gym.spaces.Dict(
            {
                "wrist_1": gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
                "front": gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
            }
        )
        
        self.hand = OrcaHand()
        self.hand.connect()
        
        self.current_hand_angles = np.zeros((17,))
        
        """
        the inner safety box is used to prevent the gripper from hitting the two walls of the bins in the center.
        it is particularly useful when there is things you want to avoid running into within the bounding box.
        it uses the intersect_line_bbox function to detect whether the gripper is going to hit the wall
        and clips actions that will lead to collision.
        """
        self.inner_safety_box = gym.spaces.Box(
            self._TARGET_POSE[:3] - np.array([0.07, 0.03, 0.001]),
            self._TARGET_POSE[:3] + np.array([0.07, 0.03, 0.04]),
            dtype=np.float64,
        )

    def intersect_line_bbox(self, p1, p2, bbox_min, bbox_max):
        # Define the parameterized line segment
        # P(t) = p1 + t(p2 - p1)
        tmin = 0
        tmax = 1

        for i in range(3):
            if p1[i] < bbox_min[i] and p2[i] < bbox_min[i]:
                return None
            if p1[i] > bbox_max[i] and p2[i] > bbox_max[i]:
                return None

            # For each axis (x, y, z), compute t values at the intersection points
            if abs(p2[i] - p1[i]) > 1e-10:  # To prevent division by zero
                t1 = (bbox_min[i] - p1[i]) / (p2[i] - p1[i])
                t2 = (bbox_max[i] - p1[i]) / (p2[i] - p1[i])

                # Ensure t1 is smaller than t2
                if t1 > t2:
                    t1, t2 = t2, t1

                tmin = max(tmin, t1)
                tmax = min(tmax, t2)

                if tmin > tmax:
                    return None

        # Compute the intersection point using the t value
        intersection = p1 + tmin * (p2 - p1)

        return intersection

    def clip_safety_box(self, pose):
        pose = super().clip_safety_box(pose)
        # Clip xyz to inner box
        if self.inner_safety_box.contains(pose[:3]):
            print(f'Command: {pose[:3]}')
            pose[:3] = self.intersect_line_bbox(
                self.currpos[:3],
                pose[:3],
                self.inner_safety_box.low,
                self.inner_safety_box.high,
            )
            print(f'Clipped: {pose[:3]}')
        return pose


    def reset(self, joint_reset=False, **kwargs):
        return super().reset(joint_reset, **kwargs)
    
    def compute_reward(self, obs) -> float:
        sgm = self.get_sgm()
        reward = is_cube_lifted(sgm)
        return reward
    
    def get_hand_angles(self):
        return self.hand.get_joint_positions()
    
    def set_hand_angles(self, angles):
        self.hand.set_joint_positions(angles)
        
    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]
        hand_action = action[7:23]
        
        
        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

        # GET ORIENTATION FROM ACTION
        self.nextpos[3:] = (
            Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
            * Rotation.from_quat(self.currpos[3:])
        ).as_quat()
        
        if not np.array_equal(self.clip_safety_box(self.nextpos), self.nextpos):
            print(f'CLIPPING OCCURED: {self.nextpos}')

        target_hand_angles = self.current_hand_angles + hand_action*5
        self.hand.set_joint_positions(target_hand_angles)

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length or reward == 1
        return ob, reward, done, False, {}
        
    def go_to_rest(self, joint_reset=False):
        """
        Move to the rest position defined in base class.
        Add a small z offset before going to rest to avoid collision with object.
        """
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.5)

        # Move up to clear the slot
        self._update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] += 0.05
        self.interpolate_move(reset_pose, timeout=1)
        
        self.hand.set_neutral_position()
        time.sleep(1)
      
        # execute the go_to_rest method from the parent class
        super().go_to_rest(joint_reset)
        
    def _update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        ps = requests.post(self.url + "getstate").json()
        self.currpos[:] = np.array(ps["pose"])
        self.currpose_euler[:] = ps['pose_euler']
        self.currvel[:] = np.array(ps["vel"])

        self.currforce[:] = np.array(ps["force"])
        self.currtorque[:] = np.array(ps["torque"])
        self.currjacobian[:] = np.reshape(np.array(ps["jacobian"]), (6, 7))

        self.q[:] = np.array(ps["q"])
        self.dq[:] = np.array(ps["dq"])

        self.current_hand_angles = self.hand.get_joint_positions()

    def _get_obs(self) -> dict:
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "hand_pose": self.current_hand_angles,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))
