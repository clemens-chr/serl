import numpy as np
import time
import requests
import copy
import cv2
import queue
import gym
import threading
import multiprocessing as mp
from scipy.spatial.transform import Rotation
from orca_core import OrcaHand, MockOrcaHand

from franka_env.envs.franka_env import FrankaEnv
from franka_env.utils.rotations import euler_2_quat
from franka_env.envs.orca_rotate.config import OrcaRotateEnvConfig
from franka_env.envs.rewards.cube_reward import CubeReward
from franka_env.camera.cube_position import CubePosition

class OrcaRotate(FrankaEnv):
    
    
    def __init__(self, **kwargs):
        
        self.hand = MockOrcaHand("/home/ccc/orca_ws/src/orca_configs/orcahand_v1_right_clemens_stanford")
        ok, msg = self.hand.connect()
        if not ok:
            raise Exception(f'Failed to connect to hand: {msg}')
        
        super().__init__(**kwargs, config=OrcaRotateEnvConfig)
        

        self.reset_hand_pose = self.config.RESET_HAND_POSE
        
        self.min_hand_pose = self.config.MIN_HAND_POSE
        self.max_hand_pose = self.config.MAX_HAND_POSE
        self.active_dof = self.config.ACTIVE_DOF
        self.action_scale = self.config.ACTION_SCALE_PER_DOF
        print(f'action_scale: {self.action_scale}')
        self.action_space_num_hand = self.config.ACTION_SPACE_NUM_HAND
        
        self.action_space_num_tcp_franka = self.config.ACTION_SPACE_NUM_TCP_FRANKA
        
        self.action_space = gym.spaces.Box(
            np.ones((self.action_space_num_hand + self.action_space_num_tcp_franka,), dtype=np.float32) * -1,
            np.ones((self.action_space_num_hand + self.action_space_num_tcp_franka,), dtype=np.float32),
        )
        
        self.observation_space["state"] = gym.spaces.Dict(
            {
                "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                "hand_pose": gym.spaces.Box(-1, 1, shape=(self.action_space_num_hand + self.action_space_num_tcp_franka,)),
                "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                "cube_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
            }
        )
        
        self.observation_space["images"] = gym.spaces.Dict(
            {
                "front": gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
                "side": gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
            }
        )
        
        self.hand = MockOrcaHand("/home/ccc/orca_ws/src/orca_configs/orcahand_v1_right_clemens_stanford")
        ok, msg = self.hand.connect()
        if not ok:
            raise Exception(f'Failed to connect to hand: {msg}')
        
        self.hand.set_joint_pos(self.reset_hand_pose, num_steps=25, step_size=0.001)    
        self.current_hand_angles = self.hand.get_joint_pos(as_list=False)
        
        self.cube_pose = np.zeros(7)
        self.cube_position_subscriber = CubePosition()
        
        self.rewarder = CubeReward()
        
            
        # Hand control thread variables
        self.hand_action_queue = queue.Queue(maxsize=3)
        self.hand_control_running = True
        self.hand_control_hertz = 30.0  # Variable hertz rate
        self.hand_control_thread = threading.Thread(target=self._hand_control_loop, daemon=True)
        self.hand_control_thread.start()
    
        # Hz monitoring with shared memory
        self.hz_monitor_running = True
        self.env_hz_data = mp.Value('d', 0.0)  # Shared double for env Hz
        self.hand_hz_data = mp.Value('d', 0.0)  # Shared double for hand Hz
        self.hz_monitor_thread = threading.Thread(target=self._hz_monitor_loop, daemon=True)
        self.hz_monitor_thread.start()
    
        self.max_episode_length = 200

        # Environment Hz monitoring
        self.last_step_time = time.time()
        self.step_count = 0
        
        """
        the inner safety box is used to prevent the gripper from hitting the two walls of the bins in the center.
        it is particularly useful when there is things you want to avoid running into within the bounding box.
        it uses the intersect_line_bbox function to detect whether the gripper is going to hit the wall
        and clips actions that will lead to collision.
        """
        self.inner_safety_box = gym.spaces.Box(
            self._TARGET_POSE[:3] - np.array([0.07, 0.03, 0.01]),
            self._TARGET_POSE[:3] + np.array([0.07, 0.03, 0.04]),
            dtype=np.float64,
        )
        
    def _hand_control_loop(self):
        """Background hand control thread running at variable hertz"""
        target_interval = 1.0 / self.hand_control_hertz
        last_time = time.time()
        num_steps = int(self.hand_control_hertz / self.hz)
        print(f'hand_control_hertz: {self.hand_control_hertz}')
        print(f'hz: {self.hz}')
        print(f'num_steps: {num_steps}')
        num_steps = 2
        
        while self.hand_control_running:
            try:
                # Get latest hand action from queue (non-blocking)
                t1 = time.time()
                try:
                    hand_action = self.hand_action_queue.get_nowait()
                    action_received = True
                except queue.Empty:
                    action_received = False
                    hand_action = None
                t1_duration = time.time() - t1
                                
                if action_received:
                    # Calculate the target hand angles
                    curr_pos = self.current_hand_angles.copy()
                    next_pos = curr_pos.copy()
              
                    for i, dof in enumerate(self.active_dof):
                        next_pos[dof] = curr_pos[dof] + hand_action[i] * self.action_scale[dof]
                        next_pos[dof] = np.clip(next_pos[dof], self.min_hand_pose[dof], self.max_hand_pose[dof])
                    
                    
                    
                    self.hand.set_joint_pos(next_pos, num_steps=num_steps, step_size=0.000)
                    
                    self.current_hand_angles = next_pos
                        
                    
            except Exception as e:
                raise e
            # Dynamic timing to maintain target hertz
            elapsed = time.time() - t1
            sleep_time = max(0, target_interval - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            # Print actual frequency for debugging
            current_time = time.time()
            actual_hertz = 1.0 / (current_time - last_time)
            last_time = current_time
            
            # Update shared memory with hand Hz data
            self.hand_hz_data.value = actual_hertz

    def _hz_monitor_loop(self):
        """Background Hz monitoring thread that sends data to endpoints"""
        while self.hz_monitor_running:
            try:
                # Send environment Hz data
                env_hz = self.env_hz_data.value
                if env_hz > 0:
                    requests.post(self.url + "env_hz", json={"env_hz": env_hz})
                
                # Send hand Hz data  
                hand_hz = self.hand_hz_data.value
                if hand_hz > 0:
                    requests.post(self.url + "hand_hz", json={"hand_hz": hand_hz})
                    
            except Exception as e:
                pass  # Silently fail if endpoints are not available
            
            time.sleep(0.1)  # Send updates every 100ms


    def clip_safety_box(self, pose):
        pose = super().clip_safety_box(pose)
        return pose


    def reset(self, joint_reset=False, **kwargs):
        self.current_hand_angles = self.hand.get_joint_pos(as_list=False)      
        time.sleep(0.1)
        self.go_to_rest()
        print('resetting cube position subscriber')
        time.sleep(0.1)
        self.cube_position_subscriber.reset_reference()

        self.goal_quat = None
        print('getting new reward quat')
        time.sleep(1)

        while self.goal_quat is None:
            pose = self.cube_position_subscriber.get_cube_pose()
            if pose is not None:
                initial_quat = pose[3:]
                self.cube_position_subscriber.get_cube_pose()  # one extra for axis calc
                self.goal_quat = self.rewarder.compute_goal_quat(initial_quat)
                print(f"Goal quat: {self.goal_quat}")
            time.sleep(0.1)
            
        
        
        return super().reset(joint_reset, **kwargs)
    
    def compute_reward(self, obs) -> float:
        reward = self.rewarder.is_cube_rotated(self.goal_quat, obs['state']['cube_pose'][3:])
        return reward
    
    def get_hand_angles(self):
        return self.hand.get_joint_pos()
    
    def get_cube_pose(self):
        return self.cube_position_subscriber.get_cube_pose()
    
    def set_hand_angles(self, angles):
        self.hand.set_joint_pos(angles)
        
    def clip_hand_angles(self, angles: dict) -> dict:
        for joint_id, angle in angles.items():
            angles[joint_id] = np.clip(angle, self.hand_open_pose[joint_id], self.hand_close_pose[joint_id])
        return angles

    def step(self, action: np.ndarray) -> tuple:   
        start_time = time.time()
        
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        franka_action = action[:self.action_space_num_tcp_franka]
        hand_action = action[self.action_space_num_tcp_franka:]
        
        if self.action_space_num_tcp_franka > 0:
            self.nextpos = self.currpos.copy()
            self.nextpos[:3] = self.nextpos[:3] + franka_action[:3] * self.action_scale[0]
            
            self.nextpos[3:] = (
                Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
                * Rotation.from_quat(self.currpos[3:])
            ).as_quat()
            
            self._send_pos_command(self.clip_safety_box(self.nextpos))

        try:
            self.hand_action_queue.put_nowait(hand_action)
        except queue.Full:
            pass

        self._update_currpos()

        ob = self._get_obs()
        reward = self.compute_reward(ob)
        print(f'reward: {reward}')
        self.curr_path_length += 1
        
        done = self.curr_path_length >= self.max_episode_length or reward == 1
        
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))
        
        # Calculate and send environment Hz
        current_time = time.time()
        self.step_count += 1
        if self.step_count > 1:  # Skip first step for accurate Hz calculation
            actual_hz = 1.0 / (current_time - self.last_step_time)
            # Update shared memory with environment Hz data
            self.env_hz_data.value = actual_hz
        self.last_step_time = current_time
        
        return ob, reward, done, False, {}
        
    def go_to_rest(self, joint_reset=False):
        """
        Move to the rest position defined in base class.
        Add a small z offset before going to rest to avoid collision with object.
        """
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.2)
        self._update_currpos()

        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] += 0.03
        self.interpolate_move(reset_pose, timeout=1)
        self.hand.set_joint_pos(self.reset_hand_pose, num_steps=25, step_size=0.001)
      
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

        self.current_hand_angles = self.hand.get_joint_pos(as_list=False)

    def _get_obs(self) -> dict:
        images = self.get_im()
        cube_pose = self.get_cube_pose()
        if cube_pose is not None:
            self.cube_pose = cube_pose
        else:
            # red warning
            print('\033[91m' + 'cube pose is None' + '\033[0m')
            
        curr_hand_angles_list = [self.current_hand_angles[dof] for dof in self.active_dof]
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "hand_pose": curr_hand_angles_list,
            "cube_pose": self.cube_pose,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))
    
    def close(self):
        """Clean up resources"""
        self.hand_control_running = False
        self.hz_monitor_running = False
        if hasattr(self, 'hand_control_thread'):
            self.hand_control_thread.join(timeout=1.0)
        if hasattr(self, 'hz_monitor_thread'):
            self.hz_monitor_thread.join(timeout=1.0)
        super().close()
