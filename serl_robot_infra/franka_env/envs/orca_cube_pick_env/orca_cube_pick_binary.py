import numpy as np
import time
import requests
import copy
import cv2
import queue
import gym
import threading
from scipy.spatial.transform import Rotation
from orca_core import OrcaHand, MockOrcaHand

from franka_env.envs.franka_env import FrankaEnv
from franka_env.utils.rotations import euler_2_quat
from franka_env.envs.orca_cube_pick_env.config_binary import OrcaCubePickBinaryEnvConfig
from franka_env.envs.rewards.cube_reward import is_cube_lifted


class OrcaCubePickBinary(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=OrcaCubePickBinaryEnvConfig)
        

        self.reset_hand_pose = self.config.RESET_HAND_POSE
        
        self.hand_open_pose = self.config.HAND_OPEN_POSE
        self.hand_close_pose = self.config.HAND_CLOSE_POSE
        
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )
        
        self.observation_space["state"] = gym.spaces.Dict(
            {
                "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                "hand_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
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
        
        self.hand.set_joint_pos(self.hand_open_pose, num_steps=25, step_size=0.001)    
        self.current_hand_pos = -1 # (open)
        
        # Hand control thread variables
        self.hand_action_queue = queue.Queue(maxsize=1)
        self.hand_control_running = True
        self.hand_control_hertz = 30.0  # Variable hertz rate
        self.hand_control_thread = threading.Thread(target=self._hand_control_loop, daemon=True)
        self.hand_control_thread.start()
    
        self.max_episode_length = 200

        
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
        
        print(f'hand_open_pose: {self.hand_open_pose}')
        print(f'hand_close_pose: {self.hand_close_pose}')
        
        

    def _hand_control_loop(self):
        """Background hand control thread running at variable hertz"""
        target_interval = 1.0 / self.hand_control_hertz
        last_time = time.time()
        num_steps = int(self.hand_control_hertz / self.hz)
        print(f'hand_control_hertz: {self.hand_control_hertz}')
        print(f'hz: {self.hz}')
        print(f'num_steps: {num_steps}')
        num_steps = 1
        
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
                    
                    
                    curr_pos = self.current_hand_pos
                    
                    next_pos = curr_pos + hand_action * self.action_scale[2]
                    
                    next_pos = np.clip(next_pos, -1, 1)
                                        
                
                    normalized_next_pos = (next_pos + 1) / 2
                    
                    target_dict = {}
                    for i, joint_id in enumerate(self.hand.joint_ids):
                        target_dict[joint_id] = self.hand_open_pose[joint_id] + (self.hand_close_pose[joint_id] - self.hand_open_pose[joint_id]) * normalized_next_pos
                    
                    target_dict = self.clip_hand_angles(target_dict)

                    # Send to hand
                    self.hand.set_joint_pos(target_dict)
                    
                    self.current_hand_pos = next_pos
                        
                    
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
            queue_size = self.hand_action_queue.qsize()
            last_time = current_time

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
        # if self.inner_safety_box.contains(pose[:3]):
        #     # print(f'Command: {pose[:3]}')
        #     pose[:3] = self.intersect_line_bbox(
        #         self.currpos[:3],
        #         pose[:3],
        #         self.inner_safety_box.low,
        #         self.inner_safety_box.high,
        #     )
        #     #print(f'Clipped: {pose[:3]}')
        return pose


    def reset(self, joint_reset=False, **kwargs):
        self.current_hand_pos = -1 # (open)
        return super().reset(joint_reset, **kwargs)
    
    def compute_reward(self, obs) -> float:
        sgm = self.get_sgm()
        reward = is_cube_lifted(sgm)
        return reward
    
    def get_hand_angles(self):
        return self.hand.get_joint_pos()
    
    def set_hand_angles(self, angles):
        self.hand.set_joint_pos(angles)
        
    def clip_hand_angles(self, angles: dict) -> dict:
        for joint_id, angle in angles.items():
            angles[joint_id] = np.clip(angle, self.hand_open_pose[joint_id], self.hand_close_pose[joint_id])
        return angles
        
    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        
        # Section 1: Action processing
        t1 = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]
        
        hand_action = action[6]  
        #print(f'hand_action: {hand_action}')
        t1_duration = time.time() - t1
        
        # Section 2: Position calculation
        t2 = time.time()
        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]
        
        self.nextpos[3:] = (
            Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
            * Rotation.from_quat(self.currpos[3:])
        ).as_quat()
        
        self._send_pos_command(self.clip_safety_box(self.nextpos))

        
        t2_duration = time.time() - t2
        
        # Section 4: Send hand action to background thread (NON-BLOCKING)
        t4 = time.time()
        try:
            self.hand_action_queue.put_nowait(hand_action)
        except queue.Full:
            # Queue is full, skip this action
            pass

        # Section 5: Robot state update
        t5 = time.time()
        self._update_currpos()

        # Section 6: Observation and reward computation
        t6 = time.time()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
       # print(f'reward: {reward}')
        t6_duration = time.time() - t6

        # Section 7: Loop timing and sleep
        t7 = time.time()
        self.curr_path_length += 1
        total_duration = time.time() - start_time
        target_duration = 1.0 / self.hz  # Should be 0.1 seconds for 10 Hz
        sleep_time = max(0, target_duration - total_duration)
        
        # Print timing breakdown
        # print(f"\n=== TIMING BREAKDOWN (Target: {target_duration:.3f}s) ===")
        # print(f"1. Action processing:     {t1_duration*1000:.1f}ms")
        # print(f"2. Position calculation:  {t2_duration*1000:.1f}ms")
        # print(f"3. Safety clipping:       {t3_duration*1000:.1f}ms")
        # print(f"4. Hand action queue:     {t4_duration*1000:.1f}ms")
        # print(f"5. Robot state update:    {t5_duration*1000:.1f}ms")
        # print(f"6. Observation/reward:    {t6_duration*1000:.1f}ms")
        # print(f"7. Loop overhead:         {(t7-t6)*1000:.1f}ms")
        # print(f"Total execution time:     {total_duration*1000:.1f}ms")
        # print(f"Sleep time:               {sleep_time*1000:.1f}ms")
        # print(f"Actual loop rate:         {1.0/total_duration:.1f} Hz")
        # print(f"Target loop rate:         {self.hz:.1f} Hz")
        # print("=" * 50)
        
        time.sleep(sleep_time)

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
        self._update_currpos()

        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] += 0.05
        self.interpolate_move(reset_pose, timeout=1)
        self.hand.set_joint_pos(self.hand_open_pose, num_steps=25, step_size=0.001)
      
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

        self.current_hand_angles = self.hand.get_joint_pos()

    def _get_obs(self) -> dict:
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "hand_pose": self.current_hand_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))
    
    def close(self):
        """Clean up resources"""
        self.hand_control_running = False
        if hasattr(self, 'hand_control_thread'):
            self.hand_control_thread.join(timeout=1.0)
        super().close()
