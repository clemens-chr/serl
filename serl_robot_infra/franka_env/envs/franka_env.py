"""Gym Interface for Franka"""
import numpy as np
import gym
import cv2
import copy
from scipy.spatial.transform import Rotation
import time
import requests
import queue
import threading
from datetime import datetime
from collections import OrderedDict
from typing import Dict
import os

from franka_env.camera.video_capture import VideoCapture
from franka_env.camera.rs_capture import RSCapture
from franka_env.utils.rotations import euler_2_quat, quat_2_euler

class ImageDisplayer(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            frame = np.concatenate(
                [v for k, v in img_array.items() if "full" not in k], axis=0
            )

            cv2.imshow("RealSense Cameras", frame)
            cv2.waitKey(1)


##############################################################################


class DefaultEnvConfig:
    """Default configuration for FrankaEnv. Fill in the values below."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS: Dict = {
        "wrist_1": "130322274175",
        "wrist_2": "317622074238",
    }
    TARGET_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    RESET_POSE = np.zeros((6,))
    RANDOM_RESET = (False,)
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    COMPLIANCE_PARAM: Dict[str, float] = {}
    PRECISION_PARAM: Dict[str, float] = {}
    BINARY_GRIPPER_THREASHOLD: float = 0.0
    APPLY_GRIPPER_PENALTY: bool = True
    GRIPPER_PENALTY: float = 0.1


##############################################################################


class FrankaEnv(gym.Env):
    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
        max_episode_length=100,
    ):
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.url = config.SERVER_URL
        self.config = config
        self.max_episode_length = max_episode_length

        # convert last 3 elements from euler to quat, from size (6,) to (7,)
        if len(config.RESET_POSE) == 6:
            self.resetpos = np.concatenate(
                [config.RESET_POSE[:3],  Rotation.from_euler('xyz', config.RESET_POSE[3:]).as_quat()]
            )
        else:
            self.resetpos = config.RESET_POSE

        self.currpos = self.resetpos.copy()
        self.currpose_euler = np.zeros((6,))
        self.currvel = np.zeros((6,))
        self.q = np.zeros((7,))
        self.dq = np.zeros((7,))
        self.currforce = np.zeros((3,))
        self.currtorque = np.zeros((3,))
        self.currjacobian = np.zeros((6, 7))

        self.curr_gripper_pos = 0.08
        self.gripper_binary_state = 0  # 0 for open, 1 for closed
        self.lastsent = time.time()
        
        # Gripper cooldown functionality (temporary flag for testing)
        self.gripper_cooldown_enabled = True  # Set to False to disable
        self.last_gripper_state_change = time.time()
        self.gripper_cooldown_duration = 2.0  # 5 seconds cooldown
        
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        self.joint_reset_cycle = 200  # reset the robot joint every 200 cycles

        if save_video:
            print("Saving videos!")
        self.save_video = save_video
        self.recording_frames = []

        # boundary box
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        "front": gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        ),
                        "side": gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        ),
                    }
                ),
            }
        )
        self.cycle_count = 0

        if fake_env:
            return

        self.cap = None
        self.init_cameras(config.REALSENSE_CAMERAS)
        self.img_queue = queue.Queue()
        self.displayer = ImageDisplayer(self.img_queue)
        self.displayer.start()
        
    
        print("Initialized Franka")

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        if np.any(pose[:3] != np.clip(pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high)):
            print("Position was clipped to safety box bounds")
        
        
        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(
                np.abs(euler[0]),
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
        )

        euler[1:] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat()

        return pose

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]
        gripper_action = action[6]
        

        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

        # GET ORIENTATION FROM ACTION
        self.nextpos[3:] = (
            Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
            * Rotation.from_quat(self.currpos[3:])
        ).as_quat()
        
        #print(f'gripper_action: {gripper_action}')
        gripper_action_effective = self._send_gripper_command(gripper_action)
        if not np.array_equal(self.clip_safety_box(self.nextpos), self.nextpos):
            print(f'CLIPPING OCCURED: {self.nextpos}')
        self._send_pos_command(self.clip_safety_box(self.nextpos))

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length or reward == 1
        return ob, reward, done, False, {}


    def compute_reward(self, obs) -> float:
        """We are using a sparse or dense reward function."""
        reward_type = 'sparse'
        
        current_pose = obs["state"]["tcp_pose"]
        euler_angles = quat_2_euler(current_pose[3:])
        # euler_angles = np.abs(euler_angles)
        current_pose = np.hstack([current_pose[:3], euler_angles])
        delta = np.abs(current_pose - self._TARGET_POSE)
            
        if reward_type == 'dense':
            if np.all(delta < self._REWARD_THRESHOLD):
                reward = 1  
            else:
                decay_rate = 10  
                reward = np.exp(-decay_rate * np.sum(delta))  
                reward = np.clip(reward, 0, 1)  
                
        elif reward_type == 'sparse':
           
            if np.all(delta < self._REWARD_THRESHOLD):
                reward = 1
            else:
                reward = 0
        else:
            raise ValueError(f"Invalid reward type: {reward_type}")

        if self.config.APPLY_GRIPPER_PENALTY and False:
            reward -= self.config.GRIPPER_PENALTY
            
            #print(f'Reward: {reward}')
        return reward

    def crop_image(self, name, image) -> np.ndarray:
        """Crop realsense images to be a square."""
        if name.startswith("side"):
            return image[:, 80:560, :]
        elif name.startswith("front"):
            return image[:, 80:560, :]
        else:
            return ValueError(f"Camera {name} not recognized in cropping")

    def get_im(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}

        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                sgm = cap.read_segmentation()
                
                cropped_rgb = self.crop_image(key, rgb)
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                
                if sgm is not None:
                    cropped_sgm = self.crop_image(key, sgm)
                    resized_sgm = cv2.resize(cropped_sgm, self.observation_space["images"][key].shape[:2][::-1])
                    display_images[key + "_sgm"] = resized_sgm
                    display_images[key + "_sgm_full"] = cropped_sgm
                                    
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im()

        self.recording_frames.append(
            np.concatenate([display_images[f"{k}_full"] for k in self.cap], axis=0)
        )
      
        self.img_queue.put(display_images)
        return images
    
    def get_sgm(self) -> Dict[str, np.ndarray]:
        """Get segmentation masks from the realsense cameras."""
        sgm = self.cap["front"].read_segmentation()
        if sgm is not None:
            cv2.imwrite(f'./sgm.png', sgm)
            #print(f'saved sgm to ./sgm.png')
        return sgm

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        steps = int(timeout * self.hz)
        self._update_currpos()
        path = np.linspace(self.currpos, goal, steps)
        for p in path:
            self._send_pos_command(p)
            time.sleep(1 / self.hz)
        self._update_currpos()
        #print(f'interpolated move to {goal}')
        time.sleep(1)

    def go_to_rest(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """
        # Change to precision mode for reset
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.2)
        

        # Perform joint reset if needed
        #print(f'performing joint reset')
        if joint_reset:
            requests.post(self.url + "jointreset")
            time.sleep(0.2)
                    

        # Perform Carteasian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._TARGET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=1.5)
        else:
            reset_pose = self.resetpos.copy()
            #print(f'moving to reset_pose: {reset_pose}')
            self.interpolate_move(reset_pose, timeout=1.5)

        # Change to compliance mode
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)

    def reset(self, joint_reset=False, **kwargs):
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True

        self.go_to_rest(joint_reset=joint_reset)
        self._recover()
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()

        return obs, {}

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                video_writer = cv2.VideoWriter(
                    f'./videos/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mp4',
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10,
                    self.recording_frames[0].shape[:2][::-1],
                )
                for frame in self.recording_frames:
                    video_writer.write(frame)
                video_writer.release()
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def init_cameras(self, name_serial_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, cam_serial in name_serial_dict.items():
            server_url = None
            if cam_name == "front":
                server_url = "ws://localhost:8765"
                
            cap = VideoCapture(
                RSCapture(name=cam_name, serial_number=cam_serial, depth=False, dummy_mode=False),
                name=cam_name,
                server_url_segment=server_url
            )
            self.cap[cam_name] = cap

    def close_cameras(self):
        """Close both wrist cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def _recover(self):
        """Internal function to recover the robot from error state."""
        requests.post(self.url + "clearerr")

    def _send_pos_command(self, pos: np.ndarray):
        """Internal function to send position command to the robot."""
        self._recover()
        arr = np.array(pos).astype(np.float32)
        data = {"arr": arr.tolist()}
        #print(f'LAST ACTION: {arr}')
        requests.post(self.url + "pose", json=data)

    def _send_gripper_command(self, pos: float, mode="binary"):
        """Internal function to send gripper command to the robot."""
        
        if mode == "binary":
            # Check if gripper cooldown is enabled and if enough time has passed
            if self.gripper_cooldown_enabled:
                current_time = time.time()
                time_since_last_change = current_time - self.last_gripper_state_change
                
                if time_since_last_change < self.gripper_cooldown_duration:
                    print(f"Gripper cooldown active: {self.gripper_cooldown_duration - time_since_last_change:.1f}s remaining")
                    return False
            
            if (
                pos < self.config.BINARY_GRIPPER_THREASHOLD
                and self.gripper_binary_state == 0
            ):  # close gripper
                requests.post(self.url + "close_gripper")
                print("Closed gripper")
                time.sleep(0.6)
                self.gripper_binary_state = 1
                if self.gripper_cooldown_enabled:
                    self.last_gripper_state_change = time.time()
                return True
            elif (
                pos > self.config.BINARY_GRIPPER_THREASHOLD
                and self.gripper_binary_state == 1
            ):  # open gripper
                requests.post(self.url + "open_gripper")
                print("Opened gripper")
                time.sleep(0.6)
                self.gripper_binary_state = 0
                if self.gripper_cooldown_enabled:
                    self.last_gripper_state_change = time.time()
                return True
            else:  # do nothing to the gripper
                return False
        elif mode == "continuous":
            print(f'Sending continuous gripper command: {pos}')
            requests.post(self.url + "move_gripper", json={"gripper_pos": pos})
            return True

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

        self.curr_gripper_pos = np.array(ps["gripper_pos"])


    def _get_obs(self) -> dict:
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))
