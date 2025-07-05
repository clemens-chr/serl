import cv2
import numpy as np
from collections import deque
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation
import time
from franka_env.camera.cube_position import CubePosition
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R


class CubeReward:
    def __init__(self, mode="velocity", queue_size=10, threshold=0.05):
        self.mode = mode
        self.position_queue = deque(maxlen=queue_size)
        self.time_queue = deque(maxlen=queue_size)
        self.threshold = threshold

    def get_white_pixel_center(self, mask_image: np.ndarray, threshold: int = 127) -> Tuple[Optional[float], Optional[float]]:
        if len(mask_image.shape) == 3:
            gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = mask_image
        white_pixels = np.where(gray > threshold)
        if len(white_pixels[0]) == 0:
            return None, None
        avg_y = np.mean(white_pixels[0])
        avg_x = np.mean(white_pixels[1])
        return float(avg_x), float(avg_y)

    def get_white_pixel_center_normalized(self, mask_image: np.ndarray, threshold: int = 127) -> Tuple[Optional[float], Optional[float]]:
        avg_x, avg_y = self.get_white_pixel_center(mask_image, threshold)
        if avg_x is None or avg_y is None:
            return None, None
        height, width = mask_image.shape[:2]
        normalized_x = avg_x / width
        normalized_y = avg_y / height
        return float(normalized_x), float(normalized_y)

    def is_left_cube(self, sgm):
        avg_x, avg_y = self.get_white_pixel_center_normalized(sgm)
        if avg_x is None or avg_y is None:
            return 0
        return 1 if 0.35 < avg_x < 0.68 and avg_y < 0.53 else 0

    def is_right_cube(self, sgm):
        avg_x, avg_y = self.get_white_pixel_center_normalized(sgm)
        if avg_x is None or avg_y is None:
            return 0
        return 1 if 0.25 < avg_x < 0.5 and avg_y < 0.53 else 0

    def is_cube_lifted(self, sgm):
        avg_x, avg_y = self.get_white_pixel_center_normalized(sgm)
        if avg_x is None or avg_y is None:
            return 0
        return 1 if 0.35 < avg_x < 0.75 and avg_y < 0.58 else 0

    def is_cube_rotated(self, goal_quat, curr_quat, type="dense"):
        """
        Reward function with two types:
        - "binary": Returns 1.0 if within 5 degrees, 0.0 otherwise
        - "dense": Returns 1.0 if within 8 degrees, 0.0 at 90 degrees, smooth falloff
        """
        if goal_quat is None or curr_quat is None:
            return 0.0
        
        # Compute angular difference
        angle_rad = self.compute_angular_difference(curr_quat, goal_quat)
        angle_deg = np.degrees(angle_rad)
        
        print(f'angle_deg: {angle_deg}')
        
        if type == "binary":
            # Binary reward: 1 if within 5 degrees, 0 otherwise
            reward = 1.0 if angle_deg <= 5.0 else 0.0
        
        elif type == "dense":
            # Dense reward: 1 at 0°, 0 at 90°, smooth falloff
            if angle_deg <= 8.0:
                reward = 1.0
            elif angle_deg >= 90.0:
                reward = 0.0
            else:
                # Linear interpolation between 8° and 90°
                # At 8°: reward = 1.0, At 90°: reward = 0.0
                reward = 1.0 - ((angle_deg - 8.0) / (90.0 - 8.0))
                reward = max(0.0, min(1.0, reward))  # Clamp between 0 and 1
        
        else:
            # Default to binary
            reward = 1.0 if angle_deg <= 5.0 else 0.0
        
        return reward
    
    def compute_goal_quat(self, initial_quat, angle_deg=90):
        """Rotate initial_quat by `angle_deg` around the z-axis (0, 0, 1)."""
        rot = R.from_quat(initial_quat)
        # Rotate around z-axis (0, 0, 1) by angle_deg
        axis_rot = R.from_euler('z', angle_deg, degrees=True)
        return (axis_rot * rot).as_quat()

    def compute_angular_difference(self, q1, q2):
        """Returns angle in radians between two quaternions."""
        q_diff = R.from_quat(q1).inv() * R.from_quat(q2)
        q_diff = q_diff.as_quat()
        q_diff = q_diff / np.linalg.norm(q_diff)
        return 2.0 * np.arcsin(np.clip(np.linalg.norm(q_diff[:3]), 0.0, 1.0))


    def get_reward(self, dominant_axis_rotation: float) -> float:
        if self.mode != "velocity" or dominant_axis_rotation is None:
            return 0.0

        t = time.monotonic()
        self.position_queue.append(dominant_axis_rotation)
        self.time_queue.append(t)

        if len(self.position_queue) < 2:
            return 0.0

        # Fit a linear slope to angle vs time (least squares velocity estimate)
        angles = np.unwrap(np.radians(self.position_queue))  # unwrap for smooth transition
        times = np.array(self.time_queue)
        A = np.vstack([times - times[0], np.ones(len(times))]).T
        slope, _ = np.linalg.lstsq(A, angles, rcond=None)[0]  # slope = rad/s

        deg_per_sec = np.degrees(slope)

        # Apply deadzone threshold
        if abs(deg_per_sec) < self.threshold:
            return 0.0

        # Cap at 1.0
        return min(abs(deg_per_sec) / 180.0, 1.0)
    
    def get_reward_90(self, principal_angle: float, angle2: float, angle3: float) -> float:
        """
        Reward function that gives maximum reward (1.0) when:
        - principal_angle > 90 degrees
        - angle2 < 10 degrees  
        - angle3 < 10 degrees
        
        Smooth falloff for other cases.
        """
        if principal_angle is None or angle2 is None or angle3 is None:
            return 0.0
        
        # Convert to absolute values for angle comparison
        principal_abs = abs(principal_angle)
        angle2_abs = abs(angle2)
        angle3_abs = abs(angle3)
        
        # Check if conditions are met for maximum reward
        principal_condition = principal_abs > 90.0
        angle2_condition = angle2_abs < 10.0
        angle3_condition = angle3_abs < 10.0
        
        if principal_condition and angle2_condition and angle3_condition:
            return 1.0
        
        # Calculate individual component rewards with smooth falloff
        # Principal angle reward: peaks at 90+ degrees, falls off smoothly
        if principal_abs >= 90.0:
            principal_reward = 1.0
        else:
            # Smooth falloff from 0 to 90 degrees
            principal_reward = max(0.0, principal_abs / 90.0)
        
        # Secondary angles reward: peaks when close to 0, falls off as they increase
        angle2_reward = max(0.0, 1.0 - (angle2_abs / 10.0))
        angle3_reward = max(0.0, 1.0 - (angle3_abs / 10.0))
        
        # Combine rewards (geometric mean for smooth combination)
        combined_reward = (principal_reward * angle2_reward * angle3_reward) ** (1/3)
        
        # Apply additional penalty for being very far from target
        if principal_abs < 45.0 or angle2_abs > 45.0 or angle3_abs > 45.0:
            combined_reward *= 0.1  # Heavy penalty for being far from target
        
        return combined_reward
    

class RewardPlotter:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.reward_times = deque(maxlen=window_size)
        self.reward_values = deque(maxlen=window_size)
        self.reward_90_values = deque(maxlen=window_size)
        self.velocity_values = deque(maxlen=window_size)
        
        # Setup the plot
        plt.ion()
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 10))
        self.fig.suptitle('Cube Rewards Over Time', fontsize=16)
        
        # Velocity reward plot
        self.ax1.set_title('Velocity Reward')
        self.ax1.set_ylabel('Reward')
        self.ax1.set_ylim(-0.1, 1.1)
        self.ax1.grid(True, alpha=0.3)
        
        # 90-degree reward plot
        self.ax2.set_title('90° Orientation Reward')
        self.ax2.set_ylabel('Reward')
        self.ax2.set_ylim(-0.1, 1.1)
        self.ax2.grid(True, alpha=0.3)
        
        # Velocity plot
        self.ax3.set_title('Velocity (deg/s)')
        self.ax3.set_ylabel('Velocity (deg/s)')
        self.ax3.set_xlabel('Time (s)')
        self.ax3.grid(True, alpha=0.3)
        
        self.reward_line, = self.ax1.plot([], [], 'b-', linewidth=2, label='Velocity Reward')
        self.reward_90_line, = self.ax2.plot([], [], 'g-', linewidth=2, label='90° Reward')
        
        self.ax1.legend()
        self.ax2.legend()
        self.ax3.legend()
        
    def update_plot(self, reward):
        current_time = time.monotonic()
        self.reward_times.append(current_time)
        self.reward_values.append(reward)
        
        # Update reward plots
        if len(self.reward_times) > 1:
            times = np.array(self.reward_times)
            rewards = np.array(self.reward_values)
            
            # Set x-axis limits for moving window
            if len(times) > self.window_size:
                start_time = times[-self.window_size]
            else:
                start_time = times[0]
            end_time = times[-1]
            
            self.ax1.set_xlim(start_time, end_time)
            self.ax2.set_xlim(start_time, end_time)
            
            self.reward_line.set_data(times, rewards)
        
        plt.pause(0.01)
    
    def close(self):
        plt.ioff()
        plt.close()


   


if __name__ == "__main__":
    cube_position = CubePosition()
    rewarder = CubeReward()
    goal_quat = None
    while True:

        while goal_quat is None:
            pose = cube_position.get_cube_pose()
            if pose is not None:
                initial_quat = pose[3:]
                cube_position.get_cube_pose()  # one extra for axis calc
                goal_quat = rewarder.compute_goal_quat(initial_quat)
            time.sleep(0.1)

        
        while True:
            pose = cube_position.get_cube_pose()
            if pose is None:
                continue

            curr_quat = pose[3:]
            angle_rad = rewarder.compute_angular_difference(curr_quat, goal_quat)
            angle_deg = np.degrees(angle_rad)

            reward = 1.0 if angle_rad < np.radians(5) else max(0.0, 1.0 - angle_rad / np.pi)
            
            if reward == 1.0:
                time.sleep(3)
                goal_quat = None
                cube_position.reset_reference()
                break

            print(f"Angular difference: {angle_deg:.2f}°, Reward: {reward:.3f}")
            time.sleep(0.1)
