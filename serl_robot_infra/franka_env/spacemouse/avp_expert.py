import time
import multiprocessing
import numpy as np
from avp_stream import VisionProStreamer
import scipy.spatial.transform
from typing import Tuple
from dataclasses import dataclass
from scipy.spatial.transform import Rotation # Import Rotation

class AVPExpertPinch:
    """
    This class provides an interface to the Apple Vision Pro to teleoperate a robotic hand.
    """

    def __init__(self, avp_ip: str = None):
        
        AVP_IP = avp_ip or "10.93.181.127"
        PINCH_THRESHOLD = 0.02
        CONTROL_LOOP_HZ = 50 
        
        # Manager to handle shared state between processes
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6
        self.latest_data["grasping"] = [0, 0, 0, 0]
        
        # This controls whether the user want to intervene
        # When left finger pinches, the user can control the robot
        self.latest_data["is_intervening"] = False
        
        self.stop_event = multiprocessing.Event()
           
        # Start a process to continuously read from the AVP
        self.process = multiprocessing.Process(
            target=self._read_avp,
            args=(AVP_IP, CONTROL_LOOP_HZ, PINCH_THRESHOLD, self.stop_event),
        )
        
        self.process.start()
    
    def _get_7d_pose_from_avp_matrix(self, matrix: np.ndarray) -> np.ndarray | None:
        """Extracts [x,y,z,qx,qy,qz,qw] from a 4x4 AVP matrix."""
        matrix = np.squeeze(matrix) # Ensure it's a 2D array
        if matrix is None or matrix.shape != (4, 4):
            print(f"Error: Invalid AVP matrix shape for pose extraction: {matrix.shape if matrix is not None else 'None'}")
            return None
        try:
            position = matrix[:3, 3]
            rotation_matrix = matrix[:3, :3]
            if np.linalg.det(rotation_matrix) < 0.1: # Check determinant is close to 1
                print(f"Warning: Possibly invalid rotation matrix (det={np.linalg.det(rotation_matrix)}). Using identity orientation.")
                # Fallback to identity quaternion or handle as error
                orientation_quat = np.array([0.0, 0.0, 0.0, 1.0]) # w is last in scipy
            else:
                orientation_quat = Rotation.from_matrix(rotation_matrix).as_quat() # [x,y,z,w]
            return np.concatenate([position, orientation_quat])

        except Exception as e:
            print(f"Error converting AVP matrix to 7D pose: {e}")
            return None

    def _read_avp(self, ip: float, control_loop_hz: float, pinch_threshold: float, stop_event: multiprocessing.Event):
        
        pinch_active = False
        reference_avp_pose = None 
        reference_franka_pose = None 
        
        stream = VisionProStreamer(ip = ip, record = True)

        while not stream and not stop_event.is_set():
            try:
                # Initialize the AVP streamer
                stream = VisionProStreamer(ip = ip, record = True)
            except Exception as e:
                print(f"Error initializing AVP streamer: {e}")
                time.sleep(10)

        last_loop_time = time.time()

        while stream.latest is None: 
            print("Waiting for AVP stream to start...")
            time.sleep(0.1) # Wait for the stream to start
            pass 

        while not stop_event.is_set():

            while not stream and not stop_event.is_set():
                try:
                    # Initialize the AVP streamer
                    stream = VisionProStreamer(ip = ip, record = True)
                except Exception as e:
                    print(f"Error initializing AVP streamer: {e}")
                    time.sleep(1)

            current_time = time.time()
            if current_time - last_loop_time < (1.0 / control_loop_hz):
                time.sleep(0.001) # Sleep briefly if looping too fast
                continue
            last_loop_time = current_time
            

            data = stream.latest
            action = [0.0] * 6
            
            
            state_right = data["right_fingers"]  # shape (25,4,4)
            pinch_right = data["right_pinch_distance"] # float
            pinch_left = data["left_pinch_distance"] # float
            
            right_wrist_matrix = data["right_wrist"]
            pinch_left = data["left_pinch_distance"]
            
            current_wrist_pose_7d = self._get_7d_pose_from_avp_matrix(right_wrist_matrix)
                        
            if pinch_right < pinch_threshold and pinch_right > 0:
                pinching_right = True
            else:
                pinching_right = False
                
            if pinch_left < pinch_threshold and pinch_left > 0:
                self.latest_data["is_intervening"] = True
            else:
                self.latest_data["is_intervening"] = False
            
                
            # Extract translation (x, y, z) from the matrix
            translation = current_wrist_pose_7d[:3]
            # rotation = scipy.spatial.transform.Rotation.from_matrix(right_wrist_matrix[:3, :3]).as_euler('xyz', degrees=False)
            rotation = [0.0, 0.0, 0.0]
            action = [
                translation[1]*0.5, -translation[0]*0.5, translation[2]*0.5,  # y, x, z
                -rotation[0], -rotation[1], -rotation[2]          # roll, pitch, yaw
            ]
            self.latest_data["action"] = action
            self.latest_data["grasping"] = pinching_right
            
    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and pinch distance of the AVP."""
        action = self.latest_data["action"]
        grasping = self.latest_data["grasping"]
        return np.array(action), grasping
    
    def is_intervening(self) -> bool:
        return self.latest_data["is_intervening"]
    
    def close(self):
        print("Stopping AVP process...")
        self.stop_event.set()
        self.process.terminate()



class AVPExpert:
    """
    This class provides an interface to the Apple Vision Pro to teleoperate a robotic hand.
    """

    def __init__(self, avp_ip: str = None):
        
        AVP_IP = avp_ip or "10.93.181.127"
        PINCH_THRESHOLD = 0.02
        CONTROL_LOOP_HZ = 50 
        
        # Manager to handle shared state between processes
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6
        self.latest_data["grasping"] = [0, 0, 0, 0]
        
        # This controls whether the user want to intervene
        # When left finger pinches, the user can control the robot
        self.latest_data["is_intervening"] = False
        
        self.stop_event = multiprocessing.Event()
           
        # Start a process to continuously read from the AVP
        self.process = multiprocessing.Process(
            target=self._read_avp,
            args=(AVP_IP, CONTROL_LOOP_HZ, PINCH_THRESHOLD, self.stop_event),
        )
        
        self.process.start()
    
    def _get_7d_pose_from_avp_matrix(self, matrix: np.ndarray) -> np.ndarray | None:
        """Extracts [x,y,z,qx,qy,qz,qw] from a 4x4 AVP matrix."""
        matrix = np.squeeze(matrix) # Ensure it's a 2D array
        if matrix is None or matrix.shape != (4, 4):
            print(f"Error: Invalid AVP matrix shape for pose extraction: {matrix.shape if matrix is not None else 'None'}")
            return None
        try:
            position = matrix[:3, 3]
            rotation_matrix = matrix[:3, :3]
            if np.linalg.det(rotation_matrix) < 0.1: # Check determinant is close to 1
                print(f"Warning: Possibly invalid rotation matrix (det={np.linalg.det(rotation_matrix)}). Using identity orientation.")
                # Fallback to identity quaternion or handle as error
                orientation_quat = np.array([0.0, 0.0, 0.0, 1.0]) # w is last in scipy
            else:
                orientation_quat = Rotation.from_matrix(rotation_matrix).as_quat() # [x,y,z,w]
            return np.concatenate([position, orientation_quat])

        except Exception as e:
            print(f"Error converting AVP matrix to 7D pose: {e}")
            return None

    def _read_avp(self, ip: float, control_loop_hz: float, pinch_threshold: float, stop_event: multiprocessing.Event):
        
        pinch_active = False
        reference_avp_pose = None 
        reference_franka_pose = None 
        
        stream = VisionProStreamer(ip = ip, record = True)

        while not stream and not stop_event.is_set():
            try:
                # Initialize the AVP streamer
                stream = VisionProStreamer(ip = ip, record = True)
            except Exception as e:
                print(f"Error initializing AVP streamer: {e}")
                time.sleep(10)

        last_loop_time = time.time()

        while stream.latest is None: 
            print("Waiting for AVP stream to start...")
            time.sleep(0.1) # Wait for the stream to start
            pass 

        while not stop_event.is_set():

            while not stream and not stop_event.is_set():
                try:
                    # Initialize the AVP streamer
                    stream = VisionProStreamer(ip = ip, record = True)
                except Exception as e:
                    print(f"Error initializing AVP streamer: {e}")
                    time.sleep(1)

            current_time = time.time()
            if current_time - last_loop_time < (1.0 / control_loop_hz):
                time.sleep(0.001) # Sleep briefly if looping too fast
                continue
            last_loop_time = current_time
            

            data = stream.latest
            action = [0.0] * 6
            
            
            state_right = data["right_fingers"]  # shape (25,4,4)
            pinch_right = data["right_pinch_distance"] # float
            pinch_left = data["left_pinch_distance"] # float
            
            right_wrist_matrix = data["right_wrist"]
            pinch_left = data["left_pinch_distance"]
            
            current_wrist_pose_7d = self._get_7d_pose_from_avp_matrix(right_wrist_matrix)
                        
            if pinch_right < pinch_threshold and pinch_right > 0:
                pinching_right = True
            else:
                pinching_right = False
                
            if pinch_left < pinch_threshold and pinch_left > 0:
                self.latest_data["is_intervening"] = True
            else:
                self.latest_data["is_intervening"] = False
            
                
            # Extract translation (x, y, z) from the matrix
            translation = current_wrist_pose_7d[:3]
            # rotation = scipy.spatial.transform.Rotation.from_matrix(right_wrist_matrix[:3, :3]).as_euler('xyz', degrees=False)
            rotation = [0.0, 0.0, 0.0]
            action = [
                translation[1]*0.5, -translation[0]*0.5, translation[2]*0.5,  # y, x, z
                -rotation[0], -rotation[1], -rotation[2]          # roll, pitch, yaw
            ]
            self.latest_data["action"] = action
            self.latest_data["grasping"] = pinching_right
            
    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and pinch distance of the AVP."""
        action = self.latest_data["action"]
        grasping = self.latest_data["grasping"]
        return np.array(action), grasping
    
    def is_intervening(self) -> bool:
        return self.latest_data["is_intervening"]
    
    def close(self):
        print("Stopping AVP process...")
        self.stop_event.set()
        self.process.terminate()
