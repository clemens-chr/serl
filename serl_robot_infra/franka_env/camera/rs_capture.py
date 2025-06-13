import numpy as np
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API
import time
from termcolor import colored


class RSCapture:
    def get_device_serial_numbers(self):
        devices = rs.context().devices
        serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
        print(colored(f"Found {len(serials)} RealSense cameras: {serials}", "green"))
        return serials

    def __init__(self, name, serial_number, dim=(640, 480), fps=15, depth=False):
        self.name = name
        print(colored(f"Initializing RealSense camera {name} with serial: {serial_number}", "green"))
        print(colored(f"Available cameras: {self.get_device_serial_numbers()}", "green"))
        
        if serial_number not in self.get_device_serial_numbers():
            raise RuntimeError(f"Camera with serial {serial_number} not found!")
            
        self.serial_number = serial_number
        self.depth = depth
        self.dim = dim
        self.fps = fps
        self.initialize_camera()

    def initialize_camera(self):
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(self.serial_number)
        self.cfg.enable_stream(rs.stream.color, self.dim[0], self.dim[1], rs.format.bgr8, self.fps)
        if self.depth:
            self.cfg.enable_stream(rs.stream.depth, self.dim[0], self.dim[1], rs.format.z16, self.fps)
        
        # Create an align object
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        print(colored(f"Starting RealSense pipeline for camera {self.name}", "green"))
        self.profile = self.pipe.start(self.cfg)

    def read(self):
        try:
            frames = self.pipe.wait_for_frames(timeout_ms=2000)
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            
            if not color_frame:
                print(colored(f"No color frame received from camera {self.name}, attempting to reconnect...", "yellow"))
                self.reconnect()
                return False, None

            if self.depth:
                depth_frame = aligned_frames.get_depth_frame()
                if not depth_frame:
                    print(colored(f"No depth frame received from camera {self.name}, attempting to reconnect...", "yellow"))
                    self.reconnect()
                    return False, None

            if color_frame.is_video_frame():
                image = np.asarray(color_frame.get_data())
                if self.depth and depth_frame.is_depth_frame():
                    depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
                    return True, np.concatenate((image, depth), axis=-1)
                else:
                    return True, image
            else:
                print(colored(f"Invalid frame type from camera {self.name}, attempting to reconnect...", "yellow"))
                self.reconnect()
                return False, None

        except Exception as e:
            print(colored(f"Error reading from camera {self.name}: {str(e)}", "red"))
            self.reconnect()
            return False, None

    def reconnect(self):
        print(colored(f"Attempting to reconnect camera {self.name}...", "yellow"))
        try:
            self.pipe.stop()
        except Exception:
            pass
        time.sleep(1)
        try:
            self.initialize_camera()
            print(colored(f"Successfully reconnected camera {self.name}", "green"))
        except Exception as e:
            print(colored(f"Failed to reconnect camera {self.name}: {str(e)}", "red"))

    def close(self):
        print(colored(f"Closing camera {self.name}", "green"))
        self.pipe.stop()
        self.cfg.disable_all_streams()
