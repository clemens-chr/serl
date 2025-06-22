import numpy as np
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API
import time
from termcolor import colored


class RSCapture:
    def get_device_serial_numbers(self):
        devices = rs.context().devices
        serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
        return serials

    def __init__(self, name, serial_number, dim=(640, 480), fps=15, depth=False, dummy_mode=False):
        self.name = name
        self.dummy_mode = dummy_mode
        self.depth = depth
        self.dim = dim
        
        if dummy_mode:
            print(colored(f"Running camera {name} in dummy mode - returning black images", "yellow"))
            return
            
        print(colored(f"Initializing RealSense camera {name} with serial: {serial_number}", "green"))
        print(colored(f"Available cameras: {self.get_device_serial_numbers()}", "green"))
        
        if serial_number not in self.get_device_serial_numbers():
            raise RuntimeError(f"Camera with serial {serial_number} not found!")
            
        self.serial_number = serial_number
        self.fps = fps
        time.sleep(1)
        self.initialize_camera()

    def get_valid_frame(self):
        """Get a valid frame with infinite retry loop (like the working example)"""
        while True:
            try:
                frames = self.pipe.wait_for_frames(timeout_ms=2000)
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                
                if not color_frame:
                    raise RuntimeError("No color frame received.")
                
                if self.depth:
                    depth_frame = aligned_frames.get_depth_frame()
                    if not depth_frame:
                        raise RuntimeError("No depth frame received.")
                    return color_frame, depth_frame
                else:
                    return color_frame, None
                    
            except Exception as e:
                print(colored(f"Error getting frame from camera {self.name}: {e}", "yellow"))
                print(colored(f"Attempting to reconnect camera {self.name}...", "yellow"))
                self.reconnect()

    def initialize_camera(self):
        # Add small delay to prevent device busy errors
        time.sleep(0.1)
        
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
        
        # Try to start pipeline with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.profile = self.pipe.start(self.cfg)
                break
            except Exception as e:
                if "Device or resource busy" in str(e) and attempt < max_retries - 1:
                    print(colored(f"Device busy, retrying in 1 second... (attempt {attempt + 1}/{max_retries})", "yellow"))
                    time.sleep(1)
                else:
                    raise e
        
        # Warmup: Get first 10 frames to adjust exposure (like the working example)
        print(colored(f"Warming up camera {self.name}...", "yellow"))
        for i in range(10):
            try:
                color_frame, depth_frame = self.get_valid_frame()
                print(colored(f"  Warmup frame {i+1}/10: ✓", "green"))
                time.sleep(0.1)
            except Exception as e:
                print(colored(f"  Warmup frame {i+1}/10: ⚠ ({str(e)})", "yellow"))
        
        print(colored(f"Camera {self.name} initialization complete!", "green"))

    def read(self):
        if self.dummy_mode:
            # Return black image with optional depth channel
            if self.depth:
                return True, np.zeros((*self.dim, 4), dtype=np.uint8)  # RGBD
            return True, np.zeros((*self.dim, 3), dtype=np.uint8)  # RGB
            
        try:
            # Use the reliable get_valid_frame method
            color_frame, depth_frame = self.get_valid_frame()
            
            if color_frame.is_video_frame():
                image = np.asarray(color_frame.get_data())
                if self.depth and depth_frame and depth_frame.is_depth_frame():
                    depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
                    return True, np.concatenate((image, depth), axis=-1)
                else:
                    return True, image
            else:
                print(colored(f"Invalid frame type from camera {self.name}", "yellow"))
                return False, None

        except Exception as e:
            print(colored(f"Unexpected error reading from camera {self.name}: {str(e)}", "red"))
            return False, None

    def reconnect(self):
        print(colored(f"Attempting to reconnect camera {self.name}...", "yellow"))
        try:
            self.pipe.stop()
            self.cfg.disable_all_streams()
            time.sleep(2)  # Give more time for resources to be released
        except Exception as e:
            print(colored(f"Error during cleanup: {str(e)}", "yellow"))
            
        try:
            self.initialize_camera()
            print(colored(f"Successfully reconnected camera {self.name}", "green"))
        except Exception as e:
            print(colored(f"Failed to reconnect camera {self.name}: {str(e)}", "red"))
            raise  # Re-raise the exception to handle it in the calling code

    def close(self):
        print(colored(f"Closing camera {self.name}", "green"))
        self.pipe.stop()
        self.cfg.disable_all_streams()
