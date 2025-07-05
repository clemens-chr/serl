import numpy as np
import pyrealsense2 as rs
import time
from termcolor import colored
import zmq
import pickle
import cv2


class RSCapture:
    def __init__(self, name, serial_number, dim=(640, 480), fps=15, depth=False,
                 dummy_mode=False, mode='hardware', stream_host="localhost", stream_port=5555):
        self.name = name
        self.serial_number = serial_number
        self.dummy_mode = dummy_mode
        self.depth = depth
        self.dim = dim
        self.fps = fps
        self.mode = mode
        self.stream_host = stream_host
        self.stream_port = stream_port

      
        if dummy_mode:
            print(colored(f"Running camera {name} in dummy mode - returning black images", "yellow"))
            return

        if mode == 'hardware':
            print(colored(f"Initializing RealSense camera {name} in hardware mode", "green"))
            if serial_number not in self.get_device_serial_numbers():
                raise RuntimeError(f"Camera with serial {serial_number} not found!")
            self.initialize_camera()

        elif mode == 'stream':
            print(colored(f"Initializing RealSense camera {name} in ZMQ stream mode", "cyan"))

            # Setup ZMQ subscriber
            self.context = zmq.Context()
            self.subscriber = self.context.socket(zmq.SUB)
            self.subscriber.setsockopt(zmq.RCVHWM, 1)
            self.subscriber.setsockopt(zmq.CONFLATE, 1)
            self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all
            self.subscriber.connect(f"tcp://{stream_host}:{stream_port}")
        else:
            raise ValueError("Invalid mode. Must be 'hardware' or 'stream'.")

    def get_device_serial_numbers(self):
        devices = rs.context().devices
        serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
        return serials

    def initialize_camera(self):
        time.sleep(0.1)
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(self.serial_number)
        self.cfg.enable_stream(rs.stream.color, self.dim[0], self.dim[1], rs.format.bgr8, self.fps)
        if self.depth:
            self.cfg.enable_stream(rs.stream.depth, self.dim[0], self.dim[1], rs.format.z16, self.fps)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        for attempt in range(3):
            try:
                self.profile = self.pipe.start(self.cfg)
                break
            except Exception as e:
                print(colored(f"Retrying RealSense pipeline start: {e}", "yellow"))
                time.sleep(1)

        print(colored(f"Warming up camera {self.name}...", "yellow"))
        for i in range(10):
            try:
                self.get_valid_frame()
                print(colored(f"  Warmup frame {i+1}/10: ✓", "green"))
                time.sleep(0.1)
            except Exception as e:
                print(colored(f"  Warmup frame {i+1}/10: ⚠ ({str(e)})", "yellow"))
        print(colored(f"Camera {self.name} ready", "green"))

    def get_valid_frame(self):
        while True:
            try:
                frames = self.pipe.wait_for_frames(timeout_ms=2000)
                aligned = self.align.process(frames)
                color = aligned.get_color_frame()
                if not color:
                    raise RuntimeError("No color frame")
                if self.depth:
                    depth = aligned.get_depth_frame()
                    if not depth:
                        raise RuntimeError("No depth frame")
                    return color, depth
                return color, None
            except Exception as e:
                print(colored(f"Frame error: {e}", "yellow"))
                self.reconnect()

    def read(self):
        if self.dummy_mode:
            shape = (*self.dim, 4) if self.depth else (*self.dim, 3)
            return True, np.zeros(shape, dtype=np.uint8)

        if self.mode == 'hardware':
            try:
                color_frame, depth_frame = self.get_valid_frame()
                image = np.asarray(color_frame.get_data())
                if self.depth and depth_frame:
                    depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
                    return True, np.concatenate((image, depth), axis=-1)
                return True, image
            except Exception as e:
                print(colored(f"Hardware read error: {e}", "red"))
                return False, None
        elif self.mode == 'stream':
            try:
                # Receive multipart message: [topic, data]
                serialized_image = self.subscriber.recv(flags=zmq.NOBLOCK)
                image = pickle.loads(serialized_image)
                if self.dim and (image.shape[1], image.shape[0]) != self.dim:
                    print(colored(f"Resizing image from {image.shape} to {self.dim}", "yellow"))
                    image = cv2.resize(image, self.dim)
                return True, image
            except zmq.Again:
                return False, None  # No message available
            except Exception as e:
                print(colored(f"Stream read error: {e}", "red"))
                return False, None

    def reconnect(self):
        print(colored(f"Reconnecting RealSense {self.name}", "yellow"))
        try:
            self.pipe.stop()
            time.sleep(2)
        except Exception as e:
            print(colored(f"Stop failed: {e}", "yellow"))
        self.initialize_camera()

    def close(self):
        if self.mode == 'hardware':
            try:
                self.pipe.stop()
            except Exception:
                pass
        elif self.mode == 'stream':
            self.subscriber.close()
            
if __name__ == "__main__":
    cam = RSCapture(
        name="front",
        serial_number="239222303782",  # Must match serialized message from stream
        mode="stream",
        stream_host="localhost",
        stream_port=5555
    )

    print("Reading stream. Press 'q' to quit.")
    try:
        while True:
            success, frame = cam.read()
            if success and frame is not None:
                cv2.imshow("Streamed Frame", frame)
            else:
                print("Waiting for frame...")
                time.sleep(0.05)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()