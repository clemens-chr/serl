import cv2
import threading
import queue
import time
import websocket
import json
import base64
import numpy as np
from termcolor import colored
from websocket import create_connection


class VideoCapture:
    def __init__(self, cap, name="Camera", server_url_segment=None):
        self.name = name
        self.cap = cap
        
        if server_url_segment is not None:
            self.server_url = server_url_segment
            self.segmentation_enabled = True
        else:
            self.segmentation_enabled = False

        self.rgb_q = queue.Queue(maxsize=1)

        self.latest_frame = None
        self.frame_lock = threading.Lock()

        self.enable = True

        self.rgb_t = threading.Thread(target=self._reader, daemon=True)
        self.rgb_t.start()
        
        if self.segmentation_enabled:
            self.segmentation_q = queue.Queue(maxsize=3)
            self.segmentation_t = threading.Thread(target=self._segmentation_reader, daemon=True)
            self.segmentation_t.start()

    def _reader(self):
        while self.enable:
            ret, frame = self.cap.read()
            if not ret:
                continue

            with self.frame_lock:
                self.latest_frame = frame

            if not self.rgb_q.full():
                self.rgb_q.put(frame)
            else:
                try:
                    self.rgb_q.get_nowait()
                    self.rgb_q.put(frame)
                except queue.Empty:
                    pass

            time.sleep(0.01)

    def _segmentation_reader(self):
        ws = None
        while self.enable:
            try:
                if ws is None or not ws.connected:
                    print(colored("Connecting to WebSocket...", "cyan"))
                    ws = create_connection(self.server_url, timeout=3)
                    print(colored("Connected to WebSocket", "green"))

                with self.frame_lock:
                    frame = self.latest_frame.copy() if self.latest_frame is not None else None

                if frame is None:
                    time.sleep(0.01)
                    continue

                image_base64 = self.encode_image(frame)
                message = {
                    "type": "image",
                    "image": image_base64,
                    "timestamp": time.time()
                }

                ws.send(json.dumps(message))
                response = ws.recv()
                response_data = json.loads(response)

                if response_data["type"] == "mask":
                    mask = self.decode_image(response_data["mask"])
                    if self.segmentation_q.full():
                        self.segmentation_q.get_nowait()
                    self.segmentation_q.put(mask)
                elif response_data["type"] == "error":
                    print(colored(f"Server error: {response_data['message']}", "red"))

            except Exception as e:
                print(colored(f"WebSocket error: {e}", "yellow"))
                time.sleep(1)
                try:
                    if ws:
                        ws.close()
                except:
                    pass
                ws = None

    def encode_image(self, frame):
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def decode_image(self, b64_string):
        img_data = base64.b64decode(b64_string)
        np_arr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

    def read(self):
        return self.rgb_q.get(timeout=5)

    def read_segmentation(self):
        if self.segmentation_enabled:
            return self.segmentation_q.get(timeout=5)
        else:
            return None

    def close(self):
        self.enable = False
        self.rgb_t.join()
        self.segmentation_t.join()
        self.cap.release()


# --- Example usage ---

if __name__ == "__main__":
    from rs_capture import RSCapture
    
    cap = RSCapture(
        name="front",
        serial_number="239222303782",  # Must match serialized message from stream
        mode="stream",
        stream_host="localhost",
        stream_port=5555
    )

    video = VideoCapture(cap, name="front", server_url_segment="ws://localhost:8765")

    try:
        while True:
            frame = video.read()
            cv2.imshow("RGB", frame)

            try:
                mask = video.read_segmentation()
                cv2.imshow("Segmentation", mask)
            except queue.Empty:
                pass

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        video.close()
        cv2.destroyAllWindows()
