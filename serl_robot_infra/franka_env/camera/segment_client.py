#!/usr/bin/env python

import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import time
import threading
import queue
from termcolor import colored


class SegmentationClient:
    def __init__(self, server_host="localhost", server_port=8765):
        self.server_host = server_host
        self.server_port = server_port
        self.server_url = f"ws://{server_host}:{server_port}"
        
        # Shared queues for communication
        self.image_queue = queue.Queue(maxsize=2)  # Input images from camera
        self.latest_mask = None
        self.latest_image = None
        self.mask_lock = threading.RLock()
        
        # Threading control
        self.running = False
        self.segmentation_thread = None
        
        print(colored(f"SegmentationClient initialized, server: {self.server_url}", "green"))

    def start(self):
        """Start the continuous segmentation loop"""
        if self.running:
            return
        
        self.running = True
        self.segmentation_thread = threading.Thread(target=self._segmentation_loop, daemon=True)
        self.segmentation_thread.start()
        print(colored("SegmentationClient loop started", "green"))

    def stop(self):
        """Stop the segmentation loop"""
        self.running = False
        if self.segmentation_thread:
            self.segmentation_thread.join(timeout=2.0)
        print(colored("SegmentationClient stopped", "green"))

    def put_image(self, image):
        """Put a new image for segmentation (non-blocking)"""
        try:
            # Remove old image if queue is full and add new one
            if self.image_queue.full():
                try:
                    self.image_queue.get_nowait()
                except queue.Empty:
                    pass
            self.image_queue.put_nowait(image.copy())
        except queue.Full:
            pass  # Skip if queue is full

    def get_latest_mask(self):
        """Get the latest segmentation mask (non-blocking)"""
        with self.mask_lock:
            return self.latest_mask.copy() if self.latest_mask is not None else None

    def get_latest_data(self):
        """Get both latest image and mask (non-blocking)"""
        with self.mask_lock:
            if self.latest_image is not None and self.latest_mask is not None:
                return self.latest_image.copy(), self.latest_mask.copy()
            return None, None

    def encode_image(self, image):
        """Encode numpy image to base64 string"""
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64

    def decode_image(self, image_base64):
        """Decode base64 string to numpy image (grayscale mask)"""
        image_bytes = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
        return image

    async def _get_mask_async(self, image):
        """Get segmentation mask for given image"""
        try:
            async with websockets.connect(self.server_url, ping_timeout=2, close_timeout=2) as websocket:
                # Encode and send image
                image_base64 = self.encode_image(image)
                message = {
                    "type": "image",
                    "image": image_base64,
                    "timestamp": time.time()
                }
                
                await websocket.send(json.dumps(message))
                
                # Wait for response
                response = await websocket.recv()
                response_data = json.loads(response)
                
                if response_data["type"] == "mask":
                    mask = self.decode_image(response_data["mask"])
                    return mask
                elif response_data["type"] == "error":
                    print(colored(f"Server error: {response_data['message']}", "red"))
                    return None
                    
        except Exception as e:
            print(colored(f"Segmentation error: {e}", "yellow"))
            return None

    def _segmentation_loop(self):
        """Main segmentation loop running continuously"""
        print(colored("Segmentation loop started", "green"))
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        frame_count = 0
        
        try:
            while self.running:
                try:
                    # Get latest image from queue (blocking with timeout)
                    image = self.image_queue.get(timeout=0.1)
                    
                    # Process segmentation
                    mask = loop.run_until_complete(self._get_mask_async(image))
                    
                    if mask is not None:
                        # Update latest data
                        with self.mask_lock:
                            self.latest_image = image.copy()
                            self.latest_mask = mask.copy()
                        
                        frame_count += 1
                        if frame_count % 50 == 0:
                            print(colored(f"SegmentationClient: {frame_count} masks processed", "cyan"))
                    
                except queue.Empty:
                    # No new images, continue loop
                    continue
                except Exception as e:
                    print(colored(f"Segmentation loop error: {e}", "red"))
                    time.sleep(0.1)
                
        except Exception as e:
            print(colored(f"Fatal segmentation loop error: {e}", "red"))
        finally:
            loop.close()
            print(colored("Segmentation loop ended", "yellow"))

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()

    def __del__(self):
        """Destructor"""
        self.stop()
