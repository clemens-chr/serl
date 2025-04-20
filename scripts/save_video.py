import os
import shutil
import pickle
import cv2
import numpy as np

# Load the .pkl file
input_file = '/home/clemens/serl/serl/examples/async_drq_sim/franka_lift_cube_image_20_trajs.pkl'
with open(input_file, 'rb') as f:
    data = pickle.load(f)

# Print the number of observations
print(f"Number of observations: {len(data)}")

# Prepare output directories
output_dir = "demo_viz"
subfolder_name = os.path.splitext(os.path.basename(input_file))[0]  # Extract file name without extension
subfolder_path = os.path.join(output_dir, subfolder_name)

# Create clean output directory
if os.path.exists(subfolder_path):
    shutil.rmtree(subfolder_path)  # Remove existing directory
os.makedirs(subfolder_path, exist_ok=True)

# Video writer setup
front_output_file = os.path.join(subfolder_path, "front_camera_video.mp4")
wrist_output_file = os.path.join(subfolder_path, "wrist_camera_video.mp4")
fps = 30  # Frames per second
frame_size = (128, 128)  # Assuming all images are 128x128

# Initialize video writers
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
front_video_writer = cv2.VideoWriter(front_output_file, fourcc, fps, frame_size)
wrist_video_writer = cv2.VideoWriter(wrist_output_file, fourcc, fps, frame_size)

# Iterate through the data and render images
for entry in data:
    if "observations" in entry:
        observations = entry["observations"]
        if 'front' in observations and 'wrist' in observations:
            # Extract images from 'front' and 'wrist'
            front_images = observations['front']  # Shape: (1, 128, 128, 3)
            wrist_images = observations['wrist']  # Shape: (1, 128, 128, 3)

            # Process and write front camera images
            if front_images.shape[0] == 1:  # Remove batch dimension if present
                front_img = front_images[0]
                front_img_bgr = cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR)
                front_video_writer.write(front_img_bgr)

            # Process and write wrist camera images
            if wrist_images.shape[0] == 1:  # Remove batch dimension if present
                wrist_img = wrist_images[0]
                wrist_img_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
                wrist_video_writer.write(wrist_img_bgr)

# Release the video writers
front_video_writer.release()
wrist_video_writer.release()
print(f"Front camera video saved as {front_output_file}")
print(f"Wrist camera video saved as {wrist_output_file}")