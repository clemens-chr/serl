import os
import shutil
import pickle
import cv2
import numpy as np
import json

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

# Function to recursively extract structure and dimensions
def extract_structure(data):
    if isinstance(data, dict):
        return {key: extract_structure(value) for key, value in data.items()}
    elif isinstance(data, (list, np.ndarray)):
        return str(np.array(data).shape)
    else:
        return str(type(data))

# Save the keys and values of the first entry to a JSON file in the same folder as the demo video
if len(data) > 0:
    first_entry = data[0]
    first_entry_summary = {key: extract_structure(value) for key, value in first_entry.items()}
    json_output_file = os.path.join(subfolder_path, "first_entry_summary.json")
    with open(json_output_file, "w") as json_file:
        json.dump(first_entry_summary, json_file, indent=4)
    print(f"First entry summary saved to {json_output_file}")

# Video writer setup
front_output_file = os.path.join(subfolder_path, "front_camera_video.mp4")
wrist_output_file = os.path.join(subfolder_path, "wrist_camera_video.mp4")
fps = 30  # Frames per second
frame_size = (128, 128)  # Assuming all images are 128x128


frames = []
# Iterate through the data and render images
for entry in data:

    if 'actions' in entry:
        print(f"Actions: {entry['actions']}")

    if "observations" in entry:
        observations = entry["observations"]

        if "rewards" in entry:
            print(f"Rewards: {entry['rewards']}")
      
        if 'front' in observations and 'wrist' in observations:
            # Extract images from 'front' and 'wrist'
            front_images = observations['front']  # Shape: (1, 128, 128, 3)
            wrist_images = observations['wrist']  # Shape: (1, 128, 128, 3)

            # Process and write front camera images
            if front_images.shape[0] == 1:  # Remove batch dimension if present
                front_img = front_images[0]
                front_img_bgr = cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR)

            # Process and write wrist camera images
            if wrist_images.shape[0] == 1:  # Remove batch dimension if present
                wrist_img = wrist_images[0]
                wrist_img_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)

            frames.append(np.concatenate((front_img, wrist_img), axis=0))


import imageio

imageio.mimsave(front_output_file, frames, fps=fps)
