import os
import pickle
import numpy as np

# Choose the replacement mode: "uniform" for gray images or "noise" for random noise
replacement_mode = "noise"  # Options: "uniform", "noise"

# Load the .pkl file
input_file = '/home/clemens/serl/serl/examples/async_drq_sim/franka_lift_cube_image_20_trajs.pkl'  # Replace with your .pkl file path
with open(input_file, 'rb') as f:
    data = pickle.load(f)

# Iterate through the data and replace images
for entry in data:
    if "observations" in entry:
        observations = entry["observations"]

        # Replace 'front' images
        if "front" in observations:
            front_shape = observations["front"].shape  # Get the shape of the front images
            if replacement_mode == "uniform":
                observations["front"] = np.full(front_shape, 128, dtype=np.uint8)  # Gray images
            elif replacement_mode == "noise":
                observations["front"] = np.random.randint(0, 256, front_shape, dtype=np.uint8)  # Random noise

        # Replace 'wrist' images
        if "wrist" in observations:
            wrist_shape = observations["wrist"].shape  # Get the shape of the wrist images
            if replacement_mode == "uniform":
                observations["wrist"] = np.full(wrist_shape, 128, dtype=np.uint8)  # Gray images
            elif replacement_mode == "noise":
                observations["wrist"] = np.random.randint(0, 256, wrist_shape, dtype=np.uint8)  # Random noise

# Save the modified data to a new .pkl file
suffix = "_gray_images.pkl" if replacement_mode == "uniform" else "_noise_images.pkl"
output_file = os.path.splitext(input_file)[0] + suffix
with open(output_file, 'wb') as f:
    pickle.dump(data, f)

print(f"Modified data saved to {output_file}")