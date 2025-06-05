import gym
from gym.spaces import Dict, Box, flatten_space, flatten
import numpy as np

# Create a nested observation space (this is like your input)
nested_observation_space = Dict({
    "hand_pos": Box(low=-1, high=1, shape=(3,)),      # x,y,z of hand
    "block_pos": Box(low=0, high=2, shape=(3,)),      # x,y,z of block
})

# Create a sample observation (this would be your actual data)
sample_observation = {
    "hand_pos": np.array([0.5, -0.3, 0.7]),    # Some hand position
    "block_pos": np.array([1.0, 1.2, 0.8])     # Some block position
}

# Show the original space and data
print("Original Observation Space:")
print(nested_observation_space)
print("\nOriginal Observation Data:")
print(sample_observation)

# Flatten the space
flat_space = flatten_space(nested_observation_space)
print("\nFlattened Space:")
print(flat_space)  # This will be a single Box with shape (6,)

# Flatten the actual observation
flat_obs = flatten(nested_observation_space, sample_observation)
print("\nFlattened Observation:")
print(flat_obs)  # This will be a 1D array with 6 values 