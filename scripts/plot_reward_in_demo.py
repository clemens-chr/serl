import os
import pickle
import matplotlib.pyplot as plt

# Load the .pkl file
input_file = '/home/clemens/serl/serl/examples/async_drq_sim/franka_lift_cube_image_20_trajs.pkl'
with open(input_file, 'rb') as f:
    data = pickle.load(f)

# Extract rewards and actions
rewards = []
actions = []

for entry in data:
    if "rewards" in entry:
        rewards.append(entry["rewards"])
    if "actions" in entry:
        actions_all = entry["actions"]
        action = actions_all[-1]
        actions.append(action)

# Check if rewards and actions were found
if not rewards:
    print("No rewards found in the data.")
    exit()
if not actions:
    print("No actions found in the data.")
    exit()

# Convert actions to a NumPy array for easier slicing
import numpy as np
actions = np.array(actions)

# Plot rewards and the last action
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot rewards on the primary y-axis
ax1.plot(rewards, label="Rewards", color="blue", marker="o")
ax1.set_xlabel("Time Step")
ax1.set_ylabel("Reward", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.grid(True)

# Plot the last action on the secondary y-axis
ax2 = ax1.twinx()
ax2.plot(actions, label="Last Action", color="orange", marker="x")
ax2.set_ylabel("Last Action", color="orange")
ax2.tick_params(axis="y", labelcolor="orange")



# Add legends
fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

# Save the plot
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "reward_and_last_action_plot.png")
plt.title("Reward and Last Action Over Time")
plt.savefig(output_file)
print(f"Reward and last action plot saved to {output_file}")

# Show the plot
plt.show()