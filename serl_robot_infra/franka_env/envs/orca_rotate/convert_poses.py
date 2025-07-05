import numpy as np
from scipy.spatial.transform import Rotation

# Define the euler pose
pose_euler = np.array([
    0.4861050577505878,
    0.03539724217748154,
    0.2713914246063738,
    2.8004016182522973,
    -0.3274430279604428,
    0.8126687532359295
])

print("Original Euler pose:")
print(pose_euler)

# Convert euler to quaternion
position = pose_euler[:3]
euler_angles = pose_euler[3:]
quaternion = Rotation.from_euler('xyz', euler_angles).as_quat()

pose_quat = np.concatenate([position, quaternion])
print("\nConverted to quaternion pose:")
print(pose_quat)

# Convert back from quaternion to euler
position_back = pose_quat[:3]
quat_back = pose_quat[3:]
euler_back = Rotation.from_quat(quat_back).as_euler('xyz')

pose_euler_back = np.concatenate([position_back, euler_back])
print("\nConverted back to euler pose:")
print(pose_euler_back)

# Verify the conversion is correct
print("\nVerification - difference between original and converted back:")
print(np.abs(pose_euler - pose_euler_back))
