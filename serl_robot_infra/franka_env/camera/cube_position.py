import zmq
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

class CubePosition:
    def __init__(self):
        self.context = zmq.Context()
        self.cube_pose = None  # [x, y, z, qx, qy, qz, qw]
        self.rel_cube_pose = None
        self.pose_sub_socket = self.context.socket(zmq.SUB)
        self.pose_sub_socket.setsockopt(zmq.RCVHWM, 1)
        self.pose_sub_socket.setsockopt(zmq.CONFLATE, 1)
        self.pose_sub_socket.connect(f"tcp://localhost:5556")
        self.pose_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # For relative rotation calculation
        self.initial_rot = None
        self.principal_axes_transform = None
        self.initial_rot_in_principal = None
        self.dominant_axis_index = None
        self.dominant_axis_rotation = None
        self.dominant_axis_label = None
        self.relative_euler = None
        self.secondary_axis_rotation = None
        self.tertiary_axis_rotation = None

    def get_cube_pose(self):
        CAMERA_TO_WORLD = np.array([[ 0.1966638,   0.9804636,  -0.00365842,  0.08884098],
                                [ 0.70710435, -0.13924596,  0.69326191, -0.17268835],
                                [ 0.67920955, -0.1389266,  -0.72067463,  0.06658885],
                                [ 0.,          0.,          0.,          1.        ]])
        
        CAMERA_TO_WORLD_ROT = R.from_matrix(CAMERA_TO_WORLD[:3, :3])
        WORLD_TO_CAMERA_ROT = CAMERA_TO_WORLD_ROT.inv()
        
        """Get the current cube pose and calculate relative rotation"""
        try:
            serialized_data = self.pose_sub_socket.recv(flags=zmq.NOBLOCK)
            data = pickle.loads(serialized_data)
            cube_pose_world = data['cube_pose']
            
            if cube_pose_world is not None:
                # Extract position and quaternion from 4x4 matrix
                rot_mat = np.array(cube_pose_world)[:3, :3]
                current_rot = R.from_matrix(rot_mat)
                position = np.array(cube_pose_world)[:3, 3]

                # Transform current rot to camera-aligned frame
                current_rot_camera = WORLD_TO_CAMERA_ROT * current_rot
                euler = current_rot_camera.as_euler('xyz', degrees=True)

                
                self.relative_euler = euler 
                self.cube_pose = np.concatenate([position, current_rot_camera.as_quat()])
                
                return self.cube_pose
                
                # Store the pose [x, y, z, qx, qy, qz, qw]
                self.cube_pose = np.concatenate([position, quaternion])
                
                # Calculate relative rotation (like the display does)
                current_rot = R.from_matrix(rotation_matrix)
                
                if self.initial_rot is None:
                    # Set initial reference frame
                    self.initial_rot = current_rot
                    initial_rotation_matrix = self.initial_rot.as_matrix()
                    
                    # Calculate principal axes transform
                    x_axis = initial_rotation_matrix[:, 0]
                    y_axis = initial_rotation_matrix[:, 1]
                    z_axis = initial_rotation_matrix[:, 2]
                    self.principal_axes_transform = np.linalg.inv(np.array([x_axis, y_axis, z_axis]).T)
                    self.initial_rot_in_principal = R.from_matrix(np.eye(3))
                    
                    # Find dominant axis (most aligned with camera view)
                    camera_view_axis = np.array([0, 0, 1])  
                    dot_products = np.abs(camera_view_axis @ initial_rotation_matrix)
                    self.dominant_axis_index = np.argmax(dot_products)
                
                # Calculate relative rotation
                if self.principal_axes_transform is not None:
                    transformed_rot_mat = self.principal_axes_transform @ current_rot.as_matrix()
                    transformed_rot = R.from_matrix(transformed_rot_mat)
                    rel_rot = transformed_rot * self.initial_rot_in_principal.inv()
                    self.rel_cube_pose = np.concatenate([position, rel_rot.as_quat()])
                    
                    # Get Euler angles
                    self.relative_euler = rel_rot.as_euler('xyz', degrees=True)
                    
                    # Get dominant axis rotation
                    if self.dominant_axis_index is not None:
                        self.dominant_axis_rotation = self.relative_euler[self.dominant_axis_index]
                        self.secondary_axis_rotation = self.relative_euler[(self.dominant_axis_index + 1) % 3]
                        self.tertiary_axis_rotation = self.relative_euler[(self.dominant_axis_index + 2) % 3]
                        
                    axis_labels = ['X (Roll)', 'Y (Pitch)', 'Z (Yaw)']
                    self.dominant_axis_label = axis_labels[self.dominant_axis_index] if self.dominant_axis_index is not None else "N/A"
                
                return self.rel_cube_pose
            else:
                return self.rel_cube_pose
                
        except zmq.Again:
            return self.cube_pose

    def get_relative_euler(self):
        """Get the relative Euler angles (like the display shows)"""
        return self.relative_euler
    
    def get_rotation_around_z(self):
        """Get the rotation around the z axis"""
        return self.relative_euler[2]

    def get_dominant_axis_rotation(self):
        """Get the rotation around the dominant axis"""
        return self.dominant_axis_rotation
    
    def get_secondary_axis_rotation(self):
        """Get the rotation around the secondary axis"""
        return self.secondary_axis_rotation
    
    def get_tertiary_axis_rotation(self):
        """Get the rotation around the tertiary axis"""
        return self.tertiary_axis_rotation

    def get_dominant_axis_label(self):
        """Get the label of the dominant axis"""
        return self.dominant_axis_label

    def reset_reference(self):
        """Reset the reference frame to current pose"""
        self.initial_rot = None
        self.principal_axes_transform = None
        self.initial_rot_in_principal = None
        self.dominant_axis_index = None
        self.dominant_axis_rotation = None
        self.dominant_axis_label = None
        self.relative_euler = None

if __name__ == "__main__":
    cube_position = CubePosition()
    while True:
        pose = cube_position.get_cube_pose()
        relative_euler = cube_position.get_relative_euler()
        dominant_rotation = cube_position.get_dominant_axis_rotation()
        dominant_label = cube_position.get_dominant_axis_label()
        
        if relative_euler is not None:
            print(f"Pose: {pose}")
            print(f"Relative Euler: Roll={relative_euler[0]:.1f}°, Pitch={relative_euler[1]:.1f}°, Yaw={relative_euler[2]:.1f}°")
            print(f"Rotation around z: {cube_position.get_rotation_around_z():.1f}°")
            print("---")
        
        time.sleep(0.1)