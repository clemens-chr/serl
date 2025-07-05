#!/usr/bin/env python3

import pickle
import numpy as np
import time
from orca_core import OrcaHand
from franka_env.spacemouse.retargeter_client import RetargeterClient
from franka_env.envs.orca_rotate.config import OrcaRotateEnvConfig

def process_pickle_file(pickle_path: str):
    """
    Process a pickle file containing raw AVP data through the retargeter
    and apply inverse action scaling.
    """
    
    # Load the pickle file
    print(f"Loading pickle file: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        transitions = pickle.load(f)
    
    print(f"Loaded {len(transitions)} transitions")
    
    # Initialize the hand and retargeter
    hand = OrcaHand("/home/ccc/orca_ws/src/orca_configs/orcahand_v1_right_clemens_stanford")
    hand.connect()
    retargeter = RetargeterClient()
    
    # Initialize retargeter
    init_response = retargeter.initialize_retargeter(
        model_path="/home/ccc/orca_ws/src/orca_configs/orcahand_v1_right_clemens_stanford",
        urdf_path="/home/ccc/orca_ws/src/orca_ros/orcahand_description/models/urdf/orcahand_right.urdf"
    )
    
    print(f"Retargeter initialization response: {init_response}")
    if init_response['status'] != 'success':
        print("Failed to initialize retargeter!")
        return
    
    time.sleep(1)
    
    # Get config for active DOFs and action scaling
    config = OrcaRotateEnvConfig()
    active_dofs = config.ACTIVE_DOF
    action_scale_per_dof = config.ACTION_SCALE_PER_DOF
    min_hand_pose = config.MIN_HAND_POSE
    max_hand_pose = config.MAX_HAND_POSE
    
    print(f"Active DOFs: {active_dofs}")
    print(f"Action scaling per DOF: {action_scale_per_dof}")
    
    all_angles = []
    # Process each transition
    for i, transition in enumerate(transitions):
        print(f"\n--- Processing transition {i+1}/{len(transitions)} ---")
        
        # Check if raw_data exists in the transition
        if 'raw_data' not in transition:
            print(f"No raw_data found in transition {i+1}")
            continue
            
        raw_data = transition['raw_data']
        
        # Run through retargeter
        retarget_response = retargeter.retarget(raw_data)
        
        if retarget_response['status'] == 'success':
            target_angles = retarget_response['target_angles']
            
            # Convert to degrees
            target_angles_deg = np.array([np.rad2deg(target_angles[joint_id]) for joint_id in hand.joint_ids])
            
            # Extract only active DOF angles
            active_angles = {}
            for dof in active_dofs:
                if dof in target_angles:
                    active_angles[dof] = np.rad2deg(target_angles[dof])
                else:
                    print(f"Warning: DOF {dof} not found in target angles")
            
            all_angles.append(active_angles)
            print(f"Active angles: {active_angles}")
            hand.set_joint_pos(active_angles)
            
            # Apply inverse action scaling
            # The forward scaling is: new_angle = current_angle + action * scale
            # So inverse is: action = (target_angle - current_angle) / scale
            
            # For this example, let's assume current angles are at the middle of the range
            # You might want to use actual current angles from the transition if available
            current_angles = {}
            for dof in active_dofs:
                current_angles[dof] = (min_hand_pose[dof] + max_hand_pose[dof]) / 2.0
            
            print(f"Current angles (middle of range): {current_angles}")
            
            # Calculate inverse scaled actions
            inverse_actions = {}
            action_array = []
            
            for dof in active_dofs:
                if dof in active_angles:
                    # Calculate the action needed to reach target from current
                    angle_diff = active_angles[dof] - current_angles[dof]
                    action = angle_diff / action_scale_per_dof[dof]
                    
                    # Clip to [-1, 1] range
                    action = np.clip(action, -1.0, 1.0)
                    
                    inverse_actions[dof] = action
                    action_array.append(action)
                else:
                    inverse_actions[dof] = 0.0
                    action_array.append(0.0)
            

        
        else:
            print(f"Retargeting failed for transition {i+1}: {retarget_response['message']}")
        
        # Add a small delay to avoid overwhelming the retargeter
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    # Default pickle file path
    pickle_path = "/home/ccc/orca_ws/src/serl/examples/orca_rotate_drq/orca_cube_pick_1_demos_2025-07-04_21-22-56.pkl"
    
    # You can change this path or pass it as a command line argument
    import sys
    if len(sys.argv) > 1:
        pickle_path = sys.argv[1]
    
    process_pickle_file(pickle_path)
