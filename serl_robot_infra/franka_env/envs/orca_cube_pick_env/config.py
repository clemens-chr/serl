import numpy as np
from franka_env.envs.franka_env import DefaultEnvConfig


class OrcaCubePickEnvConfig(DefaultEnvConfig):
    """Set the configuration for FrankaEnv."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        "front": "317622074238",
        "side": "239222303782",
    }
    TARGET_POSE = np.array(
        [
            0.4861050577505878,
            0.03539724217748154,
            0.2713914246063738,
            2.8004016182522973,
            -0.3274430279604428,
            0.8126687532359295
        ]
    )
    
    MIN_Z_POS = 0.185
    MAX_Z_POS = 0.23
    

    RESET_HAND_POSE = {'thumb_pip': -9.196288398631111, 'thumb_dip': 39.66277704372641, 'thumb_abd': 7.826453671991196, 'thumb_mcp': 4.087357406937478, 'ring_abd': -6.458179156693582, 'ring_pip': 40.39964322575463, 'ring_mcp': 31.03319460291395, 'middle_mcp': 51.93624519793229, 'middle_pip': 34.44396330497953, 'pinky_mcp': 27.368421519657417, 'pinky_pip': 14.51563707070973, 'pinky_abd': -5.214820014958331, 'middle_abd': -0.27409616187200214, 'index_abd': 83.70515603660898, 'index_mcp': 46.69700921040714, 'index_pip': 26.261932898504583, 'wrist': -29.815731106980543}
    
    HAND_OPEN_POSE = {'thumb_pip': -0.07323472332093672, 'thumb_dip': -1.4772652049629507, 'thumb_abd': 16.49086672702596, 'thumb_mcp': 20.116159870487063, 'ring_abd': -3.156067883662182, 'ring_pip': 0.24977302256021972, 'ring_mcp': 9.8305594731237, 'middle_mcp': -11.58924259254998, 'middle_pip': 23.20393300566721, 'pinky_mcp': 16.919019999970843, 'pinky_pip': -0.332470645470778, 'pinky_abd': -18.35545707938624, 'middle_abd': -0.34196031698605367, 'index_abd': -0.34322899215033686, 'index_mcp': 7.984027968722913, 'index_pip': 27.838385603913423, 'wrist': -26.060729585786774}
    HAND_CLOSE_POSE = {'thumb_pip': 5.9947852089859595, 'thumb_dip': 35.87355686856387, 'thumb_abd': 9.005569366975536, 'thumb_mcp': 28.24708820738351, 'ring_abd': -20.749542188087077, 'ring_pip': 67.67096422079197, 'ring_mcp': 32.527750583069846, 'middle_mcp': 43.302085984599934, 'middle_pip': 64.44900829145166, 'pinky_mcp': 69.32993730937116, 'pinky_pip': -0.22499762491139563, 'pinky_abd': -37.578126754979515, 'middle_abd': -2.804074599285677, 'index_abd': -2.951769332492873, 'index_mcp': 53.84104355978414, 'index_pip': 54.98989824985665, 'wrist': -26.438202156114293}
    
    
    
    RESET_POSE = TARGET_POSE + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    REWARD_THRESHOLD: np.ndarray = np.zeros(6)
    ACTION_SCALE = np.array([0.025, 0.0, 5])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.05
    RANDOM_RZ_RANGE = np.pi / 6
    ABS_POSE_LIMIT_LOW = np.array(
        [
            TARGET_POSE[0] - RANDOM_XY_RANGE,
            TARGET_POSE[1] - RANDOM_XY_RANGE,
            MIN_Z_POS,
            TARGET_POSE[3] - 0.01,
            TARGET_POSE[4] - 0.01,
            TARGET_POSE[5] - RANDOM_RZ_RANGE,
        ]
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [
            TARGET_POSE[0] + RANDOM_XY_RANGE,
            TARGET_POSE[1] + RANDOM_XY_RANGE,
            MAX_Z_POS,
            TARGET_POSE[3] + 0.01,
            TARGET_POSE[4] + 0.01,
            TARGET_POSE[5] + RANDOM_RZ_RANGE,
        ]
    )
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.006,
        "translational_clip_y": 0.006,
        "translational_clip_z": 0.005,
        "translational_clip_neg_x": 0.006,
        "translational_clip_neg_y": 0.006,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.02,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.02,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 3000,
        "translational_damping": 89,
        "rotational_stiffness": 300,
        "rotational_damping": 9,
        "translational_Ki": 0.1,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.05,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.05,
        "rotational_Ki": 0.1,
    }
