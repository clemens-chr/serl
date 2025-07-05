import numpy as np
from franka_env.envs.franka_env import DefaultEnvConfig


class OrcaRotateEnvConfig(DefaultEnvConfig):
    """Set the configuration for FrankaEnv."""

    # Hand config
    RESET_HAND_POSE = {'thumb_pip': 20.773105094940348, 'thumb_dip': 31.40728207796993, 'thumb_abd': 35.96199416203551, 'thumb_mcp': 5.399575049258402, 'ring_abd': -28.399545542470108, 'ring_pip': 21.508426767720735, 'ring_mcp': 26.35164001828454, 'middle_mcp': 31.596008656804337, 'middle_pip': 28.30536784317644, 'pinky_mcp': 33.951608498361864, 'pinky_pip': 22.138601428625293, 'pinky_abd': -32.17917591452549, 'middle_abd': -1.2058453041060133, 'index_abd': 15.105793670897135, 'index_mcp': 43.92458249316778, 'index_pip': 14.91901590472439, 'wrist': -22.444806064456465}
    
    MIN_HAND_POSE = {
    "thumb_pip": 18.857133453293358,
    "thumb_dip": 30.99766103222779,
    "thumb_abd": 3.8574857373506646,
    "thumb_mcp": 5.507566718088874,
    "ring_abd": -37.358354589772674,
    "ring_pip": 20.38353431056356,
    "ring_mcp": 20.696314026920007,
    "middle_mcp": 30.962926985010156,
    "middle_pip": 28.191974671318306,
    "pinky_mcp": 24.670699291803174,
    "pinky_pip": 21.927379710607326,
    "pinky_abd": -36.15634922885008,
    "middle_abd": -13.137230202737342,
    "index_abd": 7.766928071463546,
    "index_mcp": 44.88826820291149,
    "index_pip": 15.79198515425368,
    "wrist": -23.18653618324648
    }
    
    MAX_HAND_POSE = {
        "thumb_pip": 21.277308158531667,
        "thumb_dip": 31.304876816534396,
        "thumb_abd": 35.66745654752822,
        "thumb_mcp": 49.784016662339056,
        "ring_abd": -21.59080833678931,
        "ring_pip": 35.90699776864304,
        "ring_mcp": 25.797192914246537,
        "middle_mcp": 46.262351521895724,
        "middle_pip": 31.480359031148794,
        "pinky_mcp": 45.98618291464253,
        "pinky_pip": 22.138601428625293,
        "pinky_abd": -28.081443597511583,
        "middle_abd": -1.3327556224175297,
        "index_abd": 16.818195644098303,
        "index_mcp": 62.12756136980142,
        "index_pip": 17.8653072621767,
        "wrist": -22.39844793203209
    }
    
    ACTIVE_DOF = ['thumb_abd', 'thumb_mcp', 'ring_abd', 'ring_mcp', 'middle_mcp', 'pinky_mcp', 'pinky_abd', 'middle_abd', 'index_abd', 'index_mcp']
    ACTION_SPACE_NUM_HAND = len(ACTIVE_DOF)
        
    # Calculate action scaling for each active DOF based on 50% ROM on each side
    ACTION_SCALE_PER_DOF = {}
    for dof in ACTIVE_DOF:
        min_val = MIN_HAND_POSE[dof]
        max_val = MAX_HAND_POSE[dof]
        rom = max_val - min_val
        # Scale so that -1 moves 50% ROM to lower side, +1 moves 50% ROM to upper side
        # This means the total range covered by [-1, 1] is 100% of ROM
        # So scale = rom / 2 (since action range is 2 from -1 to 1)
        scale = rom / 2.0
        ACTION_SCALE_PER_DOF[dof] = scale
     
    HAND_OPEN_POSE = {'thumb_pip': -0.07323472332093672, 'thumb_dip': -1.4772652049629507, 'thumb_abd': 16.49086672702596, 'thumb_mcp': 20.116159870487063, 'ring_abd': -3.156067883662182, 'ring_pip': 0.24977302256021972, 'ring_mcp': 9.8305594731237, 'middle_mcp': -11.58924259254998, 'middle_pip': 23.20393300566721, 'pinky_mcp': 16.919019999970843, 'pinky_pip': -0.332470645470778, 'pinky_abd': -18.35545707938624, 'middle_abd': -0.34196031698605367, 'index_abd': -0.34322899215033686, 'index_mcp': 7.984027968722913, 'index_pip': 27.838385603913423, 'wrist': -26.060729585786774}
    HAND_CLOSE_POSE = {'thumb_pip': 5.9947852089859595, 'thumb_dip': 35.87355686856387, 'thumb_abd': 9.005569366975536, 'thumb_mcp': 28.24708820738351, 'ring_abd': -20.749542188087077, 'ring_pip': 67.67096422079197, 'ring_mcp': 32.527750583069846, 'middle_mcp': 43.302085984599934, 'middle_pip': 64.44900829145166, 'pinky_mcp': 69.32993730937116, 'pinky_pip': -0.22499762491139563, 'pinky_abd': -37.578126754979515, 'middle_abd': -2.804074599285677, 'index_abd': -2.951769332492873, 'index_mcp': 53.84104355978414, 'index_pip': 54.98989824985665, 'wrist': -26.438202156114293}
    
    # TCP config
    SERVER_URL: str = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        "side": "317622074238",
        "front": "239222303782",
    }
    PORTS = {
        "side": 5554,
        "front": 5555,
    }
    TARGET_POSE = np.array(          
        [
            0.4728129276586676,
            -0.11857788225081389,
            0.12026326395724704,
            -1.8832374992292595,
            0.7410905257263329,
            -0.8361366787667226
        ]
    )
    
    MIN_Z_POS = 0.215
    MAX_Z_POS = 0.3
    MIN_X_POS = 0.43
    MAX_X_POS = 0.52
    MIN_Y_POS = -0.04
    MAX_Y_POS = 0.04
    
    ACTION_SPACE_NUM_TCP_FRANKA = 0
    
    
    RESET_POSE = TARGET_POSE + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    REWARD_THRESHOLD: np.ndarray = np.zeros(6)
    ACTION_SCALE = np.array([0.02, 0.06, 0.5])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.05
    RANDOM_RZ_RANGE = np.pi / 4
    ABS_POSE_LIMIT_LOW = np.array(
        [
            MIN_X_POS,
            MIN_Y_POS,
            MIN_Z_POS,
            TARGET_POSE[3] - 0.005,
            TARGET_POSE[4] - 0.005,
            TARGET_POSE[5] - RANDOM_RZ_RANGE,
        ]
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [
            MAX_X_POS,
            MAX_Y_POS,
            MAX_Z_POS,
            TARGET_POSE[3] + 0.005,
            TARGET_POSE[4] + 0.005,
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
