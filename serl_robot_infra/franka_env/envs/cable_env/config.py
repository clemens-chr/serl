import numpy as np
from franka_env.envs.franka_env import DefaultEnvConfig


class CableEnvConfig(DefaultEnvConfig):
    """Set the configuration for FrankaEnv."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": "317622074238",
        "wrist_2": "239222303782",
    }
    TARGET_POSE = np.array(
        [
            0.460639895728905,
            -0.02439473272513422,
            0.026321125814908725,
            3.1331234,
            0.0182487,
            1.5824805,
        ]
    )
    RESET_POSE = TARGET_POSE + np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    REWARD_THRESHOLD: np.ndarray = np.zeros(6)
    APPLY_GRIPPER_PENALTY = False
    ACTION_SCALE = np.array([0.05, 0.3, 1])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.1
    RANDOM_RZ_RANGE = np.pi / 6
    ABS_POSE_LIMIT_LOW = np.array(
        [
            TARGET_POSE[0] - RANDOM_XY_RANGE,
            TARGET_POSE[1] - RANDOM_XY_RANGE,
            TARGET_POSE[2] - 0.001,
            TARGET_POSE[3] - 0.01,
            TARGET_POSE[4] - 0.01,
            TARGET_POSE[5] - RANDOM_RZ_RANGE,
        ]
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [
            TARGET_POSE[0] + RANDOM_XY_RANGE,
            TARGET_POSE[1] + RANDOM_XY_RANGE,
            TARGET_POSE[2] + 0.1,
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
        "translational_clip_x": 0.005,
        "translational_clip_y": 0.005,
        "translational_clip_z": 0.005,
        "translational_clip_neg_x": 0.005,
        "translational_clip_neg_y": 0.005,
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
