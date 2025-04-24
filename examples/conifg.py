from abc import abstractmethod
from typing import List

class DefaultTrainingConfig:
    """Default training configuration. """
    agent: str = "drq"
    max_traj_length: int = 1000
    batch_size: int = 256
    critic_actor_ratio: int = 4
    discount: float = 0.97

    max_steps: int = 1000000
    replay_buffer_capacity: int = 200000

    random_steps: int = 300
    training_starts: int = 300
    steps_per_update: int = 30

    log_period: int = 100
    eval_period: int = 2000
    eval_n_trajs: int = 20

    # "resnet" for ResNet10 from scratch and "resnet-pretrained" for frozen ResNet10 with pretrained weights
    encoder_type: str = "resnet-pretrained"
    render: bool = False
    save_video: bool = False
    video_period: int = 10000
    demo_path: str = None
    checkpoint_path: str = None
    reward_classifier_ckpt_path: str = None
    checkpoint_period: int = 0
    buffer_period: int = 0

    eval_checkpoint_step: int = 0
    eval_n_trajs: int = 5
    eval_checkpoint_step: int = 5000

    image_keys: List[str] = None
    classifier_keys: List[str] = None
    proprio_keys: List[str] = None
    
    debug_mode: bool = False
    log_rlds_path: str = None
    preload_rlds_path: str = None
    
    # "single-arm-learned-gripper", "dual-arm-learned-gripper" for with learned gripper, 
    # "single-arm-fixed-gripper", "dual-arm-fixed-gripper" for without learned gripper (i.e. pregrasped)
    setup_mode: str = "single-arm-fixed-gripper"

    @abstractmethod
    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        raise NotImplementedError
    
    @abstractmethod
    def process_demos(self, demo):
        raise NotImplementedError
    
