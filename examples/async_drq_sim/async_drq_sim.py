#!/usr/bin/env python3

import time
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import cv2
import os
import threading

from typing import Any, Dict, Optional
import pickle as pkl
import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gym.wrappers.record_video import RecordVideo


from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.common.evaluation import evaluate
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.utils.launcher import (
    make_drq_agent,
    make_trainer_config,
    make_wandb_logger,
    make_replay_buffer,
)
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

import franka_sim

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "PandaPickCubeVision-v0", "Name of environment.")
flags.DEFINE_string("agent", "drq", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 1000, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 128, "Batch size.")
flags.DEFINE_integer("critic_actor_ratio", 4, "critic to actor update ratio.")
flags.DEFINE_float("discount", 0.97, "Discount factor.")

flags.DEFINE_integer("max_steps", 1000000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 200000, "Replay buffer capacity.")

flags.DEFINE_integer("random_steps", 300, "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 300, "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 30, "Number of steps per update the server.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 2000, "Evaluation period.")
flags.DEFINE_integer("eval_n_trajs", 20, "Number of trajectories for evaluation.")

# flag to indicate if this is a leaner or a actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("render", False, "Render the environment.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
# "small" is a 4 layer convnet, "resnet" and "mobilenet" are frozen with pretrained weights
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
flags.DEFINE_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_integer("checkpoint_period", 0, "Period to save checkpoints.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_string(
    "reward_classifier_ckpt_path", None, "Path to reward classifier ckpt."
)
flags.DEFINE_integer(
    "eval_checkpoint_step", 0, "evaluate the policy from ckpt at this step"
)
flags.DEFINE_boolean(
    "save_video", False, "Save video of training"
)
flags.DEFINE_integer(
    "video_period", 10000, "Period to save video of training"
)

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

flags.DEFINE_string("log_rlds_path", None, "Path to save RLDS logs.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)

video_dir = None

def print_green(x):
    return print("\033[92m {}\033[00m".format(x))
 
##############################################################################

def actor(agent: DrQAgent, data_store, env, sampling_rng, video_dir=None):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """

    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            FLAGS.checkpoint_path,
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()

            print("Returned obs keys:", obs.keys())
            print("Expected space:", env.observation_space)
            print("Is valid?", env.observation_space.contains(obs))

            done = False
            start_time = time.time()
            while not done:
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=True,
                )
                actions = np.asarray(jax.device_get(actions))

                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs

                if done:
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)

                    if reward > 0.7:
                        success_counter += 1

        print_green(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print_green(f"average time: {np.mean(time_list)}")
        return  # after done eval, return and exit
    

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_store,
        wait_for_server=True,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))
        print_green("updated agent params")


    client.recv_network_callback(update_params)

    obs, _ = env.reset()
    done = False

    success_counter = 0

    # training loop
    timer = Timer()
    running_return = 0.0
    num_episodes = 0
    recording = False

    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True):
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < FLAGS.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    deterministic=False,
                )
                actions = np.asarray(jax.device_get(actions))


        with timer.context("step_env"):

            # Start video recording if the step matches the video period
            if step > 100 and step % FLAGS.video_period == 0 and FLAGS.save_video:
                env.start_video_recorder()
                print_green("Started video recording")
                recording = True

            next_obs, reward, done, truncated, info = env.step(actions)
            reward = np.asarray(reward, dtype=np.float32)
            info = np.asarray(info)
            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done or truncated,
            )
            data_store.insert(transition)

            obs = next_obs
            if done:
                print("Reset because: ", done)
                num_episodes += 1
                if reward:
                    success_counter += reward
                obs, _ = env.reset()

                # Close video recording if the environment is done
                if hasattr(env, "video_recorder") and FLAGS.save_video and recording:
                    env.close_video_recorder()
                    video_paths = [
                        os.path.join(video_dir, f)
                        for f in os.listdir(video_dir)
                        if f.endswith(".mp4")
                    ]
                    video_paths = sorted(video_paths, key=os.path.getctime)
                    video_path = video_paths[-1]
                    print_green(f"Sending video to server: {video_path}")
                    client.request("send-stats", {"video_path": video_path})
                    print_green("Closed and sent video data")
                    recording = False

        if step % FLAGS.steps_per_update == 0:
            client.update()
           
        if num_episodes % FLAGS.eval_n_trajs == 0 and num_episodes > 0:
            avg_running_return = running_return / FLAGS.eval_n_trajs
            success_rate = success_counter / FLAGS.eval_n_trajs
            print_green(f"success rate: {success_rate}")
            client.request("send-stats", {
                "success_rate": success_rate,
                "avg_running_return": avg_running_return,
            })
            success_counter = 0
            running_return = 0.0
            num_episodes = 0

        timer.tock("total")

        if step % FLAGS.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)

##############################################################################

def learner(
    rng,
    agent: DrQAgent,
    replay_buffer: MemoryEfficientReplayBufferDataStore,
    demo_buffer: Optional[MemoryEfficientReplayBufferDataStore] = None,
):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    # set up wandb and logging
    wandb_logger = make_wandb_logger(
        project="serl_dev",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )

    # To track the step in the training loop
    update_steps = 0

    PAUSE_TRAINING = threading.Event()
    PAUSE_TRAINING.clear()

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats or video requests."""
        assert type == "send-stats" or type == "pause-training" 

        if type == "pause-training":
            if payload.get("pause", False):
                print_green("Pausing training...")
                PAUSE_TRAINING.set()
            else:
                print_green("Resuming training...")
                PAUSE_TRAINING.clear()
            return {}
        
        
        if wandb_logger is not None:
            if "video_path" in payload:  # Handle a single video path
                video_path = payload["video_path"]
                print_green(f"Received video path: {video_path}")
                wandb_logger.log({"video": video_path}, step=update_steps)
            else:
                wandb_logger.log(payload, step=update_steps)
        return {} 
    
    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)


    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience if
    # demo_buffer is provided
    if demo_buffer is None:
        single_buffer_batch_size = FLAGS.batch_size
        demo_iterator = None
    else:
        single_buffer_batch_size = FLAGS.batch_size // 2
        demo_iterator = demo_buffer.get_iterator(
            sample_args={
                "batch_size": single_buffer_batch_size,
                "pack_obs_and_next_obs": True,
            },
            device=sharding.replicate(),
        )

    # create replay buffer iterator
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": single_buffer_batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()

    # show replay buffer progress bar during training
    pbar = tqdm.tqdm(
        total=FLAGS.replay_buffer_capacity,
        initial=len(replay_buffer),
        desc="replay buffer",
    )

    print_green(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting training...")

    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU


        while PAUSE_TRAINING.is_set():
            time.sleep(1)
            print_green("Waiting for actor to resume training...")

        for critic_step in range(FLAGS.critic_actor_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)

                # we will concatenate the demo data with the online data
                # if demo_buffer is provided
                if demo_iterator is not None:
                    demo_batch = next(demo_iterator)
                    batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update_critics(
                    batch,
                )

        with timer.context("train"):
            batch = next(replay_iterator)

            # we will concatenate the demo data with the online data
            # if demo_buffer is provided
            if demo_iterator is not None:
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update_high_utd(batch, utd_ratio=1)

        # publish the updated network
        if step > 0 and step % (FLAGS.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)
            print_green("sent updated network to actor")

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)

        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0 and update_steps > 0:
            assert FLAGS.checkpoint_path is not None
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path, agent.state, step=update_steps, keep=20
            )
            print_green(f"Checkpoint saved at step {update_steps} to {FLAGS.checkpoint_path}")
            

        pbar.update(len(replay_buffer) - pbar.n)  # update replay buffer bar
        update_steps += 1


##############################################################################


def main(_):
    assert FLAGS.batch_size % num_devices == 0

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)

    # create env and load dataset
    if FLAGS.render:
        env = gym.make(FLAGS.env, render_mode="human")
    else:
        env = gym.make(FLAGS.env, render_mode="rgb_array", reward_type='sparse')

    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    video_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "exp",
        FLAGS.exp_name or FLAGS.env,
        f"{timestamp}",
        "videos"
    )

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print(f"Image observations enabled: {env.image_obs}")

    if FLAGS.save_video:
        assert not FLAGS.render, "Cannot save video when render is True"
        print_green(f"Logging videos to {video_dir}")
        env = RecordVideo(
            env,
            video_dir,
            episode_trigger=lambda x: x % FLAGS.video_period == 0,
            name_prefix="video",
        )
                     
    image_keys = [key for key in env.observation_space.keys() if key != "state"]

    rng, sampling_rng = jax.random.split(rng)
    agent: DrQAgent = make_drq_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=image_keys,
        encoder_type=FLAGS.encoder_type,
        discount=FLAGS.discount,
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent: DrQAgent = jax.device_put(
        jax.tree.map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.checkpoint_path is None and FLAGS.checkpoint_period > 0:
        FLAGS.checkpoint_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "exp", FLAGS.exp_name, f"{timestamp}", "checkpoints"
            )
        if FLAGS.checkpoint_path:
            os.makedirs(FLAGS.checkpoint_path, exist_ok=True)
            print_green(f"Checkpoints will be saved to {FLAGS.checkpoint_path}")

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer = make_replay_buffer(
            env,
            capacity=FLAGS.replay_buffer_capacity,
            rlds_logger_path=FLAGS.log_rlds_path,
            type="memory_efficient_replay_buffer",
            image_keys=image_keys,
        )

        print_green("replay buffer created")
        print_green(f"replay_buffer size: {len(replay_buffer)}")

        # if demo data is provided, load it into the demo buffer
        # in the learner node, we support 2 ways to load demo data:
        # 1. load from pickle file; 2. load from tf rlds data
        if FLAGS.demo_path or FLAGS.preload_rlds_path:

            def preload_data_transform(data, metadata) -> Optional[Dict[str, Any]]:
                # NOTE: Create your own custom data transform function here if you
                # are loading this via with --preload_rlds_path with tf rlds data
                # This default does nothing
                return data

            demo_buffer = make_replay_buffer(
                env,
                capacity=FLAGS.replay_buffer_capacity,
                type="memory_efficient_replay_buffer",
                image_keys=image_keys,
                preload_rlds_path=FLAGS.preload_rlds_path,
                preload_data_transform=preload_data_transform,
            )

            if FLAGS.demo_path:
                # Check if the file exists
                if not os.path.exists(FLAGS.demo_path):
                    raise FileNotFoundError(f"File {FLAGS.demo_path} not found")

                with open(FLAGS.demo_path, "rb") as f:
                    trajs = pkl.load(f)
                    for traj in trajs:
                        demo_buffer.insert(traj)

            print(f"demo buffer size: {len(demo_buffer)}")
        else:
            demo_buffer = None

        # learner loop
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,  # None if no demo data is provided
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(2000)  # the queue size on the actor

        # actor loop
        print_green("starting actor loop")
        actor(agent, data_store, env, sampling_rng, video_dir=video_dir)

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
