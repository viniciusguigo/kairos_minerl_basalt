import datetime
import minerl
import namesgenerator
from sacred import Experiment
import basalt_utils.wrappers as wrapper_utils
from minerl.herobraine.wrappers.video_recording_wrapper import VideoRecordingWrapper
from basalt_utils.sb3_compat.policies import SpaceFlatteningActorCriticPolicy
from basalt_utils.sb3_compat.cnns import MAGICALCNN
from basalt_utils.wrappers import SaveObsAndActions
from basalt_utils.callbacks import BatchEndIntermediateRolloutEvaluator, MultiCallback, BCModelSaver
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
import collections
from imitation.algorithms.bc import BC
import imitation.data.rollout as il_rollout
import logging
import torch as th
from basalt_utils import utils
import os
import imitation.util.logger as imitation_logger
from sacred.observers import FileStorageObserver
from stable_baselines3.common.utils import get_device
from time import time

bc_baseline = Experiment("basalt_bc_baseline")

WRAPPERS = [# Maps from a string version of enum (found in the dataset) to an int version (expected for spaces.Discrete)
            (wrapper_utils.EnumStrToIntWrapper, dict()),
            # Transforms continuous camera action into discrete up/down/no-change buckets on both pitch and yaw
            (wrapper_utils.CameraDiscretizationWrapper, dict()),
            # Flattens a Dict action space into a Box, but retains memory of how to expand back out
            (wrapper_utils.ActionFlatteningWrapper, dict()),
            # Pull out only the POV observation from the observation space; transpose axes for SB3 compatibility
            (utils.ExtractPOVAndTranspose, dict())] #,


def make_unique_timestamp() -> str:
    """Make a timestamp along with a random word descriptor: e.g. 2021-06-06_1236_boring_wozniac"""
    ISO_TIMESTAMP = "%Y%m%d_%H%M"
    timestamp = datetime.datetime.now().strftime(ISO_TIMESTAMP)
    return f"{timestamp}_{namesgenerator.get_random_name()}"


@bc_baseline.config
def default_config():
    task_name = "MineRLBasaltFindCave-v0"
    train_batches = None
    train_epochs = None
    log_interval = 1
    # TODO fix this
    data_root = os.getenv('MINERL_DATA_ROOT')
    # SpaceFlatteningActorCriticPolicy is a policy that supports a flattened Dict action space by
    # maintaining multiple sub-distributions and merging their results
    policy_class = SpaceFlatteningActorCriticPolicy
    wrappers = WRAPPERS
    save_dir_base = "results/"
    save_dir = None
    policy_filename = 'trained_policy.pt'
    use_rollout_callback = False
    rollout_callback_batch_interval = 1000
    policy_save_interval = 1000
    callback_rollouts = 5
    save_videos = True
    mode = 'train'
    test_policy_path = 'train/trained_policy.pt'
    test_n_rollouts = 5
    # Note that `batch_size` needs to be less than the number of trajectories available for the task you're training on
    batch_size = 32
    n_traj = None
    buffer_size = 15000
    lr = 1e-4
    _ = locals()
    del _


@bc_baseline.config
def default_save_dir(save_dir_base, save_dir, task_name):
    """
    Calculates a save directory by combining the base `save_dir` ("results" by default) with
    the task name and a timestamp that contains both the time and a random name
    """
    if save_dir is None:
        save_dir = os.path.join(save_dir_base, task_name, make_unique_timestamp())
    _ = locals()
    del _


@bc_baseline.named_config
def normal_policy_class():
    """
    This is a sacred named_config, which means that when `normal_policy_class` is added as a parameter
    to a call of this experiment, the policy class will be set to ActorCriticCnnPolicy

    "Normal" here is just used to mean the default CNN policy from Stable Baselines, rather than the one explicitly designed
    to deal with multimodal action spaces (SpaceFlatteningActorCriticPolicy)
    """
    policy_class = ActorCriticCnnPolicy
    _ = locals()
    del _


@bc_baseline.main
def main(mode):
    if mode == 'train':
        train_bc()
    if mode == 'test':
        test_bc()


@bc_baseline.capture
def test_bc(task_name, data_root, wrappers, test_policy_path, test_n_rollouts, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Add a wrapper to the environment that records video and saves it in the
    # the `save_dir` we have constructed for this run.
    wrappers = [(VideoRecordingWrapper, {'video_directory':
                                             os.path.join(save_dir, 'videos')}),
                (SaveObsAndActions, {'save_dir':
                                         os.path.join(save_dir, 'obs_and_actions')})] + wrappers

    data_pipeline, wrapped_env = utils.get_data_pipeline_and_env(task_name, data_root, wrappers, dummy=False)
    vec_env = DummyVecEnv([lambda: wrapped_env])
    policy = th.load(test_policy_path, map_location=th.device(get_device('auto')))
    trajectories = il_rollout.generate_trajectories(policy, vec_env, il_rollout.min_episodes(test_n_rollouts))
    stats = il_rollout.rollout_stats(trajectories)
    stats = collections.OrderedDict([(key, stats[key])
                                     for key in sorted(stats)])

    # print it out
    kv_message = '\n'.join(f"  {key}={value}"
                           for key, value in stats.items())
    logging.info(f"Evaluation stats on '{task_name}': {kv_message}")


@bc_baseline.capture
def train_bc(task_name, batch_size, data_root, wrappers, train_epochs, n_traj, lr,
             policy_class, train_batches, log_interval, save_dir, policy_filename,
             use_rollout_callback, rollout_callback_batch_interval, callback_rollouts, save_videos,
             buffer_size, policy_save_interval):

    # This code is designed to let you either train for a fixed number of batches, or for a fixed number of epochs
    assert train_epochs is None or train_batches is None, \
        "Only one of train_batches or train_epochs should be set"
    assert not (train_batches is None and train_epochs is None), \
        "You cannot have both train_batches and train_epochs set to None"

    # If you've set the `save_videos` flag, add a VideoRecordingWrapper with a directory set
    # to the current `save_dir` to the environment wrappers
    if save_videos:
        wrappers = [(VideoRecordingWrapper, {'video_directory':
                                                 os.path.join(save_dir, 'videos')}),
                    (SaveObsAndActions, {'save_dir':
                                            os.path.join(save_dir, 'obs_and_actions')})] + wrappers

    # This `get_data_pipeline_and_env` utility is designed to be shared across multiple baselines
    # It takes in a task name, data root, and set of wrappers and returns

    # (1) An env object with the same environment spaces as you'd getting from making the env associated
    #     with this task and wrapping it in `wrappers`. Depending on the parameter passed into `dummy`, this is
    #     either the real wrapped environment, or a dummy environment that displays the same spaces,
    #     but without having to actually start up Minecraft
    # (2) A MineRL DataPipeline that can be used to construct a batch_iter used by BC, and also as a handle to clean
    #     up that iterator after training.
    data_pipeline, wrapped_env = utils.get_data_pipeline_and_env(task_name, data_root, wrappers,
                                                                 dummy=not use_rollout_callback)

    # This utility creates a data iterator that is basically a light wrapper around the baseline MineRL data iterator
    # that additionally:
    # (1) Applies all observation and action transformations specified by the wrappers in `wrappers`, and
    # (2) Calls `np.squeeze` recursively on all the nested dict spaces to remove the sequence dimension, since we're
    #     just doing single-frame BC here
    data_iter = utils.create_data_iterator(wrapped_env,
                                           data_pipeline=data_pipeline,
                                           batch_size=batch_size,
                                           num_epochs=train_epochs,
                                           num_batches=train_batches,
                                           buffer_size=buffer_size)
    if policy_class == SpaceFlatteningActorCriticPolicy:
        policy = policy_class(observation_space=wrapped_env.observation_space,
                              action_space=wrapped_env.action_space,
                              env=wrapped_env,
                              lr_schedule=lambda _: 1e-4,
                              features_extractor_class=MAGICALCNN)
    else:
        policy = policy_class(observation_space=wrapped_env.observation_space,
                              action_space=wrapped_env.action_space,
                              lr_schedule=lambda _: 1e-4,
                              features_extractor_class=MAGICALCNN)

    os.makedirs(save_dir, exist_ok=True)
    imitation_logger.configure(save_dir, ["stdout", "tensorboard"])
    callbacks = [BCModelSaver(policy=policy,
                              save_dir=os.path.join(save_dir, 'policy_checkpoints'),
                              save_interval_batches=policy_save_interval)]
    if use_rollout_callback:
        callbacks.append(BatchEndIntermediateRolloutEvaluator(policy=policy,
                                                              env=wrapped_env,
                                                              save_dir=os.path.join(save_dir, 'policy_rollouts'),
                                                              evaluate_interval_batches=rollout_callback_batch_interval,
                                                              n_rollouts=callback_rollouts))
    callback_op = MultiCallback(callbacks)

    bc_trainer = BC(
        observation_space=wrapped_env.observation_space,
        action_space=wrapped_env.action_space,
        policy_class= lambda **kwargs: policy,
        policy_kwargs=None,
        expert_data=data_iter,
        device='auto',
        optimizer_cls=th.optim.Adam,
        optimizer_kwargs=dict(lr=lr),
        ent_weight=1e-3,
        l2_weight=1e-5)
    bc_trainer.train(n_epochs=train_epochs,
                     n_batches=train_batches,
                     log_interval=log_interval,
                     on_batch_end=callback_op)
    bc_trainer.save_policy(policy_path=os.path.join(save_dir, policy_filename))
    bc_baseline.add_artifact(os.path.join(save_dir, policy_filename))
    bc_baseline.log_scalar(f'run_location={save_dir}', 1)
    print("Training complete; cleaning up data pipeline!")
    data_iter.close()


if __name__ == "__main__":
    bc_baseline.observers.append(FileStorageObserver("sacred_results"))
    bc_baseline.run_commandline()
