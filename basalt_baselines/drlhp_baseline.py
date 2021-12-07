import gym
from sacred import Experiment

import basalt_utils.wrappers as wrapper_utils
from basalt_utils import utils
from basalt_utils.sb3_compat.policies import SpaceFlatteningActorCriticPolicy
from drlhp import HumanPreferencesEnvWrapper

WRAPPERS = [# Transforms continuous camera action into discrete up/down/no-change buckets on both pitch and yaw
            wrapper_utils.CameraDiscretizationWrapper,
            # Flattens a Dict action space into a Box, but retains memory of how to expand back out
            wrapper_utils.ActionFlatteningWrapper,
            # Pull out only the POV observation from the observation space; transpose axes for SB3 compatibility
            utils.ExtractPOVAndTranspose,
            # Add a time limit to the environment (only relevant for testing)
            utils.Testing10000StepLimitWrapper,
            wrapper_utils.FrameSkip]

drlhp_baseline = Experiment("basalt_drlhp_baseline")

@drlhp_baseline.config
def default_config():
    task_name = "FindCaves-v0"
    train_batches = 10
    train_epochs = None
    log_interval = 1
    data_root = "/Users/cody/Code/il-representations/data/minecraft"
    # SpaceFlatteningActorCriticPolicy is a policy that supports a flattened Dict action space by
    # maintaining multiple sub-distributions and merging their results
    policy_class = SpaceFlatteningActorCriticPolicy
    wrappers = WRAPPERS
    save_location = "/Users/cody/Code/simple_bc_baseline/results"
    policy_path = 'trained_policy.pt'
    batch_size = 16
    n_traj = 16
    lr = 1e-4
    _ = locals()
    del _


@drlhp_baseline.automain
def train_drlhp(task_name, batch_size, data_root, wrappers, train_epochs, n_traj, lr,
                policy_class, train_batches, log_interval, save_location, policy_path):
    env = gym.make(task_name)
    wrapped_env = HumanPreferencesEnvWrapper(env,
                                             segment_length=100,
                                             synthetic_preferences = False,
                                             n_initial_training_steps = 10)
    breakpoint()