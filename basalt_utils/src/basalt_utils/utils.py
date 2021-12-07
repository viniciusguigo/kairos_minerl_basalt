import gym
import numpy as np
import minerl
import torch
import warnings
import os
import traceback
from stable_baselines3.common.utils import get_device
from minerl.data import BufferedBatchIter

class DummyEnv(gym.Env):
    """
    A simplistic class that lets us mock up a gym Environment that is sufficient for our purposes
    without actually going through the whole convoluted registration process.
    """
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space

    def step(self, action):
        if isinstance(self.action_space, gym.spaces.Dict):
            assert isinstance(action, dict)
        return self.observation_space.sample(), 0, True, {}

    def reset(self):
        return self.observation_space.sample()


class NestableObservationWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        if hasattr(self.env, 'observation'):
            return self._observation(self.env.observation(observation))
        else:
            return self._observation(observation)

    def _observation(self, observation):
        raise NotImplementedError


class NormalizeObservations(NestableObservationWrapper):
    def __init__(self, env, high_val=255):
        super().__init__(env)
        self.high_val = high_val

    def _observation(self, observation):
        assert observation.max() <= self.high_val, f"Observation greater than high val {self.high_val} found"
        return observation/self.high_val


class ExtractPOVAndTranspose(NestableObservationWrapper):
    """
    Basically what it says on the tin. Extracts only the POV observation out of the `obs` dict,
    and transposes those observations to be in the (C, H, W) format used by stable_baselines and imitation
    """
    def __init__(self, env):
        super().__init__(env)
        non_transposed_shape = self.env.observation_space['pov'].shape
        self.high = np.max(self.env.observation_space['pov'].high)
        transposed_shape = (non_transposed_shape[2],
                            non_transposed_shape[0],
                            non_transposed_shape[1])
        # Note: this assumes the Box is of the form where low/high values are vector but need to be scalar
        transposed_obs_space = gym.spaces.Box(low=np.min(self.env.observation_space['pov'].low),
                                              high=np.max(self.env.observation_space['pov'].high),
                                              shape=transposed_shape,
                                              dtype=np.uint8)
        self.observation_space = transposed_obs_space

    def _observation(self, observation):
        # Minecraft returns shapes in NHWC by default
        return np.swapaxes(observation['pov'], -1, -3)


class Testing10000StepLimitWrapper(gym.wrappers.TimeLimit):
    """
    A simple wrapper to impose a 10,000 step limit, for environments that don't have one built in
    """
    def __init__(self, env):
        super().__init__(env, 10000)


def wrap_env(env, wrappers):
    """
    Wrap `env` in all gym wrappers specified by `wrappers`
    """
    for wrapper, args in wrappers:
        env = wrapper(env, **args)
    return env


def optional_observation_map(env, inner_obs):
    """
    If the env implements the `observation` function (i.e. if one of the
    wrappers is an ObservationWrapper), call that `observation` transformation
    on the observation produced by the inner environment
    """
    if hasattr(env, 'observation'):
        return env.observation(inner_obs)
    else:
        return inner_obs


def optional_action_map(env, inner_action):
    """
    This is doing something slightly tricky that is explained in the documentation for
    RecursiveActionWrapper (which TODO should eventually be in MineRL)
    Basically, it needs to apply `reverse_action` transformations from the inside out
    when converting the actions stored and used in a dataset

    """
    if hasattr(env, 'wrap_action'):
        return env.wrap_action(inner_action)
    else:
        return inner_action


def recursive_squeeze(dictlike):
    """
    Take a possibly-nested dictionary-like object of which all leaf elements are numpy ar
    """
    out = {}
    for k, v in dictlike.items():
        if isinstance(v, dict):
            out[k] = recursive_squeeze(v)
        else:
            out[k] = np.squeeze(v)
    return out


def warn_on_non_image_tensor(x):
    """Do some basic checks to make sure the input image tensor looks like a
    batch of stacked square frames. Good sanity check to make sure that
    preprocessing is not being messed up somehow."""
    stack_str = None

    def do_warning(message):
        # issue a warning, but annotate it with some information about the
        # stack (specifically, basenames of code files and line number at the
        # time of exception for each stack frame except this one)
        nonlocal stack_str
        if stack_str is None:
            frames = traceback.extract_stack()
            stack_str = '/'.join(
                f'{os.path.basename(frame.filename)}:{frame.lineno}'
                # [:-1] skips the current frame
                for frame in frames[:-1])
        warnings.warn(message + f" (stack: {stack_str})")

    # check that image has rank 4
    if x.ndim != 4:
        do_warning(f"Image tensor has rank {x.ndim}, not rank 4")

    # check that H=W
    if x.shape[2] != x.shape[3]:
        do_warning(
            f"Image tensor shape {x.shape} doesn't have square images")

    # check that image is in [0,1] (approximately)
    # this is the range that SB uses
    v_min = torch.min(x).item()
    v_max = torch.max(x).item()
    if v_min < -0.01 or v_max > 1.01:
        do_warning(
            f"Input image tensor has values in range [{v_min}, {v_max}], "
            "not expected range [0, 1]")

    std = torch.std(x).item()
    if std < 0.05:
        do_warning(
            f"Input image tensor values have low stddev {std} (range "
            f"[{v_min}, {v_max}])")


def get_data_pipeline_and_env(task_name, data_root, wrappers, dummy=True):
    """
    This code loads a data pipeline object and creates an (optionally dummy) environment with the
    same observation and action space as the (wrapped) environment you want to train on

    :param task_name: The name of the MineRL task you want to get data for
    :param data_root: For manually specifying a MineRL data root
    :param wrappers: The wrappers you want to apply to both the loaded data and live environment
    """
    data_pipeline = minerl.data.make(environment=task_name,
                                     data_dir=data_root)
    if dummy:
        env = DummyEnv(action_space=data_pipeline.action_space,
                       observation_space=data_pipeline.observation_space)
    else:
        env = gym.make(task_name)
    wrapped_env = wrap_env(env, wrappers)
    return data_pipeline, wrapped_env


def create_data_iterator(
        wrapped_dummy_env: gym.Env,
        data_pipeline: minerl.data.DataPipeline,
        batch_size: int,
        buffer_size: int = 15000,
        num_epochs: int = None,
        num_batches: int = None,
        remove_no_ops: bool = False,
) -> dict:
    """
    Construct a data iterator that (1) loads data from disk, and (2) wraps it in the set of
    wrappers that have been applied to `wrapped_dummy_env`.

    :param wrapped_dummy_env: An environment that mimics the base environment and wrappers we'll be using for training,
    but doesn't actually call Minecraft
    :param data_pipeline: A MineRL DataPipeline object that can handle loading data from disk
    :param batch_size: The batch size we want the iterator to produce
    :param num_epochs: The number of epochs we want the underlying iterator to run for
    :param num_batches: The number of batches we want the underlying iterator to run for
    :param remove_no_ops: Whether to remove transitions with no-op demonstrator actions from batches
    as they are generated. For now, this corresponds to all-zeros.

    :yield: Wrapped observations and actions in a dict with the keys "obs", "acts", "rews",
         "next_obs", "dones".
    """
    buffered_iterator = BufferedBatchIter(data_pipeline, buffer_target_size=buffer_size)
    for current_obs, action, reward, next_obs, done in buffered_iterator.buffered_batch_iter(batch_size=batch_size,
                                                                                             num_epochs=num_epochs,
                                                                                             num_batches=num_batches):
        wrapped_obs = optional_observation_map(wrapped_dummy_env,
                                               recursive_squeeze(current_obs))
        wrapped_next_obs = optional_observation_map(wrapped_dummy_env,
                                                    recursive_squeeze(next_obs))
        wrapped_action = optional_action_map(wrapped_dummy_env,
                                             recursive_squeeze(action))

        if remove_no_ops:
            # This definitely makes assumptions about the action space, namely that all-zeros corresponds to a no-op
            not_no_op_indices = wrapped_action.sum(axis=1) != 0
            wrapped_obs = wrapped_obs[not_no_op_indices]
            wrapped_next_obs = wrapped_next_obs[not_no_op_indices]
            wrapped_action = wrapped_action[not_no_op_indices]

        return_dict = dict(obs=wrapped_obs,
                           acts=wrapped_action,
                           rews=reward,
                           next_obs=wrapped_next_obs,
                           dones=done)

        yield return_dict
