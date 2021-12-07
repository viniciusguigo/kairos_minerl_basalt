from functools import partial
from typing import Union, Type, Dict, List, Optional, Any, Callable

import torch as th
import torch.nn as nn
import numpy as np
import gym

from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.policies import ActorCriticPolicy, create_sde_features_extractor
from stable_baselines3.common.torch_layers import MlpExtractor, BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from basalt_utils.sb3_compat.distributions import MultimodalProbabilityDistribution
from basalt_utils.sb3_compat.features_extractor import HardcodedMinecraftFeaturesExtractor, DictFeaturesExtractor


class SpaceFlatteningActorCriticPolicy(ActorCriticPolicy):
    """
    A policy class specifically designed for use along with an environment wrapped in a
    realistic_benchmarks.space_flattening_wrappers wrapper, where the observation
    or action space of the underlying environment is a Dict, but it has been wrapped as a Box,
    with unwrapping information stored on the wrapped environment.

    This policy class handles the following modifications from the typical
    ActorCriticPolicy workflow:

    1) If the wrapped environment implements an observation-flattening wrapper, the policy __init__
    asserts that the features_extractor_class passed in is one that can handle an unwrapped Dict space.
    It also pulls the `ordered_observation_space` and `get_structured_observation` object and Callable
    respectively, out of the env. `ordered_observation_spaces` is an OrderedDict (possibly nested)
    of the observation spaces that were flattened into a Box. `get_structured_observation`
    is a method that takes in a flattened Box and returns a (possibly nested) dict observation back,
    allowing the feature extractor to handle different underlying obs spaces in different
    and respectively appropriate ways. Relevant small note here: this changes the place where
    preprocess_obs is called to be inside the features_extractor, rather than within the
    policy, since in fact `preprocess_obs` needs to be called separately on each nested observation,
    along with information about the space that observation came from

    2) If the wrapped environment implements an action-flattening wrapper, the policy creates a
    MultiModalProbabilityDistribution, which takes in the underlying action spaces (captured by the
    wrapped env) and creates individual probability distributions for each space. This allows,
    for example, a Discrete and a Box action space to be packaged together in a Box for intermediate policy
    calculations, but still allows the Discrete action to be sampled in an appropriate way, rather
    than treated as a Box for sampling purposes.

    :param observation_space: (gym.spaces.Space) Observation space
    :param action_space: (gym.spaces.Space) Action space
    :param lr_schedule: (Callable) Learning rate schedule (could be constant)
    :param env: (gym.Env) The wrapped environment on which this policy will be used.
    NOTE this is the only parameter not also required by ActorCriticPolicy
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
    :param device: (str or th.device) Device on which the code should run.
    :param activation_fn: (Type[nn.Module]) Activation function
    :param ortho_init: (bool) Whether to use or not orthogonal initialization
    :param use_sde: (bool) Whether to use State Dependent Exploration or not
    :param log_std_init: (float) Initial value for the log standard deviation
    :param full_std: (bool) Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: ([int]) Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: (bool) Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: (bool) Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: (Type[BaseFeaturesExtractor]) Features extractor to use.
    :param features_extractor_kwargs: (Optional[Dict[str, Any]]) Keyword arguments
        to pass to the feature extractor.
    :param normalize_images: (bool) Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: (Type[th.optim.Optimizer]) The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: (Optional[Dict[str, Any]]) Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Callable,
                 env: gym.Env = None,
                 net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 ortho_init: bool = True,
                 use_sde: bool = False,
                 log_std_init: float = 0.0,
                 full_std: bool = True,
                 sde_net_arch: Optional[List[int]] = None,
                 use_expln: bool = False,
                 squash_output: bool = False,
                 features_extractor_class: Type[BaseFeaturesExtractor] = HardcodedMinecraftFeaturesExtractor, # TODO changethis
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None):
        assert env is not None, "Must pass in a non-None env to initialize wrapper info"
        if isinstance(env, DummyVecEnv):
            print("You are using a VecEnv; assuming all envs have equivalent observation and action wrappers applied")
            env = env.envs[0]
        self.dict_obs_space = False
        self.dict_act_space = False
        if hasattr(env, "get_structured_observation"):
            self.dict_obs_space = True
            assert issubclass(features_extractor_class,
                              DictFeaturesExtractor), "Features extractor class must " \
                                                      "be a DictFeaturesExtractor"
            if features_extractor_kwargs is None:
                features_extractor_kwargs = dict()
            # Normalizing images is normally done within the policy, but
            # will need to be done on each nested internal space
            features_extractor_kwargs['normalize_images'] = normalize_images
            # A callable going from Box to dict of sub-observations
            features_extractor_kwargs['obs_unwrapper_function'] = env.get_structured_observation
            # A (possibly-nested) dict of observation spaces, used by the feature extractor
            # to do correct preprocessing on each
            features_extractor_kwargs['obs_space_dict'] = env.ordered_observation_spaces

        if hasattr(env, "get_structured_action"):
            self.dict_act_space = True
            self.ordered_action_spaces = env.ordered_action_spaces
            self.get_structured_action = env.get_structured_action

        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         lr_schedule=lr_schedule,
                         net_arch=net_arch,
                         activation_fn=activation_fn,
                         ortho_init=ortho_init,
                         use_sde=use_sde,
                         log_std_init=log_std_init,
                         full_std=full_std,
                         sde_net_arch=sde_net_arch,
                         use_expln=use_expln,
                         squash_output=squash_output,
                         features_extractor_class=features_extractor_class,
                         features_extractor_kwargs=features_extractor_kwargs,
                         normalize_images=normalize_images,
                         optimizer_class=optimizer_class,
                         optimizer_kwargs=optimizer_kwargs)

    def _build(self, lr_schedule: Callable):
        if not self.dict_act_space:
            super()._build(lr_schedule=lr_schedule)
        else:
            # copied directly from ActorCriticPolicy._build
            # I unfortunately am not sure of a way of doing this
            # with less duplication but still without modifying SB3 itself
            self.mlp_extractor = MlpExtractor(self.features_dim, net_arch=self.net_arch,
                                              activation_fn=self.activation_fn, device=self.device)

            latent_dim_pi = self.mlp_extractor.latent_dim_pi

            # Separate feature extractor for gSDE
            if self.sde_net_arch is not None:
                self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(self.features_dim,
                                                                                            self.sde_net_arch,
                                                                                            self.activation_fn)
            else:
                self.sde_features_extractor = latent_sde_dim = None
            ### Added
            self.action_dist = MultimodalProbabilityDistribution(self.ordered_action_spaces,
                                                                 self.get_structured_action)
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi,
                                                                      latent_sde_dim=latent_sde_dim,
                                                                      log_std_init=self.log_std_init)

            ####

            ### Below also copied from ActorCriticPolicy
            self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
            # Init weights: use orthogonal initialization
            # with small initial weight for the output
            if self.ortho_init:
                # TODO: check for features_extractor
                # Values from stable-baselines.
                # feature_extractor/mlp values are
                # originally from openai/baselines (default gains/init_scales).
                module_gains = {
                    self.features_extractor: np.sqrt(2),
                    self.mlp_extractor: np.sqrt(2),
                    self.action_net: 0.01,
                    self.value_net: 1
                }
                for module, gain in module_gains.items():
                    module.apply(partial(self.init_weights, gain=gain))

            # Setup optimizer with initial learning rate
            self.optimizer = self.optimizer_class(self.parameters(),
                                                  lr=lr_schedule(1),
                                                  **self.optimizer_kwargs)
            ####

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        assert self.features_extractor is not None, 'No feature extractor was set'
        if isinstance(self.features_extractor, DictFeaturesExtractor):
            # DictFeaturesExtractors do their own processing of each
            # internal observation space
            return self.features_extractor(obs)
        else:
            preprocessed_obs = preprocess_obs(obs,
                                              self.observation_space,
                                              normalize_images=self.normalize_images)
            return self.features_extractor(preprocessed_obs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor,
                                     latent_sde: Optional[th.Tensor] = None) -> Distribution:
        if isinstance(self.action_dist, MultimodalProbabilityDistribution):
            # if th.any(th.isnan(latent_pi)):
            #     breakpoint()
            mean_actions_dict, std_dict = self.action_net(latent_pi)
            return self.action_dist.proba_distribution(mean_actions_dict,
                                                       log_std_dict=std_dict,
                                                       latent_sde=latent_sde)
        else:
            return super()._get_action_dist_from_latent(latent_pi, latent_sde)


