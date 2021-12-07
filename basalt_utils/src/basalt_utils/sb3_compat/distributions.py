from typing import Callable, Optional, Tuple
import torch as th
from collections import OrderedDict

from stable_baselines3.common.distributions import (Distribution, make_proba_distribution,
                                                    DiagGaussianDistribution, StateDependentNoiseDistribution)


class MultimodalProbabilityDistribution(Distribution):
    """
    A Distribution that internally contains sub-Distributions corresponding to
    each action space passed in within `ordered_action_spaces`
    """
    def __init__(self,
                 ordered_action_spaces: OrderedDict,
                 action_transform_function: Callable):
        """
        :param ordered_action_spaces: A (not-nested) OrderedDict mapping the names of action spaces to the Space objects
        that define them
        :param action_transform_function A function that takes in the flattened version of the action vector and returns
        a dictionary of sub-actions, according to some externally determined logic
        """
        super(MultimodalProbabilityDistribution, self).__init__()
        self.subdistribution_types = MultimodalProbabilityDistribution.create_subdistributions(ordered_action_spaces)
        self.parametrized_subdists = None
        self.action_transform_function = action_transform_function

    @staticmethod
    def create_subdistributions(ordered_action_spaces: OrderedDict):
        """
        Iterates over the spaces in `ordered_action_space` and for each, calls `make_proba_distribution`
        from SB3 on that space. Creates a dict of subdistributions.

        A subtle but important note: these Distributions do not yet have parametrized torch distributions inside them;
        and thus can't actually produce samples or log probabilities. Parameters are only set when
        `proba_distribution` is called.

        """
        assert isinstance(ordered_action_spaces, OrderedDict), "ordered_action_spaces must be an OrderedDict"
        ordered_distributions = OrderedDict()
        for space_name, space in ordered_action_spaces.items():
            inferred_distribution = make_proba_distribution(space)
            ordered_distributions[space_name] = inferred_distribution
        return ordered_distributions

    def proba_distribution_net(self,
                               latent_dim: int,
                               latent_sde_dim: int,
                               log_std_init: float):
        """
        Creates a MultiModalActionNet. This is a nn.Module that produces a dict of `mean_actions`
        and also a dict of `log_std` for each sub-action-space

        :param latent_dim: The latent dimension of features being passed into the proba distribution
        :param latent_sde_dim: TODO figure out how SDE works
        :param log_std_init: (float) Initial value for the log standard deviation
        :return:
        """
        return MultiModalActionNet(self.subdistribution_types, latent_dim, latent_sde_dim, log_std_init)

    def proba_distribution(self,
                           mean_actions_dict: dict,
                           log_std_dict: dict,
                           latent_sde: Optional[th.Tensor] = None): # TODO is this the right type for latent_sde?
        """
        Iterates over subdistributions, and, for each, calls the underlying distribution's
        `proba_distribution` method with the mean actions (and possibly log_std) corresponding to that
        subdistribution. This returns a _parametrized_ subdistribution of the same time, which is then stored on
        self.parametrized_subdists.

        :param mean_actions_dict: A dict mapping between space names and the calculated `mean_actions` tensor
        from the policy network, used to parametrize that space's distribution
        :param log_std_dict: A dict mapping between space name and the calculated `log_std` tensor from the
        policy network. Note: not all subdistributions have standard deviations, and thus not all are represented here
        :param latent_sde: A tensor? #TODO
        :return:
        """
        self.parametrized_subdists = dict()
        for subdist_name, subdist in self.subdistribution_types.items():

            if isinstance(subdist, DiagGaussianDistribution):
                self.parametrized_subdists[subdist_name] = subdist.proba_distribution(mean_actions_dict[subdist_name],
                                                                                      log_std_dict[subdist_name])
            elif isinstance(subdist, StateDependentNoiseDistribution):
                self.parametrized_subdists[subdist_name] = subdist.proba_distribution(mean_actions_dict[subdist_name],
                                                                                      log_std_dict[subdist_name],
                                                                                      latent_sde)
            else:
                self.parametrized_subdists[subdist_name] = subdist.proba_distribution(mean_actions_dict[subdist_name])
        return self

    def assert_parametrized(self):
        assert self.parametrized_subdists is not None, "You must call call `proba_distribution` " \
                                                       "to set parametrized_subdists to not-None"

    def mode(self) -> th.Tensor:
        """
        NOTE: `proba_distribution` must be called first so that `self.parametrized_subdists` is not-None
        Iterates over internal subdistributions, calculates the mode of each, and concatenates them.
        :return:
        """
        self.assert_parametrized()

        modes = []
        for p in self.parametrized_subdists.values():
            mode = p.mode()
            if len(mode.shape) == 1:
                mode = th.unsqueeze(mode, 1)
            modes.append(mode)
        return th.cat(modes, dim=-1)

    def entropy(self) -> th.Tensor:
        """
        NOTE: `proba_distribution` must be called first so that `self.parametrized_subdists` is not-None

        Iterates over internal subdistributions, calculates the entropy of each, and then sums the
        entropies together, because the actions are assumed to be independent random variables
        :return:
        """
        self.assert_parametrized()
        return th.sum(th.stack([p.entropy() for p in self.parametrized_subdists.values()], dim=-1))

    def sample(self) -> th.Tensor:
        """
        NOTE: `proba_distribution` must be called first so that `self.parametrized_subdists` is not-None

        Iterates over internal subdistributions, samples from each, and concatenates those samples.
        :return:
        """
        self.assert_parametrized()
        samples = []
        for p in self.parametrized_subdists.values():
            sample = p.sample()
            if len(sample.shape) == 1:
                # Even if the sample is a scalar (per sample in batch) turn it into a 1D vector for concatenation
                sample = th.unsqueeze(sample, 1)
            samples.append(sample)
        return th.cat(samples, dim=-1)

    def log_prob(self,
                 actions: th.Tensor) -> th.Tensor:
        """
        NOTE: `proba_distribution` must be called first so that `self.parametrized_subdists` is not-None

        Iterates over internal subdistributions, pulls out the action corresponding to that
        subdistribution, calculates the log probability of that action under that subdistribution,
        and then sums the log probabilites together, because the actions are assumed to be independent
        """
        self.assert_parametrized()
        log_probas = []
        action_dict = self.action_transform_function(actions)
        for subspace_name, subdist in self.parametrized_subdists.items():
            log_probas.append(subdist.log_prob(action_dict[subspace_name]))
        summed_log_proba = th.sum(th.stack(log_probas), dim=0)
        assert not th.any(summed_log_proba > 0).item(), f"Log probability above 0 found: {log_probas}"
        return summed_log_proba

    def actions_from_params(self,
                            mean_actions_dict: dict,
                            log_std_dict: dict = None,
                            latent_sde: th.Tensor = None,
                            deterministic: bool = False) -> th.Tensor:
        """
        Calls `proba_distribution` to set parametrized probability distributions.
        then calls `get_actions` on each subdist, which samples or returns the mode
        according to the `deterministic` flag.
        """
        self.proba_distribution(mean_actions_dict, log_std_dict, latent_sde)
        actions = []
        for parametrized_subdist in self.parametrized_subdists.values():
            actions.append(parametrized_subdist.get_actions(deterministic))

        return th.cat(actions, dim=-1)

    def log_prob_from_params(self,
                             mean_actions_dict: dict,
                             log_std_dict: dict = None,
                             latent_sde: th.Tensor = None,
                             deterministic=False) -> Tuple[th.Tensor, th.Tensor]:
        """
        Calls `actions_from_params` which parametrizes the subdistributions, and then samples (or gets the mode)
        of actions from them. Then calls `log_prob` with those actions against the now-parametrized
        subdistributions.

        """
        actions = self.actions_from_params(mean_actions_dict, log_std_dict,
                                           latent_sde, deterministic)
        log_prob = self.log_prob(actions)

        return actions, log_prob


class MultiModalActionNet(th.nn.Module):
    def __init__(self,
                 subdistributions: dict,
                 latent_dim: int,
                 latent_sde_dim: int,
                 log_std_init: float):
        super().__init__()
        _submodules = dict()
        _parameters = dict()
        self.subdistributions = subdistributions
        # Iterates over the subdistributions (which are not parametrized yet)
        # and calls `proba_distribution_net` on each, which returns a submodule (for action_net)
        # and, where appropriate, a std value. These then get stored on a dict under the name of the
        # subspace/subdistribution
        for subdist_name, subdist in subdistributions.items():
            if isinstance(subdist, DiagGaussianDistribution):
                action_net, std = subdist.proba_distribution_net(latent_dim,
                                                                 log_std_init)
                _parameters[subdist_name] = std
            elif isinstance(subdist, StateDependentNoiseDistribution):
                action_net, std = subdist.proba_distribution_net(latent_dim,
                                                                 latent_sde_dim,
                                                                 log_std_init)
                _parameters[subdist_name] = std
            else:
                action_net = subdist.proba_distribution_net(latent_dim)

            _submodules[subdist_name] = action_net
        # Hit "forward exists" error but only for ModuleDict
        self.keys = list(_submodules.keys())
        self.submodule_list = th.nn.ModuleList(list(_submodules.values()))
        self.std_keys = list(_parameters.keys())
        self.parameter_list = th.nn.ParameterList(list(_parameters.values()))

    def forward(self, latent_pi):
        mean_actions_dict = {}
        std_dist = {}
        for ind, subdist_name in enumerate(self.keys):
            # For each subdistribution, call the submodule connected to that space
            # to map between the latent and the the `mean_actions` result for that
            # subdistribution
            mean_actions_dict[subdist_name] = self.submodule_list[ind](latent_pi)

        for ind, subdist_name in enumerate(self.std_keys):
            # For all keys that have `std` values set, return the std parameter corresponding to that key
            std_dist[subdist_name] = self.parameter_list[ind]
        return mean_actions_dict, std_dist



