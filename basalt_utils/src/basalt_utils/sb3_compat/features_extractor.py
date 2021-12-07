from typing import Callable, Dict, Iterable
from collections import OrderedDict, defaultdict

from stable_baselines3.common.torch_layers import (NatureCNN, BaseFeaturesExtractor,
                                                   create_mlp)
from stable_baselines3.common.preprocessing import preprocess_obs, is_image_space
import torch as th
import torch.nn as nn
import numpy as np
import gym


def recursive_preprocess(obs_space_dict: dict,
                         observation_dict: dict,
                         normalize_images: bool) -> dict:
    """
    Recursively iterate through `observation_dict` (a possibly-nested dictionary of actual observations),
    preprocess each observation according to the SB3 preprocessing implied by the space corresponding
    to the space name in `obs_space_dict` (a dict of actual gym Spaces), and add that to a `processed_observations`
    dict. The boolean `normalize_images` governing whether image spaces in general are normalized


    :return: A nested dictionary of processed observations
    """
    processed_observations = dict()
    for space_name, obs_space in obs_space_dict.items():
        if isinstance(obs_space, OrderedDict):
            processed_observations[space_name] = recursive_preprocess(obs_space,
                                                                      observation_dict[space_name],
                                                                      normalize_images)
        else:
            processed_observations[space_name] = preprocess_obs(observation_dict[space_name],
                                                                obs_space,
                                                                normalize_images=normalize_images)
    return processed_observations


def get_first_dim_from_shape(shape):
    if len(shape) == 0:
        return 1
    else:
        return int(shape[0])


def recursive_lookup(lookup, index_list):
    """
    Takes in indexes `index_list` in the form of a list of keys, and
    iteratively look through those keys in `lookup` until you reach the
    end of the list.
    For example, if given the list ['inventory', 'wood'] this method
    returns lookup['inventory']['wood']
    :return:
    """
    for index in index_list:
        lookup = lookup[index]
    return lookup


def recursive_lookup_from_string(lookup_dict,
                                 index_string,
                                 split_chars):
    index_list = index_string.split(split_chars)
    return recursive_lookup(lookup_dict, index_list)


class DictFeaturesExtractor(BaseFeaturesExtractor):
    """
    The base class for all FeaturesExtractors that maps between a dictionary of inputs
    and a tensor of features.

    This is not usable directly, as it does not implement the _dict_forward() method
    to actually define the logic that links the dict input and the tensor output.

    :param observation_space - The gym space of the flattened observation space
    :param obs_unwrapper_function - A Callable to map from flattened to dict observation
    :param obs_space_dict - A possibly-nested dict of the obs spaces that went into the flattened space
    :param normalize_images - A boolean for whether image spaces in general should be normalized
    :param features_dim - The dimension of features that should come out of this feature extractor
    """
    def __init__(self,
                 observation_space: gym.Space,
                 obs_unwrapper_function: Callable,
                 obs_space_dict: Dict[str, gym.Space],
                 normalize_images: bool,
                 features_dim: int = 0):

        super().__init__(observation_space, features_dim)
        self.obs_unwrapper_function = obs_unwrapper_function
        self.obs_space_dict = obs_space_dict
        self.normalize_images = normalize_images

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observation_dict = self.obs_unwrapper_function(observations)
        processed_observations = recursive_preprocess(self.obs_space_dict,
                                                      observation_dict,
                                                      self.normalize_images)
        return self._dict_forward(processed_observations)

    def _dict_forward(self, observation_dict: dict) -> th.Tensor:
        raise NotImplementedError


class HardcodedMinecraftFeaturesExtractor(DictFeaturesExtractor):
    """
    A specific hardcoded transformation, mostly relevant as an example
    of how one might write similar (actually purposeful) hardcoded extractors
    if desired. This particular one applies NatureCNN to the `pov` observation,
    and concatenates together `cameraAngle` and `inventory:dirt` before
    passing that concatenation into a MLP

    """
    def __init__(self,
                 observation_space: gym.Space,
                 obs_unwrapper_function: Callable,
                 obs_space_dict:  Dict[str, gym.Space],
                 normalize_images: bool,
                 features_dim: int = 96):

        super().__init__(observation_space, obs_unwrapper_function, obs_space_dict,
                         normalize_images, features_dim)
        self.cnn_extractor = NatureCNN(obs_space_dict['pov'], features_dim=80)
        self.camera_angle_dim = get_first_dim_from_shape(obs_space_dict['cameraAngle'].shape)
        camera_angle_modules = create_mlp(input_dim=self.camera_angle_dim,
                                          output_dim=8,
                                          net_arch=[10])
        self.camera_angle_extractor = nn.Sequential(*camera_angle_modules)
        self.dirt_inventory_dim = get_first_dim_from_shape(obs_space_dict['inventory']['dirt'].shape)
        dirt_inventory_modules = create_mlp(input_dim=self.dirt_inventory_dim,
                                            output_dim=8,
                                            net_arch=[10])
        self.dirt_inventory_extractor = nn.Sequential(*dirt_inventory_modules)

    def _dict_forward(self, observation_dict: dict) -> th.Tensor:
        cnn_latent = self.cnn_extractor(observation_dict['pov'])
        batch_size = observation_dict['pov'].shape[0]
        cam_angle_latent = self.camera_angle_extractor(observation_dict['cameraAngle']
                                                       .reshape(batch_size,
                                                                self.camera_angle_dim))
        dirt_inventory_latent = self.dirt_inventory_extractor(observation_dict['inventory']['dirt']
                                                              .reshape(batch_size,
                                                                       self.dirt_inventory_dim))
        return th.cat([cnn_latent, cam_angle_latent, dirt_inventory_latent], dim=-1)


class InferredDictFeatureExtractor(DictFeaturesExtractor):
    """
    A FeatureExtractor that:
        - Takes all spaces inferred to be image spaces, and runs them through a CNN, and produces
    a feature vector of size `cnn_feature_dim` out of each CNN
        - Takes all spaces inferred to be MLP spaces (non-image Box spaces), concatenates them,
    and runs them through a joint MLP with hidden layers specified by `mlp_net_arch`
    (in the format of a list of layer dimensions) which produces a feature vector
    of size `mlp_feature_dim` TODO maybe should be separate MLPs?
        - Takes all spaces inferred to be EMBED spaces (Discrete spaces) embeds each with a separate
    embedding table of dimension `embedding_dim`, and produces an embedding feature vector for each
        - Takes all feature vectors produced above, concatenates together, and applies a
    projection linear layer merging information from them all; passes result of that
    layer back as the extractor's final feature vector

    **NOTE**: The feature dims used here are not tuned or optimized in any way, and may well likely
    be a poor fit for your data

    :param observation_space
    :param obs_unwrapper_function: (Callable) A function that takes in a flattened observation as input,
    and returns a dict of multiple distinct observations, reshaped into their appropriate shapes
    :param obs_space_dict: (Dict[str: gym.Space])
    :param normalize_images: (bool) A boolean for whether image spaces in general should be normalized
    :param features_dim: (int) The dimension of features that should come out of this feature extractor
    :param cnn_extractor_class: (BaseFeaturesExtractor) A feature extractor class to apply to any
    spaces that are inferred to be CNN-compatible
    :param mlp_net_arch: (Iterable) An iterable of dimensions, where each corresponds to the size of
    a hidden layer in the MLP that will be used to process concatenated MLP-compatible observations
    :param mlp_feature_dim: (int) The dimension of features that will come out of the MLP
    on your MLP-compatible observations
    :param embedding_dim: (int) The dimension into which each of your embedding-compatible observations
    will be embedded
    """
    def __init__(self,
                 observation_space: gym.Space,
                 obs_unwrapper_function: Callable,
                 obs_space_dict:  Dict[str, gym.Space],
                 normalize_images: bool,
                 features_dim: int = 20,
                 cnn_extractor_class: BaseFeaturesExtractor = NatureCNN,
                 cnn_feature_dim: int = 12,
                 mlp_net_arch: Iterable = (4,),
                 mlp_feature_dim: int = 6,
                 embedding_dim: int = 6

                 ):
        super().__init__(observation_space, obs_unwrapper_function, obs_space_dict,
                         normalize_images, features_dim)

        # This gets the string obs spaces associated with each extractor
        # They're stored in a nested _ separated string
        self.split_chars = "__"
        self.inferred_extractor_mapping = self.recursive_space_infer(obs_space_dict)
        self.cnn_spaces = self.inferred_extractor_mapping['CNN']
        self.mlp_spaces = self.inferred_extractor_mapping['MLP']
        self.embed_spaces = self.inferred_extractor_mapping['EMBED']


        _cnn_extractors = []
        total_flattened_dim = 0

        # Create CNN extractors
        for space_designation in self.cnn_spaces:
            cnn_space = recursive_lookup_from_string(obs_space_dict,
                                                     space_designation,
                                                     self.split_chars)
            assert is_image_space(cnn_space)
            _cnn_extractors.append(cnn_extractor_class(cnn_space, cnn_feature_dim))
            total_flattened_dim += cnn_feature_dim
        self.cnn_extractors = nn.ModuleList(_cnn_extractors)

        # Create MLP Extractor
        total_mlp_dim = 0
        if len(self.mlp_spaces) > 0:
            for space_designation in self.mlp_spaces:
                mlp_space = recursive_lookup_from_string(obs_space_dict,
                                                         space_designation,
                                                         self.split_chars)
                assert isinstance(mlp_space, gym.spaces.Box)
                # assume if the space is multi-dimensional, we'll flatten it
                # before sending it to a MLP
                n_dim = int(np.prod(mlp_space.shape))
                total_mlp_dim += n_dim
            self.mlp_extractor = nn.Sequential(*create_mlp(total_mlp_dim,
                                                           mlp_feature_dim,
                                                           mlp_net_arch))
            total_flattened_dim += mlp_feature_dim
        else:
            self.mlp_extractor = None

        # Create Embed tables
        if len(self.embed_spaces) > 0:
            _embedding_tables = []
            for space_designation in self.embed_spaces:
                embed_space = recursive_lookup_from_string(obs_space_dict,
                                                           space_designation,
                                                           self.split_chars)
                assert isinstance(embed_space, gym.spaces.Discrete)
                space_n = embed_space.n
                _embedding_tables.append(nn.Embedding(embedding_dim=embedding_dim,
                                                      num_embeddings=space_n))
                total_flattened_dim += embedding_dim

            self.embedding_tables = nn.ModuleList(_embedding_tables)
        else:
            self.embedding_tables = None
        self.projection_layer = nn.Linear(total_flattened_dim, features_dim)

    def space_combine(self, outer, inner):
        if outer is None:
            return inner
        else:
            return f"{outer}{self.split_chars}{inner}"

    def recursive_space_infer(self, obs_space_dict, outer_space=None):
        """
        Iterates recursively over `obs_space_dict` and determines whether each internal
        Space should be handled with a CNN extractor, by being passed into a MLP, or by
        being looked up in an embedding table.

        :param obs_space_dict:
        :param outer_space: Used for recursion; a string indicating the space name in which
        `recursive_space_infer` is being called
        :return: A dictionary with keys 'CNN', 'MLP' and 'EMBED', with each mapping to a list of
        obs space names that fall under each method. if the obs space is nested, the chain of
        nested keys will be concatenated together with `self.split_char` in between

        """
        extractor_mapping = defaultdict(list)
        for space_name, space in obs_space_dict.items():
            merged_space_name = self.space_combine(outer_space, space_name)
            if isinstance(space, OrderedDict):
                inner_mapping = self.recursive_space_infer(space, space_name)
                for k, v in inner_mapping.items():
                    extractor_mapping[k] += v
            elif isinstance(space, gym.spaces.Box):
                if is_image_space(space):
                    extractor_mapping['CNN'].append(merged_space_name)
                else:
                    extractor_mapping['MLP'].append(merged_space_name)
            elif isinstance(space, gym.spaces.Discrete):
                extractor_mapping['EMBED'].append(merged_space_name)
        return extractor_mapping

    def _dict_forward(self, observation_dict: dict) -> th.Tensor:
        flat_features = []
        for ind, space_designation in enumerate(self.cnn_spaces):
            cnn_extractor = self.cnn_extractors[ind]
            cnn_observation = recursive_lookup_from_string(observation_dict,
                                                           space_designation,
                                                           self.split_chars)
            cnn_features = cnn_extractor(cnn_observation)
            flat_features.append(cnn_features)

        if len(self.mlp_spaces) > 0:
            pre_mlp_features = []
            for ind, space_designation in enumerate(self.mlp_spaces):
                mlp_observation = recursive_lookup_from_string(observation_dict,
                                                               space_designation,
                                                               self.split_chars)
                if len(mlp_observation.shape) == 1:
                    batch_size = mlp_observation.shape[0]
                    reshaped_obs = th.reshape(mlp_observation, shape=(batch_size, 1))
                else:
                    # TODO Make this able to deal with 2D batch sizes
                    reshaped_obs = th.flatten(mlp_observation, start_dim=1)
                pre_mlp_features.append(reshaped_obs)
            merged_mlp_features = th.cat(pre_mlp_features, dim=-1)
            mlp_features = self.mlp_extractor(merged_mlp_features)
            flat_features.append(mlp_features)

        for ind, space_designation in enumerate(self.embed_spaces):
            embedding_table = self.embedding_tables[ind]
            embed_observation = recursive_lookup_from_string(observation_dict,
                                                             space_designation,
                                                             self.split_chars)
            embed_observation = th.argmax(embed_observation, dim=1)
            embedding = embedding_table(embed_observation)
            flat_features.append(embedding)

        merged_flat_features = th.cat(flat_features, dim=-1)
        final_features = self.projection_layer(merged_flat_features)
        return final_features
