from gym import spaces, ObservationWrapper
from .reversible_action_wrapper import ReversibleActionWrapper
from functools import partial
from typing import Union
import numpy as np
from collections import OrderedDict
import torch as th
from stable_baselines3.common.utils import get_device


# The below two methods are just defining a shared function interface for doing
# either tensorflow or numpy reshape
def numpy_reshape(arr, shape):
    if len(shape) == 0:
        return arr[0]
    return arr.reshape(shape)


def th_reshape(arr, shape):
    if len(shape) == 0:
        return arr[0]
    return th.reshape(arr, shape)

np_dtypes_dict = {
    'int': np.int32,
    'float': np.float32
}

th_dtypes_dict = {
    'int': th.IntTensor,
    'float': th.FloatTensor
}


def numpy_cast(arr, cast_type):
    return arr.astype(np_dtypes_dict[cast_type])


def th_cast(arr, cast_type):
    return arr.type(th_dtypes_dict[cast_type]).to(get_device('auto'))


class WrappedBox(spaces.Box):
    def __init__(self, low, high, sample_func, flatten_func):
        super(WrappedBox, self).__init__(low=low, high=high)
        self.sample_func = sample_func
        self.flatten_func = flatten_func

    def sample(self):
        structured_sample = self.sample_func()
        return self.flatten_func(structured_sample)


def is_dict_tuplelike(dct):
    # If the sorted values of the dictionary keys are set-wise equivalent to an integer range of the same length
    return set(dct.keys()) == set(range(len(dct)))


def get_ndim_recursive(shape_info: dict):
    """
    Takes a possibly nested dictionary of shape information and returns the
    dimensionality of an array that is all of these shapes flattened and concatenated together

    :param shape_info: Dictionary of the form {'space_name': (shape), 'dict_space': {'inner_space': (shape}}
    :return: integer value containing the length of the flattened form of all shapes
    """
    total = 0
    if isinstance(shape_info, dict):
        for k, v in shape_info.items():
            total += get_ndim_recursive(v)
    else:
        total += np.prod(shape_info).astype(np.int64)
    return total


def extract_from_flattened(concat_arr: Union[np.ndarray, th.Tensor],
                           shape_dict: OrderedDict,
                           type_dict: OrderedDict = None,
                           is_tuple=False,
                           verbose=False):
    """
    Takes in an array that has been flattened from a possibly-nested dictionary, and inflate it back out into
    dictionary form. Note that this requires shape_dict to be an OrderedDict, because we need to specify the order in
    which elements of the dictionary were concatenated. Recursion handles unflattening a sub-array that corresponds to a
    sub-dict within a nested dict space.

    :param concat_arr: Observations flattened to a 1D array, in either array or tensor form
    :param shape_dict: An OrderedDict giving the original shapes of each space that has been collapsed down
    into the flattened array, in the correct order.
    :param type_dict: An OrderedDict giving the type expected by each space
    :param is_tuple: A flag for whether to inflate into a Tuple rather than the default Dict
    :return:
    """
    reshaped_elements = {}
    if type_dict is None:
        type_dict = {}
    ind = 0

    if isinstance(concat_arr, np.ndarray):
        reshape_op = numpy_reshape
        cast_op = numpy_cast


    elif isinstance(concat_arr, th.Tensor):
        reshape_op = th_reshape
        cast_op = th_cast
    else:
        raise TypeError("concat_arr must be either a numpy array or a tensor")

    for space_name, shape_info in shape_dict.items():
        flattened_extent = get_ndim_recursive(shape_info)
        array_subset = concat_arr[..., ind:ind + flattened_extent]
        if verbose:
            print("For space {}, array subset {} of extent {}, from index {} to {}".format(space_name, array_subset, flattened_extent, ind, ind+flattened_extent))

        if isinstance(array_subset, th.Tensor):
            shape = list(array_subset.shape)
            if shape[0] is None:
                shape[0] = -1
        else:  # numpy
            shape = array_subset.shape
        non_final_dimensions = list(shape[:-1])

        if isinstance(shape_info, OrderedDict):

            reshaped_elements[space_name] = extract_from_flattened(array_subset, shape_info,
                                                                   type_dict=type_dict.get(space_name, None),
                                                                   is_tuple=is_dict_tuplelike(shape_info))
        else:
            inflated_shape = non_final_dimensions + list(shape_info)
            try:
                if type_dict.get(space_name, None):
                    typed_subset = cast_op(array_subset, type_dict[space_name])
                else:
                    typed_subset = array_subset
                reshaped_elements[space_name] = reshape_op(typed_subset, inflated_shape)

            except ValueError as e:
                raise(ValueError("Space: {}, Shape: {}, Subset Shape: {}, Error: {}".format(space_name, concat_arr.shape, array_subset.shape, e)))
        ind += flattened_extent
    if is_tuple:
        return tuple(reshaped_elements.values())
    else:
        return reshaped_elements

def infer_batch_dimensions(original_shape, current_shape):
    num_original_dims = len(original_shape)
    if num_original_dims > 0:
        assert current_shape[-num_original_dims:] == tuple(original_shape),\
            f"Original shape {original_shape}, " \
            f"current inferred shape {current_shape[-num_original_dims:]}"
        batch_dims = current_shape[:-num_original_dims]
        return batch_dims
    else:
        # if original_shape was a scalar
        return current_shape


def flatten(structured_input, original_shapes):
    """
    Flattens an observation or action that is a (possibly nested) Dict or Tuple into a 1D array. This is done corresponding to the order
    described in (possibly nested) OrderedDict original_shapes. (Note that the values of original_shapes aren't
    actually used here, we just use it as a definition of ordering because it is a side effect created by the
    initialization process and it's easier than creating a separate keys-only artifact for use in this method)

    :param structured_observation: A tuple or dictionary of observation values
    :param original_shapes: OrderedDict specifying spaces and their shapes
    :return:
    """

    flattened = []
    batch_dim = None
    # Figure out what the batch dimension is. You can do this by doing a diff between the dimensions of the shapes of
    # each key in `structured_input` and their dimensions in `original_shapes`. Any dimension in that diff will be a
    # batch dimensions. When you flatten, you want to only flatten dimensions subsequent to this one, and
    # when you concatenate, you want to concatenate along this dimension
    # So, something that is (1, 100, 3, 3) should become (1, 100, 9) and that merged with something else should be (1, 100, 10)
    # concatenate

    if not isinstance(structured_input, dict):
        structured_input = dict(enumerate(structured_input))
    for space in original_shapes.keys():
        # check whether original_shapes[space] is a dict or tuple
        assert space in structured_input, f"Space {space} found in original_shapes, but not structured_input. " \
                                          f"original_shapes keys: {original_shapes.keys()}, structured_input keys: {structured_input.keys()}"
        if isinstance(structured_input[space], dict) or isinstance(structured_input[space], tuple):
            flattened.append(flatten(structured_input[space], original_shapes[space]))
        else:

            if np.isscalar(structured_input[space]):
                reshaped_input = [structured_input[space]]
            else:
                structured_input_shape = structured_input[space].shape
                new_batch_dim = infer_batch_dimensions(original_shapes[space], structured_input_shape)

                if batch_dim is not None:
                    if batch_dim != new_batch_dim: #f"Found inconsistent batch dimension {new_batch_dim} for space {space}"
                        break
                else:
                    batch_dim = new_batch_dim
                reshape_dims = batch_dim + (-1,)
                reshaped_input = structured_input[space].reshape(reshape_dims)

            flattened.append(reshaped_input)
    return np.concatenate(flattened, axis=-1)


def infer_flattened_dims_of_gym_space(master_gym_space: Union[spaces.Tuple, spaces.Dict]):
    """
    Infers the flattened shape that will be needed to represent all observation spaces within the existing obs_space.
    Does so recursively, so if you start out with a Dict space that contains a Tuple or Dict space inside it,
    it will unroll those as well.

    :param master_gym_space: A gym.spaces.Dict or gym.spaces.Tuple
    :return: The full shape, flattened vectors containing high and low values, and a (possibly nested) OrderedDict

    containing information about shapes of spaces, in a canonical order. Note that this will be dictionary even if the
    original space was a Tuple, because a dictionary is more general (can have int keys to mock-correspond to a Tuple)
    """
    # TODO add back in ability to do whitelisted dimensions
    if isinstance(master_gym_space, spaces.Tuple):
        space_names = range(len(master_gym_space))
        subspaces = master_gym_space.spaces
    elif isinstance(master_gym_space, spaces.Dict):
        space_names = master_gym_space.spaces.keys()
        subspaces = master_gym_space.spaces.values()
    else:
        raise TypeError("master_gym_space must be either a Tuple or Dict")

    original_shapes = OrderedDict()
    highs = []
    lows = []
    flattened_shapes = []
    ordered_original_spaces = OrderedDict()
    types_dict = OrderedDict()
    for space_name, space in zip(space_names, subspaces):

        if isinstance(space, spaces.Box):
            original_shapes[space_name] = np.atleast_1d(space.shape)
            ordered_original_spaces[space_name] = space
            types_dict[space_name] = 'float'
            flattened_shapes.append(np.prod(original_shapes[space_name]).astype(np.int64))
            low = space.low
            high = space.high
            if low.shape == 0:
                low = np.full(original_shapes[space_name], low)
                high = np.full(original_shapes[space_name], high)
            lows.append(low.flatten())
            highs.append(high.flatten())

        elif isinstance(space, spaces.Discrete):
            original_shapes[space_name] = np.atleast_1d(())
            ordered_original_spaces[space_name] = space
            types_dict[space_name] = 'int'
            flattened_shapes.append(1)
            lows.append([0])
            highs.append([space.n - 1])

        elif isinstance(space, spaces.MultiDiscrete):
            original_shapes[space_name] = space.nvec.shape
            ordered_original_spaces[space_name] = space
            types_dict[space_name] = 'int'
            flattened_shapes.append(np.prod(space.nvec.shape).astype(np.int64))
            lows.append(np.zeros(space.nvec.shape).flatten())
            highs.append((space.nvec.shape - 1).flatten())

        elif isinstance(space, spaces.Tuple) or isinstance(space, spaces.Dict):
            if isinstance(space, spaces.Dict):
                assert not is_dict_tuplelike(space.spaces), "Your dictionary's keys range from 0:N, which will cause it to be inferred to be a Tuple"
            flattened_shape, high, low, subspace_shapes, ordered_subspaces, subspace_types_dict = infer_flattened_dims_of_gym_space(space)
            flattened_shapes.append(flattened_shape)
            original_shapes[space_name] = subspace_shapes
            types_dict[space_name] = subspace_types_dict
            ordered_original_spaces[space_name] = ordered_subspaces
            highs.append(high)
            lows.append(low)

        else:
            raise TypeError(f"Gym space is of unsupported type {type(space)}")

    full_shape = (np.sum(flattened_shapes).astype(np.int64), )
    flattened_highs = np.concatenate(highs)
    flattened_lows = np.concatenate(lows)
    return full_shape, flattened_highs, flattened_lows, original_shapes, ordered_original_spaces, types_dict


class ObservationSelectionWrapper(ObservationWrapper):
    def __init__(self, env, index):
        super(ObservationSelectionWrapper, self).__init__(env)
        assert isinstance(self.env.observation_space, spaces.Dict) or isinstance(self.env.observation_space, spaces.Tuple)
        self.index = index
        self.observation_space = self.env.observation_space[self.index]

    def observation(self, obs):
        return obs[self.index]


class ObservationFlatteningWrapper(ObservationWrapper):
    """
    A wrapper that takes in an environment with a Dict or Tuple observation space, and externally exposes
    a flattened 1-dimensional Box observation space. Converts inner-space observations to flattened counterpart
    before returning them on step and reset.
    """
    def __init__(self, env):
        super(ObservationFlatteningWrapper, self).__init__(env)
        assert isinstance(self.env.observation_space, spaces.Dict) or isinstance(self.env.observation_space, spaces.Tuple)

        (flat_shape, high, low, self.original_observation_shapes,
         self.ordered_observation_spaces, self.observation_types_dict) = infer_flattened_dims_of_gym_space(self.env.observation_space)
        self.observation_space = spaces.Box(low=low, high=high)
        # These two properties are convenience functions that allow you to convert between
        # dict and array without having to pass around the shape information.
        self.get_structured_observation = partial(extract_from_flattened,
                                                  shape_dict=self.original_observation_shapes,
                                                  type_dict=self.observation_types_dict,
                                                  is_tuple=isinstance(self.env.observation_space, spaces.Tuple))

        self.flatten_observation = partial(flatten, original_shapes=self.original_observation_shapes)
        self.observation_space = WrappedBox(low=low, high=high,
                                            sample_func=env.observation_space.sample,
                                            flatten_func=self.flatten_observation)

    def observation(self, obs):
        return self.flatten_observation(obs)


class ActionFlatteningWrapper(ReversibleActionWrapper):
    """
    A wrapper that takes in an environment with a Dict or Tuple action space, and externally exposes a flattened
    1-dimensional Box action space. Converts flattened actions into their non-flat counterparts before internal env step
    """
    def __init__(self, env):
        super(ActionFlatteningWrapper, self).__init__(env)
        assert isinstance(self.env.action_space, spaces.Dict) or isinstance(self.env.action_space, spaces.Tuple)
        (flat_shape, high, low, self.original_action_shapes,
         self.ordered_action_spaces,  self.action_types_dict) = infer_flattened_dims_of_gym_space(self.env.action_space)

        self.get_structured_action = partial(extract_from_flattened,
                                             shape_dict=self.original_action_shapes,
                                             type_dict=self.action_types_dict,
                                             is_tuple=isinstance(self.env.action_space, spaces.Tuple))
        self.flatten_action = partial(flatten, original_shapes=self.original_action_shapes)
        self.action_space = WrappedBox(low=low, high=high,
                                       sample_func=env.action_space.sample,
                                       flatten_func=self.flatten_action)

    def reverse_action(self, action):
        return self.flatten_action(action)

    def action(self, action):
        return self.get_structured_action(action)


