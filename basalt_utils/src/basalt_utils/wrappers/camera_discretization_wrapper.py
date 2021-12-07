from .reversible_action_wrapper import ReversibleActionWrapper
from gym import spaces
from copy import deepcopy
from functools import partial
import numpy as np


def _get_delta(index, camera_angle):
    """
    This operation takes an index into the three-value discrete space (constant, increase, decrease) and
    converts that into a continuous camera value.

    This is done quite simply, by returning `camera_angle` for an index of 1, and -1*camera_angle
    for an index of 2


    """
    if index == 0:
        return 0
    elif index == 1:
        return camera_angle
    elif index == 2:
        return -1*camera_angle
    else:
        raise ValueError(f"Unsupported value {index}")


def _discretize(numeric_delta, camera_angle):
    """
    This operation takes a continuous `numeric_delta` camera change value (in either pitch or yaw) and
    discretizes it to be either "no change", "increase" or "decrease".

    If abs(numeric_delta) > camera_angle, it gets effectively capped at `camera_angle`, since in this discretization scheme
    the maximum you can move the camera in any direction is `camera_angle`. For `numeric_delta` values with an absolute
    value < camera angle, they are either rounded to 0 or +- 1 based on the ratio between `numeric_delta`/`camera_angle`

    :param numeric_delta: The continuous camera change value
    :param camera_angle: The angle to which the continuous value will be discretized.
    :return:
    """
    positive = numeric_delta > 0
    if abs(numeric_delta) > camera_angle:
        magnitude = 1
    else:
        magnitude = round(abs(numeric_delta)/camera_angle)
    if magnitude == 0:
        return 0
    elif magnitude == 1 and positive:
        return 1
    elif magnitude == 1 and not positive:
        return 2


class CameraDiscretizationWrapper(ReversibleActionWrapper):
    """
    This class removes the continuous camera action, and replaces it with two discrete actions.
    Each discrete action controls movement of pitch or yaw respectively, and there are
    three possible actions: keep constant, increase by `camera_angle` and decrease by `camera_angle`

    When converting continuous camera actions into this schema, if the magnitude of the `camera` action is
    less than `camera_angle`, it will be either rounded up (to increase/decrease, depending on sign) or down (to constant)
    by using a normal round operation on the ratio of `camera` to `camera_angle`. Any `camera` actions larger than
    `camera_angle` will be represented as just an increase or decrease; you can think of this as taking the
    max of abs(`camera`) and `camera_angle`
    """
    def __init__(self, env, camera_angle=45/4):
        super().__init__(env)
        self.env = env
        self.inner_action_space = deepcopy(self.env.action_space)
        existing_action_spaces = self.inner_action_space.spaces
        del existing_action_spaces['camera']
        existing_action_spaces['camera_pitch'] = spaces.Discrete(3)
        existing_action_spaces['camera_yaw'] = spaces.Discrete(3)
        self.action_space = spaces.Dict(existing_action_spaces)
        self.discretization_func = np.vectorize(partial(_discretize, camera_angle=camera_angle))
        self.get_delta_func = np.vectorize(partial(_get_delta, camera_angle=camera_angle))

    def action(self, action):
        """
        This function translates an `action` in the action space of this wrapper (`camera_pitch` and `camera_yaw`)
        and converts it into the action space of the underlying environment (a single tuple `camera` action)

        :param action: An action sampled from the action space specified by this wrapper
        :return: An action that is consistent with the action space of the env underlying this wrapper
        """

        action_copy = deepcopy(action)
        pitch_action = action['camera_pitch']
        yaw_action = action['camera_yaw']
        del action_copy['camera_pitch']
        del action_copy['camera_yaw']

        pitch_change = self.get_delta_func(pitch_action)
        yaw_change = self.get_delta_func(yaw_action)
        action_copy['camera'] = [pitch_change, yaw_change]
        return action_copy

    def reverse_action(self, action):
        """
        This function translates an `action` in the action space of the env underlying this wrapper, and converts
        it into the action space of this wrapper (camera pitch and yaw)

        :param action: An action assumed to be sampled from a Dict space containing a `camera` action key
        :return: An action with a discretized `camera_pitch` and `camera_yaw` action as defined by this wrapper
        """
        action_copy = deepcopy(action)
        camera_action = action['camera']
        del action_copy['camera']
        discretized_pitch = self.discretization_func(camera_action[..., 0])
        discretized_yaw = self.discretization_func(camera_action[..., 1])
        action_copy['camera_pitch'] = discretized_pitch
        action_copy['camera_yaw'] = discretized_yaw
        return action_copy