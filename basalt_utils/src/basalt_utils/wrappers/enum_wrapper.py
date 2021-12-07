from .reversible_action_wrapper import ReversibleActionWrapper
from minerl.herobraine.hero import spaces as minerl_spaces
from copy import deepcopy
import numpy as np



def vec_lookup(a, my_dict):
    # Copied from SO here: https://stackoverflow.com/q/16992713
    return np.vectorize(my_dict.__getitem__)(a)

class EnumStrToIntWrapper(ReversibleActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # NOTE: This needs to be called on an env that is still a dictionary action space
        assert isinstance(env.action_space, minerl_spaces.Dict)
        self.str_int_lookups = dict()
        self.int_str_lookups = dict()

        for space_name, action_space in self.action_space.spaces.items():
            if isinstance(action_space, minerl_spaces.Enum):
                self.str_int_lookups[space_name] = action_space.value_map
                self.int_str_lookups[space_name] = {v:k for k,v in action_space.value_map.items()}

    @staticmethod
    def _lookup_alternate_form(action, lookup_table):
        assert isinstance(action, dict)
        new_action = deepcopy(action)
        for space_name, lookup_dict in lookup_table.items():
            if isinstance(action[space_name], np.ndarray):
                new_action[space_name] = vec_lookup(action[space_name], lookup_table[space_name])
            else:
                new_action[space_name] = lookup_table[space_name][action[space_name]]
        return new_action

    def action(self, action):
        return self._lookup_alternate_form(action, lookup_table=self.int_str_lookups)

    def reverse_action(self, action):
        return self._lookup_alternate_form(action, lookup_table=self.str_int_lookups)

