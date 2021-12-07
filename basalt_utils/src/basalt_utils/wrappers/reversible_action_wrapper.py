from gym import ActionWrapper


class ReversibleActionWrapper(ActionWrapper):
    """
    The goal of this wrapper is to add a layer of functionality on top of the normal ActionWrapper,
    and specifically to implement a way to start:
    (1) Construct a wrapped environment, and
    (2) Take in actions in whatever action schema is dictated by the innermost env, and then apply all action
    transformations/restructurings in the order they would be applied during live environment steps:
    from the inside out

    This functionality is primarily intended for converting a dataset of actions stored in the action
    schema of the internal env into a dataset of actions stored in the schema produced by the applied set of
    ActionWrappers, so that you can train an imitation model on such a dataset and easily transfer to the action
    schema of the wrapped environment for rollouts, RL, etc.

    Mechanically, this is done by assuming that all ActionWrappers have a `reverse_action` implemented
    and recursively constructing a method to call all of the `reverse_action` methods from inside out.

    As an example:
        > wrapped_env = C(B(A(env)))
    If I assume all of (A, B, and C) are action wrappers, and I pass an action to wrapped_env.step(),
    that's equivalent to calling all of the `action` transformations from outside in:
        > env.step(A.action(B.action(C.action(act)))

    In the case covered by this wrapper, we want to perform the reverse operation, so we want to return:
        > C.reverse_action(B.reverse_action(A.reverse_action(inner_action)))

    To do this, the `wrap_action` method searches recursively for the base case where there are no more
    `ReversibleActionWrappers` (meaning we've either reached the base env, or all of the wrappers between us and the
    base env are not ReversibleActionWrappers) by checking whether `wrap_action` is implemented. Once we reach the base
    case, we return self.reverse_action(inner_action), and then call all of the self.reverse_action() methods on the way
    out of the recursion

    """
    def wrap_action(self, inner_action):
        """
        :param inner_action: An action in the format of the innermost env's action_space
        :return: An action in the format of the action space of the fully wrapped env
        """
        if hasattr(self.env, 'wrap_action'):
            return self.reverse_action(self.env.wrap_action(inner_action))
        else:
            return self.reverse_action(inner_action)

    def reverse_action(self, action):
        raise NotImplementedError("In order to use a ReversibleActionWrapper, you need to implement a `reverse_action` function"
                                  "that is the inverse of the transformation performed on an action that comes into the wrapper")