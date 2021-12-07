import gym


class FrameSkip(gym.Wrapper):
    def __init__(self, env, n_repeat=4):
        """Repeat each action `n_repeat` times.

        `stable_baselines.common.atari_wrappers` has a variation of this wrapper
        that also max-pools the last two observations."""
        super().__init__(env)
        assert n_repeat > 0
        self.n_repeat = n_repeat

    def step(self, action):
        total_reward = 0.0
        obs = None
        done = None
        info = None
        skipped_obs = []
        for i in range(self.n_repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
            if i != self.n_repeat - 1:
                skipped_obs.append(obs)
        info['skipped_obs'] = skipped_obs
        return obs, total_reward, done, info
