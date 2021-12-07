from gym import Wrapper
import pathlib
import os
import numpy as np


class SaveObsAndActions(Wrapper):
    def __init__(self, env, save_dir):
        super().__init__(env)
        self.save_dir = pathlib.Path(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.episode_ind = None
        self.saved_actions = []
        self.saved_obs = []

    def reset(self):

        if self.episode_ind is None:
            self.episode_ind = 0
        else:

            self.episode_ind += 1

        self.saved_actions = []
        self.saved_obs = []
        obs = self.env.reset()
        self.saved_obs.append(obs)
        return obs

    def step(self, action):
        self.saved_actions.append(action)
        obs, rew, done, info = super().step(action)
        self.saved_obs.append(obs)
        if done:
            np.save(file=self.save_dir / f'episode_{self.episode_ind}_actions.npy',
                    arr=np.array(self.saved_actions))
            np.save(file=self.save_dir / f'episode_{self.episode_ind}_obs.npy',
                    arr=np.array(self.saved_obs))
        return obs, rew, done, info