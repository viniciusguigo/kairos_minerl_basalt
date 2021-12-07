
import os

import gym
import atexit
import threading
import minerl
from basalt_baselines.bc import bc_baseline, WRAPPERS as bc_wrappers
import numpy as np
from test_submission_code import KAIROS_MineRLAgent, MineRLAgent, Episode, EpisodeDone, MineRLBehavioralCloningAgent
from minerl.herobraine.wrappers import downscale_wrapper
from basalt_utils.utils import wrap_env
import torch as th
# import coloredlogs
#coloredlogs.install(logging.DEBUG)


MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLBasaltFindCaveHighRes-v0')
MINERL_MAX_EVALUATION_EPISODES = int(os.getenv('MINERL_MAX_EVALUATION_EPISODES', 2))
# We only use one evaluation thread
EVALUATION_THREAD_COUNT = 1

####################
# EVALUATION CODE  #
####################


def main():
    # agent = MineRLBehavioralCloningAgent()
    agent = KAIROS_MineRLAgent(env_name=MINERL_GYM_ENV)
    agent.load_agent()

    assert MINERL_MAX_EVALUATION_EPISODES > 0
    assert EVALUATION_THREAD_COUNT > 0

    env = gym.make(MINERL_GYM_ENV)
    # Bit of sanity check
    if env.observation_space['pov'].shape[0] != 1024:
        raise RuntimeError('The MineRL environment should be a "HighRes" variant.')
    # Apply downscale wrapper to turn (1024, 1024) observations into (64, 64)
    env = downscale_wrapper.DownscaleWrapper(env)

    # Ensure that videos are closed properly

    @atexit.register
    def cleanup_env():
        env.close()

    # A simple function to evaluate on episodes!
    def evaluate(i, env):
        print("[{}] Starting evaluator.".format(i))
        for i in range(MINERL_MAX_EVALUATION_EPISODES):
            try:
                agent.run_agent_on_episode(Episode(env))
            except EpisodeDone:
                print("[{}] Episode complete".format(i))
                pass

    thread = threading.Thread(target=evaluate, args=(0, env))
    thread.start()
    thread.join()


if __name__ == "__main__":
    main()
