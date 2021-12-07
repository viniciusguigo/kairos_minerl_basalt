import gym
import torch as th
from basalt_baselines.bc import bc_baseline, WRAPPERS as bc_wrappers
from stable_baselines3.common.utils import get_device
import numpy as np
import pandas as pd
import cv2
import sys
import os
import time
import datetime as dt
from minerl.herobraine.wrappers.video_recording_wrapper import VideoRecordingWrapper
from basalt_utils.wrappers import SaveObsAndActions

from kairos_minerl.agent_utils import KAIROS_GUI, KAIROS_StateMachine, KAIROS_Vision, KAIROS_Odometry
from kairos_minerl.behavior_cloner import BehaviorCloning, BehaviorCloning_128
from kairos_minerl.state_classifier import StateMachineClassifier

from basalt_utils import utils
import basalt_utils.wrappers as wrapper_utils


KAIROS_WRAPPERS = [# Maps from a string version of enum (found in the dataset) to an int version (expected for spaces.Discrete)
            (wrapper_utils.EnumStrToIntWrapper, dict()),
            # Flattens a Dict action space into a Box, but retains memory of how to expand back out
            (wrapper_utils.ActionFlatteningWrapper, dict()),
            ] #,

# OPERATION MODE
MODEL_OP_MODE = os.getenv('MODEL_OP_MODE', None)
EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME', "PAPER_EVALUATION")
ENABLE_DEBUG_GUI = bool(os.getenv('ENABLE_DEBUG_GUI', 'False'))

# TRAINED MODELS
MODEL_STATE_CLASSIFIER = "train/state_classifier/best_state_classifier_dict.pth"

MODEL_FIND_CAVE_NAV = "train/bc_model_navigation_FINAL/find_cave_best_BC_model_dict.pth"
MODEL_MAKE_WATERFALL_NAV = "train/bc_model_navigation_FINAL/make_waterfall_best_BC_model_dict.pth"
MODEL_ANIMAL_PEN_NAV = "train/bc_model_navigation_FINAL/animal_pen_best_BC_model_dict.pth"
MODEL_VILLAGE_HOUSE_NAV = "train/bc_model_navigation_FINAL/village_house_best_BC_model_dict.pth"

MODEL_FIND_CAVE_NAV_NUM_CLASSES = 13
MODEL_MAKE_WATERFALL_NAV_NUM_CLASSES = 13
MODEL_ANIMAL_PEN_NAV_NUM_CLASSES = 13
MODEL_VILLAGE_HOUSE_NAV_NUM_CLASSES = 13

# MODEL_FIND_CAVE = "train/bc_baseline/find_cave_best_BC_model_dict.pth"
MODEL_FIND_CAVE = "train/bc_model_full_FINAL/find_cave/find_cave_BC_v1_e28_l0.621_a0.805_dict.pth"
MODEL_MAKE_WATERFALL = "train/bc_baseline/make_waterfall_best_BC_model_dict.pth"
MODEL_ANIMAL_PEN = "train/bc_baseline/animal_pen_best_BC_model_dict.pth"
MODEL_VILLAGE_HOUSE = "train/bc_baseline/village_house_best_BC_model_dict.pth"

MODEL_FIND_CAVE_NUM_CLASSES = 13
MODEL_MAKE_WATERFALL_NUM_CLASSES = 17
MODEL_ANIMAL_PEN_NUM_CLASSES = 18
MODEL_VILLAGE_HOUSE_NUM_CLASSES = 40


class EpisodeDone(Exception):
    pass


class Episode(gym.Env):
    """A class for a single episode."""
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._done = False

    def reset(self):
        if not self._done:
            return self.env.reset()

    def step(self, action):
        s, r, d, i = self.env.step(action)
        if d:
            self._done = True
            raise EpisodeDone()
        else:
            return s, r, d, i

    def wrap_env(self, wrappers):
        for wrapper, kwargs in wrappers:
            self.env = wrapper(self.env, **kwargs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space


class MineRLAgent():
    """
    To compete in the competition, you are required to implement the two
    functions in this class:
        - load_agent: a function that loads e.g. network models
        - run_agent_on_episode: a function that plays one game of MineRL
    By default this agent behaves like a random agent: pick random action on
    each step.
    """

    def load_agent(self):
        """
        This method is called at the beginning of the evaluation.
        You should load your model and do any preprocessing here.
        THIS METHOD IS ONLY CALLED ONCE AT THE BEGINNING OF THE EVALUATION.
        DO NOT LOAD YOUR MODEL ANYWHERE ELSE.
        """
        # This is a random agent so no need to do anything
        # YOUR CODE GOES HERE
        pass

    def run_agent_on_episode(self, single_episode_env: Episode):
        """This method runs your agent on a SINGLE episode.
        You should just implement the standard environment interaction loop here:
            obs  = env.reset()
            while not done:
                env.step(self.agent.act(obs))
                ...
        Args:
            env (gym.Env): The env your agent should interact with.
        """
        # An implementation of a random agent
        # YOUR CODE GOES HERE
        _ = single_episode_env.reset()
        done = False
        steps = 0
        min_steps = 500
        while not done:
            random_act = single_episode_env.action_space.sample()
            if steps < min_steps and random_act['equip'] == 'snowball':
                random_act['equip'] = 'air'
            single_episode_env.step(random_act)
            steps += 1


class MineRLBehavioralCloningAgent(MineRLAgent):
    def load_agent(self):
        # TODO not sure how to get us to be able to load the policy from the right agent here
        self.policy = th.load("train/trained_policy.pt", map_location=th.device(get_device('auto')))
        self.policy.eval()

    def run_agent_on_episode(self, single_episode_env : Episode):
        # TODO Get wrappers actually used in BC training, and wrap environment with those
        single_episode_env.wrap_env(bc_wrappers)
        obs = single_episode_env.reset()
        done = False
        while not done:

            action, _, _ = self.policy.forward(th.from_numpy(obs.copy()).unsqueeze(0).to(get_device('auto')))
            try:
                if action.device.type == 'cuda':
                    action = action.cpu()
                obs, reward, done, _ = single_episode_env.step(np.squeeze(action.numpy()))
            except EpisodeDone:
                done = True
                continue


class KAIROS_MineRLAgent(MineRLAgent):
    """
    To compete in the competition, you are required to implement the two
    functions in this class:
        - load_agent: a function that loads e.g. network models
        - run_agent_on_episode: a function that plays one game of MineRL

    By default this agent behaves like a random agent: pick random action on
    each step.
    """
    def __init__(self, env_name, debug_mode=ENABLE_DEBUG_GUI):
        self.env_name = env_name
        self.debug_mode = debug_mode
        self.steps_switch_subtask = 5 # should be >= 5 otherwise won't properly trigger important subtasks
        # default BC values when not loading models
        self.bc_num_classes = 0
        self.bc_model = None

        # setup state classifier
        self.state_classifier_model_dict_addr = MODEL_STATE_CLASSIFIER

        # define task and bc models
        if self.env_name == "MineRLBasaltFindCaveHighRes-v0" or self.env_name == "MineRLBasaltFindCave-v0":
            self.task = "find_cave"
            if MODEL_OP_MODE == "bc_only":
                self.bc_model_addr = MODEL_FIND_CAVE
                self.bc_num_classes = MODEL_FIND_CAVE_NUM_CLASSES
            elif MODEL_OP_MODE == "hybrid_navigation":
                self.bc_model_addr = MODEL_FIND_CAVE_NAV
                self.bc_num_classes = MODEL_FIND_CAVE_NAV_NUM_CLASSES

        elif self.env_name == "MineRLBasaltMakeWaterfallHighRes-v0" or self.env_name == "MineRLBasaltMakeWaterfall-v0":
            self.task = "make_waterfall"
            if MODEL_OP_MODE == "bc_only":
                self.bc_model_addr = MODEL_MAKE_WATERFALL
                self.bc_num_classes = MODEL_MAKE_WATERFALL_NUM_CLASSES
            elif MODEL_OP_MODE == "hybrid_navigation":
                self.bc_model_addr = MODEL_MAKE_WATERFALL_NAV
                self.bc_num_classes = MODEL_MAKE_WATERFALL_NAV_NUM_CLASSES

        elif self.env_name == "MineRLBasaltCreateVillageAnimalPenHighRes-v0" or self.env_name == "MineRLBasaltCreateVillageAnimalPen-v0":
            self.task = "create_pen"
            if MODEL_OP_MODE == "bc_only":
                self.bc_model_addr = MODEL_ANIMAL_PEN
                self.bc_num_classes = MODEL_ANIMAL_PEN_NUM_CLASSES
            elif MODEL_OP_MODE == "hybrid_navigation":
                self.bc_model_addr = MODEL_ANIMAL_PEN_NAV
                self.bc_num_classes = MODEL_ANIMAL_PEN_NAV_NUM_CLASSES

        elif self.env_name == "MineRLBasaltBuildVillageHouseHighRes-v0" or self.env_name == "MineRLBasaltBuildVillageHouse-v0":
            self.task = "build_house"
            if MODEL_OP_MODE == "bc_only":
                self.bc_model_addr = MODEL_VILLAGE_HOUSE
                self.bc_num_classes = MODEL_VILLAGE_HOUSE_NUM_CLASSES
            elif MODEL_OP_MODE == "hybrid_navigation":
                self.bc_model_addr = MODEL_VILLAGE_HOUSE_NAV
                self.bc_num_classes = MODEL_VILLAGE_HOUSE_NAV_NUM_CLASSES

        else:
            raise ValueError("Invalid operation mode.")

        # Add a wrapper to the environment that records video and saves it in the
        # the `save_dir` we have constructed for this run.
        save_dir = "train"
        self.wrappers = [(VideoRecordingWrapper, {'video_directory':
                                                os.path.join(save_dir, 'videos')}),
                    (SaveObsAndActions, {'save_dir':
                                            os.path.join(save_dir, 'obs_and_actions')})] + KAIROS_WRAPPERS

    def load_agent(self):
        """
        This method is called at the beginning of the evaluation.
        You should load your model and do any preprocessing here.
        THIS METHOD IS ONLY CALLED ONCE AT THE BEGINNING OF THE EVALUATION.
        DO NOT LOAD YOUR MODEL ANYWHERE ELSE.
        """
        # Loading models in PyTorch:
        # (dict with weights: https://github.com/bentoml/BentoML/issues/612#issuecomment-620736609)
        self.device = th.device(get_device('auto'))

        # loads state classifier model
        input_shape = (3, 64, 64)
        num_classes = 13
        self.state_classifier_model = StateMachineClassifier(input_shape, num_classes)
        self.state_classifier_model.load_state_dict(th.load(self.state_classifier_model_dict_addr))
        self.state_classifier_model.to(self.device)
        self.state_classifier_model.eval()

        # load models behavior cloning model for the task
        if MODEL_OP_MODE == "bc_only" or MODEL_OP_MODE == "hybrid_navigation":
            self.bc_model = BehaviorCloning(action_dim=int(self.bc_num_classes))
            self.bc_model.load_state_dict(th.load(self.bc_model_addr))
            self.bc_model.to(self.device)
            self.bc_model.eval()

    def postprocess_obs(self, obs):
        # Only use image data
        obs = obs['pov'].squeeze().astype(np.float32)
        # Transpose observations to be channel-first (BCHW instead of BHWC)
        obs = obs.transpose(2, 0, 1)
        # Normalize observations
        obs /= 255.0
        return obs


    def run_agent_on_episode(self, single_episode_env: Episode):
        """This method runs your agent on a SINGLE episode.

        You should just implement the standard environment interaction loop here:
            obs  = env.reset()
            while not done:
                env.step(self.agent.act(obs))
                ...

        Args:
            env (gym.Env): The env your agent should interact with.
        """
        # add all environment wrappers (record video, discretize camera, etc)
        single_episode_env.wrap_env(self.wrappers)

        # reset all agent's internal modules
        self._initialize_agent(env=single_episode_env, bc_model=self.bc_model)
        env_step = 0

        # evaluate agent
        obs = single_episode_env.reset()
        agent_ready_obs = self.postprocess_obs(obs)
        
        done = False
        while not done:
            # get vision features from the current observation
            state_classifier = self.vision.extract_features(agent_ready_obs)

            # check which subtask to follow
            if MODEL_OP_MODE == "bc_only":
                # do not follow subtasks or state-machine when using behavior cloning only
                self.state_machine.subtask = 'behavior_cloning'
                self.state_machine.env_step = env_step
                self.bc_in_control = True
                action = self.state_machine.compute_bc_action(agent_ready_obs)

            elif MODEL_OP_MODE == "hybrid_navigation" or MODEL_OP_MODE == "engineered_only":
                if env_step % self.steps_switch_subtask == 0:
                    self.state_machine.update_subtask(state_classifier, env_step)

                # compute actions based on vision features and current observation (state-machine)
                action = self.state_machine.compute_action(agent_ready_obs, state_classifier, env_step)

            # if intervention mode, overwrite action selected by the agent
            if self.gui.intervention_mode and self.gui.intervention_key is not None:
                action = self.gui.compute_intervention_action()

            try:
                if action.device.type == 'cuda':
                    action = action.cpu()

                # step environment
                obs, reward, done, _ = single_episode_env.step(np.squeeze(action.numpy()))
                agent_ready_obs = self.postprocess_obs(obs)
                env_step += 1                

                # update odometry
                self.odometry.update(action, env_step, state_classifier)

                if self.debug_mode:
                    # generate odometry frame
                    odom_frame = self.odometry.generate_frame()

                    # display pov frame together with odometry
                    self.gui.display_step(
                        obs['pov'], state_classifier, action, self.state_machine.subtask, odom_frame)

            except EpisodeDone:
                done = True

                if self.debug_mode:
                    self.gui.close()
                    self.odometry.close()

                continue


    def _initialize_agent(self, env, bc_model):
        # Initialize all modules
        exp_id = f'{EXPERIMENT_NAME}_{self.task}_{MODEL_OP_MODE}_{dt.datetime.now()}'

        self.vision = KAIROS_Vision(
            state_classifier_model=self.state_classifier_model,
            device=self.device)

        self.gui = KAIROS_GUI(exp_id=exp_id)

        self.odometry = KAIROS_Odometry(
            exp_id=exp_id,
            state_classifier_map=self.vision.map)

        self.state_machine = KAIROS_StateMachine(
            env=env,
            env_name=self.env_name,
            odometry=self.odometry,
            bc_model=bc_model,
            bc_num_classes=self.bc_num_classes,
            device=self.device)

    