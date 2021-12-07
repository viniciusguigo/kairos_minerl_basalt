import os, sys

from numpy.lib.function_base import copy
import cv2
import numpy as np
import pandas as pd
import torch as th
from stable_baselines3.common.utils import get_device

from kairos_minerl.gail_wrapper import (
    ActionShaping_FindCave,
    ActionShaping_Waterfall,
    ActionShaping_Animalpen,
    ActionShaping_Villagehouse,
    ActionShaping_Navigation,
)


# OPERATION MODE
MODEL_OP_MODE = os.getenv('MODEL_OP_MODE', None)


class KAIROS_GUI():
    """
    Displays agent POV and internal states relevant when debugging.
    """
    def __init__(self, exp_id, save_video=True):
        self.resolution = 512 # pixels
        self.resolution_x = 1024 # pixels
        self.resolution_y = 512 # pixels
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.text_color = (255, 255, 255)
        self.thickness = 1
        self.waitkey_delay = 1
        self.intervention_mode = False
        self.intervention_key = None

        self.action_position = (int(0.01*self.resolution), int(0.05*self.resolution))
        self.y_vision_feature_offset = int(0.04*self.resolution)
        self.state_classifier_position = (int(0.01*self.resolution), int(0.1*self.resolution))
        self.subtask_text_position = (int(0.01*self.resolution), int(0.97*self.resolution))

        self.save_video = save_video

        # setup video
        self.out = cv2.VideoWriter(
            f'train/videos/kairos_minerl_{exp_id}.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 20, (self.resolution_x, self.resolution_y))
        self.out_original = cv2.VideoWriter(
            f'train/videos/original_{exp_id}.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 20, (self.resolution, self.resolution))

    def display_step(self, obs, state_classifier, action, subtask, odom_frame):
        # setup image to display
        obs_as_rgb_img = cv2.resize(obs, dsize=[self.resolution,self.resolution])

        # reverse blue and red channels
        red = obs_as_rgb_img[:,:,2].copy()
        blue = obs_as_rgb_img[:,:,0].copy()

        obs_as_rgb_img[:,:,0] = red
        obs_as_rgb_img[:,:,2] = blue

        # save original resized frame with no labels, odometry, etc
        if self.save_video:
            self.out_original.write(obs_as_rgb_img)

        # display actions
        obs_as_rgb_img = cv2.putText(obs_as_rgb_img, f'action: {action.data}', self.action_position, self.font, 
                   self.font_scale, self.text_color, self.thickness, cv2.LINE_AA)

        # display visual features
        y_vision_feature_step = 0.
        obs_as_rgb_img = cv2.putText(obs_as_rgb_img, 'state_classifier:', self.state_classifier_position, self.font, 
                   self.font_scale, self.text_color, self.thickness, cv2.LINE_AA)
        for key, value in state_classifier.items():
            y_vision_feature_step += 1.
            single_state_classifier_position = (int(0.01*self.resolution), int(0.1*self.resolution + y_vision_feature_step*self.y_vision_feature_offset))
            obs_as_rgb_img = cv2.putText(obs_as_rgb_img, f'  {key}: {value:.2f}', single_state_classifier_position, self.font, 
                    self.font_scale, self.text_color, self.thickness, cv2.LINE_AA)

        # display subtask
        obs_as_rgb_img = cv2.putText(obs_as_rgb_img, f'subtask: {subtask}', self.subtask_text_position, self.font, 
                   self.font_scale, self.text_color, self.thickness, cv2.LINE_AA)

        # display intervention mode indicator
        if self.intervention_mode:
            obs_as_rgb_img = cv2.putText(obs_as_rgb_img, 'INTERVENTION MODE', (self.subtask_text_position[0]+320, self.subtask_text_position[1]), self.font, 
                    self.font_scale, (0,0,255), self.thickness, cv2.LINE_AA)

        # concatenate odometry frame
        obs_as_rgb_img = np.concatenate((obs_as_rgb_img, odom_frame), axis=1)
        
        # display image
        cv2.imshow("KAIROS MineRL", obs_as_rgb_img)
        key_pressed = cv2.waitKey(self.waitkey_delay)
        if key_pressed != -1:
            self.intervention_key = chr(key_pressed)
            if self.intervention_key == 'i':
                print('INTERVENTION TRIGGERED')
                self.waitkey_delay = 1-self.waitkey_delay  # flips between 1 and 0
                self.intervention_mode = True if self.waitkey_delay==0 else False

        # save frame
        if self.save_video:
            self.out.write(obs_as_rgb_img)

    def close(self):
        cv2.destroyAllWindows()
        if self.save_video:
            self.out.release()
            self.out_original.release()

    def compute_intervention_action(self):
        """
        Table of Actions

        [0] "attack"
        [1] "back"
        [2] "camera_up_down" (float, negative is UP)
        [3] "camera_right_left" (float, negative is LEFT)
        [4] "equip"
        [5] "forward"
        [6] "jump"
        [7] "left"
        [8] "right"
        [9] "sneak"
        [10] "sprint"
        [11] "use"
        """
        action = th.zeros(12)

        # compute action based on intervention key
        if self.intervention_key == 'w': # move forward
            action[5] = 1
        elif self.intervention_key == 's': # move backward
            action[1] = 1
        elif self.intervention_key == 'a': # turn left (camera)
            action[3] = -10
        elif self.intervention_key == 'd': # turn right (camera)
            action[3] = 10
        elif self.intervention_key == 'q': # turn down (camera)
            action[2] = 10
        elif self.intervention_key == 'e': # turn up (camera)
            action[2] = -10
        elif self.intervention_key == ' ': # jump forward
            action[5] = 1
            action[6] = 1

        # equip a random food
        action[4] = np.random.choice([2,12,13]) 

        # reset key so it does not apply the same actin multiple times
        self.intervention_key = None

        return action


class KAIROS_StateMachine():
    """
    Controls sequence of sub-tasks to follow for each environemnt.
    """
    def __init__(self, env, env_name, odometry, bc_model, bc_num_classes, device):
        self.env = env
        self.env_name = env_name
        self.odometry = odometry
        self.bc_model = bc_model
        self.bc_num_classes = int(bc_num_classes)
        self.device = device
        self._initialize_mapping()
        self.subtask = 0
        self.executing_multistep_subtask = False
        self.bc_in_control = False

        # define task
        if self.env_name == "MineRLBasaltFindCaveHighRes-v0" or self.env_name == "MineRLBasaltFindCave-v0":
            self.task = "FIND_CAVE"
            self.setup_cave_task()

        elif self.env_name == "MineRLBasaltMakeWaterfallHighRes-v0" or self.env_name == "MineRLBasaltMakeWaterfall-v0":
            self.task = "MAKE_WATERFALL"            
            self.setup_waterfall_task()

        elif self.env_name == "MineRLBasaltCreateVillageAnimalPenHighRes-v0" or self.env_name == "MineRLBasaltCreateVillageAnimalPen-v0":
            self.task = "CREATE_PEN"
            self.setup_pen_subtask()

        elif self.env_name == "MineRLBasaltBuildVillageHouseHighRes-v0" or self.env_name == "MineRLBasaltBuildVillageHouse-v0":
            self.task = "BUILD_HOUSE"
            self.setup_house_subtask()

        else:
            raise ValueError("Invalid environment. Check environ.sh")

        # global states
        self.good_waterfall_view = False

        # setup behavior cloning action space
        self.setup_bc_actions()

        # setup consensus
        self.triggerred_consensus = False
        self.num_consensus_steps = 50
        self.consensus_steps = 0

        # setup task-specific subtasks
        self.allow_escape_water = True
        self.setup_escape_water_subtask()

        # setup tracking of open-space areas
        self.num_open_spaces = 5
        self.open_space_tracker = self.num_open_spaces*[0]

        # setup tracking of danger_ahead
        self.num_danger_aheads = 5
        self.danger_ahead_tracker = self.num_danger_aheads*[0]

        # setup tracking of has_animals
        self.num_has_animals = 5
        self.has_animals_tracker = self.num_has_animals*[0]

        # setup tracking of top_of_waterfall
        self.num_top_of_waterfall = 5
        self.top_of_waterfall_tracker = self.num_top_of_waterfall*[0]

        # keep track of vertical camera angle
        self.default_camera_angle = 10 # positive is down
        self.goal_camera_angle = self.default_camera_angle


    # translate bc actions to original dict
    def translate_bc_to_raw_actions(self, discrete_action):
        # reset actions
        action = th.zeros(12)
        
        if discrete_action != (self.bc_num_classes-1): # additional noop action
            # convert bc action from string format to number
            bc_actions = self.bc_action_map[discrete_action]

            for bc_action in bc_actions:
                if bc_action[0] != 'camera':
                    if bc_action[0] == 'equip':
                        action[self.action_str_to_int[bc_action[0]]] = self.item_map[bc_action[1]]
                    else:
                        action[self.action_str_to_int[bc_action[0]]] = bc_action[1]
                else:
                    # turn camera left/right
                    if bc_action[1][0] == 0:
                        action[3] = bc_action[1][1]
                    # turn camera up/down
                    elif bc_action[1][1] == 0:
                        action[2] = bc_action[1][1]

        return action


    def setup_bc_actions(self):
        # initialize action shaping class used to train model
        if MODEL_OP_MODE == "hybrid_navigation":
            action_shaping = ActionShaping_Navigation(env=self.env.env.env.env)

        elif self.task == "FIND_CAVE" and MODEL_OP_MODE == "bc_only":
            action_shaping = ActionShaping_FindCave(env=self.env.env.env.env)

        elif self.task == "MAKE_WATERFALL" and MODEL_OP_MODE == "bc_only":
            action_shaping = ActionShaping_Waterfall(env=self.env.env.env.env)

        elif self.task == "CREATE_PEN" and MODEL_OP_MODE == "bc_only":
            action_shaping = ActionShaping_Animalpen(env=self.env.env.env.env)

        elif self.task == "BUILD_HOUSE" and MODEL_OP_MODE == "bc_only":
            action_shaping = ActionShaping_Villagehouse(env=self.env.env.env.env)
        else:
            action_shaping = ActionShaping_Navigation(env=self.env.env.env.env)

        # setup translation from string to int
        self.action_str_to_int = {
            "attack": 0,
            "back": 1,
            "camera_up_down": 2,
            "camera_right_left": 3,
            "equip": 4,
            "forward": 5,
            "jump": 6,
            "left": 7,
            "right": 8,
            "sneak": 9,
            "sprint": 10,
            "use": 11,
        }
        self.bc_action_map = action_shaping._actions


    def subtask_find_goal(self, obs):
        # reset actions
        action = th.zeros(12)

        if MODEL_OP_MODE == 'engineered_only':
            self.bc_in_control = False

            # move forward with random chance of jumps
            action[5] = 1 # forward
            if np.random.rand() < 0.1:
                action[3] = -10 # turn camera left            
            elif np.random.rand() > 0.9:
                action[3] = 10 # turn camera right

            # randomly jump
            if np.random.rand() < 0.25:
                action[6] = 1 # jump
        
        elif MODEL_OP_MODE == 'hybrid_navigation':
            action = self.compute_bc_action(obs)
            self.bc_in_control = True
            
        return action

    def subtask_go_to_goal(self, obs):
        # reset actions
        action = th.zeros(12)

        if MODEL_OP_MODE == 'engineered_only':
            self.bc_in_control = False

            # move forward with random chance of jumps
            action[5] = 1 # forward
            if np.random.rand() < 0.1:
                action[3] = -10 # turn camera left            
            elif np.random.rand() > 0.9:
                action[3] = 10 # turn camera right

            # randomly jump
            if np.random.rand() < 0.5:
                action[6] = 1 # jump
        
        elif MODEL_OP_MODE == 'hybrid_navigation':
            action = self.compute_bc_action(obs)
            self.bc_in_control = True

            # # check if waterfall was placed to move on to next state
            # # (equipped and used water bucket)
            # if self.task == 'MAKE_WATERFALL' and action[11] == 1 and action[4] == 11:
            #     self.reached_top = True
            #     self.built_waterfall = True
            # else:
            #     self.reached_top = False
            #     self.built_waterfall = False

        return action

    def subtask_end_episode(self):
        # reset actions
        action = th.zeros(12)

        # throw snowball
        if self.task == "BUILD_HOUSE":
            action[4] = 22 # equip it
        else:
            action[4] = 8 # equip it
        action[11] = 1 # throw it

        return action

    def track_state_classifier(self, state_classifier):
        # Keep track of open spaces
        self.open_space_tracker.pop(0)  
        self.open_space_tracker.append(state_classifier['has_open_space'])
        self.odometry.good_build_spot = True if np.mean(self.open_space_tracker)>0.75 else False

        # Keep track of danger_ahead
        self.danger_ahead_tracker.pop(0)  
        self.danger_ahead_tracker.append(state_classifier['danger_ahead'])
        self.odometry.agent_swimming = True if np.mean(self.danger_ahead_tracker)>0.4 else False

        # Keep track of animals
        self.has_animals_tracker.pop(0)  
        self.has_animals_tracker.append(state_classifier['has_animals'])
        self.odometry.has_animals_spot = True if np.mean(self.has_animals_tracker)>0.8 else False

        # Keep track of when on top of waterfalls
        self.top_of_waterfall_tracker.pop(0)  
        self.top_of_waterfall_tracker.append(state_classifier['at_the_top_of_a_waterfall'])
        self.odometry.top_of_waterfall_spot = True if np.mean(self.top_of_waterfall_tracker)>0.8 else False


    def compute_bc_action(self, obs):
        # Forward pass through model
        obs = th.Tensor(obs).unsqueeze(0).to(self.device)
        
        # Note, latest model passes out logits, so a softmax is needed for probabilities
        scores = self.bc_model(obs)
        probabilities = th.nn.functional.softmax(scores)
        
        # Into numpy
        probabilities = probabilities.detach().cpu().numpy()

        # Sample action according to the probabilities
        discrete_action = np.random.choice(np.arange(self.bc_num_classes), p=probabilities[0])

        # translate discrete action to original action space
        action = self.translate_bc_to_raw_actions(discrete_action)

        # make sure we have a weapon equipped
        action[4] = self.item_map['stone_pickaxe'] # dont have shovel in build house task

        return action


    def compute_action(self, obs, state_classifier, env_step):
        """
        Table of Actions

        [0] "attack"
        [1] "back"
        [2] "camera_up_down" (float, negative is UP)
        [3] "camera_right_left" (float, negative is LEFT)
        [4] "equip"
        [5] "forward"
        [6] "jump"
        [7] "left"
        [8] "right"
        [9] "sneak"
        [10] "sprint"
        [11] "use"
        
        Table of Subtasks

        0: "find_goal",
        1: "go_to_goal",
        2: "end_episode",
        3: "climb_up",
        4: "climb_down",
        5: "place_waterfall",
        6: "look_around",
        7: "build_pen",
        8: "go_to_location",
        9: "lure_animals",
        10: "leave_pen",
        11: "infer_biome",
        12: "build_house",
        13: "tour_inside_house",
        14: "leave_house",

        """
        # track previous relevant classified states
        self.track_state_classifier(state_classifier)        

        # Consensus is a way to look around to gather more data and make sure
        # the state classifier is outputting the correct thing
        action = th.zeros(12)
        if self.triggerred_consensus:
            action = self.step_consensus(action, state_classifier)
        else:
            # avoid danger
            if self.subtask == 'escape_water':
                action = self.step_escape_water_subtask()
                return action

            # execute subtasks
            if self.subtask == 'build_house' and not self.house_built:
                action = self.step_house_subtask()
                return action
            elif self.subtask == 'build_pen' and not self.pen_built:
                action = self.step_pen_subtask()
                return action
            elif self.subtask == 'lure_animals' and self.pen_built:
                action = self.step_lure_animals_subtask(obs)
                return action
            elif self.subtask == 'climb_up' and not self.reached_top:
                action = self.step_climb_up_subtask(state_classifier)
                return action
            elif self.subtask == 'place_waterfall' and self.reached_top:
                action = self.subtask_place_waterfall()
                return action
            elif self.subtask == 'go_to_picture_location':
                action = self.step_go_to_picture_location()
                return action
            elif self.subtask == 'find_goal':
                action = self.subtask_find_goal(obs)
            elif self.subtask == 'go_to_goal':
                action = self.subtask_go_to_goal(obs)                    
            elif self.subtask == 'end_episode':
                action = self.subtask_end_episode()            

            # # TODO: find object direction
            # if not self.triggerred_consensus:
            #     self.consensus_states = {key: [] for key, value in state_classifier.items()}
            #     self.consensus_states['heading'] = []
            #     self.triggerred_consensus = True

            # Make sure camera angle is at the desired angle
            if not self.good_waterfall_view and not self.bc_in_control:
                action = self.update_vertical_camera_angle(action)

        return action
    

    def update_subtask(self, state_classifier, env_step):
        self.env_step = env_step
        
        if not self.executing_multistep_subtask:
            # PRIORITY: escape water
            if self.odometry.agent_swimming and self.allow_escape_water:
                self.subtask = 'escape_water'
                return

            if self.task == 'FIND_CAVE':
                # timeout to find cave
                if env_step > self.timeout_to_find_cave:
                    self.subtask = 'end_episode'
                    return

                if state_classifier['inside_cave'] > 0.9:
                    self.subtask = 'end_episode'
                    return

            if self.task == 'MAKE_WATERFALL':
                if self.good_waterfall_view and self.built_waterfall:
                    self.subtask = 'end_episode'
                    return

                if self.reached_top and self.built_waterfall:
                    self.subtask = 'go_to_picture_location'
                    self.allow_escape_water = False
                    return

                if self.reached_top:# and not self.bc_in_control:
                    self.subtask = 'place_waterfall'
                    self.allow_escape_water = False
                    return

                # timeout to climb and build waterfall
                if env_step > self.timeout_to_build_waterfall and not self.reached_top:# and not self.bc_in_control:
                    self.subtask = 'climb_up'
                    self.found_mountain = True
                    self.allow_escape_water = False

                    # # OVERWRITE PILLAR CONSTRUCTION
                    # self.reached_top = True
                    # self.subtask = 'place_waterfall'

                    return

                # triggers waterfall construction based on at_the_top_of_a_waterfall and facing_wall
                if state_classifier['at_the_top_of_a_waterfall'] > 0.5 and self.moving_towards_mountain:# and not self.bc_in_control:
                    self.subtask = 'climb_up'
                    self.found_mountain = True
                    self.allow_escape_water = False

                    # # OVERWRITE PILLAR CONSTRUCTION
                    # self.reached_top = True
                    # self.subtask = 'place_waterfall'

                    return

                if self.moving_towards_mountain:
                    self.subtask = 'go_to_goal'
                    return

                if state_classifier['has_mountain'] > 0.95 and not self.found_mountain:
                    self.subtask = 'go_to_goal'
                    self.moving_towards_mountain = True
                    return


            if self.task == 'CREATE_PEN':
                if self.odometry.good_build_spot and not self.pen_built:
                    self.subtask = 'build_pen'
                    self.adjusted_head_angle = False
                    return

                # timeout to start pen construction
                if env_step > self.timeout_to_build_pen and not self.pen_built:
                    self.subtask = 'build_pen'
                    self.adjusted_head_angle = False
                    return

                # lure animals after pen is built
                if self.pen_built and not self.animals_lured:
                    self.subtask = 'lure_animals'
                    return

                # end episode after pen is built and animals are lured
                if self.pen_built and self.animals_lured:
                    self.subtask = 'end_episode'
                    return

            
            if self.task == 'BUILD_HOUSE':
                # finishes episode after house is built
                if self.house_built:
                    self.subtask = 'end_episode'
                    return

                if self.odometry.good_build_spot and not self.house_built:
                    self.subtask = 'build_house'
                    self.adjusted_head_angle = False
                    self.allow_escape_water = False
                    return

                # timeout to start house construction
                if env_step > self.timeout_to_build_house and not self.house_built:
                    self.subtask = 'build_house'
                    self.adjusted_head_angle = False
                    self.allow_escape_water = False
                    return

            # default subtask: navigation
            if self.task != 'MAKE_WATERFALL':
                self.subtask = 'find_goal'
            else:
                self.subtask = 'go_to_goal'
                self.moving_towards_mountain = False
                if env_step > self.min_time_look_for_mountain:
                    self.moving_towards_mountain = True
                    self.found_mountain = True


    def update_vertical_camera_angle(self, action):
        # randomly bounces camera up (helps escaping shallow holes)
        if np.random.rand() < 0.10:
            self.goal_camera_angle = -15
            action[6] = 1 # jump
        else:
            self.goal_camera_angle += 1
            self.goal_camera_angle = np.clip(
                self.goal_camera_angle, -15, self.default_camera_angle)

            # use high camera angles to continue jumping
            if self.goal_camera_angle < -10:
                action[6] = 1
            
        action[2] = self.goal_camera_angle-self.odometry.camera_angle
        return action

    def subtask_turn_around(self, action):
        action[5] = 0 # dont move forward
        action[3] = 15 # turn camera
        return action

    def subtask_place_waterfall(self):
        print('Placing waterfall')
        action = th.zeros(12)
        action[2] = 50 # look down
        action[4] = 11 # equip water bucket
        action[11] = 1 # use it
        self.built_waterfall = True

        return action

    def step_escape_water_subtask(self):
        action = th.zeros(12)
        if self.escape_water_step < self.total_escape_water_steps:
            action[5] = 1 # move forward
            # if np.random.rand() < 0.1:
            action[6] = 1 # jump
            if self.task == 'BUILD_HOUSE':
                action[4] = self.item_map['stone_pickaxe'] # dont have shovel in build house task
            else:
                action[4] = self.item_map['stone_shovel'] # equip shovel (breaks dirt/sand blocks if we fall in the hole)
            action[0] = 1 # attack
            # action[11] = 1 # attack
            self.goal_camera_angle = -15
            self.escape_water_step += 1
            self.executing_multistep_subtask = True
            
        else:
            self.goal_camera_angle = self.default_camera_angle
            self.escape_water_step = 0
            self.executing_multistep_subtask = False

        return action

    def step_go_to_picture_location(self):
        action = th.zeros(12)
        if self.go_to_picture_location_step < self.total_go_to_picture_location_steps:
            action[5] = 1 # move forward
            action[6] = 1 # jump
            action[2] = -95/self.total_go_to_picture_location_steps
            self.go_to_picture_location_step += 1
            self.executing_multistep_subtask = True
            
        else:
            self.allow_escape_water = False # do not get scared of waterfall
            if self.delay_to_take_picture_step > self.delay_to_take_picture:
                # turn around to take picture
                # self.goal_camera_angle = self.default_camera_angle
                self.executing_multistep_subtask = False
                self.good_waterfall_view = True            
            action[3] = 180/self.delay_to_take_picture
            self.delay_to_take_picture_step += 1

        return action


    def step_climb_up_subtask(self, state_classifier):
        # look down, place multiple blocks, jump forward, repeat
        action = th.zeros(12)

        if self.pillars_built < self.pillars_to_build:
            # first, adjust head angle
            if not self.adjusted_head_angle:
                action = th.zeros(12)
                action[2] = 90-self.odometry.camera_angle
                self.adjusted_head_angle = True

                return action

            # pauses every few steps to slow down pillar construction
            if self.climb_up_step % self.climb_up_step_frequency == 0:
                self.climb_up_step += 1
                self.executing_multistep_subtask = True
                return action

            if self.climb_up_step < self.total_climb_up_steps:
                # if np.random.rand() < 0.1:
                # action[5] = 1 # move forward
                action[6] = 1  # jump
                action[4] = 3  # equip block
                action[11] = 1 # drop block
                self.goal_camera_angle = 90
                self.climb_up_step += 1
                self.executing_multistep_subtask = True
                
            else:
                # # jump forward
                # action[5] = 1 # move forward
                # action[6] = 1  # jump

                # look back
                action[1] = 1 # move backward
                action[3] = 180

                # reset
                self.goal_camera_angle = self.default_camera_angle
                self.climb_up_step = 0
                self.adjusted_head_angle = False
                self.pillars_built += 1

        else:
            self.executing_multistep_subtask = False

            # TODO: check if reached top
            self.reached_top = True
            # if state_classifier['at_the_top_of_a_waterfall'] > 0.5:
            #     self.reached_top = True
            # else:
            #     # retry
            #     self.climb_up_step = 0
            #     self.pillars_built = 0
            #     self.adjusted_head_angle = False

        return action

    def step_consensus(self, action, state_classifier):
        # keep track of states seen during consensus
        for key, value in state_classifier.items():
            self.consensus_states[key].append(value)
        self.consensus_states['heading'].append(self.odometry.heading.item())

        # explore around
        if self.consensus_steps < self.num_consensus_steps:
            # zero out all previous actions
            action = th.zeros(12)    

            # investigate surroundings
            if self.consensus_steps > 0.25*self.num_consensus_steps and \
                self.consensus_steps < 0.75*self.num_consensus_steps:
                action[3] = np.random.randint(low=2, high=8) # turn camera right
            else:
                action[3] = -np.random.randint(low=2, high=8) # turn camera left

            # step counter
            self.consensus_steps += 1

        else:
            self.consensus_steps = 0
            self.triggerred_consensus = False

            # # DEBUG: write to disk to better explore consensus solution
            # consensus_states_df = pd.DataFrame.from_dict(self.consensus_states)
            # consensus_states_df.to_csv("data/sample_consensus_data.csv")
            

        return action

    def _initialize_mapping(self):
        self.mapping = {
            0: "find_goal",
            1: "go_to_goal",
            2: "end_episode",
            3: "climb_up",
            4: "climb_down",
            5: "place_waterfall",
            6: "look_around",
            7: "build_pen",
            8: "go_to_location",
            9: "lure_animals",
            10: "leave_pen",
            11: "infer_biome",
            12: "build_house",
            13: "tour_inside_house",
            14: "leave_house",
        }
        self.n_subtasks = len(self.mapping.keys())

    def setup_escape_water_subtask(self):
        self.escape_water_step = 0
        self.total_escape_water_steps = 10

    def setup_waterfall_task(self):
        # setup available inventory
        build_waterfall_items = [
            'air','bucket','carrot','cobblestone','fence','fence_gate','none','other',
            'snowball','stone_pickaxe','stone_shovel','water_bucket','wheat','wheat_seeds'
        ]
        self.item_map = {build_waterfall_items[i]: i for i in range(len(build_waterfall_items))}

        self.moving_towards_mountain = False
        self.found_mountain = False
        self.reached_top = False
        self.adjusted_head_angle = False
        self.built_waterfall = False
        self.go_to_picture_location_step = 0
        self.total_go_to_picture_location_steps = 70
        self.delay_to_take_picture = 40
        self.delay_to_take_picture_step = 0
        self.min_time_look_for_mountain = 1*20 # steps
        self.timeout_to_build_waterfall = 90*20

        self.climb_up_step_frequency = 5
        self.climb_up_step = 0
        self.total_climb_up_steps = 5*self.climb_up_step_frequency/2

        self.pillars_to_build = 1
        self.pillars_built = 0

    def setup_house_subtask(self):
        # setup available inventory
        build_house_items = [
            'acacia_door','acacia_fence','cactus','cobblestone','dirt','fence','flower_pot','glass','ladder',
            'log#0','log#1','log2#0','none','other','planks#0','planks#1','planks#4','red_flower','sand',
            'sandstone#0','sandstone#2','sandstone_stairs','snowball','spruce_door','spruce_fence',
            'stone_axe','stone_pickaxe','stone_stairs','torch','wooden_door','wooden_pressure_plate'
        ]
        self.item_map = {build_house_items[i]: i for i in range(len(build_house_items))}

        # clone actions
        self.build_house_actions = self.clone_human_actions(            
            dataset_addr="data/MineRLBasaltBuildVillageHouse-v0/v3_specific_quince_werewolf-1_30220-36861/rendered.npz",
            start_step=70,
            end_step=5519,
        )
        self.build_house_step = 0
        self.total_build_house_steps = self.build_house_actions.shape[1]
        self.house_built = False
        self.timeout_to_build_house = 20*30

    def step_house_subtask(self):
        # first, adjust head angle
        if not self.adjusted_head_angle:
            action = th.zeros(12)
            action[2] = 35-self.odometry.camera_angle
            action[3] = -10-self.odometry.heading
            self.adjusted_head_angle = True

            return action

        # query cloned action
        # skip frames when sending 'use' action to counter delay in tool selection:
        # https://minerl.readthedocs.io/en/latest/environments/handlers.html#tool-control-equip-and-use
        action = th.from_numpy(self.build_house_actions[:, self.build_house_step])    
        self.build_house_step += 1
        self.executing_multistep_subtask = True

        # flags end of construction
        if self.build_house_step >= self.total_build_house_steps:
            self.house_built = True
            self.executing_multistep_subtask = False

        return action
        
    def setup_cave_task(self):
        # setup available inventory
        build_cave_items = [
            'air','bucket','carrot','cobblestone','fence','fence_gate','none','other','snowball',
            'stone_pickaxe','stone_shovel','water_bucket','wheat','wheat_seeds'
        ]
        self.item_map = {build_cave_items[i]: i for i in range(len(build_cave_items))}

        self.timeout_to_find_cave = 170*20

    def setup_pen_subtask(self):
        # setup available inventory
        build_pen_items = [
            'air','bucket','carrot','cobblestone','fence','fence_gate','none','other','snowball',
            'stone_pickaxe','stone_shovel','water_bucket','wheat','wheat_seeds',
        ]
        self.item_map = {build_pen_items[i]: i for i in range(len(build_pen_items))}

        # clone actions
        self.build_pen_actions = self.clone_human_actions(            
            dataset_addr="data/MineRLBasaltCreateVillageAnimalPen-v0/v3_another_spinach_undead-5_28317-30684/rendered.npz",
            start_step=330,
            end_step=780,
        )
        self.build_pen_step = 0
        self.total_build_pen_steps = self.build_pen_actions.shape[1]
        self.pen_built = False
        self.timeout_to_build_pen = 20*30
        self.timeout_to_lure_animals_sec = 60
        self.timeout_to_find_animals_sec = 180
        self.animals_lured = False
        self.animal_locations = None
        self.confirmed_animal_location = False

    def step_pen_subtask(self):
        # first, adjust head angle
        if not self.adjusted_head_angle:
            action = th.zeros(12)
            action[2] = 35-self.odometry.camera_angle
            action[3] = -self.odometry.heading
            self.adjusted_head_angle = True

            # also store pen location
            self.odometry.pen_location = [self.odometry.x.item(), self.odometry.y.item()]
            self.odometry.pen_built_time_sec = self.odometry.t

            return action

        # query cloned action
        # skip frames when sending 'use' action to counter delay in tool selection:
        # https://minerl.readthedocs.io/en/latest/environments/handlers.html#tool-control-equip-and-use
        action = th.from_numpy(self.build_pen_actions[:, self.build_pen_step])    
        self.build_pen_step += 1
        self.executing_multistep_subtask = True

        # flags end of construction
        if self.build_pen_step >= self.total_build_pen_steps:
            self.pen_built = True
            self.executing_multistep_subtask = False

        return action


    def step_lure_animals_subtask(self, obs):
        # keep food in hand
        #  2: carrot
        # 12: wheat
        # 13: wheat seeds
        #
        # NOTE: still working on "food selection" model, equipping always wheat
        # because cows and sheeps are more likely to spawn
        action = th.zeros(12)
        action[4] = 12

        if not self.confirmed_animal_location:
            # navigate to animal location (last seen)
            if len(self.odometry.has_animals_spot_coords['x']) == 0:
                # timeout to stop looking for animals
                if self.odometry.t > self.timeout_to_find_animals_sec:
                    print('No luck finding animals.')
                    action = self.subtask_end_episode()
                    return action

                print('No animals so far.')
                # keep roaming and looking for animals
                action = self.subtask_find_goal(obs)
            else:
                num_has_animals_spot = len(self.odometry.has_animals_spot_coords['x'])
                goal_x = self.odometry.has_animals_spot_coords['x'][0]
                goal_y = self.odometry.has_animals_spot_coords['y'][0]
                distance, angular_diff = self.compute_stats_to_goal(goal_x, goal_y)

                print(f'Last seen animal at {goal_x:.2f}, {goal_y:.2f} ({num_has_animals_spot} total)')
                print(f'  distance: {distance:.2f} m')
                print(f'  angle difference: {angular_diff:.2f} deg')

                # turn camera to goal and move forward
                action = self.subtask_go_to_location(action, angular_diff)

                if distance < 1.0: # meters
                    self.confirmed_animal_location = True

            # keep executing task
            self.executing_multistep_subtask = True
            
        else:
            # confirmed animal location, go back to pen
            # TODO: need to identify animal first to lure them
            # right now, just ends the task after going back to the pen location
            goal_x = self.odometry.pen_location[0]
            goal_y = self.odometry.pen_location[1]
            distance, angular_diff = self.compute_stats_to_goal(goal_x, goal_y)
            time_after_pen_built = self.odometry.t-self.odometry.pen_built_time_sec

            print(f'Pen built at {goal_x:.2f}, {goal_y:.2f}')
            print(f'  distance: {distance:.2f} m')
            print(f'  angle difference: {angular_diff:.2f} deg')
            print(f'  time_after_pen_built: {time_after_pen_built:.2f} sec')

            # turn camera to goal and move forward
            action = self.subtask_go_to_location(action, angular_diff)

            # keep executing task
            self.executing_multistep_subtask = True

            # end episode when back to pen or if took too long
            if distance < 1.0 or time_after_pen_built > self.timeout_to_lure_animals_sec:
                self.animals_lured = True
                self.executing_multistep_subtask = False

        return action

    def compute_stats_to_goal(self, goal_x, goal_y):
        dist_x = goal_x-self.odometry.x.item()
        dist_y = goal_y-self.odometry.y.item()
        distance = np.sqrt(dist_x**2+dist_y**2)

        angular_diff = self.odometry.heading.item()-np.rad2deg(np.arctan2(
            self.odometry.x.item()-goal_x,
            goal_y-self.odometry.y.item()))
            
        if self.odometry.heading.item() < 0:
            angular_diff += 90
        else:
            angular_diff -= 90

        if angular_diff >= 360.0 or angular_diff <= 360.0:
            angular_diff = angular_diff % 360.0

        return distance, angular_diff

    def subtask_go_to_location(self, action, angular_diff):
        # turn camera towards goal heading
        cam_limit = 3.0
        action[3] = np.clip(angular_diff-self.odometry.heading.item(), -cam_limit, cam_limit)

        # move forward once heading is on track
        if action[3] < np.abs(cam_limit):
            action[5] = 1

            # randomly jump
            if np.random.rand() < 0.25:
                action[6] = 1 # jump

        return action


    def clone_human_actions(self, dataset_addr, start_step, end_step):
        # load human data
        data = dict(np.load(dataset_addr))

        # substitute item names by numbers
        last_equipped_item = 'none'
        action_equip_num = []
        for i in range(data['action$equip'].shape[0]):
            # replace 'none' equip actions by last equipped item
            # fixes mismatch between human collected dataset and minerl env as explained here:
            # https://minerl.readthedocs.io/en/latest/environments/handlers.html#tool-control-equip-and-use
            if data['action$equip'][i] == 'none':
                data['action$equip'][i] = last_equipped_item
            else:
                last_equipped_item = data['action$equip'][i]

            action_equip_num.append(self.item_map[data['action$equip'][i]])

        # stack all actions
        action_data = np.vstack((
            data['action$attack'].astype(int),
            data['action$back'].astype(int),
            data['action$camera'][:,0].astype(float),
            data['action$camera'][:,1].astype(float),
            action_equip_num,
            data['action$forward'].astype(int),
            data['action$jump'].astype(int),
            data['action$left'].astype(int),
            data['action$right'].astype(int),
            data['action$sneak'].astype(int),
            data['action$sprint'].astype(int),
            data['action$use'].astype(int),
        ))   

        # replay only a fraction of the demonstrations data
        action_data = action_data[:, start_step:end_step]

        # store original list of equipped items for debugging purposes
        self.original_equip_items = data['action$equip'][start_step:end_step]

        return action_data


class KAIROS_Vision():
    """
    Extracts vision features from agent's POV.
    """
    def __init__(self, state_classifier_model, device):
        self.state_classifier = state_classifier_model
        self.device = device
        self.num_classes = 13

        # internal count of environment steps
        self.env_step_t = 0

        # map state classifier index to names
        self.map = {
            0: 'no_labels',
            1: 'has_cave',
            2: 'inside_cave',
            3: 'danger_ahead',
            4: 'has_mountain',
            5: 'facing_wall',
            6: 'at_the_top_of_a_waterfall',
            7: 'good_view_of_waterfall',
            8: 'good_view_of_pen',
            9: 'good_view_of_house',
            10: 'has_animals',
            11: 'has_open_space',
            12: 'animals_inside pen',
        }

    def extract_features(self, obs):
        # classify input state
        # convert to torch and generate predictions
        obs = th.Tensor(obs).unsqueeze(0).to(self.device)
        state_classifier_probs = self.state_classifier(obs)        

        # build features dict
        state_classifier = {}
        for i in range(self.num_classes):
            state_classifier[self.map[i]] = state_classifier_probs[0].data[i].item()

        # keep track of environment steps
        self.env_step_t += 1

        return state_classifier


class KAIROS_Odometry():
    """
    Estimates position for the agent and point of interest based on images received
    and actions taken.
    """
    def __init__(self, exp_id, state_classifier_map):
        # initialize states
        self.t, self.x, self.y = 0., 0., 0.
        self.heading, self.vel, self.camera_angle = 0., 0., 0.

        # minecraft motion constants
        self.fps = 20
        self.dt = 1/self.fps
        self.swim_vel = 2.2 # m/s (surface)
        self.walk_vel = 4.317 # m/s
        self.sprint_vel_bonus = 5.612-self.walk_vel # m/s
        self.sprint_jump_vel_bonus = 7.127-self.walk_vel # m/s
        self.animal_range = 6. # meters
        self.detection_range = 10. # meters (crude estimate)
        self.map_resolution = 10 # scales pixels for odometry frame (higher, more precise)
        self.pen_location = [0,0]
        self.pen_built_time_sec = 0

        # save logs to disk
        self.exp_id = exp_id
        self.state_classifier_map = state_classifier_map
        self.odometry_log = {
            'env_step': [0],
            't': [self.t],
            'x': [self.x],
            'y': [self.y],
            'heading': [self.heading],
            'vel': [self.vel],
            'camera_angle': [self.camera_angle],
        }
        # add all classified states to log
        for i in range(len(self.state_classifier_map.keys())):
            self.odometry_log[self.state_classifier_map[i]] = [0.]
        self.actions_log = [np.zeros(12)]

        # colors to display features
        self.agent_color=(0,0,255)
        self.danger_ahead_color=(255,0,0)
        self.has_animals_color=(212,170,255)
        self.has_open_space_color=(86,255,86)
        self.has_cave_color=(34,48,116)
        self.at_the_top_of_a_waterfall_color=(228,245,93)

        # keep track of good build spots and if swimming
        self.good_build_spot = False
        self.agent_swimming = False
        self.has_animals_spot = False
        self.top_of_waterfall_spot = False
        self.has_animals_spot_coords = {
            't': [], 'x': [], 'y': [], 'verified': []
        }

    def update(self, action, env_step, state_classifier):
        # compute current velocity
        # [0] "attack"
        # [1] "back"
        # [2] "camera_up_down" (float)
        # [3] "camera_right_left" (float)
        # [4] "equip"
        # [5] "forward"
        # [6] "jump"
        # [7] "left"
        # [8] "right"
        # [9] "sneak"
        # [10] "sprint"
        # [11] "use"

        if action[6]: # jumping
            self.vel = (action[5]-action[1])*(self.walk_vel+self.sprint_jump_vel_bonus*action[10])
        else:
            self.vel = (action[5]-action[1])*(self.walk_vel+self.sprint_vel_bonus*action[10])

        # update heading based on camera movement
        self.heading += action[3]
        if self.heading >= 360.0 or self.heading <= -360.0:
            self.heading = self.heading % 360.0
        self.camera_angle += action[2]

        # update position based on estimated velocity
        self.t += self.dt
        self.x += self.vel*np.cos(np.deg2rad(self.heading))*self.dt
        self.y += self.vel*np.sin(np.deg2rad(self.heading))*self.dt

        # add states identified by the state classifier to the odometry logs
        for i in range(len(self.state_classifier_map.keys())):
            self.odometry_log[self.state_classifier_map[i]].append(state_classifier[self.state_classifier_map[i]])

        # update logs
        self.odometry_log['env_step'].append(env_step)
        self.odometry_log['t'].append(self.t)
        self.odometry_log['x'].append(self.x.item())
        self.odometry_log['y'].append(self.y.item())
        self.odometry_log['heading'].append(self.heading)
        self.odometry_log['camera_angle'].append(self.camera_angle)
        self.odometry_log['vel'].append(self.vel.item())
        self.actions_log.append(action.numpy().flatten())

    # Convert coordinates to pixel value to display in odometry map
    def coord_to_pixel(self, x, y):
        x_pos = int(self.map_resolution*(x+self.min_x))
        y_pos = int(self.map_resolution*(y+self.min_y))
        
        return x_pos, y_pos

    def tag_relevant_state(self, odom_frame, state_name, color, radius, confidence=0.65, fill=1):
        states = np.array(self.odometry_log[state_name])
        relevant_states = np.argwhere(states > confidence)
        for i in range(relevant_states.shape[0]):
            idx = relevant_states[i][0]
            x_pos, y_pos = self.coord_to_pixel(
                x=self.odometry_log['x'][idx],
                y=self.odometry_log['y'][idx])
            odom_frame = cv2.circle(odom_frame, (y_pos, x_pos), radius, color, fill)

    def generate_frame(self):
        # Keep track of image bounds
        all_xs = np.array(self.odometry_log['x'])
        all_ys = np.array(self.odometry_log['y'])
        self.min_x = np.abs(all_xs.min())
        self.min_y = np.abs(all_ys.min())

        # Convert odometry to pixels
        x = (self.map_resolution*(all_xs+self.min_x)).astype(int)
        y = (self.map_resolution*(all_ys+self.min_y)).astype(int)
        
        # Setup odometry image with maximum x or y dimension
        max_coord = max(x.max(), y.max())
        odom_frame = np.zeros((max_coord+1, max_coord+1, 3), np.uint8)

        # Substitute coordinates as white pixels
        odom_frame[x, y] = 255

        # Add circle to current robot position
        x_pos = x[-1]
        y_pos = y[-1]
        odom_frame = cv2.circle(odom_frame, (y_pos, x_pos), 5, self.agent_color, -1)

        # Add markers to relevant classified states
        self.tag_relevant_state(odom_frame, state_name='has_open_space',
            color=self.has_open_space_color, radius=35)
        self.tag_relevant_state(odom_frame, state_name='danger_ahead',
            color=self.danger_ahead_color, radius=15, fill=3)
        self.tag_relevant_state(odom_frame, state_name='has_animals',
            color=self.has_animals_color, radius=5, fill=-1, confidence=0.75)
        self.tag_relevant_state(odom_frame, state_name='has_cave',
            color=self.has_cave_color, radius=15, fill=-1, confidence=0.75)
        self.tag_relevant_state(odom_frame, state_name='at_the_top_of_a_waterfall',
            color=self.at_the_top_of_a_waterfall_color, radius=15, fill=-1, confidence=0.75)

        # Make sure image always has the same size
        odom_frame = cv2.resize(odom_frame, (512, 512), interpolation=cv2.INTER_LINEAR)

        # Add text with odometry info
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        font_scale = 0.5
        text_color = (255, 255, 255)
        # left column text
        odom_frame = cv2.putText(odom_frame, f'x: {self.x:.2f} m', (10, 20), font, 
                   font_scale, text_color, thickness, cv2.LINE_AA)
        odom_frame = cv2.putText(odom_frame, f'y: {self.y:.2f} m', (10, 40), font, 
                   font_scale, text_color, thickness, cv2.LINE_AA)
        odom_frame = cv2.putText(odom_frame, f'heading: {self.heading:.2f} deg', (10, 60), font, 
                   font_scale, text_color, thickness, cv2.LINE_AA)
        odom_frame = cv2.putText(odom_frame, f'camera_angle: {self.camera_angle:.2f} deg', (10, 80), font, 
                   font_scale, text_color, thickness, cv2.LINE_AA)
        if self.good_build_spot:
            odom_frame = cv2.putText(odom_frame, 'GOOD BUILD SPOT', (10, 120), font, 
                   font_scale, (0,255,0), thickness, cv2.LINE_AA)
        if self.agent_swimming:
            odom_frame = cv2.putText(odom_frame, 'AGENT SWIMMING', (10, 140), font, 
                   font_scale, (0,0,255), thickness, cv2.LINE_AA)
        if self.has_animals_spot:
            odom_frame = cv2.putText(odom_frame, 'SPOT HAS ANIMALS', (10, 160), font, 
                   font_scale, self.has_animals_color, thickness, cv2.LINE_AA)
            self.has_animals_spot_coords['t'].append(self.t)
            self.has_animals_spot_coords['x'].append(self.x.item())
            self.has_animals_spot_coords['y'].append(self.y.item())
            self.has_animals_spot_coords['verified'].append(False)
        if self.top_of_waterfall_spot:
            odom_frame = cv2.putText(odom_frame, 'GOOD SPOT FOR WATERFALL', (10, 180), font, 
                   font_scale, self.at_the_top_of_a_waterfall_color, thickness, cv2.LINE_AA)
        # right column text
        odom_frame = cv2.putText(odom_frame, f'time: {self.t:.2f} sec', (352, 20), font, 
                   font_scale, text_color, thickness, cv2.LINE_AA)
        odom_frame = cv2.putText(odom_frame, 'legend:', (352, 60), font, 
                   font_scale, text_color, thickness, cv2.LINE_AA)
        odom_frame = cv2.putText(odom_frame, 'agent', (372, 80), font, 
                   font_scale, self.agent_color, thickness, cv2.LINE_AA)
        odom_frame = cv2.putText(odom_frame, 'danger_ahead', (372, 100), font, 
                   font_scale, self.danger_ahead_color, thickness, cv2.LINE_AA)
        odom_frame = cv2.putText(odom_frame, 'has_animals', (372, 120), font, 
                   font_scale, self.has_animals_color, thickness, cv2.LINE_AA)
        odom_frame = cv2.putText(odom_frame, 'has_open_space', (372, 140), font, 
                   font_scale, self.has_open_space_color, thickness, cv2.LINE_AA)
        odom_frame = cv2.putText(odom_frame, 'has_cave', (372, 160), font, 
                   font_scale, self.has_cave_color, thickness, cv2.LINE_AA)
        odom_frame = cv2.putText(odom_frame, 'top_of_waterfall', (372, 180), font, 
                   font_scale, self.at_the_top_of_a_waterfall_color, thickness, cv2.LINE_AA)

        return odom_frame

    def close(self):
        # save logs to disk
        os.makedirs('train/odometry', exist_ok=True)
        odometry_log_df = pd.DataFrame.from_dict(self.odometry_log)
        action_log_df = pd.DataFrame(self.actions_log, columns=[
            'attack', 'back', 'equip', 'forward', 'jump', 'left', 'right',
            'sneak', 'sprint', 'use', 'camera_up_down', 'camera_right_left'])
        log_df = pd.concat([odometry_log_df, action_log_df], axis=1)
        log_df.to_csv(f'train/odometry/odometry_log_{self.exp_id}.csv', index = False, header=True)