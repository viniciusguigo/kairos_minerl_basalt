import numpy as np
import gym

#*******************************************************************
#   FIND CAVE TASK
#*******************************************************************
# custom action wrapper for complete GAIL agent for MineRL
class ActionShaping_FindCave(gym.ActionWrapper):
    def __init__(self, env, camera_angle=10, always_attack=False):
        super().__init__(env)

        self.camera_angle = camera_angle
        self.always_attack = always_attack
        self._actions = [
            [('attack', 1)], #0
            [('forward', 1)], #1
            [('forward', 1), ('jump', 1)], #2
            [('camera', [-self.camera_angle, 0])], #3
            [('camera', [self.camera_angle, 0])],  #4
            [('camera', [0, self.camera_angle])],  #5
            [('camera', [0, -self.camera_angle])], #6
            [('back', 1)], #7
            [('left', 1)], #8
            [('right', 1)], #9
            [('jump', 1)], #10
            #[('equip',11), ('use', 1)],
            [('forward', 1), ('attack', 1)], #11
        ]

        self.actions = []
        for actions in self._actions:
            act = self.env.action_space.noop()
            for a, v in actions:
                act[a] = v
            if self.always_attack:
                act['attack'] = 1
            self.actions.append(act)
        # add no-op action
        act = self.env.action_space.noop()
        self.actions.append(act)
        
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action):
        return self.actions[action]


def processed_actions_to_wrapper_actions_FindCave(dataset_actions, camera_margin=5):
    """
    Turn a batch of actions from dataset (`batch_iter`) to a numpy
    array that corresponds to batch of actions of ActionShaping wrapper (_actions).

    Camera margin sets the threshold what is considered "moving camera".

    Note: Hardcoded to work for actions in ActionShaping._actions, with "intuitive"
        ordering of actions.
        If you change ActionShaping._actions, remember to change this!

    Array elements are integers corresponding to actions, or "-1"
    for actions that did not have any corresponding discrete match.
    """
    # There are dummy dimensions of shape one
    camera_actions = dataset_actions[:,10:].astype(np.float32)
    attack_actions = dataset_actions[:,0].astype(np.float32)
    forward_actions = dataset_actions[:,3].astype(np.float32)
    jump_actions = dataset_actions[:,4].astype(np.float32)
    back_actions = dataset_actions[:,1].astype(np.float32)
    left_actions = dataset_actions[:,5].astype(np.float32)
    right_actions =  dataset_actions[:,6].astype(np.float32)
    equip_actions = dataset_actions[:,2]
    use_actions = dataset_actions[:,9].astype(np.float32)
    sneak_actions = dataset_actions[:,7].astype(np.float32)
    sprint_actions = dataset_actions[:,8].astype(np.float32)
    batch_size = len(camera_actions)
    
    actions = np.zeros((batch_size,), dtype=int)

    for i in range(len(camera_actions)):
        # Moving camera is most important (horizontal first)
        if camera_actions[i][0] < -camera_margin:
            actions[i] = 3
        elif camera_actions[i][0] > camera_margin:
            actions[i] = 4
        elif camera_actions[i][1] > camera_margin:
            actions[i] = 5
        elif camera_actions[i][1] < -camera_margin:
            actions[i] = 6
        elif forward_actions[i] == 1:
            if jump_actions[i] == 1:
                actions[i] = 2
            elif attack_actions[i] == 1:
                actions[i] = 11
            else:
                actions[i] = 1
        elif attack_actions[i] == 1:
            actions[i] = 0
        elif left_actions[i] == 1:
            actions[i] = 8
        elif right_actions[i] ==1:
            actions[i] = 9
        elif back_actions[i] == 1:
            actions[i] = 7
        elif jump_actions[i] == 1:
            actions[i] = 10
        else:
            # No reasonable mapping (would be no-op)
            actions[i] = 12
    return actions


#*******************************************************************
#   WATERFALL TASK
#*******************************************************************
# custom action wrapper for complete GAIL agent for MineRL
class ActionShaping_Waterfall(gym.ActionWrapper):
    def __init__(self, env, camera_angle=10, always_attack=False):
        super().__init__(env)

        self.camera_angle = camera_angle
        self.always_attack = always_attack
        self._actions = [
            [('attack', 1)], #0
            [('forward', 1)], #1
            [('forward', 1), ('jump', 1)], #2
            [('camera', [-self.camera_angle, 0])], #3
            [('camera', [self.camera_angle, 0])],  #4
            [('camera', [0, self.camera_angle])],  #5
            [('camera', [0, -self.camera_angle])], #6
            [('back', 1)], #7
            [('left', 1)], #8
            [('right', 1)], #9
            [('jump', 1)], #10
            [('forward', 1), ('attack', 1)], #11
            [('equip','water_bucket'), ('use', 1)], #12 #water bucket
            [('equip','stone_pickaxe'), ('use', 1)], #13 #stone pickaxe
            [('equip','stone_shovel'), ('use', 1)], #14 #stone shovel
            [('equip','cobblestone'), ('use', 1)], #15 #cobblestone
            #[('equip',1), ('use', 1)], #16 #bucket
        ]

        self.actions = []
        for actions in self._actions:
            act = self.env.action_space.noop()
            for a, v in actions:
                act[a] = v
            if self.always_attack:
                act['attack'] = 1
            self.actions.append(act)
        # add no-op action
        act = self.env.action_space.noop()
        self.actions.append(act)
        
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action):
        return self.actions[action]


def processed_actions_to_wrapper_actions_Waterfall(dataset_actions, camera_margin=5):
    """
    Turn a batch of actions from dataset (`batch_iter`) to a numpy
    array that corresponds to batch of actions of ActionShaping wrapper (_actions).

    Camera margin sets the threshold what is considered "moving camera".

    Note: Hardcoded to work for actions in ActionShaping._actions, with "intuitive"
        ordering of actions.
        If you change ActionShaping._actions, remember to change this!

    Array elements are integers corresponding to actions, or "-1"
    for actions that did not have any corresponding discrete match.
    """
    # There are dummy dimensions of shape one
    camera_actions = dataset_actions[:,10:].astype(np.float32)
    attack_actions = dataset_actions[:,0].astype(np.float32)
    forward_actions = dataset_actions[:,3].astype(np.float32)
    jump_actions = dataset_actions[:,4].astype(np.float32)
    back_actions = dataset_actions[:,1].astype(np.float32)
    left_actions = dataset_actions[:,5].astype(np.float32)
    right_actions =  dataset_actions[:,6].astype(np.float32)
    equip_actions = dataset_actions[:,2]
    use_actions = dataset_actions[:,9].astype(np.float32)
    sneak_actions = dataset_actions[:,7].astype(np.float32)
    sprint_actions = dataset_actions[:,8].astype(np.float32)
    batch_size = len(camera_actions)
    
    actions = np.zeros((batch_size,), dtype=int)


    #Enum(air,bucket,carrot,cobblestone,fence,fence_gate,none,other,snowball,stone_pickaxe,stone_shovel,water_bucket,wheat,wheat_seeds),

    equip_actions_dict = dict()
    equip_actions_dict['water_bucket'] = 12
    equip_actions_dict['stone_pickaxe'] = 13
    equip_actions_dict['stone_shovel'] = 14
    equip_actions_dict['cobblestone'] = 15
    #equip_actions_dict['bucket'] = 16
    # step through all actions 
    currently_equipped_item = 'stone_pickaxe'
    for i in range(len(camera_actions)):
        
        # keep track of what is currently equipped
        if equip_actions[i] != 'none' and equip_actions[i] in equip_actions_dict:
            currently_equipped_item = equip_actions[i]
        
        # equip and use actions are the most important
        if use_actions[i] == 1: 
            actions[i] = equip_actions_dict[currently_equipped_item]
        # Moving camera is second most important (horizontal first)
        elif camera_actions[i][0] < -camera_margin:
            actions[i] = 3
        elif camera_actions[i][0] > camera_margin:
            actions[i] = 4
        elif camera_actions[i][1] > camera_margin:
            actions[i] = 5
        elif camera_actions[i][1] < -camera_margin:
            actions[i] = 6
        elif forward_actions[i] == 1:
            if jump_actions[i] == 1:
                actions[i] = 2
            elif attack_actions[i] == 1:
                actions[i] = 11
            else:
                actions[i] = 1
        elif attack_actions[i] == 1:
            actions[i] = 0
        elif left_actions[i] == 1:
            actions[i] = 8
        elif right_actions[i] ==1:
            actions[i] = 9
        elif back_actions[i] == 1:
            actions[i] = 7
        elif jump_actions[i] == 1:
            actions[i] = 10
        else:
            # No reasonable mapping (would be no-op)
            actions[i] = 16
    return actions


#*******************************************************************
#   ANIMAL PEN TASK
#*******************************************************************
# custom action wrapper for complete GAIL agent for MineRL
class ActionShaping_Animalpen(gym.ActionWrapper):
    def __init__(self, env, camera_angle=10, always_attack=False):
        super().__init__(env)
        self.equip_mapping = {'air':0,'bucket':1,'carrot':2,'cobblestone':3,'fence':4,'fence_gate':5,
                              'none':6,'other':7,'snowball':8,'stone_pickaxe':9,'stone_shovel':10,'water_bucket':11,
                              'wheat':12,'wheat_seeds':13}
        self.camera_angle = camera_angle
        self.always_attack = always_attack
        self._actions = [
            [('attack', 1)], #0
            [('forward', 1)], #1
            [('forward', 1), ('jump', 1)], #2
            [('camera', [-self.camera_angle, 0])], #3
            [('camera', [self.camera_angle, 0])],  #4
            [('camera', [0, self.camera_angle])],  #5
            [('camera', [0, -self.camera_angle])], #6
            [('back', 1)], #7
            [('left', 1)], #8
            [('right', 1)], #9
            [('jump', 1)], #10
            [('forward', 1), ('attack', 1)], #11
            [('equip','carrot')], #12 #carrot
            [('equip','fence'), ('use', 1)], #13 #fence
            [('equip','fence_gate'), ('use', 1)], #14 #fence_gate
            [('equip','wheat')], #15 #wheat
            [('equip','wheat_seeds')], #16 #wheat_seeds
        ]

        self.actions = []
        for actions in self._actions:
            act = self.env.action_space.noop()
            for a, v in actions:
                act[a] = v
            if self.always_attack:
                act['attack'] = 1
            self.actions.append(act)
        # add no-op action
        act = self.env.action_space.noop()
        self.actions.append(act)
        
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action):
        return self.actions[action]


def processed_actions_to_wrapper_actions_Animalpen(dataset_actions, camera_margin=5):
    """
    Turn a batch of actions from dataset (`batch_iter`) to a numpy
    array that corresponds to batch of actions of ActionShaping wrapper (_actions).

    Camera margin sets the threshold what is considered "moving camera".

    Note: Hardcoded to work for actions in ActionShaping._actions, with "intuitive"
        ordering of actions.
        If you change ActionShaping._actions, remember to change this!

    Array elements are integers corresponding to actions, or "-1"
    for actions that did not have any corresponding discrete match.
    """
    # There are dummy dimensions of shape one
    camera_actions = dataset_actions[:,10:].astype(np.float32)
    attack_actions = dataset_actions[:,0].astype(np.float32)
    forward_actions = dataset_actions[:,3].astype(np.float32)
    jump_actions = dataset_actions[:,4].astype(np.float32)
    back_actions = dataset_actions[:,1].astype(np.float32)
    left_actions = dataset_actions[:,5].astype(np.float32)
    right_actions =  dataset_actions[:,6].astype(np.float32)
    equip_actions = dataset_actions[:,2]
    use_actions = dataset_actions[:,9].astype(np.float32)
    sneak_actions = dataset_actions[:,7].astype(np.float32)
    sprint_actions = dataset_actions[:,8].astype(np.float32)
    batch_size = len(camera_actions)
    
    actions = np.zeros((batch_size,), dtype=int)


    #Enum(air,bucket,carrot,cobblestone,fence,fence_gate,none,other,snowball,stone_pickaxe,stone_shovel,water_bucket,wheat,wheat_seeds)

    equip_actions_dict = dict()
    equip_actions_dict['carrot'] = 12
    equip_actions_dict['fence'] = 13
    equip_actions_dict['fence_gate'] = 14
    equip_actions_dict['wheat'] = 15
    equip_actions_dict['wheat_seeds'] = 16
    # step through all actions 
    currently_equipped_item = 'stone_pickaxe'
    for i in range(len(camera_actions)):
        
        # keep track of what is currently equipped
        if equip_actions[i] != 'none'  and equip_actions[i] in equip_actions_dict:
            currently_equipped_item = equip_actions[i]
        
        # equip and use actions are the most important
        if equip_actions[i] == 'carrot':
            actions[i] = equip_actions_dict['carrot']
        elif equip_actions[i] == 'wheat':
            actions[i] = equip_actions_dict['wheat']
        elif equip_actions[i] == 'wheat_seeds':
            actions[i] = equip_actions_dict['wheat_seeds']
        elif use_actions[i] == 1: 
            actions[i] = equip_actions_dict[currently_equipped_item]
        # Moving camera is second most important (horizontal first)
        elif camera_actions[i][0] < -camera_margin:
            actions[i] = 3
        elif camera_actions[i][0] > camera_margin:
            actions[i] = 4
        elif camera_actions[i][1] > camera_margin:
            actions[i] = 5
        elif camera_actions[i][1] < -camera_margin:
            actions[i] = 6
        elif forward_actions[i] == 1:
            if jump_actions[i] == 1:
                actions[i] = 2
            elif attack_actions[i] == 1:
                actions[i] = 11
            else:
                actions[i] = 1
        elif attack_actions[i] == 1:
            actions[i] = 0
        elif left_actions[i] == 1:
            actions[i] = 8
        elif right_actions[i] ==1:
            actions[i] = 9
        elif back_actions[i] == 1:
            actions[i] = 7
        elif jump_actions[i] == 1:
            actions[i] = 10
        else:
            # No reasonable mapping (would be no-op)
            actions[i] = 17
    return actions


#*******************************************************************
#   VILLAGE HOUSE TASK
#*******************************************************************
# custom action wrapper for complete GAIL agent for MineRL
class ActionShaping_Villagehouse(gym.ActionWrapper):
    def __init__(self, env, camera_angle=10, always_attack=False):
        super().__init__(env)
        self.equip_mapping = {'acacia_door':0,'acacia_fence':1,'cactus':2,'cobblestone':3,'dirt':4,'fence':5,'flower_pot':6,
                              'glass':7,'ladder':8,'log#0':9,'log#1':10,'log2#0':12,'none':13,'other':14,'planks#0':15,
                              'planks#1':16,'planks#4':17,'red_flower':18,'sand,sandstone#0':19,'sandstone#2':20,'sandstone_stairs':21,
                              'snowball':22,'spruce_door':23,'spruce_fence':24,'stone_axe':25,'stone_pickaxe':26,'stone_stairs':27,
                              'torch':28,'wooden_door':29,'wooden_pressure_plate':30}
        self.camera_angle = camera_angle
        self.always_attack = always_attack
        self._actions = [
            [('attack', 1)], #0
            [('forward', 1)], #1
            [('forward', 1), ('jump', 1)], #2
            [('camera', [-self.camera_angle, 0])], #3
            [('camera', [self.camera_angle, 0])],  #4
            [('camera', [0, self.camera_angle])],  #5
            [('camera', [0, -self.camera_angle])], #6
            [('back', 1)], #7
            [('left', 1)], #8
            [('right', 1)], #9
            [('jump', 1)], #10
            [('forward', 1), ('attack', 1)], #11
            [('equip','acacia_door'), ('use', 1)], #12 
            [('equip','acacia_fence'), ('use', 1)], #13
            [('equip','cactus'), ('use', 1)], #14
            [('equip','cobblestone'), ('use', 1)], #15
            [('equip','dirt'), ('use', 1)], #16 
            [('equip','fence'), ('use', 1)], #17
            [('equip','flower_pot'), ('use', 1)], #18
            [('equip','glass'), ('use', 1)], #19
            [('equip','ladder'), ('use', 1)], #20
            [('equip','log#0'), ('use', 1)], #21
            [('equip','log#1'), ('use', 1)], #22
            [('equip','log2#0'), ('use', 1)], #23
            [('equip','planks#0'), ('use', 1)], #24
            [('equip','planks#1'), ('use', 1)], #25
            [('equip','planks#4'), ('use', 1)], #26
            [('equip','red_flower'), ('use', 1)], #27 
            [('equip','sand,sandstone#0'), ('use', 1)], #28
            [('equip','sandstone#2'), ('use', 1)], #29 
            [('equip','sandstone_stairs'), ('use', 1)],#30
            [('equip','spruce_door'), ('use', 1)], #31
            [('equip','spruce_fence'), ('use', 1)], #32
            [('equip','stone_axe'), ('use', 1)], #33 
            [('equip','stone_pickaxe'), ('use', 1)], #34 
            [('equip','stone_stairs'), ('use', 1)], #35
            [('equip','torch'), ('use', 1)], #36 
            [('equip','wooden_door'), ('use', 1)], #37 
            [('equip','wooden_pressure_plate'), ('use', 1)], #38
        ]

        self.actions = []
        for actions in self._actions:
            act = self.env.action_space.noop()
            for a, v in actions:
                act[a] = v
            if self.always_attack:
                act['attack'] = 1
            self.actions.append(act)
        # add no-op action
        act = self.env.action_space.noop()
        self.actions.append(act)
        
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action):
        return self.actions[action]


def processed_actions_to_wrapper_actions_Villagehouse(dataset_actions, camera_margin=5):
    """
    Turn a batch of actions from dataset (`batch_iter`) to a numpy
    array that corresponds to batch of actions of ActionShaping wrapper (_actions).

    Camera margin sets the threshold what is considered "moving camera".

    Note: Hardcoded to work for actions in ActionShaping._actions, with "intuitive"
        ordering of actions.
        If you change ActionShaping._actions, remember to change this!

    Array elements are integers corresponding to actions, or "-1"
    for actions that did not have any corresponding discrete match.
    """
    # There are dummy dimensions of shape one
    camera_actions = dataset_actions[:,10:].astype(np.float32)
    attack_actions = dataset_actions[:,0].astype(np.float32)
    forward_actions = dataset_actions[:,3].astype(np.float32)
    jump_actions = dataset_actions[:,4].astype(np.float32)
    back_actions = dataset_actions[:,1].astype(np.float32)
    left_actions = dataset_actions[:,5].astype(np.float32)
    right_actions =  dataset_actions[:,6].astype(np.float32)
    equip_actions = dataset_actions[:,2]
    use_actions = dataset_actions[:,9].astype(np.float32)
    sneak_actions = dataset_actions[:,7].astype(np.float32)
    sprint_actions = dataset_actions[:,8].astype(np.float32)
    batch_size = len(camera_actions)
    
    actions = np.zeros((batch_size,), dtype=int)


    #Enum(acacia_door,acacia_fence,cactus,cobblestone,dirt,fence,flower_pot,glass,ladder,log#0,log#1,log2#0,none,other,planks#0,planks#1,planks#4,red_flower,sand,sandstone#0,sandstone#2,sandstone_stairs,snowball,spruce_door,spruce_fence,stone_axe,stone_pickaxe,stone_stairs,torch,wooden_door,wooden_pressure_plate)

    equip_actions_dict = dict()
    equip_actions_dict['carrot'] = 12
    equip_actions_dict['fence'] = 13
    equip_actions_dict['fence_gate'] = 14
    equip_actions_dict['wheat'] = 15
    equip_actions_dict['wheat_seeds'] = 16    
    
    equip_actions_dict['acacia_door']=12 
    equip_actions_dict['acacia_fence']=13
    equip_actions_dict['cactus']=14
    equip_actions_dict['cobblestone']=15
    equip_actions_dict['dirt']=16 
    equip_actions_dict['fence']=17
    equip_actions_dict['flower_pot']=18
    equip_actions_dict['glass']=19
    equip_actions_dict['ladder']=20
    equip_actions_dict['log#0']=21
    equip_actions_dict['log#1']=22
    equip_actions_dict['log2#0']=23
    equip_actions_dict['planks#0']=24
    equip_actions_dict['planks#1']=25
    equip_actions_dict['planks#4']=26
    equip_actions_dict['red_flower']=27 
    equip_actions_dict['sand,sandstone#0']=28
    equip_actions_dict['sandstone#2']=29 
    equip_actions_dict['sandstone_stairs']=30
    equip_actions_dict['spruce_door']=31
    equip_actions_dict['spruce_fence']=32
    equip_actions_dict['stone_axe']=33 
    equip_actions_dict['stone_pickaxe']=34 
    equip_actions_dict['stone_stairs']=35
    equip_actions_dict['torch']=36 
    equip_actions_dict['wooden_door']=37 
    equip_actions_dict['wooden_pressure_plate']=38
    
    
    # step through all actions 
    currently_equipped_item = 'stone_pickaxe'
    for i in range(len(camera_actions)):
        
        # keep track of what is currently equipped
        if equip_actions[i] != 'none' and equip_actions[i] in equip_actions_dict:
            currently_equipped_item = equip_actions[i]
        
        # equip and use actions are the most important
        if use_actions[i] == 1: 
            actions[i] = equip_actions_dict[currently_equipped_item]
        # Moving camera is second most important (horizontal first)
        elif camera_actions[i][0] < -camera_margin:
            actions[i] = 3
        elif camera_actions[i][0] > camera_margin:
            actions[i] = 4
        elif camera_actions[i][1] > camera_margin:
            actions[i] = 5
        elif camera_actions[i][1] < -camera_margin:
            actions[i] = 6
        elif forward_actions[i] == 1:
            if jump_actions[i] == 1:
                actions[i] = 2
            elif attack_actions[i] == 1:
                actions[i] = 11
            else:
                actions[i] = 1
        elif attack_actions[i] == 1:
            actions[i] = 0
        elif left_actions[i] == 1:
            actions[i] = 8
        elif right_actions[i] ==1:
            actions[i] = 9
        elif back_actions[i] == 1:
            actions[i] = 7
        elif jump_actions[i] == 1:
            actions[i] = 10
        else:
            # No reasonable mapping (would be no-op)
            actions[i] = 39
    return actions


# custom action wrapper for Simple GAIL agent for MineRL
#*******************************************************************
#   NAVIGATION SUBTASK
#*******************************************************************
# custom action wrapper for complete GAIL agent for MineRL
class ActionShaping_Navigation(gym.ActionWrapper):
    def __init__(self, env, camera_angle=10, always_attack=False):
        super().__init__(env)

        self.camera_angle = camera_angle
        self.always_attack = always_attack
        self._actions = [
            [('attack', 1)], #0
            [('forward', 1)], #1
            [('forward', 1), ('jump', 1)], #2
            [('camera', [0, self.camera_angle])],  #3  #horizontal (right)
            [('camera', [0, -self.camera_angle])], #4  #horizontal (left)
            [('camera', [-self.camera_angle, 0])], #5  #verticle
            [('camera', [self.camera_angle, 0])],  #6  #verticle
            [('back', 1)], #7
            [('left', 1)], #8
            [('right', 1)], #9
            [('jump', 1)], #10
            #[('equip',11), ('use', 1)],
            [('forward', 1), ('attack', 1)], #11
        ]

        self.actions = []
        for actions in self._actions:
            act = self.env.action_space.noop()
            for a, v in actions:
                act[a] = v
            if self.always_attack:
                act['attack'] = 1
            self.actions.append(act)
        # add no-op action
        act = self.env.action_space.noop()
        self.actions.append(act)
        
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action):
        return self.actions[action]
        return self.actions[action]


def processed_actions_to_wrapper_actions_Navigation(dataset_actions, camera_margin=5):
    """
    Turn a batch of actions from dataset (`batch_iter`) to a numpy
    array that corresponds to batch of actions of ActionShaping wrapper (_actions).

    Camera margin sets the threshold what is considered "moving camera".

    Note: Hardcoded to work for actions in ActionShaping._actions, with "intuitive"
        ordering of actions.
        If you change ActionShaping._actions, remember to change this!

    Array elements are integers corresponding to actions, or "-1"
    for actions that did not have any corresponding discrete match.
    """
    # There are dummy dimensions of shape one
    camera_actions = dataset_actions[:,10:].astype(np.float32)
    attack_actions = dataset_actions[:,0].astype(np.float32)
    forward_actions = dataset_actions[:,3].astype(np.float32)
    jump_actions = dataset_actions[:,4].astype(np.float32)
    back_actions = dataset_actions[:,1].astype(np.float32)
    left_actions = dataset_actions[:,5].astype(np.float32)
    right_actions =  dataset_actions[:,6].astype(np.float32)
    equip_actions = dataset_actions[:,2]
    use_actions = dataset_actions[:,9].astype(np.float32)
    sneak_actions = dataset_actions[:,7].astype(np.float32)
    sprint_actions = dataset_actions[:,8].astype(np.float32)
    batch_size = len(camera_actions)
    
    actions = np.zeros((batch_size,), dtype=int)

    for i in range(len(camera_actions)):
        # Moving camera is most important (horizontal first!!!)
        if camera_actions[i][1] < -camera_margin:
            actions[i] = 3
        elif camera_actions[i][1] > camera_margin:
            actions[i] = 4
        elif camera_actions[i][0] > camera_margin:
            actions[i] = 5
        elif camera_actions[i][0] < -camera_margin:
            actions[i] = 6
        elif forward_actions[i] == 1:
            if jump_actions[i] == 1:
                actions[i] = 2
            elif attack_actions[i] == 1:
                actions[i] = 11
            else:
                actions[i] = 1
        elif attack_actions[i] == 1:
            actions[i] = 0
        elif left_actions[i] == 1:
            actions[i] = 8
        elif right_actions[i] ==1:
            actions[i] = 9
        elif jump_actions[i] == 1:
            actions[i] = 10
        elif back_actions[i] == 1:
            actions[i] = 7
        elif sum(dataset_actions[i,(0,1,3,4,5,6,7,8,9)].astype(np.float32)):
            # actual noop
            actions[i] = 12
        else: #catch everthing else and remove later
            actions[i] = 99
            
    return actions




# return only image as the observation
class PovOnlyObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.env.observation_space['pov']

    def observation(self, observation):
        obs = observation['pov'].squeeze().astype(np.float32)
        # Transpose observations to be channel-first (BCHW instead of BHWC)
        obs = obs.transpose(2, 0, 1)
        # Normalize observations
        obs /= 255.0
        return obs