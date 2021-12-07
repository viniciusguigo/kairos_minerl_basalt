from tqdm import tqdm
import numpy as np
import torch as th
from torch import nn
import gym
import minerl
import os

class NatureCNN(nn.Module):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    Nicked from stable-baselines3:
        https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py

    :param input_shape: A three-item tuple telling image dimensions in (C, H, W)
    :param output_dim: Dimensionality of the output vector
    """

    def __init__(self, input_shape, output_dim):
        super().__init__()
        n_input_channels = input_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.zeros(1, *input_shape)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class PovOnlyObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.env.observation_space['pov']

    def observation(self, observation):
        return observation['pov']


class ActionShaping(gym.ActionWrapper):
    def __init__(self, env, camera_angle=10, always_attack=False):
        super().__init__(env)

        self.camera_angle = camera_angle
        self.always_attack = always_attack
        self._actions = [
            [('attack', 1)],
            [('forward', 1)],
            #             [('back', 1)],
            #             [('left', 1)],
            #             [('right', 1)],
            #             [('jump', 1)],
            #             [('forward', 1), ('attack', 1)],
            [('forward', 1), ('jump', 1)],
            [('camera', [-self.camera_angle, 0])],
            [('camera', [self.camera_angle, 0])],
            [('camera', [0, self.camera_angle])],
            [('camera', [0, -self.camera_angle])],
        ]

        self.actions = []
        for actions in self._actions:
            act = self.env.action_space.noop()
            for a, v in actions:
                act[a] = v
            if self.always_attack:
                act['attack'] = 1
            self.actions.append(act)

        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action):
        return self.actions[action]


def dataset_action_batch_to_actions(dataset_actions, camera_margin=5):
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
    camera_actions = dataset_actions["camera"].squeeze()
    attack_actions = dataset_actions["attack"].squeeze()
    forward_actions = dataset_actions["forward"].squeeze()
    jump_actions = dataset_actions["jump"].squeeze()
    batch_size = len(camera_actions)
    actions = np.zeros((batch_size,), dtype=np.int)

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
            else:
                actions[i] = 1
        elif attack_actions[i] == 1:
            actions[i] = 0
        else:
            # No reasonable mapping (would be no-op)
            actions[i] = -1
    return actions


def train():
    # Path to where MineRL dataset resides (should contain "MineRLTreechop-v0" directory)
    #DATA_DIR = r"F:\datasets\minerl_2020"
    DATA_DIR = os.getenv('MINERL_DATA_ROOT', "/Users/cody/Code/il-representations/data/minecraft")
        #"/Users/cody/Code/il-representations/data/minecraft"
    # How many times we train over dataset.
    # 3 epochs takes roughly 4 minutes on a Titan Xp GPU.
    EPOCHS = 1
    BATCH_SIZE = 32

    data = minerl.data.make("MineRLTreechop-v0",  data_dir=DATA_DIR, num_workers=4)

    # We know ActionShaping has seven discrete actions, so we create
    # a network to map images to seven values (logits), which represent
    # likelihoods of selecting those actions
    network = NatureCNN((3, 64, 64), 7) #.cuda()
    optimizer = th.optim.Adam(network.parameters(), lr=1e-4)
    loss_function = nn.CrossEntropyLoss()

    iter_count = 0
    losses = []
    for dataset_obs, dataset_actions, _, _, _ in tqdm(data.batch_iter(num_epochs=EPOCHS, batch_size=BATCH_SIZE, seq_len=1)):
        # We only use pov observations (also remove dummy dimensions)
        obs = dataset_obs["pov"].squeeze().astype(np.float32)
        # Transpose observations to be channel-first (BCHW instead of BHWC)
        obs = obs.transpose(0, 3, 1, 2)
        # Normalize observations
        obs /= 255.0

        # Actions need bit more work
        actions = dataset_action_batch_to_actions(dataset_actions)

        # Remove samples that had no corresponding action
        mask = actions != -1
        obs = obs[mask]
        actions = actions[mask]

        # Obtain logits of each action
        logits = network(th.from_numpy(obs).float()) #.cuda())

        # Minimize cross-entropy with target labels.
        # We could also compute the probability of demonstration actions and
        # maximize them.
        loss = loss_function(logits, th.from_numpy(actions).long()) #.cuda())

        # Standard PyTorch update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_count += 1
        losses.append(loss.item())
        if (iter_count % 1000) == 0:
            mean_loss = sum(losses) / len(losses)
            tqdm.write("Iteration {}. Loss {:<10.3f}".format(iter_count, mean_loss))
            losses.clear()

    th.save(network, "another_potato.pth")
    del data


def enjoy():
    network = th.load("another_potato.pth") #.cuda()

    env = gym.make('MineRLTreechop-v0')

    env = PovOnlyObservation(env)
    env = ActionShaping(env, always_attack=True)

    num_actions = env.action_space.n
    action_list = np.arange(num_actions)

    for game_i in range(10):
        obs = env.reset()
        done = False
        reward_sum = 0
        while not done:
            # Process the action:
            #   - Add/remove batch dimensions
            #   - Transpose image (needs to be channels-last)
            #   - Normalize image
            obs = th.from_numpy(obs.transpose(2, 0, 1)[None].astype(np.float32) / 255) #.cuda()
            # Turn logits into probabilities
            probabilities = th.softmax(network(obs), dim=1)[0]
            # Into numpy
            probabilities = probabilities.detach().cpu().numpy()
            # Sample action according to the probabilities
            action = np.random.choice(action_list, p=probabilities)

            obs, reward, done, info = env.step(action)
            reward_sum += reward
        print("Game {}, total reward {}".format(game_i, reward_sum))

    env.close()


if __name__ == "__main__":
    train()
    enjoy()
