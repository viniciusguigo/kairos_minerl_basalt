import argparse
import math
import sys

from tqdm import tqdm
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
import gym
import minerl
import os
from torchsummary import summary
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import WeightedRandomSampler
from kairos_minerl.gail_wrapper import PovOnlyObservation
from torch.autograd import Variable
from minerl.herobraine.hero.spaces import Box
from kairos_minerl.gail_wrapper import ActionShaping_FindCave, ActionShaping_Waterfall, ActionShaping_Animalpen, ActionShaping_Villagehouse, ActionShaping_Navigation
from kairos_minerl.gail_wrapper import processed_actions_to_wrapper_actions_FindCave, processed_actions_to_wrapper_actions_Waterfall
from kairos_minerl.gail_wrapper import processed_actions_to_wrapper_actions_Animalpen, processed_actions_to_wrapper_actions_Villagehouse
from kairos_minerl.gail_wrapper import processed_actions_to_wrapper_actions_Navigation
from minerl.herobraine.wrappers import downscale_wrapper

# GAIL policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        self.input_shape = state_dim.shape
        self.input_channels = self.input_shape[2]
        self.action_dim = action_dim
        self.state_dim = state_dim

        # features
        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        ) #output size is 3x16x16 = 768
        
        # policy
        self.policy = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(768, 128),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )  

    def forward(self, states):
        x = self.features(states)
        x = th.flatten(x,1)
        probs = th.nn.functional.softmax(self.policy(x),dim=1)
        try:
            distb = th.distributions.Categorical(probs)
        except Exception as e:
            print('***torch.distribution.Categorical Error***')
            print(e)
            print('probs')
            print(probs.detach().cpu())
        return distb
        #return probs

class BehaviorCloning(nn.Module):
    "Behavior cloning model for MineRL"

    def __init__(self, action_dim, input_shape = (3, 64, 64)):
        super().__init__()
        self.input_shape = input_shape
        self.input_channels = input_shape[0]
        self.action_dim = action_dim

        # features
        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(3),
        )  # output size is 3x16x16 = 768

        # policy
        self.policy = nn.Sequential(
            nn.Dropout(),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.features(x)
        x = th.flatten(x, 1)
        x = self.policy(x)
        return x

class BehaviorCloning_128(nn.Module):
    "Behavior cloning model for MineRL"

    def __init__(self, action_dim, input_shape = (3, 64, 64)):
        super().__init__()
        self.input_shape = input_shape
        self.input_channels = input_shape[0]
        self.action_dim = action_dim

        # features
        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(3),
        )  # output size is 3x16x16 = 768

        # policy
        self.policy = nn.Sequential(
            nn.Dropout(),
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.features(x)
        x = th.flatten(x, 1)
        x = self.policy(x)
        return x

class BehaviorCloningLSTM(nn.Module):
    "Behavior cloning model for MineRL"

    def __init__(self, action_dim, input_shape = (3, 64, 64)):
        super().__init__()
        self.input_shape = input_shape
        self.input_channels = input_shape[0]
        self.action_dim = action_dim

        # LSTM parameters
        self.n_layers = 1
        self.dropout_rate = 0
        self.batch_first = True
        self.hidden_dim = 128

        # features
        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(3),
        )  # output size is 3x16x16 = 768

        # policy layers
        self.embedding = nn.Linear(768, 128)
        self.lstm = nn.LSTM(128, self.hidden_dim, num_layers = self.n_layers,
                            batch_first = self.batch_first, dropout=self.dropout_rate)
        self.out_layer = nn.Linear(128, self.action_dim)

    def forward(self, x, hidden):
        # extract image features from conv net
        x = self.features(x)
        x = th.flatten(x, 1)
        
        # get first embedding
        x = F.relu(self.embedding(x))
        x = th.unsqueeze(x, 0)
        
        # lstm layer
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # output prediction layer
        lstm_out = th.squeeze(lstm_out)
        out = self.out_layer(lstm_out)
        #out = out.view(batch_size, -1) #?
        #out = out[:,-1] #?
        return out, hidden
    
    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", "-e", type=int, default=40)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--learning_rate", "-l", type=float, default=0.001)
    parser.add_argument("--val_interval", "-v", type=int, default=2)
    parser.add_argument("--save_interval", "-s", type=int, default=5)
    parser.add_argument("--model_type", "-m", type=str, default="combined", choices=["find_cave", "make_waterfall", "animal_pen", "village_house", "combined"])
    parser.add_argument("--navigation", "-n", type=bool, default = True)
    parser.add_argument("--rollout_model", "-r", type=str, default="")
    parser.add_argument("--high_res", type=bool, default=False)
    parser.add_argument("--gail", type=bool, default=False)
    parser.add_argument("--lstm", type=bool, default=False)
    args = parser.parse_args()
    return args

# load training data
def load_training_data(data_paths):
    data_images = None
    data_actions = None
    for data_path in data_paths: 
        if data_images is None:
            data_images = np.load(os.path.join(data_path, "images.npy"))
            data_actions = np.load(os.path.join(data_path, "actions.npy"))
        else:
            data_images = np.concatenate([data_images, np.load(os.path.join(data_path, "images.npy"))], axis=0)
            data_actions = np.concatenate([data_actions, np.load(os.path.join(data_path, "actions.npy"))], axis=0)
    
    data_images = data_images.transpose(0,3,1,2)
    
    return data_images, data_actions

def save_model(model, model_type, loss, accuracy, epoch, experiment_name, lstm=False):
    if not os.path.exists("train/{}/{}".format(experiment_name, model_type)):
        os.makedirs("train/{}/{}".format(experiment_name, model_type))
    th.save(model, "train/{}/{}/{}_BC_v1_e{}_l{:.3f}_a{:.3f}.pth".format(experiment_name, model_type, model_type, epoch , loss, accuracy))
    th.save(model.state_dict(), "train/{}/{}/{}_BC_v1_e{}_l{:.3f}_a{:.3f}_dict.pth".format(experiment_name, model_type, model_type, epoch, loss, accuracy))

cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

def save_best_model(model: BehaviorCloning, model_type, experiment_name):
    all_model_paths = os.listdir("train/{}/{}/".format(experiment_name, model_type))
    all_model_paths = [fname for fname in all_model_paths if "dict" not in fname and ".pth" in fname and "final" not in fname]

    all_model_paths_split = [(fname, float(fname.split("_")[5][1:]), index) for index, fname in enumerate(all_model_paths)]
    best_model_path = sorted(all_model_paths_split, key=lambda x: x[1], reverse=False)[0]
    best_model_state_dict_path = os.path.join("train/{}/{}".format(experiment_name, model_type), best_model_path[0].replace(".pth", "_dict.pth"))

    model.load_state_dict(th.load(best_model_state_dict_path))

    th.save(model, "train/{}/{}_best_BC_model.pth".format(experiment_name, model_type))
    th.save(model.state_dict(), "train/{}/{}_best_BC_model_dict.pth".format(experiment_name, model_type))


# compute accuracy of a trained model given a dataset
def compute_accuracy(model, dataset, loss_function, device, is_val, lstm = False):
    running_loss = 0.0
    num_correct = 0.0
    total_num = 0.0
    accuracy = 0.0
    model.eval()
    title = "Val" if is_val else "Test"
    with th.no_grad():
        #initialize lstm hidden state
        if lstm:
            hidden = model.init_hidden(1, device)
        bar = tqdm(total=len(dataset), file=sys.stdout)
        for i, data in enumerate(dataset, 0):
            inputs, labels = data
            if lstm:
                hidden = tuple([e.data for e in hidden])
                pred_labels, hidden = model(inputs.to(device), hidden)
            else:
                pred_labels = model(inputs.to(device))
            
            loss = loss_function(pred_labels, labels.long().to(device))
            
            # compute running training accuracy
            pred_labels = th.argmax(pred_labels, dim=1)
            num_correct += th.sum(labels.to(device)==pred_labels)
            total_num += len(pred_labels)
            accuracy = num_correct / total_num

            
            running_loss += loss.item()
            bar.set_description("[{}] loss: {:10.3f} | accuracy: {:10.3f}".format(title, loss, accuracy))
            bar.update()
        accuracy = (num_correct * 1.0 / len(dataset.dataset))
        bar.set_description("[{}] loss: {:10.3f} | accuracy: {:10.3f}".format(title, running_loss / len(dataset), accuracy))
        bar.close()

    accuracy = (num_correct * 1.0 / len(dataset.dataset))
    loss = running_loss * 1.0 / len(dataset)
    return loss, accuracy


# train module for training the full pipeline
def train(num_epochs = 40, batch_size = 32, learning_rate = 0.001, val_interval = 2, save_interval = 5, navigation = True,
          lstm = False, model_type = "find_cave", experiment_name = "test"):

    print("Training Behavior Cloning model for {} Task!".format(model_type))
    
    # model task type: defines the data path, and action processor/wrappers for each task
    model_type_dict = {
        "find_cave": {
            "paths": ['data/MineRLBasaltFindCave-v0'],
            "action_processor":processed_actions_to_wrapper_actions_FindCave,
            "action_wrapper": ActionShaping_FindCave
        },
        "animal_pen": {
            "paths": ['data/MineRLBasaltCreateVillageAnimalPen-v0',
                      'data/MineRLBasaltCreatePlainsAnimalPen-v0'],
            "action_processor":processed_actions_to_wrapper_actions_Animalpen,
            "action_wrapper": ActionShaping_Animalpen
        },
        "make_waterfall": {
            "paths": ['data/MineRLBasaltMakeWaterfall-v0'],
            "action_processor":processed_actions_to_wrapper_actions_Waterfall,
            "action_wrapper": ActionShaping_Waterfall
        },
        "village_house": {
            "paths": ['data/MineRLBasaltBuildVillageHouse-v0'],
            "action_processor":processed_actions_to_wrapper_actions_Villagehouse,
            "action_wrapper": ActionShaping_Villagehouse
        },
        "combined": {
            "paths": ['data/MineRLBasaltFindCave-v0',
                      'data/MineRLBasaltMakeWaterfall-v0',
                      'data/MineRLBasaltCreateVillageAnimalPen-v0',
                      'data/MineRLBasaltCreatePlainsAnimalPen-v0',
                      'data/MineRLBasaltBuildVillageHouse-v0',],
            "action_processor":processed_actions_to_wrapper_actions_Navigation,
            "action_wrapper": ActionShaping_Navigation
        },
        "navigation":{
            "action_processor":processed_actions_to_wrapper_actions_Navigation,
            "action_wrapper": ActionShaping_Navigation
            }
    }
    
    # load data
    data_paths = model_type_dict[model_type]["paths"]
    data_images, data_actions = load_training_data(data_paths)
    
    # process data
    if navigation == True:
        action_processor = model_type_dict["navigation"]["action_processor"]
    
    else:
        action_processor = model_type_dict[model_type]["action_processor"]
    
    data_actions = action_processor(data_actions)
    
    if navigation == True:
        data_images = data_images[data_actions!=99,:,:]
        data_actions=data_actions[data_actions!=99]
    
    num_classes = np.max(data_actions) + 1
    
    #data_actions = np.squeeze(np.eye(num_classes)[data_actions.reshape(-1)])
    
    input_shape = (3, 64, 64)

    # get device
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # initialize autoencoder model (and place it on the gpu)
    if lstm:
        print("Initializing LSTM BC model!")
        bc_model = BehaviorCloningLSTM(num_classes).to(device)
    else:
        print("Initializing Standard BC model!")
        bc_model = BehaviorCloning(num_classes).to(device)
        summary(bc_model, input_shape)
    
    # setup optimizer and loss
    optimizer = th.optim.Adam(bc_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_function = nn.modules.CrossEntropyLoss()

    # split data into train/test
    if lstm:
        x_train, x_val, y_train, y_val = train_test_split(data_images, data_actions, test_size=0.1, shuffle=False)
    else:
        x_train, x_val, y_train, y_val = train_test_split(data_images, data_actions, test_size=0.1, random_state=2, stratify=data_actions)
        
    # convert into pytorch dataset and wrap in a dataloader
    x_train = th.Tensor(x_train)
    x_val = th.Tensor(x_val)
    y_train = th.LongTensor(y_train)
    y_val = th.LongTensor(y_val)

    # convert to dataset
    trainset = TensorDataset(x_train, y_train)
    valset = TensorDataset(x_val, y_val)

    # wrap in dataloader
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size)    

    # place network in train mode
    bc_model.train()

    # train model
    print('Training model')

    minimum_val_loss = sys.maxsize
    for epoch in range(num_epochs):  # loop over dataset num_epoch times
        running_loss = 0.0
        num_correct = 0.0
        total_num = 0.0
        accuracy = 0.0
        bc_model.train()
        #initialize lstm hidden state
        if lstm:
            hidden = bc_model.init_hidden(1, device)
        
        bar = tqdm(total=math.ceil(len(trainset)/trainloader.batch_size), file=sys.stdout, desc="[Epoch: {}] loss: {:10.3f} | accuracy: {:10.3f}".format(epoch + 1, running_loss, accuracy))
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if lstm:
                hidden = tuple([e.data for e in hidden])
                outputs, hidden = bc_model(inputs.to(device), hidden)
            else:
                outputs = bc_model(inputs.to(device))
                
            loss = loss_function(outputs, labels.long().to(device))
            loss.backward()
            optimizer.step()

            # compute running training accuracy
            pred_labels = th.argmax(outputs, dim=1)
            num_correct += th.sum(labels.to(device)==pred_labels)
            total_num += len(pred_labels)
            accuracy = num_correct / total_num

            # print statistics
            running_loss += loss.item()
            bar.set_description("[Epoch: {}] loss: {:10.3f} | accuracy: {:10.3f}".format(epoch + 1, loss, accuracy))
            bar.update()

        # Compute training metrics at the end of each epoch
        acc = (num_correct * 1.0 / len(trainset))
        bar.set_description("[Epoch: {}] loss: {:10.3f} | accuracy: {:10.3f}".format(epoch + 1, running_loss / len(trainloader), acc))
        bar.close()

        if epoch % val_interval == 0:
            print("\nComputing validation")
            val_loss, val_accuracy = compute_accuracy(bc_model, valloader, loss_function, device, True, lstm)
            if val_loss < minimum_val_loss:
                print("\nValidation loss lower: {:10.3f} < {:10.3f}, saving...".format(val_loss, minimum_val_loss))
                save_model(
                    bc_model, model_type, val_loss, val_accuracy, epoch, experiment_name=experiment_name, lstm = lstm)
                minimum_val_loss = val_loss
            else:
                print("\nValidation loss higher: {:10.3f} > {:10.3f}".format(val_loss, minimum_val_loss))

    print()
    print('Finished Training')
    
    # save best model
    save_best_model(bc_model, model_type, experiment_name)

def main(model_type = None):
    # learning parameters
    args = parse_args()
    num_epochs = args.num_epochs  # 50
    batch_size = args.batch_size  # 32
    learning_rate = args.learning_rate  # 1e-3    
    val_interval = args.val_interval
    save_interval = args.save_interval
    navigation = args.navigation
    lstm = args.lstm
    
    if model_type is None:
        model_type = args.model_type

    train(num_epochs=num_epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    val_interval = val_interval,
    save_interval = save_interval,
    navigation = navigation, 
    lstm = lstm, 
    model_type = model_type)

    

def rollout():
    print('Testing Behavior Cloning model')

    # get device
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # get task type
    args = parse_args()
    model_type = args.model_type
    high_res = args.high_res
    if high_res:
        high_res_str = 'HighRes'
    else:
        high_res_str = ''
    lstm = args.lstm
    gail = args.gail
    
    # model task type: defines the data path, and action processor/wrappers for each task
    model_type_dict = {
        "find_cave": {
            "path": 'train/bc_model/find_cave',
            'env_name':'MineRLBasaltFindCave{}-v0'.format(high_res_str),
            "action_wrapper": ActionShaping_FindCave
        },
        "animal_pen": {
            "path": 'train/bc_model/animal_pen',
            'env_name':'MineRLBasaltCreateVillageAnimalPen{}-v0'.format(high_res_str),
            "action_wrapper": ActionShaping_Animalpen
        },
        "make_waterfall": {
            "path": 'train/bc_model/make_waterfall',
            'env_name':'MineRLBasaltMakeWaterfall{}-v0'.format(high_res_str),
            "action_wrapper": ActionShaping_Waterfall
        },
        "village_house": {
            "path": 'train/bc_model/village_house',
            'env_name':'MineRLBasaltBuildVillageHouse{}-v0'.format(high_res_str),
            "action_wrapper": ActionShaping_Villagehouse
        },
        "combined": {
            "path": 'train/bc_model_V3_CCE_False/combined',
            'env_name':'MineRLBasaltFindCave{}-v0'.format(high_res_str),
            "action_processor":processed_actions_to_wrapper_actions_Navigation,
            "action_wrapper": ActionShaping_Navigation
        },
    }
    
    # create gym environment
    env = gym.make(model_type_dict[model_type]['env_name'])
    if high_res:
        env = downscale_wrapper.DownscaleWrapper(env)
    env = PovOnlyObservation(env)
    ActionShaping = model_type_dict[model_type]['action_wrapper']
    env = ActionShaping(env)
    
    # load pre-trained BC model
    if gail:
        bc_model = PolicyNetwork(Box(-np.inf, np.inf, shape=(64,64,3)), 17).to(device)
        bc_model.load_state_dict(th.load(args.rollout_model))
    else:
        bc_model = th.load(os.path.join(model_type_dict[model_type]['path'], args.rollout_model)).to(device)
    
    
    num_actions = env.action_space.n
    action_list = np.arange(num_actions)
    
    # rollout 10 games
    for game_i in range(10):
        print("Rolling out game {} for task {}".format(game_i, model_type))
        obs = env.reset()
        done = False
        reward_sum = 0
        while not done:
            # Forward pass through model
            output = bc_model((th.tensor(np.expand_dims(obs,axis=0))).to(device))
            
            if gail:
                probabilities = output
                action = probabilities.sample().detach().cpu().numpy().squeeze()
            else:
                # Into numpy
                probabilities = th.softmax(output, 1)
                probabilities = probabilities.detach().cpu().numpy()
                # Sample action according to the probabilities
                action = probabilities
                #action = np.random.choice(action_list, p=probabilities[0])
            # Take a step in environment
            obs, reward, done, info = env.step(action)
            if info !={}:
                print(info)
            env.render()
            reward_sum += reward
        print("Game {}, total reward {}".format(game_i, reward_sum))

    env.close()

if __name__ == "__main__":
    args = parse_args()
    
    if args.rollout_model != "":
        # rollout model
        rollout()
    else:
        # train bc model
        main("make_waterfall")
        main("animal_pen")
        main("village_house")
        main("find_cave")
