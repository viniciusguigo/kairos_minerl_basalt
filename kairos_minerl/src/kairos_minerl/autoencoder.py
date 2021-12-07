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
import glob
from torch.utils.data import TensorDataset, DataLoader


class Autoencoder(nn.Module):
    "Autoencoder model for MineRL adapted from Atari"
    
    def __init__(self, input_shape, output_dim, version = 1):
        super().__init__()
        self.input_shape = input_shape
        self.n_input_channels = input_shape[0]
        self.version = version
        
        #initialize autoencoder
        if version == 1:
            self.autoencoder_v1()
        elif version == 2:
            self.autoencoder_v2()
        elif version == 3:
            self.autoencoder_v3()
        elif version == 4:
            self.autoencoder_v4()
        elif version == 5:
            self.autoencoder_v5()
                
        
    # V1 autoencoder (low-rez 64x64 image): fully conv2d layers
    def autoencoder_v1(self):
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.n_input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1),
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            
        )
        
    # V5 autoencoder (low-rez 64x64 image): fully conv2d layers
    def autoencoder_v5(self):
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.n_input_channels, 32, kernel_size=16, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 32, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 32, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(32, 32, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(32, 3, kernel_size=16, stride=1, padding=0),
            nn.Sigmoid(),
            
        )

    # V2 autoencoder (low-rez 64x64 image): fully conv2d layers
    def autoencoder_v2(self):
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.n_input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(3),
        )
        
        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )            
    
    # V3 autoencoder (low-rez 64x64 img): conv2d + dense layers
    def autoencoder_v3(self):
        self.code_size = 256
        self.flatten_length = 3*16*16
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.n_input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.BatchNorm2d(3),
            nn.Flatten(),
            nn.Linear(self.flatten_length, self.code_size),
            nn.SELU(),
            nn.BatchNorm1d(self.code_size),
        )
        
        # an intermediate layer because nn has no reshape layer :(
        self.intermediate = nn.Sequential(
            nn.Linear(self.code_size, self.flatten_length),
            nn.SELU(),
            nn.BatchNorm1d(self.flatten_length),
        )
        
        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        ) 
        
        # V3 autoencoder (low-rez 64x64 img): conv2d + dense layers
    def autoencoder_v4(self):
        self.code_size = 256
        self.flatten_length = 300
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.n_input_channels, 64, kernel_size=16, stride=2, padding=1),
            nn.SELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=8, stride=2, padding=1),
            nn.SELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 3, kernel_size=4, stride=1, padding=1),
            nn.SELU(),
            nn.BatchNorm2d(3),
            nn.Flatten(),
            nn.Linear(300, self.code_size),
            nn.SELU(),
            nn.BatchNorm1d(self.code_size),   
        )
        # an intermediate layer because nn has no reshape layer :(
        self.intermediate = nn.Sequential(
            nn.Linear(self.code_size, 300),
            nn.SELU(),
            nn.BatchNorm1d(300),
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 64, kernel_size=4, stride=1, padding=1),
            nn.SELU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 128, kernel_size=8, stride=2, padding=1),
            nn.SELU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 3, kernel_size=16, stride=2, padding=1),
            nn.Sigmoid(),
        ) 
        
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        # we handle verion 3 slightly differently        
        if self.version == 3:
            x = self.encoder(observations)
            x = self.intermediate(x)
            x = th.reshape(x, (-1, 3, 16, 16))
            return self.decoder(x)
        if self.version == 4: 
            x = self.encoder(observations)
            x = self.intermediate(x)
            x = th.reshape(x, (-1, 3, 10, 10))
            return self.decoder(x)
        else:
            return self.decoder(self.encoder(observations))


class FC_Autoencoder(nn.Module):
    "Dense, fully connected Autoencoder model for MineRL"
    def __init__(self, input_shape):
        super(FC_Autoencoder, self).__init__()
        
        # model parameters
        self.input_shape = input_shape
        self.code_size = 256
        
        # define the layers in our network
        self.fc_encoder = nn.Linear(self.input_shape[0]*self.input_shape[1]*self.input_shape[2], self.code_size)
        self.fc_decoder = nn.Linear(self.code_size, self.input_shape[0]*self.input_shape[1]*self.input_shape[2])
        
    def forward(self, x):
        
        # flatten image into vector
        x = th.flatten(x, start_dim = 1)
        x = self.fc_encoder(x)
        x = F.relu(x)
        x = self.fc_decoder(x)
        x = F.sigmoid(x)
        x = th.reshape(x, (-1, 3,64,64))
        return x
 

if __name__ == "__main__":
    
    print("Autoencoder")
    
    # learning parameters
    num_epochs = 1000
    batch_size = 64
    learning_rate = 2e-3
    input_shape = (3, 64, 64)
    output_shape = 7
    
    # get device
    device = th.device("cuda" if th.cuda.is_available() else "cpu") 
    
    # initialize autoencoder model (and place it on the gpu)
    ae_network = Autoencoder(input_shape, output_shape, version = 5).to(device)
    #ae_network = FC_Autoencoder(input_shape).to(device)
    summary(ae_network, input_shape)
    
    # setup optimizer and loss
    optimizer = th.optim.Adam(ae_network.parameters(), lr = learning_rate)
    #loss_function = nn.BCELoss()
    loss_function = nn.MSELoss()
    
    # load all minerl basalt demonstration data
    # The dataset is available in data/ directory from repository root.
    MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')
    # Find all MineRLBasalt tasks
    tasks = glob.glob(os.path.join(MINERL_DATA_ROOT, 'MineRLBasalt*'))
    
    
    data_images = np.load(tasks[0] + '/' + 'images.npy')
    for i in range(len(tasks)-1):
        print(i)
        data_images = np.concatenate((data_images, np.load(tasks[i+1] + '/' + 'images.npy')),axis=0)
    
    # Process, convert to pytorch dataset and wrap in a dataloader
    data_images = data_images.transpose(0,3,1,2)
    x_train = th.Tensor(data_images)
    # convert to dataset
    trainset = TensorDataset(x_train, x_train)
    # wrap in dataloader
    trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = True)
    
    # place network in train mode
    ae_network.train()
    
    # train model
    print('Training autoencoder model')
    iter_count = 0
    losses = []
    # iterate over epoch
    for epoch in range(num_epochs): # loop over dataset num_epoch times
        # iterate over batches (go through entire dataset once)
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, _ = data
            
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = ae_network(inputs.to(device))
            loss = loss_function(outputs, inputs.to(device))
            loss.backward()
            optimizer.step()
    
            # print statistics
            iter_count += 1
            losses.append(loss.item())
            if (iter_count % 100) == 0:
                mean_loss = sum(losses) / len(losses)
                tqdm.write("Epoch {}, Batch {} of {} Loss {:<10.5f}".format(epoch, i+1, len(trainloader), mean_loss))
                losses.clear()
                
                # show reconstruction results
                inputs = inputs.detach().cpu().numpy()
                inputs = inputs.transpose(0, 2, 3, 1)
                outputs = outputs.detach().cpu().numpy()
                outputs = (outputs.transpose(0, 2, 3, 1))
                f,(ax1, ax2) = plt.subplots(1,2)
                ax1.imshow(inputs[1,:,:,:])
                ax2.imshow(outputs[1,:,:,:])
                plt.show()
        iter_count = 0
        th.save(ae_network, "full_autoencoder_v5_epoch{}.pth".format(epoch))
    print('Finished Training')
    
    
    
    '''
    #########################################################
    ## Old Way of training useing a minerl dataset
    #########################################################
    # load data
    print("Processing data pipeline")
    DATA_DIR = os.getenv('MINERL_DATA_ROOT', "/home/nicholas/mineRL/hcxs_basalt/data")
    data = minerl.data.make("MineRLBasaltFindCave-v0",  data_dir=DATA_DIR, num_workers=4)
    
    # train model
    print("Training AE model")
    iter_count = 0
    losses = []
    for dataset_obs, dataset_actions, _, _, _ in tqdm(data.batch_iter(num_epochs=num_epochs, batch_size=batch_size, seq_len=1)):
        # We only use pov observations (also remove dummy dimensions)
        img = dataset_obs["pov"].squeeze().astype(np.float32)
        # Transpose observation images to be channel-first (BCHW instead of BHWC)
        img = img.transpose(0, 3, 1, 2)
        # Normalize observations
        img /= 255.0
        
        # do forward pass
        output_img = ae_network(th.from_numpy(img).float().to(device))
        
        #compute binary cross entropy loss
        loss = loss_function(output_img, th.from_numpy(img).float().to(device))
        
        # do backward pass (gradient decent update)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        iter_count += 1
        losses.append(loss.item())
        if (iter_count % 100) == 0:
            mean_loss = sum(losses) / len(losses)
            tqdm.write("Iteration {}. Loss {:<10.3f}".format(iter_count, mean_loss))
            losses.clear()
            
            # show reconstruction results
            img = img.transpose(0, 2, 3, 1)
            output_img = output_img.detach().cpu().numpy()
            output_img = (output_img.transpose(0, 2, 3, 1))
            f,(ax1, ax2) = plt.subplots(1,2)
            ax1.imshow(img[1,:,:,:])
            ax2.imshow(output_img[1,:,:,:])
            plt.show()
            
            
    th.save(ae_network, "find_cave_autoencoder_v3_FC.pth")
    del data
        
    '''     


