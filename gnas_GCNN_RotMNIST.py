# Group Neural Architecture Search
# Example use: 

#=======================================================Imports===========================================================================================================
import time
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import argparse
from torch.autograd import Variable
from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4, P4H2ConvZ2, P4H2ConvP4H2, P4V2ConvZ2, P4V2ConvP4V2, P4H2V2ConvZ2, P4H2V2ConvP4H2V2, H2V2ConvZ2, H2V2ConvH2V2, H2ConvZ2, H2ConvH2, V2ConvZ2, V2ConvV2, Z2ConvZ2
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling

input_size = 10 #10 by default
hidden_sizes = [400, 400, 400]
output_size = input_size
n_actions = output_size
RotMNIST_traindata = np.loadtxt('/content/drive/MyDrive/Colab Notebooks/Group convolution neural networks/Datasets/RotMNIST/mnist_all_rotation_normalized_float_test.amat')
RotMNIST_testdata = np.loadtxt('/content/drive/MyDrive/Colab Notebooks/Group convolution neural networks/Datasets/RotMNIST/mnist_all_rotation_normalized_float_train_valid.amat')

class RotMNIST(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, traindata=RotMNIST_traindata, testdata=RotMNIST_testdata, train=True, transform=None):
        self.traindata=traindata
        self.testdata=testdata
        self.train = train
        self.transform = transform

    def __len__(self):
        length = len(self.traindata)
        if self.train != True:
          length = len(self.testdata)
        return length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.traindata[idx,0:784].reshape(1,28,28)
        label = self.traindata[idx, 784]
        if self.train != True:
          image = self.testdata[idx,0:784].reshape(1,28,28)
          label = self.testdata[idx, 784]

        if self.transform:
            image = self.transform(torch.from_numpy(image)) # transforms.ToPILImage()
        return (image, label)

#=====================================================================Q-Network=========================================================================================================

class QNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(QNet, self).__init__()                   # Inherited from the parent class nn.Module
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])  # 1st Full-Connected Layer: k (input data) -> 500 (hidden node)
        self.relu = nn.ReLU()                          # Non-Linear ReLU Layer: max(0,x)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2]) # 2nd Full-Connected Layer: 500 (hidden node) -> k (output class)
        self.fc4 = nn.Linear(hidden_sizes[2], num_classes)
    
    def forward(self, x):                              # Forward pass: stacking each layer together
        x = x/10
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        return out
        
#define the policy network and target network
device = 'cuda'
policy_net = QNet(input_size, hidden_sizes, output_size).to(device)
target_net = QNet(input_size, hidden_sizes, output_size).to(device)

#===================================================================== Dictionaries for dataloader and model selection =====================================================================================
aug_dict = {
            0: transforms.RandomRotation((-30,30)),
            1: transforms.RandomHorizontalFlip(p=0.5),
            2: transforms.RandomVerticalFlip(p=0.5),
            3: transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=None, shear=None),
            4: transforms.RandomAffine(degrees=0, translate=None, scale=(0.9, 1.1), shear=None),
            5: transforms.RandomAffine(degrees=0, translate=None, scale=None, shear=10)
            }

aug_array = [0]*6

garray1_dict = {
            '000': 'Z2ConvZ2',
            '100': 'P4ConvZ2',
            '101': 'P4H2ConvZ2',
            '110': 'P4V2ConvZ2',
            '111': 'P4H2V2ConvZ2',
            '011': 'H2V2ConvZ2',
            '001': 'H2ConvZ2',
            '010': 'V2ConvZ2'
            }

garray2_dict = {
            '000': 'Z2ConvZ2',
            '100': 'P4ConvP4',
            '101': 'P4H2ConvP4H2',
            '110': 'P4V2ConvP4V2',
            '111': 'P4H2V2ConvP4H2V2',
            '011': 'H2V2ConvH2V2',
            '001': 'H2ConvH2',
            '010': 'V2ConvV2'
            }

accuracy_dict = {
            '000': 12,
            '100': 20,
            '101': 20,
            '110': 20,
            '111': 32,
            '011': 12,
            '001': 12,
            '010': 12
            }

gconv_dict = {
            'Z2ConvZ2': Z2ConvZ2,
            'P4ConvZ2': P4ConvZ2,
            'P4ConvP4': P4ConvP4,
            'P4H2ConvZ2': P4H2ConvZ2,
            'P4H2ConvP4H2': P4H2ConvP4H2,
            'P4V2ConvZ2': P4V2ConvZ2,
            'P4V2ConvP4V2': P4V2ConvP4V2,
            'P4H2V2ConvZ2': P4H2V2ConvZ2,
            'P4H2V2ConvP4H2V2': P4H2V2ConvP4H2V2,
            'H2V2ConvZ2': H2V2ConvZ2,
            'H2V2ConvH2V2': H2V2ConvH2V2,
            'H2ConvZ2': H2ConvZ2,
            'H2ConvH2': H2ConvH2,
            'V2ConvZ2': V2ConvZ2,
            'V2ConvV2': V2ConvV2
            }

splitgroup_size_dict = {
            'Z2ConvZ2': 1,
            'P4ConvZ2': 4,
            'P4ConvP4': 4,
            'P4H2ConvZ2': 8,
            'P4H2ConvP4H2': 8,
            'P4V2ConvZ2': 8,
            'P4V2ConvP4V2': 8,
            'P4H2V2ConvZ2': 16,
            'P4H2V2ConvP4H2V2': 16,
            'H2V2ConvZ2': 4,
            'H2V2ConvH2V2': 4,
            'H2ConvZ2': 2,
            'H2ConvH2': 2,
            'V2ConvZ2': 2,
            'V2ConvV2': 2
            }

channel_multiplier_dict = {
            'Z2ConvZ2': 1,
            'P4ConvZ2': 1/3,
            'P4H2ConvZ2': 1/6,
            'P4V2ConvZ2': 1/6,
            'P4H2V2ConvZ2': 1/8,
            'H2V2ConvZ2': 1/3,
            'H2ConvZ2': 1/1.8,
            'V2ConvZ2': 1/1.8
            }
dense_multiplier_dict = {
            'Z2ConvZ2': 1,
            'P4ConvZ2': 1,
            'P4H2ConvZ2': 1,
            'P4V2ConvZ2': 1,
            'P4H2V2ConvZ2': 1/1.3,
            'H2V2ConvZ2': 1,
            'H2ConvZ2': 1,
            'V2ConvZ2': 1
            }

net_size_dict = {
            1: 1.5, # 1 = large
            0: 1  # 0 = small
            }
#===================================================================== Dataloaders =================================================================================================
def rotmnist_transform_array(aug_dict=aug_dict, aug_array=torch.zeros(6), seed=12345):
  # Outputs the array of transformations from aug_array
  torch.manual_seed(seed)
  transforms_array = []
  for i in range(len(aug_array)):
    if aug_array[i] > 0:
      transforms_array.append(aug_dict[i])
  transforms_array.append(transforms.ToPILImage())
  transforms_array.append(transforms.ToTensor())
  transforms_array.append(transforms.Normalize((0.5,), (0.5,)))
  return transforms_array

def rotmnist_trainloader(aug_dict=aug_dict, aug_array=aug_array, seed=12345, train_batch_size = 64, mini_trainsize = 4000, mini_train=True):
  torch.manual_seed(seed)
  transform = transforms.Compose(rotmnist_transform_array(aug_dict=aug_dict, aug_array=aug_array, seed=seed))
  trainset = RotMNIST(traindata=RotMNIST_traindata, testdata=RotMNIST_testdata, train=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)
  if mini_train == True:
    trainset_mini, _= torch.utils.data.random_split(trainset, [mini_trainsize, len(trainset) - mini_trainsize])
    trainloader = torch.utils.data.DataLoader(trainset_mini, batch_size=train_batch_size, shuffle=True, num_workers=2)
  return trainloader

def rotmnist_testloader(aug_dict=aug_dict, aug_array=aug_array, seed=12345, test_batch_size = 1000, mini_testsize = 1000, mini_test=True):
  torch.manual_seed(seed)
  testtransform = transforms.Compose(rotmnist_transform_array(aug_dict=aug_dict, seed=seed))
  testset = RotMNIST(traindata=RotMNIST_traindata, testdata=RotMNIST_testdata, train=False, transform=testtransform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True, num_workers=2)
  if mini_test == True:
    testset_mini, _= torch.utils.data.random_split(testset, [mini_testsize, len(testset) - mini_testsize])
    testloader = torch.utils.data.DataLoader(testset_mini, batch_size=test_batch_size, shuffle=True, num_workers=2)
  return testloader

#===================================================================== Model =======================================================================================================
class GNet(nn.Module):
    def __init__(self, GConv1='Z2ConvZ2', GConv2='Z2ConvZ2', net_size=1.5):
        super(GNet, self).__init__()
        splitgroup_size = splitgroup_size_dict[GConv1]
        channel_multiplier = channel_multiplier_dict[GConv1]
        k=20*net_size
        channels = math.floor(k*channel_multiplier)
        dense_neurons = math.floor(100*dense_multiplier_dict[GConv1])
        self.conv1 = gconv_dict[GConv1](1, channels, kernel_size=3)
        self.conv2 = gconv_dict[GConv2](channels, channels, kernel_size=3)
        self.conv3 = gconv_dict[GConv2](channels, channels, kernel_size=3)
        self.conv4 = gconv_dict[GConv2](channels, channels, kernel_size=3)
        self.fc1 = nn.Linear(4*4*channels*splitgroup_size, dense_neurons)
        self.fc2 = nn.Linear(dense_neurons, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(self.fc2(x))
        return F.log_softmax(x)
#===================================================================== Train and test for child network ============================================================================
def train(net, device, trainloader, criterion, optimizer, epoch, num_epochs=20, use_cuda=True):
  
  for i, data in enumerate(trainloader, 0):   # Load a batch of images with its (index, data, class)
      inputs, labels = data
      inputs, labels = inputs.to(device='cuda', dtype=torch.float), labels.to(device='cuda',dtype=torch.long)
      optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
      outputs = net(inputs)                             # Forward pass: compute the output class given a image
      loss = criterion(outputs, labels)                 # Compute the loss: difference between the output class and the pre-given label
      loss.backward()                                   # Backward pass: compute the weight
      optimizer.step()                                  # Optimizer: update the weights of hidden nodes
      
      #if (i+1) % 40 == 0:                              # Logging
          #print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
              #%(epoch+1, num_epochs, i+1, (train_size/batch_size), loss.item()))
  return net

def test(net, device, testloader, use_cuda=True):         
  correct = 0
  total = 0

  net.eval()
  for i, data in enumerate(testloader, 0):
      inputs, labels = data
      inputs, labels = inputs.to(device='cuda', dtype=torch.float), labels.to(device='cuda',dtype=torch.long)

      outputs = net(inputs)
      _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
      total += labels.size(0)                    # Increment the total count
      correct += (predicted == labels).sum()     # Increment the correct count
      
  #print('Accuracy of the network on the 1K test images:', (100.0 * correct / total))
  return (100.0 * correct / total)
#=====================================================================Environment (keep it clean :p)============================================================================
class Environment(object):
  #describes the environment for our Q-learning problem
  def __init__(self, device, child_network_dimensions, child_train_size=6000, child_test_size=1000, child_batch_size=32, child_test_batch_size=1000, child_lr=1e-3, child_epochs=4, state_size=10):
    # Start with all zero vector
    # Initialize only once, reset as many times as needed, to avoid recomputation of base accuracies.
    print("Group Neural Architecture Search using Group Decomposition and Reinforcement Learning!")
    self.device = device
    self.k = state_size # k = g_size for fully connected and k = g_size + 4*h_size for convolutional neural networks
    self.base_state = torch.tensor([1]+[0]*9)
    self.current_state = torch.tensor([0]*10) # g_size represents the size of the array of groups for equivariance
    self.next_state = torch.tensor([0]*10)
    self.models_trained = 1 # 1 corresponds to the base case
    self.time = 0                                                    # Further, base_reward should be equal to zero.
    self.next_state_string = ''.join(str(e) for e in self.next_state)
    self.device = device
    self.child_network_dimensions = child_network_dimensions
    self.child_epochs = child_epochs
    self.child_lr = child_lr
    self.child_train_size = child_train_size
    self.child_test_size = child_test_size
    self.child_batch_size = child_batch_size
    self.child_test_batch_size = child_test_batch_size
    self.base_accuracy = self.get_state_accuracy(self.base_state)
    self.base_reward = self.get_state_reward(self.base_accuracy, self.base_accuracy)
    current_state_string = ''.join(str(e) for e in self.current_state)
    self.visited_model_rewards = {current_state_string: self.base_reward} # Dictionary of models visited and their rewards. Models are saved in the form of a binary string of length g_size
    self.visited_model_accuracies = {current_state_string: self.base_accuracy} # Dictionary of models visited and their accuracies. Models are saved in the form of a binary string of length g_size

  def step(self, action):
    #returns from QNN for the data augmentation problem; for this toy example it is going to return +100 for state = all 1s, and -1 for anything else
    reward = 0
    accuracy = 0
    new_models_trained = False
    done = False
    self.time += 1

    # Make action
    action = action.item()

    # Update states
    for i in range(self.k):
      self.next_state[i] = self.current_state[i]
    self.next_state[action] = (self.current_state[action] + 1)%2 #remove for multiple adversaries
    self.update_current_state()
    self.next_state_string = ''.join(str(e) for e in self.next_state)

    # Update reward and accuracy
    if self.next_state_string not in self.visited_model_rewards:
      new_models_trained = True
      self.models_trained += 1
      accuracy = self.get_state_accuracy(self.current_state)
      reward = self.get_state_reward(accuracy, self.base_accuracy)
      self.visited_model_accuracies[self.next_state_string] = accuracy
      self.visited_model_rewards[self.next_state_string] = reward
    else:
      accuracy = self.visited_model_accuracies[self.next_state_string]
      reward = self.visited_model_rewards[self.next_state_string]

    if self.time > 50:
      done = True
    return accuracy, reward, new_models_trained, done

  def get_state_reward(self, state_accuracy, base_accuracy):
    return (state_accuracy - base_accuracy)*math.exp(abs(state_accuracy - base_accuracy))

  def get_state_accuracy(self, state):
    # Basic Hamming distance
    # goal_state = [0,1]*6
    # accuracy = sum([abs(x-y) for (x,y) in zip(goal_state,state)])
    state = [int(i) for i in state]
    torch.manual_seed(1)
    group_array = ''.join(str(e) for e in state[0:3])
    GConv1 = garray1_dict[group_array]
    GConv2 = garray2_dict[group_array]
    aug_array = state[3:9]
    net_size = net_size_dict[state[9]]
    #print("Equivariance array",eq_array)
    child_model = GNet(GConv1=GConv1, GConv2=GConv2, net_size=net_size).to(self.device)
    train_aug_array = aug_array
    test_aug_array = [0]*6
    #print("No. of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    child_criterion = nn.CrossEntropyLoss()
    child_optimizer = optim.SGD(child_model.parameters(), lr=self.child_lr, momentum=0.9)
    trainloader = rotmnist_trainloader(aug_dict=aug_dict, aug_array=train_aug_array, seed=12345, train_batch_size = self.child_batch_size, mini_trainsize = self.child_train_size, mini_train=True)
    testloader = rotmnist_testloader(aug_dict=aug_dict, aug_array=test_aug_array, seed=12345, test_batch_size = self.child_test_batch_size, mini_testsize = self.child_test_size, mini_test=True)
    testaccuracy = 0
    for epoch in range(self.child_epochs):
      #start_time = time.time()
      child_model = train(net=child_model, device=self.device, trainloader=trainloader, criterion=child_criterion, optimizer=child_optimizer, epoch=self.child_epochs)
      #time_elapsed = time.time() - start_time
      #print("Time elapsed",time_elapsed,"secs")
      # test
      testaccuracy = max(testaccuracy, test(net=child_model, device=self.device, testloader=testloader))
    return testaccuracy

  def update_avg_reward(self,reward,time):
    self.avg_reward = (self.avg_reward*(time - 1) + reward)/self.time

  def update_avg_test_acc(self,test_acc,time):
    self.avg_test_acc = (self.avg_test_acc*(time - 1) + test_acc)/self.time

  def update_current_state(self):
    for i in range(self.k):
      self.current_state[i] = self.next_state[i]

  def reset(self):
    #reset to state 0 w.p. 0.5, rest of the time set to an uniformly random vector of length k
    if random.random() > 0.0:
      self.current_state = torch.zeros(self.k)
    else:
      a = [0]*self.k + [1]*self.k
      self.current_state = torch.tensor(random.sample(a, self.k))
    self.next_state = torch.zeros(self.k)
    print("Starting state:",self.current_state)
    self.time = 0
    print("reset!")
#=====================================================================Replay Memory===================================================================================================

# Replay memory
Transition = namedtuple('Transition',('state','action','next_state','reward'))

class ReplayMemory(object):

  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []
    self.position = 0

  def push(self,*args):
    "Saves a transition"
    if len(self.memory) < self.capacity:
      self.memory.append(None)
    self.memory[self.position] = Transition(*args)
    self.position = (self.position + 1) % self.capacity

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

#=====================================================================Q-Network=========================================================================================================

class QNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(QNet, self).__init__()                   # Inherited from the parent class nn.Module
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])  # 1st Full-Connected Layer: k (input data) -> 500 (hidden node)
        self.relu = nn.ReLU()                          # Non-Linear ReLU Layer: max(0,x)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2]) # 2nd Full-Connected Layer: 500 (hidden node) -> k (output class)
        self.fc4 = nn.Linear(hidden_sizes[2], num_classes)
    
    def forward(self, x):                              # Forward pass: stacking each layer together
        x = x/10
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        return out

#=====================================================================Select action=========================================================================================================

steps_done = 0
def select_action(state, EPS, device, n_actions, policy_net):
  global steps_done
  sample = random.random()
  steps_done += 1
  if sample > EPS:
    with torch.no_grad():
      return policy_net(state).max(0)[1].view(1,1).to(device) #returns the index instead of the value
  else:
    return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

#=====================================================================Plots===============================================================================================================

def plot_update(x_models_trained, y_accuracy):
  fig = plt.figure()
  plt.title('Deep Q-learning RotMNIST')
  plt.xlabel('Models trained')
  plt.ylabel('Accuracy')
  plt.plot(x_models_trained, y_accuracy, label="Accuray")
  plt.pause(0.001)  #pause a bit so that plots are updated
  fig.savefig('GNAS_GCNN_RotMNIST.eps', format='eps', dpi=1000)

def plot_steps_update(x_steps, y_accuracy_per_step):
  fig = plt.figure()
  plt.title('Deep Q-learning RotMNIST')
  plt.xlabel('Steps')
  plt.ylabel('Accuracy')
  plt.plot(x_steps, y_accuracy_per_step, label="Accuray")
  plt.pause(0.001)  #pause a bit so that plots are updated
  fig.savefig('GNAS_GCNN_steps_RotMNIST.eps', format='eps', dpi=1000)

def plot_overall(x_models_trained, y_accuracy):
  # average windowed plot
  # averaged epsilon plot
  window_size = 60 # window_size < 200
  EPS_MODELS_TRAINED_LIST = [0,50,100,150,200,250,300,350,400,450,500,550,600]
  EPS_LIST = [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.01]

  y_accuracy_window = [sum(y_accuracy[i:i+window_size])/window_size for i in range(len(y_accuracy)-window_size)]
  y_epsilon_accuracy = []
  v_lines = []
  x_models_trained_window = x_models_trained[window_size:]

  for i in range(len(EPS_MODELS_TRAINED_LIST)-1):
    array = y_accuracy[EPS_MODELS_TRAINED_LIST[i]:EPS_MODELS_TRAINED_LIST[i+1]]
    y_epsilon_accuracy += [sum(array)/(len(array))]*(len(array))
    v_lines.append(sum(array)/(len(array)))

  y_epsilon_accuracy = y_epsilon_accuracy[window_size:len(x_models_trained)]
  fig = plt.figure()
  plt.title('Deep Q-learning RotMNIST')
  plt.xlabel('Models trained')
  plt.ylabel('Accuracy')
  plt.plot(x_models_trained_window, y_accuracy_window, label="Rolling mean Accuray")
  plt.fill_between(x_models_trained_window, y_epsilon_accuracy, label="Average Accuracy Per Epsilon",alpha=0.5)
  plt.legend()
  x1,x2,y1,y2 = plt.axis()
  plt.axis((x1,x2,5,y2))

  for i in range(len(v_lines)):
    plt.vlines(EPS_MODELS_TRAINED_LIST[i+1],5,v_lines[i], alpha=0.3)

  fig.text(0.18,0.14,'$\epsilon=1$')
  fig.text(0.275,0.14,'$.9$')
  fig.text(0.35,0.14,'$.8$')
  fig.text(0.428,0.14,'$.7$')
  fig.text(0.490,0.14,'$.6$')
  fig.text(0.568,0.14,'$.5$')
  fig.text(0.637,0.14,'$.4$')
  fig.text(0.705,0.14,'$.3$')
  fig.text(0.762,0.14,'$.2$')
  fig.text(0.795,0.14,'$.1$')
  fig.text(0.83,0.14,'$.05$')
  plt.pause(0.001)  #pause a bit so that plots are updated
  fig.savefig('GNAS_GCNN_RotMNIST'+'.eps', format='eps', dpi=1000)

#plot_overall(x_models_trained, y_accuracy)

#=====================================================================Optimize model===============================================================================================================

def optimize_model(k, device, memory, q_optimizer, Q_BATCH_SIZE, Q_GAMMA):
  if len(memory.memory) < Q_BATCH_SIZE:
    return
  transitions = memory.sample(Q_BATCH_SIZE)
  # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
  # detailed explanation). This converts batch-array of Transitions
  # to Transition of batch-arrays.
  batch = Transition(*zip(*transitions))

  # Compute a mask of non-final states and concatenate the batch elements
  # (a final state would've been the one after which simulation ended)

  state_batch = torch.cat(batch.state).view(Q_BATCH_SIZE,k).to(device)
  action_batch = torch.cat(batch.action).to(device)
  reward_batch = torch.cat(batch.reward).to(device)
  next_state_batch = torch.cat(batch.next_state).view(Q_BATCH_SIZE,k).to(device)


  # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
  # columns of actions taken. These are the actions which would've been taken
  # for each batch state according to policy_net
  #print("State batch:",state_batch)
  state_action_values = policy_net(state_batch).gather(1, action_batch).to(device)
  #print("state_action_values:",state_action_values)

  # Compute V(s_{t+1}) for all next states.
  # Expected values of actions for non_final_next_states are computed based
  # on the "older" target_net; selecting their best reward with max(1)[0].
  # This is merged based on the mask, such that we'll have either the expected
  # state value or 0 in case the state was final.

  next_state_values = target_net(next_state_batch).max(1)[0].detach().to(device)
  #print("next_state_values:",next_state_values)
  # Compute the expected Q values
  expected_state_action_values = (next_state_values * Q_GAMMA) + reward_batch
  #print("expected_state_action_values:",expected_state_action_values)
  # Compute Huber loss
  loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

  # Optimize the model
  q_optimizer.zero_grad()
  loss.backward()
  #for param in policy_net.parameters():
    #param.grad.data.clamp_(-1, 1)
  q_optimizer.step()

def main():
  # Training settings
  # For multiple augmentations set the flag --multiple-augmentations to true
  parser = argparse.ArgumentParser(description='Deep Q-learning RotMNIST')
  parser.add_argument('--child-batch-size', type=int, default=32, metavar='N',
            help='input batch size for training (default: 32)')
  parser.add_argument('--child-test-batch-size', type=int, default=1000, metavar='N',
            help='input batch size for testing (default: 1000)')
  parser.add_argument('--child-train-size', type=int, default=6000, metavar='N',
            help='input batch size for training (default: 6000)')
  parser.add_argument('--child-test-size', type=int, default=1000, metavar='N',
            help='input batch size for training (default: 1000)')
  parser.add_argument('--child-epochs', type=int, default=4, metavar='N',
            help='number of epochs to train (default: 4)')
  parser.add_argument('--child-lr', type=float, default=1e-3, metavar='LR',
            help='learning rate (default: 1e-3)')
  parser.add_argument('--child-gamma', type=float, default=0.7, metavar='M',
            help='Learning rate step gamma (default: 0.7)')
  parser.add_argument('--Q-GAMMA', type=float, default=0.5, metavar='M',
            help='Learning rate step gamma (default: 0.7)')
  parser.add_argument('--Q-BATCH_SIZE', type=int, default=128, metavar='M',
            help='Batch size for Q-learning (default: 256)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
  parser.add_argument('--dry-run', action='store_true', default=False,
            help='quickly check a single pass')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
            help='how many batches to wait before logging training status')
  parser.add_argument('--save-model', action='store_true', default=False,
            help='For Saving the current Model')
  parser.add_argument('--aug-array-id', type=int, default=0,
            help='augmentation index to be used from aug_array_list')
  parser.add_argument('--state-size', type=int, default=10,
            help='Size of the group array')
  parser.add_argument('--max_models', type=int, default=600,
            help='Maximum number of models to be trained')
  parser.add_argument('--max_episodes', type=int, default=100,
            help='Maximum number of episodes')
  args = parser.parse_args()
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  torch.manual_seed(args.seed)
  device = torch.device("cuda" if use_cuda else "cpu")

  train_kwargs = {'batch_size': args.child_batch_size}
  test_kwargs = {'batch_size': args.child_test_batch_size}
  if use_cuda:
    cuda_kwargs = {'num_workers': 1,
             'pin_memory': True,
             'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

  k = args.state_size

  child_network_dimensions = [28*28, 20*20, 20*20, 10]

  # Setup the Q-network and its hyperparameters
  input_size = args.state_size #10 by default
  hidden_sizes = [400, 400, 400]
  output_size = input_size
  n_actions = output_size
  memory = ReplayMemory(10000)
  #define the policy network and target network

  target_net.load_state_dict(policy_net.state_dict())
  target_net.eval()

  q_optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
  memory = ReplayMemory(10000)

  #=======================================================================Setup for Q-learning=============================================================================
  Q_BATCH_SIZE = args.Q_BATCH_SIZE
  Q_GAMMA = args.Q_GAMMA # Dependency on the future
  Q_EPS_LIST = [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.01,0.01]
  Q_EPS_INDEX = 0
  Q_EPS = Q_EPS_LIST[Q_EPS_INDEX]
  Q_MODELS_TRAINED_LIST = [50,100,150,200,250,300,350,400,450,500,550,600]
  Q_TARGET_UPDATE = 2
  env = Environment(device=device, child_network_dimensions=child_network_dimensions, child_train_size=args.child_train_size, child_test_size=args.child_test_size, child_batch_size=args.child_batch_size, child_test_batch_size=args.child_test_batch_size, child_lr=args.child_lr, child_epochs=args.child_epochs, state_size=args.state_size)
  y_accuracy = [env.base_accuracy] # Compute the rolling mean accuracy and average per epsilon accuracy from here
  x_models_trained = [env.models_trained] # Should be an enumeration from 1,...,total models to be trained.
  y_accuracy_per_step = []
  steps = 0
  x_steps = []
  steps_per_model_list = []
  average_model_accuracy = env.base_accuracy
  steps_per_model = 1 # Considering the base accuracies and steps
  num_episodes = 0

  #==========================================================================Iteration loop===============================================================================
  while env.models_trained < args.max_models and num_episodes < args.max_episodes:
    # Select and perform an action
    state = torch.tensor([env.current_state[i] for i in range(env.k)]).to(device)
    action = select_action(state, Q_EPS, device, n_actions, policy_net)
    accuracy, reward, new_models_trained, done = env.step(action) # done = True when 1 episode completes, new_models_trained = True only when a new model has been trained in the step
    reward = torch.tensor([reward], device=device)
    next_state = torch.tensor([env.next_state[i] for i in range(env.k)])
    memory.push(state, action, next_state, reward)

    steps_per_model += 1
    average_model_accuracy = (average_model_accuracy*(steps_per_model-1) + accuracy)/steps_per_model

    steps += 1
    y_accuracy_per_step.append(accuracy)
    x_steps.append(steps)
    #================================================Perform one step of the optimization (on the target network)===========================================================
    optimize_model(env.k, device, memory, q_optimizer, Q_BATCH_SIZE, Q_GAMMA)
    #=======================Check if new models are trained===========================================
    if new_models_trained:
      print("Number of models trained:", env.models_trained)
      y_accuracy.append(average_model_accuracy)
      x_models_trained.append(env.models_trained)
      steps_per_model_list.append(steps_per_model)
      average_model_accuracy = 0
      steps_per_model = 0
      if env.models_trained in Q_MODELS_TRAINED_LIST:
        Q_EPS_INDEX += 1
        Q_EPS = Q_EPS_LIST[Q_EPS_INDEX]

    if done:
      num_episodes += 1
      print("Number of episodes:", num_episodes)
      print("Current state:", env.current_state)
      print("Current epsilon value:", Q_EPS)
      plot_update(x_models_trained, y_accuracy)
      env.reset()

    # Update the target network, copying all weights and biases in DQN
    if env.models_trained % Q_TARGET_UPDATE == 0:
      target_net.load_state_dict(policy_net.state_dict())
      torch.save(target_net.state_dict(), "Target_Net_RotMNIST_DQN")

  #plot_overall(x_models_trained, y_accuracy, args.aug_array_id)

  # Save network and plot data
  np.save("y_accuracy_RotMNIST",y_accuracy)
  np.save("x_models_RotMNIST",x_models_trained)
  np.save("steps_per_model_RotMNIST",steps_per_model_list)
  np.save("model_accuracies_RotMNIST",env.visited_model_accuracies)
  np.save("model_rewards_RotMNIST",env.visited_model_rewards)
  torch.save(policy_net.state_dict(), "RotMNIST_DQN")
  plot_steps_update(x_steps, y_accuracy_per_step)
  print('Complete')


if __name__ == '__main__':
  main()