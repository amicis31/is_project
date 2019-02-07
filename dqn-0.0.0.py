# Let's play randomly

from collections import namedtuple
import gym
import time
import matplotlib.pyplot as plt
import torch
from torch import nn
import random
import torch.nn.functional as F
import os
from PIL import Image
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("Centipede-v0")
observation = env.reset()

# Save file name
SAVE_FILE_NAME = 'dqn-model'

# Making replay memory structure
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
            self.capacity = capacity
            self.memory = []
            self.position = 0

    def push(self, *args):
        ''' Saves a transition.'''
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Define the DQN network
class DQN(nn.Module):
    def __init__(self, h, w):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        linear_input_size = convw * convh * 32

        self.head = nn.Linear(linear_input_size, 18)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen():
    '''This function returns the process image for the network'''
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = torch.tensor(screen, dtype=torch.float32)
    screen = screen[:215, :, :]
    mask = screen[:, :, :] > 0
    screen[mask] = 255
    return resize(screen)unsqueeze(0).to(device)

# check if model exists
if os.path.isfile(SAVE_FILE_NAME):
    # Get initial screen and size
    init_screen = get_screen()
    _, _, screen_height, screen_width = init_screen.shape

    # Initialize the network
    net = DQN(screen_height, screen_width).to(device)
    
    # Load data from training
    model_info = torch.load(SAVE_FILE_NAME, map_location='cpu')
    net.load_state_dict(model_info['policynet_state'])    
    
    # Use the network to play
    for _ in range(1000):        
        screen = get_screen()
        env.render()
        time.sleep(0.05)
        action = net(screen).max(1)[1].view(1, 1)
        observation, reward, done, info = env.step(action)


