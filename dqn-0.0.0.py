# Let's play randomly

from collections import namedtuple
from itertools import count
import gym
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import random
import math
import os
import numpy as np
from PIL import Image


env = gym.make("Centipede-v0")
observation = env.reset()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# Basic neural net
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

    





"""
for _ in range(1002):
    #print(env.observation_space)
    screen = env.render(mode='rgb_array')
# From position 215 to 250, is the point board
    screen = torch.tensor(screen)
    screen = screen[:215, :, :]
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)


for _ in range(10):   
    mask = screen[:, :, :] > 0 
    screen[mask] = 255
    plt.imshow(screen)
    plt.show()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    screen = env.render(mode='rgb_array')
    print(screen.shape)

"""


# Getting the image of the game
# Process the image only two three types of color

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen():
    '''This function returns the process image for the network'''
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    #screen = torch.tensor(screen, dtype=torch.float32)
    screen = screen[:215, :, :]
    mask = screen[:, :, :] > 0
    screen[mask] = 255
    screen = torch.tensor(screen, dtype=torch.float32)
    return resize(screen).unsqueeze(0).to(device)



# Setting action selection variables
BATCH_SIZE = 128 # RMSprop
GAMMA = 0.999 # Q-algorhtm
# Greedy epsilon policy, with exponential decay
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
# How frequent backup on the other neural net
TARGET_UPDATE = 1


# Setting networks for DQN and algorithm
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# TODO: Load the model in case that it exists
# Otherwise create the model
# Check that the file exists
steps_done = 0
policy_net = DQN(screen_height, screen_width).to(device)
target_net = DQN(screen_height, screen_width).to(device)

if os.path.isfile(SAVE_FILE_NAME):
    # Load the model
    model_info = torch.load(SAVE_FILE_NAME)
    policy_net.load_state_dict(model_info['policynet_state'])
    target_net.load_state_dict(policy_net.state_dict())
    target_net.train()
    optimizer = optim.RMSprop(policy_net.parameters())
    optimizer.load_state_dict(model_info['optimizer_state'])
    memory = model_info['replaymemory']
    steps_done = model_info['steps_done']
else:
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)

    

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # Select one of the actions
        # randomly
        return torch.tensor([[random.randrange(18)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:    
        return
    # Takes random samples from the memory
    transition = memory.sample(BATCH_SIZE)

    # Takes all the random samples and put in a transition object
    # Taking each element of transition as a tuple
    batch = Transition(*zip(*transition))

    # Sees all the next_states and check if there are None. To see if 
    # an state is a final state. 
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)
    
    # Same as the last just that if a next_state is None, it just delete it.
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # Tensors containing a concatenation of each part of transition
    # on the batch
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # policy_net(state_batch) => actions => actions.gather(1, action_batch)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Create a tensor full of zeros
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # Fill this tensor with prediction of the target net
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Take this values to predict the expected rewards
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Measure the loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

## Training loop

num_episodes = 100
for i_episode in range(num_episodes):
    env = gym.make("Centipede-v0")
    observation = env.reset()
    env.reset()    
    screen = get_screen()
    state = screen
    for t in range(200):        
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = screen
        screen = get_screen()
        if not done:
            next_state = screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        print(t)
        if done:
            break

        # Load the changes from the target network to the policy network
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            # TODO: Save the model
            model_info = {
                'policynet_state': policy_net.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'replaymemory': memory,
                'steps_done': steps_done
            }
            torch.save(model_info, SAVE_FILE_NAME)

    print('Complete')
