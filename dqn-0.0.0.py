# Let's play randomly

from collections import namedtuple
import gym
import time
import matplotlib.pyplot as plt
env = gym.make("Centipede-v0")
observation = env.reset()

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


for _ in range(1000):
    screen = env.render()
    time.sleep(0.05)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)


