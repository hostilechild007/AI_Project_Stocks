import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []  # keep track of states encountered
        self.probs = []  # log probabilities
        self.vals = []  # values that our critic calcs
        self.actions = []  # actions we took
        self.rewards = []  # rewards we received
        self.dones = []  # terminal flags

        self.batch_size = batch_size

    # for stochastic gradient ascent
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)  # indices that show where to start sampling random mini batches
        indices = np.arange(n_states, dtype=np.int64)  # get indices of our memory (interval: [0, n_states))
        np.random.shuffle(indices)  # shuffle for stochastic gradient ascent's random mini-batches

        # takes batch_size chunks of shuffled memory
        batches = [indices[i: i + self.batch_size] for i in batch_start]  # looks like a 2D matrix
        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.vals), \
               np.array(self.rewards), np.array(self.dones), batches

    def store_memory(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.probs.append(prob)
        self.vals.append(val)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    # clears memory after every trajectory
    def clear_memory(self):
        self.states = [] 
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []


