import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

T.random.manual_seed(0)


class ActorNetwork(nn.Module):
    # n_actions: number of actions
    # alpha: learning rate
    # fc1_dims: fully connected dims for 1st layer
    # fc2_dims: ... for 2nd layer
    # chkpt_dir: checkpoint directory
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, fc3_dims=256, chkpt_dir=''):
        super(ActorNetwork, self).__init__()  # calls super class's init method

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')

        # Deep Neural Network that approx agent's policy which is a prob dist over set of actions for each state
        # all stuff in Sequential are fully connected layers
        # input layer -> activation f(x) -> hidden layer -> activation f(x) -> hidden layer -> activation f(x) -> last layer -> activation f(x)
        # activation f(x) w/ non-linear transformations so network can learn better & perform more complex tasks
        self.actor = nn.Sequential(nn.Linear(*input_dims, fc1_dims),  # *: to unpack list into separate elements; 1st layer; nn.Linear(in_features, out_features)
                                   nn.LeakyReLU(),  # f(x) = max(0, x); f(x)=0 for neg x and f(x)=x for pos x; default act. f(x)
                                   nn.Linear(fc1_dims, fc2_dims),  # y = Wx + b; initializes random b w/ matrix out_features * 1 then W matrix out_features * in_features w/ random elements
                                   nn.LeakyReLU(),
                                   nn.Linear(fc2_dims, fc3_dims),
                                   nn.LeakyReLU(),
                                   nn.Linear(fc3_dims, n_actions),  # to find y, fc=nn.Linear(...) ==> y = fc(x)
                                   nn.Softmax(dim=-1)  # squash inputs to probability outputs [0, 1]; used for last layer in multi class classification problems
                                   )

        # stochastic gradient descent optimization
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)  # lr: learning rate
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')  # check if can use cuda GPU
        self.to(self.device)  # sends network to device

    # takes a single state or batch of states to
    def forward(self, state):
        # actor: given current state of the environment, determine actions to take based on probs
        distribution = self.actor(state)  # get distribution of actions
        # print("actor: ", distribution)
        distribution = Categorical(distribution)  # use categorical dist cuz dealing w/ discrete action spaces

        return distribution

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)  # self.state_dict(): the network

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

