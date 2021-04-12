import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class CriticNetwork(nn.Module):
    # don't need actions b/c it outputs the value of a particular state
    # doesn't care how many actions there are in action space
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir=''):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')

        # approx value f(x); telling actor how good each action is based on whether or not resulting state is valuable
        self.critic = nn.Sequential(nn.Linear(*input_dims, fc1_dims),  # *: to unpack list into separate elements; 1st layer
                                    nn.LeakyReLU(),  # f(x) = max(0, x); f(x)=0 for neg x and f(x)=x for pos x; default act. f(x)
                                    nn.Linear(fc1_dims, fc2_dims),  # y = Wx + b; initializes random b w/ matrix out_features * 1 then W matrix out_features * in_features w/ random elements
                                    nn.LeakyReLU(),
                                    nn.Linear(fc2_dims, 1)  # to find y, fc=nn.Linear(...) ==> y = fc(x))
                                    )

        # stochastic gradient descent optimization
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)  # lr: learning rate
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')  # check if can use cuda GPU
        self.to(self.device)  # sends network to device

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)  # self.state_dict(): the network

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
