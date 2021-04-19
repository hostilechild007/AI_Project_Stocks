import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from AI_project.Actor_Network import ActorNetwork
from AI_project.Critic_Network import CriticNetwork
from AI_project.PPO_Memory import PPOMemory


class Agent:
    # gamma: discount factor for Discounted Sum of Rewards in our Advantages
    # alpha: learning rate
    # n: num of steps before we update
    # gae_lambda: Generalized Advantage Estimation Lambda as smoothing parameter in Advantage Estimate
    def __init__(self, n_actions, input_dims, gamma=.99, alpha=.003, gae_lambda=.95, policy_clip=.2, batch_size=64,
                 n_epochs=10, fc1_dims=256, fc2_dims=256, fc3_dims=256):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs

        self.actor = ActorNetwork(n_actions, input_dims, alpha, fc1_dims, fc2_dims, fc3_dims)
        self.critic = CriticNetwork(input_dims, alpha, fc1_dims, fc2_dims, fc3_dims)
        self.memory = PPOMemory(batch_size)

    # interface between agent and its memory
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    # interface between agent and save checkpoint f(x)s for DNN
    def save_models(self):
        print("...saving models...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print("...loading models...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        # convert numpy array to Torch Tensor (multi-dimensional matrix containing elements of a single data type)
        # Note: any .tensor(np_array).to(device) converts np to torch in CPU or GPU device
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        print("state in choose_action(): ", state)
        dist = self.actor(state)  # gets prob distribution for choosing an action
        print("distributions: ", dist.probs)
        value = self.critic(state)
        action = dist.sample()  # select an action by sampling distribution

        # get rid of dimensions of size 1; example: if input shape Dim = A*1*B*C*1 ==> squeeze(Dim) = A*B*C
        # and loss works best if a scalar
        # do log_prob() for loss f(x) later
        probs = T.squeeze(dist.log_prob(action)).item()  # .item() gives an int; moves data back to CPU
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            print("epoch", _)
            states_arr, actions_arr, old_probs_arr, vals_arr, rewards_arr, dones_arr, batches = self.memory.generate_batches()

            advantage = np.zeros(len(rewards_arr), dtype=np.float32)  # stores gae's
            for t in range(len(rewards_arr) - 1):
                # print("t: ", t)
                discount = 1
                a_t = 0  # the Advantage

                for k in range(t, len(rewards_arr) - 1):
                    # print("k: ", k)
                    # mask = (1 - int(dones_arr[k])) cuz val of term state = 0 and no returns/rewards in terminal state
                    # otherwise, mask = 1 cuz not terminal state
                    delta = rewards_arr[k] + self.gamma * vals_arr[k+1] * (1 - int(dones_arr[k])) - vals_arr[k]
                    a_t += discount * delta
                    discount *= self.gamma * self.gae_lambda

                advantage[t] = a_t

            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(vals_arr).to(self.actor.device)
            # note: batch = an array with random indices
            for batch in batches:
                states = T.tensor(states_arr[batch], dtype=T.float).to(self.actor.device)
                actions = T.tensor(actions_arr[batch]).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)

                # now need pi_theta_new => take states and pass them in actor network => get new distribution to calc new probs
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)  # actor cost f(x) for new action; take log prob cuz no underflow err
                prob_ratio = new_probs.exp() / old_probs.exp()  # old_probs.exp() = e^(old_probs) cuz put in log space

                " Clipped Surrogate Objective "
                # L ^ CLIP (theta)
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # L ^ VF (theta) aka critic loss
                returns = advantage[batch] + values[batch]  # gae + V(s)
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                # don't need to add entropy cuz have 2 separate NN for actor & critic
                total_loss = actor_loss + .5*critic_loss

                # Note: optimizer is connected to .backward() cuz of the input self.parameters()
                # clears x.grad for every parameter x in the optimizer
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                # important to call this before loss.backward(),
                # otherwise you’ll accumulate the gradients from multiple passes

                # update gradients by backprop the calculated loss f(x)
                total_loss.backward()  # computes d/dx(total_loss) for every parameter x which has requires_grad=True

                # updates the value of x using the gradient x.grad (gradient descent)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

                # gradient descent combined with backprop so can make efficient updates

        self.memory.clear_memory()
