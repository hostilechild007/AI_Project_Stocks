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
    def __init__(self, n_actions, input_dims, gamma=.99, alpha=.003, gae_lambda=.95, policy_clip=.2, batch_size=64, n=2048, n_epochs=10):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
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
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist = self.actor(state)  # gives distribution for choosing an action
        value = self.critic(state)
        action = dist.sample()

        # get rid of dimensions of size 1; example: if input shape Dim = A*1*B*C*1 ==> squeeze(Dim) = A*B*C
        probs = T.squeeze(dist.log_prob(action)).item()  # .item() gives an int
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            states_arr, actions_arr, old_probs_arr, vals_arr, rewards_arr, dones_arr, batches = self.memory.generate_batches()

            advantage = np.zeros(len(rewards_arr), dtype=np.float32)

            for t in range(len(rewards_arr) - 1):
                discount = 1
                a_t = 0  # the Advantage

                for k in range(t, len(rewards_arr) - 1):
                    # (1 - int(dones_arr[k])) added for convention
                    delta = rewards_arr[k] + self.gamma * vals_arr[k+1] * (1 - int(dones_arr[k])) - vals_arr[k]
                    a_t += discount * delta
                    discount *= self.gamma * self.gae_lambda

                advantage[t] = a_t

            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(vals_arr).to(self.actor.device)
            for batch in batches:
                states = T.tensor(states_arr[batch], dtype=T.float).to(self.actor.device)
                actions = T.tensor(actions_arr[batch]).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)

                # now need pi_theta_new => take states and pass them in actor network => get new distribution to calc new probs
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()  # old_probs.exp() = e^(old_probs)

                " Clipped Surrogate Objective "
                # L ^ CLIP (theta)
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # L ^ VF (theta) aka critic loss
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + .5*critic_loss  # cuz gradient ascent

                # clears x.grad for every parameter x in the optimizer
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                # important to call this before loss.backward(),
                # otherwise you’ll accumulate the gradients from multiple passes

                total_loss.backward()  # computes d/dx(total_loss) for every parameter x which has requires_grad=True

                # updates the value of x using the gradient x.grad
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
