import gym  # toolkit for developing and comparing reinforcement learning algorithms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pyfde

from AI_project.Agent import Agent
from AI_project.StockMarketEnv import StockMarketEnv


# 0th index gamma: (.8, .995)
# 1st index alpha (learning rate): (1e-5, 1e-3)
# 2nd index gae_lambda: (0.9, 0.95)
# 3rd index policy_clip (epsilon): (.1, .3)
# 4th index num_epoch: (3, 10) *int
# 5th index batch_size: (32, 512) *int
# 6th index buffer_size (N experiences to be collected b4 gradient descent): (2048, 409600) *int
# 7th index hidden_units1: (32, 512) *int
# 8th index hidden_units2: (32, 512) *int
# 9th index hidden_units3: (32, 512) *int


def tune_ppo_parameters(guesses):
    print("guesses: ", guesses)
    gamma = guesses[0]
    alpha = guesses[1]
    gae_lambda = guesses[2]
    policy_clip = guesses[3]
    num_epoch = guesses[4]
    batch_size = guesses[5]
    buffer_size = guesses[6]
    hidden_units1 = guesses[7]
    hidden_units2 = guesses[8]
    hidden_units3 = guesses[9]

    env = StockMarketEnv(stock_history_table)

    # buffer_size = 20
    # batch_size = 5
    # buffer_size should be multiple of batch_size
    scale_multiple = buffer_size / batch_size
    scale_multiple = round(scale_multiple)
    buffer_size = round(batch_size) * scale_multiple

    # n_epochs = 4
    # alpha = 0.0003  # learning rate

    agent = Agent(n_actions=env.action_space.n, input_dims=env.observation_space.shape, gamma=gamma, alpha=alpha,
                  gae_lambda=gae_lambda, policy_clip=policy_clip, batch_size=round(batch_size),
                  n_epochs=round(num_epoch), fc1_dims=round(hidden_units1), fc2_dims=round(hidden_units2), fc3_dims=round(hidden_units3)
                  )
    n_games = 200  # episodes so have starting and terminal state so can end
    print("start observation space: ", env.observation_space)

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0  # used to know when it's time to perform learning f(x)

    start_profit = 100
    end_profit = 0
    score = 0

    for episode in range(n_games):  # for every episode....
        observation = env.reset()  # reset environment state and returns initial state
        print("starting observation space: ", observation)

        done = False  # reset terminal flag
        score = 0  # reset score

        while not done:
            # select action accord to actor network
            env.render()
            print("observation: ", observation)
            action, prob, val = agent.choose_action(observation)

            # apply action to environment, returns next state, reward, & whether or not episode reached terminal state
            observation_, reward, done, info = env.step(action)
            print("reward: ", reward)
            # print("observation_: ", observation_)
            n_steps += 1  # every time we take an action...
            score += reward
            agent.remember(observation, action, prob, val, reward, done)  # acts as storing batches in a replay buffer
            print("agent remember len: ", len(agent.memory.rewards))
            # time to perform learning f(x)
            if n_steps % buffer_size == 0:
                print("learning")
                agent.learn()
                learn_iters += 1

            observation = observation_  # current state = new state of the environment

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])  # calc mean of previous 100 games (moving average)

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', episode, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)

        end_profit = observation[0] + observation[1] * observation[2]

    with open("hyperparameter_tuning_results.txt", "w") as txt_file:
        txt_file.writelines("gamma: " + str(gamma) + "\n")
        txt_file.writelines("alpha: " + str(alpha) + "\n")
        txt_file.writelines("gae_lambda: " + str(gae_lambda) + "\n")
        txt_file.writelines("policy_clip: " + str(policy_clip) + "\n")
        txt_file.writelines("num_epoch: " + str(round(num_epoch)) + "\n")
        txt_file.writelines("batch_size: " + str(round(batch_size)) + "\n")
        txt_file.writelines("buffer_size: " + str(round(buffer_size)) + "\n")
        txt_file.writelines("hidden_units1: " + str(round(hidden_units1)) + "\n")
        txt_file.writelines("hidden_units2: " + str(round(hidden_units2)) + "\n")
        txt_file.writelines("hidden_units3: " + str(round(hidden_units3)) + "\n")
        txt_file.writelines("recent score: " + str(score) + " avg score of last 100 steps: " + str(avg_score) + "\n")
        txt_file.writelines("total profits: " + str(end_profit - start_profit) + "\n")

    return end_profit - start_profit   # try to max this out


if __name__ == '__main__':
    stock_history_table = yf.Ticker("AAPL").history(period="max")
    stock_history_table.reset_index(inplace=True)
    stock_history_table.drop("Date", axis=1, inplace=True)
    print(stock_history_table.head())
    print("rows ", len(stock_history_table.index))
    print(np.subtract(range(0, 201), 100))
    # print(stock_history_table.loc[:, "Open"])
    # stock_history_table = pd.read_csv('HDB.csv')

    # gamma=.99, alpha=.003, gae_lambda=.95, policy_clip=.2, batch_size=64, n=2048, n_epochs=10
    # get which variables need to be held constant or vary
    # 0th index gamma: (.8, .995)
    # 1st index alpha (learning rate): (1e-5, 1e-3)
    # 2nd index gae_lambda: (0.9, 0.95)
    # 3rd index policy_clip (epsilon): (.1, .3)
    # 4th index num_epoch: (3, 10)
    # 5th index batch_size: (32, 512)
    # 6th index buffer_size (N experiences to be collected b4 gradient descent): (2048, 409600)
    # 7th index hidden_units1: (32, 512)
    # 8th index hidden_units2: (32, 512)
    # 9th index hidden_units3: (32, 512)
    # if variables are held constant, make min and max the same number: (same, same)
    bounds = [(.8, .995), (1e-5, 1e-3), (0.9, 0.95), (.1, .3), (3, 10), (32, 32), (2048, 2048), #, (32, 512), (2048, 409600),
              (32, 512), (32, 512), (32, 512)
              ]
    result = pyfde.JADE(tune_ppo_parameters, limits=bounds, n_dim=10, n_pop=25)  # default is maximize
    best, fit = result.run(n_it=130)
    print(best)
    print(fit)

    """
    env = StockMarketEnv(stock_history_table)

    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003  # learning rate
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape)
    n_games = 200  # episodes so have starting and terminal state so can end
    print("start observation space: ", env.observation_space)

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0  # used to know when it's time to perform learning f(x)

    for episode in range(n_games):  # for every episode....
        observation = env.reset()  # reset environment state and returns initial state
        print("starting observation space: ", observation)

        done = False  # reset terminal flag
        score = 0  # reset score

        while not done:
            # select action accord to actor network
            env.render()
            print("observation: ", observation)
            action, prob, val = agent.choose_action(observation)

            # apply action to environment, returns next state, reward, & whether or not episode reached terminal state
            observation_, reward, done, info = env.step(action)
            print("reward: ", reward)
            # print("observation_: ", observation_)
            n_steps += 1  # every time we take an action...
            score += reward
            agent.remember(observation, action, prob, val, reward, done)  # acts as storing batches in a replay buffer

            # time to perform learning f(x)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1

            observation = observation_  # current state = new state of the environment

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])  # calc mean of previous 100 games (moving average)

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', episode, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)

    # plot shit
    print("FINISHED!!!!!")

    x = [i+1 for i in range(len(score_history))]  # x-axis for plot
    running_avg = np.zeros(len(score_history))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(score_history[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.show()
    """
