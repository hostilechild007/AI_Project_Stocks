import gym  # toolkit for developing and comparing reinforcement learning algorithms
import numpy as np
from AI_project.Agent import Agent
import matplotlib.pyplot as plt


if __name__ == '__main__':
    env = gym.make('CartPole-v0')  # create cartpole environment & can use to interact w/ envir & simulate random shit
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003  # learning rate
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape)
    n_games = 300  # episodes so have starting and terminal state so can end

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0  # used to know when it's time to perform learning f(x)

    for episode in range(n_games):  # for every episode....
        observation = env.reset()  # reset environment state and returns initial state
        done = False  # reset terminal flag
        score = 0  # reset score

        while not done:
            # select action accord to actor network
            action, prob, val = agent.choose_action(observation)

            # apply action to environment, returns next state, reward, & whether or not episode reached terminal state
            observation_, reward, done, info = env.step(action)
            n_steps += 1  # every time we take an action...
            score += reward
            agent.remember(observation, action, prob, val, reward, done)

            # time to perform learning f(x)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1

            observation = observation_  # current state = new state of the environment

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])  # calc mean of previous 100 games

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', episode, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)

    # plot shit
    x = [i+1 for i in range(len(score_history))]  # x-axis for plot
    running_avg = np.zeros(len(score_history))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(score_history[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.show()
