from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import pandas as pd
import yfinance as yf
import random
from gym.utils import seeding

# shares normalization factor
# 100 shares per trade
HMAX_NORMALIZE = 1
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE = 20
# initial amount of shares we have in our account
INITIAL_SHARES = 0
# types of stocks we have
STOCK_DIM = 1
# rewards will be too high so lower it
REWARD_SCALING = 1
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001
# quarterly days to determine when account should reset
days_per_month = 365//4


class StockMarketEnv(Env):
    def __init__(self, df):
        self.stock_history_table = df

        # action space: {-k,…,-1, 0, 1, …, k} where k & -k rep the # of shares we can buy and sell, and k ≤ h_max
        # self.action_space = Box(low=-1, high=1, shape=(STOCK_DIM,))  # have it as continuous distribution
        self.action_space = Discrete(2*HMAX_NORMALIZE+1)  # later... {-k,…,-1, 0, 1, …, k} where k is an (int) action
        # for _ in range(6):
        #     print(self.action_space.sample())

        # self.state needs to equal in size and rep of respective index
        high_state_values = np.array([np.finfo(np.float32).max,  # Account Balance
                                      np.iinfo(np.int32).max,    # Shares holding
                                      np.finfo(np.float32).max,  # HLC Avg = high, low, close average
                                      ])
        low_state_values = np.array([0,  # Account Balance
                                     0,  # Shares holding
                                     0   # HLC Avg
                                     ])
        self.observation_space = Box(low_state_values, high_state_values)
        self.observation_space = Box(low=0, high=np.inf, shape=(3,))

        self.day = 0
        self.total_days = len(self.stock_history_table.index)
        stock_row_data = self.stock_history_table.iloc[self.day, :]
        hlc_avg = (stock_row_data.loc["High"] + stock_row_data.loc["Low"] + stock_row_data.loc["Close"]) / 3
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                     [INITIAL_SHARES] + \
                     [hlc_avg]

        self.reward = 0
        self.terminal = False

        self.__seed()

    # action < 0
    def __sell_stocks(self, action):
        if self.state[1] > 0:  # if account has shares to sell

            # take min b/c the shares NN want to sell may be > the amount of shares we have
            self.state[0] += self.state[2] * min(abs(action), self.state[1]) * (1 - TRANSACTION_FEE_PERCENT)
            self.state[1] -= min(abs(action), self.state[1])

    # action > 0
    def __buy_stocks(self, action):
        if self.state[0] > 0:  # if account has money to buy
            shares_able_to_buy = self.state[0] // self.state[2]
            self.state[0] -= self.state[2] * min(action, shares_able_to_buy) * (1 + TRANSACTION_FEE_PERCENT)
            self.state[1] += min(action, shares_able_to_buy)

    def step(self, action):
        self.day += 1

        self.terminal = bool(self.day >= self.total_days or
                             self.day % days_per_month == 0 or
                             (self.state[0] == 0 and self.state[1] == 0)
                             )

        if self.terminal:
            if self.day >= self.total_days:
                self.day = 0
            return np.array(self.state), self.reward, self.terminal, {}

        else:
            action -= HMAX_NORMALIZE  # {-k,…,-1, 0, 1, …, k} where k is an (int) action
            print("action: ", action)

            # = balance + shares * stock price
            begin_total_asset = self.state[0] + self.state[1] * self.state[2]

            if action < 0:
                self.__sell_stocks(action)
            elif action > 0:
                self.__buy_stocks(action)

            # self.day += 1
            print("self.day: ", self.day)
            print("self.total_days: ", self.total_days)
            stock_row_data = self.stock_history_table.iloc[self.day, :]
            hlc_avg = (stock_row_data.loc["High"] + stock_row_data.loc["Low"] + stock_row_data.loc["Close"]) / 3
            self.state = [self.state[0]] + \
                         [self.state[1]] + \
                         [hlc_avg]

            end_total_asset = self.state[0] + self.state[1] * self.state[2]
            self.reward = end_total_asset - begin_total_asset
            self.reward *= REWARD_SCALING

        return np.array(self.state), self.reward, self.terminal, {}

    def render(self, mode='human'):
        pass

    def reset(self):
        self.reward = 0
        self.terminal = False
        if self.day >= self.total_days:
            self.day = 0
        self.day += 1
        stock_row_data = self.stock_history_table.iloc[self.day, :]
        hlc_avg = (stock_row_data.loc["High"] + stock_row_data.loc["Low"] + stock_row_data.loc["Close"]) / 3
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                     [INITIAL_SHARES] + \
                     [hlc_avg]

        return self.state

    def __seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
