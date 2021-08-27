import numpy as np
import gym
from gym import spaces
from sklearn.preprocessing import MinMaxScaler


class EnvInventory(gym.Env):
    def __init__(self, stock_penalty, shipment_penalty, min_inv=0, max_inv=100):
        ''' Intitilize patient parameters '''
        self.stock_penalty = stock_penalty
        self.shipment_penalty = shipment_penalty
        self.stockout_penalty = -10
        self.min_inv = min_inv
        self.max_inv = max_inv

        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]))
        self.lows = [-1, -1]
        self.highs = [1, 1]
        self.observation_space = spaces.Box(low=np.array(self.lows), high=np.array(self.highs), shape=(2, ))

        self.timeout_steps = 10

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit([[0], [max_inv]])

    def reset(self):
        inventory = np.random.randint(self.min_inv, self.max_inv)
        demand = np.random.randint(self.min_inv, self.max_inv)

        # normalize
        inventory = self.scaler.transform([[inventory]]).item()
        demand = self.scaler.transform([[demand]]).item()

        self.state = np.array([inventory, demand])
        self.steps_elapsed = 0

        return self.state

    def is_done(self):
        return self.steps_elapsed >= self.timeout_steps

    def step(self, action):
        if not np.isscalar(action):
            action = action.item()

        # inverse transform
        action = self.scaler.inverse_transform([[action]])
        action = int(action)

        inv, demand = self.state
        # inverse
        inv = self.scaler.inverse_transform([[inv]]).item()
        demand = self.scaler.inverse_transform([[demand]]).item()

        stockout = demand - inv if demand > inv else 0

        new_inventory = action if stockout > 0 else inv - demand + action

        if new_inventory > self.max_inv:
            new_inventory = self.max_inv

        new_demand = np.random.randint(0, 100)

        reward = self.stockout_penalty * stockout - (action > 0) * self.shipment_penalty - action - new_inventory * self.stock_penalty

        # normalize
        new_inventory = self.scaler.transform([[new_inventory]]).item()
        new_demand = self.scaler.transform([[new_demand]]).item()

        new_state = np.array([new_inventory, new_demand])
        self.state = new_state

        self.steps_elapsed += 1
        done = self.is_done()

        return new_state, float(reward), done, {}

    def render(self):
        pass

    def close(self):
        pass

    def get_obs(self):
        return self.state

    def get_state(self):
        return self.state

    def update_state(self, state):
        self.state = state

    def get_steps_elapsed(self):
        return self.steps_elapsed

    def set_steps_elapsed(self, steps):
        self.steps_elapsed = steps

