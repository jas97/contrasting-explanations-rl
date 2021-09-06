import copy

import numpy as np
import gym
from gym import spaces


class EnvCancer(gym.Env):
    def __init__(self, penalty, timeout_steps=30, transition_noise=0.0):
        ''' Intitilize patient parameters '''
        self.kde = 0.24
        self.lambda_p = 0.121
        self.k_qpp = 0.0031
        self.k_pq = 0.0295
        self.gamma = 0.729
        self.delta_qp = 0.00867
        self.k = 100
        self.dose_penalty = penalty  # TODO: make configurable
        self.timeout_steps = timeout_steps
        self.state = None
        self.steps_elapsed = 0
        self.transition_noise = transition_noise

        self.scaler = None # action and state scaler

        self.action_space = spaces.Discrete(2)  # only two actions 0/1
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([10, 10, 70, 70]), shape=(4, ))

    def reset(self):
        C = 1  # start with concentration 1
        P = 7.13 + np.random.normal(0, 4)  # experimental mean + random normal noise

        Q = 41.2 + np.random.normal(0, 10)
        Q_p = 0

        self.state = np.array([C, P, Q, Q_p])

        for i, f in enumerate(self.state):
            if self.state[i] < 0:
                self.state[i] = 0

        self.steps_elapsed = 0
        return self.state

    def is_done(self):
        return self.steps_elapsed >= self.timeout_steps

    def step(self, action):
        C, P, Q, Q_p = self.state
        P_star = P + Q + Q_p

        C_new = C
        if action == 1:
            C_new += 1

        C_new = C_new - self.kde * C_new
        P_new = (P + self.lambda_p * P * (1-P_star/self.k) + self.k_qpp * Q_p
                 - self.k_pq * P - self.gamma * C_new * self.kde * P)
        Q_new = Q + self.k_pq * P - self.gamma * C_new * self.kde * Q
        Q_p_new = (Q_p + self.gamma * C_new * self.kde * Q - self.k_qpp * Q_p
                   - self.delta_qp * Q_p)

        next_state = np.array([C_new, P_new, Q_new, Q_p_new])

        noise = 1 + self.transition_noise * np.random.randn(4)
        next_state *= noise

        for i, f in enumerate(next_state):
            if next_state[i] < 0:
                next_state[i] = 0

        self.state = next_state
        P_star_new = np.sum(self.state[1:])

        # reward = self.dose_penalty[0] * (P_star - P_star_new) - self.dose_penalty[1] * C_new
        reward = -self.dose_penalty[0] * P_star_new - self.dose_penalty[1] * C_new
        info = {}

        self.steps_elapsed += 1

        done = self.is_done()

        return next_state, reward, done, info

    def render(self):
        pass

    def close(self):
        pass

    def get_obs(self):
        return copy.deepcopy(self.state)

    def get_state(self):
        return copy.deepcopy(self.state)

    def update_state(self, state):
        self.state = copy.deepcopy(state)

    def get_steps_elapsed(self):
        return self.steps_elapsed

    def set_steps_elapsed(self, steps):
        self.steps_elapsed = steps