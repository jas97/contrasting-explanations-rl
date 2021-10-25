import math

import gym
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import MinMaxScaler


class DrivingSimulator(gym.Env):

    def __init__(self, reward_weights):
        self.merged = False
        self.since_merged = 0

        self.init_distance = 0
        self.init_lane = 1
        self.init_lane_non_autonomous = 2
        self.goal_lane = 2
        self.lane_width = 10
        self.car_length = 10

        self.init_vel = 15
        self.max_velocity = 20
        self.steering_range = [-30, 30]
        self.max_timesteps = 30
        self.steps_elapsed = 0
        self.threshold = 0.05
        self.scaler = None

        self.agent_y = []

        self.reward_weights = reward_weights

        self.gamma = 1

        self.goal_scaler = MinMaxScaler(feature_range=(0, 1))
        self.goal_scaler.fit([[0], [self.lane_width*3]])
        self.distance_scaler = MinMaxScaler(feature_range=(0, 1))
        self.distance_scaler.fit([[0], [30]])
        self.angle_scaler = MinMaxScaler(feature_range=[-1, 1])
        self.angle_scaler.fit([[-30], [30]])
        self.goal_distance_scaler = MinMaxScaler(feature_range=[0, 1])
        self.goal_distance_scaler.fit([[0], [30]])
        self.steering_angle_scaler = MinMaxScaler(feature_range=[-1, 1])
        self.steering_angle_scaler.fit([[-30], [30]])
        self.velocity_scaler = MinMaxScaler(feature_range=[0, 1])
        self.velocity_scaler.fit([[0], [self.max_velocity]])
        self.theta_scaler = MinMaxScaler(feature_range=[-1, 1])
        self.theta_scaler.fit([[-180], [180]])
        self.dev_vel_scaler = MinMaxScaler(feature_range=[0, 1])
        self.dev_vel_scaler.fit([[0], [np.maximum(self.init_vel, self.max_velocity - self.init_vel)]])

        self.observation_space = gym.spaces.Box(low=np.array([0, 0, -1, 0, -1, 0, 0, -1, 0, -1]),
                                                high=np.array([self.lane_width*3, 1000, 1, self.max_velocity, 1, self.lane_width*3, 1000, 1, self.max_velocity, 1]),
                                                shape=(10, ))

        self.action_space = gym.spaces.Discrete(5) # discretized action space

    def step(self, action):
        # angle, acc = action
        # angle = self.angle_scaler.inverse_transform([[angle]]).item()
        angle, acc = 0, 1
        if action == 0:
            acc = 1.1 # increase speed 10%
        elif action == 1:
            acc = 0.9
        elif action == 2:
            angle = +3
        elif action == 3:
            angle = -3
        elif action == 4:  # do nothing
            angle = 0
            acc = 1

        self.agent_state = self.state[0:5]
        self.nonautonomous_state = self.state[5:]
        self.agent_state, valid = self.update_car_state(self.agent_state, angle, acc)
        self.nonautonomous_state, _ = self.update_car_state(self.nonautonomous_state, 0, 1)

        self.agent_y.append(self.agent_state[1])

        self.state = self.agent_state + self.nonautonomous_state
        self.state = np.array(self.state)

        car_distance = self.get_distance()
        goal_distance = self.get_goal_distance()
        dev_from_init_vel = self.dev_from_init_vel()
        turn_reward = self.turning()
        acc_reward = self.get_acc_reward()
        progress_reward = self.progress()

        reward = - self.reward_weights['car_distance'] * car_distance + \
                 - self.reward_weights['goal_distance'] * goal_distance +\
                 - self.reward_weights['dev_from_init_vel'] * dev_from_init_vel + \
                 - self.reward_weights['turn'] * turn_reward + \
                 - self.reward_weights['acc'] * acc_reward +\
                 - self.reward_weights['progress'] * progress_reward

        self.merged = self.merged or (goal_distance < self.threshold)
        done = (self.steps_elapsed >= self.max_timesteps) or (car_distance > (1 - self.threshold))

        if not valid:
            # print('Left the road')
            reward = -10

        if self.merged:
            # done = (self.steps_elapsed >= 20) and self.merged
            if self.since_merged == 0:
                # done = True
                reward = 1000
                # print('Merged successfully at step: {} with velocity: {} and distance to other car: {} y: {} y_non_autonomous: {}'.format(self.steps_elapsed,
                #                                                                                                                           self.state[3],
                #                                                                                                                           car_distance,
                #                                                                                                                           self.state[1],
                #                                                                                                                           self.state[6]))

            self.since_merged += 1
            # if self.since_merged >= 1:  # TODO: fixed episode length
            #     print('Driven successfully after merging. X: {}, Speed: {}, Heading: {} '.format(self.state[0], self.state[3], self.state[2]))
            #     done = True
            #     reward = +1

        if car_distance > (1 - self.threshold):
            reward = -1000
            done = True

        self.steps_elapsed += 1

        return self.state, reward, done, {}

    def update_car_state(self, state, angle, acc):
        x, y, theta, vel, alpha = state

        alpha = self.steering_angle_scaler.inverse_transform([[alpha]]).item()
        theta = self.theta_scaler.inverse_transform([[theta]]).item()

        vel = vel * acc

        if vel > self.max_velocity:
            vel = self.max_velocity
        if vel < 0:
            vel = 0

        alpha += angle

        if alpha > 30:
            alpha = 30
        if alpha < -30:
            alpha = -30

        theta += (vel/self.car_length) * np.tan(math.radians(alpha))

        if theta > 180:
            theta = -180 + (theta % 180)
        if theta < -180:
            theta = 180 - (theta % 180)

        x += vel * np.sin(math.radians(theta))
        y += vel * np.cos(math.radians(theta))

        valid = True
        if x < 0:
            valid = False
            x = 0
        elif x > self.lane_width * 3:
            valid = False
            x = self.lane_width * 3

        theta = self.theta_scaler.transform([[theta]]).item()
        alpha = self.steering_angle_scaler.transform([[alpha]]).item()

        return [x, y, theta, vel, alpha], valid

    def get_distance(self):
        agent_pos = self.agent_state[0:2]
        nonautonomous_pos = self.nonautonomous_state[0:2]

        closeness = self.kernel(agent_pos, nonautonomous_pos)
        # closeness = self.distance_scaler.transform([[closeness]]).item()

        return closeness  # return closeness

    def get_goal_distance(self):
        goal_x_pos = (self.goal_lane + 0.5) * self.lane_width

        dist = abs(goal_x_pos - self.state[0])
        rew = np.maximum(0, dist)
        rew = self.goal_distance_scaler.transform([[rew]]).item()

        return rew

    def get_acc_reward(self):
        # acc = abs(self.velocities[-1] - self.velocities[-2])
        #
        # acc = self.velocity_scaler.transform([[acc]]).item()
        #
        # return acc
        return 0

    def dev_from_init_vel(self):
        dev = abs(self.state[3] - self.init_vel)
        dev = self.dev_vel_scaler.transform([[dev]]).item()

        return dev

    def turning(self):
        init_theta = 0
        # turn = np.mean(np.abs(np.subtract(self.thetas, init_theta)))
        turn = abs(init_theta - self.state[2])
        return turn

    def progress(self):
        last_pos = self.agent_y[-2]
        curr_pos = self.state[1]

        progress = abs(last_pos - curr_pos)
        scaler = MinMaxScaler(feature_range=[0, 1])
        scaler.fit([[0], [self.max_velocity]])

        progress = scaler.transform([[progress]]).item()
        return 1 - progress

    def reset(self):
        self.steps_elapsed = 0
        self.merged = False
        self.since_merged = 0

        init_pos = self.lane_width * self.init_lane + self.lane_width/2 + np.random.normal(loc=0, scale=1)
        init_pos_nonaut = self.lane_width * self.init_lane_non_autonomous + self.lane_width/2 + np.random.normal(loc=0, scale=1)

        if init_pos > 3 * self.lane_width:
            init_pos = 3 * self.lane_width
        if init_pos_nonaut > 3 * self.lane_width:
            init_pos_nonaut = 3 * self.lane_width

        y_pos = 0 # np.random.uniform(0, 2)

        self.agent_state = [init_pos, y_pos, 0, self.init_vel, 0]
        self.nonautonomous_state = [init_pos_nonaut, y_pos, 0, self.init_vel, 0]

        self.agent_y.append(self.agent_state[1])

        self.state = self.agent_state + self.nonautonomous_state

        return np.array(self.state)

    def render(self):
        print('Step: {} state = {}'.format(self.steps_elapsed, self.state))

    def close(self):
        pass

    def kernel(self, a, b):
        kern = RBF(15.0)
        dist = kern([a], [b])

        return dist.item()

    def get_state(self):
        return self.state

    def get_obs(self):
        return self.state

    def update_state(self, state):
        self.state = state
        self.agent_state = state[:5]
        self.nonautonomous_state = state[5: ]

    def get_steps_elapsed(self):
        return self.steps_elapsed

    def set_steps_elapsed(self, steps):
        self.steps_elapsed = steps