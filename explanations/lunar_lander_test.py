import gym
from stable_baselines3 import DQN

from explanations.envs.lunar_lander.lunar_lander import LunarLander


def lunar_lander_test():
    env = LunarLander()

    model = DQN.load('models/lunar-lander/experiments/model_dqn_0_0.3')


    done = False
    obs = env.reset()
    while not done:
        action, _ = model.predict(obs)
        obs, rew, done, _ = env.step(action)
        env.render()
        print('Reward:{}'.format(rew))

    env.close()


if __name__ == '__main__':
    lunar_lander_test()