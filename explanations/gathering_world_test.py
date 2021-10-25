import time

from gym_mo.envs.gridworlds import MOGatheringEnv, MOTrafficEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env


def gathering_world_test():
    env = MOGatheringEnv(from_pixels=False)
    check_env(env)

    print('Training')
    model = DQN('MlpPolicy', env, verbose=10, exploration_fraction=0.5)
    model.learn(total_timesteps=500000)
    print('Finished training')

    done = False
    obs = env.reset()
    while not done:
        print('Observation: {}'.format(obs))
        # action, _ = model.predict(obs)
        action = env.action_space.sample()
        obs, rew, done, _ = env.step(action)
        print('Reward: {}'.format(rew))
        env.render()
        time.sleep(1.5)

    env.close()


def main():
    gathering_world_test()


if __name__ == '__main__':
    main()