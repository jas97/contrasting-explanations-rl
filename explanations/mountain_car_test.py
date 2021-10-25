from stable_baselines3 import DQN

from explanations.envs.mountain_car.mountain_car_env import MountainCarEnv


def lunar_lander_test():
    env = MountainCarEnv()

    model = DQN.load('models/mountain-car/experiments/model_dqn_0_0.2')

    done = False
    obs = env.reset()
    left_acc = 0
    right_acc = 0
    while not done:
        action, _ = model.predict(obs)
        obs, rew, done, _ = env.step(action)
        env.render()
        print('Reward:{}'.format(rew))

        if action == 0:
            left_acc += 1
        elif action == 2:
            right_acc += 1

    env.close()

    print('Brake + acc used total {}'.format(left_acc + right_acc))



if __name__ == '__main__':
    lunar_lander_test()