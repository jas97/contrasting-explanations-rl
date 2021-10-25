from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from explanations.envs.driving.driving_simulator import DrivingSimulator


def test():
    env = DrivingSimulator(reward_weights={'car_distance': 10,
                                            'goal_distance': 5,
                                            'dev_from_init_vel': 0,
                                            'turn': 0,
                                            'acc': 0,
                                            'progress': 0.1})
    #
    model_path = 'models/{}/experiments/model_dqn_0_2'.format('driving')
    model = DQN.load(model_path, env)

    # print('Model baseline: {} timesteps mean reward: {}, stddev: {}'.format(1000000, *evaluate_policy(model, env, n_eval_episodes=1000, deterministic=True)))

    done = False
    obs = env.reset()
    total_rew=0

    while not done:
        action, _ = model.predict(obs)
        print('Action: {}'.format(action))
        obs, rew, done, _ = env.step(action)
        total_rew += rew
        print('Reward: {}'.format(rew))
        env.render()

    print('Total reward: {}'.format(total_rew))


if __name__ == '__main__':
    test()